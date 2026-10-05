"""Run a YAML grid spec locally: expand the axis cross product x replicates and run each cell
through orchestrate.launch_run, one GPU per concurrent run. Each call creates a new
runs/<spec>-<timestamp>/. Spec format: README.md.

Usage: uv run python -m scripts.production.run_grid <spec-name> --gpus <uuid,...>
--gpus is required; pass only free GPUs, never the backbone's.
"""

import argparse
import itertools
import os
import queue
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import yaml

from axes import AXIS_NAMES, RunSpec
from orchestrate import REPO_ROOT, launch_run

# itertools.product's leftmost axis (information) barely moves, so unshuffled it rides on execution-time order.
GRID_SHUFFLE_SEED = 20260825


def expand_grid(spec: dict) -> list[RunSpec]:
    """All cells as RunSpecs, validated up front.

    Ordered replicate-major, so an interrupted grid covers every config at n=1 first. Cells within a
    replicate are shuffled (GRID_SHUFFLE_SEED) so no axis aliases with execution order.
    """
    axes = spec["axes"]
    missing = set(AXIS_NAMES) - set(axes)
    if missing:
        raise SystemExit(f"Spec is missing axes: {sorted(missing)} -- every axis needs an explicit value list")
    unknown = set(axes) - set(AXIS_NAMES)
    if unknown:
        raise SystemExit(f"Spec has unknown axes: {sorted(unknown)} -- known axes are {list(AXIS_NAMES)}")

    combos = list(itertools.product(*(axes[name] for name in AXIS_NAMES)))
    rng = random.Random(GRID_SHUFFLE_SEED)
    cells = []
    for replicate in range(1, spec.get("replicates", 1) + 1):
        shuffled = combos.copy()
        rng.shuffle(shuffled)  # rng carries state across replicates, so each gets its own order
        for combo in shuffled:
            cells.append(RunSpec(task=spec["task"], replicate=replicate, **dict(zip(AXIS_NAMES, combo))))
    return cells


def grid_oracle(spec: dict) -> bool:
    """Whether the spec's optional top-level 'oracle' key grants oracle_check to every cell."""
    return bool(spec.get("oracle", False))


def run_grid(
    spec_name: str,
    gpus: list[str],
    base_url: str,
    max_workers: int | None,
    verbose: bool,
) -> int:
    spec_path = REPO_ROOT / "specs" / f"{spec_name}.yaml"
    if not spec_path.is_file():
        raise SystemExit(f"No such spec: {spec_path}")
    spec_dict = yaml.safe_load(spec_path.read_text())
    cells = expand_grid(spec_dict)
    oracle = grid_oracle(spec_dict)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    # specs/ is organized as specs/<task>/<name>.yaml, so spec_name itself may contain a "/" --
    # flatten that for the on-disk grid dir name (runs/ stays a flat directory of grids).
    grid_dir = REPO_ROOT / "runs" / f"{spec_name.replace('/', '-')}-{stamp}"
    grid_dir.mkdir(parents=True)

    workers = min(max_workers or len(gpus), len(gpus), len(cells))
    print(f"{len(cells)} cell(s) into {grid_dir}, {workers} at a time across {len(gpus)} GPU(s)."
          + (" oracle_check granted to every cell." if oracle else ""))
    if workers > 1:
        print(
            "Note: all workers share one vLLM backbone. At the agent-default server settings "
            "(--max-model-len 262144 --max-num-seqs 1) LLM calls serialize, so throughput is well "
            "below workers x -- retune the server for concurrency first if that matters."
        )

    # A GPU lease per concurrent run: a worker holds one UUID for the length of its cell.
    gpu_queue: queue.Queue[str] = queue.Queue()
    for uuid in gpus:
        gpu_queue.put(uuid)

    def run_cell(cell: RunSpec) -> None:
        gpu_uuid = gpu_queue.get()
        try:
            # The grid dir is new, so every cell.run_name in it is unique by construction.
            launch_run(cell, grid_dir / cell.run_name, gpu_uuid, base_url, verbose, oracle=oracle)
        finally:
            gpu_queue.put(gpu_uuid)

    failures: list[tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_cell, cell): cell for cell in cells}
        for future, cell in futures.items():
            # Collected, not raised: one cell failing to launch must not take the rest with it.
            # Futures capture BaseException, so launch_run's SystemExit surfaces here too.
            if (exc := future.exception()) is not None:
                failures.append((cell.run_name, f"{type(exc).__name__}: {exc}"))

    if failures:
        print(f"{len(failures)} cell(s) failed to run:")
        for run_name, reason in failures:
            print(f"  - {run_name}: {reason}")
    print(f"Grid finished: {grid_dir}")
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a grid of agent runs from a YAML spec (specs/<task>/<name>.yaml).")
    parser.add_argument("spec", help="Spec name without .yaml, e.g. 'redshift-estimation/smoke-small-fp8'")
    parser.add_argument(
        "--gpus",
        required=True,
        help="Comma-separated free GPU/MIG UUIDs, one concurrent run each. Exclude the backbone's GPU.",
    )
    parser.add_argument("--base-url", default=f"http://localhost:{os.environ.get('VLLM_PORT', '61000')}/v1")
    parser.add_argument("--max-workers", type=int, default=None, help="Cap concurrent runs below the number of --gpus")
    parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print live container progress (interleaves across concurrent runs).",
    )
    args = parser.parse_args()

    gpus = [uuid.strip() for uuid in args.gpus.split(",") if uuid.strip()]
    if not gpus:
        raise SystemExit("--gpus is empty")
    return run_grid(args.spec, gpus, args.base_url, args.max_workers, args.verbose)


if __name__ == "__main__":
    sys.exit(main())
