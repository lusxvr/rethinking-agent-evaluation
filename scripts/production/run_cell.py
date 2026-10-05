"""Run one grid cell into a given grid directory; used by the Slurm worker jobs.

Usage: uv run python -m scripts.production.run_cell --grid-dir <dir> --task <task> --gpu <uuid> --base-url <url> ...
"""

import argparse
import sys

from axes import AXIS_DEFAULTS, AXIS_LEVELS, RunSpec
from orchestrate import REPO_ROOT, launch_run


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one grid cell into runs/<grid-dir>/<cell.run_name>.")
    parser.add_argument("--grid-dir", required=True, help="Directory name under runs/, shared by every cell of one grid")
    parser.add_argument("--task", required=True)
    parser.add_argument("--information", default=AXIS_DEFAULTS["information"], choices=AXIS_LEVELS["information"])
    parser.add_argument("--harness", default=AXIS_DEFAULTS["harness"], choices=AXIS_LEVELS["harness"])
    parser.add_argument("--verification", default=AXIS_DEFAULTS["verification"], choices=AXIS_LEVELS["verification"])
    parser.add_argument("--budget", default=AXIS_DEFAULTS["budget"], choices=AXIS_LEVELS["budget"])
    parser.add_argument("--model", default=AXIS_DEFAULTS["model"], choices=AXIS_LEVELS["model"])
    parser.add_argument("--replicate", type=int, default=1)
    parser.add_argument("--gpu", required=True, help="GPU UUID to run the agent sandbox on")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--oracle",
        action="store_true",
        default=False,
        help="Grant oracle_check for this cell.",
    )
    args = parser.parse_args()

    spec = RunSpec(
        task=args.task,
        information=args.information,
        harness=args.harness,
        verification=args.verification,
        budget=args.budget,
        model=args.model,
        replicate=args.replicate,
    )
    # Created (not required to pre-exist) since concurrent worker jobs racing to claim the same
    # grid dir is expected -- each cell's own run_name subdirectory is still unique by construction.
    grid_dir = REPO_ROOT / "runs" / args.grid_dir
    grid_dir.mkdir(parents=True, exist_ok=True)
    return launch_run(spec, grid_dir / spec.run_name, args.gpu, args.base_url, args.verbose, oracle=args.oracle)


if __name__ == "__main__":
    sys.exit(main())
