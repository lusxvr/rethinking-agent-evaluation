"""One generation of preempted-cell recovery for one tier, submitted by run_grid_slurm.py
--preemptible and then by itself via slurm/sweep_grid.sbatch (afterany on the previous array).

Resubmits cells without score.json against the same server. Stops when none remain (teardown,
exit 0) or at --max-generations (teardown, exit 1). A dead server needs continue_grid_slurm.py.
"""

import argparse
import os
import shlex
import sys
from pathlib import Path

import yaml

from orchestrate import REPO_ROOT
from scripts.production.slurm_utils import array_spec, gres, load_cluster_profile, remaining_indices, sbatch
from scripts.production.run_grid import expand_grid, grid_oracle

SLURM_DIR = REPO_ROOT / "slurm"


def _account_args(profile: dict[str, str]) -> list[str]:
    return ["--account", profile["SLURM_ACCOUNT"]] if profile.get("SLURM_ACCOUNT") else []


def _cpu_job_args() -> list[str]:
    """--partition/--qos/--account for short CPU jobs, from the cluster profile."""
    profile = load_cluster_profile()
    args = ["--partition", profile.get("TEARDOWN_PARTITION") or "cpu", *_account_args(profile)]
    return args + (["--qos", profile["TEARDOWN_QOS"]] if profile.get("TEARDOWN_QOS") else [])


def _submit_teardown(grid_log_dir: Path, tier: str, server_job: str) -> str:
    # No --dependency: this job already ran after the previous array finished.
    return sbatch(
        [*_cpu_job_args(),
         f"--job-name=vllm-teardown-{tier}", f"--output={grid_log_dir / 'vllm-teardown-%j.out'}"],
        "vllm_teardown.sbatch", [server_job],
    )


def sweep_grid_slurm(
    grid_dir_name: str, spec_name: str, tier: str, manifest_path: Path, task: str, server_job: str,
    generation: int, max_generations: int, partition: str, worker_qos: str, max_concurrent: int,
    worker_cpus_per_task: int | None = None, worker_reservation: str | None = None,
    worker_time: str | None = None,
) -> int:
    grid_dir = REPO_ROOT / "runs" / grid_dir_name
    spec_path = REPO_ROOT / "specs" / f"{spec_name}.yaml"
    spec_dict = yaml.safe_load(spec_path.read_text())
    tier_cells = [c for c in expand_grid(spec_dict) if c.model == tier]
    grid_log_dir = SLURM_DIR / "logs" / grid_dir_name
    # Must match the oracle setting the original array used -- same spec, same grid.
    worker_env = {**os.environ, "AGENT_ORACLE": "1" if grid_oracle(spec_dict) else "0"}

    remaining = remaining_indices(manifest_path, tier_cells, grid_dir)
    if not remaining:
        print(f"Tier {tier}: all cells done after generation {generation} -- tearing down")
        teardown_job = _submit_teardown(grid_log_dir, tier, server_job)
        print(f"Teardown job {teardown_job} submitted")
        return 0

    if generation >= max_generations:
        print(f"Tier {tier}: {len(remaining)} cell(s) still missing score.json after "
              f"{max_generations} generation(s) -- giving up automated retry (task ids: {remaining}). "
              f"Run scripts.production.continue_grid_slurm by hand to keep going.")
        teardown_job = _submit_teardown(grid_log_dir, tier, server_job)
        print(f"Teardown job {teardown_job} submitted")
        return 1

    next_gen = generation + 1
    print(f"Tier {tier} generation {generation}: {len(remaining)} cell(s) still missing "
          f"score.json, resubmitting as generation {next_gen}")

    profile = load_cluster_profile()
    array_extra_args = [
        "--partition", partition, "--qos", worker_qos, *_account_args(profile),
        *shlex.split(profile.get("WORKER_NEEDS_INTERNET_FLAG") or ""),
        f"--gres={gres(profile, 1)}", f"--dependency=after:{server_job}",
        f"--array={array_spec(remaining, max_concurrent)}",
        f"--output={grid_log_dir / 'agent-worker-%A_%a.out'}",
    ]
    if worker_cpus_per_task is not None:
        array_extra_args += [f"--cpus-per-task={worker_cpus_per_task}"]
    if worker_reservation is not None:
        array_extra_args += [f"--reservation={worker_reservation}"]
    if worker_time is not None:
        # Short --time helps preemptible jobs get backfilled; a cut-off cell is simply retried.
        array_extra_args += [f"--time={worker_time}"]
    array_job = sbatch(
        array_extra_args, "agent_worker_array.sbatch",
        [str(manifest_path), grid_dir_name, task, server_job],
        env=worker_env,
    )
    print(f"Generation {next_gen} array {array_job} submitted ({len(remaining)} cells, "
          f"%{max_concurrent} throttle)")

    sweep_args = [
        grid_dir_name, "--spec", spec_name, "--tier", tier, "--manifest", str(manifest_path),
        "--task", task, "--server-job", server_job, "--generation", str(next_gen),
        "--max-generations", str(max_generations), "--partition", partition,
        "--worker-qos", worker_qos, "--max-concurrent", str(max_concurrent),
    ]
    if worker_cpus_per_task is not None:
        sweep_args += ["--worker-cpus-per-task", str(worker_cpus_per_task)]
    if worker_reservation is not None:
        sweep_args += ["--worker-reservation", worker_reservation]
    if worker_time is not None:
        sweep_args += ["--worker-time", worker_time]
    sweep_job = sbatch(
        [*_cpu_job_args(), f"--dependency=afterany:{array_job}",
         f"--job-name=sweep-grid-{tier}-gen{next_gen}",
         f"--output={grid_log_dir / f'sweep-grid-{tier}-gen{next_gen}-%j.out'}"],
        "sweep_grid.sbatch", sweep_args,
    )
    print(f"Sweep job {sweep_job} submitted, depending on generation {next_gen} array {array_job}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="One generation of automated preempted-cell recovery -- submitted automatically, not run by hand.")
    parser.add_argument("grid_dir")
    parser.add_argument("--spec", required=True)
    parser.add_argument("--tier", required=True)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--task", required=True)
    parser.add_argument("--server-job", required=True)
    parser.add_argument("--generation", required=True, type=int)
    parser.add_argument("--max-generations", required=True, type=int)
    parser.add_argument("--partition", default="gpu")
    parser.add_argument("--worker-qos", required=True)
    parser.add_argument("--max-concurrent", type=int, default=10)
    parser.add_argument("--worker-cpus-per-task", type=int, default=None)
    parser.add_argument("--worker-reservation", default=None)
    parser.add_argument(
        "--worker-time", default=None,
        help="Worker --time (default 4h), carried to later generations.",
    )
    args = parser.parse_args()
    return sweep_grid_slurm(
        args.grid_dir, args.spec, args.tier, args.manifest, args.task, args.server_job,
        args.generation, args.max_generations, args.partition, args.worker_qos,
        args.max_concurrent, args.worker_cpus_per_task, args.worker_reservation, args.worker_time,
    )


if __name__ == "__main__":
    sys.exit(main())
