"""Resume a run_grid_slurm.py grid whose vLLM server stopped early, skipping cells with score.json.

Re-expands the spec (deterministic order) to map manifest lines to run names, then submits a new
server, worker array and teardown per tier with remaining cells. orchestrate.launch_run decides
per cell whether to re-score or rerun. Run only after the old array has fully stopped, or a new
attempt may delete a directory an old one is still writing.

Granular clusters reuse the original manifest with an index list (--array=3,7,19-45%8).
Whole-node clusters write the remaining cells to a new manifest, since packed tasks map to blocks
of lines. Multi-node serving is not included. --preemptible works as in run_grid_slurm.py.

Usage: uv run python -m scripts.production.continue_grid_slurm <grid-dir-name> --spec <spec-name>
"""

import argparse
import os
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from axes import MODEL_GPU_COUNT, MODEL_LEVELS, RunSpec, provider_for
from orchestrate import REPO_ROOT
from scripts.production.slurm_utils import array_spec, gres, load_cluster_profile, remaining_indices, sbatch, write_manifest
from scripts.production.run_grid import expand_grid, grid_oracle

SLURM_DIR = REPO_ROOT / "slurm"


def _cells_by_tier(cells: list[RunSpec]) -> dict[str, list[RunSpec]]:
    by_tier: dict[str, list[RunSpec]] = {}
    for cell in cells:
        by_tier.setdefault(cell.model, []).append(cell)
    return {tier: by_tier[tier] for tier in MODEL_LEVELS if tier in by_tier}


def continue_grid_slurm(
    grid_dir_name: str, spec_name: str, partition: str, server_qos: str | None, worker_qos: str | None,
    max_concurrent: int, worker_cpus_per_task: int | None = None, server_time: str | None = None,
    server_gres: str | None = None, server_cpus_per_task: int | None = None, server_mem: str | None = None,
    server_reservation: str | None = None, worker_reservation: str | None = None,
    worker_time: str | None = None, preemptible: bool = False, max_preempt_generations: int = 100,
    account: str | None = None, profile: dict[str, str] | None = None,
    server_data_parallel_size: int | None = None,
) -> int:
    profile = profile if profile is not None else load_cluster_profile()
    whole_node_only = profile.get("WHOLE_NODE_ONLY") == "1"
    gpus_per_node = int(profile["GPUS_PER_NODE"]) if whole_node_only else None
    cores_per_node = int(profile["CORES_PER_NODE"]) if whole_node_only else None

    if preemptible and whole_node_only:
        raise SystemExit(
            "--preemptible isn't supported yet on a whole-node-only cluster (profile "
            "WHOLE_NODE_ONLY=1) -- scripts.production.sweep_grid_slurm doesn't know about "
            "whole-node packing yet, same restriction as run_grid_slurm.py's own --preemptible."
        )

    grid_dir = REPO_ROOT / "runs" / grid_dir_name
    if not grid_dir.is_dir():
        raise SystemExit(f"No such grid dir: {grid_dir}")
    spec_path = REPO_ROOT / "specs" / f"{spec_name}.yaml"
    if not spec_path.is_file():
        raise SystemExit(f"No such spec: {spec_path}")

    spec_dict = yaml.safe_load(spec_path.read_text())
    cells = expand_grid(spec_dict)
    tiers = _cells_by_tier(cells)
    # Read by agent_worker_array(_packed).sbatch's own AGENT_ORACLE lookup -- must match the
    # oracle setting the original run_grid_slurm.py grid used, since this resumes the same spec.
    worker_env = {**os.environ, "AGENT_ORACLE": "1" if grid_oracle(spec_dict) else "0"}

    if preemptible:
        preemptible_qos = profile.get("PREEMPTIBLE_QOS")
        if not preemptible_qos:
            raise SystemExit("--preemptible needs PREEMPTIBLE_QOS in the cluster profile")
        if worker_qos != preemptible_qos:
            print(f"--preemptible: overriding --worker-qos {worker_qos!r} -> {preemptible_qos!r}")
            worker_qos = preemptible_qos

    account_args = ["--account", account] if account else []
    server_qos_args = ["--qos", server_qos] if server_qos else []
    worker_qos_args = ["--qos", worker_qos] if worker_qos else []
    server_internet_args = shlex.split(profile.get("SERVER_NEEDS_INTERNET_FLAG") or "")
    worker_internet_args = shlex.split(profile.get("WORKER_NEEDS_INTERNET_FLAG") or "")
    teardown_partition = profile.get("TEARDOWN_PARTITION") or "cpu"
    teardown_qos = profile.get("TEARDOWN_QOS") or None
    teardown_qos_args = ["--qos", teardown_qos] if teardown_qos else []
    teardown_gres_args = [f"--gres={gres(profile, gpus_per_node)}"] if whole_node_only else []
    teardown_cpus_args = ["--cpus-per-task", str(cores_per_node)] if whole_node_only else []
    whole_node_args = ["--nodes", "1"] if whole_node_only else []

    grid_log_dir = SLURM_DIR / "logs" / grid_dir_name
    grid_log_dir.mkdir(parents=True, exist_ok=True)

    # Resolve tiers with remaining cells first, so --preemptible's last-tier check ignores finished tiers.
    active_tiers: list[tuple[str, list[RunSpec], Path, list[int]]] = []
    for tier, tier_cells in tiers.items():
        manifest_path = grid_dir / f".manifest-{tier}.tsv"
        if not manifest_path.is_file():
            print(f"--- Tier {tier}: no manifest at {manifest_path}, skipping ---")
            continue

        # Only selects cells without score.json; launch_run decides what to do with each.
        remaining = remaining_indices(manifest_path, tier_cells, grid_dir)
        done = len(tier_cells) - len(remaining)
        print(f"--- Tier {tier}: {done}/{len(tier_cells)} already done, {len(remaining)} remaining ---")
        if remaining:
            active_tiers.append((tier, tier_cells, manifest_path, remaining))

    # Same tier chaining as run_grid_slurm.py.
    prev_dependency: str | None = None
    all_job_ids: list[str] = []
    for tier_index, (tier, tier_cells, manifest_path, remaining) in enumerate(active_tiers):
        hosted_api = provider_for(tier) == "anthropic"
        if preemptible and hosted_api:
            raise SystemExit(
                f"--preemptible isn't supported for a hosted-API tier ({tier!r}) -- "
                f"scripts.production.sweep_grid_slurm's resubmission is keyed on --server-job, "
                f"which only means something for a locally-served vLLM tier."
            )
        if preemptible and tier_index < len(active_tiers) - 1:
            # Mirrors run_grid_slurm.py's own restriction -- see this module's docstring.
            raise SystemExit(
                f"--preemptible with more than one remaining tier isn't supported yet (tier "
                f"{tier!r} is not the last) -- the next tier's server can't be chained to wait "
                f"for a preemptible tier's teardown, since that job doesn't exist until the sweep "
                f"chain decides to submit it."
            )

        if whole_node_only:
            # Convert --max-concurrent to whole nodes, rounding up.
            node_throttle = max(1, -(-max_concurrent // gpus_per_node))  # ceil
            effective_max_num_seqs = node_throttle * gpus_per_node
            if effective_max_num_seqs != max_concurrent:
                print(f"note: --max-concurrent {max_concurrent} isn't a multiple of "
                      f"{gpus_per_node} (whole-node-only cluster) -- rounding up to "
                      f"{node_throttle} node(s), {effective_max_num_seqs} concurrent cells")
            if not hosted_api:
                tier_gpu_count = MODEL_GPU_COUNT.get(tier)
                if tier_gpu_count is None:
                    raise SystemExit(f"Tier {tier!r} has no axes.MODEL_GPU_COUNT entry -- needed on a "
                                      f"whole-node-only cluster to size the server job")
                if tier_gpu_count > gpus_per_node:
                    raise SystemExit(
                        f"Tier {tier!r} needs {tier_gpu_count} GPUs, more than this cluster's "
                        f"{gpus_per_node}-GPU nodes; multi-node serving is not included"
                    )
        else:
            effective_max_num_seqs = max_concurrent

        if hosted_api:
            # Hosted-API tier: no server to resume (see run_grid_slurm.py).
            server_job = "none"
            print(f"note: tier {tier} is a hosted API (axes.provider_for) -- no vLLM server job for this tier")
        else:
            # Never higher than effective_max_num_seqs: with exactly that many workers there's never
            # more than that many simultaneous requests (same reasoning as run_grid_slurm.py).
            server_env = {**os.environ, "VLLM_MAX_NUM_SEQS": str(effective_max_num_seqs)}
            if whole_node_only:
                # Same data-parallel fan-out as run_grid_slurm.py.
                dp = server_data_parallel_size if server_data_parallel_size is not None else gpus_per_node // tier_gpu_count
                if dp > 1:
                    per_replica_max_num_seqs = max(1, -(-effective_max_num_seqs // dp))  # ceil
                    server_env["VLLM_MAX_NUM_SEQS"] = str(per_replica_max_num_seqs)
                    existing_extra = os.environ.get("VLLM_EXTRA_ARGS", "")
                    server_env["VLLM_EXTRA_ARGS"] = f"--data-parallel-size {dp} {existing_extra}".strip()
                    print(f"note: tier {tier} only needs {tier_gpu_count}/{gpus_per_node} GPUs on this "
                          f"whole-node-only cluster -- fanning the server out to --data-parallel-size "
                          f"{dp} ({per_replica_max_num_seqs} max-num-seqs per replica, "
                          f"{per_replica_max_num_seqs * dp} total) instead of leaving the rest idle")

            server_extra_args = [
                "--partition", partition, *server_qos_args, *account_args, *server_internet_args,
                *whole_node_args,
                f"--output={grid_log_dir / 'vllm-server-%j.out'}",
            ]
            if server_time is not None:
                server_extra_args += [f"--time={server_time}"]
            if server_gres is not None:
                server_extra_args += [f"--gres={server_gres}"]
            elif whole_node_only:
                server_extra_args += [f"--gres={gres(profile, gpus_per_node)}"]
            else:
                server_extra_args += [f"--gres={gres(profile, MODEL_GPU_COUNT.get(tier, 1))}"]
            if server_cpus_per_task is not None:
                server_extra_args += [f"--cpus-per-task={server_cpus_per_task}"]
            elif whole_node_only:
                server_extra_args += [f"--cpus-per-task={cores_per_node}"]
            if server_mem is not None:
                server_extra_args += [f"--mem={server_mem}"]
            if server_reservation is not None:
                server_extra_args += [f"--reservation={server_reservation}"]
            if prev_dependency is not None:
                server_extra_args.append(f"--dependency={prev_dependency}")
            server_job = sbatch(server_extra_args, "vllm_server.sbatch", [tier], env=server_env)
            print(f"Server job {server_job} submitted (max-num-seqs={effective_max_num_seqs})")

        worker_dependency = f"after:{server_job}" if not hosted_api else prev_dependency
        worker_dependency_args = [f"--dependency={worker_dependency}"] if worker_dependency else []

        if whole_node_only:
            if worker_cpus_per_task is not None:
                print("note: --worker-cpus-per-task is ignored on a whole-node-only cluster -- "
                      "each cell's own srun step gets cores_per_node/gpus_per_node instead")
            # Packed tasks address blocks of lines, so write the remaining cells to a new manifest.
            remaining_cells = [tier_cells[i - 1] for i in remaining]
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            continue_manifest_path = grid_dir / f".manifest-{tier}.continue-{stamp}.tsv"
            write_manifest(continue_manifest_path, remaining_cells)
            n_nodes = -(-len(remaining_cells) // gpus_per_node)  # ceil
            cpus_per_slot = cores_per_node // gpus_per_node
            array_extra_args = [
                "--partition", partition, *worker_qos_args, *account_args, *worker_internet_args,
                *whole_node_args,
                f"--gres={gres(profile, gpus_per_node)}",
                *worker_dependency_args,
                f"--array=1-{n_nodes}%{node_throttle}",
                f"--output={grid_log_dir / 'agent-worker-packed-%A_%a.out'}",
            ]
            worker_script = "agent_worker_array_packed.sbatch"
            worker_script_args = [
                str(continue_manifest_path), grid_dir_name, tier_cells[0].task, server_job,
                str(gpus_per_node), str(cpus_per_slot),
            ]
            array_throttle_desc = f"%{node_throttle} throttle of nodes"
        else:
            array_extra_args = [
                "--partition", partition, *worker_qos_args, *account_args, *worker_internet_args,
                f"--gres={gres(profile, 1)}", *worker_dependency_args,
                f"--array={array_spec(remaining, max_concurrent)}",
                f"--output={grid_log_dir / 'agent-worker-%A_%a.out'}",
            ]
            if worker_cpus_per_task is not None:
                array_extra_args += [f"--cpus-per-task={worker_cpus_per_task}"]
            worker_script = "agent_worker_array.sbatch"
            worker_script_args = [str(manifest_path), grid_dir_name, tier_cells[0].task, server_job]
            array_throttle_desc = f"%{max_concurrent} throttle"
        if worker_reservation is not None:
            array_extra_args += [f"--reservation={worker_reservation}"]
        if worker_time is not None:
            # Short limits help preemptible jobs get backfilled.
            array_extra_args += [f"--time={worker_time}"]
        array_job = sbatch(array_extra_args, worker_script, worker_script_args, env=worker_env)
        print(f"Worker array {array_job} submitted ({len(remaining)} cells, {array_throttle_desc})"
              + (f", depending on server job {server_job}" if not hosted_api else ""))

        if hosted_api:
            # No server to tear down -- see run_grid_slurm.py's own comment on why the next tier
            # chains afterany off the worker array itself rather than afterok off a teardown job.
            all_job_ids += [array_job]
            prev_dependency = f"afterany:{array_job}"
        elif preemptible:
            # Same sweep chain as run_grid_slurm.py --preemptible.
            sweep_args = [
                grid_dir_name, "--spec", spec_name, "--tier", tier, "--manifest", str(manifest_path),
                "--task", tier_cells[0].task, "--server-job", server_job, "--generation", "1",
                "--max-generations", str(max_preempt_generations), "--partition", partition,
                "--worker-qos", worker_qos, "--max-concurrent", str(max_concurrent),
            ]
            if worker_cpus_per_task is not None:
                sweep_args += ["--worker-cpus-per-task", str(worker_cpus_per_task)]
            if worker_reservation is not None:
                sweep_args += ["--worker-reservation", worker_reservation]
            if worker_time is not None:
                sweep_args += ["--worker-time", worker_time]
            sweep_job = sbatch(
                ["--partition", teardown_partition, *teardown_qos_args, *account_args,
                 f"--dependency=afterany:{array_job}", f"--job-name=sweep-grid-{tier}-gen1",
                 f"--output={grid_log_dir / f'sweep-grid-{tier}-gen1-%j.out'}"],
                "sweep_grid.sbatch", sweep_args,
            )
            print(f"Sweep job {sweep_job} submitted, depending on worker array {array_job} "
                  f"(up to {max_preempt_generations} generation(s) before manual continuation is needed)")
            all_job_ids += [server_job, array_job, sweep_job]
            # Teardown is submitted later by the sweep chain; this is the last tier.
        else:
            teardown_job = sbatch(
                ["--partition", teardown_partition, *teardown_qos_args, *teardown_gres_args,
                 *teardown_cpus_args, *account_args, *whole_node_args,
                 f"--dependency=afterany:{array_job}",
                 "--time=00:02:00", f"--job-name=vllm-teardown-{tier}",
                 f"--output={grid_log_dir / 'vllm-teardown-%j.out'}"],
                "vllm_teardown.sbatch", [server_job],
            )
            print(f"Teardown job {teardown_job} submitted, depending on worker array {array_job}")

            all_job_ids += [server_job, array_job, teardown_job]
            prev_dependency = f"afterok:{teardown_job}"

    if not all_job_ids:
        print("Nothing to resume -- every cell in every tier already has a score.json.")
        return 0
    print(f"All jobs submitted: {', '.join(all_job_ids)}")
    print(f"Track with: squeue -u $USER   /   tail -f slurm/logs/{grid_dir_name}/*.out")
    return 0


def main() -> int:
    profile = load_cluster_profile()
    parser = argparse.ArgumentParser(description="Resume a run_grid_slurm.py grid, skipping cells that already have a score.json.")
    parser.add_argument("grid_dir", help="Grid directory name under runs/, e.g. 'redshift-estimation-full-large-fp8-<timestamp>'")
    parser.add_argument("--spec", required=True, help="Spec name without .yaml -- must be the same spec the grid was launched from")
    parser.add_argument(
        "--partition", default=profile.get("SLURM_PARTITION", "gpu"),
        help=f"Defaults to the active cluster's profile (CLUSTER={profile['_CLUSTER_NAME']} -> "
        f"slurm/clusters/{profile['_CLUSTER_NAME']}.env), same as run_grid_slurm.py.",
    )
    parser.add_argument(
        "--account", default=profile.get("SLURM_ACCOUNT") or None,
        help="Slurm --account for every job; empty if the cluster needs none.",
    )
    parser.add_argument("--server-qos", default=profile.get("SERVER_QOS") or None)
    parser.add_argument("--worker-qos", default=profile.get("WORKER_QOS") or None)
    parser.add_argument("--worker-cpus-per-task", type=int, default=None)
    parser.add_argument(
        "--server-time", default=profile.get("SERVER_TIME") or None,
        help="Server --time; defaults to the profile's SERVER_TIME.",
    )
    parser.add_argument("--server-gres", default=None)
    parser.add_argument(
        "--server-data-parallel-size", type=int, default=None,
        help="As in run_grid_slurm.py.",
    )
    parser.add_argument("--server-cpus-per-task", type=int, default=None)
    parser.add_argument("--server-mem", default=None)
    parser.add_argument("--server-reservation", default=None)
    parser.add_argument("--worker-reservation", default=None)
    parser.add_argument("--max-concurrent", type=int, default=10)
    parser.add_argument(
        "--worker-time", default=None,
        help="Worker --time (default 4h).",
    )
    parser.add_argument(
        "--preemptible", action="store_true",
        help="As in run_grid_slurm.py; at most one tier may have remaining cells. Granular clusters only.",
    )
    parser.add_argument(
        "--max-preempt-generations", type=int, default=100,
        help="With --preemptible: maximum resubmission rounds (default 100).",
    )
    args = parser.parse_args()
    return continue_grid_slurm(
        args.grid_dir, args.spec, args.partition, args.server_qos, args.worker_qos,
        args.max_concurrent, args.worker_cpus_per_task, args.server_time,
        args.server_gres, args.server_cpus_per_task, args.server_mem,
        args.server_reservation, args.worker_reservation, args.worker_time,
        args.preemptible, args.max_preempt_generations,
        account=args.account, profile=profile,
        server_data_parallel_size=args.server_data_parallel_size,
    )


if __name__ == "__main__":
    sys.exit(main())
