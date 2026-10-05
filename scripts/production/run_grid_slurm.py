"""Submit a grid spec to Slurm and exit. Per model tier: a vLLM server job, a throttled worker
array (agent_worker_array.sbatch, --array=1-N%--max-concurrent) that starts after the server job,
and a teardown job after the array (afterany). The next tier's server waits for the previous
teardown (afterok). --after chains the first tier onto an outside job.

Hosted-API tiers (axes.provider_for) get no server or teardown; their array gets the server id
"none" and is itself the completion job the next tier waits for (afterany).

Cluster defaults come from slurm/clusters/<CLUSTER>.env; CLI flags override them. With
WHOLE_NODE_ONLY=1, workers use agent_worker_array_packed.sbatch, one node per array task, and
--max-concurrent is rounded up to whole nodes.

--preemptible runs workers under the profile's PREEMPTIBLE_QOS and chains sweep_grid_slurm.py after the array to
resubmit preempted cells against the same server. Not supported for hosted-API tiers or on
whole-node clusters. If the server itself dies, use continue_grid_slurm.py.

Usage: uv run python -m scripts.production.run_grid_slurm <spec-name>
"""

import argparse
import os
import shlex
import sys
from datetime import datetime, timezone

import yaml

from axes import MODEL_GPU_COUNT, MODEL_LEVELS, RunSpec, provider_for
from orchestrate import REPO_ROOT
from scripts.production.slurm_utils import gres as _gres
from scripts.production.slurm_utils import load_cluster_profile as _load_cluster_profile
from scripts.production.slurm_utils import sbatch
from scripts.production.slurm_utils import write_manifest as _write_manifest
from scripts.production.run_grid import expand_grid, grid_oracle

SLURM_DIR = REPO_ROOT / "slurm"


def _cells_by_tier(cells: list[RunSpec]) -> dict[str, list[RunSpec]]:
    by_tier: dict[str, list[RunSpec]] = {}
    for cell in cells:
        by_tier.setdefault(cell.model, []).append(cell)
    # MODEL_LEVELS order, not first-appearance order, so a spec's tier sequence is deterministic
    # across re-runs regardless of how axes are ordered in the YAML.
    return {tier: by_tier[tier] for tier in MODEL_LEVELS if tier in by_tier}


def run_grid_slurm(
    spec_name: str, partition: str, server_qos: str | None, worker_qos: str | None,
    max_concurrent: int, worker_cpus_per_task: int | None = None, server_time: str | None = None,
    server_gres: str | None = None, server_cpus_per_task: int | None = None, server_mem: str | None = None,
    server_reservation: str | None = None, worker_reservation: str | None = None, after: str | None = None,
    account: str | None = None, profile: dict[str, str] | None = None,
    preemptible: bool = False, max_preempt_generations: int = 100, worker_time: str | None = None,
    server_data_parallel_size: int | None = None,
) -> int:
    profile = profile if profile is not None else _load_cluster_profile()
    whole_node_only = profile.get("WHOLE_NODE_ONLY") == "1"
    gpus_per_node = int(profile["GPUS_PER_NODE"]) if whole_node_only else None
    cores_per_node = int(profile["CORES_PER_NODE"]) if whole_node_only else None

    if preemptible:
        if whole_node_only:
            raise SystemExit(
                "--preemptible is not supported on a whole-node-only cluster (WHOLE_NODE_ONLY=1): "
                "sweep_grid_slurm does not handle whole-node packing."
            )
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
    teardown_gres_args = [f"--gres={_gres(profile, gpus_per_node)}"] if whole_node_only else []
    # Matches what an exclusive whole-node partition allocates anyway.
    teardown_cpus_args = ["--cpus-per-task", str(cores_per_node)] if whole_node_only else []
    # Some clusters require --nodes with an internet-access flag; passed on every job.
    whole_node_args = ["--nodes", "1"] if whole_node_only else []

    spec_path = REPO_ROOT / "specs" / f"{spec_name}.yaml"
    if not spec_path.is_file():
        raise SystemExit(f"No such spec: {spec_path}")
    spec_dict = yaml.safe_load(spec_path.read_text())
    cells = expand_grid(spec_dict)
    oracle = grid_oracle(spec_dict)
    # Read by agent_worker_array(_packed).sbatch's own AGENT_ORACLE lookup -- same mechanism as
    # VLLM_MAX_NUM_SEQS below, just on the worker array's sbatch() call instead of the server's.
    worker_env = {**os.environ, "AGENT_ORACLE": "1" if oracle else "0"}

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    # specs/ is organized as specs/<task>/<name>.yaml, so spec_name itself may contain a "/" --
    # flatten that for the on-disk grid dir name (runs/ and slurm/logs/ stay flat directories).
    grid_dir_name = f"{spec_name.replace('/', '-')}-{stamp}"
    grid_dir = REPO_ROOT / "runs" / grid_dir_name
    grid_dir.mkdir(parents=True)
    # Slurm logs go to slurm/logs/<grid-dir>/.
    grid_log_dir = SLURM_DIR / "logs" / grid_dir_name
    grid_log_dir.mkdir(parents=True)

    tiers = _cells_by_tier(cells)
    print(f"{len(cells)} cell(s) across {len(tiers)} model tier(s) into runs/{grid_dir_name} "
          f"(Slurm logs: slurm/logs/{grid_dir_name}), {max_concurrent} concurrent worker(s) per tier"
          + (" -- oracle_check granted to every cell" if oracle else ""))
    if after is not None:
        print(f"First tier's server job will wait on job {after} to succeed (afterok) before starting")

    # Dependency string for the next tier: afterok on a teardown, so a failed teardown blocks the
    # next server; afterany on a hosted-API array, since failed cells are normal outcomes.
    prev_dependency: str | None = f"afterok:{after}" if after is not None else None
    all_job_ids: list[str] = []
    tier_items = list(tiers.items())
    for tier_index, (tier, tier_cells) in enumerate(tier_items):
        hosted_api = provider_for(tier) == "anthropic"
        if preemptible and hosted_api:
            raise SystemExit(
                f"--preemptible isn't supported for a hosted-API tier ({tier!r}) -- "
                f"scripts.production.sweep_grid_slurm's resubmission is keyed on --server-job, "
                f"which only means something for a locally-served vLLM tier."
            )
        if preemptible and tier_index < len(tier_items) - 1:
            # A preemptible tier's teardown is submitted later by the sweep chain, so it must be the last tier.
            raise SystemExit(
                f"--preemptible with more than one remaining tier isn't supported yet (tier "
                f"{tier!r} is not the last) -- the next tier's server can't be chained to wait "
                f"for a preemptible tier's teardown, since that job doesn't exist until the sweep "
                f"chain decides to submit it."
            )
        print(f"--- Tier {tier}: {len(tier_cells)} cell(s) ---")
        if whole_node_only:
            # Convert --max-concurrent (cells) to whole nodes, rounding up.
            node_throttle = max(1, -(-max_concurrent // gpus_per_node))  # ceil
            effective_max_num_seqs = node_throttle * gpus_per_node
            if effective_max_num_seqs != max_concurrent:
                print(f"note: --max-concurrent {max_concurrent} isn't a multiple of "
                      f"{gpus_per_node} (whole-node-only cluster) -- rounding up to "
                      f"{node_throttle} node(s), {effective_max_num_seqs} concurrent cells")
        else:
            effective_max_num_seqs = max_concurrent

        if hosted_api:
            # Hosted-API tier: no server job; "none" tells the worker scripts to skip the server wait.
            server_job = "none"
            print(f"note: tier {tier} is a hosted API (axes.provider_for) -- no vLLM server job for this tier")
        else:
            # --max-num-seqs never exceeds the number of concurrent workers.
            server_n_nodes = 1
            if whole_node_only:
                tier_gpu_count = MODEL_GPU_COUNT.get(tier)
                if tier_gpu_count is None:
                    raise SystemExit(f"Tier {tier!r} has no axes.MODEL_GPU_COUNT entry -- needed on a "
                                      f"whole-node-only cluster to size the server job's node count")
                server_n_nodes = -(-tier_gpu_count // gpus_per_node)  # ceil
            if server_n_nodes > 1:
                raise SystemExit(f"Tier {tier} needs {MODEL_GPU_COUNT[tier]} GPUs, more than one "
                                 f"{gpus_per_node}-GPU node; multi-node serving is not included")
            server_node_args = whole_node_args

            server_env = {**os.environ, "VLLM_MAX_NUM_SEQS": str(effective_max_num_seqs)}
            if whole_node_only:
                # A small tier on a whole node runs data-parallel replicas on the spare GPUs.
                # VLLM_MAX_NUM_SEQS is per replica, so it is divided.
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
                *server_node_args,
                f"--output={grid_log_dir / 'vllm-server-%j.out'}",
            ]
            if server_time is not None:
                # Overrides vllm_server.sbatch's own #SBATCH --time (24h).
                server_extra_args += [f"--time={server_time}"]
            if server_gres is not None:
                server_extra_args += [f"--gres={server_gres}"]
            elif whole_node_only:
                server_extra_args += [f"--gres={_gres(profile, gpus_per_node)}"]
            else:
                server_extra_args += [f"--gres={_gres(profile, MODEL_GPU_COUNT.get(tier, 1))}"]
            if server_cpus_per_task is not None:
                server_extra_args += [f"--cpus-per-task={server_cpus_per_task}"]
            elif whole_node_only:
                server_extra_args += [f"--cpus-per-task={cores_per_node}"]
            if server_mem is not None:
                server_extra_args += [f"--mem={server_mem}"]
            if server_reservation is not None:
                # Pins the server job to that reservation's node(s); it may need a matching --server-qos.
                server_extra_args += [f"--reservation={server_reservation}"]
            if prev_dependency is not None:
                server_extra_args.append(f"--dependency={prev_dependency}")
            server_job = sbatch(server_extra_args, "vllm_server.sbatch", [tier], env=server_env)
            print(f"Server job {server_job} submitted (max-num-seqs={effective_max_num_seqs})")

        manifest_path = grid_dir / f".manifest-{tier}.tsv"
        _write_manifest(manifest_path, tier_cells)

        # No server: depend only on the previous tier's completion, if any.
        worker_dependency = f"after:{server_job}" if not hosted_api else prev_dependency
        worker_dependency_args = [f"--dependency={worker_dependency}"] if worker_dependency else []

        if whole_node_only:
            if worker_cpus_per_task is not None:
                print("note: --worker-cpus-per-task is ignored on a whole-node-only cluster -- "
                      "each cell's own srun step gets cores_per_node/gpus_per_node instead")
            n_nodes = -(-len(tier_cells) // gpus_per_node)  # ceil
            cpus_per_slot = cores_per_node // gpus_per_node
            array_extra_args = [
                "--partition", partition, *worker_qos_args, *account_args, *worker_internet_args,
                *whole_node_args,
                f"--gres={_gres(profile, gpus_per_node)}",
                *worker_dependency_args,
                f"--array=1-{n_nodes}%{node_throttle}",
                f"--output={grid_log_dir / 'agent-worker-packed-%A_%a.out'}",
            ]
            worker_script = "agent_worker_array_packed.sbatch"
            worker_script_args = [
                str(manifest_path), grid_dir_name, tier_cells[0].task, server_job,
                str(gpus_per_node), str(cpus_per_slot),
            ]
        else:
            array_extra_args = [
                "--partition", partition, *worker_qos_args, *account_args, *worker_internet_args,
                f"--gres={_gres(profile, 1)}", *worker_dependency_args,
                f"--array=1-{len(tier_cells)}%{max_concurrent}",
                f"--output={grid_log_dir / 'agent-worker-%A_%a.out'}",
            ]
            if worker_cpus_per_task is not None:
                array_extra_args += [f"--cpus-per-task={worker_cpus_per_task}"]
            worker_script = "agent_worker_array.sbatch"
            worker_script_args = [str(manifest_path), grid_dir_name, tier_cells[0].task, server_job]
        if worker_reservation is not None:
            # Workers use the server's reservation; size max_concurrent to its free GPUs minus the server's.
            array_extra_args += [f"--reservation={worker_reservation}"]
        if worker_time is not None:
            # Short limits help preemptible jobs get backfilled.
            array_extra_args += [f"--time={worker_time}"]
        array_job = sbatch(array_extra_args, worker_script, worker_script_args, env=worker_env)
        print(f"Worker array {array_job} submitted ({len(tier_cells)} cells, "
              f"%{node_throttle if whole_node_only else max_concurrent} throttle"
              f"{' of nodes' if whole_node_only else ''})"
              + (f", depending on server job {server_job}" if not hosted_api else ""))

        if hosted_api:
            # No teardown; the next tier waits for this array.
            all_job_ids += [array_job]
            prev_dependency = f"afterany:{array_job}"
        elif preemptible:
            # sweep_grid_slurm.py resubmits cells without score.json against the same server, up to
            # --max-preempt-generations, then submits teardown.
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
            # Not `sbatch --wait`: Slurm runs it after the array, nothing needs to keep watching.
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

    print(f"All jobs submitted: {', '.join(all_job_ids)}")
    print(f"Track with: squeue -u $USER   /   tail -f slurm/logs/{grid_dir_name}/*.out")
    print(f"Results land in runs/{grid_dir_name}/ as each cell finishes -- check afterward with "
          f"sacct or the analysis scripts, not this command (it doesn't wait around).")
    return 0


def main() -> int:
    profile = _load_cluster_profile()
    parser = argparse.ArgumentParser(description="Run a grid of agent runs from a YAML spec via Slurm (specs/<task>/<name>.yaml).")
    parser.add_argument("spec", help="Spec name without .yaml, e.g. 'redshift-estimation/smoke-small-fp8'")
    parser.add_argument(
        "--partition", default=profile.get("SLURM_PARTITION", "gpu"),
        help=f"Defaults to the active cluster's profile (CLUSTER={profile['_CLUSTER_NAME']} -> "
        f"slurm/clusters/{profile['_CLUSTER_NAME']}.env).",
    )
    parser.add_argument(
        "--account", default=profile.get("SLURM_ACCOUNT") or None,
        help="Slurm --account for every job; empty if the cluster needs none.",
    )
    parser.add_argument(
        "--server-qos", default=profile.get("SERVER_QOS") or None,
        help="QoS for the server job (not preemptible); omitted if empty.",
    )
    parser.add_argument(
        "--worker-qos", default=profile.get("WORKER_QOS") or None,
        help="QoS for worker jobs; omitted if empty. Use --preemptible rather than a preemptible QoS here.",
    )
    parser.add_argument(
        "--worker-cpus-per-task", type=int, default=None,
        help="Worker --cpus-per-task (default 32), for QoS with a lower CPU cap.",
    )
    parser.add_argument(
        "--server-time", default=profile.get("SERVER_TIME") or None,
        help="Server --time; defaults to the profile's SERVER_TIME, else the sbatch file's 24h.",
    )
    parser.add_argument(
        "--server-gres", default=None,
        help="Server --gres. Required for multi-GPU tiers on granular clusters; whole node on whole-node clusters.",
    )
    parser.add_argument(
        "--server-data-parallel-size", type=int, default=None,
        help="Data-parallel replicas for a small tier on a whole node (default GPUS_PER_NODE // tier GPUs). 1 disables.",
    )
    parser.add_argument(
        "--server-cpus-per-task", type=int, default=None,
        help="Server --cpus-per-task (default 16).",
    )
    parser.add_argument(
        "--server-mem", default=None,
        help="Server --mem (default 64G).",
    )
    parser.add_argument(
        "--server-reservation", default=None,
        help="Slurm reservation for the server job (may need a matching --server-qos).",
    )
    parser.add_argument(
        "--worker-reservation", default=None,
        help="Use the server's reservation for workers too (may need a matching --worker-qos).",
    )
    parser.add_argument(
        "--max-concurrent", type=int, default=10,
        help="Concurrent worker cells per tier (array %%-throttle); also the server's --max-num-seqs.",
    )
    parser.add_argument(
        "--after", default=None,
        help="Job id (another grid's teardown) the first server waits for (afterok).",
    )
    parser.add_argument(
        "--preemptible", action="store_true",
        help="Run workers under PREEMPTIBLE_QOS and resubmit preempted cells via sweep_grid_slurm. Single-tier specs, granular clusters only.",
    )
    parser.add_argument(
        "--max-preempt-generations", type=int, default=100,
        help="With --preemptible: maximum resubmission rounds (default 100).",
    )
    parser.add_argument(
        "--worker-time", default=None,
        help="Worker --time (default 4h); shorter limits help preemptible jobs get backfilled.",
    )
    args = parser.parse_args()
    return run_grid_slurm(
        args.spec, args.partition, args.server_qos, args.worker_qos,
        args.max_concurrent, args.worker_cpus_per_task, args.server_time,
        args.server_gres, args.server_cpus_per_task, args.server_mem,
        args.server_reservation, args.worker_reservation, args.after,
        args.account, profile,
        preemptible=args.preemptible, max_preempt_generations=args.max_preempt_generations,
        worker_time=args.worker_time, server_data_parallel_size=args.server_data_parallel_size,
    )


if __name__ == "__main__":
    sys.exit(main())
