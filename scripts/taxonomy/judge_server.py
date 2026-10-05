"""Taxonomy judge server lifecycle: submit vllm_server.sbatch, wait for readiness, read resource usage."""

import json
import re
import subprocess
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
JUDGE_GPUS = 4


def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def _profile_args() -> list[str]:
    """Partition, QoS, account, GPUs, CPUs, memory, time and internet flag from the cluster profile."""
    import shlex

    from scripts.production.slurm_utils import gres, load_cluster_profile

    profile = load_cluster_profile()
    args = ["--partition", profile.get("SLURM_PARTITION") or "gpu", f"--gres={gres(profile, JUDGE_GPUS)}"]
    if profile.get("WHOLE_NODE_ONLY") == "1":
        args += [f"--cpus-per-task={profile['CORES_PER_NODE']}", "--mem=0"]
    else:
        args += ["--cpus-per-task=32", "--mem=256G"]
    args += ["--time", profile["SERVER_TIME"]] if profile.get("SERVER_TIME") else []
    args += ["--qos", profile["SERVER_QOS"]] if profile.get("SERVER_QOS") else []
    args += ["--account", profile["SLURM_ACCOUNT"]] if profile.get("SLURM_ACCOUNT") else []
    return args + shlex.split(profile.get("SERVER_NEEDS_INTERNET_FLAG") or "")


def submit_server(args) -> str:
    # server_max_num_seqs sizes a server shared by several clients (run_stage2.sh); max_num_seqs
    # remains each client's own concurrency.
    server_max_num_seqs = getattr(args, "server_max_num_seqs", None) or args.max_num_seqs
    export = [f"VLLM_MAX_NUM_SEQS={server_max_num_seqs}"]
    if args.ready_timeout_s:
        export.append(f"VLLM_READY_TIMEOUT_S={args.ready_timeout_s}")
    if getattr(args, "gpu_mem_util", None):
        export.append(f"VLLM_GPU_MEM_UTIL={args.gpu_mem_util}")

    (REPO_ROOT / "slurm" / "logs" / "taxonomy-judge").mkdir(parents=True, exist_ok=True)
    log_path = "slurm/logs/taxonomy-judge/vllm-server-%j.out"
    cmd = ["sbatch", f"--export=ALL,{','.join(export)}", f"--output={log_path}", *_profile_args()]
    if getattr(args, "dependency", None):
        cmd.append(f"--dependency={args.dependency}")
    cmd += ["scripts/taxonomy/vllm_server.sbatch", args.tier, str(args.max_model_len)]
    print(f"[{args.label}] submitting: {' '.join(cmd)}")
    result = _run(cmd, cwd=REPO_ROOT)
    if result.returncode != 0:
        raise SystemExit(f"sbatch failed: {result.stdout}\n{result.stderr}")
    m = re.search(r"Submitted batch job (\d+)", result.stdout)
    if not m:
        raise SystemExit(f"could not parse job id from: {result.stdout}")
    job_id = m.group(1)
    print(f"[{args.label}] job {job_id} submitted, log at {log_path.replace('%j', job_id)}")
    return job_id


def wait_ready(args, job_id: str, cache_root: str, timeout_s: int) -> Path:
    log_path = REPO_ROOT / "slurm" / "logs" / "taxonomy-judge" / f"vllm-server-{job_id}.out"
    addr_file = Path(cache_root) / f"taxonomy-judge-{job_id}.addr"
    deadline = time.time() + timeout_s
    last_state = None
    while time.time() < deadline:
        state = _run(["squeue", "-j", job_id, "-h", "-o", "%T"]).stdout.strip()
        if state != last_state:
            print(f"[{args.label}] job {job_id} state={state or '(gone)'}")
            last_state = state
        if state == "" and not addr_file.exists():
            raise SystemExit(f"job {job_id} disappeared from the queue before becoming ready -- check sacct/{log_path}")
        if addr_file.exists():
            print(f"[{args.label}] ready ({addr_file.read_text().strip()})")
            return log_path
        time.sleep(15)
    raise SystemExit(f"[{args.label}] server did not become ready within {timeout_s}s -- check {log_path}")


def capture_resources(args, job_id: str, log_path: Path, addr: str) -> dict:
    text = log_path.read_text(errors="replace")
    port = addr.rsplit(":", 1)[1]
    gpu_probe = _run(["srun", f"--jobid={job_id}", "--overlap", "nvidia-smi",
                       "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"])
    per_gpu_mib = [int(line.split(",")[0]) for line in gpu_probe.stdout.strip().splitlines()]

    def find(pattern: str, group: int = 1, cast=str):
        m = re.search(pattern, text)
        return cast(m.group(group)) if m else None

    # The server already checked itself; a failure here is usually the srun --overlap step racing
    # Slurm, so retry.
    model_id = None
    for attempt in range(3):
        served_model = _run(["srun", f"--jobid={job_id}", "--overlap", "curl", "-sf",
                              f"http://localhost:{port}/v1/models"])
        if served_model.returncode == 0:
            model_id = json.loads(served_model.stdout)["data"][0]["id"]
            break
        if attempt < 2:
            time.sleep(5)

    return {
        "gpus": len(per_gpu_mib) or JUDGE_GPUS,
        "vram_used_mib_per_gpu": per_gpu_mib,
        "weights_gib_per_gpu": find(r"Model loading took ([\d.]+) GiB memory", cast=float),
        "kv_cache_gib_per_gpu": find(r"Available KV cache memory: ([\d.]+) GiB", cast=float),
        "max_model_len": find(r"'max_model_len': (\d+)", cast=int),
        "kv_cache_tokens": find(r"GPU KV cache size: ([\d,]+) tokens", cast=lambda s: int(s.replace(",", ""))),
        "max_concurrency_at_full_context": find(
            r"Maximum concurrency for [\d,]+ tokens per request: ([\d.]+)x", cast=float),
        "served_model_id": model_id,
    }
