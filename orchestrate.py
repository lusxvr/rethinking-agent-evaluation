import argparse
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from agent.logging_utils import STDOUT_TRACE_PATH, TRACE_STDOUT_PREFIX
from axes import AXIS_DEFAULTS, AXIS_LEVELS, RunSpec
from eval.evaluate import evaluate, format_model_choice
from eval.oracle_server import OracleServer
from scripts.analysis.trace_render import render

REPO_ROOT = Path(__file__).resolve().parent
AGENT_SIF = REPO_ROOT / "apptainer" / "agent" / "agent.sif"

# Sandbox cgroup limits. --cpus is a quota, not affinity: the container still sees every host CPU.
SANDBOX_MEMORY = "32g"
SANDBOX_CPUS = os.environ.get("SANDBOX_CPUS", "32")
SANDBOX_PIDS_LIMIT = "512"

# Run cost totals copied from run_end into score.json. The last four are Anthropic-only (0 for vLLM).
RUN_COST_FIELDS = (
    "iterations", "wallclock_s", "prompt_tokens", "completion_tokens",
    "input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens",
)


def load_environment() -> None:
    """Load .env on use, not at import, so the module imports without one."""
    if not load_dotenv(REPO_ROOT / ".env"):
        raise SystemExit(f"{REPO_ROOT / '.env'} not found -- run 'cp .env.example .env' and fill it in")


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"{name} is not set in {REPO_ROOT / '.env'} -- see .env.example for what it should hold")
    return value


def _uv_cache_host_dir() -> Path:
    # Separate from $CACHE_ROOT/uv so sandbox writes cannot affect the operator's cache.
    return Path(f"{_require_env('CACHE_ROOT')}/agent-uv")


class ResourceMonitor:
    """Polls nvidia-smi host-side at physical-GPU granularity."""

    def __init__(self, out_path: Path, interval_s: float = 5.0):
        self._out_path = out_path
        self._interval_s = interval_s
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=self._interval_s * 2)

    def _run(self) -> None:
        with self._out_path.open("a", buffering=1) as f:
            while not self._stop.is_set():
                try:
                    result = subprocess.run(
                        [
                            "nvidia-smi",
                            "--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total,power.draw",
                            "--format=csv,noheader,nounits",
                        ],
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                    rows = [line.split(", ") for line in result.stdout.strip().splitlines() if line]
                    record = {"ts": time.time(), "gpus": rows}
                except Exception as exc:  # nvidia-smi missing, timeout, etc.
                    record = {"ts": time.time(), "error": str(exc)}
                f.write(json.dumps(record) + "\n")
                self._stop.wait(self._interval_s)


def _ensure_agent_env(name: str, model_dir: Path) -> None:
    """Build models/<name>/agent/env/.venv if missing. Delete it by hand to force a rebuild."""
    venv_path = model_dir / "env" / ".venv"
    if venv_path.is_dir():
        return
    print(f"No agent env found for model {name!r} -- building it now (this can take a while)...")
    script = REPO_ROOT / "scripts" / "production" / "build_agent_model_env.sh"
    subprocess.run([str(script), name], check=True)


def check_backbone_model(base_url: str, model_name: str) -> None:
    """Fail before the container starts if the backbone does not serve this run's model."""
    try:
        client = OpenAI(base_url=base_url, api_key="EMPTY", timeout=10.0, max_retries=0)
        served = [model.id for model in client.models.list().data]
    except Exception as exc:
        raise SystemExit(f"Backbone at {base_url} is unreachable ({type(exc).__name__}: {exc})")
    if model_name not in served:
        raise SystemExit(
            f"Backbone at {base_url} serves {served}, not {model_name!r} -- start the server on "
            f"the weights this run's 'model' axis names, or run a spec whose model matches"
        )


def check_anthropic_credentials() -> None:
    """Fail before the container starts if ANTHROPIC_API_KEY is unset. Does not call the API."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit(
            "ANTHROPIC_API_KEY is not set -- add it to .env (see .env.example) before running a claude-* model."
        )


def resolve_task_dir(task: str) -> Path:
    """Return tasks/<task>/, checking that it and its data/ exist."""
    task_dir = REPO_ROOT / "tasks" / task
    if not task_dir.is_dir():
        raise SystemExit(f"No such task: {task_dir}")
    if not (task_dir / "data").is_dir():
        raise SystemExit(
            f"Task {task!r} has no data at {task_dir / 'data'} -- task data is not in git, so build "
            f"it first with the task's own preparation script, solutions/{task}/dev/prepare_data.py"
        )
    return task_dir


def compose_description(task_dir: Path, spec: RunSpec, out_path: Path) -> Path:
    """Write description.md plus this run's info/ snippets to out_path, outside run_dir.

    tasks/<t>/ is never bound wholesale, so the agent cannot read ungranted info/ snippets.
    """
    parts = [(task_dir / "description.md").read_text().rstrip("\n")]
    for name in spec.info_snippets:
        snippet_path = task_dir / "info" / f"{name}.md"
        if not snippet_path.is_file():
            raise SystemExit(
                f"Information level {spec.information!r} needs {snippet_path}, which doesn't exist "
                f"-- task {spec.task!r} can't be run at that level"
            )
        parts.append(snippet_path.read_text().rstrip("\n"))
    out_path.write_text("\n\n".join(parts) + "\n")
    return out_path


def build_apptainer_command(
    spec: RunSpec,
    task_dir: Path,
    run_dir: Path,
    description_path: Path,
    gpu_uuid: str,
    base_url: str,
    verbose: bool,
    web_allowlist: str,
    web_denylist: str,
    oracle_url: str | None,
    provider: str,
) -> tuple[list[str], dict[str, str]]:
    """Return (argv, extra_env) for the agent container.

    extra_env sets NVIDIA_VISIBLE_DEVICES for apptainer itself (--nvccli). The in-container
    --env copies restrict visibility under plain --nv, where all node GPUs are otherwise visible.
    """
    # Backs the container's $HOME and tmp dirs; deleted after the run. Only workspace/ is mounted.
    apptainer_workdir = run_dir / "apptainer-workdir"
    workspace_dir = run_dir / "workspace"
    apptainer_workdir.mkdir(exist_ok=True)
    workspace_dir.mkdir(exist_ok=True)
    uv_cache_host_dir = _uv_cache_host_dir()
    uv_cache_host_dir.mkdir(parents=True, exist_ok=True)  # apptainer errors on a missing bind source

    # API key via --env-file, not --env, so it never appears in argv (visible via ps).
    env_file_args = []
    if provider == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise SystemExit("ANTHROPIC_API_KEY is not set -- see check_anthropic_credentials")
        secrets_path = apptainer_workdir / "secrets.env"
        # Mode 0o600 at creation, so the file is never briefly world-readable.
        fd = os.open(secrets_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(f"ANTHROPIC_API_KEY={api_key}\n")
        env_file_args = ["--env-file", str(secrets_path)]

    # Each model in tasks/<task>/models.txt is mounted read-only at /models/<name>.
    model_mounts = []
    models_file = task_dir / "models.txt"
    if models_file.is_file():
        for name in models_file.read_text().splitlines():
            name = name.strip()
            if not name:
                continue
            model_dir = REPO_ROOT / "models" / name / "agent"
            if not model_dir.is_dir():
                raise SystemExit(f"models.txt lists {name!r} but {model_dir} doesn't exist")
            _ensure_agent_env(name, model_dir)
            model_mounts += ["--bind", f"{model_dir}:/models/{name}:ro"]

    # Sampling preset depends on this cell's model and harness levels.
    sampling_env = [arg for name, value in spec.sampling.items() for arg in ("--env", f"{name.upper()}={value}")]

    # cgroup limits need a systemd --user session, which Slurm batch jobs lack.
    cgroup_flags = (
        [] if "SLURM_JOB_ID" in os.environ else
        ["--memory", SANDBOX_MEMORY, "--cpus", SANDBOX_CPUS, "--pids-limit", SANDBOX_PIDS_LIMIT]
    )
    # --nvccli fails under setuid apptainer; APPTAINER_GPU_FLAG=nv falls back to --nv, which
    # needs --writable-tmpfs explicitly and relies on the --env GPU restriction below.
    gpu_flags = (
        ["--nv", "--writable-tmpfs"] if os.environ.get("APPTAINER_GPU_FLAG") == "nv" else ["--nvccli"]
    )
    argv = [
        "apptainer", "exec",
        # --contain: only explicit binds. --cleanenv: only the --env values below reach the agent.
        "--contain", "--cleanenv", *gpu_flags,
        *cgroup_flags,
        # Replaces the default 64MB tmpfs behind $HOME and tmp.
        "--workdir", str(apptainer_workdir),
        # agent.cli resolves from /app, where agent.def installs it.
        "--pwd", "/app",
        # Explicit sub-binds, so info/ and models.txt never reach the agent unfiltered.
        "--bind", f"{task_dir / 'data'}:/task/data:ro",
        "--bind", f"{description_path}:/task/description.md:ro",
        "--bind", f"{workspace_dir}:/agent_run/workspace",
        "--bind", f"{uv_cache_host_dir}:/uv_cache",
        *model_mounts,
        *env_file_args,
        "--env", f"PROVIDER={provider}",
        "--env", f"VLLM_BASE_URL={base_url}",
        # Axis labels and their resolved values; the container resolves nothing itself.
        "--env", f"INFORMATION={spec.information}",
        "--env", f"HARNESS={spec.harness}",
        "--env", f"VERIFICATION={spec.verification}",
        "--env", f"BUDGET={spec.budget}",
        "--env", f"MODEL={spec.model}",
        "--env", f"REPLICATE={spec.replicate}",
        "--env", f"MODEL_NAME={spec.model_name}",
        "--env", f"MAX_ITERATIONS={spec.max_iterations}",
        "--env", f"MAX_DURATION_S={spec.max_duration_s}",
        *sampling_env,
        "--env", f"AGENT_GPU_UUID={gpu_uuid}",
        # Restricts GPU visibility inside the container (see docstring).
        "--env", f"NVIDIA_VISIBLE_DEVICES={gpu_uuid}",
        "--env", f"CUDA_VISIBLE_DEVICES={gpu_uuid}",
        "--env", "TASK_DIR=/task",
        "--env", f"TASK_NAME={spec.task}",
        "--env", "RUN_DIR=/agent_run",
        "--env", f"TRACE_PATH={STDOUT_TRACE_PATH}",
        "--env", "UV_CACHE_DIR=/uv_cache",
        # A bare pip install would fill the 64MB tmpfs overlay; refuse it instead.
        "--env", "PIP_REQUIRE_VIRTUALENV=1",
        "--env", f"VERBOSE={'1' if verbose else '0'}",
        "--env", f"WEB_ALLOWLIST={web_allowlist}",
        "--env", f"WEB_DENYLIST={web_denylist}",
        # Set only with --oracle; its presence enables oracle_check in agent/tools.py.
        *(["--env", f"ORACLE_URL={oracle_url}"] if oracle_url else []),
        str(AGENT_SIF),
        "python", "-m", "agent.cli",
    ]
    return argv, {"NVIDIA_VISIBLE_DEVICES": gpu_uuid}


def _run_agent_container(cmd: list[str], env: dict[str, str], trace_path: Path) -> tuple[int, dict | None]:
    """Run the container, writing prefixed stdout records to trace_path. Returns (exit code, run_end).

    The trace stays host-side so the agent cannot read discarded reasoning or condition labels.
    trace_path is created on the first record, so a container that dies early leaves no trace.
    """
    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    trace_file = None
    run_end = None
    try:
        for line in process.stdout:
            if not line.startswith(TRACE_STDOUT_PREFIX):
                print(line, end="")
                continue
            record_json = line[len(TRACE_STDOUT_PREFIX):]
            if trace_file is None:
                trace_file = trace_path.open("a", buffering=1)
            trace_file.write(record_json)
            try:
                record = json.loads(record_json)
            except json.JSONDecodeError:
                continue
            if record.get("event") == "run_end":
                run_end = record
    finally:
        if trace_file is not None:
            trace_file.close()
    return process.wait(), run_end


def _host_submission_path(run_end: dict | None, run_dir: Path) -> Path | None:
    """The reported submission path rebased onto run_dir, or None."""
    if run_end is None or run_end.get("status") != "finished" or not run_end.get("submission_path"):
        return None
    path = Path(run_end["submission_path"])
    try:
        return run_dir / path.relative_to("/agent_run")
    except ValueError:
        return path


_WORKSPACE_KEEP_SUFFIXES = {".py", ".csv"}
# Directories pruned even if they contain .py/.csv files (installed packages).
_WORKSPACE_SKIP_DIR_NAMES = {".venv", "venv", "site-packages", "__pycache__", ".cache", ".uv"}


def _prune_workspace(workspace_dir: Path, submission_path: Path | None = None) -> None:
    """Delete everything under workspace/ except .py/.csv files and the submission."""
    for path in workspace_dir.rglob("*"):
        if path.is_dir():
            continue
        keep = path == submission_path or (
            path.suffix in _WORKSPACE_KEEP_SUFFIXES and not (_WORKSPACE_SKIP_DIR_NAMES & set(path.parts))
        )
        if not keep:
            path.unlink(missing_ok=True)
    # Deepest first, so a dir empties out before its parent is checked.
    for d in sorted((p for p in workspace_dir.rglob("*") if p.is_dir()), key=lambda p: -len(p.parts)):
        try:
            d.rmdir()
        except OSError:
            pass  # not empty


def _existing_run_end(run_dir: Path) -> dict | None:
    """The run_end record from run_dir/trace.jsonl, or None if the run never completed."""
    trace_path = run_dir / "trace.jsonl"
    if not trace_path.is_file():
        return None
    run_end = None
    for line in trace_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("event") == "run_end":
            run_end = record
    return run_end


def _write_trace_report(run_dir: Path, base_url: str, model: str) -> None:
    trace_path = run_dir / "trace.jsonl"
    if not trace_path.is_file():
        return
    # After evaluation, so the report includes the score.
    markdown, analysis = render(trace_path, base_url=base_url, model=model)
    (run_dir / "trace_report.md").write_text(markdown)
    (run_dir / "analysis_generated.json").write_text(json.dumps(analysis, indent=2))
    print(f"Wrote {run_dir / 'trace_report.md'} and {run_dir / 'analysis_generated.json'}")


def run_evaluation(task: str, run_dir: Path, run_end: dict | None) -> None:
    """Write score.json for every run, including runs without a submission."""
    submission_path = _host_submission_path(run_end, run_dir)
    if submission_path is None:
        status = run_end.get("status") if run_end else "no run_end record"
        result = {"valid": False, "errors": [f"no submission: run ended with status {status!r}"]}
    else:
        result = evaluate(
            task,
            submission_path,
            model_used=run_end.get("model_used"),
            expected_score=run_end.get("expected_score"),
            verification_evidence=run_end.get("verification_evidence"),
        )
    result["status"] = run_end.get("status") if run_end else None
    for field in RUN_COST_FIELDS:
        result[field] = run_end.get(field) if run_end else None
    if result["valid"]:
        print(f"Evaluation: {result['metric']} {result['score']:.4f} ({result['n']} rows)")
        gap = f", gap_closed: {result['gap_closed']:.4f}" if result["gap_closed"] is not None else ""
        print(f"Regime: {result['regime']}{gap}")
        if "model_choice_correct" in result:
            print(f"Model choice: {format_model_choice(result)}")
    else:
        print("Evaluation: no valid submission:")
        for e in result["errors"]:
            print(f"  - {e}")

    eval_path = run_dir / "score.json"
    eval_path.write_text(json.dumps(result, indent=2))
    print(f"Wrote {eval_path}")


def launch_run(
    spec: RunSpec,
    run_dir: Path,
    gpu_uuid: str,
    base_url: str,
    verbose: bool,
    web_allowlist: str = "",
    web_denylist: str = "",
    oracle: bool = False,
) -> int:
    """Run one cell (container, evaluation, trace render) and return the exit code.

    Idempotent: a scored run_dir is a no-op, one with run_end but no score.json is only re-scored,
    and any other existing run_dir is discarded and rerun.
    """
    load_environment()
    task_dir = resolve_task_dir(spec.task)

    if (run_dir / "score.json").is_file():
        print(f"Run directory {run_dir} already has score.json -- nothing to do")
        return 0

    run_end = _existing_run_end(run_dir)
    if run_end is not None:
        print(f"Run directory {run_dir} already has a finished trace (run_end) but no score.json "
              f"-- re-running evaluation only, not the agent")
        run_evaluation(spec.task, run_dir, run_end)
        _write_trace_report(run_dir, base_url, spec.model_name)
        return 0

    if run_dir.is_dir() and any(run_dir.iterdir()):
        print(f"Run directory {run_dir} exists but the agent never reached run_end -- discarding and starting fresh")
        shutil.rmtree(run_dir)

    if spec.provider == "anthropic":
        check_anthropic_credentials()
    else:
        check_backbone_model(base_url, spec.model_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}")

    monitor = ResourceMonitor(run_dir / "resources.jsonl")
    monitor.start()
    # The oracle server needs workspace/ before the container creates it.
    workspace_dir = run_dir / "workspace"
    workspace_dir.mkdir(exist_ok=True)
    oracle_server = OracleServer(spec.task, workspace_dir) if oracle else None
    # Staged outside run_dir and archived into it after the container exits.
    staging = Path(tempfile.mkdtemp(prefix="auto-research-input-"))
    try:
        if oracle_server is not None:
            oracle_server.start()
        description_path = compose_description(task_dir, spec, staging / "description.md")
        cmd, extra_env = build_apptainer_command(
            spec, task_dir, run_dir, description_path, gpu_uuid, base_url, verbose,
            web_allowlist, web_denylist, oracle_server.url if oracle_server else None,
            spec.provider,
        )
        print("Launching agent container:\n  " + " ".join(cmd))
        returncode, run_end = _run_agent_container(cmd, {**os.environ, **extra_env}, run_dir / "trace.jsonl")
    finally:
        monitor.stop()
        if oracle_server is not None:
            oracle_server.stop()
        if (staging / "description.md").is_file():
            shutil.copy(staging / "description.md", run_dir / "description.md")
        shutil.rmtree(staging, ignore_errors=True)
        shutil.rmtree(run_dir / "apptainer-workdir", ignore_errors=True)

    print(f"Agent container exited with code {returncode}. Run artifacts in {run_dir}")

    # Unconditional, so every run directory gets a score.json.
    try:
        run_evaluation(spec.task, run_dir, run_end)
    except Exception as exc:
        print(f"warning: evaluation failed ({type(exc).__name__}: {exc}); continuing to trace render")

    _write_trace_report(run_dir, base_url, spec.model_name)

    if "SLURM_JOB_ID" in os.environ:
        _prune_workspace(run_dir / "workspace", _host_submission_path(run_end, run_dir))

    return returncode


def _new_run_dir(runs_root: Path) -> Path:
    """Claim a fresh runs/<utc-timestamp> directory atomically, adding a suffix on collision."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    for attempt in range(1, 100):
        run_dir = runs_root / (stamp if attempt == 1 else f"{stamp}-{attempt}")
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
            return run_dir
        except FileExistsError:
            continue
    raise SystemExit(f"Could not claim a run directory under {runs_root} for {stamp}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch a sandboxed agent run against a task.")
    parser.add_argument("--task", required=True, help="Task name under tasks/, e.g. redshift-estimation")
    parser.add_argument("--base-url", default=f"http://localhost:{os.environ.get('VLLM_PORT', '61000')}/v1")
    parser.add_argument(
        "--information",
        default=AXIS_DEFAULTS["information"],
        choices=AXIS_LEVELS["information"],
        help="Cumulative information ladder: none, identity, interface, protocol.",
    )
    parser.add_argument(
        "--harness",
        default=AXIS_DEFAULTS["harness"],
        choices=AXIS_LEVELS["harness"],
        help="act-only: no reasoning; think-act: reasoning discarded each turn; react: reasoning kept.",
    )
    parser.add_argument(
        "--verification",
        default=AXIS_DEFAULTS["verification"],
        choices=AXIS_LEVELS["verification"],
        help="none, asked, reported (finish() reports expected score and evidence), binding.",
    )
    parser.add_argument(
        "--budget",
        default=AXIS_DEFAULTS["budget"],
        choices=AXIS_LEVELS["budget"],
        help="Wall-clock budget level (see axes.py).",
    )
    parser.add_argument(
        "--model",
        default=AXIS_DEFAULTS["model"],
        choices=AXIS_LEVELS["model"],
        help="Backbone model level (see axes.py).",
    )
    parser.add_argument("--replicate", type=int, default=1, help="Replicate index.")
    parser.add_argument(
        "--gpu",
        default=None,
        help="GPU or MIG UUID to run on. Defaults to .env's AGENT_GPU_UUID.",
    )
    parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print live progress from the container.",
    )
    parser.add_argument(
        "--web-allowlist",
        default="",
        help="Comma-separated hosts web_fetch may reach, including subdomains. Empty allows all.",
    )
    parser.add_argument(
        "--web-denylist",
        default="",
        help="Comma-separated hosts web_fetch must never reach. Loopback is always blocked.",
    )
    parser.add_argument(
        "--oracle",
        action="store_true",
        default=False,
        help="Grant oracle_check, which scores a submission against the reference mid-run.",
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
    load_environment()
    gpu_uuid = args.gpu or _require_env("AGENT_GPU_UUID")
    # Checked before claiming a directory, so a failed check leaves no empty run dir.
    if spec.provider == "anthropic":
        check_anthropic_credentials()
    else:
        check_backbone_model(args.base_url, spec.model_name)
    run_dir = _new_run_dir(REPO_ROOT / "runs")
    return launch_run(
        spec, run_dir, gpu_uuid, args.base_url, args.verbose,
        args.web_allowlist, args.web_denylist, args.oracle,
    )


if __name__ == "__main__":
    raise SystemExit(main())
