import os
from dataclasses import dataclass
from pathlib import Path

# The two axes the container interprets. (*) marks the default (axes.AXIS_DEFAULTS).
HARNESS_LEVELS = ("act-only", "think-act", "react")
#   act-only  -- no reasoning (ReAct's Act-only ablation, Yao et al. 2022)
#   think-act -- reasoning generated each turn, discarded before the next
#   react (*) -- reasoning kept in the persisted message

VERIFICATION_LEVELS = ("none", "asked", "reported", "binding")
#   none      -- all verification language stripped
#   asked (*) -- the plain ask to verify, with nothing required back
#   reported  -- finish() additionally requires expected_score + verification_evidence
#   binding   -- reported, plus: don't submit an output your own check didn't convince you of


def generates_reasoning(level: str) -> bool:
    """Whether this harness level generates reasoning (enable_thinking and sampling preset)."""
    if level not in HARNESS_LEVELS:
        raise ValueError(f"Unknown harness level {level!r}, must be one of {HARNESS_LEVELS}")
    return level != "act-only"


def persists_reasoning(level: str) -> bool:
    """Whether this level keeps reasoning in the persisted message (ReAct)."""
    if level not in HARNESS_LEVELS:
        raise ValueError(f"Unknown harness level {level!r}, must be one of {HARNESS_LEVELS}")
    return level == "react"


def is_verified(level: str) -> bool:
    """Whether this level asks finish() for expected_score and verification_evidence."""
    if level not in VERIFICATION_LEVELS:
        raise ValueError(f"Unknown verification level {level!r}, must be one of {VERIFICATION_LEVELS}")
    return level in ("reported", "binding")


@dataclass(frozen=True)
class Config:
    vllm_base_url: str
    model_name: str
    max_iterations: int
    max_duration_s: int
    task_dir: Path
    task_name: str
    models_dir: Path
    run_dir: Path
    workspace_dir: Path
    trace_path: Path
    bash_timeout_s: int
    verbose: bool
    web_allowlist: frozenset[str]
    web_denylist: frozenset[str]
    # Set only when oracle_check is granted.
    oracle_url: str | None
    gpu_uuid: str
    # vLLM-only sampling fields; None for Anthropic.
    temperature: float | None
    top_p: float | None
    top_k: int | None
    min_p: float | None
    presence_penalty: float | None
    repetition_penalty: float | None
    # Axis labels; orchestrate.py also passes their resolved values above.
    information: str
    harness: str
    verification: str
    budget: str
    model: str
    replicate: int
    # "vllm" or "anthropic", resolved host-side by axes.provider_for.
    provider: str = "vllm"
    # Anthropic-only output_config.effort; None for vLLM.
    effort: str | None = None

    @classmethod
    def from_env(cls) -> "Config":
        run_dir = Path(os.environ.get("RUN_DIR", "/agent_run"))
        workspace_dir = run_dir / "workspace"

        verification = _require("VERIFICATION")
        if verification not in VERIFICATION_LEVELS:
            raise ValueError(f"Unknown VERIFICATION {verification!r}, must be one of {VERIFICATION_LEVELS}")
        harness = _require("HARNESS")
        if harness not in HARNESS_LEVELS:
            raise ValueError(f"Unknown HARNESS {harness!r}, must be one of {HARNESS_LEVELS}")

        # PROVIDER decides which sampling fields are required.
        provider = os.environ.get("PROVIDER", "vllm")
        if provider not in ("vllm", "anthropic"):
            raise ValueError(f"Unknown PROVIDER {provider!r}, must be 'vllm' or 'anthropic'")
        if provider == "vllm":
            temperature = float(_require("TEMPERATURE"))
            top_p = float(_require("TOP_P"))
            top_k = int(_require("TOP_K"))
            min_p = float(_require("MIN_P"))
            presence_penalty = float(_require("PRESENCE_PENALTY"))
            repetition_penalty = float(_require("REPETITION_PENALTY"))
            effort = None
        else:
            temperature = top_p = top_k = min_p = presence_penalty = repetition_penalty = None
            effort = _require("EFFORT")

        return cls(
            vllm_base_url=os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1"),
            model_name=_require("MODEL_NAME"),
            max_iterations=int(_require("MAX_ITERATIONS")),
            max_duration_s=int(_require("MAX_DURATION_S")),
            task_dir=Path(os.environ.get("TASK_DIR", "/task")),
            task_name=os.environ.get("TASK_NAME", "unknown"),
            models_dir=Path(os.environ.get("MODELS_DIR", "/models")),
            run_dir=run_dir,
            workspace_dir=workspace_dir,
            # STDOUT_TRACE_PATH sends records to stdout for orchestrate.py to write host-side.
            trace_path=Path(os.environ.get("TRACE_PATH") or run_dir / "trace.jsonl"),
            bash_timeout_s=int(os.environ.get("BASH_TIMEOUT_S", "300")),
            verbose=os.environ.get("VERBOSE", "1") not in ("0", "false", "False", ""),
            # Empty allowlist allows any host not on the denylist.
            web_allowlist=_parse_host_list(os.environ.get("WEB_ALLOWLIST", "")),
            web_denylist=_parse_host_list(os.environ.get("WEB_DENYLIST", "")),
            oracle_url=os.environ.get("ORACLE_URL") or None,
            gpu_uuid=os.environ.get("AGENT_GPU_UUID", "unknown"),
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            presence_penalty=presence_penalty,
            repetition_penalty=repetition_penalty,
            information=_require("INFORMATION"),
            harness=harness,
            verification=verification,
            budget=_require("BUDGET"),
            model=_require("MODEL"),
            replicate=int(_require("REPLICATE")),
            provider=provider,
            effort=effort,
        )


def _require(name: str) -> str:
    """Read from the environment. No defaults for axis values, so axes.py stays the only source."""
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"{name} is not set -- the container is normally launched by orchestrate.py, which sets it")
    return value


def _parse_host_list(value: str) -> frozenset[str]:
    return frozenset(host.strip().lower() for host in value.split(",") if host.strip())
