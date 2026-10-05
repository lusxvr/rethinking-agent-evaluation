"""Experiment axes, their levels, and RunSpec (one grid cell). Host-side only."""

from dataclasses import dataclass

from agent.config import HARNESS_LEVELS, VERIFICATION_LEVELS, generates_reasoning

# Cumulative ladder; each level above "none" names the tasks/<t>/info/<name>.md snippet it adds.
INFORMATION_LEVELS = ("none", "identity", "interface", "protocol")

# budget: name -> wall-clock seconds
BUDGET_LEVELS = {
    "short": 300,
    "medium": 600,
    "long": 1200,
}
# Runaway guard, not part of the axis: about 3x the fastest observed iteration rate.
ITERATION_CAPS = {
    "short": 100,
    "medium": 200,
    "long": 400,
}

# Each model card's "general tasks" sampling preset per thinking mode.
_SHARED = {"top_k": 20, "min_p": 0.0, "repetition_penalty": 1.0}  # identical in every Qwen3.5 preset
_NON_THINKING = {**_SHARED, "temperature": 0.7, "top_p": 0.8, "presence_penalty": 1.5}
_THINKING = {**_SHARED, "temperature": 1.0, "top_p": 0.95, "presence_penalty": 1.5}
_THINKING_397B = {**_SHARED, "temperature": 0.6, "top_p": 0.95, "presence_penalty": 0.0}

# Step-3.7-Flash publishes no preset; StepFun's recommendation for Step-3.5-Flash is used
# (huggingface.co/stepfun-ai/Step-3.5-Flash/discussions/3). Other parameters stay at vLLM defaults.
_STEP37_FLASH_SHARED = {"top_k": -1, "min_p": 0.0, "repetition_penalty": 1.0, "presence_penalty": 0.0}
_STEP37_FLASH_NON_THINKING = {**_STEP37_FLASH_SHARED, "temperature": 0.6, "top_p": 0.95}
_STEP37_FLASH_THINKING = {**_STEP37_FLASH_SHARED, "temperature": 1.0, "top_p": 0.95}

# Sonnet 5 rejects sampling parameters; effort is its only lever. One fixed effort for both modes
# avoids confounding harness with effort; "high" is the documented maximum for disabled thinking
# on Opus 5 (unstated for Sonnet 5). See docs/anthropic-integration.md.
_CLAUDE_SONNET5 = {"effort": "high"}

# model: name -> (served id, thinking preset, non-thinking preset). Each level uses its own vendor
# preset, since presets differ within a family. For Claude the served id is the API model string.
MODEL_LEVELS = {
    "qwen35-35b-a3b-fp8": ("Qwen/Qwen3.5-35B-A3B-FP8", _THINKING, _NON_THINKING),
    "qwen35-122b-a10b-fp8": ("Qwen/Qwen3.5-122B-A10B-FP8", _THINKING, _NON_THINKING),
    "qwen35-397b-a17b-fp8": ("Qwen/Qwen3.5-397B-A17B-FP8", _THINKING_397B, _NON_THINKING),
    "claude-sonnet-5": ("claude-sonnet-5", _CLAUDE_SONNET5, _CLAUDE_SONNET5),
    # Its chat template always opens a <think> block, so thinking cannot be disabled: specs for
    # this tier must omit act-only. reasoning_effort resolves to StepFun's default, "medium".
    "step37-198b-a11b-fp8": ("stepfun-ai/Step-3.7-Flash-FP8", _STEP37_FLASH_THINKING, _STEP37_FLASH_NON_THINKING),
}

# GPUs per vLLM server; must match apptainer/vllm/resolve_tier.sh. Used by run_grid_slurm.py to
# size server jobs on whole-node clusters.
MODEL_GPU_COUNT = {
    "qwen35-35b-a3b-fp8": 1,
    "qwen35-122b-a10b-fp8": 2,
    "qwen35-397b-a17b-fp8": 8,
    "step37-198b-a11b-fp8": 4,
    # Hosted API; the agent sandbox still gets its own GPU.
    "claude-sonnet-5": 0,
}

# Family prefixes, stripped for display and pricing lookup.
_MODEL_FAMILY_PREFIXES = ("qwen35-", "step37-", "claude-")


def provider_for(model: str) -> str:
    """LLM backend (vllm or anthropic), from the family prefix."""
    return "anthropic" if model.startswith("claude-") else "vllm"


# First-party API list prices, USD per 1M tokens (input, output): Alibaba Cloud Model Studio
# (Qwen3.5), StepFun (Step-3.7-Flash), Anthropic (Sonnet 5).
MODEL_PRICING_USD_PER_1M = {
    "35b-a3b": (0.25, 2.00),
    "122b-a10b": (0.40, 3.20),
    "397b-a17b": (0.60, 3.60),
    "198b-a11b": (0.20, 1.15),
    "sonnet-5": (2.00, 10.00),
}

# Anthropic prompt-cache price multipliers on the input price.
ANTHROPIC_CACHE_WRITE_MULTIPLIER = 1.25
ANTHROPIC_CACHE_READ_MULTIPLIER = 0.10


def _pricing_key(model: str) -> str:
    """Model level without family prefix and "-fp8" suffix, the MODEL_PRICING_USD_PER_1M key."""
    for prefix in _MODEL_FAMILY_PREFIXES:
        if model.startswith(prefix):
            return model.removeprefix(prefix).removesuffix("-fp8")
    raise ValueError(f"Model {model!r} has no known family prefix ({_MODEL_FAMILY_PREFIXES})")


def estimated_cost_usd(model: str, prompt_tokens: float, completion_tokens: float) -> float:
    """Run cost at first-party list prices, without caching discounts; comparable across models.

    For Claude this is an upper bound; see anthropic_actual_cost_usd for the billed cost.
    """
    input_price, output_price = MODEL_PRICING_USD_PER_1M[_pricing_key(model)]
    return prompt_tokens / 1e6 * input_price + completion_tokens / 1e6 * output_price


def anthropic_actual_cost_usd(
    input_tokens: float, cache_creation_input_tokens: float, cache_read_input_tokens: float,
    output_tokens: float, model: str = "claude-sonnet-5",
) -> float:
    """Billed cost of a Claude run from its score.json token fields, with cache pricing."""
    input_price, output_price = MODEL_PRICING_USD_PER_1M[_pricing_key(model)]
    return (
        input_tokens / 1e6 * input_price
        + cache_creation_input_tokens / 1e6 * input_price * ANTHROPIC_CACHE_WRITE_MULTIPLIER
        + cache_read_input_tokens / 1e6 * input_price * ANTHROPIC_CACHE_READ_MULTIPLIER
        + output_tokens / 1e6 * output_price
    )


AXIS_LEVELS: dict[str, tuple[str, ...]] = {
    "information": INFORMATION_LEVELS,
    "harness": HARNESS_LEVELS,
    "verification": VERIFICATION_LEVELS,
    "budget": tuple(BUDGET_LEVELS),
    "model": tuple(MODEL_LEVELS),
}
AXIS_NAMES = tuple(AXIS_LEVELS)

# Display only: strips the model family prefix from axis labels.
def display_level(axis: str, level: str) -> str:
    if axis != "model":
        return level
    for prefix in _MODEL_FAMILY_PREFIXES:
        if level.startswith(prefix):
            return level.removeprefix(prefix)
    return level

# Default level per axis, shared by RunSpec and orchestrate.py.
AXIS_DEFAULTS = {
    "information": "protocol",
    "harness": "react",
    "verification": "asked",
    "budget": "short",
    "model": "qwen35-35b-a3b-fp8",
}


def info_snippets_for(level: str) -> tuple[str, ...]:
    """The info/ snippets an information level grants: its own and every level below it."""
    if level not in INFORMATION_LEVELS:
        raise ValueError(f"Unknown information level {level!r}, must be one of {list(INFORMATION_LEVELS)}")
    return INFORMATION_LEVELS[1 : INFORMATION_LEVELS.index(level) + 1]


def sampling_for(model: str, harness: str) -> dict[str, float | int]:
    """The model's vendor preset for the thinking mode of the harness level."""
    _, thinking, non_thinking = MODEL_LEVELS[model]
    # Copy, since presets are shared between levels.
    return dict(thinking if generates_reasoning(harness) else non_thinking)


# Defaults for consumers without a RunSpec (dev baselines, trace_render).
DEFAULT_MODEL_NAME = MODEL_LEVELS[AXIS_DEFAULTS["model"]][0]
DEFAULT_SAMPLING = sampling_for(AXIS_DEFAULTS["model"], AXIS_DEFAULTS["harness"])


@dataclass(frozen=True)
class RunSpec:
    """One grid cell. Construction validates every level."""

    task: str
    information: str = AXIS_DEFAULTS["information"]
    harness: str = AXIS_DEFAULTS["harness"]
    verification: str = AXIS_DEFAULTS["verification"]
    budget: str = AXIS_DEFAULTS["budget"]
    model: str = AXIS_DEFAULTS["model"]
    replicate: int = 1

    def __post_init__(self) -> None:
        for axis, levels in AXIS_LEVELS.items():
            value = getattr(self, axis)
            if value not in levels:
                raise ValueError(f"Unknown {axis} level {value!r}, must be one of {list(levels)}")
        if self.replicate < 1:
            raise ValueError(f"replicate must be >= 1, got {self.replicate}")

    @property
    def run_name(self) -> str:
        """Directory name of this cell inside a grid run."""
        levels = "__".join(getattr(self, axis) for axis in AXIS_NAMES)
        return f"{self.task}__{levels}__r{self.replicate}"

    @property
    def info_snippets(self) -> tuple[str, ...]:
        return info_snippets_for(self.information)

    @property
    def model_name(self) -> str:
        return MODEL_LEVELS[self.model][0]

    @property
    def provider(self) -> str:
        return provider_for(self.model)

    @property
    def sampling(self) -> dict[str, float | int]:
        return sampling_for(self.model, self.harness)

    @property
    def max_duration_s(self) -> int:
        return BUDGET_LEVELS[self.budget]

    @property
    def max_iterations(self) -> int:
        """Runaway guard, not the budget."""
        return ITERATION_CAPS[self.budget]
