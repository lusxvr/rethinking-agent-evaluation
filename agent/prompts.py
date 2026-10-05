"""The agent's system and user prompts. Tool bullets come from agent/tools.py."""

from agent.tools import active_tools, finish_parameters

_COUNT_WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")

_ACCEPTANCE_CLAUSE = (
    " Do not call finish with an output you do not believe is correct -- if your own check leaves "
    "you unconvinced, keep working within your remaining budget instead of submitting anyway."
)


def _finish_bullet(level: str) -> str:
    """The finish() bullet for the parameters this verification level offers."""
    parameters = finish_parameters(level)
    signature = ", ".join(p.name for p in parameters)
    sentences = " ".join(p.description for p in parameters)
    return f"- finish({signature}): call this when you are done. {sentences}"


def build_system_prompt(level: str, oracle_enabled: bool = False) -> str:
    """System prompt; depends only on the verification level and the oracle toggle.

    The harness level must not reach this function, or prompt wording would confound it.
    """
    tools = active_tools(oracle_enabled)
    tool_bullets = "\n".join([tool.prompt for tool in tools] + [_finish_bullet(level)])
    tool_count = _COUNT_WORDS[len(tools) + 1]  # +1: finish() isn't in TOOLS

    # 'reported' acts only through _finish_bullet's extra parameters.
    verify_intro = "" if level == "none" else " Verify your own work rather than assuming it's correct."
    verify_closing = "" if level == "none" else ", and verify your output matches what the task described"
    acceptance = _ACCEPTANCE_CLAUSE if level == "binding" else ""

    return f"""You are an autonomous research agent. You solve tasks end-to-end, \
including ones that call for domain expertise you might not already have. Investigate before you \
implement, and ground your methodology in whatever authoritative \
sources you can find (code, papers, the web) rather than guessing.{verify_intro}

You're given a task description and any data it provides, and one or more \
pretrained models that may or may not be relevant to solving it; figuring out whether and how to \
use them is part of the task. You have a private workspace to do your own work in. Your sandbox has \
GPU access -- check `torch.cuda.is_available()` (or run `nvidia-smi`) to see what's available, and \
use it for any GPU-capable workload (e.g. model inference) rather than defaulting to CPU.

Your context window is limited. Data files can be large (e.g. thousands of CSV rows) -- do NOT read \
entire large data files with read_file, since its output is truncated and wastes context. Instead \
use run_bash with tools like `head`, `wc -l`, or a short pandas/python snippet to inspect a sample \
(e.g. the first few rows, column dtypes, shape) before writing code that processes the whole file.

You have {tool_count} tools:
{tool_bullets}

Work step by step: understand what the task is asking for, gather whatever data or information you \
need, do the work{verify_closing} before calling finish. You must call finish exactly once when your final output \
is ready. Do not stop before calling finish.{acceptance}
"""


def build_user_message(task_description: str, max_duration_s: int) -> str:
    """User message stating the wall-clock budget; the iteration cap is deliberately not mentioned."""
    return (
        "Task description (also available at /task/description.md):\n\n"
        f"{task_description}\n\n"
        "Any data files for the task are under /task/data. Produce your final output under "
        "/agent_run/workspace and call finish with its path when done.\n\n"
        f"You have {max_duration_s // 60} minutes of wall-clock time to finish this task. A running "
        "status line after each tool result tells you how much you have left. If the run ends "
        "without you calling finish, you get no credit at all -- so if you're running low, submit "
        "your best attempt so far rather than continuing to investigate and risking no submission."
    )
