"""Validate tasks/<t>, solutions/<t> and the models they reference against README.md's task contract.

Usage: uv run python -m scripts.production.check_task <task-name>
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from axes import INFORMATION_LEVELS
from eval.evaluate import evaluate, load_task_config
from eval.metrics import METRICS

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
# In ladder order -- the cumulative-prefix check below depends on it.
INFO_SNIPPETS = INFORMATION_LEVELS[1:]
REQUIRED_EVAL_CONFIG_FIELDS = (
    "SOLUTION_FILE", "ID_COLUMN", "TARGET_COLUMN", "METRIC", "TARGET_TYPE",
    "HIGHER_IS_BETTER", "EXPECTED_ROWS", "REFERENCE", "BACKBONE", "TRIVIAL", "MARGIN",
)


class Check:
    """Collects problems instead of raising, so one run reports everything wrong at once."""

    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def error(self, msg: str) -> None:
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)


def check_task_dir(task: str, c: Check) -> list[str]:
    """Structural checks under tasks/<task>/; returns the model names listed in models.txt."""
    task_dir = REPO_ROOT / "tasks" / task
    description = task_dir / "description.md"
    if not description.is_file() or not description.read_text().strip():
        c.error(f"{description} missing or empty")

    models_txt = task_dir / "models.txt"
    models: list[str] = []
    if not models_txt.is_file():
        c.error(f"{models_txt} missing")
    else:
        models = [line.strip() for line in models_txt.read_text().splitlines() if line.strip()]
        if not models:
            c.error(f"{models_txt} lists no models")

    data_dir = task_dir / "data"
    if not data_dir.is_dir() or not any(data_dir.iterdir()):
        c.warn(f"{data_dir} missing or empty -- run solutions/{task}/download.sh")

    check_info_ladder(task_dir, models, c)
    return models


def check_info_ladder(task_dir: Path, models: list[str], c: Check) -> None:
    """Structural checks of the information ladder (README, "The information ladder")."""
    info_dir = task_dir / "info"
    present = {p.stem for p in info_dir.glob("*.md")} if info_dir.is_dir() else set()
    if not present:
        c.warn(f"{info_dir} has none of {list(INFO_SNIPPETS)} -- information levels above 'none' will error at launch")
        return
    if unknown := present - set(INFO_SNIPPETS):
        c.warn(f"{info_dir} has unrecognized file(s) {sorted(unknown)}, ignored by the information ladder")

    # The ladder is cumulative, so the snippets present must be a prefix of it: a task with
    # protocol.md but no interface.md can be run at 'protocol' but not at 'interface'.
    for higher, lower in zip(INFO_SNIPPETS[1:], INFO_SNIPPETS):
        if higher in present and lower not in present:
            c.error(f"{info_dir}/{higher}.md exists but {lower}.md doesn't -- the ladder is cumulative, so rungs present must be a prefix of {list(INFO_SNIPPETS)}")

    if "identity" in present:
        identity = (info_dir / "identity.md").read_text()
        # 'which artifact?' and nothing else: a name, no operational content.
        if not any(name in identity for name in models):
            c.error(f"{info_dir}/identity.md names none of models.txt's {models} -- the identity rung's only job is to name the relevant model")
        if "```" in identity:
            c.error(f"{info_dir}/identity.md has a code block -- how to call the model is the interface rung, not identity")

    if "interface" in present:
        interface = (info_dir / "interface.md").read_text()
        # 'how do I invoke it?': facts from the model's own docs, valid for any task using it.
        if not any(name in interface for name in models):
            c.warn(f"{info_dir}/interface.md names none of models.txt's {models} -- the interface rung should say what it is describing")
        # Naming this task's data is the commonest way task-specific content slips a rung down;
        # a fact that only holds for this task belongs in protocol.md.
        if "data/" in interface:
            c.warn(f"{info_dir}/interface.md refers to this task's data/ -- the interface rung should hold only facts valid for any task using the model; move task-specific content to protocol.md")

    if "protocol" in present and "```" not in (info_dir / "protocol.md").read_text():
        c.warn(f"{info_dir}/protocol.md has no code block -- the protocol rung is this task's reference recipe, normally code")


def check_eval_config(task: str, c: Check):
    try:
        cfg = load_task_config(task)
    except Exception as exc:
        c.error(f"solutions/{task}/eval_config.py failed to load: {exc}")
        return None

    for field in REQUIRED_EVAL_CONFIG_FIELDS:
        if not hasattr(cfg, field):
            c.error(f"eval_config.py missing required field {field}")
    if hasattr(cfg, "METRIC") and cfg.METRIC not in METRICS:
        c.error(f"METRIC {cfg.METRIC!r} not in eval/metrics.py's registry {sorted(METRICS)}")
    if hasattr(cfg, "TARGET_TYPE") and cfg.TARGET_TYPE not in ("continuous", "categorical"):
        c.error(f"TARGET_TYPE must be 'continuous' or 'categorical', got {cfg.TARGET_TYPE!r}")
    if hasattr(cfg, "MARGIN") and cfg.MARGIN <= 0:
        c.error(f"MARGIN must be > 0, got {cfg.MARGIN}")
    if hasattr(cfg, "SOLUTION_FILE") and not cfg.SOLUTION_FILE.is_file():
        c.error(f"SOLUTION_FILE {cfg.SOLUTION_FILE} does not exist")
    return cfg


def check_solution(task: str, cfg, c: Check) -> None:
    solution = pd.read_csv(cfg.SOLUTION_FILE)
    for col in (cfg.ID_COLUMN, cfg.TARGET_COLUMN):
        if col not in solution.columns:
            c.error(f"solution.csv missing column {col!r}")
            return
    if solution[cfg.ID_COLUMN].duplicated().any():
        c.error(f"solution.csv's {cfg.ID_COLUMN} has duplicate values")
    if len(solution) != cfg.EXPECTED_ROWS:
        c.error(f"solution.csv has {len(solution)} rows, EXPECTED_ROWS says {cfg.EXPECTED_ROWS}")
    if cfg.TARGET_TYPE == "categorical":
        valid = getattr(cfg, "VALID_TARGET_VALUES", None)
        if valid is not None and not set(solution[cfg.TARGET_COLUMN].unique()) <= valid:
            c.error(f"solution.csv's {cfg.TARGET_COLUMN} has values outside VALID_TARGET_VALUES {valid}")

    # Grade the solution against itself; checks the evaluate() wiring, not the anchors.
    self_graded = evaluate(task, cfg.SOLUTION_FILE)
    if not self_graded["valid"]:
        c.error(f"grading solution.csv against itself failed: {self_graded['errors']}")


def check_dev_scripts(task: str, c: Check) -> None:
    dev_dir = REPO_ROOT / "solutions" / task / "dev"
    for name in ("run_reference.py", "run_backbone.py"):
        if not (dev_dir / name).is_file():
            c.error(f"solutions/{task}/dev/{name} missing")
    if not (REPO_ROOT / "solutions" / task / "download.sh").is_file():
        c.error(f"solutions/{task}/download.sh missing")


def check_models(models: list[str], c: Check) -> None:
    for name in dict.fromkeys(models):  # dedup, preserve order
        model_dir = REPO_ROOT / "models" / name
        if not model_dir.is_dir():
            c.error(f"models/{name} does not exist (listed in models.txt)")
            continue
        agent_dir = model_dir / "agent"
        if not (agent_dir / "README.md").is_file():
            c.error(f"models/{name}/agent/README.md missing")
        weights_dir = agent_dir / "weights"
        if not weights_dir.is_dir() or not any(weights_dir.iterdir()):
            c.warn(f"models/{name}/agent/weights/ missing or empty -- run models/{name}/download.sh")
        if not (agent_dir / "env").is_dir():
            c.warn(f"models/{name}/agent/env/ not built -- run scripts/production/build_agent_model_env.sh {name} (orchestrate.py also auto-builds it)")
        if not (model_dir / "download.sh").is_file():
            c.error(f"models/{name}/download.sh missing")
        if not (model_dir / "verify.py").is_file():
            c.error(f"models/{name}/verify.py missing")
        if not (model_dir / "dev_env" / "pyproject.toml").is_file():
            c.error(f"models/{name}/dev_env/pyproject.toml missing")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task")
    args = parser.parse_args()

    c = Check()
    models = check_task_dir(args.task, c)
    cfg = check_eval_config(args.task, c)
    if cfg is not None:
        check_solution(args.task, cfg, c)
        domain_model = getattr(cfg, "DOMAIN_MODEL", None)
        if domain_model is not None and domain_model not in models:
            c.error(f"eval_config.py's DOMAIN_MODEL {domain_model!r} is not listed in models.txt {models}")
    check_dev_scripts(args.task, c)
    check_models(models, c)

    for warning in c.warnings:
        print(f"warn:  {warning}")
    for error in c.errors:
        print(f"error: {error}")
    print(f"\n{args.task}: {len(c.errors)} error(s), {len(c.warnings)} warning(s)")
    return 1 if c.errors else 0


if __name__ == "__main__":
    sys.exit(main())
