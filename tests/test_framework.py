"""Unit tests for the experiment's pure functions. No GPU, container or backbone needed."""

import json
import re
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

import agent.config
import orchestrate
from agent.agent import _assistant_message_dict, run
from agent.config import (
    HARNESS_LEVELS,
    VERIFICATION_LEVELS,
    Config,
    generates_reasoning,
    is_verified,
    persists_reasoning,
)
from agent.prompts import build_system_prompt
from agent.tools import TOOLS, SandboxViolation, Tools, active_tools, build_tool_schemas
from axes import (
    ANTHROPIC_CACHE_READ_MULTIPLIER,
    AXIS_DEFAULTS,
    AXIS_LEVELS,
    AXIS_NAMES,
    DEFAULT_SAMPLING,
    MODEL_GPU_COUNT,
    MODEL_LEVELS,
    RunSpec,
    anthropic_actual_cost_usd,
    estimated_cost_usd,
    provider_for,
)
from eval.evaluate import evaluate
from eval.oracle_server import OracleServer
from scripts.production.run_grid import grid_oracle
from scripts.analysis.trace_reconstruct import reconstruct_input
from scripts.analysis.trace_render import _manipulation_checks, parse_verification_assessment

# --- axes: the information ladder and the defaults registry -------------------------------------


@pytest.mark.parametrize(
    "information,expected",
    [
        ("none", ()),
        ("identity", ("identity",)),
        ("interface", ("identity", "interface")),
        ("protocol", ("identity", "interface", "protocol")),
    ],
)
def test_information_ladder_is_cumulative(information, expected):
    assert RunSpec(task="t", information=information).info_snippets == expected


def test_runspec_rejects_unknown_level():
    with pytest.raises(ValueError, match="harness"):
        RunSpec(task="t", harness="chain-of-thought")


def test_run_name_covers_every_axis():
    spec = RunSpec(task="t", replicate=3)
    assert spec.run_name == "t__" + "__".join(getattr(spec, a) for a in AXIS_NAMES) + "__r3"


def test_grid_oracle_defaults_off_and_is_not_a_runspec_axis():
    """oracle is a spec-level toggle (run_grid_slurm.py's AGENT_ORACLE), not a RunSpec field --
    it must stay orthogonal to the verification axis, see agent/tools.py's active_tools."""
    assert grid_oracle({"task": "t", "axes": {}}) is False
    assert grid_oracle({"task": "t", "axes": {}, "oracle": True}) is True
    assert "oracle" not in AXIS_NAMES


def test_config_level_tables_mark_the_real_defaults():
    """agent/config.py can't import axes.py, so its (*) markers are prose -- keep them true."""
    marked = re.findall(r"^#\s+(\S+) \(\*\) +--", Path(agent.config.__file__).read_text(), re.MULTILINE)
    assert set(marked) == {AXIS_DEFAULTS["harness"], AXIS_DEFAULTS["verification"]}


def test_every_axes_import_in_the_repo_still_resolves():
    """solutions/*/dev/ scripts import from axes.py but need a model env to run, so nothing else
    here executes them -- a renamed export breaks them silently. Parsed, not imported."""
    import ast

    import axes

    # Explicit source roots: rglob from the repo root would walk runs/ agent workspaces.
    repo_root = Path(orchestrate.__file__).parent
    sources = list(repo_root.glob("*.py"))
    for directory in ("agent", "eval", "scripts", "tests", "solutions"):
        sources += (repo_root / directory).rglob("*.py")
    for path in sources:
        if ".venv" in path.parts or "__pycache__" in path.parts:
            continue
        imported = [
            alias.name
            for node in ast.walk(ast.parse(path.read_text()))
            if isinstance(node, ast.ImportFrom) and node.module == "axes"
            for alias in node.names
        ]
        for name in imported:
            assert hasattr(axes, name), f"{path.relative_to(repo_root)} imports axes.{name}, which no longer exists"


def test_axis_defaults_match_runspec_and_are_legal():
    spec = RunSpec(task="t")
    for axis, level in AXIS_DEFAULTS.items():
        assert level in AXIS_LEVELS[axis], f"{axis} default is not a legal level"
        assert getattr(spec, axis) == level, f"{axis} CLI default and RunSpec default disagree"


# --- the finish() contract: one definition behind both the prompt and the schema ----------------


@pytest.mark.parametrize("level", VERIFICATION_LEVELS)
def test_prompt_and_schema_agree_on_finish_parameters(level):
    schema = build_tool_schemas(level)[-1]["function"]
    names = list(schema["parameters"]["properties"])
    # The prompt's bullet names exactly the parameters the schema offers, in the same order.
    assert f"- finish({', '.join(names)}):" in build_system_prompt(level)
    for name in names:
        assert schema["parameters"]["properties"][name]["description"] in build_system_prompt(level)


@pytest.mark.parametrize("level", VERIFICATION_LEVELS)
def test_verified_levels_require_the_verification_parameters(level):
    required = build_tool_schemas(level)[-1]["function"]["parameters"]["required"]
    verified_only = {"expected_score", "verification_evidence"}
    assert (verified_only <= set(required)) is is_verified(level)
    assert "submission_path" in required and "summary" in required


def test_unverified_levels_drop_volunteered_verification_arguments(tmp_path):
    (tmp_path / "workspace").mkdir()
    submission = tmp_path / "workspace" / "out.csv"
    submission.write_text("id,y\n")
    volunteered = {
        "submission_path": str(submission),
        "summary": "done",
        "expected_score": 0.9,
        "verification_evidence": "checked",
    }

    _, signal = Tools(tmp_path, tmp_path / "workspace", 5, "asked").dispatch("finish", volunteered)
    assert signal.expected_score is None and signal.verification_evidence is None

    _, signal = Tools(tmp_path, tmp_path / "workspace", 5, "reported").dispatch("finish", volunteered)
    assert signal.expected_score == 0.9 and signal.verification_evidence == "checked"


def test_every_advertised_tool_is_one_the_sandbox_can_run(tmp_path):
    """A ToolSpec the model is offered but dispatch has no branch for fails only at call time."""
    tools = Tools(tmp_path, tmp_path / "workspace", 5, "asked")
    for spec in TOOLS:
        assert callable(getattr(tools, spec.name, None)), f"{spec.name} is advertised but not implemented"
    with pytest.raises(ValueError, match="Unknown tool"):
        tools.dispatch("no_such_tool", {})


def test_finish_rejects_a_path_that_is_not_a_file(tmp_path):
    tools = Tools(tmp_path, tmp_path / "workspace", 5, "asked")
    with pytest.raises(FileNotFoundError):
        tools.finish(str(tmp_path / "workspace" / "missing.csv"), "done")


# --- the sandbox's path containment -------------------------------------------------------------


def test_reads_are_confined_to_the_mounted_roots(tmp_path):
    (tmp_path / "task").mkdir()
    (tmp_path / "task" / "description.md").write_text("task")
    (tmp_path / "secret.txt").write_text("host-side")
    tools = Tools(tmp_path / "task", tmp_path / "workspace", 5, "asked")

    assert tools.read_file(str(tmp_path / "task" / "description.md")) == "task"
    with pytest.raises(SandboxViolation):
        tools.read_file(str(tmp_path / "secret.txt"))
    with pytest.raises(SandboxViolation):
        tools.read_file(str(tmp_path / "task" / ".." / "secret.txt"))


def test_writes_are_confined_to_the_workspace_including_via_symlink(tmp_path):
    workspace = tmp_path / "workspace"
    tools = Tools(tmp_path / "task", workspace, 5, "asked")
    (workspace / "escape").symlink_to(tmp_path)

    tools.write_file("inside.txt", "ok")
    assert (workspace / "inside.txt").read_text() == "ok"
    with pytest.raises(SandboxViolation):
        tools.write_file(str(tmp_path / "outside.txt"), "no")
    with pytest.raises(SandboxViolation):
        tools.write_file(str(workspace / "escape" / "outside.txt"), "no")


# --- the harness axis: what each arm carries into the next turn ----------------------------------


def _message(content="answer", reasoning="because"):
    return SimpleNamespace(content=content, reasoning=reasoning, tool_calls=None)


def test_react_persists_reasoning_and_think_act_discards_it():
    persisted = _assistant_message_dict(_message(), persist_reasoning=True, thought_number=4)
    discarded = _assistant_message_dict(_message(), persist_reasoning=False, thought_number=4)
    # "Thought N:" is ReAct's delimiter.
    assert persisted["content"] == "Thought 4: because\n\nanswer"
    assert discarded["content"] == "answer"


def test_reasoning_only_response_still_carries_the_thought():
    message = _assistant_message_dict(_message(content=None), persist_reasoning=True, thought_number=1)
    assert message["content"] == "Thought 1: because"


def test_harness_flags_agree_with_the_level_table():
    """The two flags the loop derives from a harness level; agent.py reads them from config.py."""
    assert [generates_reasoning(level) for level in HARNESS_LEVELS] == [False, True, True]
    assert [persists_reasoning(level) for level in HARNESS_LEVELS] == [False, False, True]
    for helper in (generates_reasoning, persists_reasoning):
        with pytest.raises(ValueError, match="harness"):
            helper("chain-of-thought")


# --- sampling: a model level is weights plus the preset their own authors publish ----------------


def test_every_vllm_model_level_carries_both_vendor_presets():
    """Only vLLM tiers have a vendor sampling table to compare against DEFAULT_SAMPLING -- Claude
    Sonnet 5 has no such table at all (see the dedicated claude-sonnet-5 tests below)."""
    for name, (model_id, thinking, non_thinking) in MODEL_LEVELS.items():
        assert model_id, f"{name} has no served model id"
        if provider_for(name) != "vllm":
            continue
        for preset in (thinking, non_thinking):
            assert set(preset) == set(DEFAULT_SAMPLING), f"{name}'s presets must set the same knobs"


def test_sampling_follows_both_the_model_and_the_harness_level():
    """act-only runs the template's non-thinking mode, so it takes that mode's own preset -- the
    arm is the vendor's non-thinking configuration, not thinking-mode settings with thinking off."""
    vllm_models = {name: v for name, v in MODEL_LEVELS.items() if provider_for(name) == "vllm"}
    for model, (_, _, non_thinking) in vllm_models.items():
        thinking = {RunSpec(task="t", model=model, harness=h).sampling["temperature"] for h in ("think-act", "react")}
        assert len(thinking) == 1, "the two thinking arms must sample identically"
        assert RunSpec(task="t", model=model, harness="act-only").sampling == non_thinking
    # The reason the preset can't be one shared table: this family's presets are not uniform.
    presets = {name: thinking["temperature"] for name, (_, thinking, _) in vllm_models.items()}
    assert len(set(presets.values())) > 1, f"if every level agreed, one shared preset would do: {presets}"


def test_claude_model_level_has_no_gpu_and_uses_effort_not_sampling_params():
    """claude-sonnet-5 leases no server GPU and sets effort instead of sampling parameters."""
    assert provider_for("claude-sonnet-5") == "anthropic"
    assert MODEL_GPU_COUNT["claude-sonnet-5"] == 0
    for harness in HARNESS_LEVELS:
        assert RunSpec(task="t", model="claude-sonnet-5", harness=harness).sampling == {"effort": "high"}


def test_claude_pricing_key_resolves():
    assert estimated_cost_usd("claude-sonnet-5", 1_000_000, 1_000_000) == pytest.approx(2.00 + 10.00)


def test_anthropic_actual_cost_prices_cache_reads_and_writes_separately():
    # Cache reads are cheap, so the billed cost is below estimated_cost_usd's upper bound.
    cheap = anthropic_actual_cost_usd(
        input_tokens=0, cache_creation_input_tokens=0, cache_read_input_tokens=1_000_000, output_tokens=0,
    )
    assert cheap == pytest.approx(2.00 * ANTHROPIC_CACHE_READ_MULTIPLIER)
    assert cheap < estimated_cost_usd("claude-sonnet-5", 1_000_000, 0)


# --- eval: the regime/gap arithmetic and the anchors it is derived from --------------------------


def _task_config(tmp_path, **overrides):
    """A minimal eval_config for a 2-row continuous task, plus the solution file it names."""
    solution = tmp_path / "solution.csv"
    solution.write_text("id,y\n1,1.0\n2,3.0\n")
    fields = dict(
        SOLUTION_FILE=solution, ID_COLUMN="id", TARGET_COLUMN="y", METRIC="r2",
        TARGET_TYPE="continuous", HIGHER_IS_BETTER=True, EXPECTED_ROWS=2,
        REFERENCE=0.8, BACKBONE=0.2, TRIVIAL=0.0, MARGIN=0.1,
    )
    return SimpleNamespace(**{**fields, **overrides})


def _submission(tmp_path, values):
    path = tmp_path / "submission.csv"
    path.write_text("id,y\n" + "".join(f"{i},{v}\n" for i, v in zip((1, 2), values)))
    return path


@pytest.mark.parametrize(
    "reference,backbone,regime",
    [(0.8, 0.2, "gap_positive"), (0.2, 0.8, "gap_negative"), (0.5, 0.55, "gap_negligible")],
)
def test_regime_follows_the_anchors(tmp_path, monkeypatch, reference, backbone, regime):
    cfg = _task_config(tmp_path, REFERENCE=reference, BACKBONE=backbone)
    monkeypatch.setattr("eval.evaluate.load_task_config", lambda task: cfg)
    result = evaluate("fake", _submission(tmp_path, [1.0, 3.0]))
    assert result["regime"] == regime
    # gap_closed is only meaningful where there is a gap to close.
    assert (result["gap_closed"] is not None) is (regime == "gap_positive")


def test_anchors_are_recorded_so_the_result_stays_interpretable(tmp_path, monkeypatch):
    cfg = _task_config(tmp_path)
    monkeypatch.setattr("eval.evaluate.load_task_config", lambda task: cfg)
    result = evaluate("fake", _submission(tmp_path, [1.0, 3.0]))
    assert (result["reference"], result["backbone"]) == (cfg.REFERENCE, cfg.BACKBONE)
    assert (result["trivial"], result["margin"]) == (cfg.TRIVIAL, cfg.MARGIN)
    assert result["higher_is_better"] is True


def test_gap_closed_reaches_one_at_the_reference(tmp_path, monkeypatch):
    # A perfect submission scores r2=1.0; with REFERENCE=1.0 that is exactly full recovery.
    cfg = _task_config(tmp_path, REFERENCE=1.0, BACKBONE=0.0)
    monkeypatch.setattr("eval.evaluate.load_task_config", lambda task: cfg)
    result = evaluate("fake", _submission(tmp_path, [1.0, 3.0]))
    assert result["score"] == pytest.approx(1.0)
    assert result["gap_closed"] == pytest.approx(1.0)
    assert result["beat_trivial"] is True and result["below_both"] is False


def test_lower_is_better_flips_every_comparison(tmp_path, monkeypatch):
    # Same numbers, opposite direction: the specialist's lower REFERENCE is now the better anchor.
    cfg = _task_config(tmp_path, HIGHER_IS_BETTER=False, REFERENCE=0.2, BACKBONE=0.8, TRIVIAL=1.0)
    monkeypatch.setattr("eval.evaluate.load_task_config", lambda task: cfg)
    result = evaluate("fake", _submission(tmp_path, [1.0, 3.0]))
    assert result["regime"] == "gap_positive"
    assert result["calibration_error"] is None


def test_calibration_error_is_positive_when_the_agent_overestimates(tmp_path, monkeypatch):
    cfg = _task_config(tmp_path)
    monkeypatch.setattr("eval.evaluate.load_task_config", lambda task: cfg)
    result = evaluate("fake", _submission(tmp_path, [1.0, 2.0]), expected_score=1.0)
    assert result["score"] < 1.0
    assert result["calibration_error"] > 0


@pytest.mark.parametrize(
    "reference,backbone,model_used,expected_correct",
    [
        (0.8, 0.2, "expert", True),    # gap_positive: using the specialist is right
        (0.8, 0.2, None, False),
        (0.2, 0.8, None, True),        # gap_negative: declining it is right
        (0.2, 0.8, "expert", False),
        (0.5, 0.55, "expert", None),   # gap_negligible: not scoreable either way
    ],
)
def test_model_choice_is_scored_against_the_anchors_not_the_domain_model(
    tmp_path, monkeypatch, reference, backbone, model_used, expected_correct
):
    cfg = _task_config(tmp_path, REFERENCE=reference, BACKBONE=backbone, DOMAIN_MODEL="expert")
    monkeypatch.setattr("eval.evaluate.load_task_config", lambda task: cfg)
    result = evaluate("fake", _submission(tmp_path, [1.0, 3.0]), model_used=model_used)
    assert result["model_choice_correct"] is expected_correct


def test_format_errors_are_reported_without_a_score(tmp_path, monkeypatch):
    monkeypatch.setattr("eval.evaluate.load_task_config", lambda task: _task_config(tmp_path))
    path = tmp_path / "bad.csv"
    path.write_text("id,y\n1,1.0\n1,2.0\n3,3.0\n")
    result = evaluate("fake", path)
    assert result["valid"] is False
    assert any("duplicate" in e for e in result["errors"])
    assert any("rows" in e for e in result["errors"])


# --- the oracle tool -------------------------------------------------------------------------------


def _running_oracle_server(tmp_path, monkeypatch, task="fake", **cfg_overrides):
    """A live OracleServer for one test, scoring against _task_config's 2-row solution."""
    cfg = _task_config(tmp_path, **cfg_overrides)
    monkeypatch.setattr("eval.evaluate.load_task_config", lambda t: cfg)
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    server = OracleServer(task, workspace)
    server.start()
    return server, workspace


def test_active_tools_only_adds_oracle_when_enabled():
    assert [t.name for t in active_tools(False)] == [t.name for t in TOOLS]
    assert [t.name for t in active_tools(True)] == [t.name for t in TOOLS] + ["oracle_check"]


@pytest.mark.parametrize("level", VERIFICATION_LEVELS)
def test_oracle_tool_is_orthogonal_to_the_verification_level(level):
    """oracle_check depends only on oracle_enabled, never on the verification level."""
    names = lambda enabled: {s["function"]["name"] for s in build_tool_schemas(level, oracle_enabled=enabled)}
    assert "oracle_check" not in names(False)
    assert "oracle_check" in names(True)
    assert "oracle_check" not in build_system_prompt(level, oracle_enabled=False)
    assert "oracle_check" in build_system_prompt(level, oracle_enabled=True)


def test_oracle_check_is_unavailable_without_an_oracle_url(tmp_path):
    tools = Tools(tmp_path, tmp_path / "workspace", 5, "asked")
    with pytest.raises(RuntimeError, match="not available"):
        tools.oracle_check("submission.csv")


def test_oracle_server_returns_only_the_score_never_the_anchors(tmp_path, monkeypatch):
    server, workspace = _running_oracle_server(tmp_path, monkeypatch)
    try:
        (workspace / "submission.csv").write_text("id,y\n1,1.0\n2,3.0\n")
        response = httpx.post(f"{server.url}/check", json={"relative_path": "submission.csv"})
        result = response.json()
    finally:
        server.stop()
    assert set(result) == {"valid", "score", "metric", "n", "errors"}
    assert result["score"] == pytest.approx(1.0)
    # Anchors (here 0.8/0.2/0.0/0.1) must never reach the oracle payload.
    assert "reference" not in result and "backbone" not in result and "regime" not in result


def test_oracle_check_dispatches_through_tools_and_reports_the_score(tmp_path, monkeypatch):
    server, workspace = _running_oracle_server(tmp_path, monkeypatch)
    try:
        (workspace / "submission.csv").write_text("id,y\n1,1.0\n2,3.0\n")
        tools = Tools(tmp_path / "task", workspace, 5, "asked", oracle_url=server.url)
        result_text, finish_signal = tools.dispatch("oracle_check", {"submission_path": "submission.csv"})
    finally:
        server.stop()
    assert finish_signal is None  # oracle_check must never end the run the way finish() does
    assert result_text == "r2: 1.0000 (2 rows)"


def test_oracle_check_reports_format_errors_without_a_score(tmp_path, monkeypatch):
    server, workspace = _running_oracle_server(tmp_path, monkeypatch)
    try:
        (workspace / "bad.csv").write_text("wrong,columns\n1,2\n")
        tools = Tools(tmp_path / "task", workspace, 5, "asked", oracle_url=server.url)
        result_text = tools.oracle_check("bad.csv")
    finally:
        server.stop()
    assert "INVALID" in result_text


def test_oracle_server_rejects_a_relative_path_that_escapes_the_workspace(tmp_path, monkeypatch):
    server, workspace = _running_oracle_server(tmp_path, monkeypatch)
    try:
        response = httpx.post(f"{server.url}/check", json={"relative_path": "../secret.csv"})
        result = response.json()
    finally:
        server.stop()
    assert result["valid"] is False


# --- manipulation checks: did each axis's intervention actually reach the agent? -----------------


def _llm_event(reasoning=None, content="", persisted=None):
    """One llm_response: raw_output is the model's own reply, message is what went into history."""
    return {
        "event": "llm_response",
        "raw_output": {"reasoning": reasoning, "content": content},
        "message": {"role": "assistant", "content": persisted if persisted is not None else content},
    }


def _checks(harness, events, **run_start):
    return _manipulation_checks(events, {"harness": harness, **run_start}, {})


def test_manipulation_check_counts_reasoning_generated_and_persisted():
    checks = _checks(
        "react",
        [_llm_event(reasoning="why", content="a", persisted="Thought 1: why\n\na")],
    )
    assert (checks["check_reasoning_turns"], checks["check_persisted_turns"]) == (1, 1)
    # Both char counts come off raw_output, so folding the thought in can't double-count it.
    assert (checks["check_reasoning_chars"], checks["check_content_chars"]) == (3, 1)
    assert checks["check_violations"] is None


def test_manipulation_check_catches_a_react_arm_that_did_not_persist():
    """The failure this exists for: <think> stripped from history makes react == think-act at the
    model's input, and the resulting null reads as "persistence doesn't matter"."""
    checks = _checks("react", [_llm_event(reasoning="why", content="a")])
    assert checks["check_reasoning_turns"] == 1 and checks["check_persisted_turns"] == 0
    assert "should persist reasoning into context" in checks["check_violations"]


def test_manipulation_check_catches_an_act_only_arm_that_reasoned():
    checks = _checks("act-only", [_llm_event(reasoning="why", content="a")])
    assert "should not generate reasoning" in checks["check_violations"]


def test_manipulation_check_passes_a_clean_think_act_arm():
    assert _checks("think-act", [_llm_event(reasoning="why", content="a")])["check_violations"] is None


def test_manipulation_check_flags_an_information_level_that_never_reached_the_description():
    checks = _manipulation_checks(
        [_llm_event(content="a")],
        {"harness": "think-act", "information": "identity", "task_name": "redshift-estimation",
         "seed_messages": [{"role": "user", "content": "the bare description"}]},
        {},
    )
    assert checks["check_info_snippets_granted"] == 1 and checks["check_info_snippets_found"] == 0
    assert "missing from the description" in checks["check_violations"]


def test_manipulation_check_catches_a_withheld_rung_leaking_into_the_description():
    """The contamination direction: information=none is the control, so a rung it withholds
    appearing in the description makes that cell measure nothing."""
    protocol = (Path(orchestrate.__file__).parent / "tasks/redshift-estimation/info/protocol.md").read_text()
    checks = _manipulation_checks(
        [_llm_event(reasoning="why", content="a")],
        {"harness": "think-act", "information": "none", "task_name": "redshift-estimation",
         "seed_messages": [{"role": "user", "content": f"the description\n\n{protocol}"}]},
        {},
    )
    assert "withholds the rungs above it" in checks["check_violations"]
    # Counted, not skipped: 'none' grants nothing, which is a checkable claim rather than no claim.
    assert checks["check_info_snippets_granted"] == 0


def test_manipulation_check_does_not_mistake_the_models_own_text_for_persistence():
    """Persistence is the two channels differing, not a prefix -- a model writing "Thought 2:"
    itself at a non-persisting arm must not read as the harness manipulation having run."""
    event = _llm_event(reasoning="why", content="Thought 2: let me try X")
    assert _checks("think-act", [event])["check_violations"] is None


@pytest.mark.parametrize(
    "level,run_end,expected",
    [
        ("reported", {"status": "finished"}, "requires expected_score"),
        ("reported", {"status": "finished", "expected_score": 0.5, "verification_evidence": "checked"}, None),
        ("asked", {"status": "finished", "expected_score": 0.5, "verification_evidence": "checked"}, "should not accept"),
        # An unfinished run never reached finish(), so it can't have reported anything.
        ("reported", {"status": "max_duration_reached"}, None),
    ],
)
def test_manipulation_check_holds_finish_to_the_verification_level(level, run_end, expected):
    # reasoning="why" keeps the think-act arm itself clean, so only the verification check can fire.
    events = [_llm_event(reasoning="why", content="a")]
    violations = _manipulation_checks(events, {"harness": "think-act", "verification": level}, run_end)["check_violations"]
    if expected is None:
        assert violations is None
    else:
        assert expected in violations


def test_manipulation_check_reports_whether_the_budget_bound():
    events, start = [_llm_event(content="a")], {"harness": "think-act"}
    assert _manipulation_checks(events, start, {"status": "max_duration_reached"})["budget_bound"] is True
    assert _manipulation_checks(events, start, {"status": "finished"})["budget_bound"] is False


# --- trace handling ------------------------------------------------------------------------------


def test_submission_path_is_rebased_from_the_container_onto_the_host(tmp_path):
    run_end = {"status": "finished", "submission_path": "/agent_run/workspace/out.csv"}
    assert orchestrate._host_submission_path(run_end, tmp_path) == tmp_path / "workspace" / "out.csv"


@pytest.mark.parametrize(
    "run_end",
    [None, {"status": "crashed"}, {"status": "finished", "submission_path": ""}],
)
def test_no_submission_path_without_a_finished_run(tmp_path, run_end):
    assert orchestrate._host_submission_path(run_end, tmp_path) is None


def test_reconstruct_input_replays_the_conversation_up_to_an_iteration(tmp_path):
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        "\n".join(
            json.dumps(record)
            for record in [
                {"event": "run_start", "seed_messages": [{"role": "system", "content": "s"}]},
                {"event": "llm_response", "iteration": 0, "message": {"role": "assistant", "content": "a0"}},
                {"event": "tool_call", "iteration": 0, "message": {"role": "tool", "content": "r0"}},
                {"event": "llm_response", "iteration": 1, "message": {"role": "assistant", "content": "a1"}},
            ]
        )
    )
    assert [m["content"] for m in reconstruct_input(trace, 0)] == ["s"]
    assert [m["content"] for m in reconstruct_input(trace, 1)] == ["s", "a0", "r0"]
    with pytest.raises(ValueError):
        reconstruct_input(trace, 2)


def test_verification_assessment_parses_into_fields():
    parsed = parse_verification_assessment(
        "VERIFICATION: Superficial\n"
        "VERIFICATION_REASON: only checked the file parsed\n"
        "FAILURE_STAGE: preprocessing\n"
        "**FAILURE_MODE:** `silent`\n"
        "FAILURE_REASON: skipped ToRGB\n"
    )
    assert parsed["verification"] == "superficial"
    assert parsed["failure_mode"] == "silent"
    assert parsed["failure_stage"] == "preprocessing"
    assert parsed["verification_reason"] == "only checked the file parsed"


def test_verification_assessment_tolerates_a_missing_or_off_format_reply():
    assert parse_verification_assessment(None)["verification"] is None
    assert parse_verification_assessment("I could not tell.")["failure_mode"] is None
    # An out-of-enum answer is kept verbatim rather than silently reading as "not assessed".
    assert parse_verification_assessment("VERIFICATION: mostly fine")["verification"] == "mostly fine"


# --- the agent loop, end to end against a stub backbone ------------------------------------------


class _StubCompletions:
    def __init__(self, responses):
        self._responses = list(responses)

    def create(self, **kwargs):
        response = self._responses.pop(0)
        if isinstance(response, BaseException):  # KeyboardInterrupt is not an Exception
            raise response
        usage = SimpleNamespace(model_dump=lambda: {"prompt_tokens": 100, "completion_tokens": 10})
        return SimpleNamespace(choices=[SimpleNamespace(message=response)], usage=usage)


def _stub_openai(responses):
    def factory(**kwargs):
        return SimpleNamespace(chat=SimpleNamespace(completions=_StubCompletions(responses)))

    return factory


def _assistant(tool_calls=None, content="", reasoning=None):
    message = SimpleNamespace(content=content, reasoning=reasoning, tool_calls=tool_calls)
    message.model_dump = lambda: {"content": content, "reasoning": reasoning}
    return message


def _call(name, **arguments):
    return SimpleNamespace(
        id=f"call-{name}", function=SimpleNamespace(name=name, arguments=json.dumps(arguments))
    )


def _config(tmp_path, **overrides):
    task_dir = tmp_path / "task"
    task_dir.mkdir(exist_ok=True)
    (task_dir / "description.md").write_text("Predict something.")
    fields = dict(
        vllm_base_url="http://stub", model_name="stub-model", max_iterations=10, max_duration_s=300,
        task_dir=task_dir, task_name="fake", models_dir=tmp_path / "models", run_dir=tmp_path,
        workspace_dir=tmp_path / "workspace", trace_path=tmp_path / "trace.jsonl", bash_timeout_s=5,
        verbose=False, web_allowlist=frozenset(), web_denylist=frozenset(), oracle_url=None, gpu_uuid="none",
        temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0,
        information="none", harness="react", verification="asked", budget="medium", model="m", replicate=1,
    )
    return Config(**{**fields, **overrides})


def _events(config):
    return [json.loads(line) for line in Path(config.trace_path).read_text().splitlines() if line]


def test_loop_runs_to_finish_and_records_every_turn(tmp_path, monkeypatch):
    config = _config(tmp_path)
    submission = tmp_path / "workspace" / "out.csv"
    monkeypatch.setattr(
        "agent.agent.OpenAI",
        _stub_openai([
            _assistant(tool_calls=[_call("write_file", path=str(submission), content="id,y\n")]),
            _assistant(tool_calls=[_call("finish", submission_path=str(submission), summary="done")]),
        ]),
    )

    assert run(config) == 0
    events = _events(config)
    run_end = events[-1]
    assert run_end["event"] == "run_end" and run_end["status"] == "finished"
    assert run_end["submission_path"] == str(submission)
    # The budget status line, the one message built outside the LLM/tool paths.
    assert [e["event"] for e in events].count("status_message") == 1
    assert "elapsed" in next(e for e in events if e["event"] == "status_message")["message"]["content"]


def test_run_end_records_what_the_run_cost(tmp_path, monkeypatch):
    """Totals come from run_end, so every reader of a run agrees on them without re-deriving."""
    config = _config(tmp_path)
    submission = tmp_path / "workspace" / "out.csv"
    monkeypatch.setattr(
        "agent.agent.OpenAI",
        _stub_openai([
            _assistant(tool_calls=[_call("write_file", path=str(submission), content="id,y\n")]),
            _assistant(tool_calls=[_call("finish", submission_path=str(submission), summary="done")]),
        ]),
    )

    run(config)
    run_end = _events(config)[-1]
    assert run_end["iterations"] == 2
    assert run_end["prompt_tokens"] == 200 and run_end["completion_tokens"] == 20
    assert run_end["wallclock_s"] >= 0


def test_a_failing_tool_is_reported_to_the_model_not_fatal(tmp_path, monkeypatch):
    config = _config(tmp_path)
    submission = tmp_path / "workspace" / "out.csv"
    monkeypatch.setattr(
        "agent.agent.OpenAI",
        _stub_openai([
            _assistant(tool_calls=[_call("read_file", path="/etc/passwd")]),
            _assistant(tool_calls=[_call("write_file", path=str(submission), content="id,y\n")]),
            _assistant(tool_calls=[_call("finish", submission_path=str(submission), summary="done")]),
        ]),
    )

    assert run(config) == 0
    tool_results = [e["message"]["content"] for e in _events(config) if e["event"] == "tool_call"]
    assert tool_results[0].startswith("error:") and "outside the allowed read roots" in tool_results[0]


def test_an_unexpected_crash_still_records_a_run_end(tmp_path, monkeypatch):
    config = _config(tmp_path)
    monkeypatch.setattr("agent.agent.build_system_prompt", lambda level, oracle_enabled=False: 1 / 0)

    assert run(config) == 1
    run_end = _events(config)[-1]
    assert run_end["status"] == "crashed"
    assert "ZeroDivisionError" in run_end["error"] and "traceback" in run_end
    assert run_end["iterations"] == 0 and "wallclock_s" in run_end  # recorded even when nothing ran


def test_an_interrupt_still_records_a_run_end(tmp_path, monkeypatch):
    config = _config(tmp_path)
    monkeypatch.setattr("agent.agent.OpenAI", _stub_openai([KeyboardInterrupt()]))

    assert run(config) == 1
    assert _events(config)[-1]["status"] == "interrupted"


def test_the_budget_ends_the_run_before_the_iteration_cap(tmp_path, monkeypatch):
    config = _config(tmp_path, max_duration_s=0)
    monkeypatch.setattr("agent.agent.OpenAI", _stub_openai([]))

    assert run(config) == 1
    assert _events(config)[-1]["status"] == "max_duration_reached"


def test_an_llm_that_never_calls_a_tool_stalls_out(tmp_path, monkeypatch):
    config = _config(tmp_path)
    monkeypatch.setattr("agent.agent.OpenAI", _stub_openai([_assistant(content="thinking...")] * 5))

    assert run(config) == 1
    events = _events(config)
    assert events[-1]["status"] == "stalled_no_tool_calls"
    assert [e["event"] for e in events].count("nudge_message") == 4
