"""Tests for agent/backends/claude.py against real anthropic.types objects and a stubbed client."""

import json
from pathlib import Path

import anthropic
import httpx2
import pytest
from anthropic.types import Message, TextBlock, ThinkingBlock, ToolUseBlock, Usage

import orchestrate
from agent.agent import run
from agent.backends.claude import Conversation, _create_with_retry, _is_transient, _to_claude_tools
from agent.config import Config
from agent.tools import build_tool_schemas
from axes import RunSpec
from scripts.analysis.trace_render import _llm_section

_READ_FILE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read a file",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
    },
}


def _usage(input_tokens=100, output_tokens=10, cache_creation=0, cache_read=0):
    return Usage(
        input_tokens=input_tokens, output_tokens=output_tokens,
        cache_creation_input_tokens=cache_creation, cache_read_input_tokens=cache_read,
    )


def _message(content, stop_reason="tool_use", id="msg_1"):
    return Message(
        id=id, type="message", role="assistant", model="claude-sonnet-5",
        content=content, stop_reason=stop_reason, stop_sequence=None, usage=_usage(),
    )


def _tool_use(name, id, **input_):
    return ToolUseBlock(type="tool_use", id=id, name=name, input=input_)


def _thinking(text="because", signature="sig"):
    return ThinkingBlock(type="thinking", thinking=text, signature=signature)


def _text(text):
    return TextBlock(type="text", text=text, citations=None)


class _NullTrace:
    def log(self, *args, **kwargs):
        pass


class _FakeStreamContext:
    """Stand-in for anthropic's MessageStreamManager; `result` is a Message or an exception."""

    def __init__(self, result):
        self._result = result

    def __enter__(self):
        if isinstance(self._result, BaseException):
            raise self._result
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def get_final_message(self):
        return self._result


class _StubMessages:
    """Pops one scripted anthropic.types.Message (or exception) per call, in order; records the
    kwargs of the last call so a test can inspect exactly what was sent."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.last_kwargs: dict | None = None

    def stream(self, **kwargs):
        self.last_kwargs = kwargs
        return _FakeStreamContext(self._responses.pop(0))


class _StubClient:
    def __init__(self, responses):
        self.messages = _StubMessages(responses)


def _config(tmp_path, **overrides):
    task_dir = tmp_path / "task"
    task_dir.mkdir(exist_ok=True)
    (task_dir / "description.md").write_text("Predict something.")
    fields = dict(
        vllm_base_url="http://stub", model_name="claude-sonnet-5", max_iterations=10, max_duration_s=300,
        task_dir=task_dir, task_name="fake", models_dir=tmp_path / "models", run_dir=tmp_path,
        workspace_dir=tmp_path / "workspace", trace_path=tmp_path / "trace.jsonl", bash_timeout_s=5,
        verbose=False, web_allowlist=frozenset(), web_denylist=frozenset(), oracle_url=None, gpu_uuid="none",
        temperature=None, top_p=None, top_k=None, min_p=None, presence_penalty=None, repetition_penalty=None,
        information="none", harness="react", verification="asked", budget="medium", model="claude-sonnet-5",
        replicate=1, provider="anthropic", effort="high",
    )
    return Config(**{**fields, **overrides})


def _events(config):
    return [json.loads(line) for line in Path(config.trace_path).read_text().splitlines() if line]


def _complete(conv, client, messages, tool_schemas, enable_thinking, persist_reasoning, deadline_ts, trace, thought_number):
    """get_response() followed by parse_response()."""
    response = conv.get_response(
        client, messages, tool_schemas, enable_thinking, persist_reasoning,
        deadline_ts=deadline_ts, trace=trace, thought_number=thought_number,
    )
    return conv.parse_response(response, persist_reasoning, thought_number=thought_number)


def _run_two_turns(tmp_path, harness: str):
    """Two tool-calling turns, each with its own thinking block, fed through _complete() -- the
    shared setup for the message-translation tests below."""
    config = _config(tmp_path, harness=harness)
    conv = Conversation(config)
    trace = _NullTrace()
    client = _StubClient([
        _message([_thinking("t1"), _tool_use("read_file", "toolu_1", path="/a")]),
        _message([_thinking("t2"), _tool_use("read_file", "toolu_2", path="/b")]),
    ])
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "go"}]

    msg1, _, _, _ = _complete(
        conv, client, messages, [_READ_FILE_SCHEMA], True, False, deadline_ts=1e12, trace=trace, thought_number=1,
    )
    messages += [msg1, {"role": "tool", "tool_call_id": "toolu_1", "content": "ok a"}]
    msg2, _, _, _ = _complete(
        conv, client, messages, [_READ_FILE_SCHEMA], True, False, deadline_ts=1e12, trace=trace, thought_number=2,
    )
    messages += [msg2, {"role": "tool", "tool_call_id": "toolu_2", "content": "ok b"}]
    return conv, messages


# --- tool schema conversion: strict mode ----------------------------------------------------------


def test_claude_tools_are_strict_with_additional_properties_closed():
    openai_schemas = build_tool_schemas("binding")  # the level where expected_score/verification_evidence apply
    claude_tools = _to_claude_tools(openai_schemas)

    finish_tool = next(t for t in claude_tools if t["name"] == "finish")
    assert finish_tool["strict"] is True
    assert finish_tool["input_schema"]["additionalProperties"] is False
    # model_used stays optional under strict mode.
    assert "model_used" in finish_tool["input_schema"]["properties"]
    assert "model_used" not in finish_tool["input_schema"]["required"]
    assert "expected_score" in finish_tool["input_schema"]["required"]
    assert "verification_evidence" in finish_tool["input_schema"]["required"]


def test_claude_tools_conversion_does_not_mutate_the_shared_openai_schema():
    """_to_claude_tools must not mutate the shared OpenAI schemas."""
    openai_schemas = build_tool_schemas("binding")
    finish_schema = next(s for s in openai_schemas if s["function"]["name"] == "finish")
    params_before = dict(finish_schema["function"]["parameters"])

    _to_claude_tools(openai_schemas)

    assert finish_schema["function"]["parameters"] == params_before
    assert "additionalProperties" not in finish_schema["function"]["parameters"]


# --- message translation: the harness axis's persistence rule, implemented on native content -----


def test_think_act_keeps_only_the_mandatory_last_turns_thinking(tmp_path):
    """think-act keeps thinking only for the turn whose tool_result is in this request."""
    conv, messages = _run_two_turns(tmp_path, "think-act")
    built = conv._build_request_messages(messages[1:], persist_reasoning=False)
    assistant_turns = [m for m in built if m["role"] == "assistant"]
    assert len(assistant_turns) == 2
    assert not any(b["type"] == "thinking" for b in assistant_turns[0]["content"]), "earlier turn must be stripped"
    assert any(b["type"] == "thinking" for b in assistant_turns[1]["content"]), "last turn is mandatory, must survive"


def test_think_act_strips_a_last_turn_with_no_tool_use_immediately(tmp_path):
    """think-act drops thinking of a last turn without tool_use."""
    config = _config(tmp_path, harness="think-act")
    conv = Conversation(config)
    client = _StubClient([_message([_thinking("why"), _text("just thinking out loud")], stop_reason="end_turn")])
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "go"}]

    msg, _, _, raw_tool_calls = _complete(
        conv, client, messages, [_READ_FILE_SCHEMA], True, False, deadline_ts=1e12, trace=_NullTrace(), thought_number=1,
    )
    assert not raw_tool_calls
    messages += [msg, {"role": "user", "content": "nudge: you must call a tool"}]

    built = conv._build_request_messages(messages[1:], persist_reasoning=False)
    assistant_turns = [m for m in built if m["role"] == "assistant"]
    assert len(assistant_turns) == 1
    assert not any(b["type"] == "thinking" for b in assistant_turns[0]["content"])


def test_react_keeps_every_turns_thinking_block(tmp_path):
    conv, messages = _run_two_turns(tmp_path, "react")
    built = conv._build_request_messages(messages[1:], persist_reasoning=True)
    assistant_turns = [m for m in built if m["role"] == "assistant"]
    assert len(assistant_turns) == 2
    assert all(any(b["type"] == "thinking" for b in t["content"]) for t in assistant_turns)


def test_system_prompt_is_extracted_from_messages_not_sent_as_a_message(tmp_path):
    config = _config(tmp_path, harness="act-only")
    conv = Conversation(config)
    client = _StubClient([_message([_text("done")], stop_reason="end_turn")])
    messages = [{"role": "system", "content": "you are a helpful agent"}, {"role": "user", "content": "go"}]

    _complete(conv, client, messages, [_READ_FILE_SCHEMA], False, False, deadline_ts=1e12, trace=_NullTrace(), thought_number=1)

    sent = client.messages.last_kwargs
    assert sent["system"] == "you are a helpful agent"
    assert all(m["role"] != "system" for m in sent["messages"])
    assert sent["thinking"] == {"type": "disabled"}  # act-only
    assert sent["output_config"] == {"effort": "high"}


def test_thinking_requests_summarized_display_when_enabled(tmp_path):
    """display="summarized" is required; the default returns empty thinking text."""
    config = _config(tmp_path, harness="react")
    conv = Conversation(config)
    client = _StubClient([_message([_thinking("why"), _text("go")], stop_reason="end_turn")])
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "go"}]

    conv.get_response(client, messages, [_READ_FILE_SCHEMA], True, True, deadline_ts=1e12, trace=_NullTrace(), thought_number=1)

    assert client.messages.last_kwargs["thinking"] == {"type": "adaptive", "display": "summarized"}


def test_parallel_tool_calls_bundle_into_one_tool_result_message(tmp_path):
    config = _config(tmp_path, harness="act-only")
    conv = Conversation(config)
    client = _StubClient([_message([_tool_use("read_file", "toolu_1", path="/a"), _tool_use("read_file", "toolu_2", path="/b")])])
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "go"}]

    msg, _, _, _ = _complete(conv, client, messages, [_READ_FILE_SCHEMA], False, False, deadline_ts=1e12, trace=_NullTrace(), thought_number=1)
    assert len(msg["tool_calls"]) == 2
    messages += [
        msg,
        {"role": "tool", "tool_call_id": "toolu_1", "content": "content a"},
        {"role": "tool", "tool_call_id": "toolu_2", "content": "content b"},
    ]

    built = conv._build_request_messages(messages[1:], persist_reasoning=False)
    tool_result_messages = [
        m for m in built
        if m["role"] == "user" and isinstance(m["content"], list) and m["content"][0]["type"] == "tool_result"
    ]
    assert len(tool_result_messages) == 1, "must be one message, not two -- splitting trains Claude off parallel calls"
    assert [b["tool_use_id"] for b in tool_result_messages[0]["content"]] == ["toolu_1", "toolu_2"]


def test_a_failed_tool_result_is_flagged_is_error(tmp_path):
    config = _config(tmp_path, harness="act-only")
    conv = Conversation(config)
    body = [{"role": "tool", "tool_call_id": "toolu_1", "content": "error: SandboxViolation: nope"}]
    built = conv._build_request_messages(body, persist_reasoning=False)
    assert built == [{"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "toolu_1", "content": "error: SandboxViolation: nope", "is_error": True},
    ]}]


def test_reasoning_folds_into_content_only_when_persisting(tmp_path):
    config = _config(tmp_path, harness="react")
    conv = Conversation(config)
    client = _StubClient([_message([_thinking("why"), _text("answer")], stop_reason="end_turn")])
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "go"}]

    msg, raw, _, _ = _complete(conv, client, messages, [_READ_FILE_SCHEMA], True, True, deadline_ts=1e12, trace=_NullTrace(), thought_number=3)
    assert msg["content"] == "Thought 3: why\n\nanswer"
    assert raw["reasoning"] == "why" and raw["content"] == "answer"  # raw_output keeps both channels separate


# --- usage normalization: cache reads/writes counted, but folded into prompt_tokens for parity ----


def test_usage_normalizes_cache_tokens_into_prompt_tokens(tmp_path):
    config = _config(tmp_path, harness="act-only")
    conv = Conversation(config)
    response = _message([_text("done")], stop_reason="end_turn")
    response.usage = _usage(input_tokens=50, output_tokens=20, cache_creation=30, cache_read=100)
    client = _StubClient([response])
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "go"}]

    _, _, usage, _ = _complete(conv, client, messages, [_READ_FILE_SCHEMA], False, False, deadline_ts=1e12, trace=_NullTrace(), thought_number=1)
    assert usage["prompt_tokens"] == 50 + 30 + 100
    assert usage["completion_tokens"] == 20
    assert usage["cache_read_input_tokens"] == 100 and usage["cache_creation_input_tokens"] == 30


# --- retry classification ------------------------------------------------------------------------


def test_transient_errors_are_retried_until_success():
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    calls = {"n": 0}

    class _FlakyMessages:
        def stream(self, **kwargs):
            calls["n"] += 1
            if calls["n"] < 3:
                return _FakeStreamContext(anthropic.APIConnectionError(request=request))
            return _FakeStreamContext("ok")

    client = type("C", (), {"messages": _FlakyMessages()})()
    result = _create_with_retry(client, {}, deadline_ts=time_ahead(), trace=_NullTrace())
    assert result == "ok" and calls["n"] == 3


def test_non_transient_errors_are_not_retried_forever():
    class _AlwaysFailsMessages:
        def stream(self, **kwargs):
            return _FakeStreamContext(ValueError("bad request shape"))

    client = type("C", (), {"messages": _AlwaysFailsMessages()})()
    with pytest.raises(ValueError):
        _create_with_retry(client, {}, deadline_ts=time_ahead(), trace=_NullTrace())


def time_ahead(seconds: float = 60.0) -> float:
    import time

    return time.time() + seconds


def test_is_transient_classifies_by_exception_type():
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    assert _is_transient(anthropic.APIConnectionError(request=request)) is True
    assert _is_transient(ValueError("nope")) is False


# --- the agent loop, end to end against a stub Claude client --------------------------------------


def test_loop_runs_to_finish_with_claude_backend(tmp_path, monkeypatch):
    config = _config(tmp_path)
    submission = tmp_path / "workspace" / "out.csv"
    responses = [
        _message([_thinking(), _tool_use("write_file", "toolu_1", path=str(submission), content="id,y\n")]),
        _message([_tool_use("finish", "toolu_2", submission_path=str(submission), summary="done")]),
    ]
    monkeypatch.setattr("agent.agent.claude_backend.build_client", lambda: _StubClient(responses))

    assert run(config) == 0
    events = _events(config)
    run_end = events[-1]
    assert run_end["event"] == "run_end" and run_end["status"] == "finished"
    assert run_end["submission_path"] == str(submission)
    assert run_end["prompt_tokens"] == 200 and run_end["completion_tokens"] == 20


# --- trace_render.py's LLM calls route by provider -------------------------------------------------


def test_llm_section_uses_the_anthropic_client_for_a_claude_model(monkeypatch):
    """trace_render's LLM sections use the Anthropic client for Claude runs."""
    captured = {}

    class _FakeAnthropicClient:
        def __init__(self, **kwargs):
            captured["client_kwargs"] = kwargs
            self.messages = self

        def create(self, **kwargs):
            captured["create_kwargs"] = kwargs
            return _message([_text("a real summary")], stop_reason="end_turn")

    monkeypatch.setattr("scripts.analysis.trace_render.anthropic.Anthropic", _FakeAnthropicClient)
    result = _llm_section("system prompt", "user content", base_url="http://unused", model="claude-sonnet-5", max_tokens=100)

    assert result == "a real summary"
    assert captured["create_kwargs"]["model"] == "claude-sonnet-5"
    assert captured["create_kwargs"]["system"] == "system prompt"
    assert captured["create_kwargs"]["thinking"] == {"type": "disabled"}


def test_llm_section_still_uses_openai_for_a_vllm_model(monkeypatch):
    calls = []

    class _FakeOpenAIClient:
        def __init__(self, **kwargs):
            self.chat = self
            self.completions = self

        def create(self, **kwargs):
            calls.append(kwargs)
            message = type("M", (), {"content": "vllm summary"})()
            return type("R", (), {"choices": [type("C", (), {"message": message})()]})()

    monkeypatch.setattr("scripts.analysis.trace_render.OpenAI", _FakeOpenAIClient)
    result = _llm_section("sys", "user", base_url="http://localhost:8000/v1", model="qwen35-35b-a3b-fp8", max_tokens=100)

    assert result == "vllm summary"
    assert len(calls) == 1


def test_run_start_records_effort_not_sampling_params_for_claude(tmp_path, monkeypatch):
    config = _config(tmp_path)
    responses = [_message([_tool_use("finish", "toolu_1", submission_path="x", summary="s")], stop_reason="tool_use")]
    (tmp_path / "workspace").mkdir(exist_ok=True)
    (tmp_path / "workspace" / "x").write_text("id,y\n")
    monkeypatch.setattr("agent.agent.claude_backend.build_client", lambda: _StubClient(responses))

    run(config)
    run_start = _events(config)[0]
    assert run_start["provider"] == "anthropic"
    assert run_start["sampling"] == {"effort": "high"}


# --- the API key never touches this process's own argv --------------------------------------------


def _apptainer_command_inputs(tmp_path, monkeypatch, api_key="sk-ant-secret-value"):
    monkeypatch.setenv("CACHE_ROOT", str(tmp_path / "cache"))
    if api_key is None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    else:
        monkeypatch.setenv("ANTHROPIC_API_KEY", api_key)
    spec = RunSpec(task="t", model="claude-sonnet-5")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    description_path = tmp_path / "description.md"
    description_path.write_text("desc")
    return dict(
        spec=spec, task_dir=task_dir, run_dir=run_dir, description_path=description_path,
        gpu_uuid="none", base_url="http://x", verbose=False, web_allowlist="", web_denylist="",
        oracle_url=None, provider="anthropic",
    )


def test_api_key_reaches_the_container_via_env_file_not_argv(tmp_path, monkeypatch):
    """A --env NAME=value argument is part of this process's own argv, visible to any other user
    on a shared HPC node via `ps`/`/proc` -- a paid API key must never appear there."""
    kwargs = _apptainer_command_inputs(tmp_path, monkeypatch)
    argv, _ = orchestrate.build_apptainer_command(**kwargs)

    assert "sk-ant-secret-value" not in " ".join(argv)
    assert "--env-file" in argv
    secrets_path = Path(argv[argv.index("--env-file") + 1])
    assert secrets_path.read_text() == "ANTHROPIC_API_KEY=sk-ant-secret-value\n"
    assert oct(secrets_path.stat().st_mode)[-3:] == "600"
    assert "PROVIDER=anthropic" in argv


def test_missing_api_key_fails_before_container_launch(tmp_path, monkeypatch):
    kwargs = _apptainer_command_inputs(tmp_path, monkeypatch, api_key=None)
    with pytest.raises(SystemExit, match="ANTHROPIC_API_KEY"):
        orchestrate.build_apptainer_command(**kwargs)


# --- the network-call/parsing exception boundary ---------------------------------------------------


def test_a_parsing_bug_crashes_the_run_instead_of_being_reported_as_an_llm_error(tmp_path, monkeypatch):
    """A parsing bug ends the run as crashed, not llm_error."""
    config = _config(tmp_path)
    responses = [_message([_text("hi")], stop_reason="end_turn")]
    monkeypatch.setattr("agent.agent.claude_backend.build_client", lambda: _StubClient(responses))
    monkeypatch.setattr(
        "agent.backends.claude.Conversation.parse_response",
        lambda self, response, persist_reasoning, thought_number: (_ for _ in ()).throw(TypeError("boom")),
    )

    assert run(config) == 1
    run_end = _events(config)[-1]
    assert run_end["status"] == "crashed"
    assert "TypeError" in run_end["error"] and "traceback" in run_end


# --- per-call timeout is capped by the run's own remaining budget ----------------------------------


def test_call_timeout_is_capped_by_remaining_run_budget(tmp_path):
    """Each call's timeout is capped by the remaining run budget."""
    import time

    from agent.backends.claude import MAX_CALL_TIMEOUT_S, MIN_CALL_TIMEOUT_S, _create_with_retry

    client = _StubClient([_message([_text("hi")], stop_reason="end_turn")])
    short_deadline = time.time() + 120  # well under MAX_CALL_TIMEOUT_S
    _create_with_retry(client, {}, deadline_ts=short_deadline, trace=_NullTrace())
    assert MIN_CALL_TIMEOUT_S < client.messages.last_kwargs["timeout"] <= 120

    client2 = _StubClient([_message([_text("hi")], stop_reason="end_turn")])
    long_deadline = time.time() + 10_000  # well over MAX_CALL_TIMEOUT_S
    _create_with_retry(client2, {}, deadline_ts=long_deadline, trace=_NullTrace())
    assert client2.messages.last_kwargs["timeout"] == MAX_CALL_TIMEOUT_S

    client3 = _StubClient([_message([_text("hi")], stop_reason="end_turn")])
    almost_expired_deadline = time.time() + 1  # less than MIN_CALL_TIMEOUT_S left
    _create_with_retry(client3, {}, deadline_ts=almost_expired_deadline, trace=_NullTrace())
    assert client3.messages.last_kwargs["timeout"] == MIN_CALL_TIMEOUT_S
