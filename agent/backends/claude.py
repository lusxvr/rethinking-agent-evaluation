"""Claude API backend: translates the agent's OpenAI-shaped messages and tools to the Messages API.

Design rationale: docs/anthropic-integration.md.
"""

import json
import time
from typing import Any

import anthropic

from agent.config import Config
from agent.logging_utils import TraceLogger
from agent.tools import RawToolCall

# Per-call timeout ceiling; each attempt is also capped at the remaining run budget.
MAX_CALL_TIMEOUT_S = 600.0
# Floor so a call near the end of the budget still has a chance to return.
MIN_CALL_TIMEOUT_S = 30.0
RETRY_DELAY_S = 5.0
# Per-turn output cap (thinking plus tool call). Streaming allows values above the SDK's
# non-streaming limit.
MAX_TOKENS = 32000

_THINKING_BLOCK_TYPES = {"thinking", "redacted_thinking"}


def build_client() -> anthropic.Anthropic:
    """Client reading ANTHROPIC_API_KEY from the environment; retries are handled by _create_with_retry."""
    return anthropic.Anthropic(timeout=MAX_CALL_TIMEOUT_S, max_retries=0)


def _to_claude_tools(openai_tool_schemas: list[dict]) -> list[dict]:
    """Convert OpenAI tool schemas to Claude tools.

    strict=True, since without it Claude can omit required properties such as expected_score.
    Properties missing from "required" stay optional under strict mode.
    """
    tools = []
    for t in openai_tool_schemas:
        params = t["function"]["parameters"]
        tools.append({
            "name": t["function"]["name"],
            "description": t["function"]["description"],
            "input_schema": {**params, "additionalProperties": False},
            "strict": True,
        })
    return tools


def _is_transient(exc: Exception) -> bool:
    return (
        isinstance(exc, (anthropic.APIConnectionError, anthropic.RateLimitError))
        or (isinstance(exc, anthropic.APIStatusError) and exc.status_code >= 500)
    )


def _create_with_retry(client: anthropic.Anthropic, kwargs: dict, deadline_ts: float, trace: TraceLogger):
    """Stream one request. Transient errors retry until deadline_ts; others retry once, then raise."""
    attempt = 0
    while True:
        call_timeout = min(MAX_CALL_TIMEOUT_S, max(deadline_ts - time.time(), MIN_CALL_TIMEOUT_S))
        try:
            with client.messages.stream(**kwargs, timeout=call_timeout) as stream:
                return stream.get_final_message()
        except Exception as exc:
            attempt += 1
            transient = _is_transient(exc)
            if transient:
                remaining = deadline_ts - time.time()
                if remaining <= 0:
                    raise
                delay = min(RETRY_DELAY_S, remaining)
            else:
                if attempt >= 2:
                    raise
                delay = RETRY_DELAY_S
            trace.log(
                "llm_retry", attempt=attempt, transient=transient, delay_s=delay,
                error=f"{type(exc).__name__}: {exc}",
            )
            print(
                f"warning: Claude API call failed (attempt {attempt}, {type(exc).__name__}: {exc}); "
                f"retrying in {delay:.0f}s" + (f" ({remaining:.0f}s of run budget left)" if transient else "")
            )
            time.sleep(delay)


class Conversation:
    """Keeps each assistant turn's native content blocks, including thinking signatures.

    Claude requires the turn that produced a tool call to be replayed unchanged alongside its
    tool_result. Harness levels: react keeps all thinking blocks, think-act keeps only the one
    still required, act-only has none. See docs/anthropic-integration.md.
    """

    def __init__(self, config: Config):
        self._config = config
        self._claude_tools: list[dict] | None = None  # set on first get_response() call
        self._native_content: list[list[dict]] = []  # one entry per assistant turn, in order

    def get_response(
        self, client: anthropic.Anthropic, messages: list[dict], tool_schemas: list[dict],
        enable_thinking: bool, persist_reasoning: bool, deadline_ts: float, trace: TraceLogger,
        thought_number: int,
    ) -> anthropic.types.Message:
        """Network call for one turn; raises on failure. Parsing is in parse_response."""
        if self._claude_tools is None:
            self._claude_tools = _to_claude_tools(tool_schemas)

        system_text = messages[0]["content"]
        claude_messages = self._build_request_messages(messages[1:], persist_reasoning)

        kwargs: dict[str, Any] = dict(
            model=self._config.model_name,
            max_tokens=MAX_TOKENS,
            system=system_text,
            tools=self._claude_tools,
            tool_choice={"type": "auto"},
            # act-only disables thinking; effort stays fixed (axes._CLAUDE_SONNET5).
            # display="summarized" is required: the default returns empty thinking text.
            thinking={"type": "adaptive", "display": "summarized"} if enable_thinking else {"type": "disabled"},
            output_config={"effort": self._config.effort},
            messages=claude_messages,
            # Caches the request prefix; each turn pays full price only for new content.
            cache_control={"type": "ephemeral"},
        )
        response = _create_with_retry(client, kwargs, deadline_ts, trace)

        if response.stop_reason == "refusal":
            trace.log(
                "llm_refusal", iteration=thought_number - 1,
                category=getattr(response.stop_details, "category", None),
            )
        elif response.stop_reason == "max_tokens":
            # Not fatal: partial content still goes through parse_response.
            trace.log("llm_truncated", iteration=thought_number - 1, note="hit max_tokens -- response may be incomplete")
        return response

    def parse_response(
        self, response: anthropic.types.Message, persist_reasoning: bool, thought_number: int,
    ) -> tuple[dict, dict, dict, list[RawToolCall]]:
        """Convert a response into the vLLM path's shapes: (message, raw_output, usage, tool_calls)."""
        native_content = [block.model_dump() for block in response.content]
        self._native_content.append(native_content)

        thinking_text = "".join(
            block.get("thinking", "") for block in native_content if block.get("type") == "thinking"
        )
        text = "".join(block.get("text", "") for block in native_content if block.get("type") == "text")
        tool_use_blocks = [b for b in native_content if b.get("type") == "tool_use"]

        content = text
        if persist_reasoning and thinking_text:
            # Same "Thought N:" prefix as the vLLM path.
            thought = f"Thought {thought_number}: {thinking_text}"
            content = f"{thought}\n\n{content}" if content else thought

        raw_tool_calls = [RawToolCall(id=b["id"], name=b["name"], arguments=json.dumps(b["input"])) for b in tool_use_blocks]

        assistant_message: dict = {"role": "assistant", "content": content}
        if tool_use_blocks:
            assistant_message["tool_calls"] = [
                {"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": tc.arguments}}
                for tc in raw_tool_calls
            ]

        raw_output = {
            "reasoning": thinking_text or None,
            "content": text,
            "stop_reason": response.stop_reason,
            "id": response.id,
            # For exact replay only; analysis reads the fields above.
            "native_content": native_content,
        }

        u = response.usage
        usage = {
            # Includes cached tokens, as on the vLLM path; cache pricing is in axes.anthropic_actual_cost_usd.
            "prompt_tokens": u.input_tokens + (u.cache_creation_input_tokens or 0) + (u.cache_read_input_tokens or 0),
            "completion_tokens": u.output_tokens,
            "input_tokens": u.input_tokens,
            "output_tokens": u.output_tokens,
            "cache_creation_input_tokens": u.cache_creation_input_tokens or 0,
            "cache_read_input_tokens": u.cache_read_input_tokens or 0,
        }
        return assistant_message, raw_output, usage, raw_tool_calls

    def _build_request_messages(self, body: list[dict], persist_reasoning: bool) -> list[dict]:
        """Build Claude messages from the OpenAI-shaped history without the system message."""
        n_assistant_total = sum(1 for m in body if m["role"] == "assistant")
        claude_messages: list[dict] = []
        pending_tool_results: list[dict] = []
        assistant_index = 0

        def flush_tool_results() -> None:
            nonlocal pending_tool_results
            if pending_tool_results:
                # One user message per turn's tool_results; splitting them discourages parallel calls.
                claude_messages.append({"role": "user", "content": pending_tool_results})
                pending_tool_results = []

        for m in body:
            role = m["role"]
            if role == "tool":
                content_str = m["content"]
                block = {"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": content_str}
                if content_str.startswith("error:"):
                    block["is_error"] = True
                pending_tool_results.append(block)
            elif role == "user":
                flush_tool_results()
                claude_messages.append({"role": "user", "content": [{"type": "text", "text": m["content"]}]})
            elif role == "assistant":
                flush_tool_results()
                native = self._native_content[assistant_index]
                # A turn whose tool_result is in this request must replay its thinking unchanged.
                # react keeps every turn; think-act drops the rest.
                is_last = assistant_index == n_assistant_total - 1
                has_tool_use = any(b.get("type") == "tool_use" for b in native)
                content = native if (persist_reasoning or (is_last and has_tool_use)) else [
                    b for b in native if b.get("type") not in _THINKING_BLOCK_TYPES
                ]
                claude_messages.append({"role": "assistant", "content": content})
                assistant_index += 1
            else:
                raise ValueError(f"Unexpected role in history: {role!r}")
        flush_tool_results()
        return claude_messages
