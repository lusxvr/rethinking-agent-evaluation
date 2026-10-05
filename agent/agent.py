import json
import re
import subprocess
import sys
import time
import traceback

from openai import APIConnectionError, APIStatusError, OpenAI

from agent.backends import claude as claude_backend
from agent.config import Config, generates_reasoning, persists_reasoning
from agent.logging_utils import TraceLogger
from agent.prompts import build_system_prompt, build_user_message
from agent.tools import FinishSignal, RawToolCall, Tools, build_tool_schemas

MAX_CONSECUTIVE_NO_TOOL_CALLS = 5
LLM_RETRY_DELAY_S = 5.0
LLM_TIMEOUT_S = 180.0


def _preview(text: str, limit: int = 160) -> str:
    """Length-capped preview for verbose stdout logging; trace.jsonl always has the full value."""
    text = text.replace("\n", " \\n ")
    return text if len(text) <= limit else text[:limit] + "..."


def _describe_gpu() -> str:
    """GPU name plus MIG profile (e.g. "1g.35gb") from `nvidia-smi -L`."""
    try:
        result = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=10)
        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        gpu_match = re.search(r"GPU \d+: (.+?) \(UUID", lines[0]) if lines else None
        gpu_name = gpu_match.group(1) if gpu_match else "unknown"
        mig_line = next((line for line in lines if line.startswith("MIG")), None)
        return f"{gpu_name}, MIG {mig_line.split()[1]}" if mig_line else gpu_name
    except Exception:
        return "unknown"


def _format_duration(seconds: float) -> str:
    m, s = divmod(round(seconds), 60)
    return f"{m}m{s:02d}s"


def _assistant_message_dict(message, persist_reasoning: bool, thought_number: int = 1) -> dict:
    """Assistant message for history; persisted reasoning is prefixed as ReAct's `Thought N:`.

    Actions stay in the native tool-call format, which the model is post-trained on.
    """
    content = message.content or ""
    reasoning = getattr(message, "reasoning", None)
    if persist_reasoning and reasoning:
        # The chat template drops reasoning fields and <think> tags; only content text survives.
        thought = f"Thought {thought_number}: {reasoning}"
        content = f"{thought}\n\n{content}" if content else thought
    result: dict = {"role": "assistant", "content": content}
    if message.tool_calls:
        result["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": _sanitized_arguments(tc.function.arguments)},
            }
            for tc in message.tool_calls
        ]
    return result


def _sanitized_arguments(arguments: str | None) -> str:
    """Tool-call arguments for history, replaced by "{}" if not valid JSON.

    The chat template re-parses history, so invalid JSON would fail every later request.
    trace.jsonl keeps the original in raw_output.
    """
    try:
        json.loads(arguments or "{}")
    except json.JSONDecodeError:
        return "{}"
    return arguments or "{}"


def _reasoning_effort_for(model: str, enable_thinking: bool) -> str | None:
    """reasoning_effort for Step-3.7-Flash, whose template ignores enable_thinking; None otherwise.

    "none" does not actually disable thinking, so Step-3.7 specs omit act-only (see axes.py).
    """
    if not model.startswith("step37-"):
        return None
    return "medium" if enable_thinking else "none"


def _complete(
    client: OpenAI, config: Config, messages: list[dict], tool_schemas: list, enable_thinking: bool,
    deadline_ts: float, trace: TraceLogger,
):
    """One chat completion. Transient errors (timeout/connection/5xx) retry until deadline_ts.
    Any other error retries once, then raises."""
    attempt = 0
    reasoning_effort = _reasoning_effort_for(config.model, enable_thinking)
    while True:
        try:
            return client.chat.completions.create(
                model=config.model_name,
                messages=messages,
                tools=tool_schemas,
                # Verify "auto" when adding a model; some need "required".
                tool_choice="auto",
                temperature=config.temperature,
                top_p=config.top_p,
                presence_penalty=config.presence_penalty,
                # Non-OpenAI sampling fields go through extra_body.
                extra_body={
                    "top_k": config.top_k,
                    "min_p": config.min_p,
                    "repetition_penalty": config.repetition_penalty,
                    # False only for act-only.
                    "chat_template_kwargs": {"enable_thinking": enable_thinking},
                    # Only set for Step-3.7-Flash.
                    **({"reasoning_effort": reasoning_effort} if reasoning_effort is not None else {}),
                },
            )
        except Exception as exc:
            attempt += 1
            transient = isinstance(exc, APIConnectionError) or (
                isinstance(exc, APIStatusError) and exc.status_code >= 500
            )
            if transient:
                remaining = deadline_ts - time.time()
                if remaining <= 0:
                    raise
                delay = min(LLM_RETRY_DELAY_S, remaining)
            else:
                if attempt >= 2:
                    raise
                delay = LLM_RETRY_DELAY_S
            trace.log(
                "llm_retry", attempt=attempt, transient=transient, delay_s=delay,
                error=f"{type(exc).__name__}: {exc}",
            )
            print(
                f"warning: LLM call failed (attempt {attempt}, {type(exc).__name__}: {exc}); "
                f"retrying in {delay:.0f}s"
                + (f" ({remaining:.0f}s of run budget left)" if transient else "")
            )
            time.sleep(delay)


def run(config: Config) -> int:
    """Run the agent loop. Every exit goes through end(), so a missing run_end means an external kill."""
    trace = TraceLogger(config.trace_path)

    def vprint(msg: str) -> None:
        # Live progress on stdout; trace.jsonl has the full record.
        if config.verbose:
            print(msg)

    # Run totals. prompt_tokens sums every call's re-sent history (cost, not context size).
    iterations = 0
    prompt_tokens = 0
    completion_tokens = 0
    # Anthropic-only cache breakdown for axes.anthropic_actual_cost_usd; 0 for vLLM.
    input_tokens = 0
    output_tokens = 0
    cache_creation_input_tokens = 0
    cache_read_input_tokens = 0
    start_ts = time.time()

    def end(status: str, note: str, **fields) -> int:
        trace.log(
            "run_end",
            status=status,
            iterations=iterations,
            wallclock_s=time.time() - start_ts,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_creation_input_tokens=cache_creation_input_tokens,
            cache_read_input_tokens=cache_read_input_tokens,
            **fields,
        )
        vprint(f"=== {note} ===")
        trace.close()
        return 0 if status == "finished" else 1

    try:
        tools = Tools(
            config.task_dir,
            config.workspace_dir,
            config.bash_timeout_s,
            config.verification,
            models_dir=config.models_dir,
            web_allowlist=config.web_allowlist,
            web_denylist=config.web_denylist,
            oracle_url=config.oracle_url,
        )
        # max_retries=0: _complete and the Claude backend handle retries themselves.
        if config.provider == "anthropic":
            client = claude_backend.build_client()
            conversation = claude_backend.Conversation(config)
        else:
            client = OpenAI(base_url=config.vllm_base_url, api_key="EMPTY", timeout=LLM_TIMEOUT_S, max_retries=0)
            conversation = None

        oracle_enabled = config.oracle_url is not None
        task_description = (config.task_dir / "description.md").read_text()
        tool_schemas = build_tool_schemas(config.verification, oracle_enabled=oracle_enabled)
        messages: list[dict] = [
            {"role": "system", "content": build_system_prompt(config.verification, oracle_enabled=oracle_enabled)},
            {"role": "user", "content": build_user_message(task_description, config.max_duration_s)},
        ]
        # Logged once; later events carry each appended message under "message".
        trace.log(
            "run_start",
            model_name=config.model_name,
            task_dir=str(config.task_dir),
            task_name=config.task_name,
            gpu_uuid=config.gpu_uuid,
            gpu=_describe_gpu(),
            max_iterations=config.max_iterations,
            max_duration_s=config.max_duration_s,
            provider=config.provider,
            # Only the provider's own sampling fields.
            sampling=(
                {"effort": config.effort}
                if config.provider == "anthropic"
                else {
                    "temperature": config.temperature,
                    "top_p": config.top_p,
                    "top_k": config.top_k,
                    "min_p": config.min_p,
                    "presence_penalty": config.presence_penalty,
                    "repetition_penalty": config.repetition_penalty,
                }
            ),
            # Labels beside resolved values, so later axes.py edits cannot reinterpret old runs.
            information=config.information,
            harness=config.harness,
            verification=config.verification,
            oracle_enabled=oracle_enabled,
            budget=config.budget,
            model=config.model,
            replicate=config.replicate,
            seed_messages=messages,
        )
        vprint(
            f"=== task={config.task_name!r} model={config.model_name!r} harness={config.harness!r} "
            f"budget={config.max_duration_s}s (iteration cap {config.max_iterations}) ==="
        )

        # Harness level as two flags (see agent/config.py).
        enable_thinking = generates_reasoning(config.harness)
        persist_reasoning = persists_reasoning(config.harness)

        def out_of_budget(iteration: int) -> int | None:
            """Exit code if the wall clock has run out, else None. Checked before and after tool calls."""
            if time.time() - start_ts < config.max_duration_s:
                return None
            return end(
                "max_duration_reached",
                f"max duration ({config.max_duration_s}s) reached at iteration {iteration}",
            )

        consecutive_no_tool_calls = 0
        for iteration in range(config.max_iterations):
            if (returncode := out_of_budget(iteration)) is not None:
                return returncode

            # Only the network call is inside this try, so parsing bugs surface as crashes.
            try:
                if conversation is not None:
                    response = conversation.get_response(
                        client, messages, tool_schemas, enable_thinking, persist_reasoning,
                        deadline_ts=start_ts + config.max_duration_s, trace=trace, thought_number=iteration + 1,
                    )
                else:
                    response = _complete(
                        client, config, messages, tool_schemas, enable_thinking,
                        deadline_ts=start_ts + config.max_duration_s, trace=trace,
                    )
            except Exception as exc:
                if (returncode := out_of_budget(iteration)) is not None:
                    return returncode
                return end(
                    "llm_error",
                    f"LLM call failed at iteration {iteration}: {type(exc).__name__}: {exc}",
                    error=f"{type(exc).__name__}: {exc}",
                )

            if conversation is not None:
                # 1-indexed: ReAct's trajectories start at "Thought 1".
                assistant_message, raw_output, usage, raw_tool_calls = conversation.parse_response(
                    response, persist_reasoning, thought_number=iteration + 1,
                )
            else:
                message = response.choices[0].message
                usage = response.usage.model_dump() if response.usage else None
                assistant_message = _assistant_message_dict(
                    message, persist_reasoning=persist_reasoning, thought_number=iteration + 1
                )
                raw_output = message.model_dump()  # includes fields not carried into history, e.g. reasoning
                raw_tool_calls = [
                    RawToolCall(id=tc.id, name=tc.function.name, arguments=tc.function.arguments)
                    for tc in (message.tool_calls or [])
                ]
            trace.log(
                "llm_response",
                iteration=iteration,
                message=assistant_message,
                raw_output=raw_output,
                usage=usage,
            )
            iterations += 1
            if usage:
                prompt_tokens += usage.get("prompt_tokens", 0)
                completion_tokens += usage.get("completion_tokens", 0)
                input_tokens += usage.get("input_tokens", 0)
                output_tokens += usage.get("output_tokens", 0)
                cache_creation_input_tokens += usage.get("cache_creation_input_tokens", 0)
                cache_read_input_tokens += usage.get("cache_read_input_tokens", 0)
            messages.append(assistant_message)

            if not raw_tool_calls:
                consecutive_no_tool_calls += 1
                vprint(f"[iter {iteration}] no tool call ({consecutive_no_tool_calls}/{MAX_CONSECUTIVE_NO_TOOL_CALLS})")
                if consecutive_no_tool_calls >= MAX_CONSECUTIVE_NO_TOOL_CALLS:
                    return end(
                        "stalled_no_tool_calls",
                        f"stalled: no tool call for {MAX_CONSECUTIVE_NO_TOOL_CALLS} turns in a row",
                    )
                nudge_message = {
                    "role": "user",
                    "content": "You must call a tool to make progress. Call `finish` once the "
                    "submission file is ready.",
                }
                trace.log("nudge_message", iteration=iteration, message=nudge_message)
                messages.append(nudge_message)
                continue
            consecutive_no_tool_calls = 0

            for tool_call in raw_tool_calls:
                name = tool_call.name
                try:
                    arguments = json.loads(tool_call.arguments or "{}")
                except json.JSONDecodeError as exc:
                    result_text = f"error: could not parse arguments as JSON: {exc}"
                    tool_message = {"role": "tool", "tool_call_id": tool_call.id, "content": result_text}
                    trace.log(
                        "tool_call",
                        iteration=iteration,
                        name=name,
                        arguments_raw=tool_call.arguments,
                        message=tool_message,
                    )
                    vprint(f"[iter {iteration}] {name}({tool_call.arguments!r}) -> {result_text}")
                    messages.append(tool_message)
                    continue

                vprint(f"[iter {iteration}] {name}({_preview(json.dumps(arguments))})")
                try:
                    result_text, finish_signal = tools.dispatch(name, arguments)
                except Exception as exc:
                    # A tool failure is the model's to react to, never the run's to die on.
                    result_text = f"error: {type(exc).__name__}: {exc}"
                    finish_signal = None

                tool_message = {"role": "tool", "tool_call_id": tool_call.id, "content": result_text}
                trace.log(
                    "tool_call",
                    iteration=iteration,
                    name=name,
                    arguments=arguments,
                    message=tool_message,
                )
                vprint(f"[iter {iteration}]   -> {_preview(result_text)}")
                messages.append(tool_message)

                if isinstance(finish_signal, FinishSignal):
                    return end(
                        "finished",
                        f"finished: {finish_signal.submission_path}",
                        submission_path=finish_signal.submission_path,
                        summary=finish_signal.summary,
                        model_used=finish_signal.model_used,
                        expected_score=finish_signal.expected_score,
                        verification_evidence=finish_signal.verification_evidence,
                    )

            if (returncode := out_of_budget(iteration)) is not None:
                return returncode

            # The model's only signal about the remaining budget.
            elapsed_s = time.time() - start_ts
            status_message = {
                "role": "user",
                "content": (
                    f"[Status: {config.max_duration_s - elapsed_s:.0f}s left "
                    f"({_format_duration(elapsed_s)} elapsed of {_format_duration(config.max_duration_s)}).]"
                ),
            }
            trace.log("status_message", iteration=iteration, message=status_message)
            messages.append(status_message)

        # Runaway guard (axes.ITERATION_CAPS), not the budget.
        return end(
            "iteration_cap_reached",
            f"iteration cap ({config.max_iterations}) reached without finishing",
        )
    except KeyboardInterrupt:
        # Host Ctrl-C; recorded separately from a crash.
        return end("interrupted", "interrupted")
    except Exception as exc:
        return end(
            "crashed",
            f"crashed: {type(exc).__name__}: {exc}",
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
        )


def main() -> int:
    config = Config.from_env()
    return run(config)


if __name__ == "__main__":
    sys.exit(main())
