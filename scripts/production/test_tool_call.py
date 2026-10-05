"""Sanity check that the running vLLM server can parse tool calls (and reasoning, for thinking
models) for the served model -- run before trusting the agent loop, and after any tool_choice/model change (reliability isn't portable across models)."""

import argparse
import json
import os
import sys

from openai import OpenAI

from axes import DEFAULT_MODEL_NAME

TOOL = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read the contents of a file",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Path to the file to read"}},
            "required": ["path"],
        },
    },
}


def main() -> int:
    parser = argparse.ArgumentParser()
    # Same default port as apptainer/vllm/serve.sh.
    parser.add_argument("--base-url", default=f"http://localhost:{os.environ.get('VLLM_PORT', '61000')}/v1")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--tool-choice", default="auto", choices=["required", "auto"])
    parser.add_argument("--enable-thinking", action="store_true", default=True)
    parser.add_argument("--no-enable-thinking", dest="enable_thinking", action="store_false")
    args = parser.parse_args()

    client = OpenAI(base_url=args.base_url, api_key="EMPTY")
    response = client.chat.completions.create(
        model=args.model,
        messages=[
            {
                "role": "user",
                "content": "Use the read_file tool to read the file at /task/description.md. "
                "Call the tool, do not just describe what you would do.",
            }
        ],
        tools=[TOOL],
        tool_choice=args.tool_choice,
        extra_body={"chat_template_kwargs": {"enable_thinking": args.enable_thinking}},
    )
    message = response.choices[0].message
    reasoning = getattr(message, "reasoning", None)

    print(f"tool_choice={args.tool_choice!r} enable_thinking={args.enable_thinking}")
    print("reasoning:", repr(reasoning)[:500] if reasoning else None)
    print("content:", message.content)
    print("tool_calls:", message.tool_calls)

    if not message.tool_calls:
        print("FAIL: no tool_calls returned")
        return 1

    call = message.tool_calls[0]
    if call.function.name != "read_file":
        print(f"FAIL: expected tool 'read_file', got '{call.function.name}'")
        return 1

    try:
        args_dict = json.loads(call.function.arguments)
    except json.JSONDecodeError as exc:
        print(f"FAIL: tool_call arguments not valid JSON: {exc}")
        return 1

    if args_dict.get("path") != "/task/description.md":
        print(f"WARN: expected path '/task/description.md', got {args_dict.get('path')!r}")

    print("PASS: tool call parsed correctly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
