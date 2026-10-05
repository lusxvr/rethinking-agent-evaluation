"""Check that the Anthropic API returns tool calls and thinking in the shape agent/backends/claude.py
expects. Makes one billed request.
"""

import sys
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]

TOOL = {
    "name": "read_file",
    "description": "Read the contents of a file",
    "input_schema": {
        "type": "object",
        "properties": {"path": {"type": "string", "description": "Path to the file to read"}},
        "required": ["path"],
    },
}


def main() -> int:
    if not load_dotenv(REPO_ROOT / ".env"):
        print(f"{REPO_ROOT / '.env'} not found -- run 'cp .env.example .env' and set ANTHROPIC_API_KEY")
        return 1

    client = Anthropic()  # reads ANTHROPIC_API_KEY from the environment .env just loaded
    response = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=4096,
        thinking={"type": "adaptive"},
        output_config={"effort": "high"},
        tools=[TOOL],
        tool_choice={"type": "auto"},
        messages=[{
            "role": "user",
            "content": "Use the read_file tool to read the file at /task/description.md. "
            "Call the tool, do not just describe what you would do.",
        }],
    )

    thinking = "".join(b.thinking for b in response.content if b.type == "thinking")
    text = "".join(b.text for b in response.content if b.type == "text")
    tool_use_blocks = [b for b in response.content if b.type == "tool_use"]

    print(f"stop_reason={response.stop_reason!r}")
    print("thinking:", repr(thinking)[:500] if thinking else None)
    print("text:", text)
    print("tool_use blocks:", tool_use_blocks)
    print(
        "usage:", response.usage.model_dump()
        if hasattr(response.usage, "model_dump") else response.usage,
    )

    if response.stop_reason == "refusal":
        print(f"FAIL: model refused (category={getattr(response.stop_details, 'category', None)})")
        return 1

    if not tool_use_blocks:
        print("FAIL: no tool_use block returned")
        return 1

    call = tool_use_blocks[0]
    if call.name != "read_file":
        print(f"FAIL: expected tool 'read_file', got {call.name!r}")
        return 1

    if call.input.get("path") != "/task/description.md":
        print(f"WARN: expected path '/task/description.md', got {call.input.get('path')!r}")

    print("PASS: tool call parsed correctly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
