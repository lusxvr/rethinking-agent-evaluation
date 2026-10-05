"""Reconstruct the exact messages list sent at a given iteration by replaying the trace's message deltas."""

import argparse
import json
import sys
from pathlib import Path


def reconstruct_input(trace_path: Path, target_iteration: int) -> list[dict]:
    events = [json.loads(line) for line in trace_path.read_text().splitlines() if line]

    run_start = next((e for e in events if e["event"] == "run_start"), None)
    if run_start is None:
        raise ValueError(f"no run_start event in {trace_path}")
    messages = list(run_start["seed_messages"])

    total_iterations = sum(1 for e in events if e["event"] == "llm_response")
    if not 0 <= target_iteration < total_iterations:
        raise ValueError(
            f"trace has iterations 0..{total_iterations - 1}, requested {target_iteration}"
        )

    llm_response_count = 0
    for e in events:
        if e["event"] == "llm_response":
            if llm_response_count == target_iteration:
                break
            llm_response_count += 1
        if "message" in e:
            messages.append(e["message"])

    return messages


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reconstruct the exact model input at a given iteration from a run's trace.jsonl"
    )
    parser.add_argument("trace", type=Path, help="Path to a run's trace.jsonl")
    parser.add_argument("--iteration", type=int, required=True, help="Iteration to reconstruct the input for")
    args = parser.parse_args()

    messages = reconstruct_input(args.trace, args.iteration)
    print(json.dumps(messages, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
