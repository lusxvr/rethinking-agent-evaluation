"""BACKBONE anchor: the backbone LLM alone on mmlu-astronomy, greedy (dev/results/backbone.json) and
with the agent's default sampling (backbone-agent-sampling.json).

Usage:
    cd models/astrosage/dev_env
    uv run --project . python ../../../solutions/mmlu-astronomy/dev/run_backbone.py
"""

import argparse
import csv
import json
import sys
from pathlib import Path

from mcq import build_prompt, parse_letter
from openai import OpenAI
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
from axes import DEFAULT_MODEL_NAME, DEFAULT_SAMPLING  # noqa: E402

DATA_DIR = REPO_ROOT / "tasks" / "mmlu-astronomy" / "data"
RESULTS_DIR = REPO_ROOT / "solutions" / "mmlu-astronomy" / "dev" / "results"

# Thinking length varies widely, so truncated answers are retried at a larger budget.
MAX_TOKENS_INITIAL = 2048
MAX_TOKENS_RETRY = 8192

# Same keys in both configs. Greedy uses vLLM defaults otherwise; AGENT_PARAMS is axes.DEFAULT_SAMPLING.
GREEDY_PARAMS = {**DEFAULT_SAMPLING, "temperature": 0.0, "top_p": 1.0, "top_k": -1, "presence_penalty": 0.0}
AGENT_PARAMS = dict(DEFAULT_SAMPLING)


def _load_questions() -> list[dict]:
    with (DATA_DIR / "questions.csv").open(newline="") as f:
        return list(csv.DictReader(f))


def _ask(client: OpenAI, model: str, prompt: str, max_tokens: int, params: dict):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=params["temperature"],
        top_p=params["top_p"],
        presence_penalty=params["presence_penalty"],
        # top_k/min_p/repetition_penalty aren't standard OpenAI fields; vLLM reads them from extra_body.
        extra_body={
            "top_k": params["top_k"],
            "min_p": params["min_p"],
            "repetition_penalty": params["repetition_penalty"],
            "chat_template_kwargs": {"enable_thinking": True},
        },
    )
    return response.choices[0], response.usage.completion_tokens


def _run_config(client: OpenAI, model: str, questions: list[dict], name: str, params: dict) -> dict:
    predictions = {}
    still_truncated = []
    for i, row in enumerate(tqdm(questions, desc=name)):
        choices = [row["choice_a"], row["choice_b"], row["choice_c"], row["choice_d"]]
        prompt = build_prompt(row["question"], choices)
        choice, completion_tokens = _ask(client, model, prompt, MAX_TOKENS_INITIAL, params)
        retried = False
        if choice.finish_reason == "length":
            retried = True
            choice, completion_tokens = _ask(client, model, prompt, MAX_TOKENS_RETRY, params)

        generated = choice.message.content or ""
        letter = parse_letter(generated)
        predictions[row["question_id"]] = letter or "?"
        if choice.finish_reason == "length":
            still_truncated.append(row["question_id"])
        flag = " [retried]" if retried else ""
        if i % 50 == 0:
            tqdm.write(
                f"[{i + 1}/{len(questions)}] predicted={letter!r} finish={choice.finish_reason} "
                f"completion_tokens={completion_tokens}{flag} raw={generated.strip()[:40]!r}"
            )

    if still_truncated:
        print(f"WARNING ({name}): {len(still_truncated)} questions still truncated even at {MAX_TOKENS_RETRY} tokens: {still_truncated}")
    return predictions


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N questions (for a quick check)")
    args = parser.parse_args()

    questions = _load_questions()
    if args.limit:
        questions = questions[: args.limit]
    client = OpenAI(base_url=args.base_url, api_key="EMPTY")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for suffix, params in [("backbone", GREEDY_PARAMS), ("backbone-agent-sampling", AGENT_PARAMS)]:
        predictions = _run_config(client, args.model, questions, suffix, params)
        path = RESULTS_DIR / f"{suffix}.json"
        path.write_text(json.dumps(predictions, indent=2))
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
