"""BACKBONE anchor: one chat completion per test sequence with the task's question, no tools.
Writes dev/results/backbone_prediction_{no_thinking,thinking}.csv. Needs a running vLLM server.

Usage:
    cd models/dnabert-2/dev_env
    uv run --project . python ../../../solutions/promoter-prediction/dev/run_backbone.py --base-url http://localhost:PORT/v1 [--thinking]
"""

import argparse
import csv
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
from axes import DEFAULT_MODEL_NAME, DEFAULT_SAMPLING  # noqa: E402
from eval.metrics import _mcc  # noqa: E402

RESULTS_DIR = REPO_ROOT / "solutions" / "promoter-prediction" / "dev" / "results"
GUE_CONFIG = "prom_core_tata"

# The task's own question, so the anchor sees what the agent sees.
QUESTION_TEMPLATE = (
    "Below is a 70-nucleotide human DNA sequence. Determine whether it is a functional core "
    "promoter region (the segment where RNA polymerase II initiates transcription) or not.\n\n"
    "Sequence: {sequence}\n\n"
    "Respond with exactly one character: 1 if it is a promoter, 0 if it is not."
)

MAX_TOKENS_NO_THINKING = 16  # greedy, no thinking -- a one-character answer needs no more
# Retry with a larger budget instead of giving every call one.
MAX_TOKENS_THINKING_INITIAL = 2048
MAX_TOKENS_THINKING_RETRY = 8192

# Greedy in both modes, so only the reasoning channel differs.
GREEDY_PARAMS = {**DEFAULT_SAMPLING, "temperature": 0.0, "top_p": 1.0, "top_k": -1, "presence_penalty": 0.0}


def parse_label(text: str) -> int | None:
    match = re.search(r"[01]", text)
    return int(match.group()) if match else None


def _ask(client: OpenAI, model: str, sequence: str, max_tokens: int, thinking: bool):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": QUESTION_TEMPLATE.format(sequence=sequence)}],
        max_tokens=max_tokens,
        temperature=GREEDY_PARAMS["temperature"],
        top_p=GREEDY_PARAMS["top_p"],
        presence_penalty=GREEDY_PARAMS["presence_penalty"],
        extra_body={
            "top_k": GREEDY_PARAMS["top_k"],
            "min_p": GREEDY_PARAMS["min_p"],
            "repetition_penalty": GREEDY_PARAMS["repetition_penalty"],
            "chat_template_kwargs": {"enable_thinking": thinking},
        },
    )
    return response.choices[0]


def _process_one(client: OpenAI, model: str, idx: int, sequence: str, thinking: bool) -> tuple[int, int | None, bool]:
    if thinking:
        choice = _ask(client, model, sequence, MAX_TOKENS_THINKING_INITIAL, thinking=True)
        if choice.finish_reason == "length":
            choice = _ask(client, model, sequence, MAX_TOKENS_THINKING_RETRY, thinking=True)
    else:
        choice = _ask(client, model, sequence, MAX_TOKENS_NO_THINKING, thinking=False)
    truncated = choice.finish_reason == "length"
    label = parse_label(choice.message.content or "")
    return idx, label, truncated


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N test examples (for a quick check)")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--thinking", action="store_true", help="Enable the reasoning channel (default: off)")
    args = parser.parse_args()

    ds = load_dataset("leannmlindsey/GUE", GUE_CONFIG)["test"]
    if args.limit:
        ds = ds.select(range(args.limit))
    client = OpenAI(base_url=args.base_url, api_key="EMPTY", timeout=3600.0 if args.thinking else 600.0)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "thinking" if args.thinking else "no_thinking"
    output_path = RESULTS_DIR / f"backbone_prediction_{suffix}.csv"

    predictions: dict[int, int] = {}
    unparsed = []
    still_truncated = []
    failed = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {
            pool.submit(_process_one, client, args.model, i, row["sequence"], args.thinking): i
            for i, row in enumerate(ds)
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="backbone"):
            idx = futures[future]
            try:
                idx, label, truncated = future.result()
            except Exception as exc:
                print(f"WARNING: example {idx} failed ({exc!r}), skipping", file=sys.stderr)
                failed.append(idx)
                continue
            if label is None:
                unparsed.append(idx)
                label = 0  # unparseable counts as label 0, not a dropped row
            predictions[idx] = label
            if truncated:
                still_truncated.append(idx)

    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sequence_id", "label"])
        for i in range(len(ds)):
            if i in predictions:
                writer.writerow([f"query_{i:04d}", predictions[i]])

    true = [ds[i]["label"] for i in sorted(predictions)]
    pred = [predictions[i] for i in sorted(predictions)]
    import pandas as pd

    mcc = _mcc(pd.Series(true), pd.Series(pred))

    if unparsed:
        print(f"WARNING: {len(unparsed)} responses had no parseable 0/1: {unparsed[:20]}{'...' if len(unparsed) > 20 else ''}")
    if still_truncated:
        budget = MAX_TOKENS_THINKING_RETRY if args.thinking else MAX_TOKENS_NO_THINKING
        print(f"WARNING: {len(still_truncated)} still truncated at {budget} tokens: {still_truncated}")
    if failed:
        print(f"WARNING: {len(failed)} requests failed and were skipped: {failed}")
    print(f"Wrote {output_path}")
    print(f"[summary] n={len(predictions)} backbone_mcc={mcc:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
