"""BACKBONE anchor: one chat completion per test sequence asking for its dot-bracket structure.
Writes dev/results/backbone_prediction_{no_thinking,thinking}.csv. Needs a running vLLM server.

Usage:
    cd models/rinalmo/dev_env
    uv run --project . python ../../../solutions/rna-folding/dev/run_backbone.py --base-url http://localhost:PORT/v1 [--thinking]
"""

import argparse
import csv
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
from axes import AXIS_DEFAULTS, MODEL_LEVELS  # noqa: E402
from eval.metrics import _sec_struct_f1  # noqa: E402

QUERY_PATH = REPO_ROOT / "tasks" / "rna-folding" / "data" / "query_sequences.csv"
SOLUTION_PATH = REPO_ROOT / "solutions" / "rna-folding" / "solution.csv"
RESULTS_DIR = REPO_ROOT / "solutions" / "rna-folding" / "dev" / "results"

# The task's own question, so the anchor sees what the agent sees.
QUESTION_TEMPLATE = (
    "Below is an RNA sequence. Predict its secondary structure using extended dot-bracket "
    "notation: '.' for an unpaired base, and one of the four bracket-pair types (), [], {{}}, <> "
    "for base-paired positions (a base pair uses two matching characters of the SAME type; use a "
    "different type only for a pseudoknotted/crossing base pair). Your answer must be exactly "
    "{length} characters long, the same length as the sequence, with no other text.\n\n"
    "Sequence: {sequence}\n\n"
    "Structure:"
)

# Budget scales with sequence length (up to ~500nt here) rather than a fixed constant -- a
# character-for-character structure string needs roughly one token per position at minimum.
MAX_TOKENS_NO_THINKING_PER_CHAR = 3
MAX_TOKENS_NO_THINKING_FLOOR = 64
MAX_TOKENS_THINKING_INITIAL = 4096
MAX_TOKENS_THINKING_RETRY = 16384

# The model's sampling preset, not greedy: greedy decoding degenerates into repeated tokens on
# this long output.
_, THINKING_PARAMS, NON_THINKING_PARAMS = MODEL_LEVELS[AXIS_DEFAULTS["model"]]


def _ask(client: OpenAI, model: str, sequence: str, max_tokens: int, thinking: bool):
    params = THINKING_PARAMS if thinking else NON_THINKING_PARAMS
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": QUESTION_TEMPLATE.format(sequence=sequence, length=len(sequence))}],
        max_tokens=max_tokens,
        temperature=params["temperature"],
        top_p=params["top_p"],
        presence_penalty=params["presence_penalty"],
        extra_body={
            "top_k": params["top_k"],
            "min_p": params["min_p"],
            "repetition_penalty": params["repetition_penalty"],
            "chat_template_kwargs": {"enable_thinking": thinking},
        },
    )
    return response.choices[0]


def _process_one(client: OpenAI, model: str, sequence_id: str, sequence: str, thinking: bool) -> tuple[str, str, bool]:
    if thinking:
        choice = _ask(client, model, sequence, MAX_TOKENS_THINKING_INITIAL, thinking=True)
        if choice.finish_reason == "length":
            choice = _ask(client, model, sequence, MAX_TOKENS_THINKING_RETRY, thinking=True)
    else:
        max_tokens = max(MAX_TOKENS_NO_THINKING_FLOOR, len(sequence) * MAX_TOKENS_NO_THINKING_PER_CHAR)
        choice = _ask(client, model, sequence, max_tokens, thinking=False)
    truncated = choice.finish_reason == "length"
    structure = (choice.message.content or "").strip()
    return sequence_id, structure, truncated


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default=MODEL_LEVELS[AXIS_DEFAULTS["model"]][0])
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N query examples (for a quick check)")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--thinking", action="store_true", help="Enable the reasoning channel (default: off)")
    args = parser.parse_args()

    queries = pd.read_csv(QUERY_PATH, dtype={"sequence_id": str})
    if args.limit:
        queries = queries.head(args.limit)
    client = OpenAI(base_url=args.base_url, api_key="EMPTY", timeout=3600.0 if args.thinking else 600.0)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "thinking" if args.thinking else "no_thinking"
    output_path = RESULTS_DIR / f"backbone_prediction_{suffix}.csv"

    predictions: dict[str, str] = {}
    still_truncated = []
    failed = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {
            pool.submit(_process_one, client, args.model, row["sequence_id"], row["sequence"], args.thinking): row["sequence_id"]
            for _, row in queries.iterrows()
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="backbone"):
            sequence_id = futures[future]
            try:
                sequence_id, structure, truncated = future.result()
            except Exception as exc:
                print(f"WARNING: {sequence_id} failed ({exc!r}), skipping", file=sys.stderr)
                failed.append(sequence_id)
                continue
            predictions[sequence_id] = structure
            if truncated:
                still_truncated.append(sequence_id)

    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sequence_id", "structure"])
        for _, row in queries.iterrows():
            if row["sequence_id"] in predictions:
                writer.writerow([row["sequence_id"], predictions[row["sequence_id"]]])

    solution = pd.read_csv(SOLUTION_PATH, dtype={"sequence_id": str})
    submission = pd.DataFrame(predictions.items(), columns=["sequence_id", "structure"])
    merged = solution.merge(submission, on="sequence_id", suffixes=("_true", "_pred"))
    f1 = _sec_struct_f1(merged["structure_true"], merged["structure_pred"])

    if still_truncated:
        budget = MAX_TOKENS_THINKING_RETRY if args.thinking else "length-scaled"
        print(f"WARNING: {len(still_truncated)} still truncated at {budget} tokens: {still_truncated[:20]}")
    if failed:
        print(f"WARNING: {len(failed)} requests failed and were skipped: {failed}")
    print(f"Wrote {output_path}")
    print(f"[summary] n={len(merged)} backbone_sec_struct_f1={f1:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
