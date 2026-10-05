"""Llama-3.1-8B-Instruct (AstroSage's base model) on mmlu-astronomy, with run_reference.py's prompt
(base-llama.json) and with Llama's chat template (base-llama-chat.json). Gated: needs HF_TOKEN.

Usage:
    cd models/astrosage/dev_env
    uv run --project . python ../../../solutions/mmlu-astronomy/dev/run_base_llama.py
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import torch
from dotenv import load_dotenv
from mcq import ASTROSAGE_PREFIX, build_prompt, parse_letter
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "tasks" / "mmlu-astronomy" / "data"
RESULTS_DIR = REPO_ROOT / "solutions" / "mmlu-astronomy" / "dev" / "results"
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"


def _load_questions() -> list[dict]:
    with (DATA_DIR / "questions.csv").open(newline="") as f:
        return list(csv.DictReader(f))


def _generate(model, tokenizer, encoded) -> str:
    """encoded: a BatchEncoding with input_ids (+ attention_mask) -- both tokenizer(...) and
    apply_chat_template(..., return_tensors="pt") return this, not a bare tensor."""
    with torch.no_grad():
        output = model.generate(**encoded, max_new_tokens=50, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(output[0][encoded["input_ids"].shape[-1] :], skip_special_tokens=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N questions (for a quick check)")
    args = parser.parse_args()

    if not load_dotenv(REPO_ROOT / ".env"):
        raise SystemExit(f"{REPO_ROOT / '.env'} not found -- run 'cp .env.example .env' and set CACHE_ROOT")
    os.environ.setdefault("HF_HOME", f"{os.environ['CACHE_ROOT']}/huggingface")

    if "HF_TOKEN" not in os.environ:
        raise SystemExit(
            f"HF_TOKEN not set in {REPO_ROOT / '.env'} -- meta-llama/Llama-3.1-8B-Instruct is "
            "gated, accept its license on huggingface.co and set a token with access"
        )

    questions = _load_questions()
    if args.limit:
        questions = questions[: args.limit]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map=device)
    model.eval()

    identical, chat = {}, {}
    for row in tqdm(questions, desc="base-llama"):
        choices = [row["choice_a"], row["choice_b"], row["choice_c"], row["choice_d"]]
        prompt = build_prompt(row["question"], choices)

        inputs = tokenizer(ASTROSAGE_PREFIX + prompt, return_tensors="pt").to(model.device)
        identical[row["question_id"]] = parse_letter(_generate(model, tokenizer, inputs)) or "?"

        chat_inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)
        chat[row["question_id"]] = parse_letter(_generate(model, tokenizer, chat_inputs)) or "?"

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for suffix, predictions in [("base-llama", identical), ("base-llama-chat", chat)]:
        path = RESULTS_DIR / f"{suffix}.json"
        path.write_text(json.dumps(predictions, indent=2))
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
