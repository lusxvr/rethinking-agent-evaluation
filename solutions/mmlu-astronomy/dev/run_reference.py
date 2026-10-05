"""REFERENCE anchor: AstroSage-8B on mmlu-astronomy (0-shot with ASTROSAGE_PREFIX), writing
dev/results/astrosage-8b.json. Uses the agent's weights but the host dev_env.

Usage:
    cd models/astrosage/dev_env
    uv run --project . python ../../../solutions/mmlu-astronomy/dev/run_reference.py
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from mcq import ASTROSAGE_PREFIX, build_prompt, parse_letter
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[3]
WEIGHTS_DIR = REPO_ROOT / "models" / "astrosage" / "agent" / "weights"
DATA_DIR = REPO_ROOT / "tasks" / "mmlu-astronomy" / "data"
RESULTS_PATH = REPO_ROOT / "solutions" / "mmlu-astronomy" / "dev" / "results" / "astrosage-8b.json"


def _load_questions() -> list[dict]:
    with (DATA_DIR / "questions.csv").open(newline="") as f:
        return list(csv.DictReader(f))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N questions (for a quick check)")
    args = parser.parse_args()

    questions = _load_questions()
    if args.limit:
        questions = questions[: args.limit]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(WEIGHTS_DIR)
    model = AutoModelForCausalLM.from_pretrained(WEIGHTS_DIR, dtype=torch.bfloat16, device_map=device)
    model.eval()

    predictions = {}
    for row in tqdm(questions, desc="astrosage-8b"):
        choices = [row["choice_a"], row["choice_b"], row["choice_c"], row["choice_d"]]
        prompt = ASTROSAGE_PREFIX + build_prompt(row["question"], choices)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=50, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        generated = tokenizer.decode(output[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
        predictions[row["question_id"]] = parse_letter(generated) or "?"

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(predictions, indent=2))
    print(f"Wrote {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
