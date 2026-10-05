"""Sanity check that agent/weights load and produce coherent output. Runs in dev_env, not agent/env.

Usage:
    cd models/astrosage/dev_env && uv run --project . python ../verify.py
"""

import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

WEIGHTS_DIR = Path(__file__).resolve().parent / "agent" / "weights"

SAMPLE_QUESTIONS = [
    "What is the Chandrasekhar limit and why is it significant for the fate of white dwarf stars?",
    "Briefly explain the difference between Type Ia and Type II supernovae.",
]

# No tokenizer.chat_template is shipped -- the model card's documented usage is a plain
# prompt fed to the tokenizer directly, not apply_chat_template.
PROMPT_TEMPLATE = "You are an expert in general astrophysics. Your task is to answer the following question:\n{question}\n"


def main() -> int:
    if not WEIGHTS_DIR.is_dir() or not any(WEIGHTS_DIR.iterdir()):
        print(f"error: no weights found in {WEIGHTS_DIR} -- run ./download.sh first")
        return 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading AstroSage-8B from {WEIGHTS_DIR} onto {device}...")
    tokenizer = AutoTokenizer.from_pretrained(WEIGHTS_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        WEIGHTS_DIR,
        dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()

    for question in SAMPLE_QUESTIONS:
        prompt = PROMPT_TEMPLATE.format(question=question)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        start = time.time()
        with torch.no_grad():
            output = model.generate(
                **inputs, max_new_tokens=200, do_sample=False, pad_token_id=tokenizer.eos_token_id
            )
        elapsed = time.time() - start

        generated = tokenizer.decode(
            output[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True
        )
        print(f"\n=== Q: {question}\n--- A ({elapsed:.1f}s): {generated.strip()}")

        if len(generated.strip()) < 10:
            print("error: output looks empty/degenerate")
            return 1

    print("\nOK: model loaded and produced coherent-looking output for all sample questions.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
