"""Shared 0-shot MCQ prompting and parsing for every run_*.py here, so their scores are comparable."""

import re

LETTERS = ["A", "B", "C", "D"]

# From AstroSage-8B's model card; used by run_reference.py and run_base_llama.py, not run_backbone.py.
ASTROSAGE_PREFIX = "You are an expert in general astrophysics. Your task is to answer the following question:\n"


def build_prompt(question: str, choices: list[str]) -> str:
    options = "\n".join(f"{letter}. {choice}" for letter, choice in zip(LETTERS, choices))
    return (
        f"Question: {question}\n{options}\n"
        "Answer with a single letter (A, B, C, or D) only. If unsure, still guess one letter."
    )


def parse_letter(text: str) -> str | None:
    match = re.search(r"\b([ABCD])\b", text.strip().upper())
    return match.group(1) if match else None
