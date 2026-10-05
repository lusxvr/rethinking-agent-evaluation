## Protocol

Prompt each question 0-shot (no few-shot examples). Prefix every prompt with the model's own
recommended framing:

```
You are an expert in general astrophysics. Your task is to answer the following question:
```

followed by:

```
Question: {question}
A. {choice_a}
B. {choice_b}
C. {choice_c}
D. {choice_d}
Answer with a single letter (A, B, C, or D) only. If unsure, still guess one letter.
```

Generate greedily and short -- `max_new_tokens=50, do_sample=False` -- then parse the first
standalone `A`/`B`/`C`/`D` token in the generated text as the answer. If no such token appears,
still write one of `A`/`B`/`C`/`D` for that row (pick a fixed default) rather than leaving it
blank or writing any other placeholder -- any value outside those four invalidates the entire
submission, not just that row.

152 sequential single-prompt `generate()` calls is slow enough to risk the shorter budgets --
batch multiple questions per call (pad the tokenizer, e.g. `tokenizer.pad_token =
tokenizer.eos_token`, `padding=True`) instead of one call per question.
