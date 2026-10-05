## Task

For each RNA sequence, predict its secondary structure: which bases pair with which. Represent
the structure as a string the same length as the sequence, using extended dot-bracket notation:

- `.` for a base that isn't paired with any other base.
- One of four bracket-pair types -- `()`, `[]`, `{}`, `<>` -- for a paired base. A base pair is
  written as two matching characters of the *same* type at the two paired positions (the opening
  character at the lower-index position, the closing character at the higher-index one). Most
  pairs can use `()`. A different type is only needed when two pairs "cross" each other (a
  pseudoknot) -- e.g. positions 2 and 8 pair, and positions 4 and 10 also pair, so a plain single
  stack can't represent both correctly at once.

Example: the sequence `GGGAAACCC`, if positions 1-3 pair with 9-7 (in reverse) and positions 4-6
are unpaired, would be written `(((...)))`.

Your predictions will be scored against the true structures using a relaxed, per-sequence F1
score (a predicted pair one position off from the true one still partially counts), averaged
across all query sequences.

## What you have

- `data/train.csv` -- 300 labeled example sequences (`sequence_id`, `sequence`, `structure`), for
  checking your own approach against known-correct structures. This task doesn't require training
  anything -- these are for verifying your pipeline produces sensible output, not for fitting a
  model.
- `data/query_sequences.csv` -- 300 RNA sequences (`sequence_id`, `sequence`) to predict
  structures for. No structure is attached to these -- that is what you need to determine.
- One or more pretrained models may be available in your environment under `/models/<name>` --
  each has its own documentation describing what it is and how to use it. It's up to you to decide
  whether any of them are useful for this task.

## Output

Write a CSV file to your workspace with exactly these columns: `sequence_id,structure` -- one row
per query sequence (300 rows total, matching the rows in `data/query_sequences.csv`), each
`structure` value exactly as long as its sequence. Then call `finish` with the path to this file.
