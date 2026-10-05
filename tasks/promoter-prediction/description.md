## Task

For each DNA sequence, determine whether it is a functional core promoter region (the segment
where RNA polymerase II initiates transcription) or not. Respond with 1 if it is, 0 if it is not.
Your predictions will be scored against the true labels using Matthews correlation coefficient
(MCC) across all query sequences.

## What you have

- `data/train.csv` -- 4,904 labeled example DNA sequences (`sequence_id`, `sequence`, `label`),
  useful for training or calibrating your approach.
- `data/query_sequences.csv` -- 613 DNA sequences (`sequence_id`, `sequence`) to classify. No
  label is attached to these -- that is what you need to determine.
- One or more pretrained models may be available in your environment under `/models/<name>` --
  each has its own documentation describing what it is and how to use it. It's up to you to decide
  whether any of them are useful for this task.

## Output

Write a CSV file to your workspace with exactly these columns: `sequence_id,label` -- one row per
query sequence (613 rows total, matching the rows in `data/query_sequences.csv`). Then call
`finish` with the path to this file.
