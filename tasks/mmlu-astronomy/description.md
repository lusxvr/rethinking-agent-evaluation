## Task

For each of these astronomy multiple-choice questions, choose the correct answer. Respond with a
single letter: A, B, C, or D. Your predictions will be scored against the true answers using
accuracy across all questions.

## What you have

- `data/questions.csv` -- astronomy multiple-choice questions, one per row: `question_id`,
  `question`, `choice_a`, `choice_b`, `choice_c`, `choice_d`. No answer is attached to these --
  that is what you need to determine.
- One or more pretrained models may be available in your environment under `/models/<name>` --
  each has its own documentation describing what it is and how to use it. It's up to you to decide
  whether any of them are useful for this task.

## Output

Write a CSV file to your workspace with exactly these columns: `question_id,answer` -- one row per
question, matching the rows in `data/questions.csv`. Then call `finish` with the path to this file.
