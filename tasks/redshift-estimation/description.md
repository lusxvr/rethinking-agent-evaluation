## Task

For each galaxy image below, estimate its cosmic redshift. Respond with a single decimal number
per galaxy. Your predictions will be scored against the true redshift using R² (coefficient of
determination) across all query galaxies.

## What you have

- `data/query_images/<targetid>.npy` -- 20 galaxy imaging cutouts, one file per galaxy, each a
  `(152, 152, 3)` float32 array (three photometric bands). No redshift is attached to these --
  that is what you need to estimate.
- `data/reference_catalog.npz` -- a supporting reference catalog of 126,907 other galaxies with
  known redshifts, useful for grounding your estimate.
- One or more pretrained models may be available in your environment under `/models/<name>` --
  each has its own documentation describing what it is and how to use it. It's up to you to decide
  whether any of them are useful for this task.

## Output

Write a CSV file to your workspace with exactly these columns: `targetid,redshift` -- one row per
query galaxy (20 rows total, matching the files in `data/query_images/`). Then call `finish` with
the path to this file.
