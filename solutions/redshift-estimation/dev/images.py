"""Image extraction shared by this solution's dev scripts (prepare_data.py, build_final_query_set.py)."""

import numpy as np
import pyarrow as pa

IMAGE_SHAPE = (152, 152, 3)


def extract_images(table: pa.Table, indices: np.ndarray, progress=None) -> np.ndarray:
    """Images at sorted `indices`, extracted per ~1,000-row chunk with vectorized pyarrow (a row loop
    or one whole-table take() is too slow or overflows). `progress` may wrap the chunk loop.
    """
    image_col = table.column("image")
    chunk_lengths = np.array([len(chunk) for chunk in image_col.chunks])
    chunk_starts = np.concatenate([[0], np.cumsum(chunk_lengths)])
    chunk_id = np.searchsorted(chunk_starts, indices, side="right") - 1
    local_idx = indices - chunk_starts[chunk_id]

    images = np.empty((len(indices), *IMAGE_SHAPE), dtype=np.float32)
    chunk_ids = np.unique(chunk_id)
    for cid in (progress(chunk_ids) if progress else chunk_ids):
        mask = chunk_id == cid
        taken = image_col.chunk(int(cid)).take(pa.array(local_idx[mask]))
        for _ in range(3):  # list<list<list<float>>> -> flat float array, no Python objects
            taken = taken.flatten()
        images[mask] = taken.to_numpy(zero_copy_only=False).reshape(mask.sum(), *IMAGE_SHAPE)
    return images
