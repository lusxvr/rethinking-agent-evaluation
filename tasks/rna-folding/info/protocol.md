## Protocol

Run every sequence in `data/query_sequences.csv` through the model (see interface.md for loading)
to get its base-pair probability matrix, then turn that into a valid structure with the following
post-processing, in order:

**1. Mask out disallowed pairs** before thresholding -- a real base pair can't be closer than 4
positions apart in the sequence (it would create a physically impossible sharp loop), and can only
occur between Watson-Crick or wobble pairs (`AU`, `UA`, `GC`, `CG`, `GU`, `UG`):

```python
import numpy as np

def allowed_pairs_mask(seq: str) -> np.ndarray:
    seq = seq.upper().replace("T", "U")
    seq_len = len(seq)

    mask = np.logical_not(np.eye(seq_len, dtype=bool))  # no self-pairing
    sharp_loop = np.eye(seq_len, k=0, dtype=bool)
    for k in range(1, 4):
        sharp_loop |= np.eye(seq_len, k=k, dtype=bool) | np.eye(seq_len, k=-k, dtype=bool)
    mask &= ~sharp_loop

    canonical = {"AU", "UA", "GC", "CG", "GU", "UG"}
    canonical_mask = np.zeros((seq_len, seq_len), dtype=bool)
    for i, a in enumerate(seq):
        for j, b in enumerate(seq):
            if f"{a}{b}" in canonical:
                canonical_mask[i, j] = True
    mask &= canonical_mask

    return mask
```

**2. Threshold** the masked probabilities at the checkpoint's own tuned `threshold` (not 0.5):

```python
probs = probs[0].numpy()
probs[~allowed_pairs_mask(sequence)] = 0.0
sec_struct = (probs > threshold).astype(int)
```

**3. Resolve conflicts** so each base ends with at most one partner: repeatedly commit the
highest-probability remaining pair, then forbid every other pairing for both of its bases, until
none remain:

```python
def clean_sec_struct(sec_struct: np.ndarray, probs: np.ndarray) -> np.ndarray:
    clean = np.copy(sec_struct)
    tmp = np.copy(probs)
    tmp[sec_struct < 1] = 0.0
    while np.sum(tmp > 0.0) > 0:
        i, j = np.unravel_index(np.argmax(tmp, axis=None), tmp.shape)
        tmp[i, :] = tmp[j, :] = 0.0
        clean[i, :] = clean[j, :] = 0
        tmp[:, i] = tmp[:, j] = 0.0
        clean[:, i] = clean[:, j] = 0
        clean[i, j] = clean[j, i] = 1
    return clean
```

**4. Encode as extended dot-bracket notation.** The result of step 3 is a symmetric matrix where
`sec_struct[i, j] == 1` means positions `i` and `j` are paired. Assign each pair to one of the
four bracket types, greedily: use the first type that doesn't *cross* any pair already assigned to
it (two pairs `(i, j)` and `(a, b)` with `i < a` cross when `i < a < j < b`) -- pairs that are
nested or don't overlap at all can freely share a type; only genuinely crossing (pseudoknotted)
pairs need a different one:

```python
def matrix_to_dot_bracket(sec_struct: np.ndarray) -> str:
    seq_len = sec_struct.shape[0]
    pairs = [(i, j) for i in range(seq_len) for j in range(i + 1, seq_len) if sec_struct[i, j] > 0]
    bracket_types = [("(", ")"), ("[", "]"), ("{", "}"), ("<", ">")]

    def crosses(i1, j1, i2, j2):
        return (i1 < i2 < j1 < j2) or (i2 < i1 < j2 < j1)

    assigned = [[] for _ in bracket_types]
    for i, j in pairs:
        for bucket in assigned:
            if all(not crosses(i, j, a, b) for a, b in bucket):
                bucket.append((i, j))
                break
        # a pair that fits none of the 4 types is dropped -- rare in practice

    out = ["."] * seq_len
    for (open_c, close_c), bucket in zip(bracket_types, assigned):
        for i, j in bucket:
            out[i], out[j] = open_c, close_c
    return "".join(out)
```

Write `sequence_id,structure` for all 300 query sequences and call `finish`.
