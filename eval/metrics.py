"""Metric registry for eval/evaluate.py: name -> callable(true, pred) -> float."""

import pandas as pd


def _accuracy(true: pd.Series, pred: pd.Series) -> float:
    return float((true == pred).mean())


def _r2(true: pd.Series, pred: pd.Series) -> float:
    residual = pred - true
    total = true - true.mean()
    return float(1 - (residual**2).sum() / (total**2).sum())


def _mcc(true: pd.Series, pred: pd.Series) -> float:
    """Binary MCC, matching sklearn.metrics.matthews_corrcoef (0.0 on a zero denominator)."""
    tp = int(((true == 1) & (pred == 1)).sum())
    tn = int(((true == 0) & (pred == 0)).sum())
    fp = int(((true == 0) & (pred == 1)).sum())
    fn = int(((true == 1) & (pred == 0)).sum())
    denom = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    if denom == 0:
        return 0.0
    return float((tp * tn - fp * fn) / denom)


# Extended dot-bracket brackets (pseudoknots). Ported from RiNALMo's rinalmo/utils/sec_struct.py
# (Apache 2.0): dot_bracket_to_2d_mat, _relax_ss, ss_precision/ss_recall/ss_f1.
_BRACKET_OPEN_TO_CLOSE = {"(": ")", "[": "]", "{": "}", "<": ">"}
_BRACKET_CLOSE_TO_OPEN = {v: k for k, v in _BRACKET_OPEN_TO_CLOSE.items()}


def _dot_bracket_to_pairs(db: str) -> set[frozenset] | None:
    """Base pairs as frozenset({i, j}) from extended dot-bracket notation; None if malformed."""
    stacks: dict[str, list[int]] = {c: [] for c in _BRACKET_OPEN_TO_CLOSE}
    pairs: set[frozenset] = set()
    for i, ch in enumerate(db):
        if ch == ".":
            continue
        elif ch in _BRACKET_OPEN_TO_CLOSE:
            stacks[ch].append(i)
        elif ch in _BRACKET_CLOSE_TO_OPEN:
            open_stack = stacks[_BRACKET_CLOSE_TO_OPEN[ch]]
            if not open_stack:
                return None
            pairs.add(frozenset((i, open_stack.pop())))
        else:
            return None
    if any(stacks.values()):
        return None
    return pairs


def _has_pair(pairs: set[frozenset], a: int, b: int) -> bool:
    return a != b and frozenset((a, b)) in pairs


def _dilated_contains(pairs: set[frozenset], a: int, b: int) -> bool:
    """Whether (a, b) or a 4-neighbor is paired; the pair-set form of _relax_ss's dilation."""
    return (
        _has_pair(pairs, a, b)
        or _has_pair(pairs, a - 1, b)
        or _has_pair(pairs, a + 1, b)
        or _has_pair(pairs, a, b - 1)
        or _has_pair(pairs, a, b + 1)
    )


_F1_EPSILON = 1e-5  # matches RiNALMo's own ss_f1 zero-denominator guard


def _relaxed_ss_f1(target_pairs: set[frozenset], pred_pairs: set[frozenset]) -> float:
    if pred_pairs:
        precision = sum(_dilated_contains(target_pairs, *p) for p in pred_pairs) / len(pred_pairs)
    else:
        precision = 0.0
    if target_pairs:
        recall = sum(_dilated_contains(pred_pairs, *p) for p in target_pairs) / len(target_pairs)
    else:
        recall = 0.0
    if precision + recall < _F1_EPSILON:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _sec_struct_f1(true: pd.Series, pred: pd.Series) -> float:
    """Relaxed F1 per sequence (RiNALMo's bpRNA protocol), macro-averaged; malformed rows score 0."""
    scores = []
    for true_db, pred_db in zip(true, pred):
        true_db, pred_db = str(true_db), str(pred_db)
        target_pairs = _dot_bracket_to_pairs(true_db)
        pred_pairs = _dot_bracket_to_pairs(pred_db)
        if target_pairs is None or pred_pairs is None or len(true_db) != len(pred_db):
            scores.append(0.0)
            continue
        scores.append(_relaxed_ss_f1(target_pairs, pred_pairs))
    return float(sum(scores) / len(scores)) if scores else 0.0


METRICS = {
    "accuracy": _accuracy,
    "r2": _r2,
    "mcc": _mcc,
    "sec_struct_f1": _sec_struct_f1,
}
