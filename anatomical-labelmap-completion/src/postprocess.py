"""
Volume-chain reconstruction + 3D consistency smoothing.

Slabs are overlapping sliding windows, so within a source volume the rows form a chain
(row i's next-visible slice == row j's center-visible slice). Target structures are 3D-continuous,
so averaging a row's class probabilities with its chain neighbours (same (x,y) grid across adjacent
slices) reinforces true structures and suppresses per-slice noise. Fold/leak-safe: chain neighbours
share the same source volume, hence the same train/val (or test) membership.
"""
import numpy as np


def _key(a):
    return a.astype(np.int16).tobytes()


def build_chain(V):
    """Return next[], prev[] arrays (row index or -1) from the sliding-window overlap."""
    n = len(V)
    cen2row = {}
    for i in range(n):
        cen2row.setdefault(_key(V[i, 1]), []).append(i)
    nxt = [-1] * n
    prv = [-1] * n
    for i in range(n):
        for j in cen2row.get(_key(V[i, 2]), []):
            if _key(V[i, 1]) == _key(V[j, 0]):
                nxt[i] = j
                break
        for j in cen2row.get(_key(V[i, 0]), []):
            if _key(V[i, 1]) == _key(V[j, 2]):
                prv[i] = j
                break
    return nxt, prv


def smooth3d(proba, V, alpha=0.5, width=2):
    """Average each row's (C,H,W) proba with up to `width` chain neighbours each side, weight alpha^step."""
    n = len(proba)
    nxt, prv = build_chain(V)
    out = np.empty_like(proba)
    for i in range(n):
        acc = proba[i].astype(np.float64).copy()
        w = 1.0
        cur, wt = i, 1.0
        for _ in range(width):
            cur = nxt[cur]
            if cur < 0:
                break
            wt *= alpha
            acc += wt * proba[cur]
            w += wt
        cur, wt = i, 1.0
        for _ in range(width):
            cur = prv[cur]
            if cur < 0:
                break
            wt *= alpha
            acc += wt * proba[cur]
            w += wt
        out[i] = (acc / w).astype(np.float32)
    out /= (out.sum(1, keepdims=True) + 1e-9)
    return out
