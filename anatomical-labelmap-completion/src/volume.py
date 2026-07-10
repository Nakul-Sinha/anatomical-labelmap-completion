"""
Volume reconstruction + extended multi-slice context.

Slabs are overlapping sliding windows, so rows chain into source volumes (row i's next-visible
slice == row j's center-visible slice). This lets us assemble, for any row, a K-slice visible
context centered on its center slice (offset 0 = center, +1 = its next, -1 = its prev, +2 = the
next row's next, ...), reconstructed by walking the chain. Target structures are 3D-continuous,
so a wider slice context should predict the center-slice targets better than the given 3 slices.

Everything uses only the provided visible maps (train or test) reorganised -- no labels leak.
"""
import numpy as np


def _key(a):
    return a.astype(np.int16).tobytes()


def build_chain(V):
    """next[i], prev[i] = adjacent rows in the same volume (or -1)."""
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


def extended_context(V, K):
    """Return (N, K, 32, 32) visible context, K odd, centered on each row's center slice.
    Offsets beyond the chain ends are zero-padded (background)."""
    assert K % 2 == 1
    r = (K - 1) // 2
    n = len(V)
    nxt, prv = build_chain(V)
    out = np.zeros((n, K, 32, 32), dtype=V.dtype)
    for i in range(n):
        out[i, r] = V[i, 1]                       # offset 0 = center
        # forward offsets +1..+r
        cur = i
        for d in range(1, r + 1):
            if d == 1:
                out[i, r + d] = V[i, 2]
                cur = nxt[i]
            else:
                if cur < 0:
                    break
                out[i, r + d] = V[cur, 2]
                cur = nxt[cur]
        # backward offsets -1..-r
        cur = i
        for d in range(1, r + 1):
            if d == 1:
                out[i, r - d] = V[i, 0]
                cur = prv[i]
            else:
                if cur < 0:
                    break
                out[i, r - d] = V[cur, 0]
                cur = prv[cur]
    return out, nxt, prv


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import common as C
    tr = C.load_split("train")
    for K in (3, 7, 11):
        ec, nxt, prv = extended_context(tr["V"], K)
        # fraction of non-padded outer slices (chain coverage)
        outer = ec[:, 0]
        cov = (outer.reshape(len(ec), -1) != 0).any(1).mean()
        print(f"K={K}: context {ec.shape}, outermost-slice non-empty fraction {cov:.2f}")
    nxt, prv = build_chain(tr["V"])
    print("rows with a next neighbour:", sum(1 for x in nxt if x >= 0), "/", len(tr["V"]))
