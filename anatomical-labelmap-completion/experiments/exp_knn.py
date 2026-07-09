"""
k-NN retrieval / voting experiment.

Rationale (see docs/notes.md): copying the target map from the most-similar training row scores
CV ~0.69. Here we (a) design better similarity metrics that focus on anatomy rather than background,
(b) aggregate top-k neighbours by weighted voting, (c) enforce the hard center-zero constraint, and
(d) sweep hyper-parameters under fixed 5-fold CV. Also diagnoses train-CV vs test neighbour similarity.
"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import common as C

tr = C.load_split("train")
V, T, Tidx = tr["V"].astype(np.int16), tr["T"].astype(np.int64), tr["Tidx"].astype(np.int64)
N = len(V)
Vf = V.reshape(N, -1).astype(np.int16)          # (N, 3072) full slab
Cf = V[:, 1].reshape(N, -1).astype(np.int16)    # (N, 1024) center slice
Pf = V[:, 0].reshape(N, -1).astype(np.int16)
Nf = V[:, 2].reshape(N, -1).astype(np.int16)


def sim_matrix(Aq, At, kind):
    """Similarity of each query row (Aq) to each train row (At). Higher = more similar."""
    if kind == "center_agree":
        q, t = V[Aq, 1].reshape(len(Aq), -1), V[At, 1].reshape(len(At), -1)
        return (q[:, None, :] == t[None, :, :]).sum(2)
    raise ValueError(kind)


def sim_center_anat(qi, train_idx):
    """Anatomy-focused center similarity for one query vs many train rows.
    counts cells where both center-visible nonzero and equal (matching anatomy),
    minus a small penalty for mismatched nonzero cells."""
    q = Cf[qi]
    t = Cf[train_idx]                      # (m,1024)
    qnz = q != 0
    both_eq = (t == q) & qnz               # matching anatomy cells
    match = both_eq.sum(1)
    # penalise cells where one is anatomy and the other differs (shape disagreement)
    either_nz = (t != 0) | qnz
    disagree = ((t != q) & either_nz).sum(1)
    return match.astype(np.float64) - 0.25 * disagree.astype(np.float64)


def sim_full_anat(qi, train_idx, wc=2.0):
    out = np.zeros(len(train_idx), np.float64)
    for sl, w in ((0, 1.0), (1, wc), (2, 1.0)):
        q = V[qi, sl].reshape(-1)
        t = V[train_idx, sl].reshape(len(train_idx), -1)
        qnz = q != 0
        out += w * (((t == q) & qnz).sum(1) - 0.25 * (((t != q) & ((t != 0) | qnz)).sum(1)))
    return out


def vote(neigh_idx, weights, center_zero_mask, soft=True, thresh=0.5):
    """Aggregate neighbours' target maps into a single (32,32) opaque-label prediction."""
    acc = np.zeros((C.NUM_CLASSES, 1024), np.float64)
    for j, w in zip(neigh_idx, weights):
        ti = Tidx[j].reshape(-1)                     # 0..17
        acc[ti, np.arange(1024)] += w
    wsum = weights.sum() + 1e-9
    # background handling: argmax over classes; but bias toward target when target mass is high
    fg_mass = acc[1:].sum(0) / wsum                  # fraction of weight that is a target
    best_fg = acc[1:].argmax(0) + 1
    best_fg_frac = acc[1:].max(0) / wsum
    pred_idx = np.where(best_fg_frac >= thresh, best_fg, 0)
    # enforce hard constraint: targets only on center-zero cells
    pred_idx = np.where(center_zero_mask.reshape(-1), pred_idx, 0)
    return C.idx_to_labels(pred_idx).reshape(32, 32)


def run_cv(kind, k, thresh, wc=2.0):
    folds = C.make_folds(N, 5, seed=42)
    preds_all = np.zeros((N, 32, 32), np.int64)
    for tri, vai in folds:
        for qi in vai:
            if kind == "center_anat":
                s = sim_center_anat(qi, tri)
            elif kind == "full_anat":
                s = sim_full_anat(qi, tri, wc)
            elif kind == "center_agree":
                s = (Cf[tri] == Cf[qi]).sum(1).astype(np.float64)
            order = np.argsort(-s)[:k]
            nbr = tri[order]
            w = s[order] - s[order].min() + 1e-3
            czm = V[qi, 1] == 0
            preds_all[qi] = vote(nbr, w, czm, thresh=thresh)
    sc = C.score_rows([preds_all[i] for i in range(N)], [T[i] for i in range(N)])
    return sc, preds_all


if __name__ == "__main__":
    t0 = time.time()
    # quick baseline: 1-NN center anatomy copy
    print("Sweeping k-NN configs (5-fold CV)...")
    results = []
    for kind in ["center_anat", "full_anat", "center_agree"]:
        for k in [1, 3, 5, 7, 11, 15, 21]:
            for thresh in ([0.0] if k == 1 else [0.25, 0.35, 0.5, 0.6]):
                sc, _ = run_cv(kind, k, thresh)
                results.append((sc, kind, k, thresh))
                print(f"  {kind:12s} k={k:2d} thr={thresh:.2f} -> CV {sc:.4f}")
    results.sort(reverse=True)
    print("\nTOP 5:")
    for sc, kind, k, thr in results[:5]:
        print(f"  {sc:.4f}  {kind} k={k} thr={thr}")
    best = results[0]
    print(f"\nBEST: {best}")
    print(f"(elapsed {time.time()-t0:.0f}s)")
