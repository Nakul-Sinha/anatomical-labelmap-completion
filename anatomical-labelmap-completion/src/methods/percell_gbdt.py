"""
Per-cell gradient-boosted classifier -> soft per-cell class probabilities (ensemble format).

A DIFFERENT inductive bias from k-NN retrieval. k-NN matches whole center slices to
near-duplicate neighbours; that collapses across volumes (honest group-CV ~0.05) because a
volume-disjoint test set has no near-duplicate to copy. This method instead learns a LOCAL,
translation-invariant mapping from the visible-anatomy configuration around each hole to the
removed target label, so it can generalise to unseen volumes.

Each center-zero cell is one sample. Two stages:
  A) is-target        : binary HistGradientBoosting on every center-zero cell
  B) which label 1..17 : multiclass HistGradientBoosting on target cells only
combined per cell as  P(bg)=1-pA,  P(class c)=pA * P(c | target).

Features are translation-invariant local context (NO absolute position, which would just
memorise a volume):
  - visible non-zero counts in windows r=1,2,3,5 for prev/center/next slices
  - cross-slice hole structure: prev==0, next==0, both-zero, zero-counts in windows
  - distance-to-nearest-visible-anatomy (EDT) per slice; enclosure flag (zero region not
    connected to the image border -> a removed-anatomy hole, not open background)
  - per-label window counts (r=2,4) for the most frequent visible labels: WHICH anatomy
    borders the hole (the core signal for the removed label's identity)

HONEST cross-validation via common.make_group_folds (volume-disjoint). Random folds put
adjacent sliding-window slabs of the same volume in both train and val and inflate the score
massively. All model fits use fold-train cells only; the test model is fit on all train cells.

Memory-frugal: features are gathered straight into a (n_center_zero_cells, F) matrix (no
(N,32,32,F) tensor), because the machine is shared with other methods.
"""
import os, sys, time
import numpy as np
from scipy import ndimage
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
import common as C
from sklearn.ensemble import HistGradientBoostingClassifier as HGB

TOPK = 45  # number of most-frequent visible labels given dedicated window-count features


def _vocab(V, topk=TOPK):
    vals, cnts = np.unique(V, return_counts=True)
    nz = sorted([(int(v), int(c)) for v, c in zip(vals, cnts) if v != 0], key=lambda x: -x[1])
    return [v for v, _ in nz[:topk]]


def _box_counts(mask, r):
    """(N,H,W) numeric -> sum over (2r+1)x(2r+1) window (border-clipped), same shape."""
    N, H, W = mask.shape
    m = mask.astype(np.float32)
    ii = np.zeros((N, H + 1, W + 1), np.float32)
    np.cumsum(np.cumsum(m, axis=1), axis=2, out=ii[:, 1:, 1:])
    ar = np.arange(H); ac = np.arange(W)
    i0 = np.clip(ar - r, 0, H); i1 = np.clip(ar + r + 1, 0, H)
    j0 = np.clip(ac - r, 0, W); j1 = np.clip(ac + r + 1, 0, W)
    return ii[:, i1][:, :, j1] - ii[:, i0][:, :, j1] - ii[:, i1][:, :, j0] + ii[:, i0][:, :, j0]


def _dist_enc(V):
    """distance-to-anatomy (EDT) per slice (N,3,32,32) and center-slice enclosure (N,32,32)."""
    N = len(V)
    dt = np.zeros((N, 3, 32, 32), np.float32)
    enc = np.zeros((N, 32, 32), np.float32)
    for i in range(N):
        for s in range(3):
            dt[i, s] = ndimage.distance_transform_edt(V[i, s] == 0)
        z = (V[i, 1] == 0)
        lab, _ = ndimage.label(z)
        bl = set(lab[0, :]).union(lab[-1, :]).union(lab[:, 0]).union(lab[:, -1]); bl.discard(0)
        enc[i] = (z & ~np.isin(lab, list(bl))).astype(np.float32)
    return dt, enc


def _n_features(vocab):
    return 3 * 4 + 3 + 3 * 2 + 3 + 1 + len(vocab) * 3 * 2


def cell_matrix(V, vocab):
    """Build (n_center_zero_cells, F) feature matrix + (rows,ii,jj) indices. Fixed F layout."""
    prev, cen, nxt = V[:, 0], V[:, 1], V[:, 2]
    rows, ii, jj = np.where(cen == 0)
    n = len(rows)
    F = _n_features(vocab)
    X = np.empty((n, F), np.float32)
    dt, enc = _dist_enc(V)
    k = [0]

    def add(grid):
        X[:, k[0]] = grid[rows, ii, jj]
        k[0] += 1

    for arr in (prev, cen, nxt):
        nzm = (arr != 0)
        for r in (1, 2, 3, 5):
            add(_box_counts(nzm, r))
    add((prev == 0).astype(np.float32))
    add((nxt == 0).astype(np.float32))
    add(((prev == 0) & (nxt == 0)).astype(np.float32))
    for arr in (prev, cen, nxt):
        for r in (2, 3):
            add(_box_counts((arr == 0), r))
    for s in range(3):
        add(dt[:, s])
    add(enc)
    for L in vocab:
        for arr in (prev, cen, nxt):
            m = (arr == L)
            add(_box_counts(m, 2))
            add(_box_counts(m, 4))
    assert k[0] == F, (k[0], F)
    return X, rows, ii, jj


def _fit_stage(Xtr, ytr, is_target_stage, seed):
    if is_target_stage:
        clf = HGB(max_iter=350, learning_rate=0.08, max_leaf_nodes=31, min_samples_leaf=80,
                  l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
                  n_iter_no_change=15, random_state=seed)
        clf.fit(Xtr, (ytr > 0).astype(np.int32))
    else:
        clf = HGB(max_iter=400, learning_rate=0.08, max_leaf_nodes=31, min_samples_leaf=20,
                  l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
                  n_iter_no_change=15, random_state=seed)
        m = ytr > 0
        clf.fit(Xtr[m], ytr[m])
    return clf


def _predict_cells(clfA, clfB, X):
    """X: (ncells,F) -> per-cell 18-class probs (col0=bg)."""
    pA = np.clip(clfA.predict_proba(X)[:, 1], 0.0, 1.0)
    pB = clfB.predict_proba(X)
    out = np.zeros((len(X), C.NUM_CLASSES), np.float32)
    out[:, 0] = 1.0 - pA
    for kk, c in enumerate(clfB.classes_):
        out[:, c] = pA * pB[:, kk]
    return out


def _scatter(cellprobs, rows, ii, jj, nrows):
    proba = np.zeros((nrows, C.NUM_CLASSES, 32, 32), np.float32)
    proba[:, 0] = 1.0
    proba[rows, 0, ii, jj] = cellprobs[:, 0]
    for c in range(1, C.NUM_CLASSES):
        proba[rows, c, ii, jj] = cellprobs[:, c]
    return proba


def run(seed=42, save=True, name="percell_gbdt", report_random=False):
    t_all = time.time()
    tr = C.load_split("train")
    te = C.load_split("test")
    V, Tidx = tr["V"], tr["Tidx"].astype(np.int64)
    Vte = te["V"]
    N, Nte = len(V), len(Vte)

    vocab = _vocab(V)
    Xtr, rtr, itr, jtr = cell_matrix(V, vocab)      # all train center-zero cells
    ytr_all = Tidx[rtr, itr, jtr]

    def compute_oof(folds):
        oof = np.zeros((N, C.NUM_CLASSES, 32, 32), np.float32)
        oof[:, 0] = 1.0
        row_fold = np.full(N, -1, np.int64)
        for fidx, (_, vai) in enumerate(folds):
            row_fold[vai] = fidx
        cell_fold = row_fold[rtr]
        for fidx in range(len(folds)):
            trm = cell_fold != fidx
            vam = cell_fold == fidx
            clfA = _fit_stage(Xtr[trm], ytr_all[trm], True, seed)
            clfB = _fit_stage(Xtr[trm], ytr_all[trm], False, seed)
            cp = _predict_cells(clfA, clfB, Xtr[vam])
            oof[rtr[vam], 0, itr[vam], jtr[vam]] = cp[:, 0]
            for c in range(1, C.NUM_CLASSES):
                oof[rtr[vam], c, itr[vam], jtr[vam]] = cp[:, c]
        return oof

    # ---- HONEST volume-grouped OOF ----
    oof = compute_oof(C.make_group_folds(5, seed=seed))

    # ---- TEST: fit on ALL train cells, predict test cells ----
    clfA = _fit_stage(Xtr, ytr_all, True, seed)
    clfB = _fit_stage(Xtr, ytr_all, False, seed)
    Xte, rte, ite, jte = cell_matrix(Vte, vocab)
    cp = _predict_cells(clfA, clfB, Xte)
    test = _scatter(cp, rte, ite, jte, Nte)
    del Xte

    if save:
        sys.path.insert(0, os.path.join(_HERE, ".."))
        from ensemble import save_preds
        save_preds(name, oof, test)
    print(f"[percell_gbdt] done in {time.time()-t_all:.0f}s  oof{oof.shape} test{test.shape}", flush=True)

    if report_random:
        sys.path.insert(0, os.path.join(_HERE, ".."))
        import decide2 as D
        rand_oof = compute_oof(C.make_folds(N, 5, seed=seed))
        s_r, at_r, ma_r = D.tune_decision(rand_oof, V, tr["T"])
        print(f"[percell_gbdt] random-CV (leaky reference, inflated) {s_r:.4f} @ thr={at_r} min_area={ma_r}", flush=True)
    return oof, test


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(_HERE, ".."))
    import decide2 as D
    oof, test = run(report_random=True)
    tr = C.load_split("train")
    s_g, at, ma = D.tune_decision(oof, tr["V"], tr["T"])
    print(f"percell_gbdt GROUP-CV {s_g:.4f} @ thr={at} min_area={ma}   <-- HONEST (volume-disjoint)", flush=True)
