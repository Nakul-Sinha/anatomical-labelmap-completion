"""
k-NN retrieval method -> soft per-cell class probabilities in the ensemble format.

Similarity = center-slice cell agreement (best in CV). Probabilities for a query cell are the
similarity-weighted class distribution of its top-k neighbours' target maps. Enforcing the
center-zero constraint is left to the shared decision rule (src/ensemble.decide).
"""
import os, sys
import numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
import common as C


def _center_agree(qcen_flat, tcen_flat):
    # qcen_flat: (1024,), tcen_flat: (m,1024) -> (m,) count of matching cells
    return (tcen_flat == qcen_flat).sum(1).astype(np.float64)


def _proba_for_query(qi_cen, train_cen, train_Tidx, k, tau):
    s = _center_agree(qi_cen, train_cen)          # (m,)
    order = np.argsort(-s)[:k]
    w = s[order].astype(np.float64)
    w = w - w.min() + 1e-3
    w = w ** tau
    acc = np.zeros((C.NUM_CLASSES, 1024), np.float64)
    for j, wj in zip(order, w):
        ti = train_Tidx[j].reshape(-1)
        acc[ti, np.arange(1024)] += wj
    acc /= (acc.sum(0, keepdims=True) + 1e-9)
    return acc.reshape(C.NUM_CLASSES, 32, 32)


def run(k=3, tau=1.0, seed=42, save=True, name="knn"):
    tr = C.load_split("train")
    te = C.load_split("test")
    Vtr, Tidx = tr["V"], tr["Tidx"].astype(np.int64)
    cen_tr = Vtr[:, 1].reshape(len(Vtr), -1).astype(np.int16)
    N = len(Vtr)

    # OOF
    oof = np.zeros((N, C.NUM_CLASSES, 32, 32), np.float32)
    for tri, vai in C.make_folds(N, 5, seed=seed):
        tc = cen_tr[tri]; tt = Tidx[tri]
        for qi in vai:
            oof[qi] = _proba_for_query(cen_tr[qi], tc, tt, k, tau)

    # TEST (all train)
    Vte = te["V"]
    cen_te = Vte[:, 1].reshape(len(Vte), -1).astype(np.int16)
    test = np.zeros((len(Vte), C.NUM_CLASSES, 32, 32), np.float32)
    for qi in range(len(Vte)):
        test[qi] = _proba_for_query(cen_te[qi], cen_tr, Tidx, k, tau)

    if save:
        from ensemble import save_preds
        save_preds(name, oof, test)
    return oof, test


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(_HERE, ".."))
    import ensemble as E
    oof, test = run()
    tr = C.load_split("train")
    s, th = E.tune_thresh(oof, tr["V"], tr["T"])
    print(f"kNN OOF unified score {s:.4f} @ thr={th}; saved proba shapes oof{oof.shape} test{test.shape}")
