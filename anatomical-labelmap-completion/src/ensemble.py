"""
Soft-probability ensembling infrastructure.

Every method (k-NN, CNN, ...) emits per-cell class probabilities of shape (N, 18, 32, 32)
for out-of-fold train predictions and for the test set, saved to artifacts/preds/<name>.npz
with keys `oof` and `test`. This module blends them and applies a single, CV-tuned decision rule:

    decide(proba, center_zero_mask, thresh):
        for each cell, take the highest-probability TARGET class (1..17);
        assign it iff its probability >= thresh AND the cell is center-zero (hard constraint);
        else background.

Blend weights and threshold are tuned on OOF against the exact grader metric.
"""
import os, sys, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

PRED_DIR = os.path.join(C.CHALLENGE_ROOT, "artifacts", "preds")
os.makedirs(PRED_DIR, exist_ok=True)


def save_preds(name, oof_proba, test_proba):
    oof_proba = np.asarray(oof_proba, dtype=np.float32)
    test_proba = np.asarray(test_proba, dtype=np.float32)
    assert oof_proba.shape[1:] == (C.NUM_CLASSES, 32, 32), oof_proba.shape
    assert test_proba.shape[1:] == (C.NUM_CLASSES, 32, 32), test_proba.shape
    path = os.path.join(PRED_DIR, f"{name}.npz")
    np.savez_compressed(path, oof=oof_proba, test=test_proba)
    return path


def load_preds(name):
    z = np.load(os.path.join(PRED_DIR, f"{name}.npz"))
    return z["oof"], z["test"]


def list_preds():
    return sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(PRED_DIR, "*.npz")))


def decide(proba, czmask, thresh):
    """proba: (18,32,32) -> (32,32) opaque labels. czmask: (32,32) bool (center-visible==0)."""
    fg = proba[1:]                      # (17,32,32)
    best = fg.argmax(0) + 1             # 1..17 class index
    best_p = fg.max(0)                  # (32,32)
    idx = np.where((best_p >= thresh) & czmask, best, 0)
    return C.idx_to_labels(idx)


def decide_stack(proba_stack, Vslabs, thresh):
    """proba_stack: (N,18,32,32); Vslabs: (N,3,32,32). Returns list of (32,32) opaque preds."""
    preds = []
    for i in range(len(proba_stack)):
        czm = Vslabs[i, 1] == 0
        preds.append(decide(proba_stack[i], czm, thresh))
    return preds


def score_proba(proba_stack, Vslabs, T, thresh):
    preds = decide_stack(proba_stack, Vslabs, thresh)
    return C.score_rows(preds, [T[i] for i in range(len(T))])


def tune_thresh(proba_stack, Vslabs, T, grid=None):
    if grid is None:
        grid = np.round(np.arange(0.15, 0.71, 0.025), 3)
    best = (-1, None)
    for th in grid:
        s = score_proba(proba_stack, Vslabs, T, th)
        if s > best[0]:
            best = (s, float(th))
    return best  # (score, thresh)


def blend(probas, weights=None):
    """probas: list of (N,18,32,32). Weighted average (renormalised per cell)."""
    if weights is None:
        weights = [1.0] * len(probas)
    acc = np.zeros_like(probas[0])
    for p, w in zip(probas, weights):
        acc += w * p
    acc /= (acc.sum(1, keepdims=True) + 1e-9)
    return acc


if __name__ == "__main__":
    tr = C.load_split("train")
    names = list_preds()
    print("available preds:", names)
    for nm in names:
        oof, _ = load_preds(nm)
        s, th = tune_thresh(oof, tr["V"], tr["T"])
        print(f"  {nm:16s} OOF best {s:.4f} @ thr={th}")
