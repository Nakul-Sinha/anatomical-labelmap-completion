#!/usr/bin/env python
"""
Official Project Eris solution -- Small Anatomical Labelmap Completion.

Reproducible end-to-end: reads ./dataset/public/{train,test}.csv, trains from scratch, and
writes ./working/submission.csv through the declared modelling pipeline. No cached predictions
or precomputed submission are read.

Pipeline (every hyper-parameter chosen on HONEST volume-grouped CV -- see below):
  1. k-NN retrieval proba   : per test row, similarity-weighted vote of its top-k train
                              center-slice neighbours (center-cell agreement).
  2. CNN segmentation proba : an augmented U-Net (label embeddings, masked CE + soft-Dice,
                              joint dihedral augmentation) seed-ensembled on ALL train rows,
                              with 8-pose test-time augmentation.
  3. blend                  : fixed convex weights.
  4. 3D volume-consistency  : smooth probabilities along each source volume's sliding-window
                              slice chain (target structures are 3D-continuous).
  5. decision               : center-zero-constrained argmax target class with a probability
                              threshold and a tiny-region drop (both tuned on group CV).

WHY GROUP CV. The slabs are overlapping sliding windows, so adjacent slices of one source
volume otherwise leak across a random train/val split (k-NN inflates from 0.048 to 0.706).
The private test is volume-DISJOINT (verified: 0/300 exact slice matches; test neighbour
similarity 0.840 matches held-out-volume similarity 0.852). So only volume-grouped CV
(common.make_group_folds) estimates test performance; all params below are tuned on it.
Honest group-CV of this pipeline: ~0.057.
"""
import os
import sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "src"))
sys.path.insert(0, os.path.join(_HERE, "src", "methods"))

import common as C
import postprocess as P
import decide2 as D
import knn as KNN
import cnn_unet as CNN

# ---- Hyper-parameters (locked on honest volume-grouped CV) -----------------
KNN_K = 3
KNN_TAU = 1.0
W_KNN = 0.5          # blend weight for retrieval
W_CNN = 0.5          # blend weight for the CNN
SMOOTH_ALPHA = 0.7   # 3D chain smoothing decay
SMOOTH_WIDTH = 2     # neighbours each side along the slice chain
THRESH = 0.375       # min target probability to assign a cell
MIN_AREA = 0         # drop predicted labels with fewer than this many cells
CNN_TEST_SEEDS = (0, 1, 2)


def knn_test_proba(tr, te):
    cen_tr = tr["V"][:, 1].reshape(len(tr["V"]), -1).astype(np.int16)
    Tidx = tr["Tidx"].astype(np.int64)
    cen_te = te["V"][:, 1].reshape(len(te["V"]), -1).astype(np.int16)
    out = np.zeros((len(te["V"]), C.NUM_CLASSES, 32, 32), np.float32)
    for i in range(len(cen_te)):
        out[i] = KNN._proba_for_query(cen_te[i], cen_tr, Tidx, KNN_K, KNN_TAU)
    return out


def cnn_test_proba(tr, te):
    lut, n_vocab = CNN.build_vocab()
    cfg = {**CNN.DEFAULT_CFG}
    cfg["bg_idx"] = int(lut[0])
    idx_all, cz_all, y_all = CNN.make_tensors(tr["V"], lut, tr["Tidx"])
    idx_te, cz_te, _ = CNN.make_tensors(te["V"], lut)
    acc = np.zeros((len(te["V"]), C.NUM_CLASSES, 32, 32), np.float32)
    for sd in CNN_TEST_SEEDS:
        model = CNN.train_model(idx_all, cz_all, y_all, n_vocab, cfg, seed=777 + sd)
        acc += CNN.predict_proba(model, idx_te, cz_te, tta=cfg["tta"], bg_idx=cfg["bg_idx"])
    return acc / len(CNN_TEST_SEEDS)


def main():
    tr = C.load_split("train")
    te = C.load_split("test")

    knn = knn_test_proba(tr, te)
    cnn = cnn_test_proba(tr, te)

    blend = W_KNN * knn + W_CNN * cnn
    blend /= (blend.sum(1, keepdims=True) + 1e-9)
    blend = P.smooth3d(blend, te["V"], alpha=SMOOTH_ALPHA, width=SMOOTH_WIDTH)

    preds = [D.decide_adv(blend[i], te["V"][i, 1] == 0, THRESH, MIN_AREA)
             for i in range(len(te["ids"]))]

    out = os.path.join(_HERE, "working", "submission.csv")
    C.write_submission(te["ids"], preds, out)
    print(f"wrote {out}  ({len(preds)} rows)")


if __name__ == "__main__":
    main()
