#!/usr/bin/env python
"""
Official Project Eris solution -- Small Anatomical Labelmap Completion.

Reproducible end-to-end: reads ./dataset/public/{train,test}.csv, trains from scratch, and
writes ./working/submission.csv through the declared modelling pipeline. No cached predictions
or precomputed submission are read.

Pipeline (every hyper-parameter chosen on HONEST volume-grouped CV -- see below):
  1. k-NN retrieval proba   : per test row, similarity-weighted vote of its top-k train
                              center-slice neighbours (center-cell agreement).
  2. U-Net proba            : plain U-Net over embedded label maps (masked CE + soft-Dice),
                              seed-ensembled on ALL train rows.
  3. dilated-context proba  : a dilated fully-conv net (different receptive field) for
                              architectural diversity, seed-ensembled on ALL train rows.
  4. blend                  : fixed convex weights.
  5. 3D volume-consistency  : smooth probabilities along each source volume's sliding-window
                              slice chain (target structures are 3D-continuous).
  6. decision               : center-zero-constrained argmax target class with a probability
                              threshold (and optional tiny-region drop).

WHY GROUP CV, and WHY NO AUGMENTATION. The slabs are overlapping sliding windows, so adjacent
slices of one source volume otherwise leak across a random train/val split (k-NN inflates from
0.048 to 0.706). The private test is volume-DISJOINT (verified: 0/300 exact slice matches;
test neighbour similarity 0.840 matches held-out-volume 0.852), so only volume-grouped CV
(common.make_group_folds) estimates test performance. This data is an ATLAS: the oriented local
configuration of anatomy around a hole carries real signal, so geometric augmentation / TTA /
absolute-coordinate channels all HURT on group CV and are disabled (verified by ablation).
Honest group-CV of this pipeline: ~0.079 (blend U-Net 0.45 + dilated-context 0.45 + k-NN 0.10,
then 3D smoothing). Single methods: U-Net 0.064, dilated-context 0.060, k-NN 0.048.
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
import cnn_unet as UNET
import cnn_context as CTX

# ---- Hyper-parameters (locked on honest volume-grouped CV) -----------------
KNN_K, KNN_TAU = 3, 1.0
W_KNN, W_UNET, W_CTX = 0.10, 0.45, 0.45
SMOOTH_ALPHA, SMOOTH_WIDTH = 0.7, 3
THRESH, MIN_AREA = 0.225, 0
UNET_SEEDS = (0, 1, 2, 3)
CTX_SEEDS = (0, 1, 2)
CTX_CFG = dict(emb=16, w=96, dilations=(1, 2, 4, 8, 4, 2, 1), groups=8, dropout=0.30,
               bs=64, lr=2e-3, wd=5e-4, epochs=120, bg_weight=0.10, dice_w=1.0,
               aug=False, shift=0, tta=False, use_coord=False)


def knn_test_proba(tr, te):
    cen_tr = tr["V"][:, 1].reshape(len(tr["V"]), -1).astype(np.int16)
    Tidx = tr["Tidx"].astype(np.int64)
    cen_te = te["V"][:, 1].reshape(len(te["V"]), -1).astype(np.int16)
    out = np.zeros((len(te["V"]), C.NUM_CLASSES, 32, 32), np.float32)
    for i in range(len(cen_te)):
        out[i] = KNN._proba_for_query(cen_te[i], cen_tr, Tidx, KNN_K, KNN_TAU)
    return out


def unet_test_proba(tr, te):
    lut, n_vocab = UNET.build_vocab()
    cfg = {**UNET.DEFAULT_CFG}
    cfg["bg_idx"] = int(lut[0])
    idx_all, cz_all, y_all = UNET.make_tensors(tr["V"], lut, tr["Tidx"])
    idx_te, cz_te, _ = UNET.make_tensors(te["V"], lut)
    acc = np.zeros((len(te["V"]), C.NUM_CLASSES, 32, 32), np.float32)
    for sd in UNET_SEEDS:
        m = UNET.train_model(idx_all, cz_all, y_all, n_vocab, cfg, seed=777 + sd)
        acc += UNET.predict_proba(m, idx_te, cz_te, tta=cfg["tta"], bg_idx=cfg["bg_idx"])
    return acc / len(UNET_SEEDS)


def ctx_test_proba(tr, te):
    import torch
    lut, n_vocab = CTX.build_vocab()
    dev = CTX.DEVICE
    vidx_all = torch.tensor(CTX.encode_V(tr["V"], lut), device=dev)
    tidx_all = torch.tensor(tr["Tidx"].astype(np.int64), device=dev)
    vidx_te = torch.tensor(CTX.encode_V(te["V"], lut), device=dev)
    acc = np.zeros((len(te["V"]), C.NUM_CLASSES, 32, 32), np.float32)
    for sd in CTX_SEEDS:
        m = CTX._make_model(n_vocab, CTX_CFG)
        CTX._train(m, vidx_all, tidx_all, CTX_CFG, seed=100 + sd)
        acc += CTX._predict(m, vidx_te, tta=CTX_CFG["tta"])
    return acc / len(CTX_SEEDS)


def main():
    tr = C.load_split("train")
    te = C.load_split("test")

    knn = knn_test_proba(tr, te)
    unet = unet_test_proba(tr, te)
    ctx = ctx_test_proba(tr, te)

    blend = W_KNN * knn + W_UNET * unet + W_CTX * ctx
    blend /= (blend.sum(1, keepdims=True) + 1e-9)
    blend = P.smooth3d(blend, te["V"], alpha=SMOOTH_ALPHA, width=SMOOTH_WIDTH)

    preds = [D.decide_adv(blend[i], te["V"][i, 1] == 0, THRESH, MIN_AREA)
             for i in range(len(te["ids"]))]

    out = os.path.join(_HERE, "working", "submission.csv")
    C.write_submission(te["ids"], preds, out)
    print(f"wrote {out}  ({len(preds)} rows)")


if __name__ == "__main__":
    main()
