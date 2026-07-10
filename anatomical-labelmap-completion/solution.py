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
import deepnet as DEEP

W = {"deep7": 0.13, "deep9": 0.15, "deep11": 0.13, "deep13": 0.11, "deep15": 0.13,
     "unet": 0.11, "ctx": 0.06, "knn": 0.18}
SMOOTH_ALPHA, SMOOTH_WIDTH = 0.5, 3
THRESH, MIN_AREA = 0.375, 0
UNET_SEEDS = (0, 1)
CTX_SEEDS = (0, 1)
DEEP_SEEDS = (0, 1, 2)
CTX_CFG = dict(emb=16, w=96, dilations=(1, 2, 4, 8, 4, 2, 1), groups=8, dropout=0.30, bs=64,
               lr=2e-3, wd=5e-4, epochs=120, bg_weight=0.10, dice_w=1.0, aug=False, shift=0,
               tta=False, use_coord=False)


def knn_test(tr, te):
    cen_tr = tr["V"][:, 1].reshape(len(tr["V"]), -1).astype(np.int16)
    Tidx = tr["Tidx"].astype(np.int64)
    cen_te = te["V"][:, 1].reshape(len(te["V"]), -1).astype(np.int16)
    out = np.zeros((len(te["V"]), C.NUM_CLASSES, 32, 32), np.float32)
    for i in range(len(cen_te)):
        out[i] = KNN._proba_for_query(cen_te[i], cen_tr, Tidx, 3, 1.0)
    return out


def unet_test(tr, te):
    lut, nv = UNET.build_vocab()
    cfg = {**UNET.DEFAULT_CFG}; cfg["bg_idx"] = int(lut[0])
    ia, ca, ya = UNET.make_tensors(tr["V"], lut, tr["Tidx"])
    it, ct, _ = UNET.make_tensors(te["V"], lut)
    acc = np.zeros((len(te["V"]), C.NUM_CLASSES, 32, 32), np.float32)
    for sd in UNET_SEEDS:
        m = UNET.train_model(ia, ca, ya, nv, cfg, seed=777 + sd)
        acc += UNET.predict_proba(m, it, ct, tta=cfg["tta"], bg_idx=cfg["bg_idx"])
    return acc / len(UNET_SEEDS)


def ctx_test(tr, te):
    import torch
    lut, nv = CTX.build_vocab(); dev = CTX.DEVICE
    va = torch.tensor(CTX.encode_V(tr["V"], lut), device=dev)
    ta = torch.tensor(tr["Tidx"].astype(np.int64), device=dev)
    vt = torch.tensor(CTX.encode_V(te["V"], lut), device=dev)
    acc = np.zeros((len(te["V"]), C.NUM_CLASSES, 32, 32), np.float32)
    for sd in CTX_SEEDS:
        m = CTX._make_model(nv, CTX_CFG); CTX._train(m, va, ta, CTX_CFG, seed=100 + sd)
        acc += CTX._predict(m, vt, tta=CTX_CFG["tta"])
    return acc / len(CTX_SEEDS)


def deep_test(K):
    _, test = DEEP.run(K=K, cfg={"base": 40, "epochs": 170}, seeds_oof=(), seeds_test=DEEP_SEEDS,
                       save=False, name=f"deep{K}")
    return test


def main():
    tr = C.load_split("train"); te = C.load_split("test")
    proba = {"deep7": deep_test(7), "deep9": deep_test(9), "deep11": deep_test(11),
             "deep13": deep_test(13), "deep15": deep_test(15),
             "unet": unet_test(tr, te), "ctx": ctx_test(tr, te), "knn": knn_test(tr, te)}
    blend = sum(W[k] * proba[k] for k in W)
    blend /= (blend.sum(1, keepdims=True) + 1e-9)
    blend = P.smooth3d(blend, te["V"], alpha=SMOOTH_ALPHA, width=SMOOTH_WIDTH)
    preds = [D.decide_adv(blend[i], te["V"][i, 1] == 0, THRESH, MIN_AREA) for i in range(len(te["ids"]))]
    out = os.path.join(_HERE, "working", "submission.csv")
    C.write_submission(te["ids"], preds, out)
    print(f"wrote {out}  ({len(preds)} rows)")


if __name__ == "__main__":
    main()
