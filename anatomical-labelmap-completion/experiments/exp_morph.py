"""Decision-side morphology: after decoding, adjust each predicted label's region (dilate / close /
largest-component) within center-zero cells. Cheap test of whether region sizing caps per-label IoU."""
import sys, os
import numpy as np
from scipy import ndimage
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
import common as C, decide2 as D, postprocess as P
name = sys.argv[1] if len(sys.argv) > 1 else "ctx11"
oof = np.load(os.path.join(C.ARTIFACT_DIR, f"{name}_oof.npy"))
tr = C.load_split("train"); V, T = tr["V"], tr["T"]
b = oof / (oof.sum(1, keepdims=True) + 1e-9)
sm = P.smooth3d(b, V, 0.7, 3)
base, th, ma = D.tune_decision(sm, V, T)
print(f"{name}: base +3D {base:.4f} @ thr={th}")


def decode(proba, cz, thr):
    fg = proba[1:]; best = fg.argmax(0) + 1; bp = fg.max(0)
    return np.where((bp >= thr) & cz, best, 0)


def morph(idx, cz, op, k):
    out = idx.copy()
    for L in range(1, 18):
        m = idx == L
        if m.sum() == 0:
            continue
        if op == "dilate":
            m2 = ndimage.binary_dilation(m, iterations=k) & cz & ((idx == 0) | (idx == L))
        elif op == "close":
            m2 = ndimage.binary_closing(m, iterations=k) & cz
        elif op == "fill":
            m2 = ndimage.binary_fill_holes(m) & cz
        else:
            m2 = m
        out[(m2) & (~m) & (idx == 0)] = L        # only grow into background cells
        if op in ("close", "fill"):
            out[m2] = np.where(idx[m2] == 0, L, idx[m2])
    return out


for op, ks in [("dilate", [1, 2]), ("close", [1, 2]), ("fill", [0])]:
    for k in ks:
        preds = []
        for i in range(len(oof)):
            cz = V[i, 1] == 0
            idx = C.labels_to_idx(decode(sm[i], cz, th))
            idx2 = morph(idx, cz, op, k)
            preds.append(C.idx_to_labels(idx2))
        s = C.score_rows(preds, [T[i] for i in range(len(T))])
        print(f"  {op} k={k}: {s:.4f}  ({s-base:+.4f})")
