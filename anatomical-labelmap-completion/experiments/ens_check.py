import sys, os
import numpy as np
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
import common as C, ensemble as E, decide2 as D, postprocess as P
tr = C.load_split("train"); V, T = tr["V"], tr["T"]
P_ = {n: E.load_preds(n)[0] for n in ["cnn_unet", "cnn_context", "knn"]}
for n in ["ctx9", "ctx11"]:
    P_[n] = np.load(os.path.join(C.ARTIFACT_DIR, f"{n}_oof.npy"))


def ev(bl, tag):
    b = bl / (bl.sum(1, keepdims=True) + 1e-9); best = (-1, 0, 0)
    for a in (0.5, 0.7):
        for w in (2, 3):
            s = D.tune_decision(P.smooth3d(b, V, a, w), V, T)[0]
            if s > best[0]:
                best = (s, a, w)
    print(f"{tag:34s} +3D {best[0]:.4f} (a={best[1]},w={best[2]})", flush=True)


un, ct, kn, c9, c11 = (P_[k] for k in ["cnn_unet", "cnn_context", "knn", "ctx9", "ctx11"])
print("singles+3D:", flush=True)
for n in P_:
    ev(P_[n], "  " + n)
ev(0.5 * c9 + 0.5 * c11, "ctx9+ctx11")
ev(0.4 * c9 + 0.4 * c11 + 0.2 * un, "ctx9+ctx11+unet")
ev(0.3 * c9 + 0.3 * c11 + 0.2 * un + 0.2 * ct, "+ctx2d")
ev(0.28 * c9 + 0.28 * c11 + 0.16 * un + 0.16 * ct + 0.12 * kn, "all5")
