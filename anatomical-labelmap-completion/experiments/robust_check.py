import sys, os
import numpy as np
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
import common as C, ensemble as E, decide2 as D, postprocess as P
tr = C.load_split("train"); V, T = tr["V"], tr["T"]
O = {n: E.load_preds(n)[0] for n in ["deepnet_k11", "deepnet_k13", "cnn_unet", "cnn_context", "knn"]}


def ev(wd, tag):
    tot = sum(wd.values()); bl = sum(wd[k] / tot * O[k] for k in wd)
    b = bl / (bl.sum(1, keepdims=True) + 1e-9)
    s, th, ma = D.tune_decision(P.smooth3d(b, V, 0.5, 3), V, T)
    print(f"{tag:14s} {s:.4f} @ thr={th}", flush=True)


ev({"deepnet_k11": .083, "deepnet_k13": .25, "cnn_unet": .333, "cnn_context": .083, "knn": .25}, "greedy")
ev({"deepnet_k11": .35, "deepnet_k13": .35, "cnn_unet": .10, "cnn_context": .05, "knn": .15}, "deep-heavy")
ev({"deepnet_k11": .30, "deepnet_k13": .30, "cnn_unet": .12, "cnn_context": .08, "knn": .20}, "deep+knn")
ev({"deepnet_k11": .28, "deepnet_k13": .28, "cnn_unet": .16, "cnn_context": .10, "knn": .18}, "balanced")
ev({"deepnet_k11": .5, "deepnet_k13": .5}, "deep-only")
ev({"deepnet_k11": .30, "deepnet_k13": .25, "cnn_unet": .20, "cnn_context": .10, "knn": .15}, "mix")
