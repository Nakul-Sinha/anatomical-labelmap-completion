import sys, os
import numpy as np
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
import common as C, ensemble as E, decide2 as D, postprocess as P
tr = C.load_split("train"); V, T = tr["V"], tr["T"]
O = {n: E.load_preds(n)[0] for n in ["deepnet_k9", "deepnet_k11", "deepnet_k13", "cnn_unet", "cnn_context", "knn"]}
# precompute the smoothed blend once per config (smooth is the expensive part)
def ev(wd, tag):
    tot = sum(wd.values()); bl = sum(wd.get(k, 0) / tot * O[k] for k in O)
    b = bl / (bl.sum(1, keepdims=True) + 1e-9)
    s, th, ma = D.tune_decision(P.smooth3d(b, V, 0.5, 3), V, T)
    print(f"{tag:24s} {s:.4f} @ thr={th}", flush=True)

ev({"deepnet_k9": 1}, "single k9")
ev({"deepnet_k9": .34, "deepnet_k11": .33, "deepnet_k13": .33}, "3deep-equal")
ev({"deepnet_k9": .4, "deepnet_k11": .3, "deepnet_k13": .3}, "3deep-k9heavy")
ev({"deepnet_k9": .30, "deepnet_k11": .24, "deepnet_k13": .22, "cnn_unet": .10, "knn": .14}, "3deep+unet+knn")
ev({"deepnet_k9": .26, "deepnet_k11": .22, "deepnet_k13": .20, "cnn_unet": .10, "cnn_context": .06, "knn": .16}, "all6-balanced")
ev({"deepnet_k9": .24, "deepnet_k11": .20, "deepnet_k13": .18, "cnn_unet": .14, "cnn_context": .08, "knn": .16}, "all6-diverse")
ev({"deepnet_k9": .22, "deepnet_k11": .20, "deepnet_k13": .18, "cnn_unet": .20, "cnn_context": .08, "knn": .12}, "all6-unetheavy")
ev({"deepnet_k9": .28, "deepnet_k11": .24, "deepnet_k13": .22, "cnn_unet": .06, "cnn_context": .04, "knn": .16}, "deep-forward")
