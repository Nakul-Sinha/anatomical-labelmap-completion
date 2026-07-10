"""Fast greedy weight search: pre-smooth each model's OOF once (smooth3d is ~linear), then search
weights on the smoothed probas. Finds the OOF-optimal blend (test tracks CV, so higher CV -> higher test)."""
import sys, os
import numpy as np
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
import common as C, ensemble as E, decide2 as D, postprocess as P
tr = C.load_split("train"); V, T = tr["V"], tr["T"]
NAMES = ["deepnet_k7","deepnet_k9","deepnet_k11","deepnet_k13","deepnet_k15","cnn_unet","cnn_context","knn"]
SM = {}
for n in NAMES:
    o = E.load_preds(n)[0]; b = o / (o.sum(1, keepdims=True) + 1e-9)
    SM[n] = P.smooth3d(b, V, 0.5, 3)
print("pre-smoothed", flush=True)


def sc_fast(weights, thr=0.35):
    tot = sum(weights.values())
    bl = sum(weights[k] / tot * SM[k] for k in weights)
    bl /= bl.sum(1, keepdims=True) + 1e-9
    preds = [D.decide_adv(bl[i], V[i, 1] == 0, thr, 0) for i in range(len(bl))]
    return (C.score_rows(preds, [T[i] for i in range(len(T))]),)


def sc(weights):
    tot = sum(weights.values())
    bl = sum(weights[k] / tot * SM[k] for k in weights)
    bl /= bl.sum(1, keepdims=True) + 1e-9
    return D.tune_decision(bl, V, T)


# greedy add-with-replacement (Caruana) on integer counts -> robust
counts = {n: 0 for n in NAMES}
singles = sorted(((sc_fast({n: 1})[0], n) for n in NAMES), reverse=True)
for s, n in singles:
    print(f"  single {n:14s} {s:.4f}", flush=True)
counts[singles[0][1]] = 1
cur_best = singles[0][0]
for step in range(20):
    trial = []
    for n in NAMES:
        c2 = dict(counts); c2[n] += 1
        trial.append((sc_fast({k: c2[k] for k in NAMES if c2[k] > 0})[0], n))
    trial.sort(reverse=True)
    if trial[0][0] >= cur_best - 1e-5:
        cur_best = trial[0][0]; counts[trial[0][1]] += 1
    else:
        break
tot = sum(counts.values())
weights = {k: round(counts[k] / tot, 3) for k in NAMES if counts[k] > 0}
final = sc(weights)
print(f"\nGREEDY ensemble group-CV {final[0]:.4f}  thr={final[1]} ma={final[2]}", flush=True)
print(f"  weights {weights}", flush=True)
