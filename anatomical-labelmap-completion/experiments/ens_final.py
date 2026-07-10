"""Final ensemble tuning on honest group-CV OOF: greedy weight search over all members + 3D + decision."""
import sys, os, itertools
import numpy as np
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
import common as C, ensemble as E, decide2 as D, postprocess as P
tr = C.load_split("train"); V, T = tr["V"], tr["T"]

NAMES = ["deepnet_k9", "deepnet_k11", "deepnet_k13", "cnn_unet", "cnn_context", "knn"]
O = {}
for n in NAMES:
    try:
        O[n] = E.load_preds(n)[0]
    except Exception as e:
        print("missing", n, e)
NAMES = [n for n in NAMES if n in O]


def score(bl, a=0.5, w=2):
    b = bl / (bl.sum(1, keepdims=True) + 1e-9)
    return D.tune_decision(P.smooth3d(b, V, a, w), V, T)


print("singles (+3D best):")
for n in NAMES:
    best = max((score(O[n], a, w)[0], a, w) for a in (0.5, 0.7) for w in (2, 3))
    print(f"  {n:14s} {best[0]:.4f}")

# greedy weight search (Caruana-style with fractional grid)
grid = [0.0, 0.1, 0.2, 0.3, 0.4]
best = (-1, None, None, None)
# seed with top single
base = {n: 0.0 for n in NAMES}
order = sorted(NAMES, key=lambda n: -score(O[n])[0])
base[order[0]] = 1.0
for _ in range(6):
    improved = False
    for n in NAMES:
        for wv in grid:
            trial = dict(base); trial[n] = wv
            tot = sum(trial.values())
            if tot == 0:
                continue
            bl = sum(trial[k] / tot * O[k] for k in NAMES)
            for a in (0.5, 0.7):
                for w in (2, 3):
                    s = score(bl, a, w)
                    if s[0] > best[0]:
                        best = (s[0], {k: round(trial[k] / tot, 3) for k in NAMES if trial[k] > 0}, (a, w), (s[1], s[2]))
                        base = dict(trial); improved = True
    if not improved:
        break
print(f"\nBEST ensemble group-CV {best[0]:.4f}")
print(f"  weights {best[1]}")
print(f"  smooth {best[2]}  decision thr/ma {best[3]}")
