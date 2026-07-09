"""
Config selection over ALL available methods on HONEST group CV:
  ensemble weights (Caruana) -> 3D smoothing (alpha,width) -> decision (thr,min_area).
Reports the winning composition so it can be baked into solution.py. (This script does not
produce the official submission; solution.py regenerates predictions from scratch.)
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import common as C
import ensemble as E
import decide2 as D
import postprocess as P

tr = C.load_split("train")
V, T = tr["V"], tr["T"]
names = E.list_preds()
print("methods:", names)
named_oof = {n: E.load_preds(n)[0] for n in names}

print("\n-- singles (group CV) --")
for n in names:
    s, th, ma = D.tune_decision(named_oof[n], V, T)
    print(f"  {n:14s} {s:.4f} @ thr={th} ma={ma}")

# Caruana ensemble on raw OOF
if len(names) >= 2:
    weights, sc, _, _ = D.ensemble_selection(named_oof, V, T)
else:
    weights = {names[0]: 1.0}
print(f"\nensemble weights: {weights}")
blend = D.blend_named(named_oof, weights)

print("\n-- + 3D smoothing --")
best = (-1, None, None, None, None)
for a in [0.5, 0.7, 1.0]:
    for w in [1, 2, 3]:
        sm = P.smooth3d(blend, V, alpha=a, width=w)
        s, th, ma = D.tune_decision(sm, V, T)
        if s > best[0]:
            best = (s, a, w, th, ma)
        print(f"  a={a} w={w}: {s:.4f} @ thr={th} ma={ma}")
print(f"\nFINAL: group-CV {best[0]:.4f}  weights={weights}  smooth(a={best[1]},w={best[2]})  thr={best[3]} min_area={best[4]}")
