"""Presence-aware decision: decouple WHICH labels (per-label mass threshold) from WHERE (argmax).
Tuned on honest group-CV OOF. Targets the ~0.026 label-set gap found by diag_decision."""
import sys, os
import numpy as np
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
import common as C, decide2 as D

name = sys.argv[1] if len(sys.argv) > 1 else "ctx9"
oof = np.load(os.path.join(C.ARTIFACT_DIR, f"{name}_oof.npy"))
tr = C.load_split("train"); V, T = tr["V"], tr["T"]
CZ = (V[:, 1] == 0)


def decide_presence(proba, cz, tau, sthr, agg="sum"):
    fg = proba[1:]                                   # (17,H,W)
    m = cz[None]
    mass = (fg * m).sum((1, 2)) if agg == "sum" else (fg * m).max(1).max(1)
    inc = mass >= tau                                # which labels present
    fgm = np.where(inc[:, None, None], fg, -1.0)
    best = fgm.argmax(0)                             # 0..16
    bestp = np.take_along_axis(fg, best[None], 0)[0]
    idx = np.where((bestp >= sthr) & cz & inc[best], best + 1, 0)
    return C.idx_to_labels(idx)


def score(tau, sthr, agg):
    preds = [decide_presence(oof[i], CZ[i], tau, sthr, agg) for i in range(len(oof))]
    return C.score_rows(preds, [T[i] for i in range(len(T))])


base, bth, bma = D.tune_decision(oof, V, T)
print(f"{name}: baseline argmax decision {base:.4f} @ thr={bth}")
best = (-1, None, None, None)
for agg in ("sum", "max"):
    grid_tau = np.round(np.arange(0.2, 6.0, 0.2), 2) if agg == "sum" else np.round(np.arange(0.2, 0.9, 0.05), 2)
    for tau in grid_tau:
        for sthr in np.round(np.arange(0.05, 0.5, 0.05), 2):
            s = score(tau, sthr, agg)
            if s > best[0]:
                best = (s, tau, sthr, agg)
print(f"{name}: presence-decision BEST {best[0]:.4f}  (tau={best[1]}, sthr={best[2]}, agg={best[3]})")
print(f"  gain over baseline: {best[0]-base:+.4f}")
