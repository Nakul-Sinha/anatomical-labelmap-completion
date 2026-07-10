"""Is the label-SET the bottleneck? Compare my decision vs oracle-label-set on a saved OOF."""
import sys, os
import numpy as np
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
import common as C, decide2 as D

name = sys.argv[1] if len(sys.argv) > 1 else "ctx9"
oof = np.load(os.path.join(C.ARTIFACT_DIR, f"{name}_oof.npy"))
tr = C.load_split("train"); V, T, Tidx = tr["V"], tr["T"], tr["Tidx"].astype(np.int64)

s, th, ma = D.tune_decision(oof, V, T)
print(f"{name}: my decision  group-CV {s:.4f} @ thr={th} ma={ma}")

# Oracle label set: for each row, forbid target classes NOT present in truth (move their mass to bg).
oracle = oof.copy()
for i in range(len(oof)):
    present = set(np.unique(Tidx[i][Tidx[i] > 0]).tolist())
    for c in range(1, C.NUM_CLASSES):
        if c not in present:
            oracle[i, 0] += oracle[i, c]
            oracle[i, c] = 0.0
so, tho, mao = D.tune_decision(oracle, V, T)
print(f"{name}: ORACLE label-set group-CV {so:.4f} @ thr={tho}  (gap from label-set errors: {so-s:.4f})")

# Oracle presence recall: how many true labels does my argmax decision even predict?
from decide2 import decide_adv
tp = fp = fn = 0
for i in range(len(oof)):
    p = decide_adv(oof[i], V[i, 1] == 0, th, ma)
    ps = set(np.unique(p[p > 0]).tolist()); ts = set(np.unique(T[i][T[i] > 0]).tolist())
    tp += len(ps & ts); fp += len(ps - ts); fn += len(ts - ps)
prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
print(f"{name}: label-set precision={prec:.3f} recall={rec:.3f}  (avg extra {fp/len(oof):.2f}/row, missed {fn/len(oof):.2f}/row)")
