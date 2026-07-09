"""Regenerate kNN OOF with HONEST volume-group folds; keep the (fold-independent) test proba."""
import os, sys
import numpy as np
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "src", "methods"))
import common as C
import ensemble as E
from knn import _proba_for_query

tr = C.load_split("train")
Vtr = tr["V"]; Tidx = tr["Tidx"].astype(np.int64); N = len(Vtr)
cen = Vtr[:, 1].reshape(N, -1).astype(np.int16)

oof = np.zeros((N, C.NUM_CLASSES, 32, 32), np.float32)
for tri, vai in C.make_group_folds(5, 42):
    tc = cen[tri]; tt = Tidx[tri]
    for qi in vai:
        oof[qi] = _proba_for_query(cen[qi], tc, tt, 3, 1.0)

_, test = E.load_preds("knn")           # test proba unchanged (retrieves from all train)
E.save_preds("knn", oof, test)
s, th = E.tune_thresh(oof, Vtr, tr["T"])
print(f"knn GROUP-honest OOF {s:.4f} @ thr={th} (overwrote knn.npz oof)")
