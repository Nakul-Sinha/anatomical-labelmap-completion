import sys, os
import numpy as np
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
import common as C, ensemble as E, decide2 as D, postprocess as P
tr = C.load_split("train"); V, T, Tidx = tr["V"], tr["T"], tr["Tidx"].astype(np.int64)
P_ = {n: E.load_preds(n)[0] for n in ["cnn_unet", "cnn_context", "knn"]}
for n in ["ctx9", "ctx11"]:
    P_[n] = np.load(os.path.join(C.ARTIFACT_DIR, f"{n}_oof.npy"))
bl = 0.28 * P_["ctx9"] + 0.28 * P_["ctx11"] + 0.16 * P_["cnn_unet"] + 0.16 * P_["cnn_context"] + 0.12 * P_["knn"]
bl /= bl.sum(1, keepdims=True) + 1e-9
sm = P.smooth3d(bl, V, 0.5, 2)
s, th, ma = D.tune_decision(sm, V, T)
print(f"ensemble +3D: {s:.4f} @ thr={th}")
from decide2 import decide_adv
rows = []
for i in range(len(sm)):
    p = decide_adv(sm[i], V[i, 1] == 0, th, ma)
    rs = C.score_rows([p], [T[i]])
    nlab = len(np.unique(Tidx[i][Tidx[i] > 0])); area = int((Tidx[i] > 0).sum())
    rows.append((rs, nlab, area))
rows = np.array(rows)
print("per-row score: mean %.4f" % rows[:, 0].mean())
print("\nby #true labels:")
for nl in range(1, 8):
    m = rows[:, 1] == nl
    if m.sum() > 3:
        print(f"  {nl} labels: n={int(m.sum()):3d} mean-score={rows[m,0].mean():.3f}  share-of-total={rows[m,0].sum()/rows[:,0].sum()*100:.0f}%")
print("\nby target area quartile:")
q = np.quantile(rows[:, 2], [0.25, 0.5, 0.75])
for lo, hi, tag in [(0, q[0], "small"), (q[0], q[1], "med-"), (q[1], q[2], "med+"), (q[2], 1e9, "large")]:
    m = (rows[:, 2] >= lo) & (rows[:, 2] < hi)
    if m.sum() > 3:
        print(f"  {tag}: n={int(m.sum()):3d} mean-score={rows[m,0].mean():.3f}")
print(f"\nrows scoring >0.2: {(rows[:,0]>0.2).sum()} ; >0.1: {(rows[:,0]>0.1).sum()} ; ==0: {(rows[:,0]==0).sum()}")
