"""
Shared harness for the Small Anatomical Labelmap Completion challenge.

Everything here is import-safe and path-robust (locates ./dataset/public relative
to the challenge root, regardless of the current working directory).

Exposes:
  - TARGET_LABELS, LABEL_TO_IDX, IDX_TO_LABEL  (fixed 17-label opaque space + background)
  - load_split('train'|'test')                 -> dict(ids, V, [T, Tidx])
  - score_rows(preds, truths)                  -> exact grader metric (squared active-macro-IoU)
  - make_folds(n, k=5, seed=42)                -> list of (train_idx, val_idx)
  - write_submission(ids, preds, path)         -> valid answer_json CSV
"""
from __future__ import annotations
import os, json, functools
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Paths (robust to cwd): challenge root is the parent of this file's src/ dir.
# ----------------------------------------------------------------------------
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
CHALLENGE_ROOT = os.path.dirname(_SRC_DIR)


def _find_data_dir() -> str:
    """Locate the public dataset dir. Prefer ./dataset/public, fall back to ./dataset."""
    for cand in (
        os.path.join(CHALLENGE_ROOT, "dataset", "public"),
        os.path.join(CHALLENGE_ROOT, "dataset"),
        os.path.join(os.getcwd(), "dataset", "public"),
        os.path.join(os.getcwd(), "dataset"),
    ):
        if os.path.exists(os.path.join(cand, "train.csv")):
            return cand
    raise FileNotFoundError("Could not locate dataset/public with train.csv")


DATA_DIR = _find_data_dir()
ARTIFACT_DIR = os.path.join(CHALLENGE_ROOT, "artifacts")
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# ----------------------------------------------------------------------------
# Fixed label space (verified identical across every train + test row).
# ----------------------------------------------------------------------------
TARGET_LABELS = (1893, 2126, 2660, 2823, 3127, 3680, 4638, 4877,
                 5094, 5170, 6102, 6798, 7200, 7760, 7775, 7989, 8525)
NUM_CLASSES = len(TARGET_LABELS) + 1  # 18 (background + 17 targets)
LABEL_TO_IDX = {0: 0}
for _i, _l in enumerate(TARGET_LABELS, start=1):
    LABEL_TO_IDX[_l] = _i
IDX_TO_LABEL = {v: k for k, v in LABEL_TO_IDX.items()}

_L2I_ARR = np.zeros(max(TARGET_LABELS) + 1, dtype=np.int64)
for _l, _i in LABEL_TO_IDX.items():
    _L2I_ARR[_l] = _i
_I2L_ARR = np.array([IDX_TO_LABEL[i] for i in range(NUM_CLASSES)], dtype=np.int64)


def labels_to_idx(a: np.ndarray) -> np.ndarray:
    """Map opaque target labels -> contiguous 0..17 indices (0 = background)."""
    return _L2I_ARR[a]


def idx_to_labels(a: np.ndarray) -> np.ndarray:
    """Map contiguous 0..17 indices -> opaque target labels."""
    return _I2L_ARR[a]


# ----------------------------------------------------------------------------
# Data loading (with npz cache for speed).
# ----------------------------------------------------------------------------
@functools.lru_cache(maxsize=4)
def load_split(which: str = "train"):
    assert which in ("train", "test")
    cache = os.path.join(ARTIFACT_DIR, f"{which}_arrays.npz")
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        out = {"ids": z["ids"], "V": z["V"]}
        if which == "train":
            out["T"] = z["T"]
            out["Tidx"] = z["Tidx"]
        return out

    df = pd.read_csv(os.path.join(DATA_DIR, f"{which}.csv"))
    ids, Vs = [], []
    Ts = [] if which == "train" else None
    for _, r in df.iterrows():
        ids.append(r["id"])
        ctx = json.loads(r["context_json"])
        v = np.asarray(ctx["visible_label_maps"], dtype=np.int64).reshape(3, 32, 32)
        Vs.append(v)
        if which == "train":
            t = np.asarray(json.loads(r["answer_json"])["target_label_map"],
                           dtype=np.int64).reshape(32, 32)
            Ts.append(t)
    ids = np.array(ids)
    V = np.stack(Vs).astype(np.int16)          # (N,3,32,32)
    out = {"ids": ids, "V": V}
    save = {"ids": ids, "V": V}
    if which == "train":
        T = np.stack(Ts).astype(np.int16)      # (N,32,32) opaque labels
        Tidx = labels_to_idx(T.astype(np.int64)).astype(np.int8)  # 0..17
        out["T"] = T
        out["Tidx"] = Tidx
        save.update({"T": T, "Tidx": Tidx})
    np.savez_compressed(cache, **save)
    return out


# ----------------------------------------------------------------------------
# Exact grader metric: mean over rows of (active-macro-IoU)^2.
# preds / truths: iterables of (32,32) int arrays of OPAQUE labels (0 = bg).
# ----------------------------------------------------------------------------
def score_rows(preds, truths) -> float:
    total = 0.0
    n = 0
    for p, t in zip(preds, truths):
        p = np.asarray(p); t = np.asarray(t)
        labs = set(np.unique(p[p > 0]).tolist()) | set(np.unique(t[t > 0]).tolist())
        if not labs:
            total += 1.0
        else:
            ious = []
            for L in labs:
                pi = p == L; ti = t == L
                inter = int(np.count_nonzero(pi & ti))
                union = int(np.count_nonzero(pi | ti))
                ious.append(inter / union if union > 0 else 0.0)
            total += float(np.mean(ious)) ** 2
        n += 1
    return total / max(n, 1)


def row_score(p, t) -> float:
    return score_rows([p], [t])


# ----------------------------------------------------------------------------
# Fixed cross-validation folds (shared across all experiments for comparability).
# ----------------------------------------------------------------------------
def make_folds(n: int, k: int = 5, seed: int = 42):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    folds = np.array_split(idx, k)
    out = []
    for i in range(k):
        val = folds[i]
        tr = np.concatenate([folds[j] for j in range(k) if j != i])
        out.append((np.sort(tr), np.sort(val)))
    return out


# ----------------------------------------------------------------------------
# Submission writer (valid answer_json schema).
# ----------------------------------------------------------------------------
def write_submission(ids, preds, path: str):
    rows = []
    for i, p in zip(ids, preds):
        p = np.asarray(p, dtype=np.int64).reshape(32, 32)
        obj = {"grid_shape": [32, 32], "target_label_map": p.reshape(-1).tolist()}
        rows.append({"id": i, "answer_json": json.dumps(obj, separators=(",", ":"))})
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    pd.DataFrame(rows, columns=["id", "answer_json"]).to_csv(path, index=False)
    return path


if __name__ == "__main__":
    tr = load_split("train"); te = load_split("test")
    print("train V", tr["V"].shape, "T", tr["T"].shape, "Tidx", tr["Tidx"].shape)
    print("test  V", te["V"].shape)
    print("num classes", NUM_CLASSES, "folds:", [(len(a), len(b)) for a, b in make_folds(len(tr["ids"]))])
    # sanity: perfect prediction scores 1.0
    print("perfect score:", score_rows(tr["T"][:20], tr["T"][:20]))
    print("empty-vs-truth:", score_rows([np.zeros((32, 32), int)] * 20, tr["T"][:20]))
