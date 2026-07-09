"""
Advanced decision rule + robust ensemble selection for the squared-macro-IoU metric.

Decision: per cell take the argmax target class; keep it iff prob >= assign_thr AND center-zero.
Then drop predicted labels whose region is tiny (< min_area) — such labels are usually false
positives that zero-out a row's macro-IoU (label "present in either" gets IoU~0, then squared).

Ensemble: Caruana-style greedy selection with replacement over method probabilities — robust to
overfitting blend weights, returns an implicit weighting + a final tuned threshold.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import ensemble as E


def decide_adv(proba, czmask, assign_thr=0.4, min_area=0):
    fg = proba[1:]
    best = fg.argmax(0) + 1
    best_p = fg.max(0)
    idx = np.where((best_p >= assign_thr) & czmask, best, 0)
    if min_area > 0:
        for L in range(1, C.NUM_CLASSES):
            m = idx == L
            if 0 < m.sum() < min_area:
                idx[m] = 0
    return C.idx_to_labels(idx)


def score_stack(proba_stack, Vslabs, T, assign_thr, min_area=0):
    preds = []
    for i in range(len(proba_stack)):
        preds.append(decide_adv(proba_stack[i], Vslabs[i, 1] == 0, assign_thr, min_area))
    return C.score_rows(preds, [T[i] for i in range(len(T))])


def tune_decision(proba_stack, Vslabs, T, thr_grid=None, area_grid=(0, 1, 2, 3, 4)):
    if thr_grid is None:
        thr_grid = np.round(np.arange(0.2, 0.66, 0.025), 3)
    best = (-1, None, None)
    for th in thr_grid:
        for ma in area_grid:
            s = score_stack(proba_stack, Vslabs, T, th, ma)
            if s > best[0]:
                best = (s, float(th), int(ma))
    return best  # (score, assign_thr, min_area)


def ensemble_selection(named_oof, Vslabs, T, n_iter=25, init_top=2, thr=0.4):
    """Greedy Caruana selection with replacement. named_oof: dict name->(N,18,32,32).
    Returns (weights dict, cv_score, assign_thr, min_area)."""
    names = list(named_oof)
    # seed with the best single methods
    singles = []
    for nm in names:
        s = score_stack(named_oof[nm], Vslabs, T, thr)
        singles.append((s, nm))
    singles.sort(reverse=True)
    selected = [nm for _, nm in singles[:init_top]]
    counts = {nm: 0 for nm in names}
    for nm in selected:
        counts[nm] += 1

    def blended(sel):
        acc = np.zeros_like(named_oof[names[0]])
        for nm in sel:
            acc += named_oof[nm]
        acc /= len(sel)
        return acc

    cur = blended(selected)
    best_s = score_stack(cur, Vslabs, T, thr)
    for _ in range(n_iter):
        trial = []
        for nm in names:
            newsel = selected + [nm]
            b = blended(newsel)
            s = score_stack(b, Vslabs, T, thr)
            trial.append((s, nm))
        trial.sort(reverse=True)
        if trial[0][0] >= best_s - 1e-6:
            best_s = trial[0][0]
            selected.append(trial[0][1])
            counts[trial[0][1]] += 1
        else:
            break
    tot = sum(counts.values())
    weights = {nm: counts[nm] / tot for nm in names if counts[nm] > 0}
    # final blended proba with these weights, then tune decision
    acc = np.zeros_like(named_oof[names[0]])
    for nm, w in weights.items():
        acc += w * named_oof[nm]
    sc, at, ma = tune_decision(acc, Vslabs, T)
    return weights, sc, at, ma


def blend_named(named, weights):
    names = list(weights)
    acc = np.zeros_like(named[names[0]])
    for nm, w in weights.items():
        acc += w * named[nm]
    acc /= (acc.sum(1, keepdims=True) + 1e-9)
    return acc


if __name__ == "__main__":
    tr = C.load_split("train")
    names = E.list_preds()
    print("preds available:", names)
    named_oof = {nm: E.load_preds(nm)[0] for nm in names}
    for nm in names:
        s, at, ma = tune_decision(named_oof[nm], tr["V"], tr["T"])
        print(f"  {nm:16s} single OOF {s:.4f} @ thr={at} min_area={ma}")
    if len(names) >= 2:
        w, sc, at, ma = ensemble_selection(named_oof, tr["V"], tr["T"])
        print(f"\nENSEMBLE weights={w}\n  OOF {sc:.4f} @ thr={at} min_area={ma}")
