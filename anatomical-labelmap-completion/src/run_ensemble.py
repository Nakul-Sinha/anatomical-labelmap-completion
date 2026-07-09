"""
Experiment driver: ensemble the available method probabilities under HONEST group CV, then
produce + validate the test submission.

Assumes each method saved group-honest OOF (generated with common.make_group_folds) and test proba
in artifacts/preds/<name>.npz. Retrieval-type methods that don't generalize on group CV are excluded
automatically (they fall out of Caruana selection).
"""
import os, sys, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import ensemble as E
import decide2 as D


def main(exclude=(), write=True):
    tr = C.load_split("train")
    te = C.load_split("test")
    names = [n for n in E.list_preds() if n not in exclude]
    if not names:
        print("no predictions available yet.")
        return
    named_oof = {n: E.load_preds(n)[0] for n in names}
    named_test = {n: E.load_preds(n)[1] for n in names}

    print("=== single-method group-CV (honest) ===")
    singles = []
    for n in names:
        s, at, ma = D.tune_decision(named_oof[n], tr["V"], tr["T"])
        singles.append((s, n, at, ma))
        print(f"  {n:16s} {s:.4f}  (thr={at}, min_area={ma})")
    singles.sort(reverse=True)

    if len(names) == 1:
        weights = {names[0]: 1.0}
        sc, at, ma = singles[0][0], singles[0][2], singles[0][3]
    else:
        weights, sc, at, ma = D.ensemble_selection(named_oof, tr["V"], tr["T"])
    print(f"\n=== ensemble ===\n  weights={weights}\n  group-CV OOF {sc:.4f} @ thr={at} min_area={ma}")

    # blend test proba and decide
    test_blend = D.blend_named(named_test, weights)
    preds = [D.decide_adv(test_blend[i], te["V"][i, 1] == 0, at, ma) for i in range(len(te["ids"]))]
    if write:
        out = os.path.join(C.CHALLENGE_ROOT, "working", "submission.csv")
        C.write_submission(te["ids"], preds, out)
        print(f"\nwrote {out}")
        import validate_submission as VS
        VS.validate(out)
    return weights, sc, at, ma


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--no-write", action="store_true")
    a = ap.parse_args()
    main(exclude=tuple(a.exclude), write=not a.no_write)
