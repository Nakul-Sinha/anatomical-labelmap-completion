"""
Strict submission validator — mirrors the grader's acceptance checks so we never ship an
invalid file. Usage: python src/validate_submission.py working/submission.csv
"""
import os, sys, json
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

VALID_LABELS = set([0]) | set(C.TARGET_LABELS)


def validate(sub_path: str) -> bool:
    errs = []
    df = pd.read_csv(sub_path)
    # columns: exactly id, answer_json (order-independent)
    if set(df.columns) != {"id", "answer_json"}:
        errs.append(f"columns must be exactly {{id, answer_json}}, got {list(df.columns)}")
        print("INVALID:", errs); return False

    te = C.load_split("test")
    test_ids = list(te["ids"])
    sub_ids = df["id"].astype(str).tolist()
    if len(sub_ids) != len(test_ids):
        errs.append(f"row count {len(sub_ids)} != {len(test_ids)}")
    if len(set(sub_ids)) != len(sub_ids):
        errs.append("duplicate ids present")
    if set(sub_ids) != set(map(str, test_ids)):
        miss = set(map(str, test_ids)) - set(sub_ids)
        extra = set(sub_ids) - set(map(str, test_ids))
        errs.append(f"id mismatch: {len(miss)} missing, {len(extra)} extra")

    bad_json = bad_keys = bad_shape = bad_len = bad_lab = 0
    for a in df["answer_json"]:
        try:
            o = json.loads(a)
        except Exception:
            bad_json += 1; continue
        if set(o.keys()) != {"grid_shape", "target_label_map"}:
            bad_keys += 1; continue
        if list(o["grid_shape"]) != [32, 32]:
            bad_shape += 1
        m = o["target_label_map"]
        if len(m) != 1024:
            bad_len += 1
        else:
            u = set(int(x) for x in np.unique(m).tolist())
            if not u.issubset(VALID_LABELS):
                bad_lab += 1
    for nm, cnt in [("unparseable json", bad_json), ("wrong top-level keys", bad_keys),
                    ("wrong grid_shape", bad_shape), ("wrong length", bad_len),
                    ("labels out of namespace", bad_lab)]:
        if cnt:
            errs.append(f"{cnt} rows: {nm}")

    if errs:
        print("INVALID submission:")
        for e in errs:
            print("  -", e)
        return False
    print(f"VALID submission: {len(sub_ids)} rows, ids match, all answer_json well-formed.")
    return True


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(C.CHALLENGE_ROOT, "working", "submission.csv")
    ok = validate(path)
    sys.exit(0 if ok else 1)
