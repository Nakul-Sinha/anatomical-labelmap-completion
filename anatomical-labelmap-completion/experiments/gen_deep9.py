import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src")); sys.path.insert(0, os.path.join(_ROOT, "src", "methods"))
import common as C, decide2 as D
import deepnet
tr = C.load_split("train")
oof, test = deepnet.run(K=9, cfg={"base": 40, "epochs": 170}, seeds_oof=(0, 1, 2), seeds_test=(0, 1, 2, 3), name="deepnet_k9")
print(f"deepnet_k9: group-CV {D.tune_decision(oof, tr['V'], tr['T'])[0]:.4f}", flush=True)
