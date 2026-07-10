import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src")); sys.path.insert(0, os.path.join(_ROOT, "src", "methods"))
import common as C, decide2 as D
import ctx25d
for K, nm in [(11, "ctx25d_k11"), (9, "ctx25d_k9")]:
    oof, test = ctx25d.run(K=K, seeds_oof=(0, 1), seeds_test=(0, 1, 2), name=nm)
    tr = C.load_split("train")
    print(f"{nm}: group-CV {D.tune_decision(oof, tr['V'], tr['T'])[0]:.4f}", flush=True)
