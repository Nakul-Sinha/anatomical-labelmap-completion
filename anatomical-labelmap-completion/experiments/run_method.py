"""Controlled runner: run one method's run(), save proba, report honest group-CV. Unbuffered."""
import sys, os, time
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "src", "methods"))
import common as C
import decide2 as D

name = sys.argv[1]
kwargs = eval(sys.argv[2]) if len(sys.argv) > 2 else {}
mod = __import__(name)
t = time.time()
oof, test = mod.run(save=True, name=name, **kwargs)[:2]
tr = C.load_split("train")
s, at, ma = D.tune_decision(oof, tr["V"], tr["T"])
print(f"\n>>> {name} group-CV {s:.4f} @ thr={at} min_area={ma}  ({time.time()-t:.0f}s)", flush=True)
