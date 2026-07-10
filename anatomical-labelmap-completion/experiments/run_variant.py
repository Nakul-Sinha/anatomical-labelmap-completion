"""Run a higher-capacity PLAIN U-Net variant (controlled) and report honest group-CV."""
import sys, os, time
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "src", "methods"))
import common as C
import decide2 as D
import cnn_unet

t = time.time()
oof, test = cnn_unet.run(
    cfg={"base": 72, "dropout": 0.30, "epochs": 140, "aug": False, "tta": False, "use_coords": False},
    seeds_oof=(0, 1), seeds_test=(0, 1, 2), save=True, name="unet_big", verbose=True)
tr = C.load_split("train")
s, at, ma = D.tune_decision(oof, tr["V"], tr["T"])
print(f"\n>>> unet_big group-CV {s:.4f} @ thr={at} ma={ma}  ({time.time()-t:.0f}s)", flush=True)
