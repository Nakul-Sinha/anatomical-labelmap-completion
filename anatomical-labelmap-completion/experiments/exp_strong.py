"""Stack the winning levers: extended context (K) + elastic + label-dropout + capacity + Lovasz.
Fast ranking on honest group CV (1 seed). Both context and elastic already help (0.054->0.072)."""
import sys, os, time
import numpy as np
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src")); sys.path.insert(0, os.path.join(_ROOT, "experiments"))
import common as C, ensemble as E, decide2 as D
import volume as VOL
import torch, torch.nn.functional as F
from exp_context import UNetK, build_vocab
from exp_elastic import rand_grid, warp_labels
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def lovasz_grad(gt_sorted):
    p = len(gt_sorted); gts = gt_sorted.sum()
    inter = gts - gt_sorted.float().cumsum(0); union = gts + (1 - gt_sorted).float().cumsum(0)
    jacc = 1. - inter / union
    if p > 1:
        jacc[1:p] = jacc[1:p] - jacc[0:-1]
    return jacc


def lovasz_softmax(prob, lab, mask, classes=range(1, 18)):
    # prob (B,C,H,W) softmax; lab (B,H,W); mask (B,H,W) bool -> mean Lovasz over target classes present
    losses = []
    B, Cc, H, W = prob.shape
    pf = prob.permute(0, 2, 3, 1).reshape(-1, Cc)[mask.reshape(-1)]
    lf = lab.reshape(-1)[mask.reshape(-1)]
    for c in classes:
        fg = (lf == c).float()
        if fg.sum() == 0:
            continue
        err = (fg - pf[:, c]).abs()
        order = torch.argsort(err, descending=True)
        losses.append(torch.dot(err[order], lovasz_grad(fg[order])))
    if not losses:
        return prob.sum() * 0.0
    return torch.stack(losses).mean()


def run(K=7, base=48, elastic_a=3.5, res=4, ldrop=0.0, epochs=120, lovasz=False, seed=0,
        p=0.30, name=None, save=False):
    lut, vocab = build_vocab()
    tr = C.load_split("train"); T = tr["T"]; Tidx = tr["Tidx"].astype(np.int64)
    ctx, _, _ = VOL.extended_context(tr["V"], K); idxA = lut[ctx.astype(np.int64)]; r = (K - 1) // 2
    N = len(idxA); bg = int(lut[0])
    oof = np.zeros((N, 18, 32, 32), np.float32); w = torch.ones(18, device=DEV); w[0] = 0.10
    t0 = time.time()
    for tri, vai in C.make_group_folds(5, 42):
        torch.manual_seed(seed); np.random.seed(seed)
        gen = torch.Generator(device=DEV); gen.manual_seed(seed)
        net = UNetK(vocab, K, base=base, p=p).to(DEV)
        opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=5e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        Xi = torch.tensor(idxA[tri], device=DEV); Yi = torch.tensor(Tidx[tri], device=DEV)
        n = len(tri); bs = 128
        for ep in range(epochs):
            net.train(); perm = torch.randperm(n, device=DEV)
            for s in range(0, n, bs):
                b = perm[s:s + bs]; xb = Xi[b]; yb = Yi[b]
                if elastic_a > 0:
                    grid = rand_grid(len(b), 32, 32, elastic_a, res, DEV, gen)
                    xb = warp_labels(xb, grid); yb = warp_labels(yb[:, None], grid)[:, 0]
                if ldrop > 0:
                    m = (torch.rand(xb.shape, device=DEV, generator=gen) < ldrop) & (xb != bg)
                    xb = torch.where(m, torch.full_like(xb, bg), xb)
                cz = (xb[:, r:r + 1] == bg).float()
                logit = net(xb, cz); prob = logit.softmax(1)
                mm = cz[:, 0]
                ce = (F.cross_entropy(logit, yb, weight=w, reduction="none") * mm).sum() / mm.sum().clamp_min(1)
                oh = F.one_hot(yb, 18).permute(0, 3, 1, 2).float(); pp = prob[:, 1:] * cz; gg = oh[:, 1:] * cz
                dice = 1 - ((2 * (pp * gg).sum((0, 2, 3)) + 1) / ((pp + gg).sum((0, 2, 3)) + 1)).mean()
                loss = ce + dice
                if lovasz:
                    loss = loss + lovasz_softmax(prob, yb, mm.bool())
                opt.zero_grad(); loss.backward(); opt.step()
            sch.step()
        net.eval()
        with torch.no_grad():
            Xv = torch.tensor(idxA[vai], device=DEV); Zv = (Xv[:, r:r + 1] == bg).float()
            oof[vai] = net(Xv, Zv).softmax(1).cpu().numpy()
    s, th, ma = D.tune_decision(oof, tr["V"], T)
    print(f"K={K} base={base} el={elastic_a} ldrop={ldrop} lov={int(lovasz)} ep={epochs}: "
          f"group-CV {s:.4f} @ thr={th}  ({time.time()-t0:.0f}s)", flush=True)
    if save and name:
        np.save(os.path.join(C.ARTIFACT_DIR, f"{name}_oof.npy"), oof)
    return s


if __name__ == "__main__":
    run(K=9, base=48, elastic_a=3.5, epochs=140, save=True, name="ctx9")
    run(K=11, base=48, elastic_a=3.5, epochs=140, save=True, name="ctx11")
    run(K=13, base=48, elastic_a=3.5, epochs=140, save=True, name="ctx13")
    run(K=9, base=48, elastic_a=2.5, epochs=140)
    run(K=9, base=36, elastic_a=3.5, epochs=140)                 # smaller (more reg)
