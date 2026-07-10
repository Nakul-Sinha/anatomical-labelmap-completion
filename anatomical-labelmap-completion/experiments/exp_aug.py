"""Heavy realistic augmentation (small affine + elastic) + longer training. Small-angle affine is
anatomical variation (unlike 90-deg D4 which breaks atlas orientation). Group-CV, K=9."""
import sys, os, time, math
import numpy as np
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src")); sys.path.insert(0, os.path.join(_ROOT, "experiments"))
import common as C, decide2 as D
import volume as VOL
import torch, torch.nn.functional as F
from exp_context import UNetK, build_vocab
from exp_elastic import warp_labels
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def aug_grid(B, H, W, rot, sc, sh, tr_px, el_a, res, device, gen):
    ang = (torch.rand(B, device=device, generator=gen) * 2 - 1) * (rot * math.pi / 180)
    s = 1 + (torch.rand(B, device=device, generator=gen) * 2 - 1) * sc
    shx = (torch.rand(B, device=device, generator=gen) * 2 - 1) * sh
    tx = (torch.rand(B, device=device, generator=gen) * 2 - 1) * (tr_px * 2 / W)
    ty = (torch.rand(B, device=device, generator=gen) * 2 - 1) * (tr_px * 2 / H)
    cos, sin = torch.cos(ang), torch.sin(ang)
    theta = torch.zeros(B, 2, 3, device=device)
    theta[:, 0, 0] = cos * s; theta[:, 0, 1] = -sin * s + shx; theta[:, 0, 2] = tx
    theta[:, 1, 0] = sin * s; theta[:, 1, 1] = cos * s; theta[:, 1, 2] = ty
    grid = F.affine_grid(theta, (B, 1, H, W), align_corners=True)
    if el_a > 0:
        small = torch.randn(B, 2, res, res, device=device, generator=gen)
        flow = F.interpolate(small, size=(H, W), mode="bicubic", align_corners=True)
        flow = flow / flow.flatten(2).abs().amax(2, keepdim=True).clamp_min(1e-6).unsqueeze(-1) * el_a
        grid = grid + flow.permute(0, 2, 3, 1) * torch.tensor([2 / (W - 1), 2 / (H - 1)], device=device)
    return grid


def run(K=9, base=48, rot=12, sc=0.12, sh=0.06, tr_px=2, el_a=3.0, res=4, epochs=200, seed=0,
        save=False, name=None):
    lut, vocab = build_vocab()
    tr = C.load_split("train"); T = tr["T"]; Tidx = tr["Tidx"].astype(np.int64)
    ctx, _, _ = VOL.extended_context(tr["V"], K); idxA = lut[ctx.astype(np.int64)]; r = (K - 1) // 2
    bg = int(lut[0]); N = len(idxA)
    oof = np.zeros((N, 18, 32, 32), np.float32); w = torch.ones(18, device=DEV); w[0] = 0.10
    t0 = time.time()
    for tri, vai in C.make_group_folds(5, 42):
        torch.manual_seed(seed); np.random.seed(seed)
        gen = torch.Generator(device=DEV); gen.manual_seed(seed)
        net = UNetK(vocab, K, base=base).to(DEV)
        opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=5e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        Xi = torch.tensor(idxA[tri], device=DEV); Yi = torch.tensor(Tidx[tri], device=DEV)
        n = len(tri); bs = 128
        for ep in range(epochs):
            net.train(); perm = torch.randperm(n, device=DEV)
            for s in range(0, n, bs):
                b = perm[s:s + bs]; xb = Xi[b]; yb = Yi[b]
                grid = aug_grid(len(b), 32, 32, rot, sc, sh, tr_px, el_a, res, DEV, gen)
                xb = warp_labels(xb, grid); yb = warp_labels(yb[:, None], grid)[:, 0]
                cz = (xb[:, r:r + 1] == bg).float()
                logit = net(xb, cz); prob = logit.softmax(1); mm = cz[:, 0]
                ce = (F.cross_entropy(logit, yb, weight=w, reduction="none") * mm).sum() / mm.sum().clamp_min(1)
                oh = F.one_hot(yb, 18).permute(0, 3, 1, 2).float(); pp = prob[:, 1:] * cz; gg = oh[:, 1:] * cz
                dice = 1 - ((2 * (pp * gg).sum((0, 2, 3)) + 1) / ((pp + gg).sum((0, 2, 3)) + 1)).mean()
                (ce + dice).backward(); opt.step(); opt.zero_grad()
            sch.step()
        net.eval()
        with torch.no_grad():
            Xv = torch.tensor(idxA[vai], device=DEV); Zv = (Xv[:, r:r + 1] == bg).float()
            oof[vai] = net(Xv, Zv).softmax(1).cpu().numpy()
    s, th, ma = D.tune_decision(oof, tr["V"], T)
    print(f"K={K} rot={rot} sc={sc} sh={sh} el={el_a} ep={epochs}: group-CV {s:.4f} @ thr={th}  ({time.time()-t0:.0f}s)", flush=True)
    if save and name:
        np.save(os.path.join(C.ARTIFACT_DIR, f"{name}_oof.npy"), oof)
    return s


if __name__ == "__main__":
    run(K=9, rot=0, sc=0, sh=0, tr_px=0, el_a=3.0, epochs=200)       # elastic only, long
    run(K=9, rot=12, sc=0.12, sh=0.06, tr_px=2, el_a=3.0, epochs=200)  # heavy affine+elastic
    run(K=9, rot=20, sc=0.18, sh=0.10, tr_px=3, el_a=4.0, epochs=200)  # heavier
