"""
Elastic-deformation augmentation test. D4 rotations/flips hurt (they break the atlas's absolute
orientation), but SMOOTH elastic warps simulate cross-subject anatomical variation while preserving
orientation -- the standard lever for cross-subject medical-image generalization, never tested here.
Applied jointly (nearest) to the K input label slices and the target. Compared on honest group CV.
"""
import sys, os, time
import numpy as np
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "experiments"))
import common as C, ensemble as E, decide2 as D
import volume as VOL
import torch, torch.nn.functional as F
from exp_context import UNetK, build_vocab, loss_fn
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def rand_grid(B, H, W, alpha_px, res, device, gen):
    """Smooth random sampling grid (identity + low-freq displacement of ~alpha_px pixels)."""
    small = torch.randn(B, 2, res, res, device=device, generator=gen)
    flow = F.interpolate(small, size=(H, W), mode="bicubic", align_corners=True)
    flow = flow / flow.flatten(2).abs().amax(2, keepdim=True).clamp_min(1e-6).unsqueeze(-1) * alpha_px
    ys, xs = torch.meshgrid(torch.linspace(-1, 1, H, device=device),
                            torch.linspace(-1, 1, W, device=device), indexing="ij")
    base = torch.stack([xs, ys], -1).unsqueeze(0).expand(B, -1, -1, -1)
    disp = flow.permute(0, 2, 3, 1) * torch.tensor([2 / (W - 1), 2 / (H - 1)], device=device)
    return base + disp


def warp_labels(x, grid):  # x: (B,C,H,W) long -> nearest sampled long
    xf = x.float()
    out = F.grid_sample(xf, grid, mode="nearest", padding_mode="zeros", align_corners=True)
    return out.round().long()


def run(K=7, use_elastic=True, alpha_px=2.5, res=5, seed=0, epochs=140, name=None, save=False):
    lut, vocab = build_vocab()
    tr = C.load_split("train"); te = C.load_split("test"); T = tr["T"]; Tidx = tr["Tidx"].astype(np.int64)
    ctx, _, _ = VOL.extended_context(tr["V"], K)
    idxA = lut[ctx.astype(np.int64)]; r = (K - 1) // 2
    N = len(idxA)
    oof = np.zeros((N, 18, 32, 32), np.float32)
    w = torch.ones(18, device=DEV); w[0] = 0.10
    t0 = time.time()
    for tri, vai in C.make_group_folds(5, 42):
        torch.manual_seed(seed); np.random.seed(seed)
        gen = torch.Generator(device=DEV); gen.manual_seed(seed)
        net = UNetK(vocab, K).to(DEV)
        opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=5e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        Xi = torch.tensor(idxA[tri], device=DEV); Yi = torch.tensor(Tidx[tri], device=DEV)
        n = len(tri); bs = 128
        for ep in range(epochs):
            net.train(); perm = torch.randperm(n, device=DEV)
            for s in range(0, n, bs):
                b = perm[s:s + bs]
                xb = Xi[b]; yb = Yi[b]
                if use_elastic:
                    grid = rand_grid(len(b), 32, 32, alpha_px, res, DEV, gen)
                    xb = warp_labels(xb, grid)
                    yb = warp_labels(yb[:, None], grid)[:, 0]
                cz = (xb[:, r:r + 1] == lut[0]).float()
                loss = loss_fn(net(xb, cz), yb, cz, w)
                opt.zero_grad(); loss.backward(); opt.step()
            sch.step()
        net.eval()
        with torch.no_grad():
            Xv = torch.tensor(idxA[vai], device=DEV); Zv = (Xv[:, r:r + 1] == lut[0]).float()
            oof[vai] = net(Xv, Zv).softmax(1).cpu().numpy()
    s, th, ma = D.tune_decision(oof, tr["V"], T)
    tag = f"K={K} elastic={use_elastic} a={alpha_px} res={res}"
    print(f"{tag}: group-CV {s:.4f} @ thr={th}  ({time.time()-t0:.0f}s)", flush=True)
    if save and name:
        np.save(os.path.join(C.ARTIFACT_DIR, f"{name}_oof.npy"), oof)
    return s


if __name__ == "__main__":
    EP = 90
    run(K=3, use_elastic=False, epochs=EP)                        # baseline (~cnn_unet 0.064)
    run(K=7, use_elastic=False, epochs=EP)                        # context effect
    run(K=3, use_elastic=True, alpha_px=3.5, res=4, epochs=EP)    # elastic effect
    run(K=7, use_elastic=True, alpha_px=3.5, res=4, epochs=EP)    # context + elastic
    run(K=7, use_elastic=True, alpha_px=6.0, res=4, epochs=EP)    # stronger elastic
