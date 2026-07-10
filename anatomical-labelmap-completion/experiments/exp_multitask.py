"""
Multi-task: segmentation + a dedicated per-label PRESENCE head (which of 17 labels are in the
center target). The label-set is the bottleneck (oracle set -> +0.026). Segmentation mass can't
separate spurious vs missed labels; a global presence classifier should predict the set better.
Decision: predicted-present set filters the segmentation placement.
"""
import sys, os, time
import numpy as np
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src")); sys.path.insert(0, os.path.join(_ROOT, "experiments"))
import common as C, decide2 as D
import volume as VOL
import torch, torch.nn as nn, torch.nn.functional as F
from exp_context import build_vocab
from exp_elastic import rand_grid, warp_labels
DEV = "cuda" if torch.cuda.is_available() else "cpu"


class MTUNet(nn.Module):
    def __init__(self, vocab, K, emb=16, base=48, p=0.30):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb, padding_idx=0)
        cin = K * emb + 1

        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.GroupNorm(8, o), nn.SiLU(),
                                 nn.Conv2d(o, o, 3, padding=1), nn.GroupNorm(8, o), nn.SiLU())
        self.e0 = blk(cin, base); self.e1 = blk(base, base * 2); self.e2 = blk(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)
        self.u1 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2); self.d1 = blk(base * 4, base * 2)
        self.u2 = nn.ConvTranspose2d(base * 2, base, 2, 2); self.d0 = blk(base * 2, base)
        self.drop = nn.Dropout2d(p); self.head = nn.Conv2d(base, 18, 1)
        self.pres = nn.Sequential(nn.Linear(base * 8, base * 2), nn.SiLU(), nn.Dropout(0.3),
                                  nn.Linear(base * 2, 17))

    def forward(self, idx, cz):
        B, K, H, W = idx.shape
        e = self.emb(idx).permute(0, 1, 4, 2, 3).reshape(B, -1, H, W)
        x = torch.cat([e, cz], 1)
        x0 = self.e0(x); x1 = self.e1(self.pool(x0)); x2 = self.e2(self.pool(x1))
        pooled = torch.cat([x2.mean((2, 3)), x2.amax((2, 3))], 1)     # (B, base*8)
        pres = self.pres(pooled)
        d1 = self.d1(torch.cat([self.u1(x2), x1], 1)); d0 = self.d0(torch.cat([self.u2(d1), x0], 1))
        return self.head(self.drop(d0)), pres


def run(K=9, base=48, elastic_a=3.0, res=4, epochs=140, pres_w=0.5, seed=0, save=False, name="mt9"):
    lut, vocab = build_vocab()
    tr = C.load_split("train"); T = tr["T"]; Tidx = tr["Tidx"].astype(np.int64)
    ctx, _, _ = VOL.extended_context(tr["V"], K); idxA = lut[ctx.astype(np.int64)]; r = (K - 1) // 2
    bg = int(lut[0]); N = len(idxA)
    pres_true = np.zeros((N, 17), np.float32)
    for i in range(N):
        for c in np.unique(Tidx[i][Tidx[i] > 0]):
            pres_true[i, c - 1] = 1.0
    seg_oof = np.zeros((N, 18, 32, 32), np.float32); pres_oof = np.zeros((N, 17), np.float32)
    w = torch.ones(18, device=DEV); w[0] = 0.10
    t0 = time.time()
    for tri, vai in C.make_group_folds(5, 42):
        torch.manual_seed(seed); np.random.seed(seed)
        gen = torch.Generator(device=DEV); gen.manual_seed(seed)
        net = MTUNet(vocab, K, base=base).to(DEV)
        opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=5e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        Xi = torch.tensor(idxA[tri], device=DEV); Yi = torch.tensor(Tidx[tri], device=DEV)
        Pi = torch.tensor(pres_true[tri], device=DEV)
        n = len(tri); bs = 128
        for ep in range(epochs):
            net.train(); perm = torch.randperm(n, device=DEV)
            for s in range(0, n, bs):
                b = perm[s:s + bs]; xb = Xi[b]; yb = Yi[b]; pb = Pi[b]
                if elastic_a > 0:
                    grid = rand_grid(len(b), 32, 32, elastic_a, res, DEV, gen)
                    xb = warp_labels(xb, grid); yb = warp_labels(yb[:, None], grid)[:, 0]
                cz = (xb[:, r:r + 1] == bg).float()
                seg, pres = net(xb, cz); prob = seg.softmax(1); mm = cz[:, 0]
                ce = (F.cross_entropy(seg, yb, weight=w, reduction="none") * mm).sum() / mm.sum().clamp_min(1)
                oh = F.one_hot(yb, 18).permute(0, 3, 1, 2).float(); pp = prob[:, 1:] * cz; gg = oh[:, 1:] * cz
                dice = 1 - ((2 * (pp * gg).sum((0, 2, 3)) + 1) / ((pp + gg).sum((0, 2, 3)) + 1)).mean()
                bce = F.binary_cross_entropy_with_logits(pres, pb)
                loss = ce + dice + pres_w * bce
                opt.zero_grad(); loss.backward(); opt.step()
            sch.step()
        net.eval()
        with torch.no_grad():
            Xv = torch.tensor(idxA[vai], device=DEV); Zv = (Xv[:, r:r + 1] == bg).float()
            seg, pres = net(Xv, Zv)
            seg_oof[vai] = seg.softmax(1).cpu().numpy(); pres_oof[vai] = torch.sigmoid(pres).cpu().numpy()

    # baseline (seg only)
    sb, thb, mab = D.tune_decision(seg_oof, tr["V"], T)
    # presence-filtered decision
    CZ = (tr["V"][:, 1] == 0)

    def decide_pres(tau_p, sthr):
        preds = []
        for i in range(N):
            inc = pres_oof[i] >= tau_p
            fg = seg_oof[i, 1:]
            fgm = np.where(inc[:, None, None], fg, -1.0)
            best = fgm.argmax(0); bestp = np.take_along_axis(fg, best[None], 0)[0]
            idx = np.where((bestp >= sthr) & CZ[i] & inc[best], best + 1, 0)
            preds.append(C.idx_to_labels(idx))
        return C.score_rows(preds, [T[i] for i in range(N)])
    best = (-1, None, None)
    for tp in np.round(np.arange(0.15, 0.7, 0.05), 2):
        for st in np.round(np.arange(0.05, 0.45, 0.05), 2):
            s = decide_pres(tp, st)
            if s > best[0]:
                best = (s, tp, st)
    # presence set quality at best tau
    inc = pres_oof >= best[1]
    tp_ = (inc & (pres_true > 0)).sum(); fp_ = (inc & (pres_true == 0)).sum(); fn_ = (~inc & (pres_true > 0)).sum()
    prec = tp_ / max(tp_ + fp_, 1); rec = tp_ / max(tp_ + fn_, 1)
    print(f"{name} K={K}: seg-only {sb:.4f} | +presence-head {best[0]:.4f} (tau_p={best[1]},sthr={best[2]}) "
          f"| set prec={prec:.3f} rec={rec:.3f}  ({time.time()-t0:.0f}s)", flush=True)
    if save:
        np.save(os.path.join(C.ARTIFACT_DIR, f"{name}_seg_oof.npy"), seg_oof)
        np.save(os.path.join(C.ARTIFACT_DIR, f"{name}_pres_oof.npy"), pres_oof)
    return best[0]


if __name__ == "__main__":
    run(K=9, pres_w=0.5, epochs=140)
    run(K=9, pres_w=1.0, epochs=140)
    run(K=11, pres_w=1.0, epochs=140)
