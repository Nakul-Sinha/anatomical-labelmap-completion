"""Deeper U-Net (3 downsamples -> near-global receptive field) + optional bottleneck self-attention,
to test whether long-range spatial reasoning improves target localization. Group CV, K=11, elastic."""
import sys, os, time
import numpy as np
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src")); sys.path.insert(0, os.path.join(_ROOT, "experiments"))
import common as C, decide2 as D, postprocess as P
import volume as VOL
import torch, torch.nn as nn, torch.nn.functional as F
from exp_context import build_vocab
from exp_elastic import rand_grid, warp_labels
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def blk(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.GroupNorm(8, o), nn.SiLU(),
                         nn.Conv2d(o, o, 3, padding=1), nn.GroupNorm(8, o), nn.SiLU())


class DeepUNet(nn.Module):
    def __init__(self, vocab, K, emb=16, base=40, p=0.30, attn=True):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb, padding_idx=0)
        cin = K * emb + 1
        self.e0 = blk(cin, base); self.e1 = blk(base, base * 2); self.e2 = blk(base * 2, base * 4)
        self.e3 = blk(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)
        self.attn = attn
        if attn:
            self.pos = nn.Parameter(torch.zeros(1, 16, base * 8))
            self.mha = nn.MultiheadAttention(base * 8, 4, batch_first=True)
            self.ln = nn.LayerNorm(base * 8)
        self.u2 = nn.ConvTranspose2d(base * 8, base * 4, 2, 2); self.d2 = blk(base * 8, base * 4)
        self.u1 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2); self.d1 = blk(base * 4, base * 2)
        self.u0 = nn.ConvTranspose2d(base * 2, base, 2, 2); self.d0 = blk(base * 2, base)
        self.drop = nn.Dropout2d(p); self.head = nn.Conv2d(base, 18, 1)

    def forward(self, idx, cz):
        B, K, H, W = idx.shape
        e = self.emb(idx).permute(0, 1, 4, 2, 3).reshape(B, -1, H, W)
        x = torch.cat([e, cz], 1)
        x0 = self.e0(x); x1 = self.e1(self.pool(x0)); x2 = self.e2(self.pool(x1)); x3 = self.e3(self.pool(x2))
        if self.attn:
            b, c, h, w = x3.shape
            t = x3.flatten(2).transpose(1, 2) + self.pos
            a, _ = self.mha(t, t, t)
            t = self.ln(t + a)
            x3 = t.transpose(1, 2).reshape(b, c, h, w)
        d2 = self.d2(torch.cat([self.u2(x3), x2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), x1], 1))
        d0 = self.d0(torch.cat([self.u0(d1), x0], 1))
        return self.head(self.drop(d0))


def run(K=11, base=40, attn=True, elastic_a=3.0, res=4, epochs=150, seed=0, save=False, name=None):
    lut, vocab = build_vocab()
    tr = C.load_split("train"); T = tr["T"]; Tidx = tr["Tidx"].astype(np.int64)
    ctx, _, _ = VOL.extended_context(tr["V"], K); idxA = lut[ctx.astype(np.int64)]; r = (K - 1) // 2
    bg = int(lut[0]); N = len(idxA)
    oof = np.zeros((N, 18, 32, 32), np.float32); w = torch.ones(18, device=DEV); w[0] = 0.10
    t0 = time.time()
    for tri, vai in C.make_group_folds(5, 42):
        torch.manual_seed(seed); np.random.seed(seed)
        gen = torch.Generator(device=DEV); gen.manual_seed(seed)
        net = DeepUNet(vocab, K, base=base, attn=attn).to(DEV)
        opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=5e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        Xi = torch.tensor(idxA[tri], device=DEV); Yi = torch.tensor(Tidx[tri], device=DEV)
        n = len(tri); bs = 128
        for ep in range(epochs):
            net.train(); perm = torch.randperm(n, device=DEV)
            for s in range(0, n, bs):
                b = perm[s:s + bs]; xb = Xi[b]; yb = Yi[b]
                if elastic_a > 0:
                    g = rand_grid(len(b), 32, 32, elastic_a, res, DEV, gen)
                    xb = warp_labels(xb, g); yb = warp_labels(yb[:, None], g)[:, 0]
                cz = (xb[:, r:r + 1] == bg).float()
                lg = net(xb, cz); prob = lg.softmax(1); mm = cz[:, 0]
                ce = (F.cross_entropy(lg, yb, weight=w, reduction="none") * mm).sum() / mm.sum().clamp_min(1)
                oh = F.one_hot(yb, 18).permute(0, 3, 1, 2).float(); pp = prob[:, 1:] * cz; gg = oh[:, 1:] * cz
                dice = 1 - ((2 * (pp * gg).sum((0, 2, 3)) + 1) / ((pp + gg).sum((0, 2, 3)) + 1)).mean()
                (ce + dice).backward(); opt.step(); opt.zero_grad()
            sch.step()
        net.eval()
        with torch.no_grad():
            Xv = torch.tensor(idxA[vai], device=DEV)
            oof[vai] = net(Xv, (Xv[:, r:r + 1] == bg).float()).softmax(1).cpu().numpy()
    b3 = P.smooth3d(oof / (oof.sum(1, keepdims=True) + 1e-9), tr["V"], 0.7, 3)
    s = D.tune_decision(b3, tr["V"], T)[0]
    print(f"deep K={K} base={base} attn={attn} ep={epochs}: +3D group-CV {s:.4f}  ({time.time()-t0:.0f}s)", flush=True)
    if save and name:
        np.save(os.path.join(C.ARTIFACT_DIR, f"{name}_oof.npy"), oof)
    return s


if __name__ == "__main__":
    run(K=11, base=48, attn=False, epochs=160, save=True, name="deep_b48")
    run(K=11, base=56, attn=False, epochs=160)
    run(K=13, base=48, attn=False, epochs=160)
    run(K=9, base=48, attn=False, epochs=160)
