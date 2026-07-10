"""Add explicit geometric input features to the deep net: distance-to-anatomy (EDT), ray-cast
nearest visible label in 4 directions (embedded), and interior-hole enclosure flag. Targets the
localization bottleneck (the model currently learns anatomy-relative geometry only implicitly).
Group CV, K=9, deep net."""
import sys, os, time
import numpy as np
from scipy import ndimage
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src")); sys.path.insert(0, os.path.join(_ROOT, "experiments"))
import common as C, decide2 as D, postprocess as P
import volume as VOL
import torch, torch.nn as nn, torch.nn.functional as F
from exp_context import build_vocab
from exp_elastic import rand_grid, warp_labels
from exp_deep import blk
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def geom_feats(cen):
    """cen: (N,32,32) center visible labels -> ray4 (N,4,32,32) label ids, dist (N,32,32), enc (N,32,32)."""
    N = len(cen)
    ray = np.zeros((N, 4, 32, 32), np.int64); dist = np.zeros((N, 32, 32), np.float32); enc = np.zeros((N, 32, 32), np.float32)
    for k in range(N):
        c = cen[k]
        up = np.zeros((32, 32), np.int64); dn = np.zeros((32, 32), np.int64); lf = np.zeros((32, 32), np.int64); rt = np.zeros((32, 32), np.int64)
        for x in range(32):
            last = 0
            for y in range(32):
                up[y, x] = last
                if c[y, x] != 0: last = c[y, x]
            last = 0
            for y in range(31, -1, -1):
                dn[y, x] = last
                if c[y, x] != 0: last = c[y, x]
        for y in range(32):
            last = 0
            for x in range(32):
                lf[y, x] = last
                if c[y, x] != 0: last = c[y, x]
            last = 0
            for x in range(31, -1, -1):
                rt[y, x] = last
                if c[y, x] != 0: last = c[y, x]
        ray[k] = np.stack([up, dn, lf, rt])
        dist[k] = ndimage.distance_transform_edt(c == 0).astype(np.float32) / 10.0
        z = (c == 0); lab, _ = ndimage.label(z)
        bl = set(lab[0]).union(lab[-1]).union(lab[:, 0]).union(lab[:, -1]); bl.discard(0)
        enc[k] = (z & ~np.isin(lab, list(bl))).astype(np.float32)
    return ray, dist, enc


class FeatUNet(nn.Module):
    def __init__(self, vocab, K, emb=16, base=40, p=0.30, use_feat=True):
        super().__init__()
        self.use_feat = use_feat
        self.emb = nn.Embedding(vocab, emb, padding_idx=0)
        cin = K * emb + 1 + (4 * emb + 2 if use_feat else 0)
        self.e0 = blk(cin, base); self.e1 = blk(base, base * 2); self.e2 = blk(base * 2, base * 4); self.e3 = blk(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)
        self.u2 = nn.ConvTranspose2d(base * 8, base * 4, 2, 2); self.d2 = blk(base * 8, base * 4)
        self.u1 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2); self.d1 = blk(base * 4, base * 2)
        self.u0 = nn.ConvTranspose2d(base * 2, base, 2, 2); self.d0 = blk(base * 2, base)
        self.drop = nn.Dropout2d(p); self.head = nn.Conv2d(base, 18, 1)

    def forward(self, idx, cz, ray=None, dist=None, enc=None):
        B, K, H, W = idx.shape
        e = self.emb(idx).permute(0, 1, 4, 2, 3).reshape(B, -1, H, W)
        feats = [e, cz]
        if self.use_feat:
            re = self.emb(ray).permute(0, 1, 4, 2, 3).reshape(B, -1, H, W)
            feats += [re, dist, enc]
        x = torch.cat(feats, 1)
        x0 = self.e0(x); x1 = self.e1(self.pool(x0)); x2 = self.e2(self.pool(x1)); x3 = self.e3(self.pool(x2))
        d2 = self.d2(torch.cat([self.u2(x3), x2], 1)); d1 = self.d1(torch.cat([self.u1(d2), x1], 1)); d0 = self.d0(torch.cat([self.u0(d1), x0], 1))
        return self.head(self.drop(d0))


def run(K=9, base=40, use_feat=True, epochs=160, seed=0):
    lut, vocab = build_vocab(); bg = int(lut[0])
    tr = C.load_split("train"); T = tr["T"]; Tidx = tr["Tidx"].astype(np.int64)
    ctx, _, _ = VOL.extended_context(tr["V"], K); idxA = lut[ctx.astype(np.int64)]; r = (K - 1) // 2
    ray, dist, enc = geom_feats(tr["V"][:, 1]); rayI = lut[ray]
    N = len(idxA); oof = np.zeros((N, 18, 32, 32), np.float32); w = torch.ones(18, device=DEV); w[0] = 0.10
    t0 = time.time()
    for tri, vai in C.make_group_folds(5, 42):
        torch.manual_seed(seed); np.random.seed(seed); gen = torch.Generator(device=DEV); gen.manual_seed(seed)
        net = FeatUNet(vocab, K, base=base, use_feat=use_feat).to(DEV)
        opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=5e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        Xi = torch.tensor(idxA[tri], device=DEV); Yi = torch.tensor(Tidx[tri], device=DEV)
        Ri = torch.tensor(rayI[tri], device=DEV); Di = torch.tensor(dist[tri, None], device=DEV); Ei = torch.tensor(enc[tri, None], device=DEV)
        n = len(tri); bs = 128
        for ep in range(epochs):
            net.train(); perm = torch.randperm(n, device=DEV)
            for s in range(0, n, bs):
                b = perm[s:s + bs]; xb = Xi[b]; yb = Yi[b]; rb = Ri[b]; db = Di[b]; eb = Ei[b]
                if True:
                    g = rand_grid(len(b), 32, 32, 3.0, 4, DEV, gen)
                    xb = warp_labels(xb, g); yb = warp_labels(yb[:, None], g)[:, 0]
                    rb = warp_labels(rb, g); db = F.grid_sample(db, g, mode="bilinear", padding_mode="border", align_corners=True); eb = warp_labels(eb, g).float()
                cz = (xb[:, r:r + 1] == bg).float(); mm = cz[:, 0]
                lg = net(xb, cz, rb, db, eb); prob = lg.softmax(1)
                ce = (F.cross_entropy(lg, yb, weight=w, reduction="none") * mm).sum() / mm.sum().clamp_min(1)
                oh = F.one_hot(yb, 18).permute(0, 3, 1, 2).float(); pp = prob[:, 1:] * cz; gg = oh[:, 1:] * cz
                dice = 1 - ((2 * (pp * gg).sum((0, 2, 3)) + 1) / ((pp + gg).sum((0, 2, 3)) + 1)).mean()
                (ce + dice).backward(); opt.step(); opt.zero_grad()
            sch.step()
        net.eval()
        with torch.no_grad():
            Xv = torch.tensor(idxA[vai], device=DEV); cz = (Xv[:, r:r + 1] == bg).float()
            Rv = torch.tensor(rayI[vai], device=DEV); Dv = torch.tensor(dist[vai, None], device=DEV); Ev = torch.tensor(enc[vai, None], device=DEV)
            oof[vai] = net(Xv, cz, Rv, Dv, Ev).softmax(1).cpu().numpy()
    s = D.tune_decision(P.smooth3d(oof / (oof.sum(1, keepdims=True) + 1e-9), tr["V"], 0.7, 3), tr["V"], T)[0]
    print(f"K={K} use_feat={use_feat}: +3D group-CV {s:.4f}  ({time.time()-t0:.0f}s)", flush=True)
    return s


if __name__ == "__main__":
    run(K=9, use_feat=False, epochs=160)
    run(K=9, use_feat=True, epochs=160)
