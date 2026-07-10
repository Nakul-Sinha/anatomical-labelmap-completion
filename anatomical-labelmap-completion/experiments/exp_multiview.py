"""
Multi-view: model predicts the target for ALL K context slices (deep 3D supervision). At inference
each slice is predicted by every row whose window overlaps it, and those views are averaged -- a
genuine per-slice ensemble. Fold-safe: overlapping rows share the row's volume, hence its fold.
"""
import sys, os, time
import numpy as np
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src")); sys.path.insert(0, os.path.join(_ROOT, "experiments"))
import common as C, decide2 as D, postprocess as POST
import volume as VOL
import torch, torch.nn as nn, torch.nn.functional as F
from exp_context import build_vocab
from exp_elastic import rand_grid, warp_labels
DEV = "cuda" if torch.cuda.is_available() else "cpu"


class MVUNet(nn.Module):
    def __init__(self, vocab, K, emb=16, base=48, p=0.30):
        super().__init__()
        self.K = K
        self.emb = nn.Embedding(vocab, emb, padding_idx=0)
        cin = K * emb + 1

        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.GroupNorm(8, o), nn.SiLU(),
                                 nn.Conv2d(o, o, 3, padding=1), nn.GroupNorm(8, o), nn.SiLU())
        self.e0 = blk(cin, base); self.e1 = blk(base, base * 2); self.e2 = blk(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)
        self.u1 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2); self.d1 = blk(base * 4, base * 2)
        self.u2 = nn.ConvTranspose2d(base * 2, base, 2, 2); self.d0 = blk(base * 2, base)
        self.drop = nn.Dropout2d(p); self.head = nn.Conv2d(base, K * 18, 1)

    def forward(self, idx, cz):
        B, K, H, W = idx.shape
        e = self.emb(idx).permute(0, 1, 4, 2, 3).reshape(B, -1, H, W)
        x = torch.cat([e, cz], 1)
        x0 = self.e0(x); x1 = self.e1(self.pool(x0)); x2 = self.e2(self.pool(x1))
        d1 = self.d1(torch.cat([self.u1(x2), x1], 1)); d0 = self.d0(torch.cat([self.u2(d1), x0], 1))
        return self.head(self.drop(d0)).reshape(B, self.K, 18, H, W)


def multi_targets(V, Tidx, K):
    nxt, prv = VOL.build_chain(V); r = (K - 1) // 2; N = len(V)
    tgt = np.zeros((N, K, 32, 32), np.int64); valid = np.zeros((N, K), bool)
    for i in range(N):
        tgt[i, r] = Tidx[i]; valid[i, r] = True
        cur = i
        for d in range(1, r + 1):
            cur = nxt[cur] if cur >= 0 else -1
            if cur < 0:
                break
            tgt[i, r + d] = Tidx[cur]; valid[i, r + d] = True
        cur = i
        for d in range(1, r + 1):
            cur = prv[cur] if cur >= 0 else -1
            if cur < 0:
                break
            tgt[i, r - d] = Tidx[cur]; valid[i, r - d] = True
    return tgt, valid, nxt, prv


def reconcile(pred_KC, V, K):
    nxt, prv = VOL.build_chain(V); r = (K - 1) // 2; N = len(V)
    out = np.zeros((N, 18, 32, 32), np.float32)
    for i in range(N):
        acc = pred_KC[i, r].astype(np.float64).copy(); cnt = 1
        cur = i
        for d in range(1, r + 1):
            cur = nxt[cur] if cur >= 0 else -1
            if cur < 0:
                break
            acc += pred_KC[cur, r - d]; cnt += 1
        cur = i
        for d in range(1, r + 1):
            cur = prv[cur] if cur >= 0 else -1
            if cur < 0:
                break
            acc += pred_KC[cur, r + d]; cnt += 1
        out[i] = (acc / cnt).astype(np.float32)
    return out


def run(K=9, base=48, elastic_a=3.0, res=4, epochs=160, seed=0, save=False, name="mv9"):
    lut, vocab = build_vocab()
    tr = C.load_split("train"); T = tr["T"]; Tidx = tr["Tidx"].astype(np.int64)
    ctx, _, _ = VOL.extended_context(tr["V"], K); idxA = lut[ctx.astype(np.int64)]; r = (K - 1) // 2
    tgt, valid, _, _ = multi_targets(tr["V"], Tidx, K)
    bg = int(lut[0]); N = len(idxA)
    predKC = np.zeros((N, K, 18, 32, 32), np.float32)
    w = torch.ones(18, device=DEV); w[0] = 0.10
    t0 = time.time()
    for tri, vai in C.make_group_folds(5, 42):
        torch.manual_seed(seed); np.random.seed(seed)
        gen = torch.Generator(device=DEV); gen.manual_seed(seed)
        net = MVUNet(vocab, K, base=base).to(DEV)
        opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=5e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        Xi = torch.tensor(idxA[tri], device=DEV); Yi = torch.tensor(tgt[tri], device=DEV)
        Vi = torch.tensor(valid[tri], device=DEV)
        n = len(tri); bs = 96
        for ep in range(epochs):
            net.train(); perm = torch.randperm(n, device=DEV)
            for s in range(0, n, bs):
                b = perm[s:s + bs]; xb = Xi[b]; yb = Yi[b]; vb = Vi[b]
                if elastic_a > 0:
                    grid = rand_grid(len(b), 32, 32, elastic_a, res, DEV, gen)
                    xb = warp_labels(xb, grid)
                    yb = warp_labels(yb.reshape(-1, 1, 32, 32).float(), grid.repeat_interleave(K, 0)).reshape(len(b), K, 32, 32).long()
                out = net(xb, (xb[:, r:r + 1] == bg).float())        # (B,K,18,H,W)
                loss = 0.0
                for kk in range(K):
                    m = vb[:, kk]
                    if m.sum() == 0:
                        continue
                    cz = (xb[m, kk:kk + 1] == bg).float()
                    lg = out[m, kk]; yy = yb[m, kk]; mm = cz[:, 0]
                    ce = (F.cross_entropy(lg, yy, weight=w, reduction="none") * mm).sum() / mm.sum().clamp_min(1)
                    prob = lg.softmax(1); oh = F.one_hot(yy, 18).permute(0, 3, 1, 2).float()
                    pp = prob[:, 1:] * cz; gg = oh[:, 1:] * cz
                    dice = 1 - ((2 * (pp * gg).sum((0, 2, 3)) + 1) / ((pp + gg).sum((0, 2, 3)) + 1)).mean()
                    loss = loss + ce + dice
                (loss / K).backward(); opt.step(); opt.zero_grad()
            sch.step()
        net.eval()
        with torch.no_grad():
            for s in range(0, len(vai), 128):
                vi = vai[s:s + 128]
                Xv = torch.tensor(idxA[vi], device=DEV)
                predKC[vi] = net(Xv, (Xv[:, r:r + 1] == bg).float()).softmax(2).cpu().numpy()
    # center-only vs reconciled
    center = predKC[:, r]
    sc = D.tune_decision(center, tr["V"], T)[0]
    rec = reconcile(predKC, tr["V"], K)
    sr = D.tune_decision(rec, tr["V"], T)[0]
    recs = POST.smooth3d(rec / (rec.sum(1, keepdims=True) + 1e-9), tr["V"], 0.7, 3)
    srs = D.tune_decision(recs, tr["V"], T)[0]
    print(f"{name} K={K}: center-only {sc:.4f} | multi-view {sr:.4f} | +3D {srs:.4f}  ({time.time()-t0:.0f}s)", flush=True)
    if save:
        np.save(os.path.join(C.ARTIFACT_DIR, f"{name}_oof.npy"), rec)
    return srs


if __name__ == "__main__":
    run(K=9, epochs=160, save=True, name="mv9")
