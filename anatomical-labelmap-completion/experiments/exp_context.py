"""Does wider 3D slice context help? Train a K-slice U-Net, compare K on honest group-CV."""
import sys, os, time
import numpy as np
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
import common as C, ensemble as E, decide2 as D
import volume as VOL
import torch, torch.nn as nn, torch.nn.functional as F
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def build_vocab():
    tr = C.load_split("train"); te = C.load_split("test")
    uv = np.unique(np.concatenate([tr["V"].reshape(-1), te["V"].reshape(-1)]))
    lut = np.zeros(int(uv.max()) + 1, np.int64)
    for i, l in enumerate(uv, 1):
        lut[int(l)] = i
    return lut, len(uv) + 1


class UNetK(nn.Module):
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

    def forward(self, idx, cz):
        B, K, H, W = idx.shape
        e = self.emb(idx).permute(0, 1, 4, 2, 3).reshape(B, -1, H, W)
        x = torch.cat([e, cz], 1)
        x0 = self.e0(x); x1 = self.e1(self.pool(x0)); x2 = self.e2(self.pool(x1))
        d1 = self.d1(torch.cat([self.u1(x2), x1], 1)); d0 = self.d0(torch.cat([self.u2(d1), x0], 1))
        return self.head(self.drop(d0))


def loss_fn(logit, y, cz, w):
    m = cz[:, 0]
    ce = (F.cross_entropy(logit, y, weight=w, reduction="none") * m).sum() / m.sum().clamp_min(1)
    prob = logit.softmax(1); oh = F.one_hot(y, 18).permute(0, 3, 1, 2).float()
    p = prob[:, 1:] * cz; g = oh[:, 1:] * cz
    dice = 1 - ((2 * (p * g).sum((0, 2, 3)) + 1) / ((p + g).sum((0, 2, 3)) + 1)).mean()
    return ce + dice


def run_K(K, seed=0, epochs=130):
    lut, vocab = build_vocab()
    tr = C.load_split("train"); T = tr["T"]; Tidx = tr["Tidx"].astype(np.int64)
    ctx, _, _ = VOL.extended_context(tr["V"], K)          # (N,K,32,32)
    idxA = lut[ctx.astype(np.int64)]
    r = (K - 1) // 2
    cz = (ctx[:, r] == 0).astype(np.float32)
    N = len(idxA)
    oof = np.zeros((N, 18, 32, 32), np.float32)
    w = torch.ones(18, device=DEV); w[0] = 0.10
    for tri, vai in C.make_group_folds(5, 42):
        torch.manual_seed(seed); np.random.seed(seed)
        net = UNetK(vocab, K).to(DEV)
        opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=5e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        Xi = torch.tensor(idxA[tri], device=DEV); Zi = torch.tensor(cz[tri, None], device=DEV)
        Yi = torch.tensor(Tidx[tri], device=DEV)
        n = len(tri); bs = 128
        for ep in range(epochs):
            net.train(); perm = torch.randperm(n, device=DEV)
            for s in range(0, n, bs):
                b = perm[s:s + bs]
                loss = loss_fn(net(Xi[b], Zi[b]), Yi[b], Zi[b], w)
                opt.zero_grad(); loss.backward(); opt.step()
            sch.step()
        net.eval()
        with torch.no_grad():
            Xv = torch.tensor(idxA[vai], device=DEV); Zv = torch.tensor(cz[vai, None], device=DEV)
            oof[vai] = net(Xv, Zv).softmax(1).cpu().numpy()
    s, th, ma = D.tune_decision(oof, tr["V"], T)
    print(f"K={K}: group-CV {s:.4f} @ thr={th}", flush=True)
    np.save(os.path.join(C.ARTIFACT_DIR, f"ctxK{K}_oof.npy"), oof)
    return s


if __name__ == "__main__":
    t = time.time()
    for K in (3, 7, 11):
        run_K(K)
    print(f"elapsed {time.time()-t:.0f}s")
