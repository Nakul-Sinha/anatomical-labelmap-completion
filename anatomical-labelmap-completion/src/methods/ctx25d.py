"""
2.5D extended-context U-Net -> ensemble-format OOF + test proba.

Reconstructs the source volume and feeds a K-slice context (default K=11) centered on each row's
center slice; trains a plain U-Net (no aug except ELASTIC deformation, which helps cross-subject
generalization; D4/affine hurt on this atlas) with masked CE + soft-Dice. Seed-ensembled.
Best single method on honest group CV (~0.076).
"""
import os, sys, time
import numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
import common as C
import volume as VOL
import torch, torch.nn as nn, torch.nn.functional as F
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def build_vocab():
    uv = np.unique(np.concatenate([C.load_split("train")["V"].reshape(-1),
                                   C.load_split("test")["V"].reshape(-1)]))
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


def _rand_grid(B, H, W, a, res, dev, gen):
    small = torch.randn(B, 2, res, res, device=dev, generator=gen)
    flow = F.interpolate(small, size=(H, W), mode="bicubic", align_corners=True)
    flow = flow / flow.flatten(2).abs().amax(2, keepdim=True).clamp_min(1e-6).unsqueeze(-1) * a
    ys, xs = torch.meshgrid(torch.linspace(-1, 1, H, device=dev), torch.linspace(-1, 1, W, device=dev), indexing="ij")
    base = torch.stack([xs, ys], -1).unsqueeze(0).expand(B, -1, -1, -1)
    return base + flow.permute(0, 2, 3, 1) * torch.tensor([2 / (W - 1), 2 / (H - 1)], device=dev)


def _warp(x, grid):
    return F.grid_sample(x.float(), grid, mode="nearest", padding_mode="zeros", align_corners=True).round().long()


def _train(idxA, Tidx, tri, vocab, K, cfg, seed, bg):
    torch.manual_seed(seed); np.random.seed(seed)
    gen = torch.Generator(device=DEV); gen.manual_seed(seed)
    net = UNetK(vocab, K, base=cfg["base"], p=cfg["dropout"]).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg["epochs"])
    w = torch.ones(18, device=DEV); w[0] = cfg["bg_w"]; r = (K - 1) // 2
    Xi = torch.tensor(idxA[tri], device=DEV); Yi = torch.tensor(Tidx[tri], device=DEV)
    n = len(tri); bs = cfg["bs"]; a = cfg["elastic_a"]
    for ep in range(cfg["epochs"]):
        net.train(); perm = torch.randperm(n, device=DEV)
        for s in range(0, n, bs):
            b = perm[s:s + bs]; xb = Xi[b]; yb = Yi[b]
            if a > 0:
                g = _rand_grid(len(b), 32, 32, a, cfg["res"], DEV, gen)
                xb = _warp(xb, g); yb = _warp(yb[:, None], g)[:, 0]
            cz = (xb[:, r:r + 1] == bg).float()
            lg = net(xb, cz); prob = lg.softmax(1); mm = cz[:, 0]
            ce = (F.cross_entropy(lg, yb, weight=w, reduction="none") * mm).sum() / mm.sum().clamp_min(1)
            oh = F.one_hot(yb, 18).permute(0, 3, 1, 2).float(); pp = prob[:, 1:] * cz; gg = oh[:, 1:] * cz
            dice = 1 - ((2 * (pp * gg).sum((0, 2, 3)) + 1) / ((pp + gg).sum((0, 2, 3)) + 1)).mean()
            (ce + dice).backward(); opt.step(); opt.zero_grad()
        sch.step()
    net.eval()
    return net


@torch.no_grad()
def _predict(net, idxA, ids, K, bg):
    r = (K - 1) // 2
    out = np.zeros((len(ids), 18, 32, 32), np.float32)
    for s in range(0, len(ids), 128):
        b = ids[s:s + 128]; Xv = torch.tensor(idxA[b], device=DEV)
        out[s:s + len(b)] = net(Xv, (Xv[:, r:r + 1] == bg).float()).softmax(1).cpu().numpy()
    return out


DEFAULT = dict(base=48, dropout=0.30, lr=2e-3, wd=5e-4, epochs=150, bs=128, bg_w=0.10,
               elastic_a=3.0, res=4)


def run(K=11, cfg=None, seeds_oof=(0, 1), seeds_test=(0, 1, 2), save=True, name="ctx25d"):
    cfg = {**DEFAULT, **(cfg or {})}
    lut, vocab = build_vocab()
    tr = C.load_split("train"); te = C.load_split("test")
    Tidx = tr["Tidx"].astype(np.int64)
    bg = int(lut[0])
    ctx_tr, _, _ = VOL.extended_context(tr["V"], K); idx_tr = lut[ctx_tr.astype(np.int64)]
    ctx_te, _, _ = VOL.extended_context(te["V"], K); idx_te = lut[ctx_te.astype(np.int64)]
    N = len(idx_tr)
    oof = np.zeros((N, 18, 32, 32), np.float32)
    t0 = time.time()
    for tri, vai in C.make_group_folds(5, 42):
        acc = np.zeros((len(vai), 18, 32, 32), np.float32)
        for sd in seeds_oof:
            net = _train(idx_tr, Tidx, tri, vocab, K, cfg, seed=1000 + sd, bg=bg)
            acc += _predict(net, idx_tr, np.array(vai), K, bg)
        oof[vai] = acc / len(seeds_oof)
    test = np.zeros((len(idx_te), 18, 32, 32), np.float32)
    allidx = np.arange(N)
    for sd in seeds_test:
        net = _train(idx_tr, Tidx, allidx, vocab, K, cfg, seed=2000 + sd, bg=bg)
        test += _predict(net, idx_te, np.arange(len(idx_te)), K, bg)
    test /= len(seeds_test)
    if save:
        from ensemble import save_preds
        save_preds(name, oof, test)
    print(f"ctx25d K={K} done ({time.time()-t0:.0f}s)", flush=True)
    return oof, test


if __name__ == "__main__":
    import ensemble as E, decide2 as D
    oof, test = run(K=11)
    tr = C.load_split("train")
    print("ctx25d group-CV:", D.tune_decision(oof, tr["V"], tr["T"])[0])
