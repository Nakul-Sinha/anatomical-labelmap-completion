import os
import sys
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

DEV = "cuda" if torch.cuda.is_available() else "cpu"
_HERE = os.path.dirname(os.path.abspath(__file__))

TARGET_LABELS = (1893, 2126, 2660, 2823, 3127, 3680, 4638, 4877,
                 5094, 5170, 6102, 6798, 7200, 7760, 7775, 7989, 8525)
NUM_CLASSES = len(TARGET_LABELS) + 1
_L2I = {0: 0}
for _i, _l in enumerate(TARGET_LABELS, 1):
    _L2I[_l] = _i
_I2L = np.array([{v: k for k, v in _L2I.items()}[i] for i in range(NUM_CLASSES)], dtype=np.int64)
_L2I_ARR = np.zeros(max(TARGET_LABELS) + 1, dtype=np.int64)
for _l, _i in _L2I.items():
    _L2I_ARR[_l] = _i


def labels_to_idx(a):
    return _L2I_ARR[a]


def idx_to_labels(a):
    return _I2L[a]


def _find_data_dir():
    cands = [os.path.join(_HERE, "dataset", "public"), os.path.join(_HERE, "dataset"),
             os.path.join(os.getcwd(), "dataset", "public"), os.path.join(os.getcwd(), "dataset"),
             os.path.join(_HERE, "..", "dataset", "public"), os.path.join(_HERE, "..", "dataset")]
    for c in cands:
        if os.path.exists(os.path.join(c, "train.csv")):
            return c
    raise FileNotFoundError("dataset with train.csv not found near solution.py or cwd")


_CACHE = {}


def load_split(which):
    if which in _CACHE:
        return _CACHE[which]
    df = pd.read_csv(os.path.join(_find_data_dir(), which + ".csv"))
    ids, Vs, Ts = [], [], []
    for _, r in df.iterrows():
        ids.append(r["id"])
        Vs.append(np.asarray(json.loads(r["context_json"])["visible_label_maps"], dtype=np.int64).reshape(3, 32, 32))
        if which == "train":
            Ts.append(np.asarray(json.loads(r["answer_json"])["target_label_map"], dtype=np.int64).reshape(32, 32))
    out = {"ids": np.array(ids), "V": np.stack(Vs).astype(np.int16)}
    if which == "train":
        T = np.stack(Ts).astype(np.int16)
        out["T"] = T
        out["Tidx"] = labels_to_idx(T.astype(np.int64)).astype(np.int8)
    _CACHE[which] = out
    return out


def write_submission(ids, preds, path):
    rows = []
    for i, p in zip(ids, preds):
        obj = {"grid_shape": [32, 32], "target_label_map": np.asarray(p, dtype=np.int64).reshape(-1).tolist()}
        rows.append({"id": i, "answer_json": json.dumps(obj, separators=(",", ":"))})
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    pd.DataFrame(rows, columns=["id", "answer_json"]).to_csv(path, index=False)


def _key(a):
    return a.astype(np.int16).tobytes()


def build_chain(V):
    n = len(V)
    cen2row = {}
    for i in range(n):
        cen2row.setdefault(_key(V[i, 1]), []).append(i)
    nxt = [-1] * n
    prv = [-1] * n
    for i in range(n):
        for j in cen2row.get(_key(V[i, 2]), []):
            if _key(V[i, 1]) == _key(V[j, 0]):
                nxt[i] = j
                break
        for j in cen2row.get(_key(V[i, 0]), []):
            if _key(V[i, 1]) == _key(V[j, 2]):
                prv[i] = j
                break
    return nxt, prv


def extended_context(V, K):
    r = (K - 1) // 2
    n = len(V)
    nxt, prv = build_chain(V)
    out = np.zeros((n, K, 32, 32), dtype=V.dtype)
    for i in range(n):
        out[i, r] = V[i, 1]
        cur = i
        for d in range(1, r + 1):
            if d == 1:
                out[i, r + d] = V[i, 2]
                cur = nxt[i]
            else:
                if cur < 0:
                    break
                out[i, r + d] = V[cur, 2]
                cur = nxt[cur]
        cur = i
        for d in range(1, r + 1):
            if d == 1:
                out[i, r - d] = V[i, 0]
                cur = prv[i]
            else:
                if cur < 0:
                    break
                out[i, r - d] = V[cur, 0]
                cur = prv[cur]
    return out


def smooth3d(proba, V, alpha=0.5, width=3):
    n = len(proba)
    nxt, prv = build_chain(V)
    out = np.empty_like(proba)
    for i in range(n):
        acc = proba[i].astype(np.float64).copy()
        w = 1.0
        cur, wt = i, 1.0
        for _ in range(width):
            cur = nxt[cur]
            if cur < 0:
                break
            wt *= alpha
            acc += wt * proba[cur]
            w += wt
        cur, wt = i, 1.0
        for _ in range(width):
            cur = prv[cur]
            if cur < 0:
                break
            wt *= alpha
            acc += wt * proba[cur]
            w += wt
        out[i] = (acc / w).astype(np.float32)
    out /= (out.sum(1, keepdims=True) + 1e-9)
    return out


def decide_adv(proba, czmask, assign_thr, min_area):
    fg = proba[1:]
    best = fg.argmax(0) + 1
    best_p = fg.max(0)
    idx = np.where((best_p >= assign_thr) & czmask, best, 0)
    if min_area > 0:
        for L in range(1, NUM_CLASSES):
            m = idx == L
            if 0 < m.sum() < min_area:
                idx[m] = 0
    return idx_to_labels(idx)


def build_vocab1():
    tr = load_split("train"); te = load_split("test")
    uv = np.unique(np.concatenate([tr["V"].reshape(-1), te["V"].reshape(-1)]))
    lut = np.zeros(int(uv.max()) + 1, np.int64)
    for i, l in enumerate(uv, 1):
        lut[int(l)] = i
    return lut, len(uv) + 1


def build_vocab0():
    tr = load_split("train"); te = load_split("test")
    vals = np.unique(np.concatenate([tr["V"].reshape(-1), te["V"].reshape(-1)])).astype(np.int64)
    lut = np.zeros(int(vals.max()) + 1, np.int64)
    for i, v in enumerate(vals):
        lut[int(v)] = i
    return lut, len(vals)


def _rand_grid(B, H, W, a, res, gen):
    small = torch.randn(B, 2, res, res, device=DEV, generator=gen)
    flow = F.interpolate(small, size=(H, W), mode="bicubic", align_corners=True)
    flow = flow / flow.flatten(2).abs().amax(2, keepdim=True).clamp_min(1e-6).unsqueeze(-1) * a
    ys, xs = torch.meshgrid(torch.linspace(-1, 1, H, device=DEV), torch.linspace(-1, 1, W, device=DEV), indexing="ij")
    base = torch.stack([xs, ys], -1).unsqueeze(0).expand(B, -1, -1, -1)
    return base + flow.permute(0, 2, 3, 1) * torch.tensor([2 / (W - 1), 2 / (H - 1)], device=DEV)


def _warp(x, grid):
    return F.grid_sample(x.float(), grid, mode="nearest", padding_mode="zeros", align_corners=True).round().long()


def _dice_ce(logits, y, cz, w):
    mm = cz[:, 0]
    ce = (F.cross_entropy(logits, y, weight=w, reduction="none") * mm).sum() / mm.sum().clamp_min(1)
    prob = logits.softmax(1)
    oh = F.one_hot(y, NUM_CLASSES).permute(0, 3, 1, 2).float()
    pp = prob[:, 1:] * cz; gg = oh[:, 1:] * cz
    dice = 1 - ((2 * (pp * gg).sum((0, 2, 3)) + 1) / ((pp + gg).sum((0, 2, 3)) + 1)).mean()
    return ce + dice


def _cosine(opt, epochs):
    return torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)


class ConvBlock(nn.Module):
    def __init__(self, cin, cout, groups=8, p=0.0):
        super().__init__()
        g = max(1, min(groups, cout))
        while cout % g != 0:
            g -= 1
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1); self.n1 = nn.GroupNorm(g, cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1); self.n2 = nn.GroupNorm(g, cout)
        self.act = nn.SiLU(); self.drop = nn.Dropout2d(p)

    def forward(self, x):
        x = self.act(self.n1(self.c1(x)))
        x = self.drop(x)
        return self.act(self.n2(self.c2(x)))


class UNet2D(nn.Module):
    def __init__(self, n_vocab, embed_dim=16, base=48, n_classes=18, p=0.15):
        super().__init__()
        self.emb = nn.Embedding(n_vocab, embed_dim, padding_idx=0)
        in_ch = 3 * embed_dim + 1
        self.enc0 = ConvBlock(in_ch, base); self.enc1 = ConvBlock(base, base * 2); self.enc2 = ConvBlock(base * 2, base * 4, p=p)
        self.pool = nn.MaxPool2d(2)
        self.up1 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2); self.dec1 = ConvBlock(base * 4, base * 2)
        self.up2 = nn.ConvTranspose2d(base * 2, base, 2, stride=2); self.dec0 = ConvBlock(base * 2, base)
        self.head = nn.Conv2d(base, n_classes, 1)

    def forward(self, idx, czmask):
        B = idx.shape[0]
        e = self.emb(idx).permute(0, 1, 4, 2, 3).reshape(B, -1, 32, 32)
        x = torch.cat([e, czmask], 1)
        x0 = self.enc0(x); x1 = self.enc1(self.pool(x0)); x2 = self.enc2(self.pool(x1))
        d1 = self.dec1(torch.cat([self.up1(x2), x1], 1)); d0 = self.dec0(torch.cat([self.up2(d1), x0], 1))
        return self.head(d0)


UNET_CFG = dict(embed_dim=16, base=48, dropout=0.30, lr=2e-3, wd=5e-4, epochs=135,
                batch_size=128, bg_weight=0.10)


def unet_test(tr, te):
    lut, nv = build_vocab1()
    idx_all = torch.from_numpy(lut[tr["V"].astype(np.int64)]).long()
    cz_all = torch.from_numpy((tr["V"][:, 1] == 0)[:, None].astype(np.float32))
    y_all = torch.from_numpy(tr["Tidx"].astype(np.int64))
    idx_te = torch.from_numpy(lut[te["V"].astype(np.int64)]).long()
    cz_te = torch.from_numpy((te["V"][:, 1] == 0)[:, None].astype(np.float32))
    acc = np.zeros((len(te["V"]), NUM_CLASSES, 32, 32), np.float32)
    for sd in UNET_SEEDS:
        torch.manual_seed(777 + sd); np.random.seed(777 + sd)
        net = UNet2D(nv, UNET_CFG["embed_dim"], UNET_CFG["base"], NUM_CLASSES, UNET_CFG["dropout"]).to(DEV)
        opt = torch.optim.AdamW(net.parameters(), lr=UNET_CFG["lr"], weight_decay=UNET_CFG["wd"])
        sch = _cosine(opt, UNET_CFG["epochs"])
        w = torch.ones(NUM_CLASSES, device=DEV); w[0] = UNET_CFG["bg_weight"]
        ia = idx_all.to(DEV); ca = cz_all.to(DEV); ya = y_all.to(DEV)
        n = len(ia); bs = UNET_CFG["batch_size"]
        g = torch.Generator(device="cpu"); g.manual_seed(777 + sd)
        for ep in range(UNET_CFG["epochs"]):
            net.train(); perm = torch.randperm(n, generator=g).to(DEV)
            for s in range(0, n, bs):
                b = perm[s:s + bs]
                loss = _dice_ce(net(ia[b], ca[b]), ya[b], ca[b], w)
                opt.zero_grad(); loss.backward(); opt.step()
            sch.step()
        net.eval()
        with torch.no_grad():
            it = idx_te.to(DEV); ct = cz_te.to(DEV)
            for s in range(0, len(it), 256):
                acc[s:s + 256] += net(it[s:s + 256], ct[s:s + 256]).softmax(1).cpu().numpy()
    return acc / len(UNET_SEEDS)


class ResDilBlock(nn.Module):
    def __init__(self, w, d, groups, p):
        super().__init__()
        self.n1 = nn.GroupNorm(groups, w); self.c1 = nn.Conv2d(w, w, 3, padding=d, dilation=d)
        self.n2 = nn.GroupNorm(groups, w); self.c2 = nn.Conv2d(w, w, 3, padding=d, dilation=d)
        self.drop = nn.Dropout2d(p); self.act = nn.GELU()

    def forward(self, x):
        h = self.c1(self.act(self.n1(x)))
        h = self.c2(self.drop(self.act(self.n2(h))))
        return x + h


class DilCNN(nn.Module):
    def __init__(self, n_vocab, emb=16, w=96, dilations=(1, 2, 4, 8, 4, 2, 1), groups=8, p=0.10):
        super().__init__()
        self.emb = nn.Embedding(n_vocab, emb)
        in_ch = 3 * emb + 1
        self.stem = nn.Conv2d(in_ch, w, 3, padding=1)
        self.blocks = nn.ModuleList([ResDilBlock(w, d, groups, p) for d in dilations])
        self.head_n = nn.GroupNorm(groups, w); self.act = nn.GELU(); self.head = nn.Conv2d(w, NUM_CLASSES, 1)

    def forward(self, vidx):
        B = vidx.shape[0]
        e = self.emb(vidx).permute(0, 1, 4, 2, 3).reshape(B, -1, 32, 32)
        czm = (vidx[:, 1:2] == 0).float()
        x = self.stem(torch.cat([e, czm], 1))
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.act(self.head_n(x)))


CTX_CFG = dict(emb=16, w=96, dilations=(1, 2, 4, 8, 4, 2, 1), groups=8, dropout=0.30, bs=64,
               lr=2e-3, wd=5e-4, epochs=120, bg_weight=0.10, dice_w=1.0)


def ctx_test(tr, te):
    lut, nv = build_vocab0()
    va = torch.tensor(lut[tr["V"].astype(np.int64)], device=DEV)
    ta = torch.tensor(tr["Tidx"].astype(np.int64), device=DEV)
    vt = torch.tensor(lut[te["V"].astype(np.int64)], device=DEV)
    acc = np.zeros((len(te["V"]), NUM_CLASSES, 32, 32), np.float32)
    for sd in CTX_SEEDS:
        torch.manual_seed(42 + sd); np.random.seed(42 + sd)
        net = DilCNN(nv, CTX_CFG["emb"], CTX_CFG["w"], CTX_CFG["dilations"], CTX_CFG["groups"], CTX_CFG["dropout"]).to(DEV)
        opt = torch.optim.AdamW(net.parameters(), lr=CTX_CFG["lr"], weight_decay=CTX_CFG["wd"])
        n = len(va); bs = CTX_CFG["bs"]
        sch = _cosine(opt, CTX_CFG["epochs"] * max(1, (n + bs - 1) // bs))
        w = torch.ones(NUM_CLASSES, device=DEV); w[0] = CTX_CFG["bg_weight"]
        rng = np.random.RandomState(42 + sd)
        for ep in range(CTX_CFG["epochs"]):
            net.train(); order = rng.permutation(n)
            for s in range(0, n, bs):
                b = order[s:s + bs]; vb = va[b]; tb = ta[b]
                czm = (vb[:, 1:2] == 0).float()
                logits = net(vb); mm = czm[:, 0]
                ce = (F.cross_entropy(logits, tb, weight=w, reduction="none") * mm).sum() / czm.sum().clamp_min(1)
                p = logits.softmax(1) * czm; y = F.one_hot(tb, NUM_CLASSES).permute(0, 3, 1, 2).float() * czm
                inter = (p[:, 1:] * y[:, 1:]).sum((0, 2, 3)); den = p[:, 1:].sum((0, 2, 3)) + y[:, 1:].sum((0, 2, 3))
                dice = (2 * inter + 1) / (den + 1)
                loss = ce + CTX_CFG["dice_w"] * (1 - dice.mean())
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sch.step()
        net.eval()
        with torch.no_grad():
            for s in range(0, len(vt), 256):
                acc[s:s + 256] += net(vt[s:s + 256]).softmax(1).cpu().numpy()
    return acc / len(CTX_SEEDS)


def _blk(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.GroupNorm(8, o), nn.SiLU(),
                         nn.Conv2d(o, o, 3, padding=1), nn.GroupNorm(8, o), nn.SiLU())


class DeepUNet(nn.Module):
    def __init__(self, vocab, K, emb=16, base=40, p=0.30):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb, padding_idx=0)
        cin = K * emb + 1
        self.e0 = _blk(cin, base); self.e1 = _blk(base, base * 2); self.e2 = _blk(base * 2, base * 4); self.e3 = _blk(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)
        self.u2 = nn.ConvTranspose2d(base * 8, base * 4, 2, 2); self.d2 = _blk(base * 8, base * 4)
        self.u1 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2); self.d1 = _blk(base * 4, base * 2)
        self.u0 = nn.ConvTranspose2d(base * 2, base, 2, 2); self.d0 = _blk(base * 2, base)
        self.drop = nn.Dropout2d(p); self.head = nn.Conv2d(base, 18, 1)

    def forward(self, idx, cz):
        B, K, H, W = idx.shape
        e = self.emb(idx).permute(0, 1, 4, 2, 3).reshape(B, -1, H, W)
        x = torch.cat([e, cz], 1)
        x0 = self.e0(x); x1 = self.e1(self.pool(x0)); x2 = self.e2(self.pool(x1)); x3 = self.e3(self.pool(x2))
        d2 = self.d2(torch.cat([self.u2(x3), x2], 1)); d1 = self.d1(torch.cat([self.u1(d2), x1], 1)); d0 = self.d0(torch.cat([self.u0(d1), x0], 1))
        return self.head(self.drop(d0))


DEEP_CFG = dict(base=40, dropout=0.30, lr=2e-3, wd=5e-4, epochs=170, bs=128, bg_w=0.10, elastic_a=3.0, res=4)


def deep_test(K):
    lut, vocab = build_vocab1(); bg = int(lut[0])
    tr = load_split("train"); te = load_split("test"); Tidx = tr["Tidx"].astype(np.int64)
    idx_tr = lut[extended_context(tr["V"], K).astype(np.int64)]
    idx_te = lut[extended_context(te["V"], K).astype(np.int64)]
    r = (K - 1) // 2; N = len(idx_tr)
    test = np.zeros((len(idx_te), NUM_CLASSES, 32, 32), np.float32)
    for sd in DEEP_SEEDS:
        torch.manual_seed(2000 + sd); np.random.seed(2000 + sd)
        gen = torch.Generator(device=DEV); gen.manual_seed(2000 + sd)
        net = DeepUNet(vocab, K, base=DEEP_CFG["base"], p=DEEP_CFG["dropout"]).to(DEV)
        opt = torch.optim.AdamW(net.parameters(), lr=DEEP_CFG["lr"], weight_decay=DEEP_CFG["wd"])
        sch = _cosine(opt, DEEP_CFG["epochs"])
        w = torch.ones(NUM_CLASSES, device=DEV); w[0] = DEEP_CFG["bg_w"]
        Xi = torch.tensor(idx_tr, device=DEV); Yi = torch.tensor(Tidx, device=DEV)
        bs = DEEP_CFG["bs"]
        for ep in range(DEEP_CFG["epochs"]):
            net.train(); perm = torch.randperm(N, device=DEV)
            for s in range(0, N, bs):
                b = perm[s:s + bs]; xb = Xi[b]; yb = Yi[b]
                g = _rand_grid(len(b), 32, 32, DEEP_CFG["elastic_a"], DEEP_CFG["res"], gen)
                xb = _warp(xb, g); yb = _warp(yb[:, None], g)[:, 0]
                cz = (xb[:, r:r + 1] == bg).float()
                loss = _dice_ce(net(xb, cz), yb, cz, w)
                opt.zero_grad(); loss.backward(); opt.step()
            sch.step()
        net.eval()
        with torch.no_grad():
            Xt = torch.tensor(idx_te, device=DEV)
            for s in range(0, len(Xt), 128):
                xb = Xt[s:s + 128]
                test[s:s + 128] += net(xb, (xb[:, r:r + 1] == bg).float()).softmax(1).cpu().numpy()
    return test / len(DEEP_SEEDS)


def knn_test(tr, te):
    cen_tr = tr["V"][:, 1].reshape(len(tr["V"]), -1).astype(np.int16)
    Tidx = tr["Tidx"].astype(np.int64)
    cen_te = te["V"][:, 1].reshape(len(te["V"]), -1).astype(np.int16)
    out = np.zeros((len(te["V"]), NUM_CLASSES, 32, 32), np.float32)
    for i in range(len(cen_te)):
        s = (cen_tr == cen_te[i]).sum(1).astype(np.float64)
        order = np.argsort(-s)[:3]; wv = s[order]; wv = wv - wv.min() + 1e-3
        acc = np.zeros((NUM_CLASSES, 1024), np.float64)
        for j, wj in zip(order, wv):
            acc[Tidx[j].reshape(-1), np.arange(1024)] += wj
        acc /= (acc.sum(0, keepdims=True) + 1e-9)
        out[i] = acc.reshape(NUM_CLASSES, 32, 32)
    return out


W = {"deep7": 0.13, "deep9": 0.15, "deep11": 0.13, "deep13": 0.11, "deep15": 0.13,
     "unet": 0.11, "ctx": 0.06, "knn": 0.18}
SMOOTH_ALPHA, SMOOTH_WIDTH = 0.5, 3
THRESH, MIN_AREA = 0.375, 0
UNET_SEEDS = (0, 1)
CTX_SEEDS = (0, 1)
DEEP_SEEDS = (0, 1, 2)


def main():
    tr = load_split("train"); te = load_split("test")
    proba = {"deep7": deep_test(7), "deep9": deep_test(9), "deep11": deep_test(11),
             "deep13": deep_test(13), "deep15": deep_test(15),
             "unet": unet_test(tr, te), "ctx": ctx_test(tr, te), "knn": knn_test(tr, te)}
    blend = sum(W[k] * proba[k] for k in W)
    blend /= (blend.sum(1, keepdims=True) + 1e-9)
    blend = smooth3d(blend, te["V"], alpha=SMOOTH_ALPHA, width=SMOOTH_WIDTH)
    preds = [decide_adv(blend[i], te["V"][i, 1] == 0, THRESH, MIN_AREA) for i in range(len(te["ids"]))]
    out = os.path.join(_HERE, "working", "submission.csv")
    write_submission(te["ids"], preds, out)
    print("wrote", out, len(preds), "rows")


if __name__ == "__main__":
    main()
