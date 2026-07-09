"""
U-Net segmentation method -> soft per-cell class probabilities in the ensemble format.

The 3 visible label-map slices are encoded with a learned EMBEDDING over the visible-label
vocabulary (built from all distinct labels in train+test inputs; index 0 reserved for
unknown/pad). Each cell of each slice is embedded (dim ~16); the three embedded slices are
concatenated channel-wise together with a binary center-is-zero mask and normalized (x,y)
coordinate channels. A small U-Net (32->16->8, GroupNorm skip connections, ~1M params) maps
this to 18 per-cell logits -> softmax.

Loss = weighted cross-entropy (background down-weighted) + soft multiclass Dice over the 17
target classes. Both terms are evaluated ONLY on center-zero cells (the only place a target can
live; the shared decision rule enforces this at inference), which focuses capacity on the hard
~2.6%-positive region. The Dice term (smoothed) penalizes spurious target labels, aligning with
the squared active-macro-IoU grader metric.

OOF uses the canonical 5-fold split (never leaking a val row into its own training fold); the
test prediction averages several models trained on all 600 rows. Everything is seeded.
"""
import os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
import common as C

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------------
# Vocabulary over visible labels (train + test inputs; these are inputs, not labels).
# ----------------------------------------------------------------------------
def build_vocab():
    tr = C.load_split("train"); te = C.load_split("test")
    allv = np.concatenate([tr["V"].reshape(-1), te["V"].reshape(-1)])
    uv = np.unique(allv)                                   # distinct opaque labels (incl. 0)
    lut = np.zeros(int(uv.max()) + 1, dtype=np.int64)      # opaque label -> vocab index
    for i, l in enumerate(uv, start=1):                    # 0 reserved for unknown/pad
        lut[int(l)] = i
    n_vocab = len(uv) + 1
    return lut, n_vocab


# ----------------------------------------------------------------------------
# Model.
# ----------------------------------------------------------------------------
class ConvBlock(nn.Module):
    def __init__(self, cin, cout, groups=8, p=0.0):
        super().__init__()
        g = max(1, min(groups, cout))
        while cout % g != 0:
            g -= 1
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.n1 = nn.GroupNorm(g, cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.n2 = nn.GroupNorm(g, cout)
        self.act = nn.SiLU()
        self.drop = nn.Dropout2d(p)

    def forward(self, x):
        x = self.act(self.n1(self.c1(x)))
        x = self.drop(x)
        x = self.act(self.n2(self.c2(x)))
        return x


class UNet(nn.Module):
    def __init__(self, n_vocab, embed_dim=16, base=48, n_classes=18, p=0.15, use_coords=True):
        super().__init__()
        self.use_coords = use_coords
        self.emb = nn.Embedding(n_vocab, embed_dim, padding_idx=0)
        in_ch = 3 * embed_dim + 1 + (2 if use_coords else 0)  # 3 slices + czmask (+2 coords)
        self.enc0 = ConvBlock(in_ch, base)
        self.enc1 = ConvBlock(base, base * 2)
        self.enc2 = ConvBlock(base * 2, base * 4, p=p)     # bottleneck (dropout)
        self.pool = nn.MaxPool2d(2)
        self.up1 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec1 = ConvBlock(base * 4, base * 2)
        self.up2 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec0 = ConvBlock(base * 2, base)
        self.head = nn.Conv2d(base, n_classes, 1)
        ys, xs = torch.meshgrid(torch.linspace(-1, 1, 32),
                                torch.linspace(-1, 1, 32), indexing="ij")
        self.register_buffer("coords", torch.stack([xs, ys], 0).unsqueeze(0))  # (1,2,32,32)

    def forward(self, idx, czmask):
        # idx: (B,3,32,32) long; czmask: (B,1,32,32) float
        B = idx.shape[0]
        e = self.emb(idx)                                  # (B,3,32,32,embed)
        e = e.permute(0, 1, 4, 2, 3).reshape(B, -1, 32, 32)  # (B,3*embed,32,32)
        feats = [e, czmask]
        if self.use_coords:
            feats.append(self.coords.expand(B, -1, -1, -1))
        x = torch.cat(feats, dim=1)
        x0 = self.enc0(x)                                  # 32, base
        x1 = self.enc1(self.pool(x0))                      # 16, base*2
        x2 = self.enc2(self.pool(x1))                      # 8,  base*4
        u1 = self.up1(x2)                                  # 16, base*2
        d1 = self.dec1(torch.cat([u1, x1], 1))
        u2 = self.up2(d1)                                  # 32, base
        d0 = self.dec0(torch.cat([u2, x0], 1))
        return self.head(d0)                               # (B,18,32,32)


# ----------------------------------------------------------------------------
# Loss (evaluated on center-zero cells only).
# ----------------------------------------------------------------------------
def compute_loss(logits, y, cz, ce_weight_vec, ce_w=1.0, dice_w=1.0, dice_eps=1.0):
    czf = cz.float()                                       # (B,1,H,W)
    m = czf.squeeze(1)                                     # (B,H,W)
    denom = m.sum().clamp_min(1.0)
    ce = F.cross_entropy(logits, y, weight=ce_weight_vec, reduction="none")  # (B,H,W)
    ce = (ce * m).sum() / denom
    prob = F.softmax(logits, dim=1)                        # (B,18,H,W)
    onehot = F.one_hot(y, logits.shape[1]).permute(0, 3, 1, 2).float()
    p = prob[:, 1:] * czf                                  # (B,17,H,W) predicted target mass
    g = onehot[:, 1:] * czf                                # (B,17,H,W) truth target mass
    inter = (p * g).sum(dim=(2, 3))                        # (B,17)
    tot = p.sum(dim=(2, 3)) + g.sum(dim=(2, 3))            # (B,17)
    dice = (2 * inter + dice_eps) / (tot + dice_eps)       # (B,17)
    dice_loss = 1.0 - dice.mean()
    return ce_w * ce + dice_w * dice_loss


# ----------------------------------------------------------------------------
# Data tensors.
# ----------------------------------------------------------------------------
def make_tensors(V, lut, Tidx=None):
    idx = torch.from_numpy(lut[V.astype(np.int64)]).long()          # (N,3,32,32)
    cz = torch.from_numpy((V[:, 1] == 0)[:, None].astype(np.float32))  # (N,1,32,32)
    y = None
    if Tidx is not None:
        y = torch.from_numpy(Tidx.astype(np.int64))                 # (N,32,32)
    return idx, cz, y


def _seed_all(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_model(idx_tr, cz_tr, y_tr, n_vocab, cfg, seed):
    _seed_all(seed)
    model = UNet(n_vocab, embed_dim=cfg["embed_dim"], base=cfg["base"], p=cfg["dropout"],
                 use_coords=cfg.get("use_coords", True)).to(DEVICE)
    wvec = torch.ones(C.NUM_CLASSES, device=DEVICE)
    wvec[0] = cfg["bg_weight"]
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    epochs = cfg["epochs"]
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    N = idx_tr.shape[0]
    bs = cfg["batch_size"]
    idx_tr = idx_tr.to(DEVICE); cz_tr = cz_tr.to(DEVICE); y_tr = y_tr.to(DEVICE)
    g = torch.Generator(device="cpu"); g.manual_seed(seed)
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(N, generator=g).to(DEVICE)
        for s in range(0, N, bs):
            b = perm[s:s + bs]
            xb, cb, yb = idx_tr[b], cz_tr[b], y_tr[b]
            if cfg.get("flip", False):
                if torch.rand(1, generator=g).item() < 0.5:
                    xb = torch.flip(xb, dims=[3]); cb = torch.flip(cb, dims=[3]); yb = torch.flip(yb, dims=[2])
            logits = model(xb, cb)
            loss = compute_loss(logits, yb, cb, wvec,
                                ce_w=cfg["ce_w"], dice_w=cfg["dice_w"], dice_eps=cfg["dice_eps"])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    model.eval()
    return model


@torch.no_grad()
def predict_proba(model, idx, cz, bs=256):
    idx = idx.to(DEVICE); cz = cz.to(DEVICE)
    out = np.zeros((idx.shape[0], C.NUM_CLASSES, 32, 32), np.float32)
    model.eval()
    for s in range(0, idx.shape[0], bs):
        logits = model(idx[s:s + bs], cz[s:s + bs])
        out[s:s + bs] = F.softmax(logits, dim=1).cpu().numpy()
    return out


DEFAULT_CFG = dict(
    embed_dim=16, base=48, dropout=0.15,
    lr=2e-3, wd=1e-4, epochs=110, batch_size=96,
    bg_weight=0.10, ce_w=1.0, dice_w=1.0, dice_eps=1.0,
    flip=False, use_coords=True,
)


def run(cfg=None, seeds_oof=(0,), seeds_test=(0, 1, 2), save=True, name="cnn_unet", verbose=True):
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    lut, n_vocab = build_vocab()
    tr = C.load_split("train"); te = C.load_split("test")
    V, Tidx = tr["V"], tr["Tidx"]
    N = len(V)
    idx_all, cz_all, y_all = make_tensors(V, lut, Tidx)

    # ---- OOF (VOLUME-GROUPED folds; test is volume-disjoint, so this is the honest
    #      estimate. No volume ever spans a train/val boundary.) ----
    oof = np.zeros((N, C.NUM_CLASSES, 32, 32), np.float32)
    t0 = time.time()
    for fi, (tri, vai) in enumerate(C.make_group_folds(5, seed=42)):
        acc = np.zeros((len(vai), C.NUM_CLASSES, 32, 32), np.float32)
        for sd in seeds_oof:
            model = train_model(idx_all[tri], cz_all[tri], y_all[tri], n_vocab, cfg, seed=1000 * sd + fi)
            acc += predict_proba(model, idx_all[vai], cz_all[vai])
        oof[vai] = acc / len(seeds_oof)
        if verbose:
            print(f"  fold {fi}: done ({time.time()-t0:.0f}s)")

    # ---- TEST (all 600 rows; average several seeds) ----
    idx_te, cz_te, _ = make_tensors(te["V"], lut)
    test = np.zeros((len(te["V"]), C.NUM_CLASSES, 32, 32), np.float32)
    for sd in seeds_test:
        model = train_model(idx_all, cz_all, y_all, n_vocab, cfg, seed=777 + sd)
        test += predict_proba(model, idx_te, cz_te)
    test /= len(seeds_test)

    if save:
        from ensemble import save_preds
        save_preds(name, oof, test)
    return oof, test


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(_HERE, ".."))
    import ensemble as E
    oof, test = run()
    tr = C.load_split("train")
    s, th = E.tune_thresh(oof, tr["V"], tr["T"])
    print(f"cnn_unet OOF unified score {s:.4f} @ thr={th}; oof{oof.shape} test{test.shape}")
