"""
Deeper, RESIDUAL plain U-Net -> soft per-cell class probabilities in the ensemble format.

This is a higher-capacity sibling of methods/cnn_unet.py (the base-48, 2-level reference).
It keeps that model's proven inputs and loss verbatim (imported, not re-implemented) and only
changes the *architecture*:

  * 3 downsampling levels (32 -> 16 -> 8 -> 4) instead of 2, so the bottleneck sees the whole
    slab and the decoder rebuilds detail through 3 skip connections;
  * RESIDUAL conv blocks (conv-GN-SiLU, conv-GN, + projected skip, SiLU) instead of plain ones,
    which ease optimisation of the deeper stack;
  * base channels 64 with multipliers (1,2,3,3) -> ~3.5M params (vs ~1M reference).

Everything else is identical to the reference recipe, which was already tuned to maximise the
HONEST volume-grouped CV (common.make_group_folds): the 3 visible slices are encoded with a
learned embedding over the visible-label vocabulary (dim 16) and concatenated with a binary
center-is-zero mask channel; the loss is background-down-weighted (0.10) cross-entropy + smoothed
soft multiclass Dice over the 17 target classes, BOTH masked to center-zero cells (the only place
a target can live). NO geometric augmentation, NO test-time augmentation, NO absolute-coordinate
channels -- all three were verified on volume-grouped CV to HURT (this is atlas-aligned data, so
absolute orientation/position carry real signal that invariance discards). Regularisation is
dropout (0.30 at the bottleneck, lighter mid) + weight decay 5e-4.

HONEST CV: OOF uses common.make_group_folds(5, seed=42) so no source volume ever spans a
train/val boundary (random folds leak overlapping sliding-window slabs and inflate ~9x; the
private test set is volume-disjoint). OOF averages 2 seeds/fold; the test prediction averages
several seeds trained on all 600 rows. Everything is seeded.

Goal: beat the base U-Net's group-CV as a single model AND add ensemble diversity (deeper +
residual -> a genuinely different feature hierarchy from the base-48 net).
"""
import os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))   # src/
sys.path.insert(0, _HERE)                        # src/methods/
import common as C
# Reuse the reference model's proven, frozen pieces (read-only import; cnn_unet is not modified).
import cnn_unet as base_unet
from cnn_unet import build_vocab, make_tensors, compute_loss, predict_proba, _seed_all

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Fixed 32x32 input sizes -> let cuDNN pick fastest kernels; TF32 for throughput on the 4050.
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# ----------------------------------------------------------------------------
# Residual conv block: conv-GN-SiLU -> dropout -> conv-GN -> (+ projected skip) -> SiLU.
# GroupNorm group count is chosen to divide the channel count.
# ----------------------------------------------------------------------------
def _gn(groups, c):
    g = max(1, min(groups, c))
    while c % g != 0:
        g -= 1
    return nn.GroupNorm(g, c)


class ResBlock(nn.Module):
    def __init__(self, cin, cout, groups=8, p=0.0):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.n1 = _gn(groups, cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.n2 = _gn(groups, cout)
        self.act = nn.SiLU()
        self.drop = nn.Dropout2d(p)
        self.proj = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x):
        h = self.act(self.n1(self.c1(x)))
        h = self.drop(h)
        h = self.n2(self.c2(h))
        return self.act(h + self.proj(x))


# ----------------------------------------------------------------------------
# Deeper residual U-Net (32 -> 16 -> 8 -> 4).  Same embedding+czmask input as the
# reference; 18 per-cell logits out.
# ----------------------------------------------------------------------------
class UNetDeep(nn.Module):
    def __init__(self, n_vocab, embed_dim=16, base=64, mult=(1, 2, 3, 3),
                 n_classes=18, p=0.30, p_mid=0.15):
        super().__init__()
        self.emb = nn.Embedding(n_vocab, embed_dim, padding_idx=0)
        in_ch = 3 * embed_dim + 1                      # 3 embedded slices + center-zero mask
        c = [base * m for m in mult]                   # e.g. (64,128,192,192)
        self.enc0 = ResBlock(in_ch, c[0])
        self.enc1 = ResBlock(c[0], c[1], p=p_mid)
        self.enc2 = ResBlock(c[1], c[2], p=p_mid)
        self.enc3 = ResBlock(c[2], c[3], p=p)          # bottleneck (heaviest dropout)
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(c[3], c[2], 2, stride=2)
        self.dec2 = ResBlock(c[2] + c[2], c[2], p=p_mid)
        self.up1 = nn.ConvTranspose2d(c[2], c[1], 2, stride=2)
        self.dec1 = ResBlock(c[1] + c[1], c[1])
        self.up0 = nn.ConvTranspose2d(c[1], c[0], 2, stride=2)
        self.dec0 = ResBlock(c[0] + c[0], c[0])
        self.head = nn.Conv2d(c[0], n_classes, 1)

    def forward(self, idx, czmask):
        # idx: (B,3,32,32) long; czmask: (B,1,32,32) float
        B = idx.shape[0]
        e = self.emb(idx)                                    # (B,3,32,32,embed)
        e = e.permute(0, 1, 4, 2, 3).reshape(B, -1, 32, 32)  # (B,3*embed,32,32)
        x = torch.cat([e, czmask], dim=1)
        x0 = self.enc0(x)                                    # 32, c0
        x1 = self.enc1(self.pool(x0))                        # 16, c1
        x2 = self.enc2(self.pool(x1))                        # 8,  c2
        x3 = self.enc3(self.pool(x2))                        # 4,  c3 (bottleneck)
        u2 = self.up2(x3)                                    # 8,  c2
        d2 = self.dec2(torch.cat([u2, x2], 1))
        u1 = self.up1(d2)                                    # 16, c1
        d1 = self.dec1(torch.cat([u1, x1], 1))
        u0 = self.up0(d1)                                    # 32, c0
        d0 = self.dec0(torch.cat([u0, x0], 1))
        return self.head(d0)                                 # (B,18,32,32)


# ----------------------------------------------------------------------------
# Config + training (loss/vocab/tensors/predict reused from the reference).
# ----------------------------------------------------------------------------
DEFAULT_CFG = dict(
    embed_dim=16, base=64, mult=(1, 2, 3, 3), dropout=0.30, dropout_mid=0.15,
    lr=2e-3, wd=5e-4, epochs=150, batch_size=64,
    bg_weight=0.10, ce_w=1.0, dice_w=1.0, dice_eps=1.0,
)


def build_model(n_vocab, cfg):
    return UNetDeep(n_vocab, embed_dim=cfg["embed_dim"], base=cfg["base"],
                    mult=tuple(cfg["mult"]), p=cfg["dropout"], p_mid=cfg["dropout_mid"]).to(DEVICE)


def train_model(idx_tr, cz_tr, y_tr, n_vocab, cfg, seed):
    _seed_all(seed)
    model = build_model(n_vocab, cfg)
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
            logits = model(idx_tr[b], cz_tr[b])
            loss = compute_loss(logits, y_tr[b], cz_tr[b], wvec,
                                ce_w=cfg["ce_w"], dice_w=cfg["dice_w"], dice_eps=cfg["dice_eps"])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    model.eval()
    return model


def run(cfg=None, seeds_oof=(0, 1), seeds_test=(0, 1, 2, 3), save=True, name="unet_deep", verbose=True):
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    lut, n_vocab = build_vocab()
    tr = C.load_split("train"); te = C.load_split("test")
    V, Tidx = tr["V"], tr["Tidx"]
    N = len(V)
    idx_all, cz_all, y_all = make_tensors(V, lut, Tidx)

    if verbose:
        m = build_model(n_vocab, cfg)
        n_par = sum(p.numel() for p in m.parameters())
        print(f"  UNetDeep base={cfg['base']} mult={tuple(cfg['mult'])} params={n_par/1e6:.2f}M "
              f"epochs={cfg['epochs']} bs={cfg['batch_size']}", flush=True)
        del m

    # ---- OOF (VOLUME-GROUPED folds; the honest, volume-disjoint estimate) ----
    oof = np.zeros((N, C.NUM_CLASSES, 32, 32), np.float32)
    t0 = time.time()
    for fi, (tri, vai) in enumerate(C.make_group_folds(5, seed=42)):
        acc = np.zeros((len(vai), C.NUM_CLASSES, 32, 32), np.float32)
        for sd in seeds_oof:
            model = train_model(idx_all[tri], cz_all[tri], y_all[tri], n_vocab, cfg, seed=1000 * sd + fi)
            acc += predict_proba(model, idx_all[vai], cz_all[vai])
        oof[vai] = acc / len(seeds_oof)
        if verbose:
            print(f"  [oof] group-fold {fi} done ({time.time()-t0:.0f}s)", flush=True)

    # ---- TEST (all 600 rows; average several seeds) ----
    test = np.zeros((len(te["V"]), C.NUM_CLASSES, 32, 32), np.float32)
    if seeds_test:
        idx_te, cz_te, _ = make_tensors(te["V"], lut)
        for sd in seeds_test:
            model = train_model(idx_all, cz_all, y_all, n_vocab, cfg, seed=777 + sd)
            test += predict_proba(model, idx_te, cz_te)
            if verbose:
                print(f"  [test] all-600 seed {sd} done ({time.time()-t0:.0f}s)", flush=True)
        test /= len(seeds_test)

    if save:
        from ensemble import save_preds
        save_preds(name, oof, test)
    return oof, test


if __name__ == "__main__":
    import argparse
    sys.path.insert(0, os.path.join(_HERE, ".."))
    import ensemble as E
    import decide2 as D
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="single-seed OOF, no test, no save")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--base", type=int, default=None)
    ap.add_argument("--mult", type=str, default=None, help="comma list, e.g. 1,2,3,4")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--dropout", type=float, default=None)
    ap.add_argument("--dropout_mid", type=float, default=None)
    ap.add_argument("--seeds_oof", type=str, default=None)
    ap.add_argument("--seeds_test", type=str, default=None)
    a = ap.parse_args()

    cfg = {}
    if a.epochs is not None: cfg["epochs"] = a.epochs
    if a.base is not None: cfg["base"] = a.base
    if a.mult is not None: cfg["mult"] = tuple(int(x) for x in a.mult.split(","))
    if a.lr is not None: cfg["lr"] = a.lr
    if a.dropout is not None: cfg["dropout"] = a.dropout
    if a.dropout_mid is not None: cfg["dropout_mid"] = a.dropout_mid

    def _seeds(s, default):
        return tuple(int(x) for x in s.split(",")) if s else default

    if a.quick:
        seeds_oof = _seeds(a.seeds_oof, (0,))
        seeds_test = _seeds(a.seeds_test, ())
        save = False
    else:
        seeds_oof = _seeds(a.seeds_oof, (0, 1))
        seeds_test = _seeds(a.seeds_test, (0, 1, 2, 3))
        save = True

    t0 = time.time()
    oof, test = run(cfg=cfg, seeds_oof=seeds_oof, seeds_test=seeds_test, save=save)
    tr = C.load_split("train")
    s, at, ma = D.tune_decision(oof, tr["V"], tr["T"])
    fine = np.round(np.concatenate([np.arange(0.02, 0.15, 0.01), np.arange(0.15, 0.71, 0.025)]), 3)
    s2, th2 = E.tune_thresh(oof, tr["V"], tr["T"], grid=fine)
    print(f"\n>>> unet_deep group-CV {s:.4f} @ thr={at} min_area={ma}   "
          f"(tune_thresh {s2:.4f} @ {th2})   oof{oof.shape} test{test.shape}  ({time.time()-t0:.0f}s)",
          flush=True)
