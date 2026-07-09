"""
Dilated / residual context-aggregation CNN -> soft per-cell class probabilities.

Architecturally DIFFERENT from a U-Net (no encoder/decoder, no heavy downsampling):
the whole network runs at full 32x32 resolution and aggregates a wide anatomical
neighbourhood through a stack of residual conv blocks with INCREASING DILATION
(1,2,4,8,...). This decorrelates it from a U-Net in the ensemble.

Input encoding (per row):
  - a learned Embedding over the visible-label vocabulary (built from train+test V),
    applied to each of the 3 slices (prev / center / next) and concatenated;
  - a center-is-zero mask channel (the hard-constraint region where targets can live);
  - normalized (x,y) coordinate channels.

Output: 18 logits per cell -> softmax -> proba[c,i,j] with class 0 = background.
The hard center-zero constraint is enforced downstream by ensemble.decide.

OOF is produced with the 5 canonical folds (train fold-train only, predict fold-val).
TEST is the average of the 5 fold models (k-fold bag). Seeded for reproducibility.
"""
import os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
import common as C

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Visible-label vocabulary (compact indices for the embedding table).
# ---------------------------------------------------------------------------
def build_vocab():
    tr = C.load_split("train")
    te = C.load_split("test")
    vals = np.unique(np.concatenate([tr["V"].reshape(-1), te["V"].reshape(-1)]))
    vals = vals.astype(np.int64)                      # sorted ascending; 0 is first -> index 0
    lut = np.zeros(int(vals.max()) + 1, dtype=np.int64)
    for i, v in enumerate(vals):
        lut[int(v)] = i
    return lut, len(vals)


def encode_V(V, lut):
    """(N,3,32,32) opaque visible labels -> (N,3,32,32) compact vocab indices (int64)."""
    return lut[V.astype(np.int64)]


# ---------------------------------------------------------------------------
# Network: residual dilated conv blocks at full resolution.
# ---------------------------------------------------------------------------
class ResDilBlock(nn.Module):
    def __init__(self, w, d, groups, p):
        super().__init__()
        self.n1 = nn.GroupNorm(groups, w)
        self.c1 = nn.Conv2d(w, w, 3, padding=d, dilation=d)
        self.n2 = nn.GroupNorm(groups, w)
        self.c2 = nn.Conv2d(w, w, 3, padding=d, dilation=d)
        self.drop = nn.Dropout2d(p)
        self.act = nn.GELU()

    def forward(self, x):
        h = self.c1(self.act(self.n1(x)))
        h = self.c2(self.drop(self.act(self.n2(h))))
        return x + h


class DilCNN(nn.Module):
    def __init__(self, n_vocab, emb=16, w=96, dilations=(1, 2, 4, 8, 4, 2, 1),
                 groups=8, p=0.10):
        super().__init__()
        self.emb = nn.Embedding(n_vocab, emb)
        in_ch = 3 * emb + 1 + 2
        self.stem = nn.Conv2d(in_ch, w, 3, padding=1)
        self.blocks = nn.ModuleList([ResDilBlock(w, d, groups, p) for d in dilations])
        self.head_n = nn.GroupNorm(groups, w)
        self.act = nn.GELU()
        self.head = nn.Conv2d(w, C.NUM_CLASSES, 1)
        ys, xs = torch.meshgrid(torch.linspace(-1, 1, 32),
                                torch.linspace(-1, 1, 32), indexing="ij")
        self.register_buffer("coord", torch.stack([xs, ys], 0).unsqueeze(0))  # (1,2,32,32)

    def forward(self, vidx):
        B = vidx.shape[0]
        e = self.emb(vidx)                              # (B,3,32,32,emb)
        e = e.permute(0, 1, 4, 2, 3).reshape(B, -1, 32, 32)
        czm = (vidx[:, 1:2] == 0).float()               # (B,1,32,32) center-zero mask
        x = torch.cat([e, czm, self.coord.expand(B, -1, -1, -1)], 1)
        x = self.stem(x)
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.act(self.head_n(x)))      # (B,18,32,32) logits


# ---------------------------------------------------------------------------
# Loss: masked weighted CE + batch-pooled soft-Dice over the 17 target classes.
# ---------------------------------------------------------------------------
def _loss(logits, tgt_idx, czmask, class_w, dice_w):
    # czmask: (B,1,32,32) float (1 on center-zero cells, where predictions are made).
    ce = F.cross_entropy(logits, tgt_idx, weight=class_w, reduction="none")   # (B,32,32)
    ce = (ce * czmask[:, 0]).sum() / czmask.sum().clamp_min(1.0)

    p = F.softmax(logits, 1) * czmask                    # (B,18,32,32)
    y = F.one_hot(tgt_idx, C.NUM_CLASSES).permute(0, 3, 1, 2).float() * czmask
    pf, yf = p[:, 1:], y[:, 1:]                          # target classes only
    dims = (0, 2, 3)
    inter = (pf * yf).sum(dims)
    denom = pf.sum(dims) + yf.sum(dims)
    dice = (2 * inter + 1.0) / (denom + 1.0)
    return ce + dice_w * (1.0 - dice.mean())


def _class_weights(bg_weight):
    w = np.ones(C.NUM_CLASSES, dtype=np.float32)
    w[0] = bg_weight
    return torch.tensor(w, device=DEVICE)


def _make_model(n_vocab, cfg):
    return DilCNN(n_vocab, emb=cfg["emb"], w=cfg["w"], dilations=cfg["dilations"],
                  groups=cfg["groups"], p=cfg["dropout"]).to(DEVICE)


def _train(model, vidx_tr, tidx_tr, cfg, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    n = vidx_tr.shape[0]
    class_w = _class_weights(cfg["bg_weight"])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    epochs, bs = cfg["epochs"], cfg["bs"]
    steps = epochs * max(1, (n + bs - 1) // bs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    model.train()
    rng = np.random.RandomState(seed)
    for ep in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, bs):
            b = order[s:s + bs]
            vb = vidx_tr[b]
            tb = tidx_tr[b]
            czm = (vb[:, 1:2] == 0).float()
            logits = model(vb)
            loss = _loss(logits, tb, czm, class_w, cfg["dice_w"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
    return model


@torch.no_grad()
def _predict(model, vidx, bs=256):
    model.eval()
    out = np.zeros((vidx.shape[0], C.NUM_CLASSES, 32, 32), np.float32)
    for s in range(0, vidx.shape[0], bs):
        vb = vidx[s:s + bs]
        p = F.softmax(model(vb), 1)
        out[s:s + bs] = p.cpu().numpy()
    return out


DEFAULT_CFG = dict(
    emb=16, w=96, dilations=(1, 2, 4, 8, 4, 2, 1), groups=8, dropout=0.10,
    bs=64, lr=2e-3, wd=5e-4, epochs=90, bg_weight=0.10, dice_w=1.0,
    test_seeds=3,
)


def run_config(cfg=None, seed=42, verbose=True):
    """HONEST group-fold OOF + all-600-trained TEST proba. Returns (oof, test).

    OOF: the 5 canonical VOLUME-GROUP folds (make_group_folds) so no source volume
    is ever split across train and val -- the private test set is volume-disjoint, so
    this is the only honest estimate. TEST: a small seed-ensemble trained on ALL 600
    rows (the group folds are only for honest scoring, not for the deliverable model).
    """
    if cfg is None:
        cfg = dict(DEFAULT_CFG)
    t0 = time.time()
    lut, n_vocab = build_vocab()
    tr = C.load_split("train")
    te = C.load_split("test")
    N = len(tr["V"])

    vidx_tr_all = torch.tensor(encode_V(tr["V"], lut), device=DEVICE)          # (N,3,32,32)
    tidx_all = torch.tensor(tr["Tidx"].astype(np.int64), device=DEVICE)        # (N,32,32)
    vidx_te = torch.tensor(encode_V(te["V"], lut), device=DEVICE)              # (Nte,3,32,32)

    # ---- OOF on honest volume-group folds ----
    oof = np.zeros((N, C.NUM_CLASSES, 32, 32), np.float32)
    for fi, (tri, vai) in enumerate(C.make_group_folds(5, seed=42)):
        tri_t = torch.tensor(tri, device=DEVICE)
        model = _make_model(n_vocab, cfg)
        _train(model, vidx_tr_all[tri_t], tidx_all[tri_t], cfg, seed=seed + fi)
        oof[vai] = _predict(model, vidx_tr_all[torch.tensor(vai, device=DEVICE)])
        if verbose:
            print(f"  [oof] group-fold {fi} done ({time.time()-t0:.0f}s)")

    # ---- TEST: seed-ensemble trained on ALL 600 rows ----
    test_acc = np.zeros((len(te["V"]), C.NUM_CLASSES, 32, 32), np.float32)
    ns = cfg.get("test_seeds", 3)
    for si in range(ns):
        model = _make_model(n_vocab, cfg)
        _train(model, vidx_tr_all, tidx_all, cfg, seed=seed + 100 + si)
        test_acc += _predict(model, vidx_te)
        if verbose:
            print(f"  [test] all-600 seed {si} done ({time.time()-t0:.0f}s)")
    test = test_acc / ns
    if verbose:
        print(f"  run_config elapsed {time.time()-t0:.0f}s")
    return oof, test


def run(save=True, name="cnn_context", seed=42, cfg=None):
    import ensemble as E
    oof, test = run_config(cfg, seed=seed, verbose=True)
    tr = C.load_split("train")
    s, th = E.tune_thresh(oof, tr["V"], tr["T"])
    print(f"cnn_context OOF score {s:.4f} @ thr={th}  oof{oof.shape} test{test.shape}")
    if save:
        E.save_preds(name, oof, test)
        print(f"saved artifacts/preds/{name}.npz")
    return oof, test, s, th


if __name__ == "__main__":
    run()
