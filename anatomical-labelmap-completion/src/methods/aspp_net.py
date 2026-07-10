"""
Multi-scale ASPP-style segmentation net -> soft per-cell class probabilities.

Architecturally DIFFERENT from both the U-Net (encoder/decoder, downsampling) and the
serial dilated-context CNN (a *chain* of increasing-dilation residual blocks). Here the
whole network runs at full 32x32 resolution and its core is a PARALLEL multi-dilation
(Atrous Spatial Pyramid Pooling) block: several 3x3 conv branches with different dilation
rates (3,6,9,12) run side-by-side, alongside a 1x1 (local) branch and an image-level
GLOBAL-context branch (global average pool -> 1x1 -> broadcast). Their outputs are
concatenated and fused by a 1x1 conv, wrapped in a residual connection with GroupNorm.
Three such blocks are stacked. Because every block simultaneously samples local + several
mid-range + whole-image receptive fields (rather than growing the field one block at a
time), it makes prediction errors that are decorrelated from the serial nets -> a useful
ensemble member. ~1.3M params.

Input encoding (identical to the other CNN members, for a fair blend):
  - a learned Embedding over the visible-label vocabulary (built from train+test V), applied
    to each of the 3 slices (prev/center/next) and concatenated channel-wise;
  - a center-is-zero mask channel (the hard-constraint region where a target can live).
NO absolute-coordinate channels, NO geometric augmentation, NO test-time augmentation --
all three were shown on volume-grouped CV to HURT (this is atlas-aligned data, so oriented
local anatomy is real signal that invariance/coords discard/overfit).

Output: 18 logits per cell -> softmax; class 0 = background. The hard center-zero constraint
is enforced downstream by the shared decision rule (decide2 / ensemble.decide).

Loss = background-down-weighted cross-entropy (bg weight 0.10) + smoothed soft multiclass
Dice over the 17 target classes, BOTH masked to center-zero cells (the only place a target
can appear), aligning the objective with the squared active-macro-IoU grader.

HONEST CV: the slabs are overlapping sliding windows, so a random split leaks adjacent
slices of the same volume across train/val. OOF here therefore uses the 5 canonical
VOLUME-GROUP folds (common.make_group_folds(5, seed=42)); no source volume spans a
train/val boundary, matching the volume-disjoint private test. OOF averages 2 seeds/fold;
TEST is a seed-ensemble trained on ALL 600 rows. Everything is seeded.
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
torch.backends.cudnn.benchmark = True  # fixed 32x32 input -> autotune is a free speedup


# ---------------------------------------------------------------------------
# Visible-label vocabulary (compact indices for the embedding table).
# Index 0 is reserved for unknown/pad; distinct visible labels map to 1.. .
# ---------------------------------------------------------------------------
def build_vocab():
    tr = C.load_split("train"); te = C.load_split("test")
    allv = np.concatenate([tr["V"].reshape(-1), te["V"].reshape(-1)])
    uv = np.unique(allv)
    lut = np.zeros(int(uv.max()) + 1, dtype=np.int64)
    for i, l in enumerate(uv, start=1):
        lut[int(l)] = i
    return lut, len(uv) + 1


def _valid_groups(groups, ch):
    g = max(1, min(groups, ch))
    while ch % g != 0:
        g -= 1
    return g


# ---------------------------------------------------------------------------
# Parallel multi-dilation (ASPP) block with a global-context branch + residual.
# ---------------------------------------------------------------------------
class ASPPBlock(nn.Module):
    def __init__(self, w, branch_ch, dilations=(3, 6, 9, 12), groups=8, p=0.15):
        super().__init__()
        gb = _valid_groups(groups, branch_ch)
        # local 1x1 branch
        self.b1 = nn.Conv2d(w, branch_ch, 1, bias=False)
        self.n1 = nn.GroupNorm(gb, branch_ch)
        # parallel dilated 3x3 branches
        self.dconv = nn.ModuleList(
            [nn.Conv2d(w, branch_ch, 3, padding=d, dilation=d, bias=False) for d in dilations])
        self.dnorm = nn.ModuleList([nn.GroupNorm(gb, branch_ch) for _ in dilations])
        # image-level global-context branch
        self.gpool = nn.AdaptiveAvgPool2d(1)
        self.gconv = nn.Conv2d(w, branch_ch, 1, bias=False)
        self.gnorm = nn.GroupNorm(gb, branch_ch)
        # fuse concatenated branches back to width w, residual add
        cat_ch = (2 + len(dilations)) * branch_ch          # 1x1 + global + dilated
        gw = _valid_groups(groups, w)
        self.fuse = nn.Conv2d(cat_ch, w, 1, bias=False)
        self.fnorm = nn.GroupNorm(gw, w)
        self.act = nn.GELU()
        self.drop = nn.Dropout2d(p)

    def forward(self, x):
        H, W = x.shape[-2], x.shape[-1]
        feats = [self.act(self.n1(self.b1(x)))]
        for conv, norm in zip(self.dconv, self.dnorm):
            feats.append(self.act(norm(conv(x))))
        g = self.act(self.gnorm(self.gconv(self.gpool(x))))
        feats.append(g.expand(-1, -1, H, W))
        y = torch.cat(feats, 1)
        y = self.drop(self.act(self.fnorm(self.fuse(y))))
        return x + y                                       # residual


class ASPPNet(nn.Module):
    def __init__(self, n_vocab, embed_dim=16, w=128, branch_ch=64, n_blocks=3,
                 dilations=(3, 6, 9, 12), groups=8, p_block=0.15, p_head=0.30, n_classes=18):
        super().__init__()
        self.emb = nn.Embedding(n_vocab, embed_dim, padding_idx=0)
        in_ch = 3 * embed_dim + 1                          # 3 embedded slices + center-zero mask
        gw = _valid_groups(groups, w)
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, w, 3, padding=1, bias=False), nn.GroupNorm(gw, w), nn.GELU())
        self.blocks = nn.ModuleList(
            [ASPPBlock(w, branch_ch, dilations, groups, p_block) for _ in range(n_blocks)])
        self.pre_head = nn.Sequential(
            nn.Conv2d(w, w, 3, padding=1, bias=False), nn.GroupNorm(gw, w), nn.GELU(),
            nn.Dropout2d(p_head))
        self.head = nn.Conv2d(w, n_classes, 1)

    def forward(self, idx, czmask):
        # idx: (B,3,32,32) long; czmask: (B,1,32,32) float
        B = idx.shape[0]
        e = self.emb(idx)                                  # (B,3,32,32,embed)
        e = e.permute(0, 1, 4, 2, 3).reshape(B, -1, 32, 32)
        x = torch.cat([e, czmask], 1)
        x = self.stem(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.pre_head(x)
        return self.head(x)                                # (B,18,32,32) logits


# ---------------------------------------------------------------------------
# Loss (masked to center-zero cells): bg-down-weighted CE + soft multiclass Dice.
# ---------------------------------------------------------------------------
def compute_loss(logits, y, cz, ce_weight_vec, ce_w=1.0, dice_w=1.0, dice_eps=1.0):
    czf = cz.float()                                       # (B,1,H,W)
    m = czf.squeeze(1)                                     # (B,H,W)
    denom = m.sum().clamp_min(1.0)
    ce = F.cross_entropy(logits, y, weight=ce_weight_vec, reduction="none")
    ce = (ce * m).sum() / denom
    prob = F.softmax(logits, dim=1)
    onehot = F.one_hot(y, logits.shape[1]).permute(0, 3, 1, 2).float()
    p = prob[:, 1:] * czf                                  # predicted target mass
    g = onehot[:, 1:] * czf                                # truth target mass
    inter = (p * g).sum(dim=(2, 3))
    tot = p.sum(dim=(2, 3)) + g.sum(dim=(2, 3))
    dice = (2 * inter + dice_eps) / (tot + dice_eps)
    return ce_w * ce + dice_w * (1.0 - dice.mean())


# ---------------------------------------------------------------------------
# Data tensors.
# ---------------------------------------------------------------------------
def make_tensors(V, lut, Tidx=None):
    idx = torch.from_numpy(lut[V.astype(np.int64)]).long()               # (N,3,32,32)
    cz = torch.from_numpy((V[:, 1] == 0)[:, None].astype(np.float32))    # (N,1,32,32)
    y = None if Tidx is None else torch.from_numpy(Tidx.astype(np.int64))
    return idx, cz, y


def _seed_all(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_model(idx_tr, cz_tr, y_tr, n_vocab, cfg, seed):
    _seed_all(seed)
    model = ASPPNet(n_vocab, embed_dim=cfg["embed_dim"], w=cfg["w"], branch_ch=cfg["branch_ch"],
                    n_blocks=cfg["n_blocks"], dilations=cfg["dilations"], groups=cfg["groups"],
                    p_block=cfg["p_block"], p_head=cfg["p_head"]).to(DEVICE)
    wvec = torch.ones(C.NUM_CLASSES, device=DEVICE)
    wvec[0] = cfg["bg_weight"]
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    epochs, bs = cfg["epochs"], cfg["batch_size"]
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    N = idx_tr.shape[0]
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
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        sched.step()
    model.eval()
    return model


@torch.no_grad()
def predict_proba(model, idx, cz, bs=256):
    idx = idx.to(DEVICE); cz = cz.to(DEVICE)
    out = np.zeros((idx.shape[0], C.NUM_CLASSES, 32, 32), np.float32)
    model.eval()
    for s in range(0, idx.shape[0], bs):
        prob = F.softmax(model(idx[s:s + bs], cz[s:s + bs]), dim=1)
        out[s:s + bs] = prob.cpu().numpy()
    return out


DEFAULT_CFG = dict(
    embed_dim=16, w=128, branch_ch=64, n_blocks=3, dilations=(3, 6, 9, 12), groups=8,
    p_block=0.15, p_head=0.30,
    lr=2e-3, wd=5e-4, epochs=150, batch_size=64,
    bg_weight=0.10, ce_w=1.0, dice_w=1.0, dice_eps=1.0,
)


def run(save=True, name="aspp_net", cfg=None, seeds_oof=(0, 1), seeds_test=(0, 1, 2), verbose=True):
    """HONEST volume-group OOF + all-600 seed-ensemble TEST proba. Returns (oof, test).

    OOF: 5 canonical volume-group folds (make_group_folds), 2 seeds/fold averaged, so no
    source volume spans a train/val boundary (the private test is volume-disjoint).
    TEST: seed-ensemble trained on ALL 600 rows.
    """
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    lut, n_vocab = build_vocab()
    tr = C.load_split("train"); te = C.load_split("test")
    V, Tidx = tr["V"], tr["Tidx"]
    N = len(V)
    idx_all, cz_all, y_all = make_tensors(V, lut, Tidx)

    oof = np.zeros((N, C.NUM_CLASSES, 32, 32), np.float32)
    t0 = time.time()
    for fi, (tri, vai) in enumerate(C.make_group_folds(5, seed=42)):
        acc = np.zeros((len(vai), C.NUM_CLASSES, 32, 32), np.float32)
        for sd in seeds_oof:
            model = train_model(idx_all[tri], cz_all[tri], y_all[tri], n_vocab, cfg,
                                 seed=1000 * sd + fi)
            acc += predict_proba(model, idx_all[vai], cz_all[vai])
        oof[vai] = acc / len(seeds_oof)
        if verbose:
            print(f"  [oof] group-fold {fi} done ({time.time()-t0:.0f}s)", flush=True)

    idx_te, cz_te, _ = make_tensors(te["V"], lut)
    test = np.zeros((len(te["V"]), C.NUM_CLASSES, 32, 32), np.float32)
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
    sys.path.insert(0, os.path.join(_HERE, ".."))
    import decide2 as D
    oof, test = run()
    tr = C.load_split("train")
    s, at, ma = D.tune_decision(oof, tr["V"], tr["T"])
    print(f"aspp_net GROUP-CV {s:.4f} @ thr={at} min_area={ma}; oof{oof.shape} test{test.shape}")
