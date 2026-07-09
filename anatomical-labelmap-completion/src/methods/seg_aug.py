"""
Augmented U-Net segmentation over embedded label maps -> 18-class per-cell probabilities.

Focus: cross-volume GENERALIZATION (honest group CV). Key ingredients:
  - learned embedding of opaque visible labels (vocab from train+test inputs),
  - loss masked to center-zero cells (only place targets can live), weighted CE + soft Dice,
  - heavy dihedral (8-orientation) augmentation applied jointly to input slabs + target,
  - 8-orientation test-time augmentation.
"""
import os, sys, time
import numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
import common as C
import torch
import torch.nn as nn
import torch.nn.functional as F

DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------- vocab ----------------
def build_vocab():
    tr = C.load_split("train"); te = C.load_split("test")
    labs = set(np.unique(tr["V"]).tolist()) | set(np.unique(te["V"]).tolist())
    labs = sorted(labs)
    vocab = {0: 0}
    for l in labs:
        if l not in vocab:
            vocab[l] = len(vocab)
    maxlab = max(labs)
    lut = np.zeros(maxlab + 1, dtype=np.int64)
    for l, i in vocab.items():
        lut[l] = i
    return lut, len(vocab)


# ---------------- model ----------------
class UNet(nn.Module):
    def __init__(self, vocab, emb=16, base=64, ncls=18):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb)
        cin = 3 * emb + 3  # 3 embedded slices + czmask + 2 coords

        def blk(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1), nn.GroupNorm(8, o), nn.ReLU(inplace=True),
                nn.Conv2d(o, o, 3, padding=1), nn.GroupNorm(8, o), nn.ReLU(inplace=True))

        self.e1 = blk(cin, base)
        self.e2 = blk(base, base * 2)
        self.bott = blk(base * 2, base * 4)
        self.d2 = blk(base * 4 + base * 2, base * 2)
        self.d1 = blk(base * 2 + base, base)
        self.drop = nn.Dropout2d(0.1)
        self.head = nn.Conv2d(base, ncls, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, idx, cz, coord):
        # idx: (B,3,H,W) long ; cz: (B,1,H,W) ; coord: (B,2,H,W)
        B, _, H, W = idx.shape
        e = self.emb(idx)                       # (B,3,H,W,emb)
        e = e.permute(0, 1, 4, 2, 3).reshape(B, -1, H, W)
        x = torch.cat([e, cz, coord], 1)
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        b = self.bott(self.pool(e2))
        d2 = self.d2(torch.cat([F.interpolate(b, scale_factor=2, mode="nearest"), e2], 1))
        d1 = self.d1(torch.cat([F.interpolate(d2, scale_factor=2, mode="nearest"), e1], 1))
        return self.head(self.drop(d1))


# ---------------- dihedral transforms ----------------
def dihedral(arr, t, axes=(-2, -1)):
    r, f = t % 4, t // 4
    a = np.rot90(arr, r, axes=axes)
    if f:
        a = np.flip(a, axis=axes[-1])
    return np.ascontiguousarray(a)


def dihedral_inv_logits(logit, t):
    # invert transform t on a (ncls,H,W) array
    r, f = t % 4, t // 4
    a = logit
    if f:
        a = np.flip(a, axis=-1)
    a = np.rot90(a, -r, axes=(-2, -1))
    return np.ascontiguousarray(a)


COORD = None
def _coord():
    global COORD
    if COORD is None:
        ys, xs = np.mgrid[0:32, 0:32]
        COORD = np.stack([(ys - 15.5) / 15.5, (xs - 15.5) / 15.5]).astype(np.float32)
    return COORD


def _to_tensors(Vidx, cz, coord):
    idx = torch.as_tensor(Vidx, dtype=torch.long, device=DEV)
    czt = torch.as_tensor(cz, dtype=torch.float32, device=DEV).unsqueeze(1)
    co = torch.as_tensor(np.broadcast_to(coord, (len(Vidx), 2, 32, 32)).copy(), device=DEV)
    return idx, czt, co


def soft_dice_loss(prob, tgt_oh, cz):
    # prob,tgt_oh: (B,C,H,W); cz mask (B,1,H,W). foreground classes 1..C-1
    p = prob[:, 1:] * cz
    t = tgt_oh[:, 1:] * cz
    num = 2 * (p * t).sum((0, 2, 3)) + 1.0
    den = (p + t).sum((0, 2, 3)) + 1.0
    return (1 - num / den).mean()


def train_model(Vidx_tr, cz_tr, Tidx_tr, vocab, epochs=120, lr=2e-3, wd=1e-4,
                bg_w=0.1, seed=0, log=False):
    torch.manual_seed(seed); np.random.seed(seed)
    net = UNet(vocab).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    coord = _coord()
    w = torch.ones(18, device=DEV); w[0] = bg_w
    n = len(Vidx_tr)
    bs = 64
    for ep in range(epochs):
        net.train()
        perm = np.random.permutation(n)
        for s in range(0, n, bs):
            bidx = perm[s:s + bs]
            t = np.random.randint(0, 8)                     # one dihedral per batch
            vb = dihedral(Vidx_tr[bidx], t)                 # (b,3,H,W)
            cb = dihedral(cz_tr[bidx], t)
            tb = dihedral(Tidx_tr[bidx], t)
            idx, czt, co = _to_tensors(vb, cb, coord)
            tgt = torch.as_tensor(tb, dtype=torch.long, device=DEV)
            logit = net(idx, czt, co)
            ce = F.cross_entropy(logit, tgt, weight=w, reduction="none")
            ce = (ce * czt[:, 0]).sum() / (czt.sum() + 1e-6)
            prob = logit.softmax(1)
            tgt_oh = F.one_hot(tgt, 18).permute(0, 3, 1, 2).float()
            dl = soft_dice_loss(prob, tgt_oh, czt)
            loss = ce + dl
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    return net


@torch.no_grad()
def predict_tta(net, Vidx, cz, tta=True):
    net.eval()
    coord = _coord()
    acc = np.zeros((len(Vidx), 18, 32, 32), np.float32)
    ts = range(8) if tta else [0]
    for t in ts:
        vb = dihedral(Vidx, t); cb = dihedral(cz, t)
        idx, czt, co = _to_tensors(vb, cb, coord)
        pr = net(idx, czt, co).softmax(1).cpu().numpy()
        for i in range(len(Vidx)):
            acc[i] += dihedral_inv_logits(pr[i], t)
    acc /= len(list(ts))
    return acc


def run(epochs=120, seed=0, save=True, name="seg_aug", tta=True, folds=None, verbose=True):
    lut, vocab = build_vocab()
    tr = C.load_split("train"); te = C.load_split("test")
    Vidx = lut[tr["V"].astype(np.int64)]          # (N,3,32,32)
    cz = (tr["V"][:, 1] == 0).astype(np.float32)  # (N,32,32)
    Tidx = tr["Tidx"].astype(np.int64)
    N = len(Vidx)
    if folds is None:
        folds = C.make_group_folds(5, 42)

    oof = np.zeros((N, 18, 32, 32), np.float32)
    t0 = time.time()
    for fi, (tri, vai) in enumerate(folds):
        net = train_model(Vidx[tri], cz[tri], Tidx[tri], vocab, epochs=epochs, seed=seed)
        oof[vai] = predict_tta(net, Vidx[vai], cz[vai], tta=tta)
        if verbose:
            import ensemble as E
            s, th = E.tune_thresh(oof[vai], tr["V"][vai], tr["T"][vai])
            print(f"  fold {fi}: val {s:.4f}@{th}  ({time.time()-t0:.0f}s)")

    import ensemble as E
    s, th = E.tune_thresh(oof, tr["V"], tr["T"])
    print(f"seg_aug group-CV OOF {s:.4f} @ thr={th}  ({time.time()-t0:.0f}s)")

    # test: train on ALL
    Vte = lut[te["V"].astype(np.int64)]
    czte = (te["V"][:, 1] == 0).astype(np.float32)
    net = train_model(Vidx, cz, Tidx, vocab, epochs=epochs, seed=seed)
    test = predict_tta(net, Vte, czte, tta=tta)
    if save:
        E2 = __import__("ensemble"); E2.save_preds(name, oof, test)
    return oof, test, s, th


if __name__ == "__main__":
    ep = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    run(epochs=ep)
