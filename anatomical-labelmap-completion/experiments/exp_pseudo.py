"""
Transductive pseudo-labeling / test-time adaptation, validated honestly on group CV.

Motivation: test scores ~0.022 ABOVE group-CV -- more training volumes generalize better. So adding
the target volumes' anatomy to training (via the model's own pseudo-labels) should help the disjoint
domain. Validation mimics the test scenario: for each group fold, pseudo-label the held-out val
VOLUMES with a first-stage model, retrain on train(real) + val(pseudo), then predict val. If group-CV
rises, the technique transfers to test. Uses only inputs + model predictions (no answers/metadata).
"""
import sys, os, time
import numpy as np
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src")); sys.path.insert(0, os.path.join(_ROOT, "experiments"))
import common as C, decide2 as D, postprocess as P
import volume as VOL
import torch, torch.nn.functional as F
from exp_deep import DeepUNet
from exp_context import build_vocab
from exp_elastic import rand_grid, warp_labels
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _train(net, idxA, Tidx, sample_ids, K, bg, epochs, elastic_a, res, gen, seed, sample_w=None):
    torch.manual_seed(seed); np.random.seed(seed)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=5e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    w = torch.ones(18, device=DEV); w[0] = 0.10; r = (K - 1) // 2
    Xi = torch.tensor(idxA[sample_ids], device=DEV); Yi = torch.tensor(Tidx[sample_ids], device=DEV)
    W = None if sample_w is None else torch.tensor(sample_w[sample_ids], device=DEV, dtype=torch.float32)
    n = len(sample_ids); bs = 128
    for ep in range(epochs):
        net.train(); perm = torch.randperm(n, device=DEV)
        for s in range(0, n, bs):
            b = perm[s:s + bs]; xb = Xi[b]; yb = Yi[b]
            g = rand_grid(len(b), 32, 32, elastic_a, res, DEV, gen)
            xb = warp_labels(xb, g); yb = warp_labels(yb[:, None], g)[:, 0]
            cz = (xb[:, r:r + 1] == bg).float(); mm = cz[:, 0]
            lg = net(xb, cz); prob = lg.softmax(1)
            ce_map = F.cross_entropy(lg, yb, weight=w, reduction="none") * mm
            if W is not None:
                ce_map = ce_map * W[b][:, None, None]
            ce = ce_map.sum() / mm.sum().clamp_min(1)
            oh = F.one_hot(yb, 18).permute(0, 3, 1, 2).float(); pp = prob[:, 1:] * cz; gg = oh[:, 1:] * cz
            dice = 1 - ((2 * (pp * gg).sum((0, 2, 3)) + 1) / ((pp + gg).sum((0, 2, 3)) + 1)).mean()
            (ce + dice).backward(); opt.step(); opt.zero_grad()
        sch.step()
    return net


@torch.no_grad()
def _pred(net, idxA, ids, K, bg):
    r = (K - 1) // 2; out = np.zeros((len(ids), 18, 32, 32), np.float32)
    for s in range(0, len(ids), 128):
        b = ids[s:s + 128]; Xv = torch.tensor(idxA[b], device=DEV)
        out[s:s + len(b)] = net(Xv, (Xv[:, r:r + 1] == bg).float()).softmax(1).cpu().numpy()
    return out


def run(K=11, base=40, epochs=150, seed=0, pseudo_w=0.5, rounds=1):
    lut, vocab = build_vocab(); bg = int(lut[0])
    tr = C.load_split("train"); T = tr["T"]; Tidx = tr["Tidx"].astype(np.int64)
    ctx, _, _ = VOL.extended_context(tr["V"], K); idxA = lut[ctx.astype(np.int64)]
    N = len(idxA)
    oof_std = np.zeros((N, 18, 32, 32), np.float32)
    oof_ps = np.zeros((N, 18, 32, 32), np.float32)
    t0 = time.time()
    for tri, vai in C.make_group_folds(5, 42):
        gen = torch.Generator(device=DEV); gen.manual_seed(seed)
        # stage 1
        net = DeepUNet(vocab, K, base=base, attn=False).to(DEV)
        _train(net, idxA, Tidx, tri, K, bg, epochs, 3.0, 4, gen, seed)
        net.eval(); oof_std[vai] = _pred(net, idxA, vai, K, bg)
        # pseudo-label val volumes, retrain on tri(real)+vai(pseudo)
        pseudo = Tidx.copy().astype(np.int64)
        for rd in range(rounds):
            pv = oof_std[vai] if rd == 0 else _pred(net, idxA, vai, K, bg)
            for j, i in enumerate(vai):
                cz = tr["V"][i, 1] == 0
                fg = pv[j, 1:]; best = fg.argmax(0) + 1; bp = fg.max(0)
                pseudo[i] = np.where((bp >= 0.30) & cz, best, 0)
            allids = np.concatenate([tri, vai])
            sw = np.ones(N, np.float32); sw[vai] = pseudo_w
            net = DeepUNet(vocab, K, base=base, attn=False).to(DEV)
            _train(net, idxA, pseudo, allids, K, bg, epochs, 3.0, 4, gen, seed + 1, sample_w=sw)
            net.eval()
        oof_ps[vai] = _pred(net, idxA, vai, K, bg)
    for tag, o in [("standard", oof_std), ("pseudo", oof_ps)]:
        b = o / (o.sum(1, keepdims=True) + 1e-9)
        s = D.tune_decision(P.smooth3d(b, tr["V"], 0.7, 3), tr["V"], T)[0]
        print(f"{tag:10s} K={K} pw={pseudo_w} r={rounds}: group-CV {s:.4f}", flush=True)
    print(f"(elapsed {time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    run(K=11, epochs=140, pseudo_w=0.5, rounds=1)
    run(K=11, epochs=140, pseudo_w=1.0, rounds=1)
