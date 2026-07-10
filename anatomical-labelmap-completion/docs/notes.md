# Small Anatomical Labelmap Completion: findings & strategy

## Task
- Input per row: 3 adjacent 32×32 **integer** label maps (prev / center / next slice), with 17 target
  structures removed (set to 0). Output: the center slice's removed target labels as a 32×32 map
  (0 background, or one of 17 fixed opaque target ids).
- Metric: mean over rows of `(active_macro_IoU)^2`, where the macro-IoU averages per-label IoU over
  every nonzero label present in **either** prediction or truth. Empty-vs-empty row = 1.0.
  Squaring + "labels present in either" makes both spurious labels and missed labels expensive.

## Key EDA findings (see experiments/ + scratch EDA)
1. **Label space is fixed**: the same 17 target ids appear as `target_label_ids` in every train & test
   row. Test visible labels (66) ⊆ train visible labels (71). 18 output classes total.
2. **Targets never appear in any visible slice** (0.0% in prev/center/next). This is atlas-based label
   *inpainting*, not cross-slice copy/interpolation.
3. **Hard constraint**: 100% of target cells are `0` in the center-visible map. Never predict a target
   on a cell where center-visible ≠ 0. (Center-zero cells are ~82% of the grid; only 2.6% are targets.)
4. **Position alone is nearly useless** (per-cell argmax CV ≈ 0.00): the target region is a tiny 2.6%
   of the large background-zero area, so a positional prior can't localize it.
5. **Cross-slice hole alignment barely helps** (P(target | center-zero) = 2.6%, vs 2.6% if prev&next
   also zero), background is background across all three slices.
6. **Targets are not enclosed holes**: 77% of target cells have zero nonzero 4-neighbors; 95% lie in
   the exterior background component. Morphology alone won't localize them.
7. **Retrieval is the breakthrough**: copying the target map from the most-similar training row (by
   visible-cell agreement) scores **CV ≈ 0.67 (full slab) / 0.69 (center-only)**. The visible→hidden
   mapping is highly deterministic and the anatomy recurs across rows.

## Strategy
Treat as instance-based retrieval + learned refinement (legitimate: uses the intended visible-map
input, not ids/row-order/fingerprints):
- **k-NN voting** over similar rows (tune k, distance, slice weighting, per-label thresholds).
- **Learned U-Net / CNN** over embedded label maps → 18-class per-cell segmentation, masked to
  center-zero cells, trained with CE + region (Dice/Lovász) loss.
- **Ensemble** k-NN + CNN; calibrate per-label inclusion to the squared-macro-IoU metric.
- Validate with fixed 5-fold CV (seed 42) via `src/common.make_folds`. Guard against overfitting
  (single-1-NN overfits; k-NN voting + CNN generalize).

## ⚠️ CRITICAL: the random-fold CV is LEAKY (adjacency leak)
The slabs are **overlapping sliding windows** over source volumes: 523/600 train rows' prev/next
visible slice exactly equals another train row's center slice. A random split therefore places
adjacent slices of the same volume in both train and val, and retrieval simply copies its near-
duplicate neighbour.

- Reconstructed **61 volumes** by unioning rows that share any visible slice (`common.reconstruct_groups`).
- The private **TEST set is volume-disjoint** from train (0/300 exact center matches; nearest-neighbour
  cell-agreement 0.84 on test vs 0.98 on random train-holdout). So **volume-grouped CV
  (`common.make_group_folds`) is the only honest estimate of test performance.**
- k-NN retrieval: **0.7056 random-CV → 0.0483 group-CV**. The 0.70 was entirely leak.
- Local ray-cast "socket" signature (translation-invariant): group-CV 0.02.

**Consequence:** retrieval / whole-slab nearest-neighbour is useless on the real test. The task is
genuine **cross-volume generalization**: learn the visible-anatomy → hidden-target mapping that holds
on unseen volumes. All model selection, thresholds, and ensembling use **group CV**. Retrieval methods
are dropped from the final ensemble unless they demonstrably generalize on group CV.

## Baselines
| Method | random-CV (leaky) | group-CV (honest) |
|---|---|---|
| predict empty | 0.000 | 0.000 |
| positional argmax | ~0.000 | ~0.000 |
| 1-NN copy (center-only) | 0.691 |  |
| k-NN soft vote (k=3) | 0.7056 | 0.0483 |
| local ray-cast signature |  | 0.020 |

## Learned-model honest results (volume-grouped CV)
| Method | group-CV |
|---|---|
| k-NN soft vote (k=3) | 0.0483 |
| augmented U-Net (embeddings, masked CE+Dice, D4 aug + 8-pose TTA) | 0.0348 |
| **blend k-NN + U-Net (0.5/0.5)** | 0.0540 |
| **+ 3D volume-consistency smoothing (a=0.7, w=2)** | **0.0567** |

The learned CNN *under*performs retrieval on unseen volumes, i.e. there is little volume-invariant
rule to learn; most of the signal is instance similarity that does not transfer. Blending + 3D
continuity extract a bit more. Extrapolating k-NN's per-row score onto the test neighbour-similarity
distribution gives an expected **test score ~0.03 to 0.05** (test is slightly harder than group-holdout).

## Final pipeline (solution.py)
k-NN retrieval proba + augmented-U-Net seed-ensemble proba (trained on all rows, TTA) → 0.5/0.5 blend
→ 3D slice-chain smoothing → center-zero-constrained decision (thr 0.375). All params tuned on honest
group CV. Retrieval-forward, so it also captures any test↔train similarity if the test were less
disjoint than measured. Runs end-to-end from ./dataset/public/ in a few minutes.

## Operational notes
- Local background jobs are killed at ~600s; the context-CNN (dilated, aug+TTA) and per-cell GBDT
  are correct and honest but exceed that locally. They are available as ensemble members
  (src/methods/cnn_context.py, percell_gbdt.py) and fit the platform's 30-min A10G budget, but add
  only marginal group-CV over the k-NN+U-Net+3D core, so the shipped solution keeps the fast core.

## FINAL (supersedes the table above): augmentation ablation + best ensemble
Key correction: geometric **augmentation HURTS** on this atlas data, the oriented local
configuration of anatomy around a hole is real signal, and D4 rotations/flips/translations,
8-pose TTA, and absolute-coordinate channels all destroy it. Removing them lifted both CNNs a lot:

| Method (volume-grouped CV) | group-CV |
|---|---|
| k-NN soft vote (k=3) | 0.0483 |
| U-Net, augmented (D4 + TTA) | 0.0348 |
| **U-Net, plain (no aug/TTA/coords, dropout 0.30)** | **0.0640** |
| dilated-context CNN, augmented | 0.0386 |
| **dilated-context CNN, plain** | **0.0601** |
| per-cell HistGBDT (local features) | 0.0330 |
| U-Net + dilated-context (0.5/0.5) + 3D | 0.0770 |
| **U-Net 0.45 + dilated-context 0.45 + k-NN 0.10 + 3D (a=0.7,w=3,thr=0.225)** | **0.0791** |

Final pipeline = the last row (solution.py). +39% over the pre-ablation 0.0567. k-NN kept at a
small weight: it adds group-CV value AND hedges the (unlikely) case that the private test is less
volume-disjoint than measured. Expected honest test ~0.05-0.08 (test ~as hard as held-out volumes).
GBDT (0.033) not selected. All params tuned on group CV; no aug/TTA in the final models.

## Final ensemble update (4 models)
Adding a higher-capacity plain U-Net (base 72, single group-CV 0.0614) as a 4th, decorrelated
member gives a small but consistent lift:

| Ensemble (+ 3D smoothing) | group-CV |
|---|---|
| U-Net(48) .45 + dilated .45 + kNN .10 | 0.0791 |
| **U-Net(48) .30 + U-Net(72) .30 + dilated .30 + kNN .10** | **0.0804** |

Shipped solution.py = the 4-model row (thr 0.25, smooth a=0.7 w=3). Honest group-CV **0.0804**
(+42% over the pre-ablation 0.0567; +66% over the k-NN baseline 0.0483). Two plain U-Nets of
different width + a dilated-context net + a small k-NN weight; variance reduction from the extra
decorrelated model. No augmentation/TTA anywhere (verified harmful on this atlas data).

## Deep-dive research (pushing past the 0.08 plateau; leader at 0.138)
Confirmed the test is genuinely volume-disjoint (0/300 slice matches at ANY position; test NN
similarity 0.84 == group-holdout 0.85; volume reconstruction is clean, each slice in <=3 rows).
So group-CV is well-calibrated and 0.138 is achievable on disjoint volumes by a stronger method.

**Levers tested on honest group-CV (single-seed unless noted):**
| Lever | result |
|---|---|
| wider 3D context (reconstruct volume, K-slice window): K=3→7→9→11 | 0.054 → 0.067 → 0.077 (helps) |
| elastic deformation aug (simulates cross-subject variation) | +0.009 (helps; D4/affine HURT) |
| **deep U-Net (3 downsamples, near-global receptive field), base=40** | **0.083 (best single)** |
| base sweep: 32 / 40 / 48 / 56 | 0.076 / 0.083 / 0.074 / 0.070 (40 optimal) |
| bottleneck self-attention | hurts (overfits) |
| seed-ensembling deep K=11 (3 seeds) | 0.0847 raw → 0.086 +3D |
| **final ensemble** (deep K11+K13 + 2D U-Net + dilated CNN + kNN + 3D) | **~0.098 (robust) / 0.103 (greedy, overfit)** |

**Levers that did NOT help:** affine/rotation aug, dedicated presence head (multi-task; set is
irreducibly hard, 96% recall @ 77% precision), presence/mass decision thresholds, morphological
region growing (targets too small, floods), multi-view slice reconciliation (far views dilute),
larger base / bottleneck attention (overfit on 61 volumes).

**Key diagnostic:** oracle label-set gives only +0.026 (0.072→0.098) and is unreachable (predicting
which labels are present on a new subject is genuinely hard). Per-row scores are uniform (~0.09), 
no easy subset to exploit. The fundamental limit is **cross-subject localization with only 61 source
volumes**; the deeper receptive field + wide 3D context + elastic aug + diverse ensembling extract
the most signal I found. Shipped robust ensemble ≈ 0.098 group-CV (2.05x the honest kNN baseline).
