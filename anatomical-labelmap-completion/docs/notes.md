# Small Anatomical Labelmap Completion — findings & strategy

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
   also zero) — background is background across all three slices.
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
| 1-NN copy (center-only) | 0.691 | — |
| k-NN soft vote (k=3) | 0.7056 | 0.0483 |
| local ray-cast signature | — | 0.020 |

## Learned-model honest results (volume-grouped CV)
| Method | group-CV |
|---|---|
| k-NN soft vote (k=3) | 0.0483 |
| augmented U-Net (embeddings, masked CE+Dice, D4 aug + 8-pose TTA) | 0.0348 |
| **blend k-NN + U-Net (0.5/0.5)** | 0.0540 |
| **+ 3D volume-consistency smoothing (a=0.7, w=2)** | **0.0567** |

The learned CNN *under*performs retrieval on unseen volumes — i.e. there is little volume-invariant
rule to learn; most of the signal is instance similarity that does not transfer. Blending + 3D
continuity extract a bit more. Extrapolating k-NN's per-row score onto the test neighbour-similarity
distribution gives an expected **test score ~0.03–0.05** (test is slightly harder than group-holdout).

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
Key correction: geometric **augmentation HURTS** on this atlas data — the oriented local
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
