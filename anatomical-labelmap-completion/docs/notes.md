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

## Baselines (CV, 5-fold seed 0/42)
| Method | CV score |
|---|---|
| predict empty | 0.000 |
| positional argmax | ~0.000 |
| 1-NN copy (full slab) | 0.669 |
| 1-NN copy (center-only) | 0.691 |
