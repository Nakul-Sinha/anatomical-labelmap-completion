# Anatomical Labelmap Completion

## The problem

Each row gives me three adjacent 32x32 integer label maps, the previous, center
and next slice of a small anatomical volume, with 17 target structures deleted
from the center slice. I have to put those labels back.

Scoring is the mean of the squared active macro IoU, where the macro IoU averages
per label IoU over every non zero label present in either my prediction or the
truth, and a row that is genuinely empty and that I leave empty scores a full 1.0.

The squaring, combined with the rule that a label counts if it appears in either
side, is what makes this awkward. Inventing a structure that is not there costs
me as much as missing one, so a lot of the work is teaching the model when to
predict nothing at all.

## What I did

I lean on the two neighboring slices, since anatomy is continuous and most of a
missing structure is still visible one slice away. The shipped solution is a fast
core of k-nearest-neighbor matching plus a U-Net plus a 3D context pass. A
dilated context CNN and a per-cell GBDT exist as ensemble members but only add
marginally on grouped CV, so I left them out of the shipped path.

The result I did not expect: geometric augmentation actively hurts here. The
oriented local configuration of anatomy around a hole is real signal, and D4
rotations, flips, translations, 8-pose TTA, and absolute coordinate channels all
destroy it. Removing them lifted both CNNs substantially.

## Layout

Solution code, experiments and notes live under
`anatomical-labelmap-completion/`. `reference.txt` is the working checklist for
how I run these. Datasets are not committed.
