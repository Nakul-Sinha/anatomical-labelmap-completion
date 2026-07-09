# Project Eris — competition workspace

Git-backed workspace for Project Eris ML challenges. One directory per challenge at the repo root
(see `reference.txt` for the operating rules).

## Challenges

| Directory | Challenge | Status |
|---|---|---|
| [`anatomical-labelmap-completion/`](anatomical-labelmap-completion/) | Small Anatomical Labelmap Completion | in progress |

## Workflow

- Every change lands via a feature branch → GitHub PR → merge (no direct commits to `main` after setup).
- Each challenge is self-contained: reads `./dataset/public/`, writes `./working/submission.csv`,
  and the official `solution.py` regenerates the submission end-to-end from a declared modeling pipeline.
- Only libraries available in the standard Kaggle/Eris runtime are used (numpy, pandas, scipy,
  scikit-learn, pytorch). No `pip install`, no internet, no external data.
