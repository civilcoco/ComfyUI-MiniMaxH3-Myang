# MotionContext continuity study

This folder records the reproducible audit and experiments for cross-segment
identity, scene, motion, and seam continuity in the Myang H3 Director pipeline.

## Scope

- Baseline: the existing five-segment Director run at 768×1376, 24 fps.
- Primary issue: small character-face identity drift while motion and scene
  continuity remain acceptable.
- Comparability rule: change one factor at a time and keep source, seed, prompts,
  resolution, sampler, step count, and segment lengths fixed unless explicitly
  named as the experiment variable.
- Measurements are split into immediate seam continuity and within-segment
  identity persistence. A good cut does not by itself prove stable identity.

Generated contact sheets and local metric reports are intentionally excluded
from the public repository because model outputs and source media may carry
separate terms. Reproduce them locally with the documented probe instead.

- `evaluation_contract.md` defines the comparable baseline and ablation order.
- `findings.md` records the source, prompt, visual, and metric evidence.
- `tools/continuity_probe.py` reproduces the local face/CLIP/HOG/HSV report.
