# Evaluation contract

## Baseline

- Output: `ComfyUI/output/Video/H3_导演台_成片_00002_.mp4`
- 872 frames at 24 fps, 768×1376.
- Segment cuts: frames 192, 362, 532, and 702.
- One-pass: 540P, 8-step Turbo, context 22 frames.
- Detail pass: 768P, 4 steps, denoise 0.2, neural 3D latent upscale.
- Seed: 280745050315927.
- Prompt cache: `ComfyUI/user/h3_longscript_cache/fdd235944d359ce02a2ca91ffebc7c2a.json`.

## Metrics

- Immediate cut continuity and within-segment identity are reported separately.
- Face crops use the installed local FACE detector. Low-confidence, very small,
  or profile faces remain a measurement limitation and must not be treated as
  an absolute identity score.
- CLIP cosine measures perceptual similarity; HOG cosine measures normalized
  facial texture/shape; HSV cosine measures color distribution.
- A result is accepted only if motion/scene continuity does not regress and
  recurring identity accessories, facial geometry, eye design, and headwear are
  at least as stable as the baseline.

## Ablation order

1. Existing baseline.
2. Prompt identity lock with detail pass disabled.
3. Prompt identity lock with detail pass enabled and high-resolution detail
   latent chaining enabled.
4. If needed, reduce detail denoise from 0.20 to 0.10 while holding everything
   else fixed.
5. Test context 39 only after the identity changes above; context length mainly
   affects the transition window and costs delivered duration.

Each run keeps the source image, screenplay, cached five-shot plan, seed,
samplers, steps, resolution, and segment durations fixed unless that row names
the changed variable.
