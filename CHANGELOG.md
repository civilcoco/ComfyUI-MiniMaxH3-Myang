# Changelog

All notable changes to this project are documented here.

## 0.2.0 - Release candidate

This release adds the Director rough-cut, reusable templates, asset catalogue,
audio refinement and the native second-pass reference-weight path. It also
scopes temporary memory reservations to the active H3 sampler, preserves
unrelated learned Skill summaries, and validates the complete clean-checkout
test suite before publication.

## Unreleased - MiniMax H3 improvements (research branch)

This branch brings the MiniMax H3 half of the author's in-progress local tree
back onto the public 0.1.0 checkout. The AI image studio, Live2D, psd2live and
See-through modules are deliberately left out; `pyproject.toml` therefore no
longer declares `psd-tools`.

### Second pass (二采)

- Fixed reference weighting never reaching the sampler. `H3Condition` attaches
  `reference_weight` to each official ref block and the attention forwards scale
  the matching Value rows, but nothing recorded which packed rows belonged to
  which reference: the stock `PackedLayout` has no such field, so
  `layout.reference_value_scales` was never set and `transformer_options` never
  carried `minimax_reference_value_scales`. The layout patch in `anchors.py`
  now resolves the token ranges per weighted reference (image rows, a video's
  audio rows, then its visual rows) and a `DIFFUSION_MODEL` wrapper installed by
  the sampler copies them into `transformer_options` on every step. Unit
  weights leave the attention path untouched. `H3Condition` installs the layout
  patch itself when any weight differs from 1.0, so single-segment runs get it
  too; single-segment graphs never reached the long-video anchor installer.
- The dynamic-VRAM pass-2 OOM retry now always restores the AIMDO pressure
  baseline afterwards. It only did so when the setter reported success, so a
  ComfyUI build whose aimdo library lacks `set_simple_vram_headroom` (this
  machine's) skipped the restore path entirely.
- Second-pass conditioning is reused from the first pass by default
  (`reuse_condition`), with an explicit opt-out that rebuilds it at the
  second-pass canvas. Memory profiles (`memory_profile`, custom reserve,
  preview cadence), a `H3RefineMemoryBarrier` before the pass-2 model loads, and
  a `H3SegmentMemoryBarrier` that keeps only the CPU tail of the detail latent
  between segments replace the previous always-resident behaviour.
- `H3LatentOverlapSeed` seeds the previous segment's high-resolution tail into
  the zero-noise region of the next detail latent instead of registering it a
  second time as a condition keyframe, which was inflating packed attention
  tokens from segment 2 onwards.
- The bislerp latent option was removed from the second-pass menu; it ghosts on
  the temporally compressed H3 latent. `H3PixelUpscale` handles `仅放大` with
  pixel/VSR methods without a VAE round trip, and `H3VsrEnhance` runs a 1:1
  RTX VSR pass after a same-resolution second pass.
- Neural 3D temporal chunks are cross-faded with replicated edge context;
  chunk `0` runs the whole clip in one pass and is the new default.
- Experimental continuous-Sigma second pass shares one noise trajectory with the
  first pass and needs no separate `二采模型`.
- Pass-1 checkpoints can be saved and reused so a second pass can be re-run
  without sampling again; a single saved pass-1 video can also enter the second
  pass directly.

### Turbo, audio, Director

- The Turbo loader no longer mounts TE-Speed or Spectrum; the `speed_cache`
  widget stays for positional compatibility and is ignored. Up to three
  ordinary H3 effect LoRAs can be stacked after the official Turbo LoRA.
- Native packed-AV audio refinement (`audio_refine.py`, adapted from
  ComfyUI-H3-AudioRefine, MIT) and duration-preserving audio seams.
- Director: rough-cut timeline, reusable templates, asset catalogue, storyboard
  revision routes, per-segment skill bundles, action-transfer resume, reference
  weight controls on media cards, and first-pass memory profiles.
- Optional MAINodes / H3-FaceRefine enhancements are gated behind Director
  switches; their tests skip when the packs are absent.

### Tests and tooling

- `tools/run_tests.ps1` runs every `tests/test_*.py` and `.mjs` file and treats
  a `SkipTest` as a skip rather than a failure.
- Removed a stray `import core` in the bounded-reference-decode test that only
  resolved when the package's own directory was on `sys.path`.
- The default action-transfer prompt no longer asks the model to remove a
  watermark.

## 0.1.0 - Unreleased

- Initial public release of ComfyUI-MiniMaxH3-Myang.
- Added the Myang Director for manual storyboards, LLM-assisted script splitting,
  text-to-video, video continuation, and motion-transfer jobs.
- Added native MiniMax H3 long-video expansion on the 17-frame temporal grid,
  with arbitrary-position multi-keyframes and segment-to-segment context.
- Added synchronized 24 fps video and 40 Hz audio trimming, seam blending, and
  optional color-drift correction.
- Added shared and per-shot media management, reference-tag validation, and an
  optional Media Agent with LLM/VLM assistance.
- Added action-transfer resume support and per-segment output saving.
- Added implemented LightX2V Turbo profiles with task, step, scheduler, and AV
  shift validation.
- Added optional pixel, latent, neural 3D, and NVIDIA RTX VSR second-pass paths.
- Added the local LLM/VLM service-management panel with protected API-key
  handling and provider routing.
- Added regression coverage for ComfyUI v0.33.2 and v0.34.0 layouts, frontend
  panels, release metadata, and package hygiene.
