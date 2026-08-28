# Changelog

All notable changes to this project are documented here.

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
- Added optional low-sigma refinement and pixel, latent, neural 3D, and NVIDIA
  RTX VSR second-pass paths.
- Added the local LLM/VLM service-management panel with protected API-key
  handling and provider routing.
- Added regression coverage for ComfyUI v0.33.2 and v0.34.0 layouts, frontend
  panels, release metadata, and package hygiene.
