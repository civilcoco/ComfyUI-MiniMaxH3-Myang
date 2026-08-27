# Changelog

All notable changes are documented here.

## 0.1.0 - Unreleased

- Fixed the LLM service settings panel failing to render because its route
  status presentation helper was missing. Added safe status labels for ready,
  cooling, quota-blocked, disabled, and unknown routes.
- Consolidated the active native long-video implementation into `nodes.py`.
  Removed the version-suffixed `nodes_v2.py` module and the unreachable legacy
  Motion Context compatibility path while preserving public node IDs and
  serialized widget positions. Removed the redundant `core_v2.py` re-export,
  renamed the active anchor compatibility module by responsibility, and dropped
  the disabled legacy workflow builder.
- Made synchronized anchor trimming pad a short audio tail with silence as well
  as truncate a long one, preventing per-segment AV duration drift. The complete
  suite now passes against both ComfyUI v0.33.2 and v0.34.0 layouts.
- Prepared the public release boundary: removed unverified third-party Skills,
  generated H3 samples, runtime caches and private-path reports from the
  distributable tree; sanitized example media names and removed watermark-
  removal instructions.
- Hardened Skill routes against path traversal. Local directory import is now
  disabled unless an administrator sets `MINIMAX_H3_SKILLS_IMPORT_DIR`, and
  imported packages must remain inside that root, contain UTF-8 text only,
  avoid symlinks and fit bounded file/package sizes.
- Added GPL/MIT/Apache provenance headers, `NOTICE`, Comfy Registry metadata,
  `.comfyignore`, release hygiene auditing and baseline GitHub Actions checks.
- Fixed collapsed Director setting cards and aligned stale splitter tests with
  the current fixed-count planner and deterministic local fallback contracts.
- Simplified the Director's left-side inputs. Base and Turbo models now share
  one model socket, first-pass steps stay in the Director control, and new
  nodes no longer expose duplicate Agent prompt, legacy detail-settings,
  separate Turbo-model, or Turbo recommended-step sockets. Loaded workflows
  migrate an existing Turbo-model link to the single model socket.
- Fixed the smart split silently producing N copies of the whole script. An
  empty or non-JSON LLM answer used to be padded with `{"prompt": <the entire
  script>}` repeated N times, which looks like a successful run until the video
  plays back as N renders of the same shot. The split now runs the Media
  Agent's fallback ladder (retry with a fresh seed, then a shorter request with
  a lower token ceiling — Skill dropped first, then the material rules) and
  raises an actionable error if every rung comes back empty. Padding is still
  allowed when the model returned *some* segments.
- Explained the segment arithmetic to the model: `5 x 8.00s` of generation is
  36.3s of film because adjacent segments overlap by 0.92s at the seam. Given
  only the two totals, reasoning models spent their whole budget trying to
  reconcile them and answered with an empty string.
- Segment transitions are now chosen per boundary by the LLM (`transition`:
  承接 / 切镜 / 开场) instead of forcing every boundary to be a smooth
  continuation, and the panel badges them. The seam anchor is always physically
  present, so the rules tell the model to cut through shot content and never to
  ask for a fade or flash the pipeline cannot honour.
- `call_vlm` retries under a provider's token cap (glm-4v-flash allows 1..1024
  and answers HTTP 400 rather than clamping) and defaults to 1024; unrelated
  400s still propagate. `call_llm` now logs `finish_reason` and the reasoning
  channel's size when content comes back empty.
- Added a segment-prompt panel to the Director. Agent/long-script prompts are
  authored at runtime by the LLM, so until now the only readout was one
  truncated line of whichever segment was sampling. `H3DirectorPlanValue` — the
  one place holding both the finished plan and the panel id for all three
  source modes — now broadcasts the whole plan, and the panel renders each
  segment with material chips and dialogue highlighting, marking the segment
  currently generating.
- Fixed the Director hard-erroring with "『起始段』只在动作迁移下生效" after
  switching away from action transfer. Both resume controls are only rendered
  inside the transfer panel, so their values in other modes are leftover state
  the user cannot see or reset — they are now ignored with a log line, and the
  frontend resets the start segment when the task mode leaves transfer.
- Gave the Director's Agent/long-script mode the Media Agent's Skill and media
  handling instead of a second, weaker copy: `skill_preset` (with `auto`
  routing through the cached index and the LLM-learned digest), `skill_text`
  for pasted overrides, and `vlm_service` so a VLM describes every shared asset
  before the split. Segment prompts now follow the Skill's output structure and
  tag conventions, while the segment count and per-segment duration stay under
  the Director's control. Skill text and manifest both feed the split cache key.
- `agent_nodes` grew two public seams — `resolve_skill` and `media_whitelist` —
  so the agent and the splitter cannot drift apart. `_media_manifest` now
  numbers within a type by input index, matching `H3Condition`.
- Added a shared-materials bucket to the Director: character art, scenery,
  clips and music can be uploaded on the node itself instead of wiring a
  separate Media Agent. In Agent/long-script mode the manifest goes to the LLM
  with the script, so it picks which segment references which asset; manual
  shot cards reach them through 叠加全局素材. Uploads stack after a connected
  Media Agent, and video uploads are rejected for 动作迁移/视频续写.
- Fixed the material manifest handed to the splitter's LLM: it numbered assets
  by `input_index` across all types, so a bundle of one image plus one clip
  advertised `@视频2` — a tag `H3Condition` rejects as a dangling reference.
  Numbering now comes from `core.media_rows`, the same per-type counter the
  generator uses, and the manifest carries each asset's subject so the model
  can match assets to shots. The split cache key now includes the manifest.
- Added action-transfer resume to the Director: a `起始段` start-segment widget
  plus optional `前段视频` / `前段音频` inputs restart an interrupted run at any
  segment. The reference video keeps its original split windows, the previous
  cut is used only as the seam anchor (never as a ref2va reference), and saved
  segments keep their absolute numbering.
- Initial public-release preparation.
- Native multi-keyframe long-video expansion and synchronized AV trim.
- Seam blending and color-drift correction.
- LightX2V Turbo schedule validation.
- Optional low-sigma refinement and second-pass detail upscale.
- Moved the user-authored MiniMax H3 Media Agent, viewer, dialogue audit and
  frontend into Myang; it no longer imports another custom-node package at runtime.
- Added GPL-3.0 licensing, provenance notices and sanitized example workflow.
- Added the Myang Director control surface with manual variable-duration shot
  cards and Agent/long-script smart splitting, while keeping the standalone
  pipeline nodes available.
- Moved all LLM/VLM execution and configuration into `Myang_node`; service IDs
  remain stable internally while node dropdowns use display names. Fixed true
  deletion, ID-edit duplicates, legacy aliases and API-key exposure.
- Added optional learned 3D H3 latent upscaling, bounded temporal chunks,
  fp16/fp32/bf16 selection, multi-pass refine and post-upscale GPU offload.
- Combined ComfyUI's model-only LoRA loading with the LightX2V joint AV Turbo
  schedule node, added filename-based FL2VA/Ref2VA profile detection, and let
  Director prefer an optional Turbo model while synchronizing its official NFE.
- Fixed Director text fields losing their caret after every keystroke by
  separating hidden timeline persistence from structural DOM rendering.
- Split Director task semantics: action transfer now accepts one video directly
  in its control panel, uses one prompt and automatically segments the source
  with synchronized audio, while
  continuation uses the previous video's tail only as first-segment Motion
  Context and then continues from the preceding generated latent.
- Added an instance-scoped Director progress panel with segment/step state,
  current prompt and decoded frame previews; standalone long-video progress no
  longer receives Director-owned events.
- Replaced Director prompt textareas with stable rich material mentions. Typing
  or inserting `@图片1` / `@视频1` / `@音频1` now resolves to an atomic preview
  chip using the same local/global ordinal contract as H3 conditioning.
- Shortened rich mention chips to `图片1` / `视频1` / `音频1`; filenames remain
  available in menus/tooltips, while an optional bound subject adds the only
  visible suffix (for example `图片1 · 小爱`).
- Kept first-pass steps visible with Turbo connected. Valid official alternatives
  such as 4 steps in the LightX2V v1.0 8/4 profile are preserved, while invalid
  legacy values fall back to that profile's recommendation.
- Added Director reference-video preprocessing for action transfer and Motion
  Context continuation: preserve source resolution, select the same 360P–1080P
  short-edge presets as first pass, or set a custom orientation-aware 1080P cap.
- Added an always-visible actual Shift summary to the joint Turbo loader and an
  explicit advanced override switch. Official profiles remain authoritative by
  default; overrides are stored in model metadata and exposed through outputs.
