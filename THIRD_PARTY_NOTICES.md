# Third-party notices and provenance

Copyright (C) 2026 沐阳Myang

This repository is distributed under GPL-3.0-only. It contains adaptations
from GPL-compatible projects and uses the runtime platform and documented
integrations listed below. This file records provenance; it does not replace the licenses of model
weights, LoRAs, SDKs or other packages that users download separately.

## Adapted code

### ComfyUI-H3-Motion-Context

- Source: https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context
- Audited revision: `c140ae99b8c38f782ebd8564c267b42aacade6a4`
- Copyright: Copyright (C) 2026 NikoDemon80
- License: GNU General Public License version 3
- Use here: portions of `anchors.py`, including reference-segment mapping,
  temporal latent slicing, marked H3 layout/payload handling and synchronized
  audio/video trim logic. Myang substantially modified and extended this work
  for arbitrary-position multi-keyframes, marker-gated composition, long-video
  graph expansion and seam scheduling.

The full GPL-3.0 license is provided in `LICENSE`.

### ComfyUI MiniMax H3 Director

- Source: https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director
- Audited revision: `a267324a9f88141ff4e4b0e8c1a6ed90b4e45db7`
- Copyright: Copyright 2026 ComfyUI-Bernini Contributors
- License: Apache License 2.0
- Use here: `latent_upscale_3d.py` adapts the repository's in-package 3D H3
  latent-upscaler runtime and checkpoint contract. Myang adds its own temporal
  memory bound, precision controls, canvas validation and GPU-offload policy.
  The Director UI and execution node in this repository were written for the
  existing Myang pipeline; the upstream repository was used as product and
  architecture reference, including the per-shot material workflow, rather
  than copied wholesale. Myang's implementation stores portable ComfyUI input
  references and resolves them through its own Media Agent-compatible bundle.

The Apache-2.0 license copy is provided in `LICENSES/Apache-2.0.txt`.

### LBH-123-AI MiniMax H3 Latent Upscaler weights

- Model page: https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler
- Companion code: https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler
- Model-card license: Apache-2.0
- Use here: compatible optional user-downloaded 3D weights and the published
  24-channel normalization statistics. No checkpoint is bundled. The GitHub
  code checkout audited on 2026-08-21 did not contain a root LICENSE file, so
  this repository does not copy that checkout's source files.

## Content intentionally not redistributed

This repository does not bundle MiniMax Hub Skills, the MiniMax H3 prompt
manuals, or generated H3 samples. Users may install their own Skills after
checking the exact source terms. Removing those materials from the release is
not a statement that they can never be redistributed; it means this project
does not currently have sufficient provenance and permission to do so.

## Runtime platform and documented integrations

### ComfyUI

- Source: https://github.com/Comfy-Org/ComfyUI
- Tested revision: `43cb4fffc89bba20ab7bd61467a36d0339338dab`
- License: GPL-3.0
- Use here: runtime APIs and the official MiniMax H3 nodes/model interfaces.

### LightX2V MiniMax H3 Turbo

- Source: https://github.com/ModelTC/Minimax-H3-Turbo
- Model page: https://huggingface.co/lightx2v/Minimax-h3-Turbo
- Repository/model-card license marker: Apache-2.0
- Use here: documented shift/NFE/task-family presets and checkpoint-name
  compatibility validation in `turbo.py`. LoRA application delegates to
  ComfyUI's loader; no LightX2V source file or model weight is bundled.

The Turbo LoRA is derived from MiniMax H3. Users must also comply with the
MiniMax H3 Community License; an Apache-2.0 model-card marker does not cancel
the base model's separate restrictions.

### SolAttn

- Source: https://github.com/kijai/ComfyUI-SolAttn_triton
- Use here: optional runtime compatibility detection. No SolAttn source is
  copied or bundled.

### ComfyUI-Prompt-Assistant

- Source: https://github.com/yawiii/ComfyUI-Prompt-Assistant
- License marker: GPL-3.0
- Use here: optional one-time read-only migration source for existing LLM/VLM
  service configuration. Myang's runtime calls providers directly and does not
  import Prompt-Assistant code. No Prompt-Assistant source file is bundled.

### NVIDIA RTX Video Super Resolution

`detail.py` can optionally call a separately installed NVIDIA VFX Python
runtime. This repository does not distribute NVIDIA SDK files, binaries or
models. Users are responsible for installing them and accepting NVIDIA's
applicable terms.

## Workflow-only inspiration

- The user-supplied “NanFeng H3 V4 Public Package” workflow was used to identify
  low-sigma schedule densification as an experiment. No file or code from that
  package is redistributed.
- The user-supplied “二采重绘放大版” workflow was used to study the general
  low-resolution first pass → pixel upscale → VAE encode → low-denoise second
  pass pipeline. Its JSON is not redistributed.
- `wjluoxiao/ComfyUI-JZL-MiniMax-H3` (MIT, audited revision
  `7719a53ca79ae47325cd483a0b62d1974483ad20`) was reviewed for its reference
  area multiplier and V3 Autogrow interface. No JZL source file is copied; the
  current JZL repository does not implement the second-pass sampler used here.

Their licenses were not available in the publication workspace. Do not add
their original JSON, screenshots, documentation or assets to this repository
without first obtaining a compatible license or explicit permission.
