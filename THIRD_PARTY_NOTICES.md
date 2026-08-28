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
  The Director UI, execution node and per-shot media workflow in this repository
  are Myang implementations and are not copied from this source.

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
- Tested releases:
  - v0.33.2: `7cee3ceb1a35503172e0dfb8dbdbdedee2aba8aa`
  - v0.34.0: `12d5279438bfefc058a269eae805ceab6047777f`
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

### Optional speed-cache integrations

- `TE-Speed-MiniMaxH3`: `turbo.py` can discover a separately installed sibling
  package with this exact directory name. No source file or model is bundled;
  if it is absent, Myang warns and continues without that cache.
- [ComfyUI-Spectrum-MiniMax-H3](https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3):
  optional Spectrum cache discovery under the same no-bundling and fallback
  behavior.

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
