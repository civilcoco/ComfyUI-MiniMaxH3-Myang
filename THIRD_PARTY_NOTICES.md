# Third-party notices and provenance

This repository is distributed under GPL-3.0-only. It contains adaptations
from GPL-compatible projects and interoperates with optional projects listed
below. This file records provenance; it does not replace the licenses of model
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

### ComfyUI-H3-AudioRefine

- Source: https://github.com/Adudeguyman/ComfyUI-H3-AudioRefine
- Audited revision: `d0ed019b6f1c4ceb0caf3d69c502a313d1f6da9d`
- Copyright: Copyright (c) 2026 Adudeguyman
- License: MIT
- Use here: `audio_refine.py` adapts the native packed-AV validation,
  per-stream noise-mask construction and exact audio-only sampling path. Myang
  does not bundle the optional frozen-video cache; it adds its own
  duration-preserving long-video audio-boundary smoother and Director wiring.

MIT License

Copyright (c) 2026 Adudeguyman

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

### ComfyUI-MiniMaxH3-Easy

- Source: https://github.com/nkxx188/ComfyUI-MiniMaxH3-Easy
- Copyright: Copyright (c) 2026 nkxx188
- License: MIT
- Use here: portions of media normalization, media-bundle compatibility and
  reference mention handling in `core.py`, `agent_media.py` and `nodes.py`.
  The Myang Media Agent, dialogue audit and LLM orchestration are not attributed
  to this project; the public upstream repository does not contain that node.

MIT License

Copyright (c) 2026 nkxx188

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

### ComfyUI MiniMax H3 Director

- Source: https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director
- Audited revision: `a267324a9f88141ff4e4b0e8c1a6ed90b4e45db7`
- Copyright: AIMixer contributors
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

### See-through (vendored inference code)

- Source: https://github.com/shitagaki-lab/see-through
- Paper: *See-through: Single-image Layer Decomposition for Anime Characters*,
  ACM SIGGRAPH 2026 Conference Papers, doi:10.1145/3799902.3811209,
  arXiv:2602.03749
- Copyright: Copyright the See-through authors (Jian Lin, Chengze Li, Haoyun
  Qin, Kwun Wang Chan, Yanghua Jin, Hanyuan Liu, Stephen Chun Wang Choy,
  Xueting Liu)
- License: Apache License 2.0 (see `LICENSES/Apache-2.0.txt`)
- Use here: `seethrough/` vendors the inference path only —
  `common/modules/layerdiffuse/{layerdiff3d,diffusers_kdiffusion_sdxl,vae,
  transformer3d,utils}.py` (kept as `layerdiff3d.py`, `kdiffusion_sdxl.py`,
  `trans_vae.py`, `transformer3d.py`, `ld_utils.py`),
  `common/modules/marigold/` (as `seethrough/marigold/`, with
  `marigold_depth_pipeline.py` renamed `pipeline.py`), and the reachable subset
  of `common/utils/{cv,torch_utils,torchcv}.py` collected into
  `seethrough/st_utils.py`. Bodies are upstream's; the only edits are relative
  imports, LaMa routed through this pack's own loader, and a PIL image load
  replacing See-through's `io_utils`. Training code, the Qt UI, the taggers and
  the mmcv/mmdet/detectron2 annotators are not included.
  `live2d_seethrough.py` is Myang's own driver: it reimplements
  `inference/scripts/inference_psd.py`'s two-pass orchestration in memory rather
  than through PNGs under `workspace/`, and adds the ComfyUI node, VRAM
  handling, NF4 loading and reporting.
- Weights are not bundled. `tools/fetch_seethrough_weights.py` downloads
  `24yearsold/seethroughv0.0.2_layerdiff3d_nf4` and
  `24yearsold/seethroughv0.0.1_marigold_nf4` (Apache-2.0 model cards) into
  `ComfyUI/models/seethrough/` on request.

### LayerDiffuse and Marigold (upstream of See-through's models)

- LayerDiffuse: https://github.com/lllyasviel/LayerDiffuse_DiffusersCLI —
  transparent layer diffusion, Apache-2.0. See-through's LayerDiff 3D builds on
  it; the vendored `layerdiff3d.py` / `trans_vae.py` derive from that lineage.
- Marigold: https://github.com/prs-eth/Marigold — Copyright 2023-2025 Marigold
  Team, ETH Zürich, Apache-2.0. See-through's depth model is a Marigold
  fine-tune; the vendored pipeline keeps ETH Zürich's file header.

### AutoLive2d (interoperability target)

- Source: https://github.com/Fenglin-Maple/AutoLive2d
- License: Apache License 2.0
- Use here: no code is copied. `live2d_autolive.py` writes AutoLive2d's v1
  `RigProject` / `RigTemplate` / finished-pack format, with the per-part rig
  constants (kinds, `recommendedZ`, parent bones, bones, parameters, physics
  templates, mesh density, per-vertex pseudo-depth, deformer keyframes)
  transcribed from its `src/lib/classify.ts`, `defaults.ts`, `mesh.ts`,
  `template.ts` and `finishedPack.ts` so an exported pack matches what its own
  editor would build. AutoLive2d is a standalone web app, not a ComfyUI
  extension, and is not required to use this pack.

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
