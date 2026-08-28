<div align="center">

# ComfyUI-MiniMaxH3-Myang

Long-video direction, segmented generation, multi-keyframe continuity, and AV stitching for MiniMax H3.

<a href="./README.md"><img src="https://img.shields.io/badge/🇬🇧_English-0b8cf5" alt="English"></a>
<a href="./README.ZH_CN.md"><img src="https://img.shields.io/badge/🇨🇳_中文简体-e9e9e9" alt="中文简体"></a>

Author and maintainer: **沐阳Myang**<br>
Bilibili: [**沐阳Myang**](https://space.bilibili.com/506587111) · GitHub: [@civilcoco](https://github.com/civilcoco)

</div>

This node pack turns MiniMax H3 clip generation into a long-video workflow. It plans shots on H3's
time grid, generates picture and sound segment by segment, carries the tail of one segment into the
next as temporal context, trims the overlap, and joins the delivered clips.

The core nodes call ComfyUI's official MiniMax H3 interfaces directly and do not require another
third-party custom-node pack. Model weights, LoRAs, and demo media are not distributed here.

> [!IMPORTANT]
> The node code is licensed under **GPL-3.0-only**. The continuity implementation includes GPL-3.0
> code adapted and further developed from
> [ComfyUI-H3-Motion-Context](https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context).
> See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for audited revisions, modification boundaries,
> and the other third-party notices.

> [!CAUTION]
> MiniMax H3 models and their outputs are governed by a separate community license that may include
> regional restrictions. Before downloading a model or publishing generated material, read
> [LEGAL.md](LEGAL.md) and the model publisher's current terms.

## Features

- **Director** — organize text-to-video, video continuation, and motion-transfer jobs in one node.
- **Storyboards and long video** — arrange shots manually or ask an LLM to split a script by duration.
- **AV continuity** — extract temporal context from the previous latent and pin it at the head of the next segment.
- **Media management** — upload, preview, number, and validate shared or shot-specific media.
- **Media Agent** — optionally use an LLM/VLM to understand media and write valid H3 reference tags.
- **Turbo scheduling** — load a LightX2V Turbo LoRA and validate task family, steps, scheduler, and AV shift.
- **Second-pass upscale** — choose pixel, latent, neural 3D, or NVIDIA RTX VSR paths.
- **Resume support** — continue a motion-transfer job from a selected segment with the preceding AV seam context.

## Installation

Use a recent ComfyUI build that includes the official MiniMax H3 nodes. Clone this repository into
`ComfyUI/custom_nodes`:

```powershell
cd ComfyUI\custom_nodes
git clone https://github.com/civilcoco/ComfyUI-MiniMaxH3-Myang.git
```

Restart ComfyUI. The nodes appear under the `沐阳 H3` category. If the browser still shows the
pre-install UI, perform a hard refresh.

Provide your own copies of:

- a MiniMax H3 diffusion model;
- a Qwen text encoder;
- a video VAE;
- an audio VAE;
- any optional LoRA or upscaler required by your chosen path.

The core package declares no additional Python packages beyond the standard ComfyUI environment.
Install `openai-whisper` only when you select local Whisper transcription in the Media Agent. The
NVIDIA RTX VSR path requires a separately installed NVIDIA VFX runtime.

## Quick start

Start with:

[`example_workflows/Minimax_H3_Myang_Director_CN.json`](example_workflows/Minimax_H3_Myang_Director_CN.json)

1. Select the diffusion model, CLIP, video VAE, and audio VAE in `沐阳 H3 加载器`.
2. Choose text-to-video, video continuation, or motion transfer in the Director.
3. Enter a script, or fill in the title, prompt, and duration on each manual shot card.
4. Upload shared or shot-specific media. Refer to a specific item with
   `@图片N`, `@视频N`, or `@音频N`.
5. Render only two segments first. Use 22 context frames and inspect the picture, motion, lip sync,
   and audio at the join.
6. Once the seam is sound, increase the segment count or enable Turbo, low-sigma refinement, or a second pass.

For a graph with individually wired stages, open:

[`example_workflows/Minimax_H3_Myang_LongVideo_CN.json`](example_workflows/Minimax_H3_Myang_LongVideo_CN.json)

That example also demonstrates optional integrations from VideoHelperSuite, KJNodes, SolAttn,
Spectrum, ReservedVRAM, and Easy-Use. They are not Python import dependencies of Myang. Install the
packages needed by the branches you use, or bypass/remove those nodes.

Both examples have been sanitized. Select your own model names, media, prompts, output name, and seed after loading.

## Director

`沐阳 H3 · 导演台（全功能）` offers two ways to build a timeline:

- **Manual storyboard** — set a title, prompt, duration, and media on each shot card. Durations snap
  to H3's supported `17k+5` frame grid. Each shot accepts up to 9 images, 3 videos, and 3 audio files.
- **Agent / long-script split** — let an LLM build the timeline from the total duration, per-segment
  duration, media inventory, and writing rules. With the LLM disabled, the input prompt follows the
  local splitting path and consumes no tokens.

Use the Director's shared-media area for character images, locations, reference video, or music used
throughout the job. Media on a shot card belongs to that shot. Each card can use only its own media or
append the shared inventory. Workflows store portable `ComfyUI/input` references; they do not embed
the media payload in the JSON.

A motion-transfer shot can select its own action source. When it does not, the Director uses the global
`ref_video` input. Motion transfer and video continuation accept one direct reference video; image and
audio references remain available as normal.

### Media references

Each media type has an independent index:

```text
@图片1  @图片2
@视频1  @视频2
@音频1  @音频2
```

The Media Agent converts these editor-facing labels into the official H3 `<Picture N>`, `<Video N>`,
and `<Audio N>` forms, and checks that every label maps to connected media. A VLM can describe image
and video contents; Whisper can add an audio transcript.

### LLM services

Open ComfyUI Settings, select `Myang_node`, and open **LLM Service Settings**. The panel supports:

- OpenAI-compatible APIs;
- Ollama;
- multiple URL/API-key routes in one service;
- round-robin or primary-route-first selection;
- route cooldown and failover after rate limits, timeouts, or service errors.

The configuration read endpoint never returns API keys to the browser. Leave the key field empty when
editing a service to keep its stored value. Configuration is saved under
`user/default/Myang_node/config/llm_services.json` in the ComfyUI user directory. LLM/VLM calls are
implemented by this package; Prompt Assistant is not required.

## Continuity and seam handling

```text
segment N temporal latent
        ↓
extract tail video and audio context
        ↓
pin multi-keyframe context at the head of segment N+1
        ↓
sample → trim the overlap in sync → blend the seam → concatenate
```

Video follows a 24fps timeline. Audio placement follows H3's 40Hz time grid. The seam node applies a
short waveform blend at the real cut and trims each audio segment to the delivered picture duration.

In a manually wired graph, `H3ScriptSplitter.overlap_frames` must match
`H3LongVideo.context_length`.

| Context frames | Temporal blocks | Suggested use |
|---:|---:|---|
| 5 | 2 | Experimental short context; faster with weaker constraint |
| 22 | 7 | Recommended starting point; about 0.92 seconds |
| 39 | 12 | Stronger motion and composition continuity |
| 56 | 17 | Longest context; highest token and trimming cost |

Context consumes conditioning tokens and is trimmed from the delivered portion of every later segment.
Establish a two-segment baseline with 22 frames before comparing another length on the same inputs.

The smart splitter labels each boundary as a continuation or a cut. Both use temporal anchors. A cut is
described by the next shot's prompt; it is not a black frame, flash, or post-production hard cut.

## Turbo and low-sigma refinement

`沐阳 H3 · Turbo LoRA 联合音画加载调度` calls ComfyUI's LoRA loader and applies the paired H3
video/audio shifts. The following table lists the presets implemented by this package:

| LightX2V preset | Video/audio shift | Inference NFE | Training resolution |
|---|---:|---:|---|
| v1.0 8-step | 12 / 3 | 8 or 4; start with 8 | Mixed 544p ratios |
| v1.0 4-step 768P | 6 / 3 | 4 | 1344×768 |
| v0.1 4-step | 12 / 3 | 4 | Mixed 544p ratios |
| Ref2VA v0.1 4-step | 12 / 3 | 4 | Mixed 544p ratios |

Turbo follows a fixed NFE trajectory and requires the `simple` scheduler, `denoise=1.0`, and an Euler
sampler. Low-sigma insertion changes that trajectory, so the two modes cannot be enabled together.
Prefer an FL2VA/T2VA preset for generation and a Ref2VA preset for motion transfer.

The upstream project may publish additional checkpoints. In v0.1.0, the upstream **8-step v1.0
768P** checkpoint has no dedicated Myang profile; do not use filename-based Auto detection for it.
Use only a preset listed above unless you have independently validated a manual shift, or wait for a
package update that adds the checkpoint explicitly.

The Turbo node also exposes optional TE-Speed and Spectrum cache modes. They require separately
installed sibling custom-node packages; when one is unavailable, Myang warns and continues with that
cache disabled.

Preset sources:
[LightX2V model page](https://huggingface.co/lightx2v/Minimax-h3-Turbo) and
[publisher ComfyUI workflows](https://github.com/ModelTC/Minimax-H3-Turbo/tree/main/example_workflows).

Without Turbo, **Low Sigma Refinement** can add integration points to the low-noise end of the sampling
trajectory. Compare Off and Balanced with identical media, seed, and settings before using it for a long job.

## Second-pass upscale

The Director and `沐阳 H3 · 二采放大设置` provide three modes:

- upscale, then run a second pass;
- second pass at the same resolution;
- upscale only, with no second pass.

Available paths include pixel/VAE, bislerp latent, neural 3D latent, and NVIDIA RTX VSR. Long videos are
processed one segment at a time. Final audio comes directly from the first pass and receives only seam
handling and duration trimming. Use a Ref2VA base model without a Turbo LoRA for the second pass.

The neural 3D path accepts LBH-123-AI's 24-channel H3 Latent Upscaler weights. Download the checkpoint
separately and place it in:

```text
ComfyUI/models/latent_upscale_models/
```

Weights and documentation:
[LBH-123-AI/Minimax_h3_latent_Upscaler](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler)

Start with `fp16` and a temporal chunk of `16`. Try `8` when memory is tight, or `fp32` if the result
shows precision artifacts or color blocks.

## Main nodes

| Node | Purpose |
|---|---|
| 沐阳 H3 · 导演台（全功能） | Organize storyboard, media, generation, seams, and optional second pass |
| 沐阳 H3 加载器 | Configure Ref2VA/FL2VA models, CLIP, video VAE, and audio VAE |
| 沐阳 H3 条件（提示词 + 素材） | Build official H3 conditioning and reference media |
| 沐阳 H3 · Media Agent | Preview, number, describe, and validate media references |
| 沐阳 H3 · 分段计划 | Calculate segment count, frame lengths, and overlaps |
| 沐阳 H3 · 长视频（原生多关键帧） | Expand and execute a multi-segment sampling chain |
| 沐阳 H3 · 任意位置关键帧 | Pin a keyframe at an arbitrary supported position; chainable |
| 沐阳 H3 · 段间多关键帧 | Carry temporal latent context between adjacent segments |
| 沐阳 H3 · 锚点同步裁剪 | Trim anchored picture and sound in sync |
| H3 接缝淡化 | Blend picture/waveform cuts and enforce the delivery duration |
| 沐阳 H3 · Turbo LoRA 联合音画加载调度 | Load a LoRA and validate the Turbo parameter contract |
| 沐阳 H3 · 二采放大设置 | Configure second-pass modes and parameters for long video |
| 沐阳 H3 · 二采放大精修（像素路径） | Run pixel upscale and low-denoise redraw |
| 沐阳 H3 · Latent 直接放大（极速双采） | Run a latent-space upscale path |
| 沐阳 H3 · 段间漂移校正 | Optionally correct accumulated brightness and color drift |

Nodes marked `内部` are managed by the Director or long-video expansion and normally do not need manual wiring.

## Compatibility and limitations

- CPU structural regression tests cover the H3 layouts in ComfyUI `v0.33.2` and `v0.34.0`. After a
  ComfyUI update, validate a two-segment render before starting a long job.
- Resolution must remain constant between segments when continuity uses temporal latents.
- Long chains can accumulate losses in picture detail, timbre, brightness, and saturation. A clean seam
  does not guarantee that content quality will remain constant down the chain.
- Turbo, cache nodes, attention patches, and second-pass processing all change speed, memory use, or
  quality. Troubleshoot against a plain two-segment baseline first.
- ComfyUI-H3-Motion-Context is not an installation dependency. If it is also present in `custom_nodes`,
  use only one H3 continuity implementation in a job and restart ComfyUI before switching graphs.
- CPU tests verify graph structure, time coordinates, reference order, and trim lengths. They cannot
  replace a real-model visual and audio review.

## Testing

Run from PowerShell:

```powershell
pwsh tools\run_tests.ps1 -ComfyRoot D:\path\to\ComfyUI
```

The suite covers stock head/tail equivalence, multi-keyframes, image/video/audio reference ordering,
5/22/39/56-frame temporal blocks, the 40Hz audio grid, seam trimming, the Media Agent, LLM configuration,
Turbo contracts, the Director, and second-pass expansion. When Node.js is available, it also runs the
frontend structure and LLM service-panel tests.

## Provenance and license

- Repository license: [GPL-3.0-only](LICENSE).
- Primary adapted source:
  [ComfyUI-H3-Motion-Context](https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context), GPL-3.0.
- Neural latent-upscale runtime reference:
  [ComfyUI MiniMax H3 Director](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director), Apache-2.0.
- Full copyright, audited revisions, modification boundaries, and optional integration notes:
  [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
- Version history: [CHANGELOG.md](CHANGELOG.md).

Please report issues with reproducible examples through
[GitHub Issues](https://github.com/civilcoco/ComfyUI-MiniMaxH3-Myang/issues).
