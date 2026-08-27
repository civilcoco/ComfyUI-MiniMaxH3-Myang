"""沐阳 H3 · 自己的采样核心。

SPDX-License-Identifier: GPL-3.0-only

Copyright (C) 2026 Myang

Built straight on ComfyUI's own MiniMax H3 support (`comfy_extras.nodes_minimax_h3`
and `comfy/ldm/minimax`), which is the official implementation. Nothing here
needs a third-party pack.

The core node does the real work -- reference encoding, the Qwen presentation,
the DiT payload. What this module adds is the part the core deliberately leaves
open: a single loader for the four models H3 needs, resolution presets instead
of raw pixel counts, seconds instead of frame counts, and the editor's
``@图片1`` syntax translated into the ``<Picture 1>`` tags the tokenizer expects.

Deliberately a thin layer over the core call rather than a reimplementation of
it: when ComfyUI updates its H3 support, this follows along.

"""

import logging
import math
import re

import torch

from .media_catalog import (
    audio_track,
    catalog_rows,
    image_batch,
    iter_catalog,
    video_stream,
)

logger = logging.getLogger(__name__)

# Presets name the short edge; the canvas is derived from it and the aspect,
# each axis rounded to 32. No area cap here: the core's 768*1344 cap belongs to
# *reference* canvases (adapt_canvas), not to what is being generated, whose
# width/height the core accepts up to MAX_RESOLUTION. Applying it to the output
# is what silently shrank the 832P-and-up presets.
RESOLUTION_PRESETS = ["360P", "416P", "480P", "540P", "640P", "720P",
                      "768P", "832P", "928P", "1024P", "1080P", "自定义"]
ASPECT_RATIOS = ["1:1", "2:3", "3:2", "3:4", "4:3", "9:16", "16:9", "21:9"]

# Area budgets, not short edges. At the same short edge a 21:9 reference costs
# more than twice a 1:1 one, and reference tokens ride through every sampling
# step, so budgeting by area is what actually bounds the cost.
REF_AREA_MATCH = "匹配生成分辨率"
REF_AREA_ORIGINAL = "匹配素材（原尺寸）"
REF_IMAGE_SIZES = {
    REF_AREA_MATCH: None,            # computed from the generation canvas
    "最大1K面积": 1024 * 1024,
    "最大1.5K面积": 1536 * 1536,
    "最大2K面积": 2048 * 2048,
    REF_AREA_ORIGINAL: 0,            # no resample
}
REF_SIZE_SEARCH_RADIUS = 4

MENTION_INDEX = "按编号（@图片1）"
MENTION_FILENAME = "按文件名（@角色.png）"
MENTION_MODES = [MENTION_INDEX, MENTION_FILENAME]

FPS = 24
CANVAS_MULTIPLE = 32
AIMDO_HOSTBUF_ALIGNMENT_SLACK = 64 * 1024 ** 2
AIMDO_HOSTBUF_PATCH = "_myang_h3_hostbuf_model_reserve_v2"

# The editor writes @图片1 / @视频1 / @音频1; the tokenizer wants <Picture 1> /
# <Video 1> / <Audio 1>. Same ordinals, so this is a pure relabel.
MENTION_RE = re.compile(r"@(图片|视频|音频)[ \t_]*(\d+)")
MENTION_TAG = {"图片": "Picture", "视频": "Video", "音频": "Audio"}
# Filename mode: @ up to the next space or sentence punctuation, so
# "@角色.png，然后" stops before the comma.
MENTION_FILE_RE = re.compile(r"@([^\s，。、；：！？,;:!?]+)")


def _core():
    import comfy_extras.nodes_minimax_h3 as core
    return core


def canvas_for(resolution: str, aspect: str, width: int, height: int) -> tuple[int, int]:
    """Preset + aspect -> a canvas H3 will accept: short edge, aspect, round 32."""
    if str(resolution) == "自定义":
        return (max(CANVAS_MULTIPLE, int(width) // CANVAS_MULTIPLE * CANVAS_MULTIPLE),
                max(CANVAS_MULTIPLE, int(height) // CANVAS_MULTIPLE * CANVAS_MULTIPLE))
    short = int(str(resolution).rstrip("P"))
    try:
        a, b = (float(x) for x in str(aspect).split(":"))
    except Exception:
        a, b = 16.0, 9.0
    ratio = a / b
    nom_w, nom_h = (short * ratio, float(short)) if ratio >= 1.0 else (float(short), short / ratio)
    return (max(CANVAS_MULTIPLE, round(nom_w / CANVAS_MULTIPLE) * CANVAS_MULTIPLE),
            max(CANVAS_MULTIPLE, round(nom_h / CANVAS_MULTIPLE) * CANVAS_MULTIPLE))


def aligned_size(image_w: int, image_h: int, target_area: int) -> tuple[int, int]:
    """Largest 32-aligned size within `target_area` that keeps the aspect.

    Rounding each axis independently distorts the aspect, and on a face that
    reads as a different person. Searching a few steps around the nominal size
    and scoring aspect error well above area error keeps identity intact.
    """
    if target_area <= 0 or image_w <= 0 or image_h <= 0:
        return int(image_w), int(image_h)
    area = image_w * image_h
    if area <= target_area:
        # Already within budget. Round *down* to the 32 grid: rounding to the
        # nearest would enlarge a 1080-tall reference to 1088, and upscaling a
        # reference costs tokens on every sampling step while inventing detail
        # the source never had.
        return (max(CANVAS_MULTIPLE, int(image_w) // CANVAS_MULTIPLE * CANVAS_MULTIPLE),
                max(CANVAS_MULTIPLE, int(image_h) // CANVAS_MULTIPLE * CANVAS_MULTIPLE))
    scale = math.sqrt(target_area / area)
    ratio = image_w / image_h
    base_w = max(1, round(image_w * scale / CANVAS_MULTIPLE))
    base_h = max(1, round(image_h * scale / CANVAS_MULTIPLE))
    best = (base_w * CANVAS_MULTIPLE, base_h * CANVAS_MULTIPLE)
    best_score = None
    for dw in range(-REF_SIZE_SEARCH_RADIUS, REF_SIZE_SEARCH_RADIUS + 1):
        for dh in range(-REF_SIZE_SEARCH_RADIUS, REF_SIZE_SEARCH_RADIUS + 1):
            w = (base_w + dw) * CANVAS_MULTIPLE
            h = (base_h + dh) * CANVAS_MULTIPLE
            if w < CANVAS_MULTIPLE or h < CANVAS_MULTIPLE or w * h > target_area:
                continue
            if w > image_w or h > image_h:
                continue
            score = abs((w / h) - ratio) / ratio * 20.0 + abs(w * h - target_area) / target_area
            if best_score is None or score < best_score:
                best, best_score = (w, h), score
    return best


def length_for(seconds: float, fps: float = FPS) -> int:
    """Seconds -> a frame count on H3's 17k+5 grid."""
    return _core().align_frame_count(max(5, round(float(seconds) * float(fps))))


def resolve_mentions(prompt: str, counts: dict[str, int], by_filename=None) -> str:
    """@图片1 -> <Picture 1>, and in filename mode @角色.png -> <Picture 1> too.

    An out-of-range or unknown mention is left as written rather than silently
    pointed at someone else's media: it then shows up as obviously wrong text
    in the prompt instead of as a wrong face in the video.
    """
    text = str(prompt or "")
    if by_filename:
        def by_name(match):
            hit = by_filename.get(match.group(1))
            return f"<{hit[0]} {hit[1]}>" if hit else match.group(0)
        text = MENTION_FILE_RE.sub(by_name, text)

    def by_index(match):
        tag = MENTION_TAG.get(match.group(1))
        ordinal = int(match.group(2))
        if not tag or ordinal < 1 or ordinal > int(counts.get(tag, 0)):
            return match.group(0)
        return f"<{tag} {ordinal}>"
    return MENTION_RE.sub(by_index, text)


def _to_24fps(frames, source_fps):
    source_fps = float(source_fps or 24.0)
    if abs(source_fps - 24.0) < 0.01:
        return frames
    count = max(1, round(int(frames.shape[0]) * 24.0 / source_fps))
    indices = torch.linspace(
        0, int(frames.shape[0]) - 1, count,
        device=frames.device).round().long()
    return frames[indices]


class H3Bundle:
    """The models H3 needs, carried as one link.

    H3 ships two transformers: ref2va drives reference-to-video, fl2va drives
    text/first-last-frame. Only one is ever resident -- they are ~32GB each --
    so the bundle names both and loads on demand, swapping when the task
    changes. Holding both at once is what an eager loader would cost.
    """

    def __init__(self, clip, video_vae, audio_vae, names, weight_dtype="default"):
        self.clip = clip
        self.video_vae = video_vae
        self.audio_vae = audio_vae
        self.names = names
        self.weight_dtype = weight_dtype
        self._cached_kind = None
        self._cached_model = None

    def model_for(self, kind: str):
        kind = "fl2va" if str(kind).lower().startswith("fl") else "ref2va"
        if self._cached_kind == kind and self._cached_model is not None:
            return self._cached_model
        name = self.names.get(kind)
        if not name:
            raise ValueError(f"加载器没有配置 {kind} 模型")
        _ensure_aimdo_hostbuf_headroom()
        import nodes
        model, = nodes.NODE_CLASS_MAPPINGS["UNETLoader"]().load_unet(name, self.weight_dtype)
        self._cached_kind, self._cached_model = kind, model
        logger.info("H3-Myang: 已加载 %s 模型 %s", kind, name)
        return model

    def __repr__(self):
        return f"<H3Bundle ref2va={self.names.get('ref2va', '?')}>"


def _ensure_aimdo_hostbuf_headroom():
    """Reserve the model's address range without raising the pinned-RAM budget.

    Windows caps registered pinned memory at 40% of RAM, while the HostBuffer
    virtual reservation defaults to twice that cap. An unpruned H3 checkpoint
    can be larger than this heuristic, so Aimdo eventually tries to append a
    weight past the reserved address range and logs hostbuf_grow errors before
    falling back to pin stealing. Reserve at least the aligned model size; this
    changes only the virtual address ceiling. ``MAX_PINNED_MEMORY`` and
    ``ensure_pin_budget`` still own physical pinned-RAM pressure.
    """
    try:
        import comfy.model_management as mm
    except Exception:
        return False
    current = mm.pinned_hostbuf_size
    if getattr(current, AIMDO_HOSTBUF_PATCH, False):
        return True
    original = current

    def with_headroom(size):
        reserved = int(original(size))
        cap = int(getattr(mm, "MAX_PINNED_MEMORY", -1))
        high_ram = bool(getattr(getattr(mm, "args", None), "high_ram", False))
        if (bool(getattr(mm, "WINDOWS", False)) and not high_ram
                and cap > 0 and int(size) > cap and reserved > 0):
            return max(reserved, int(size) + AIMDO_HOSTBUF_ALIGNMENT_SLACK)
        return reserved

    setattr(with_headroom, AIMDO_HOSTBUF_PATCH, True)
    setattr(with_headroom, "_myang_original", original)
    mm.pinned_hostbuf_size = with_headroom
    logger.info(
        "H3-Myang: aimdo 大模型 host buffer 按实际模型大小预留；"
        "40%% pinned RAM 上限不变")
    return True


class H3Loader:
    CATEGORY = "沐阳 H3"
    FUNCTION = "load"
    RETURN_TYPES = ("MYANG_H3",)
    RETURN_NAMES = ("h3",)
    DESCRIPTION = ("一次配好 H3 需要的模型：ref2va（参考生视频）、fl2va（文/首尾帧生视频）、"
                   "文本编码器、画面 VAE、声音 VAE。两个扩散模型按用到哪个才加载哪个，"
                   "不会同时占显存。模型用「沐阳 H3 取模型」引出来挂补丁链。")

    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        unet = folder_paths.get_filename_list("diffusion_models")
        return {
            "required": {
                "ref2va_model": (unet, {"tooltip": "参考生视频用（动作迁移 / 续写）"}),
                "fl2va_model": (unet, {"tooltip": "文生视频、首尾帧用"}),
                "text_encoder": (folder_paths.get_filename_list("text_encoders"),),
                "video_vae": (folder_paths.get_filename_list("vae"),),
                "audio_vae": (folder_paths.get_filename_list("vae"),),
                "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"],
                                 {"default": "default"}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return "|".join(str(kwargs.get(k, "")) for k in
                        ("ref2va_model", "fl2va_model", "text_encoder",
                         "video_vae", "audio_vae", "weight_dtype"))

    def load(self, ref2va_model, fl2va_model, text_encoder, video_vae, audio_vae, weight_dtype):
        import nodes
        clip, = nodes.NODE_CLASS_MAPPINGS["CLIPLoader"]().load_clip(text_encoder, "minimax_h3")
        vvae, = nodes.VAELoader().load_vae(video_vae)
        avae, = nodes.VAELoader().load_vae(audio_vae)
        return (H3Bundle(clip, vvae, avae,
                         {"ref2va": ref2va_model, "fl2va": fl2va_model,
                          "clip": text_encoder, "video_vae": video_vae, "audio_vae": audio_vae},
                         weight_dtype),)


class H3Model:
    CATEGORY = "沐阳 H3"
    FUNCTION = "get"
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    DESCRIPTION = "从加载器取出扩散模型，用来挂 SageAttn / LowVRAM / Lora 补丁链。"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "h3": ("MYANG_H3",),
                "kind": (["ref2va", "fl2va"], {
                    "default": "ref2va",
                    "tooltip": "要和主节点的任务模式对上：动作迁移 / 续写用 ref2va，"
                               "首尾帧 / 纯文生用 fl2va"}),
            },
        }

    def get(self, h3, kind):
        return (h3.model_for(kind),)


class H3Condition:
    CATEGORY = "沐阳 H3"
    FUNCTION = "build"
    RETURN_TYPES = ("CONDITIONING", "LATENT", "INT", "FLOAT")
    RETURN_NAMES = ("positive", "latent", "frames", "fps")
    DESCRIPTION = ("提示词 + 参考素材 → 条件和空 latent。走 ComfyUI 官方的 "
                   "MiniMaxH3ReferenceToVideo / MiniMaxH3ImageToVideo，"
                   "这里负责分辨率档位、秒数换帧数、参考图面积预算、"
                   "以及把 @图片1 或 @文件名 翻译成 <Picture 1>。")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "h3": ("MYANG_H3",),
                "prompt": ("STRING", {"forceInput": True}),
                "resolution": (RESOLUTION_PRESETS, {"default": "480P"}),
                "aspect_ratio": (ASPECT_RATIOS, {"default": "16:9"}),
                "width": ("INT", {"default": 864, "min": 32, "max": 16384, "step": 32}),
                "height": ("INT", {"default": 480, "min": 32, "max": 16384, "step": 32}),
                "seconds": ("FLOAT", {"default": 5.0, "min": 0.2, "max": 30.0, "step": 0.1}),
                "ref_image_size": (list(REF_IMAGE_SIZES), {
                    "default": REF_AREA_MATCH,
                    "tooltip": "参考图的面积预算。按面积而不是短边：同样短边下 21:9 的图"
                               "比 1:1 贵一倍多，而参考 token 每一步都要算"}),
                "reference_mention_mode": (MENTION_MODES, {"default": MENTION_INDEX}),
            },
            "optional": {
                "media": ("MINIMAX_H3_MEDIA", {
                    "tooltip": "素材包。图片、视频、音频按包里的顺序编号，"
                               "和提示词里的 @图片1 / @视频1 对应"}),
                "ref_video": ("IMAGE", {
                    "tooltip": "循环内部切出的分段参考视频。它排在素材包前面，"
                               "所以永远是 @视频1"}),
                "ref_audio": ("AUDIO", {
                    "tooltip": "与分段参考视频同步切出的音频；它排在素材包音频前面"}),
                "first_frame": ("IMAGE", {"tooltip": "接上就走首尾帧模式（官方 fl2va 通路）"}),
                "last_frame": ("IMAGE",),
            },
        }

    def build(self, h3, prompt, resolution, aspect_ratio, width, height, seconds,
              ref_image_size, reference_mention_mode=MENTION_INDEX, media=None,
              ref_video=None, ref_audio=None, first_frame=None, last_frame=None):
        core = _core()
        w, hgt = canvas_for(resolution, aspect_ratio, width, height)
        length = length_for(seconds)
        frames_out = core.align_frame_count(max(5, length))

        # Keyframes are a different task: the official fl2va path takes the
        # keyframes and the prompt, and no references at all.
        if first_frame is not None or last_frame is not None:
            cond, latent = core.MiniMaxH3ImageToVideo.execute(
                clip=h3.clip, vae=h3.video_vae, prompt=resolve_mentions(prompt, {}),
                width=w, height=hgt, length=length,
                first_frame=image_batch(first_frame) if first_frame is not None else None,
                last_frame=image_batch(last_frame) if last_frame is not None else None)
            logger.info("H3-Myang: 首尾帧 %dx%d %d 帧 (%.2fs)", w, hgt, frames_out, frames_out / FPS)
            return (cond, latent, frames_out, float(FPS))

        budget = REF_IMAGE_SIZES.get(str(ref_image_size), None)
        if budget is None:
            budget = w * hgt

        images, videos, video_audios, audios = {}, {}, {}, {}
        counts = {"Picture": 0, "Video": 0, "Audio": 0}
        by_filename = {}

        # A direct segment slice comes first by contract: loop prompts call it
        # @视频1 whenever no Agent bundle owns the numbering.
        if ref_video is not None:
            clip_frames, soundtrack, source_fps = video_stream(ref_video)
            counts["Video"] = 1
            videos["ref_video_1"] = _to_24fps(clip_frames, source_fps)
            if soundtrack is not None:
                counts["Audio"] += 1
                video_audios["ref_video_audio_1"] = soundtrack

        if ref_audio is not None:
            if ref_video is not None:
                video_audios["ref_video_audio_1"] = audio_track(ref_audio)
            else:
                counts["Audio"] += 1
                audios["direct_audio_1"] = audio_track(ref_audio)

        for kind, payload, name in iter_media(media):
            if kind == "image":
                counts["Picture"] += 1
                # Budget the reference here rather than leaving it to the core's
                # two fixed choices, then hand it over at "max" so the core --
                # which only ever scales down -- leaves it exactly as sized.
                images[f"ref_image_{counts['Picture']}"] = _fit_reference(image_batch(payload), budget)
                if name:
                    by_filename[name] = ("Picture", counts["Picture"])
            elif kind == "video":
                clip_frames, soundtrack, source_fps = video_stream(payload)
                counts["Video"] += 1
                ordinal = counts["Video"]
                videos[f"ref_video_{ordinal}"] = _to_24fps(clip_frames, source_fps)
                if soundtrack is not None:
                    counts["Audio"] += 1
                    video_audios[f"ref_video_audio_{ordinal}"] = soundtrack
                if name:
                    by_filename[name] = ("Video", ordinal)
            elif kind == "audio":
                counts["Audio"] += 1
                audios[f"ref_audio_{counts['Audio']}"] = audio_track(payload)
                if name:
                    by_filename[name] = ("Audio", counts["Audio"])

        names = by_filename if str(reference_mention_mode) == MENTION_FILENAME else None
        text = resolve_mentions(prompt, counts, names)

        # An unresolved mention is left as literal text by design, so that a
        # typo cannot silently point at somebody else's media. The cost is that
        # it fails quietly: the model just never receives a <Picture 1> and the
        # render comes out looking like the reference was ignored. Say so here
        # instead, since that is the one failure nobody can see from the output.
        leftover = MENTION_RE.findall(text)
        if leftover:
            missing = "、".join(f"@{kind}{ordinal}" for kind, ordinal in leftover)
            have = f"图片 {counts['Picture']}、视频 {counts['Video']}、音频 {counts['Audio']}"
            raise ValueError(
                f"提示词里的 {missing} 找不到对应素材（这次实际收到：{have}）。\n"
                "最常见的原因是换素材时新建了加载节点：Agent 的素材登记还指向旧节点，"
                "新节点没被登记，所以没进素材包。\n"
                "解决：把新的加载节点重新拖进 Agent 的 media 插槽（重新连线才会重新登记）；"
                "或者不要新建节点，直接在原来的加载节点里换文件。")

        cond, latent = core.MiniMaxH3ReferenceToVideo.execute(
            clip=h3.clip, vae=h3.video_vae, audio_vae=h3.audio_vae, prompt=text,
            width=w, height=hgt, length=length, ref_image_size="max",
            ref_images=images or None, ref_videos=videos or None,
            ref_video_audios=video_audios or None, ref_audios=audios or None,
        )
        logger.info("H3-Myang: %dx%d %d 帧 (%.2fs) | 图%d 视频%d 音频%d | 参考预算 %s",
                    w, hgt, frames_out, frames_out / FPS,
                    counts["Picture"], counts["Video"], counts["Audio"], ref_image_size)
        if by_filename:
            logger.info("H3-Myang: 本次素材 %s",
                        "，".join(f"<{tag} {num}>={nm}" for nm, (tag, num) in by_filename.items()))

        # The one thing that cannot be read off the finished video: whether the
        # reference tags actually made it into the prompt, and how much of the
        # attention sequence each reference is worth. A portrait that loses to
        # the reference video by two orders of magnitude will be ignored no
        # matter how the prompt is worded.
        shown = text if len(text) <= 300 else text[:300] + "…"
        logger.info("H3-Myang: 送入采样器的提示词: %s", shown)
        weights = []
        for name, tensor in images.items():
            weights.append("%s %dx%d=%d token"
                           % (name, int(tensor.shape[2]), int(tensor.shape[1]),
                              (int(tensor.shape[2]) // 16) * (int(tensor.shape[1]) // 16)))
        for name, tensor in videos.items():
            t = core.video_latent_t(int(tensor.shape[0]))
            weights.append("%s %dx%d×%d帧=%d token"
                           % (name, int(tensor.shape[2]), int(tensor.shape[1]), t,
                              t * (int(tensor.shape[2]) // 16) * (int(tensor.shape[1]) // 16)))
        if weights:
            logger.info("H3-Myang: 参考素材权重 %s", " | ".join(weights))
        return (cond, latent, frames_out, float(FPS))


def _fit_reference(image, target_area: int):
    """Resize a reference image to the area budget, keeping the aspect."""
    if image is None or target_area <= 0:
        return image
    import comfy.utils
    h, w = int(image.shape[1]), int(image.shape[2])
    tw, th = aligned_size(w, h, int(target_area))
    if (tw, th) == (w, h):
        return image
    samples = image[..., :3].movedim(-1, 1)
    samples = comfy.utils.common_upscale(samples, tw, th, "lanczos", "disabled")
    return samples.movedim(1, -1)


def media_rows(media):
    """Yield catalog rows using the exact ordering consumed by H3Condition."""
    yield from catalog_rows(media)


def iter_media(media):
    """Yield normalized catalog assets, grouped image/video/audio."""
    yield from iter_catalog(media)



NODE_CLASS_MAPPINGS = {"H3Loader": H3Loader, "H3Model": H3Model, "H3Condition": H3Condition}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3Loader": "沐阳 H3 加载器",
    "H3Model": "沐阳 H3 取模型",
    "H3Condition": "沐阳 H3 条件（提示词 + 素材）",
}
