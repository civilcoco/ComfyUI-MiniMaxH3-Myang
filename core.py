"""沐阳 H3 · 自己的采样核心。

SPDX-License-Identifier: GPL-3.0-only

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

Media normalization and reference-editor compatibility contain adaptations
from nkxx188/ComfyUI-MiniMaxH3-Easy (MIT).  See THIRD_PARTY_NOTICES.md.
"""

import collections
import logging
import math
import re

import torch

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
MODEL_LAYOUT_SEPARATE = "双模型（原生 FL2VA + Ref2VA）"
MODEL_LAYOUT_HYBRID = "混合模型（实验·FL2VA/Ref2VA 共用）"
MODEL_LAYOUTS = [MODEL_LAYOUT_SEPARATE, MODEL_LAYOUT_HYBRID]
NO_HYBRID_MODEL = "未选择混合模型"

# Reference weights deliberately compensate only part of a token imbalance.
# Exact inverse-token balancing would turn a small portrait beside a long video
# into a 20x-50x attention multiplier, far outside the model's training range.
REFERENCE_AUTO_EXPONENT = 0.20
REFERENCE_AUTO_MIN = 0.50
REFERENCE_AUTO_MAX = 2.25
REFERENCE_MANUAL_MIN = 0.25
REFERENCE_MANUAL_MAX = 3.00

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


def _frames(value):
    if isinstance(value, torch.Tensor):
        if value.ndim == 3:
            return value.unsqueeze(0)
        if value.ndim == 4:
            return value
    if isinstance(value, (tuple, list)):
        tensors = []
        for item in value:
            if isinstance(item, torch.Tensor):
                tensors.append(item if item.ndim == 4 else item.unsqueeze(0))
        if tensors:
            return torch.cat(tensors, dim=0)
    raise ValueError(
        "H3-Myang: 无法从 %s 提取视频帧" % type(value).__name__)


def _audio(value):
    if value is None:
        return None
    if isinstance(value, dict) and value.get("waveform") is not None:
        sample_rate = int(
            value.get("sample_rate") or value.get("samplerate")
            or value.get("sampler_rate") or 32000)
        waveform = value["waveform"]
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0).unsqueeze(0)
        elif waveform.ndim == 2:
            waveform = waveform.unsqueeze(0)
        return {"waveform": waveform, "sample_rate": sample_rate}
    if hasattr(value, "audio"):
        return _audio(getattr(value, "audio"))
    if isinstance(value, (tuple, list)):
        for item in value:
            if isinstance(item, dict) and (
                    "waveform" in item or "sample_rate" in item):
                return _audio(item)
    raise ValueError(
        "H3-Myang: 无法从 %s 提取声音" % type(value).__name__)


def _video_parts(value):
    """Normalise IMAGE batches and modern VIDEO objects to official inputs."""
    if value is None:
        raise ValueError("H3-Myang: 参考视频为空")

    if hasattr(value, "get_components"):
        components = value.get_components()
        images = getattr(components, "images", None)
        if images is not None:
            return (_frames(images),
                    _audio(getattr(components, "audio", None))
                    if getattr(components, "audio", None) is not None else None,
                    float(getattr(components, "frame_rate", 24.0) or 24.0))

    if isinstance(value, torch.Tensor):
        return _frames(value), None, 24.0

    if isinstance(value, dict):
        images = next((value.get(name) for name in
                       ("images", "frames", "video", "samples")
                       if value.get(name) is not None), None)
        if images is not None:
            soundtrack = value.get("audio")
            if soundtrack is None:
                soundtrack = value.get("soundtrack")
            rate = float(value.get("fps") or value.get("frame_rate")
                         or value.get("framerate") or 24.0)
            return (_frames(images),
                    _audio(soundtrack) if soundtrack is not None else None,
                    rate)

    if isinstance(value, (tuple, list)):
        images, soundtrack, rate = None, None, 24.0
        for item in value:
            if isinstance(item, torch.Tensor) and images is None:
                images = item
            elif isinstance(item, dict) and (
                    "waveform" in item or "sample_rate" in item):
                soundtrack = item
            elif isinstance(item, (int, float)) and not isinstance(item, bool):
                if 1.0 <= float(item) <= 240.0:
                    rate = float(item)
            elif isinstance(item, dict) and images is None and any(
                    name in item for name in ("images", "frames", "video")):
                return _video_parts(item)
        if images is not None:
            return (_frames(images),
                    _audio(soundtrack) if soundtrack is not None else None,
                    rate)

    raise ValueError(
        "H3-Myang: 不支持的参考视频类型 %s" % type(value).__name__)


def _to_24fps(frames, source_fps):
    source_fps = float(source_fps or 24.0)
    if abs(source_fps - 24.0) < 0.01:
        return frames
    count = max(1, round(int(frames.shape[0]) * 24.0 / source_fps))
    indices = torch.linspace(
        0, int(frames.shape[0]) - 1, count,
        device=frames.device).round().long()
    return frames[indices]


_BoundedPlan = collections.namedtuple("_BoundedPlan", "reformat turns canvas")

_REFORMAT_EXTRA = None


def _reformat_extra():
    """Downscale with lanczos when this pyav build exposes swscale filters."""
    global _REFORMAT_EXTRA
    if _REFORMAT_EXTRA is None:
        try:
            from av.video.reformatter import Interpolation
            _REFORMAT_EXTRA = {"interpolation": Interpolation.LANCZOS}
        except Exception:
            _REFORMAT_EXTRA = {}
    return _REFORMAT_EXTRA


def _bounded_frame_estimate(stream, step):
    """How many frames a bounded decode keeps, for the up-front budget check."""
    total = int(getattr(stream, "frames", 0) or 0)
    if total <= 0:
        rate = float(stream.average_rate or 0)
        if stream.duration and stream.time_base and rate > 0:
            total = int(float(stream.duration * stream.time_base) * rate)
    return int(total / step) if total > 0 else 0


def _bounded_plan(frame, size_policy, stream, step, path):
    """Fix the decode canvas from the first frame and vet what it will cost.

    Rotation is applied after scaling, so a clip tagged 90 degrees has to be
    scaled to the transposed canvas for the finished frame to come out at the
    size the policy asked for.

    The budget check exists because the alternative is grinding through the
    whole clip and then dying inside ``DefaultCPUAllocator``, which reads like
    a VRAM problem and is not one. Peak is the uint8 accumulation plus the
    float32 IMAGE it gets promoted to: five bytes per subpixel.
    """
    rotation = getattr(frame, "rotation", 0) or 0
    turns = int(round(rotation // 90)) % 4
    width, height = int(frame.width), int(frame.height)
    shown = (height, width) if turns % 2 else (width, height)
    canvas = tuple(size_policy(*shown)) if size_policy else shown
    canvas = (max(2, int(canvas[0])), max(2, int(canvas[1])))
    reformat = {"format": "rgb24", **_reformat_extra()}
    reformat["width"], reformat["height"] = (
        (canvas[1], canvas[0]) if turns % 2 else canvas)

    kept = _bounded_frame_estimate(stream, step)
    needed = kept * canvas[0] * canvas[1] * 3 * 5
    gib = float(1024 ** 3)
    logger.info(
        "H3-Myang: 参考视频解码画布 %dx%d | 约 %d 帧 | 预计峰值内存 %.1f GiB",
        canvas[0], canvas[1], kept, needed / gib)
    try:
        import psutil
        available = int(psutil.virtual_memory().available)
    except Exception:
        available = 0
    if kept > 0 and available > 0 and needed > available:
        raise ValueError(
            "参考视频解码需要约 %.1f GiB 内存（%d 帧 × %dx%d），当前只有 %.1f GiB 可用。"
            "请把『参考视频分辨率』调低，或先把动作视频转成更小的画布再上传：%s"
            % (needed / gib, kept, canvas[0], canvas[1], available / gib, path))
    return _BoundedPlan(reformat, turns, canvas)


def _bounded_picture(frame, plan):
    import numpy
    picture = frame.reformat(**plan.reformat).to_ndarray(format="rgb24")
    if plan.turns:
        picture = numpy.rot90(picture, k=plan.turns, axes=(0, 1))
    return numpy.ascontiguousarray(picture)


def decode_video_bounded(path, size_policy=None, target_fps=24.0):
    """Decode a clip straight onto the canvas that will actually be used.

    ComfyUI's own ``VideoFromFile.get_components`` decodes every frame to
    float32 RGB at native size and stacks the whole clip, so a 2160x3840/60fps
    phone reference asks for ~78GiB of contiguous CPU RAM -- and it asks
    *before* any downstream resize node can shrink anything, so the run dies in
    the CPU allocator with the GPU still idle.

    Here the resize and the frame decimation happen inside the decode loop, so
    peak memory follows the canvas that gets used rather than the one that was
    filmed. Frames accumulate as uint8 and are promoted to float once.

    Returns the same ``(IMAGE, AUDIO|None, fps)`` triple as ``_video_parts``, so
    callers can keep piping it through ``_to_24fps``; that stays a no-op when
    the decimation already happened here.
    """
    import av
    import numpy

    with av.open(str(path)) as container:
        video = next((s for s in container.streams if s.type == "video"), None)
        if video is None:
            raise ValueError("H3-Myang: 参考视频没有画面轨道：%s" % path)
        audio = next((s for s in container.streams if s.type == "audio"), None)
        source_fps = float(video.average_rate or target_fps or 24.0)
        wanted_fps = float(target_fps or 0.0)
        step = (source_fps / wanted_fps
                if wanted_fps > 0.0 and source_fps > wanted_fps + 0.01 else 1.0)

        streams, resampler = [video], None
        if audio is not None:
            resampler = av.audio.resampler.AudioResampler(format="fltp")
            streams.append(audio)

        pictures, waves, plan = [], [], None
        index, due = 0, 0.0
        for packet in container.demux(*streams):
            if packet.stream.type != "video":
                if resampler is not None:
                    for frame in packet.decode():
                        waves.extend(chunk.to_ndarray()
                                     for chunk in resampler.resample(frame))
                continue
            try:
                decoded = list(packet.decode())
            except av.error.InvalidDataError:
                logger.info("H3-Myang: 参考视频有损坏的画面包，已跳过")
                continue
            for frame in decoded:
                take = index >= due - 1e-9
                index += 1
                if not take:
                    continue
                due += step
                if plan is None:
                    plan = _bounded_plan(frame, size_policy, video, step, path)
                pictures.append(_bounded_picture(frame, plan))

        if not pictures:
            raise ValueError("H3-Myang: 参考视频没有解出可用画面：%s" % path)
        images = torch.from_numpy(numpy.stack(pictures))
        pictures.clear()
        images = images.float().div_(255.0)

        soundtrack = None
        if waves:
            soundtrack = {
                "waveform": torch.from_numpy(
                    numpy.concatenate(waves, axis=1)).unsqueeze(0),
                "sample_rate": int(audio.sample_rate or 0) or 32000}
    return images, soundtrack, (wanted_fps if step > 1.0 else source_fps)


class H3Bundle:
    """The models H3 needs, carried as one link.

    H3 ships two transformers: ref2va drives reference-to-video, fl2va drives
    text/first-last-frame. Only one is ever resident -- they are ~32GB each --
    so the bundle names both and loads on demand, swapping when the task
    changes. Holding both at once is what an eager loader would cost.
    """

    def __init__(self, clip, video_vae, audio_vae, names, weight_dtype="default",
                 model_layout=MODEL_LAYOUT_SEPARATE):
        self.clip = clip
        self.video_vae = video_vae
        self.audio_vae = audio_vae
        self.names = names
        self.weight_dtype = weight_dtype
        self.model_layout = (
            MODEL_LAYOUT_HYBRID
            if str(model_layout) == MODEL_LAYOUT_HYBRID
            else MODEL_LAYOUT_SEPARATE)
        self._cached_kind = None
        self._cached_name = None
        self._cached_model = None

    def model_for(self, kind: str):
        kind = "fl2va" if str(kind).lower().startswith("fl") else "ref2va"
        name = (self.names.get("hybrid")
                if self.model_layout == MODEL_LAYOUT_HYBRID
                else self.names.get(kind))
        if not name:
            raise ValueError(f"加载器没有配置 {kind} 模型")
        # A hybrid checkpoint deliberately serves both condition families.
        # Cache by filename rather than requested family so asking for FL2VA
        # after Ref2VA does not reload the same ~20GB model a second time.
        if self._cached_name == name and self._cached_model is not None:
            self._cached_kind = kind
            return self._cached_model
        _ensure_aimdo_hostbuf_headroom()
        import nodes
        model, = nodes.NODE_CLASS_MAPPINGS["UNETLoader"]().load_unet(name, self.weight_dtype)
        from .progress import _install_reference_weight_bridge
        _install_reference_weight_bridge(model)
        self._cached_kind, self._cached_name, self._cached_model = kind, name, model
        logger.info(
            "H3-Myang: 已加载 %s 模型 %s%s", kind, name,
            "（FL2VA/Ref2VA 混合共用）"
            if self.model_layout == MODEL_LAYOUT_HYBRID else "")
        return model

    def __repr__(self):
        if self.model_layout == MODEL_LAYOUT_HYBRID:
            return f"<H3Bundle hybrid={self.names.get('hybrid', '?')}>"
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
                   "不会同时占显存；也可选择一个兼容两种条件的混合模型来避免切换。"
                   "模型用「沐阳 H3 取模型」引出来挂补丁链。")

    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        unet = folder_paths.get_filename_list("diffusion_models")
        hybrid_choices = [NO_HYBRID_MODEL] + list(unet)
        return {
            "required": {
                "ref2va_model": (unet, {"tooltip": "参考生视频用（动作迁移 / 续写）"}),
                "fl2va_model": (unet, {"tooltip": "文生视频、首尾帧用"}),
                "text_encoder": (folder_paths.get_filename_list("text_encoders"),),
                "video_vae": (folder_paths.get_filename_list("vae"),),
                "audio_vae": (folder_paths.get_filename_list("vae"),),
                "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"],
                                 {"default": "default"}),
                # Append-only: saved H3Loader widget values are positional.
                "model_layout": (MODEL_LAYOUTS, {
                    "default": MODEL_LAYOUT_SEPARATE,
                    "tooltip": "双模型保持官方分工；混合档让 FL2VA/Ref2VA 共用同一"
                               "checkpoint，切换任务时不再换入另一套大权重。混合权重"
                               "属于第三方实验模型，请自行确认兼容性。"}),
                "hybrid_model": (hybrid_choices, {
                    "default": NO_HYBRID_MODEL,
                    "tooltip": "仅混合模型档生效；文件放在 models/diffusion_models。"}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return "|".join(str(kwargs.get(k, "")) for k in
                        ("ref2va_model", "fl2va_model", "text_encoder",
                         "video_vae", "audio_vae", "weight_dtype",
                         "model_layout", "hybrid_model"))

    def load(self, ref2va_model, fl2va_model, text_encoder, video_vae,
             audio_vae, weight_dtype, model_layout=MODEL_LAYOUT_SEPARATE,
             hybrid_model=NO_HYBRID_MODEL):
        import nodes
        layout = (MODEL_LAYOUT_HYBRID
                  if str(model_layout) == MODEL_LAYOUT_HYBRID
                  else MODEL_LAYOUT_SEPARATE)
        hybrid_name = str(hybrid_model or "")
        if layout == MODEL_LAYOUT_HYBRID and (
                not hybrid_name or hybrid_name == NO_HYBRID_MODEL):
            raise ValueError(
                "已选择『混合模型』档位，但还没有选择混合 FL2VA/Ref2VA 权重")
        clip, = nodes.NODE_CLASS_MAPPINGS["CLIPLoader"]().load_clip(text_encoder, "minimax_h3")
        vvae, = nodes.VAELoader().load_vae(video_vae)
        avae, = nodes.VAELoader().load_vae(audio_vae)
        return (H3Bundle(clip, vvae, avae,
                         {"ref2va": ref2va_model, "fl2va": fl2va_model,
                          "hybrid": hybrid_name if layout == MODEL_LAYOUT_HYBRID else "",
                          "clip": text_encoder, "video_vae": video_vae, "audio_vae": audio_vae},
                         weight_dtype, model_layout=layout),)


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
                # Append-only: the action-transfer video normally arrives
                # through this direct socket rather than the Media Agent
                # bundle.  Keeping these fields at the end preserves the
                # optional-input order of existing saved workflows.
                "ref_video_weight_mode": (["off", "auto", "manual"], {
                    "default": "auto",
                    "tooltip": "动作迁移参考视频权重：auto 按 token 数自动计算，"
                               "manual 使用下方数值；旧工作流省略时按 auto。"}),
                "ref_video_weight": ("FLOAT", {
                    "default": 1.0, "min": REFERENCE_MANUAL_MIN,
                    "max": REFERENCE_MANUAL_MAX, "step": 0.05}),
            },
        }

    def build(self, h3, prompt, resolution, aspect_ratio, width, height, seconds,
              ref_image_size, reference_mention_mode=MENTION_INDEX, media=None,
              ref_video=None, ref_audio=None, first_frame=None, last_frame=None,
              ref_video_weight_mode="auto", ref_video_weight=1.0):
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
                first_frame=_frames(first_frame) if first_frame is not None else None,
                last_frame=_frames(last_frame) if last_frame is not None else None)
            logger.info("H3-Myang: 首尾帧 %dx%d %d 帧 (%.2fs)", w, hgt, frames_out, frames_out / FPS)
            return (cond, latent, frames_out, float(FPS))

        budget = REF_IMAGE_SIZES.get(str(ref_image_size), None)
        if budget is None:
            budget = w * hgt

        images, videos, video_audios, audios = {}, {}, {}, {}
        reference_policies = {"image": {}, "video": {}, "audio": {}}
        counts = {"Picture": 0, "Video": 0, "Audio": 0}
        by_filename = {}

        # A direct segment slice comes first by contract: loop prompts call it
        # @视频1 whenever no Agent bundle owns the numbering.
        if ref_video is not None:
            clip_frames, soundtrack, source_fps = _video_parts(ref_video)
            counts["Video"] = 1
            videos["ref_video_1"] = _to_24fps(clip_frames, source_fps)
            direct_mode = str(ref_video_weight_mode or "auto").strip().lower()
            if direct_mode not in {"off", "auto", "manual"}:
                direct_mode = "auto"
            try:
                direct_weight = max(
                    REFERENCE_MANUAL_MIN,
                    min(REFERENCE_MANUAL_MAX, float(ref_video_weight or 1.0)))
            except (TypeError, ValueError):
                direct_weight = 1.0
            reference_policies["video"]["ref_video_1"] = (
                direct_mode, direct_weight)
            if soundtrack is not None:
                counts["Audio"] += 1
                video_audios["ref_video_audio_1"] = soundtrack

        if ref_audio is not None:
            if ref_video is not None:
                video_audios["ref_video_audio_1"] = _audio(ref_audio)
            else:
                counts["Audio"] += 1
                audios["direct_audio_1"] = _audio(ref_audio)
                reference_policies["audio"]["direct_audio_1"] = ("off", 1.0)

        for item, link in _sorted_media(media):
            kind = str(getattr(item, "media_type", "image")).lower()
            payload = getattr(item, "value", None)
            name = str(link.get("filename") or link.get("subject") or "")
            mode = str(getattr(item, "reference_weight_mode", "off") or "off").lower()
            if mode not in {"auto", "manual"}:
                mode = "off"
            manual_weight = max(REFERENCE_MANUAL_MIN, min(
                REFERENCE_MANUAL_MAX,
                float(getattr(item, "reference_weight", 1.0) or 1.0)))
            if kind == "image":
                counts["Picture"] += 1
                # Budget the reference here rather than leaving it to the core's
                # two fixed choices, then hand it over at "max" so the core --
                # which only ever scales down -- leaves it exactly as sized.
                key = f"ref_image_{counts['Picture']}"
                images[key] = _fit_reference(_frames(payload), budget)
                reference_policies["image"][key] = (mode, manual_weight)
                if name:
                    by_filename[name] = ("Picture", counts["Picture"])
            elif kind == "video":
                clip_frames, soundtrack, source_fps = _video_parts(payload)
                counts["Video"] += 1
                ordinal = counts["Video"]
                key = f"ref_video_{ordinal}"
                videos[key] = _to_24fps(clip_frames, source_fps)
                reference_policies["video"][key] = (mode, manual_weight)
                if soundtrack is not None:
                    counts["Audio"] += 1
                    video_audios[f"ref_video_audio_{ordinal}"] = soundtrack
                if name:
                    by_filename[name] = ("Video", ordinal)
            elif kind == "audio":
                counts["Audio"] += 1
                key = f"ref_audio_{counts['Audio']}"
                audios[key] = _audio(payload)
                reference_policies["audio"][key] = (mode, manual_weight)
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
        weight_plan = reference_weight_plan(
            core, images, videos, audios, reference_policies, frames_out)
        cond = apply_reference_weight_plan(cond, weight_plan)
        if any(abs(float(item["weight"]) - 1.0) > 1e-9 for item in weight_plan):
            # The stock PackedLayout does not know which packed rows belong to
            # which reference.  Myang's layout patch records that mapping so
            # the sampler can scale those rows; install it now, because a
            # single-segment run never reaches the long-video anchor path that
            # would otherwise install it.
            from .anchor_compat import ensure_anchors
            ensure_anchors()
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
        if weight_plan:
            logger.info(
                "H3-Myang: 参考素材有效权重 %s",
                " | ".join(
                    "%s %s rows=%d weight=%.2f%s" % (
                        item["name"], item["kind"], item["rows"],
                        item["weight"],
                        "(手动)" if item["mode"] == "manual"
                        else "(自动)" if item["mode"] == "auto"
                        else "(关闭)")
                    for item in weight_plan))
        return (cond, latent, frames_out, float(FPS))


def _official_reference_rows(core_module, kind, value, frame_limit):
    """Estimate the exact visual rows that H3 packs after its own resize."""
    if kind == "audio" or value is None:
        return 0
    height, width = int(value.shape[1]), int(value.shape[2])
    if kind == "image":
        short_edge = float(getattr(core_module, "REF_IMAGE_SHORT_EDGE", 2048))
        scale = min(1.0, short_edge / min(width, height))
        width = max(CANVAS_MULTIPLE, round(width * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        height = max(CANVAS_MULTIPLE, round(height * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        return max(1, (width // 32) * (height // 32))

    adapt = getattr(core_module, "adapt_canvas", None)
    if callable(adapt):
        canvas_w, canvas_h = adapt(width, height)
    else:
        canvas_w = max(CANVAS_MULTIPLE, round(width / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        canvas_h = max(CANVAS_MULTIPLE, round(height / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
    if width * height < canvas_w * canvas_h:
        canvas_w = max(CANVAS_MULTIPLE, round(width / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        canvas_h = max(CANVAS_MULTIPLE, round(height / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
    frames = min(int(value.shape[0]), int(frame_limit))
    while frames >= 5 and frames % 17 != 5:
        frames -= 1
    if frames < 5:
        return 0
    to_latent_t = getattr(core_module, "video_latent_t", None)
    latent_t = int(to_latent_t(frames) if callable(to_latent_t) else (frames + 3) // 4)
    return max(1, latent_t * (canvas_w // 32) * (canvas_h // 32))


def reference_weight_plan(core_module, images, videos, audios, policies,
                          frame_limit):
    """Resolve automatic/manual settings to bounded per-reference weights.

    The first action/reference video is the 1.0 anchor. A fifth-root token
    correction gives small still references enough influence without attempting
    the destructive 1/N equalization that a long video would otherwise demand.
    """
    items = []
    for kind, values in (("image", images), ("video", videos), ("audio", audios)):
        for name, value in values.items():
            policy = policies.get(kind, {}).get(name)
            if policy is None:
                # Old graphs predate the direct-video weight inputs. Their
                # action clip must still follow the UI's automatic policy.
                default_mode = ("auto" if kind == "video"
                                and name == "ref_video_1" else "off")
                mode, manual = default_mode, 1.0
            else:
                mode, manual = policy
            items.append({
                "kind": kind,
                "name": name,
                "rows": _official_reference_rows(
                    core_module, kind, value, frame_limit),
                "mode": mode,
                "manual": manual,
            })
    video_anchor = next(
        (item["rows"] for item in items
         if item["kind"] == "video" and item["rows"] > 0), 0)
    if not video_anchor:
        visual = [item["rows"] for item in items if item["rows"] > 0]
        video_anchor = max(visual, default=1)
    for item in items:
        if item["mode"] == "manual":
            weight = item["manual"]
        elif item["mode"] == "auto" and item["rows"] > 0:
            weight = (video_anchor / item["rows"]) ** REFERENCE_AUTO_EXPONENT
            weight = max(REFERENCE_AUTO_MIN, min(REFERENCE_AUTO_MAX, weight))
        else:
            weight = 1.0
        item["weight"] = round(float(weight), 4)
    return items


def apply_reference_weight_plan(conditioning, plan):
    """Attach weights to the official ref blocks without changing their data."""
    if not plan or not isinstance(conditioning, (list, tuple)):
        return conditioning
    result = []
    for entry in conditioning:
        if not isinstance(entry, (list, tuple)) or len(entry) < 2:
            result.append(entry)
            continue
        metadata = dict(entry[1])
        refs = metadata.get("minimax_refs")
        if isinstance(refs, (list, tuple)):
            weighted = []
            for index, block in enumerate(refs):
                copy = dict(block)
                if index < len(plan):
                    copy["reference_weight"] = plan[index]["weight"]
                    copy["reference_weight_mode"] = plan[index]["mode"]
                weighted.append(copy)
            metadata["minimax_refs"] = weighted
        result.append([entry[0], metadata])
    return result


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


def _sorted_media(media):
    """Yield (item, link) in the one order every consumer must agree on.

    Ordering is the contract with the prompt: the core numbers references per
    type in the order they are handed over, and the editor numbers its chips
    the same way, so the two only agree if this order is stable. Within a type
    the original input index decides, so swapping one clip for another (as the
    loop does per segment) cannot renumber the rest.
    """
    if media is None:
        return
    items = getattr(media, "items", None) or ()
    links = {int(l.get("order", i + 1)): l
             for i, l in enumerate(getattr(media, "links", None) or ())}
    order = {"image": 0, "video": 1, "audio": 2}

    def key(item):
        kind = str(getattr(item, "media_type", "image")).lower()
        return (order.get(kind, 0), int(getattr(item, "input_index", 0)))

    for item in sorted(items, key=key):
        yield item, links.get(int(getattr(item, "input_index", 0)), {})


def media_rows(media):
    """Yield (kind, ordinal, subject, filename) with the prompt's own numbering.

    The ordinal is the N in ``@图片N`` / ``<Picture N>``: counted per type over
    :func:`_sorted_media`, exactly like ``H3Condition`` counts it.  Callers that
    describe a bundle to a human or an LLM must use this and not ``input_index``,
    which runs across all types at once.
    """
    counts = {"image": 0, "video": 0, "audio": 0}
    for item, link in _sorted_media(media):
        kind = str(getattr(item, "media_type", "image")).lower()
        if kind not in counts:
            continue
        counts[kind] += 1
        yield (kind, counts[kind],
               str(link.get("subject") or "").strip(),
               str(link.get("filename") or "").strip())


def iter_media(media):
    """Yield (kind, payload, filename) from a media bundle, images then video/audio."""
    for item, link in _sorted_media(media):
        name = str(link.get("filename") or link.get("subject") or "")
        yield (str(getattr(item, "media_type", "image")).lower(),
               getattr(item, "value", None), name)



NODE_CLASS_MAPPINGS = {"H3Loader": H3Loader, "H3Model": H3Model, "H3Condition": H3Condition}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3Loader": "沐阳 H3 加载器",
    "H3Model": "沐阳 H3 取模型",
    "H3Condition": "沐阳 H3 条件（提示词 + 素材）",
}
