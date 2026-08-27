"""Reuse the Myang Media Agent's bundle inside the segment loop.

The Agent node is where media actually gets authored: it previews every
connected image and clip, numbers them, validates that each ``@图片1`` /
``@视频1`` in the prompt points at something real, and emits the prompt in the
form the H3 core expects. Rebuilding any of that inside a long-video node
would be a worse copy of it.

What the loop needs that the Agent cannot provide is a *different* reference
clip per segment: segment i watches frames ``(i-1)*(frames-overlap)`` onward.
So the Agent keeps ownership of the whole bundle and this node swaps one entry
in it — same slot, same ordinal, same media type, different frames. The prompt
the Agent wrote keeps resolving to the media the operator wired up.
"""

import json
import logging
import os

import torch

import folder_paths
from comfy_api.latest import InputImpl

from . import core
from .media_catalog import MyangMediaAsset, MyangMediaCatalog, video_stream

logger = logging.getLogger(__name__)

SHOT_MEDIA_LIMITS = {"image": 9, "video": 3, "audio": 3}
REFERENCE_VIDEO_ORIGINAL = "匹配参考视频原分辨率"
REFERENCE_VIDEO_RESOLUTIONS = [
    REFERENCE_VIDEO_ORIGINAL, *core.RESOLUTION_PRESETS,
]


def _safe_input_path(asset):
    """Resolve a Director upload without accepting an arbitrary filesystem path."""

    file_ref = asset.get("file") if isinstance(asset.get("file"), dict) else asset
    name = os.path.basename(str(file_ref.get("name") or "").strip())
    subfolder = os.path.normpath(str(file_ref.get("subfolder") or "").strip())
    if not name or subfolder.startswith("..") or os.path.isabs(subfolder):
        raise ValueError("导演台素材文件引用无效")
    root = os.path.abspath(folder_paths.get_input_directory())
    path = os.path.abspath(os.path.join(root, subfolder, name))
    if os.path.commonpath((root, path)) != root or not os.path.isfile(path):
        raise ValueError("导演台素材不存在或不在 ComfyUI/input 内：%s" % name)
    return path, name


def _empty_video():
    return torch.zeros((0, 1, 1, 3), dtype=torch.float32)


def _empty_audio():
    return {"waveform": torch.zeros((1, 1, 0), dtype=torch.float32),
            "sample_rate": 32000}


class H3ShotMedia:
    """Load the lightweight file references stored in one Director shot card."""

    CATEGORY = "沐阳 H3/导演台"
    FUNCTION = "resolve"
    RETURN_TYPES = ("MINIMAX_H3_MEDIA", "IMAGE", "AUDIO")
    RETURN_NAMES = ("本镜头素材", "动作源视频", "动作源音频")
    DESCRIPTION = (
        "导演台内部节点：运行到对应镜头时才加载该镜头的图片/视频/音频，"
        "避免所有镜头素材同时解码并占用内存。")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "assets_json": ("STRING", {"multiline": True, "default": "[]"}),
                "required_frames": ("INT", {"default": 125, "min": 5, "max": 10000}),
                "asset_mode": (["仅本镜头", "叠加全局素材"], {"default": "仅本镜头"}),
            },
            "optional": {"media": ("MINIMAX_H3_MEDIA",)},
        }

    def resolve(self, assets_json, required_frames, asset_mode="仅本镜头", media=None):
        try:
            assets = json.loads(str(assets_json or "[]"))
        except json.JSONDecodeError as error:
            raise ValueError("导演台镜头素材数据不是有效 JSON：%s" % error) from error
        if not isinstance(assets, list):
            raise ValueError("导演台镜头素材必须是列表")

        normalized = []
        counts = {kind: 0 for kind in SHOT_MEDIA_LIMITS}
        for raw in assets:
            if not isinstance(raw, dict):
                continue
            kind = str(raw.get("kind") or raw.get("media_type") or "").lower()
            if kind == "picture":
                kind = "image"
            if kind not in counts:
                continue
            counts[kind] += 1
            if counts[kind] > SHOT_MEDIA_LIMITS[kind]:
                raise ValueError("单个镜头最多可添加图片9个、视频3个、音频3个")
            normalized.append((kind, raw))

        keep_global = str(asset_mode) == "叠加全局素材"
        assets_out = list(media.assets) if keep_global and isinstance(media, MyangMediaCatalog) else []
        next_index = max([asset.slot for asset in assets_out] + [0]) + 1
        action_frames = None
        action_audio = None

        for kind, asset in normalized:
            path, name = _safe_input_path(asset)
            if kind == "image":
                value = InputImpl.VideoFromFile(path).get_components().images
                if value.shape[0] < 1:
                    raise ValueError("无法读取镜头图片：%s" % name)
                value = value[:1]
            elif kind == "video":
                value = InputImpl.VideoFromFile(path)
                if str(asset.get("role") or "reference") == "action":
                    if action_frames is not None:
                        raise ValueError("每个镜头只能指定一个动作源视频")
                    frames, soundtrack, source_fps = video_stream(value)
                    frames = core._to_24fps(frames, source_fps)
                    needed = int(required_frames)
                    if int(frames.shape[0]) < needed:
                        raise ValueError(
                            "镜头动作源『%s』只有 %d 帧，当前镜头需要 %d 帧（24fps）" %
                            (name, int(frames.shape[0]), needed))
                    action_frames = frames[:needed]
                    action_audio = soundtrack
                    value = {"images": action_frames, "fps": 24.0}
                    if soundtrack is not None:
                        value["audio"] = soundtrack
            else:
                from comfy_extras.nodes_audio import load as load_audio
                waveform, sample_rate = load_audio(path)
                value = {"waveform": waveform.unsqueeze(0),
                         "sample_rate": int(sample_rate)}

            assets_out.append(MyangMediaAsset(
                slot=next_index,
                kind=kind,
                payload=value,
                filename=name,
                label=str(asset.get("label") or name),
                origin="director_shot",
            ))
            next_index += 1

        bundle = MyangMediaCatalog(tuple(assets_out))
        return (bundle, action_frames if action_frames is not None else _empty_video(),
                action_audio if action_audio is not None else _empty_audio())

class H3MediaSwapClip:
    CATEGORY = "沐阳 H3"
    FUNCTION = "swap"
    RETURN_TYPES = ("MINIMAX_H3_MEDIA",)
    RETURN_NAMES = ("media",)
    DESCRIPTION = ("把 Agent 素材包里的第 N 段参考视频换成本段的切片，"
                   "其余素材和编号原样保留，所以提示词里的 @视频1 还是指同一个位置。")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "media": ("MINIMAX_H3_MEDIA",),
                "clip": ("IMAGE", {"tooltip": "本段的参考视频切片"}),
                "video_ordinal": ("INT", {"default": 1, "min": 1, "max": 3,
                                          "tooltip": "换掉第几个视频素材（@视频N 的 N）"}),
            },
        }

    def swap(self, media, clip, video_ordinal=1):
        if not isinstance(media, MyangMediaCatalog):
            raise ValueError(
                "media 不是 Agent 输出的素材包。把 MiniMaxH3MediaAgent 的 media 输出接过来。")

        updated, replaced = media.replacing_video(int(video_ordinal), clip)
        if replaced:
            return (updated,)

        # No video in the bundle: the operator wired only stills to the Agent.
        # Append the clip so the loop still works; it lands last, so it is the
        # highest-numbered media and the prompt's @视频1 refers to it.
        ordinal = sum(asset.kind == "video" for asset in media.assets) + 1
        appended = media.appended(MyangMediaAsset(
            slot=media.next_slot(), kind="video", payload=clip, origin="segment_swap"))
        logger.info("H3-Myang: 素材包里没有目标视频，参考视频切片作为 @视频%d 追加", ordinal)
        return (appended,)


class H3ReferenceClip:
    CATEGORY = "沐阳 H3/导演台/内部"
    FUNCTION = "slice"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("分段动作参考",)
    DESCRIPTION = "导演台内部节点：按时间切单个动作参考视频，尾段不足 H3 帧网格时重复末帧补齐。"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "start_frame": ("INT", {"default": 0, "min": 0, "max": 1000000}),
            "frame_count": ("INT", {"default": 125, "min": 1, "max": 10000}),
        }}

    def slice(self, image, start_frame, frame_count):
        available = int(image.shape[0])
        if available < 1:
            raise ValueError("动作参考视频没有可用帧")
        start = max(0, min(int(start_frame), available - 1))
        frame_count = int(frame_count)
        clip = image[start:min(available, start + frame_count)].clone()
        missing = frame_count - int(clip.shape[0])
        if missing > 0:
            clip = torch.cat((clip, image[-1:].repeat((missing, 1, 1, 1))), dim=0)
        return (clip,)


class H3ReferenceAudioClip:
    CATEGORY = "沐阳 H3/导演台/内部"
    FUNCTION = "slice"
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("分段动作音频",)
    DESCRIPTION = "导演台内部节点：按动作参考视频的帧区间同步切分音频。"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "audio": ("AUDIO",),
            "start_frame": ("INT", {"default": 0, "min": 0, "max": 1000000}),
            "frame_count": ("INT", {"default": 125, "min": 1, "max": 10000}),
            "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0}),
        }}

    def slice(self, audio, start_frame, frame_count, fps=24.0):
        waveform = audio.get("waveform")
        sample_rate = int(audio.get("sample_rate", 0))
        if waveform is None or sample_rate <= 0:
            raise ValueError("动作参考音频缺少 waveform 或 sample_rate")
        start = round(int(start_frame) * sample_rate / float(fps))
        length = max(1, round(int(frame_count) * sample_rate / float(fps)))
        chunk = waveform[..., start:start + length].clone()
        missing = length - int(chunk.shape[-1])
        if missing > 0:
            chunk = torch.nn.functional.pad(chunk, (0, missing))
        return ({**audio, "waveform": chunk, "sample_rate": sample_rate},)


def reference_video_size(image_width, image_height, resolution, width=1920, height=1080):
    """Return an aspect-preserving reference canvas capped to a 1080P frame.

    Presets preserve the source aspect ratio.  The cap is orientation-aware:
    landscape fits inside 1920x1080, portrait inside 1080x1920.  This prevents
    an ultra-wide/custom typo from silently creating a multi-gigabyte tensor.
    """
    source_w, source_h = int(image_width), int(image_height)
    if source_w < 1 or source_h < 1:
        raise ValueError("参考视频分辨率无效")
    if str(resolution) == REFERENCE_VIDEO_ORIGINAL:
        return source_w, source_h
    if str(resolution) == "自定义":
        target_w, target_h = int(width), int(height)
    else:
        try:
            short = int(str(resolution).rstrip("P"))
        except ValueError as error:
            raise ValueError("未知参考视频分辨率：%s" % resolution) from error
        scale = short / float(min(source_w, source_h))
        target_w = round(source_w * scale)
        target_h = round(source_h * scale)

    target_w = max(32, target_w)
    target_h = max(32, target_h)
    max_w, max_h = ((1920, 1080) if target_w >= target_h else (1080, 1920))
    scale = min(1.0, max_w / float(target_w), max_h / float(target_h))
    target_w = max(32, round(target_w * scale))
    target_h = max(32, round(target_h * scale))
    return min(max_w, target_w), min(max_h, target_h)


class H3ReferenceResize:
    CATEGORY = "沐阳 H3/导演台/内部"
    FUNCTION = "resize"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("参考视频",)
    DESCRIPTION = "导演台内部节点：保持原比例分块缩放参考视频，最高限制到 1080P 画布。"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "resolution": (REFERENCE_VIDEO_RESOLUTIONS, {
                "default": REFERENCE_VIDEO_ORIGINAL}),
            "width": ("INT", {"default": 1920, "min": 32, "max": 1920, "step": 32}),
            "height": ("INT", {"default": 1080, "min": 32, "max": 1920, "step": 32}),
        }}

    def resize(self, image, resolution=REFERENCE_VIDEO_ORIGINAL,
               width=1920, height=1080):
        if image is None or not hasattr(image, "shape") or len(image.shape) != 4:
            raise ValueError("参考视频必须是 ComfyUI IMAGE 帧批次")
        source_h, source_w = int(image.shape[1]), int(image.shape[2])
        target_w, target_h = reference_video_size(
            source_w, source_h, resolution, width, height)
        if (target_w, target_h) == (source_w, source_h):
            return (image,)
        import comfy.utils
        chunks = []
        # 长参考视频分块缩放，避免一次把所有帧的 NCHW 临时张量压进显存。
        for frames in image.split(16, dim=0):
            nchw = frames[..., :3].movedim(-1, 1)
            resized = comfy.utils.common_upscale(
                nchw, target_w, target_h, "lanczos", "disabled")
            chunks.append(resized.movedim(1, -1))
        return (torch.cat(chunks, dim=0),)


NODE_CLASS_MAPPINGS = {
    "H3MediaSwapClip": H3MediaSwapClip,
    "H3ShotMedia": H3ShotMedia,
    "H3ReferenceClip": H3ReferenceClip,
    "H3ReferenceAudioClip": H3ReferenceAudioClip,
    "H3ReferenceResize": H3ReferenceResize,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3MediaSwapClip": "沐阳 H3 · 分段素材替换",
    "H3ShotMedia": "沐阳 H3 · 镜头素材（内部）",
    "H3ReferenceClip": "沐阳 H3 · 动作视频自动分段（内部）",
    "H3ReferenceAudioClip": "沐阳 H3 · 动作音频自动分段（内部）",
    "H3ReferenceResize": "沐阳 H3 · 参考视频分辨率（内部）",
}
