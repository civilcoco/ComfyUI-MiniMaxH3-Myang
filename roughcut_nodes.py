"""Execution nodes connecting a rough-cut selection to MiniMax H3."""

from __future__ import annotations

import json
import os
from fractions import Fraction

import folder_paths
import torch
from comfy_api.latest import InputImpl, Types

from . import core
from .roughcut import (
    boundary_clip, boundary_window, normalize_project, overwrite_selection,
    selection_contract,
)
from .roughcut_library import resolve_asset, resolve_input, resolve_output


def _empty_image():
    return torch.zeros((0, 1, 1, 3), dtype=torch.float32)


def _empty_audio():
    return {"waveform": torch.zeros((1, 1, 0), dtype=torch.float32),
            "sample_rate": 32000}


def _crop_context_audio(audio, frame_count, fps, side):
    """Keep exactly the sound window adjacent to I/O.

    Decoders may return codec padding.  The I-side must end exactly at I while
    the O-side must begin exactly at O; otherwise the visual MotionContext and
    its audio anchor drift in opposite directions.
    """
    if not isinstance(audio, dict) or audio.get("waveform") is None:
        return _empty_audio()
    waveform = audio["waveform"]
    sample_rate = int(audio.get("sample_rate") or 32000)
    wanted = max(1, round(int(frame_count) / float(fps) * sample_rate))
    available = int(waveform.shape[-1])
    if available > wanted:
        waveform = (waveform[..., available - wanted:]
                    if side == "first" else waveform[..., :wanted])
    elif available < wanted:
        # Preserve boundary alignment: pad away from the cut, never between
        # the available waveform and I/O.
        missing = wanted - available
        waveform = (torch.nn.functional.pad(waveform, (missing, 0))
                    if side == "first"
                    else torch.nn.functional.pad(waveform, (0, missing)))
    return {"waveform": waveform, "sample_rate": sample_rate}


def _clip_path(clip):
    source = clip["source"]
    if source["type"] == "library":
        return resolve_asset(source["library_id"], source["relative_path"])
    if source["type"] == "input":
        return resolve_input(source["filename"], source.get("subfolder", ""))
    return resolve_output(source["filename"], source.get("subfolder", ""))


def _boundary_image(project, side):
    resolved = boundary_clip(project, side)
    if resolved is None:
        label = "入点" if side == "first" else "出点"
        raise ValueError(f"粗剪时间轴的{label}没有可读取的视频或图片素材")
    clip, source_frame = resolved
    if clip["kind"] == "audio":
        raise ValueError("首尾帧约束不能从音频片段提取画面")
    path = _clip_path(clip)
    if clip["kind"] == "image":
        images = InputImpl.VideoFromFile(str(path)).get_components().images
    else:
        source_fps = max(1.0, float(clip.get("source_fps") or 24.0))
        images = InputImpl.VideoFromFile(
            str(path), start_time=float(source_frame) / source_fps,
            duration=max(1.0 / source_fps, 0.001)).get_components().images
    if images is None or int(images.shape[0]) < 1:
        raise ValueError("粗剪首尾约束帧读取失败")
    return images[:1]


def _boundary_context(project, side, frame_count):
    resolved = boundary_window(project, side, frame_count)
    if resolved is None:
        label = "入点之前" if side == "first" else "出点之后"
        raise ValueError(
            f"粗剪时间轴{label}没有连续 {int(frame_count)} 帧的视频，"
            "请缩短 MotionContext 窗口、移动 I/O，或改用单帧约束")
    clip, source_start, source_count = resolved
    if clip["kind"] == "image":
        raise ValueError("静态图片只能用于单帧约束，不能提供 MotionContext 运动窗口")
    source_fps = max(1.0, float(clip.get("source_fps") or 24.0))
    video = InputImpl.VideoFromFile(
        str(_clip_path(clip)), start_time=source_start / source_fps,
        duration=source_count / source_fps)
    frames, audio, decoded_fps = core._video_parts(video)
    frames = core._to_24fps(frames, decoded_fps)
    wanted = int(frame_count)
    if int(frames.shape[0]) < wanted:
        raise ValueError(
            f"粗剪 MotionContext 解码后只有 {int(frames.shape[0])} 帧，需要 {wanted} 帧")
    frames = frames[-wanted:] if side == "first" else frames[:wanted]

    project_fps = float(project["settings"]["fps"])
    soundtrack = _crop_context_audio(audio, wanted, project_fps, side)
    # A separately edited A1 clip is authoritative over the video's embedded
    # track.  This makes audio files dragged onto the timeline useful for both
    # head and tail bridging while keeping embedded sound as the fallback.
    audio_window = boundary_window(
        project, side, frame_count, track_kind="audio")
    if audio_window is not None:
        audio_clip, audio_start, audio_count = audio_window
        audio_fps = max(1.0, float(audio_clip.get("source_fps") or project_fps))
        source = InputImpl.VideoFromFile(
            str(_clip_path(audio_clip)), start_time=audio_start / audio_fps,
            duration=audio_count / audio_fps)
        components = source.get_components()
        decoded_audio = getattr(components, "audio", None)
        if decoded_audio is not None:
            soundtrack = _crop_context_audio(
                core._audio(decoded_audio), wanted, project_fps, side)
    return frames, soundtrack


class H3RoughCutBoundaryFrames:
    CATEGORY = "沐阳 H3/导演台/内部"
    FUNCTION = "load"
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "AUDIO", "IMAGE", "AUDIO")
    RETURN_NAMES = ("首帧约束", "尾帧约束", "入点MotionContext", "入点声音",
                    "出点MotionContext", "出点声音")
    DESCRIPTION = "从粗剪时间轴 I/O 边界提取单帧约束或前后 MotionContext 窗口。"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "project_json": ("STRING", {"forceInput": True}),
            "load_first": ("BOOLEAN", {"default": False}),
            "load_last": ("BOOLEAN", {"default": False}),
            "load_head_motion": ("BOOLEAN", {"default": False}),
            "load_tail_motion": ("BOOLEAN", {"default": False}),
            "context_frames": ("INT", {"default": 22, "min": 5, "max": 56}),
        }}

    def load(self, project_json, load_first=False, load_last=False,
             load_head_motion=False, load_tail_motion=False, context_frames=22):
        project = normalize_project(project_json)
        first = _boundary_image(project, "first") if load_first else _empty_image()
        last = _boundary_image(project, "last") if load_last else _empty_image()
        head, head_audio = (_boundary_context(project, "first", context_frames)
                            if load_head_motion else (_empty_image(), _empty_audio()))
        tail, tail_audio = (_boundary_context(project, "last", context_frames)
                            if load_tail_motion else (_empty_image(), _empty_audio()))
        return first, last, head, head_audio, tail, tail_audio


def _fit_audio(audio, target_samples):
    if not isinstance(audio, dict) or audio.get("waveform") is None:
        return audio
    waveform = audio["waveform"]
    current = int(waveform.shape[-1])
    if current > target_samples:
        waveform = waveform[..., :target_samples].clone()
    elif current < target_samples:
        padding = torch.zeros(
            (*waveform.shape[:-1], target_samples - current),
            device=waveform.device, dtype=waveform.dtype)
        waveform = torch.cat((waveform, padding), dim=-1)
    return {**audio, "waveform": waveform}


class H3RoughCutSave:
    CATEGORY = "沐阳 H3/导演台/内部"
    FUNCTION = "save"
    RETURN_TYPES = ("IMAGE", "AUDIO", "STRING")
    RETURN_NAMES = ("images", "audio", "project_json")
    OUTPUT_NODE = True
    DESCRIPTION = "把导演台生成结果精确裁到粗剪 I/O 区间并覆盖写回时间轴。"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "audio": ("AUDIO",),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0}),
                "project_json": ("STRING", {"forceInput": True}),
                "filename_prefix": ("STRING", {"default": "video/H3_导演台_粗剪"}),
                "owner_id": ("STRING", {"default": ""}),
            },
            "optional": {
                # Expanded graphs created before this guard omit the value and
                # therefore fail closed.  Only the current Director explicitly
                # authorises a write-back with True.
                "writeback_enabled": ("BOOLEAN", {"default": False}),
            },
        }

    def save(self, images, audio, fps, project_json, filename_prefix, owner_id="",
             writeback_enabled=False):
        if writeback_enabled is not True:
            return {"ui": {}, "result": (images, audio, str(project_json or ""))}
        project = normalize_project(project_json)
        contract = selection_contract(project)
        target_frames = max(1, round(float(contract["seconds"]) * float(fps)))
        available = int(images.shape[0])
        if available < 1:
            raise ValueError("导演台没有生成可写入粗剪时间轴的画面")
        if available >= target_frames:
            fitted_images = images[:target_frames].clone()
        else:
            padding = images[-1:].repeat(target_frames - available, 1, 1, 1)
            fitted_images = torch.cat((images, padding), dim=0)

        sample_rate = int(audio.get("sample_rate") or 32000) if isinstance(audio, dict) else 32000
        target_samples = max(1, round(target_frames / float(fps) * sample_rate))
        fitted_audio = _fit_audio(audio, target_samples)
        video = InputImpl.VideoFromComponents(
            Types.VideoComponents(
                images=fitted_images, audio=fitted_audio,
                frame_rate=Fraction(round(float(fps) * 1000), 1000)),
            bit_depth=8)
        width, height = video.get_dimensions()
        # Keep the editable timeline canvas in sync with the actual Director
        # result.  Without this, a portrait generation written into an empty
        # project would later be exported on the legacy 1920x1080 canvas.
        project["settings"]["width"] = int(width)
        project["settings"]["height"] = int(height)
        full_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            str(filename_prefix or "video/H3_导演台_粗剪"),
            folder_paths.get_output_directory(), width, height)
        saved_name = f"{filename}_{counter:05}_.mp4"
        video.save_to(
            os.path.join(full_folder, saved_name),
            format=Types.VideoContainer.MP4,
            codec=Types.VideoCodec.H264)
        updated = overwrite_selection(project, {
            "type": "output", "filename": saved_name, "subfolder": subfolder,
        }, source_fps=float(fps), source_frames=target_frames)
        serialized = json.dumps(updated, ensure_ascii=False, separators=(",", ":"))
        event = {
            "owner_id": str(owner_id or ""),
            "project_json": serialized,
            "video": {"filename": saved_name, "subfolder": subfolder, "type": "output"},
            "target_frames": target_frames,
            "fps": float(fps),
        }
        try:
            from server import PromptServer
            instance = getattr(PromptServer, "instance", None)
            if instance is not None and hasattr(instance, "send_sync"):
                instance.send_sync("myh3_roughcut_commit", event)
        # The file is already saved at this point.  A disconnected browser or
        # an unavailable PromptServer must never turn that successful render
        # into a failed ComfyUI execution.
        except Exception:
            pass
        return {
            "ui": {"videos": [event["video"]], "roughcut_project": [serialized]},
            "result": (fitted_images, fitted_audio, serialized),
        }


NODE_CLASS_MAPPINGS = {
    "H3RoughCutBoundaryFrames": H3RoughCutBoundaryFrames,
    "H3RoughCutSave": H3RoughCutSave,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3RoughCutBoundaryFrames": "沐阳 H3 · 粗剪首尾帧（内部）",
    "H3RoughCutSave": "沐阳 H3 · 粗剪覆盖写入（内部）",
}
