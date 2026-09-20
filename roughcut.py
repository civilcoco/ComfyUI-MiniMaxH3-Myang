"""Frame-accurate project model for the Director rough-cut workspace."""

from __future__ import annotations

import copy
import json
import uuid
from pathlib import PurePosixPath
from typing import Any


PROJECT_FORMAT = "myang.roughcut"
PROJECT_VERSION = 1
TRACK_KINDS = frozenset({"video", "audio"})
SOURCE_TYPES = frozenset({"library", "input", "output"})


def new_project(fps: float = 24.0, width: int = 1920, height: int = 1080) -> dict[str, Any]:
    return {
        "format": PROJECT_FORMAT,
        "version": PROJECT_VERSION,
        "id": f"roughcut_{uuid.uuid4().hex}",
        "revision": 0,
        "settings": {
            "fps": float(fps),
            "width": int(width),
            "height": int(height),
            "auto_align": True,
            "snap_enabled": True,
            # A project must explicitly opt into generation write-back.  This
            # prevents stale hidden ComfyUI widget values from trimming a new
            # Director run after the user has turned the timeline off.
            "writeback_enabled": False,
        },
        "selection": {
            "in_frame": 0,
            "out_frame": max(1, round(float(fps) * 5.0)),
            "in_set": False,
            "out_set": False,
            "anchor_side": "",
            "duration_mode": "director",
            "target_track": "video_1",
            "start_mode": "off",
            "end_mode": "off",
            "use_first_frame": False,
            "use_last_frame": False,
        },
        "tracks": [
            {"id": "video_1", "kind": "video", "name": "V1", "clips": []},
            {"id": "audio_1", "kind": "audio", "name": "A1", "clips": []},
        ],
    }


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} 必须是整数")
    try:
        number = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} 必须是整数") from error
    if number < minimum:
        raise ValueError(f"{name} 不能小于 {minimum}")
    return number


def _source(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("时间轴片段缺少素材引用")
    source_type = str(value.get("type") or "").strip().lower()
    if source_type not in SOURCE_TYPES:
        raise ValueError("时间轴素材来源只能是 library、input 或 output")
    normalized = {"type": source_type}
    if source_type == "library":
        library_id = str(value.get("library_id") or "").strip()
        relative_path = str(value.get("relative_path") or "").replace("\\", "/").strip()
        relative = PurePosixPath(relative_path)
        if (not library_id or not relative_path or relative.is_absolute()
                or ".." in relative.parts):
            raise ValueError("时间轴素材库引用无效")
        normalized.update({"library_id": library_id, "relative_path": relative_path})
    else:
        filename = str(value.get("filename") or "").strip()
        subfolder = str(value.get("subfolder") or "").replace("\\", "/").strip("/")
        if not filename or PurePosixPath(filename).name != filename:
            raise ValueError("时间轴文件引用无效")
        if subfolder and (PurePosixPath(subfolder).is_absolute()
                          or ".." in PurePosixPath(subfolder).parts):
            raise ValueError("时间轴文件目录引用无效")
        normalized.update({"filename": filename, "subfolder": subfolder})
    return normalized


def _clip(value: Any, track_kind: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("时间轴片段必须是对象")
    start = _integer(value.get("timeline_start"), "timeline_start")
    end = _integer(value.get("timeline_end"), "timeline_end", 1)
    if end <= start:
        raise ValueError("时间轴片段结束帧必须大于开始帧")
    source_in = _integer(value.get("source_in", 0), "source_in")
    source_out = _integer(
        value.get("source_out", source_in + end - start), "source_out", 1)
    if source_out <= source_in:
        raise ValueError("素材出点必须大于入点")
    kind = str(value.get("kind") or track_kind).strip().lower()
    if kind not in {"video", "image", "audio", "generated"}:
        raise ValueError("时间轴片段类型无效")
    if track_kind == "audio" and kind != "audio":
        raise ValueError("音频轨只能放置音频片段")
    if track_kind == "video" and kind == "audio":
        raise ValueError("视频轨不能放置音频片段")
    return {
        "id": str(value.get("id") or f"clip_{uuid.uuid4().hex}"),
        "kind": kind,
        "name": str(value.get("name") or "未命名片段"),
        "timeline_start": start,
        "timeline_end": end,
        "source_in": source_in,
        "source_out": source_out,
        "source_fps": float(value.get("source_fps") or 24.0),
        "source": _source(value.get("source")),
    }


def normalize_project(value: str | dict[str, Any] | None) -> dict[str, Any]:
    if value is None or str(value).strip() == "":
        return new_project()
    if isinstance(value, str):
        try:
            raw = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError(f"粗剪工程不是有效 JSON：{error}") from error
    elif isinstance(value, dict):
        raw = copy.deepcopy(value)
    else:
        raise ValueError("粗剪工程必须是 JSON 对象")
    if str(raw.get("format") or "") != PROJECT_FORMAT:
        raise ValueError("不是沐阳粗剪时间轴工程")
    if _integer(raw.get("version"), "version", 1) != PROJECT_VERSION:
        raise ValueError("粗剪工程版本不受支持")

    settings = raw.get("settings") if isinstance(raw.get("settings"), dict) else {}
    fps = float(settings.get("fps") or 24.0)
    if not 1.0 <= fps <= 240.0:
        raise ValueError("粗剪工程 fps 必须在 1～240 之间")
    width = _integer(settings.get("width", 1920), "width", 32)
    height = _integer(settings.get("height", 1080), "height", 32)
    auto_align = bool(settings.get("auto_align", True))
    snap_enabled = bool(settings.get("snap_enabled", True))
    # Only a real JSON boolean enables write-back.  Missing legacy values and
    # strings such as "true" intentionally fail closed.
    writeback_enabled = settings.get("writeback_enabled") is True

    tracks = []
    track_ids = set()
    for item in raw.get("tracks") or []:
        if not isinstance(item, dict):
            continue
        track_id = str(item.get("id") or "").strip()
        kind = str(item.get("kind") or "").strip().lower()
        if not track_id or track_id in track_ids or kind not in TRACK_KINDS:
            raise ValueError("粗剪轨道 ID 重复或类型无效")
        track_ids.add(track_id)
        clips = [_clip(clip, kind) for clip in (item.get("clips") or [])]
        clips.sort(key=lambda clip: (clip["timeline_start"], clip["timeline_end"], clip["id"]))
        tracks.append({
            "id": track_id,
            "kind": kind,
            "name": str(item.get("name") or track_id),
            "clips": clips,
        })
    if not tracks:
        tracks = new_project(fps, width, height)["tracks"]
        track_ids = {track["id"] for track in tracks}

    selection = raw.get("selection") if isinstance(raw.get("selection"), dict) else {}
    in_frame = _integer(selection.get("in_frame", 0), "in_frame")
    out_frame = _integer(selection.get("out_frame", in_frame + round(fps * 5)), "out_frame", 1)
    if out_frame <= in_frame:
        raise ValueError("粗剪出点必须晚于入点")
    duration_mode = str(selection.get("duration_mode") or "director")
    if duration_mode not in {"director", "timeline"}:
        duration_mode = "director"
    explicit_point_state = (
        "in_set" in selection or "out_set" in selection or
        "anchor_side" in selection or "duration_mode" in selection)
    has_clips = any(track["clips"] for track in tracks)
    untouched_legacy_default = (
        not explicit_point_state and
        _integer(raw.get("revision", 0), "revision") == 0 and
        not has_clips and in_frame == 0 and
        out_frame == max(1, round(fps * 5.0)))
    in_set = bool(selection.get(
        "in_set", False if untouched_legacy_default else True))
    out_set = bool(selection.get(
        "out_set", False if untouched_legacy_default else True))
    anchor_side = str(selection.get("anchor_side") or "")
    if anchor_side not in {"", "in", "out"}:
        anchor_side = ""
    if duration_mode == "director" and anchor_side == "":
        if in_set and not out_set:
            anchor_side = "in"
        elif out_set and not in_set:
            anchor_side = "out"
    target_track = str(selection.get("target_track") or "video_1")
    if target_track not in track_ids:
        target_track = next((track["id"] for track in tracks if track["kind"] == "video"), tracks[0]["id"])

    start_mode = str(selection.get("start_mode") or
                     ("frame" if selection.get("use_first_frame") else "off"))
    end_mode = str(selection.get("end_mode") or
                   ("frame" if selection.get("use_last_frame") else "off"))
    if start_mode not in {"off", "frame", "motion"}:
        start_mode = "off"
    if end_mode not in {"off", "frame", "motion"}:
        end_mode = "off"

    return {
        "format": PROJECT_FORMAT,
        "version": PROJECT_VERSION,
        "id": str(raw.get("id") or f"roughcut_{uuid.uuid4().hex}"),
        "revision": _integer(raw.get("revision", 0), "revision"),
        "settings": {
            "fps": fps, "width": width, "height": height,
            "auto_align": auto_align, "snap_enabled": snap_enabled,
            "writeback_enabled": writeback_enabled,
        },
        "selection": {
            "in_frame": in_frame,
            "out_frame": out_frame,
            "in_set": in_set,
            "out_set": out_set,
            "anchor_side": anchor_side,
            "duration_mode": duration_mode,
            "target_track": target_track,
            "start_mode": start_mode,
            "end_mode": end_mode,
            "use_first_frame": start_mode == "frame",
            "use_last_frame": end_mode == "frame",
        },
        "tracks": tracks,
    }


def project_json(value: str | dict[str, Any] | None) -> str:
    return json.dumps(normalize_project(value), ensure_ascii=False, separators=(",", ":"))


def selection_contract(value: str | dict[str, Any]) -> dict[str, Any]:
    project = normalize_project(value)
    selection = project["selection"]
    frames = int(selection["out_frame"]) - int(selection["in_frame"])
    duration_mode = str(selection.get("duration_mode") or "director")
    in_set = bool(selection.get("in_set", False))
    out_set = bool(selection.get("out_set", False))
    ready = ((in_set or out_set) if duration_mode == "director"
             else (in_set and out_set))
    return {
        **selection,
        "frames": frames,
        "fps": float(project["settings"]["fps"]),
        "seconds": frames / float(project["settings"]["fps"]),
        "ready": ready,
    }


def _split_clip(clip: dict[str, Any], start: int, end: int, project_fps: float) -> list[dict[str, Any]]:
    clip_start = int(clip["timeline_start"])
    clip_end = int(clip["timeline_end"])
    if clip_end <= start or clip_start >= end:
        return [clip]
    source_rate = float(clip.get("source_fps") or project_fps) / project_fps
    pieces = []
    if clip_start < start:
        left = copy.deepcopy(clip)
        left["timeline_end"] = start
        left["source_out"] = min(
            int(clip["source_out"]),
            int(clip["source_in"]) + round((start - clip_start) * source_rate))
        pieces.append(left)
    if clip_end > end:
        right = copy.deepcopy(clip)
        right["id"] = f"{clip['id']}_r_{uuid.uuid4().hex[:8]}"
        right["timeline_start"] = end
        right["source_in"] = max(
            int(clip["source_in"]),
            int(clip["source_out"]) - round((clip_end - end) * source_rate))
        pieces.append(right)
    return pieces


def overwrite_selection(value: str | dict[str, Any], generated_source: dict[str, Any],
                        name: str = "导演台生成片段", source_fps: float | None = None,
                        source_frames: int | None = None) -> dict[str, Any]:
    project = normalize_project(value)
    selection = project["selection"]
    start, end = int(selection["in_frame"]), int(selection["out_frame"])
    target = str(selection["target_track"])
    duration = end - start
    target_track = next((track for track in project["tracks"] if track["id"] == target), None)
    if target_track is None or target_track["kind"] != "video":
        raise ValueError("粗剪生成结果只能写入视频轨")
    kept = []
    for clip in target_track["clips"]:
        kept.extend(_split_clip(clip, start, end, float(project["settings"]["fps"])))
    kept.append({
        "id": f"generated_{uuid.uuid4().hex}",
        "kind": "generated",
        "name": str(name or "导演台生成片段"),
        "timeline_start": start,
        "timeline_end": end,
        "source_in": 0,
        "source_out": int(source_frames if source_frames is not None else duration),
        "source_fps": float(source_fps or project["settings"]["fps"]),
        "source": _source(generated_source),
    })
    target_track["clips"] = sorted(
        kept, key=lambda clip: (clip["timeline_start"], clip["timeline_end"], clip["id"]))
    project["revision"] += 1
    return project


def boundary_clip(value: str | dict[str, Any], side: str) -> tuple[dict[str, Any], int] | None:
    project = normalize_project(value)
    selection = project["selection"]
    target = str(selection["target_track"])
    track = next((item for item in project["tracks"] if item["id"] == target), None)
    if track is None:
        return None
    point = int(selection["in_frame"] if side == "first" else selection["out_frame"])
    clips = list(track["clips"])
    if side == "first":
        candidates = [clip for clip in clips if clip["timeline_start"] <= point < clip["timeline_end"]]
        if not candidates:
            candidates = [clip for clip in clips if clip["timeline_end"] == point]
        if not candidates:
            return None
        clip = max(candidates, key=lambda item: (item["timeline_start"], item["timeline_end"]))
        timeline_frame = point if point < clip["timeline_end"] else point - 1
    else:
        candidates = [clip for clip in clips if clip["timeline_start"] <= point < clip["timeline_end"]]
        if not candidates:
            candidates = [clip for clip in clips if clip["timeline_start"] == point]
        if not candidates:
            return None
        clip = min(candidates, key=lambda item: (item["timeline_start"], item["timeline_end"]))
        timeline_frame = point
    project_fps = float(project["settings"]["fps"])
    source_fps = float(clip.get("source_fps") or project_fps)
    offset = max(0, timeline_frame - int(clip["timeline_start"]))
    source_frame = int(clip["source_in"]) + round(offset * source_fps / project_fps)
    source_frame = min(source_frame, max(int(clip["source_in"]), int(clip["source_out"]) - 1))
    return clip, source_frame


def boundary_window(value: str | dict[str, Any], side: str,
                    frame_count: int, track_kind: str = "video"
                    ) -> tuple[dict[str, Any], int, int] | None:
    project = normalize_project(value)
    selection = project["selection"]
    wanted_kind = str(track_kind or "video").strip().lower()
    if wanted_kind not in TRACK_KINDS:
        raise ValueError("粗剪边界窗口轨道只能是 video 或 audio")
    if wanted_kind == "video":
        target = str(selection["target_track"])
        track = next((item for item in project["tracks"]
                      if item["id"] == target and item["kind"] == "video"), None)
    else:
        # Phase one exposes one A1 track.  Choosing by kind instead of a fixed
        # id keeps imported projects and future renamed tracks compatible.
        track = next((item for item in project["tracks"]
                      if item["kind"] == "audio"), None)
    if track is None:
        return None
    count = _integer(frame_count, "context_frames", 1)
    point = int(selection["in_frame"] if side == "first" else selection["out_frame"])
    if side == "first":
        candidates = [clip for clip in track["clips"]
                      if clip["timeline_start"] < point <= clip["timeline_end"]]
        window_start, window_end = point - count, point
    else:
        candidates = [clip for clip in track["clips"]
                      if clip["timeline_start"] <= point < clip["timeline_end"]]
        window_start, window_end = point, point + count
    candidates = [clip for clip in candidates
                  if ((wanted_kind == "audio" and clip["kind"] == "audio")
                      or (wanted_kind == "video" and clip["kind"] != "audio"))
                  and int(clip["timeline_start"]) <= window_start
                  and int(clip["timeline_end"]) >= window_end]
    if not candidates:
        return None
    clip = max(candidates, key=lambda item: (item["timeline_start"], item["timeline_end"]))
    project_fps = float(project["settings"]["fps"])
    source_fps = float(clip.get("source_fps") or project_fps)
    source_count = max(1, round(count * source_fps / project_fps))
    source_start = int(clip["source_in"]) + round(
        (window_start - int(clip["timeline_start"])) * source_fps / project_fps)
    if source_start < int(clip["source_in"]) or source_start + source_count > int(clip["source_out"]):
        return None
    return clip, source_start, source_count
