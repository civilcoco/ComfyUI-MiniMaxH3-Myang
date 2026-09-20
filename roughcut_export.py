"""Local, streaming MP4 export for the Director rough-cut timeline."""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path, PurePosixPath
from typing import Any, Iterator

import av
import folder_paths
import numpy as np

from .roughcut import normalize_project
from .roughcut_library import resolve_asset, resolve_input, resolve_output


EXPORT_ROUTE = "/minimax-h3-myang/roughcut/export"
MAX_EXPORT_SECONDS = 6 * 60 * 60
MAX_EXPORT_EDGE = 8192
AUDIO_RATE = 48000
AUDIO_CHUNK = 4096
_ROUTES_REGISTERED = False
_EXPORT_LOCK = asyncio.Lock()


@dataclass(frozen=True)
class TimelineSpan:
    start: int
    end: int
    clip: dict[str, Any] | None


def timeline_frames(project: dict[str, Any]) -> int:
    return max((int(clip["timeline_end"])
                for track in project["tracks"]
                for clip in track["clips"]), default=0)


def track_spans(track: dict[str, Any] | None, total_frames: int) -> list[TimelineSpan]:
    clips = list(track["clips"]) if track else []
    boundaries = {0, int(total_frames)}
    for clip in clips:
        boundaries.add(max(0, min(int(total_frames), int(clip["timeline_start"]))))
        boundaries.add(max(0, min(int(total_frames), int(clip["timeline_end"]))))
    points = sorted(boundaries)
    spans: list[TimelineSpan] = []
    for start, end in zip(points, points[1:]):
        if end <= start:
            continue
        active = [clip for clip in clips
                  if int(clip["timeline_start"]) <= start
                  and int(clip["timeline_end"]) >= end]
        clip = active[-1] if active else None
        if spans and spans[-1].clip is clip and spans[-1].end == start:
            previous = spans[-1]
            spans[-1] = TimelineSpan(previous.start, end, clip)
        else:
            spans.append(TimelineSpan(start, end, clip))
    return spans


def _clip_path(clip: dict[str, Any]) -> Path:
    source = clip["source"]
    if source["type"] == "library":
        return resolve_asset(source["library_id"], source["relative_path"])
    if source["type"] == "input":
        return resolve_input(source["filename"], source.get("subfolder", ""))
    return resolve_output(source["filename"], source.get("subfolder", ""))


def _validate_prefix(filename_prefix: str) -> str:
    value = str(filename_prefix or "video/H3_粗剪导出").strip().replace("\\", "/")
    relative = PurePosixPath(value)
    if (not value or relative.is_absolute() or ".." in relative.parts
            or any(part in {"", "."} for part in relative.parts)):
        raise ValueError("粗剪导出文件名必须位于 ComfyUI/output 内")
    if relative.suffix.lower() == ".mp4":
        relative = relative.with_suffix("")
    if relative.suffix:
        raise ValueError("粗剪导出文件名不要填写其他扩展名")
    return relative.as_posix()


def _output_path(filename_prefix: str, width: int, height: int) -> tuple[Path, dict[str, str]]:
    prefix = _validate_prefix(filename_prefix)
    root = Path(folder_paths.get_output_directory()).resolve()
    full_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
        prefix, str(root), width, height)
    folder = Path(full_folder).resolve()
    try:
        folder.relative_to(root)
    except ValueError as error:
        raise ValueError("粗剪导出路径越出了 ComfyUI/output") from error
    folder.mkdir(parents=True, exist_ok=True)
    saved_name = f"{filename}_{counter:05}_.mp4"
    target = (folder / saved_name).resolve()
    try:
        target.relative_to(root)
    except ValueError as error:
        raise ValueError("粗剪导出路径越出了 ComfyUI/output") from error
    return target, {"filename": saved_name, "subfolder": subfolder, "type": "output"}


def _video_time(frame: av.VideoFrame, index: int, fallback_rate: float) -> float:
    if frame.pts is not None and frame.time_base is not None:
        return float(frame.pts * frame.time_base)
    return index / fallback_rate


def _fit_frame(frame: av.VideoFrame, width: int, height: int) -> np.ndarray:
    source_width, source_height = int(frame.width), int(frame.height)
    if source_width < 1 or source_height < 1:
        raise ValueError("粗剪素材包含无效画面尺寸")
    scale = min(width / source_width, height / source_height)
    fitted_width = max(2, min(width, round(source_width * scale)))
    fitted_height = max(2, min(height, round(source_height * scale)))
    fitted_width -= fitted_width % 2
    fitted_height -= fitted_height % 2
    resized = frame.reformat(width=fitted_width, height=fitted_height, format="rgb24")
    image = np.zeros((height, width, 3), dtype=np.uint8)
    top = (height - fitted_height) // 2
    left = (width - fitted_width) // 2
    image[top:top + fitted_height, left:left + fitted_width] = resized.to_ndarray()
    return image


def _image_frame(path: Path, width: int, height: int) -> np.ndarray:
    with av.open(str(path), mode="r") as container:
        stream = next((item for item in container.streams if item.type == "video"), None)
        if stream is None:
            raise ValueError(f"图片素材无法解码：{path.name}")
        frame = next(container.decode(stream), None)
        if frame is None:
            raise ValueError(f"图片素材没有画面：{path.name}")
        return _fit_frame(frame, width, height)


def _video_span_frames(clip: dict[str, Any], span: TimelineSpan, project_fps: float,
                       width: int, height: int) -> Iterator[np.ndarray]:
    path = _clip_path(clip)
    source_fps = max(0.001, float(clip.get("source_fps") or project_fps))
    first_source = int(clip["source_in"]) + round(
        (span.start - int(clip["timeline_start"])) * source_fps / project_fps)
    source_limit = max(first_source + 1, int(clip["source_out"]))
    with av.open(str(path), mode="r") as container:
        stream = next((item for item in container.streams if item.type == "video"), None)
        if stream is None:
            raise ValueError(f"视频素材没有画面轨：{path.name}")
        start_time = max(0.0, first_source / source_fps)
        if stream.time_base:
            container.seek(max(0, int(start_time / float(stream.time_base))),
                           stream=stream, backward=True, any_frame=False)
        decoded = iter(container.decode(stream))
        fallback_rate = float(stream.average_rate or source_fps)
        decoded_index = 0
        current: av.VideoFrame | None = None
        following = next(decoded, None)
        following_time = (_video_time(following, decoded_index, fallback_rate)
                          if following is not None else float("inf"))
        cached: np.ndarray | None = None
        for timeline_frame in range(span.start, span.end):
            source_frame = int(clip["source_in"]) + round(
                (timeline_frame - int(clip["timeline_start"]))
                * source_fps / project_fps)
            source_frame = min(source_frame, source_limit - 1)
            target_time = source_frame / source_fps
            changed = False
            while following is not None and following_time <= target_time + 1e-7:
                current = following
                changed = True
                decoded_index += 1
                following = next(decoded, None)
                following_time = (_video_time(following, decoded_index, fallback_rate)
                                  if following is not None else float("inf"))
            selected = current or following
            if selected is None:
                if cached is None:
                    raise ValueError(f"视频素材在所选入点没有可解码画面：{path.name}")
                yield cached
                continue
            if changed or cached is None:
                cached = _fit_frame(selected, width, height)
            yield cached


def _video_frames(project: dict[str, Any], total_frames: int,
                  width: int, height: int) -> Iterator[np.ndarray]:
    fps = float(project["settings"]["fps"])
    target_id = str(project["selection"].get("target_track") or "")
    track = next((item for item in project["tracks"]
                  if item["id"] == target_id and item["kind"] == "video"), None)
    if track is None:
        track = next((item for item in project["tracks"] if item["kind"] == "video"), None)
    blank = np.zeros((height, width, 3), dtype=np.uint8)
    for span in track_spans(track, total_frames):
        clip = span.clip
        if clip is None:
            for _ in range(span.end - span.start):
                yield blank
        elif clip["kind"] == "image":
            image = _image_frame(_clip_path(clip), width, height)
            for _ in range(span.end - span.start):
                yield image
        else:
            yield from _video_span_frames(clip, span, fps, width, height)


def _silent_samples(count: int) -> Iterator[np.ndarray]:
    while count > 0:
        size = min(AUDIO_CHUNK, count)
        yield np.zeros((2, size), dtype=np.float32)
        count -= size


def _resampled_audio(path: Path, start_seconds: float, sample_count: int,
                     sample_rate: int = AUDIO_RATE) -> Iterator[np.ndarray]:
    if sample_count <= 0:
        return
    cursor = 0
    with av.open(str(path), mode="r") as container:
        stream = next((item for item in container.streams if item.type == "audio"), None)
        if stream is None:
            yield from _silent_samples(sample_count)
            return
        seek_time = max(0.0, start_seconds - 0.25)
        container.seek(int(seek_time * av.time_base), backward=True, any_frame=False)
        resampler = av.audio.resampler.AudioResampler(
            format="fltp", layout="stereo", rate=sample_rate)
        for packet in container.demux(stream):
            for decoded in packet.decode():
                for frame in resampler.resample(decoded):
                    if frame.pts is not None and frame.time_base is not None:
                        frame_start = float(frame.pts * frame.time_base)
                    else:
                        frame_start = start_seconds + cursor / sample_rate
                    values = frame.to_ndarray().astype(np.float32, copy=False)
                    offset = round((frame_start - start_seconds) * sample_rate)
                    left = max(0, -offset)
                    destination = max(0, offset)
                    if destination > cursor:
                        gap = min(sample_count, destination) - cursor
                        if gap > 0:
                            yield from _silent_samples(gap)
                            cursor += gap
                    if destination + values.shape[1] <= cursor:
                        continue
                    left += max(0, cursor - destination)
                    available = min(values.shape[1] - left, sample_count - cursor)
                    if available > 0:
                        yield np.ascontiguousarray(values[:, left:left + available])
                        cursor += available
                    if cursor >= sample_count:
                        return
        for frame in resampler.resample(None):
            values = frame.to_ndarray().astype(np.float32, copy=False)
            available = min(values.shape[1], sample_count - cursor)
            if available > 0:
                yield np.ascontiguousarray(values[:, :available])
                cursor += available
    if cursor < sample_count:
        yield from _silent_samples(sample_count - cursor)


def _track_audio(project: dict[str, Any], track: dict[str, Any] | None,
                 total_frames: int, sample_rate: int = AUDIO_RATE) -> Iterator[np.ndarray]:
    fps = float(project["settings"]["fps"])
    for span in track_spans(track, total_frames):
        sample_start = round(span.start / fps * sample_rate)
        sample_end = round(span.end / fps * sample_rate)
        count = max(0, sample_end - sample_start)
        clip = span.clip
        if clip is None or clip["kind"] == "image":
            yield from _silent_samples(count)
            continue
        source_fps = max(0.001, float(clip.get("source_fps") or fps))
        source_frame = int(clip["source_in"]) + round(
            (span.start - int(clip["timeline_start"])) * source_fps / fps)
        yield from _resampled_audio(
            _clip_path(clip), source_frame / source_fps, count, sample_rate)


def _fixed_chunks(parts: Iterator[np.ndarray], total_samples: int,
                  size: int = AUDIO_CHUNK) -> Iterator[np.ndarray]:
    pending = np.zeros((2, 0), dtype=np.float32)
    emitted = 0
    for part in parts:
        if part.shape[1] == 0:
            continue
        pending = np.concatenate((pending, part), axis=1)
        while pending.shape[1] >= size and emitted + size <= total_samples:
            yield pending[:, :size]
            pending = pending[:, size:]
            emitted += size
    remaining = total_samples - emitted
    if remaining > 0:
        if pending.shape[1] < remaining:
            pending = np.concatenate(
                (pending, np.zeros((2, remaining - pending.shape[1]), dtype=np.float32)), axis=1)
        yield pending[:, :remaining]


def _audio_chunks(project: dict[str, Any], total_frames: int,
                  sample_rate: int = AUDIO_RATE) -> Iterator[np.ndarray]:
    fps = float(project["settings"]["fps"])
    total_samples = round(total_frames / fps * sample_rate)
    video_id = str(project["selection"].get("target_track") or "")
    video_track = next((item for item in project["tracks"]
                        if item["id"] == video_id and item["kind"] == "video"), None)
    if video_track is None:
        video_track = next((item for item in project["tracks"] if item["kind"] == "video"), None)
    audio_track = next((item for item in project["tracks"] if item["kind"] == "audio"), None)
    video = _fixed_chunks(_track_audio(project, video_track, total_frames, sample_rate), total_samples)
    audio = _fixed_chunks(_track_audio(project, audio_track, total_frames, sample_rate), total_samples)
    for video_part, audio_part in zip(video, audio):
        yield np.clip(video_part + audio_part, -1.0, 1.0)


def export_project(project_json: str | dict[str, Any], filename_prefix: str = "video/H3_粗剪导出",
                   crf: int = 18) -> dict[str, Any]:
    project = normalize_project(project_json)
    fps = float(project["settings"]["fps"])
    width, height = int(project["settings"]["width"]), int(project["settings"]["height"])
    if width > MAX_EXPORT_EDGE or height > MAX_EXPORT_EDGE or width % 2 or height % 2:
        raise ValueError(f"粗剪导出宽高必须是偶数，且不能超过 {MAX_EXPORT_EDGE}")
    total_frames = timeline_frames(project)
    if total_frames < 1:
        raise ValueError("粗剪时间轴没有可导出的片段")
    if total_frames / fps > MAX_EXPORT_SECONDS:
        raise ValueError("单次粗剪导出最长为 6 小时")
    quality = int(crf)
    if not 0 <= quality <= 51:
        raise ValueError("CRF 必须在 0～51 之间")

    # Resolve every source before creating the destination, so an invalid or
    # missing clip cannot leave a partial MP4 in output.
    for track in project["tracks"]:
        for clip in track["clips"]:
            _clip_path(clip)

    target, media = _output_path(filename_prefix, width, height)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp.mp4")
    frame_rate = Fraction(round(fps * 1000), 1000)
    try:
        with av.open(str(temporary), mode="w", format="mp4",
                     options={"movflags": "+faststart"}) as output:
            video_stream = output.add_stream("h264", rate=frame_rate)
            video_stream.width = width
            video_stream.height = height
            video_stream.pix_fmt = "yuv420p"
            video_stream.options = {"crf": str(quality)}
            audio_stream = output.add_stream("aac", rate=AUDIO_RATE, layout="stereo")

            for index, image in enumerate(_video_frames(project, total_frames, width, height)):
                frame = av.VideoFrame.from_ndarray(image, format="rgb24")
                frame.pts = index
                frame.time_base = Fraction(1, 1) / frame_rate
                for packet in video_stream.encode(frame):
                    output.mux(packet)
            for packet in video_stream.encode(None):
                output.mux(packet)

            audio_pts = 0
            for samples in _audio_chunks(project, total_frames):
                frame = av.AudioFrame.from_ndarray(
                    np.ascontiguousarray(samples), format="fltp", layout="stereo")
                frame.sample_rate = AUDIO_RATE
                frame.pts = audio_pts
                frame.time_base = Fraction(1, AUDIO_RATE)
                audio_pts += int(samples.shape[1])
                for packet in audio_stream.encode(frame):
                    output.mux(packet)
            for packet in audio_stream.encode(None):
                output.mux(packet)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "ok": True,
        "video": media,
        "frames": total_frames,
        "fps": fps,
        "width": width,
        "height": height,
        "duration": total_frames / fps,
    }


def register_routes() -> None:
    global _ROUTES_REGISTERED
    if _ROUTES_REGISTERED:
        return
    from aiohttp import web
    from server import PromptServer

    prompt_server = getattr(PromptServer, "instance", None)
    if prompt_server is None:
        return

    @prompt_server.routes.post(EXPORT_ROUTE)
    async def export_roughcut(request):
        remote = str(getattr(request, "remote", "") or "")
        if remote not in {"127.0.0.1", "::1", "localhost"}:
            raise web.HTTPForbidden(text="粗剪导出只允许本机调用")
        try:
            payload = await request.json()
        except (json.JSONDecodeError, ValueError, TypeError) as error:
            raise web.HTTPBadRequest(text="粗剪导出请求不是有效 JSON") from error
        if not isinstance(payload, dict):
            raise web.HTTPBadRequest(text="粗剪导出请求必须是 JSON 对象")
        try:
            # get_save_image_path allocates its counter from the current files.
            # Serialize exports so simultaneous button presses cannot claim the
            # same destination before either encoder has committed its file.
            async with _EXPORT_LOCK:
                result = await asyncio.to_thread(
                    export_project, payload.get("project_json"),
                    payload.get("filename_prefix", "video/H3_粗剪导出"),
                    payload.get("crf", 18))
        except ValueError as error:
            raise web.HTTPBadRequest(text=str(error)) from error
        except (OSError, av.error.FFmpegError) as error:
            raise web.HTTPBadRequest(
                text="粗剪导出失败：素材无法解码或本机 MP4 编码器不可用") from error
        return web.json_response(result)

    _ROUTES_REGISTERED = True
