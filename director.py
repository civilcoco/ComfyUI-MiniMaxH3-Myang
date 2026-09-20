"""Myang H3 Director: one control surface over the existing native pipeline.

SPDX-License-Identifier: GPL-3.0-only

The Director deliberately composes Myang's public nodes instead of duplicating
their sampling code.  Existing workflows and standalone nodes remain valid.
"""

from __future__ import annotations

import json
import logging
import math
import time

from . import core
from . import detail
from . import media_catalog
from . import media as director_media
from . import nodes as legacy
from . import roughcut as roughcut_timeline
from . import turbo


logger = logging.getLogger(__name__)


DIRECTOR_TIMELINE = "导演台分镜卡（手动逐镜头）"
DIRECTOR_SCRIPT = "Agent / 长剧本智能切分"
DIRECTOR_SOURCES = [DIRECTOR_TIMELINE, DIRECTOR_SCRIPT]

TURNAROUND_LORA = "minimax_h3_five_view_1024cont_s600.safetensors"
FACE_DETECTOR = "bbox\\face_yolov8m.pt"


def _input_enabled(value) -> bool:
    """Read a ComfyUI BOOLEAN without treating every non-empty string as true.

    Old saved workflows can temporarily queue optional widget values with the
    previous positional layout while the browser is still using a cached node
    definition.  ``bool("False")`` is True in Python, which used to make an
    unrelated legacy value enable the rough-cut parser.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value == 1
    if isinstance(value, str):
        return value.strip().casefold() in {
            "true", "1", "yes", "on", "开启", "启用",
        }
    return False


def _send_director_preparation(owner: str, activity: str,
                               total_segments: int = 1) -> None:
    """Publish graph-planning work before H3LongVideo can start sampling."""
    owner = str(owner or "").strip()
    if not owner:
        return
    try:
        from server import PromptServer
        instance = getattr(PromptServer, "instance", None)
        if instance is not None and hasattr(instance, "send_sync"):
            instance.send_sync("myh3_progress", {
                "owner_id": owner,
                "segment_index": 1,
                "total_segments": max(1, int(total_segments or 1)),
                "stage": "preparing",
                "activity": str(activity or "导演台准备中"),
            })
    except Exception as error:  # progress must never block generation
        logger.debug("H3-Myang: 导演台准备状态推送失败：%s", error)


def _turnaround_loras():
    try:
        import folder_paths
        names = list(folder_paths.get_filename_list("loras"))
    except Exception:
        names = []
    matches = [name for name in names
               if "five_view" in str(name).casefold()
               or "turnaround" in str(name).casefold()]
    if TURNAROUND_LORA not in matches:
        matches.insert(0, TURNAROUND_LORA)
    return matches


def _face_detectors():
    try:
        import folder_paths
        names = list(folder_paths.get_filename_list("ultralytics"))
    except Exception:
        names = []
    matches = [name for name in names if "face" in str(name).casefold()]
    if FACE_DETECTOR not in matches:
        matches.insert(0, FACE_DETECTOR)
    return matches

DEFAULT_TIMELINE = {
    "version": 2,
    "shots": [{
        "id": "shot_1",
        "enabled": True,
        "duration_seconds": 5.0,
        "brief": "镜头 1",
        "prompt": "",
        "transition": "开场",
        "asset_mode": "仅本镜头",
        "assets": [],
    }],
}


def _timeline_payload(raw, mode: str | None = None) -> dict | list:
    """Select one task-mode bucket from the Director timeline envelope.

    Version 4 timelines keep prompts and public assets per generation mode.
    Older workflows used a single top-level bucket; those remain readable and
    are treated as the requested mode's bucket for backward compatibility.
    """
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
    except json.JSONDecodeError as error:
        raise ValueError("导演台分镜数据不是有效 JSON：%s" % error) from error
    if isinstance(data, dict) and isinstance(data.get("modes"), dict):
        requested = str(mode or data.get("active_mode") or legacy.TASK_FRESH)
        bucket = data["modes"].get(requested)
        if isinstance(bucket, dict):
            return bucket
        # Unknown task values must not borrow another mode's data.
        return {}
    return data


def _timeline_shots(raw, mode: str | None = None) -> list[dict]:
    try:
        data = _timeline_payload(raw, mode)
    except ValueError:
        raise
    if isinstance(data, list):
        shots = data
    elif isinstance(data, dict):
        shots = data.get("shots") or data.get("segments") or []
    else:
        shots = []
    shots = [shot for shot in shots
             if isinstance(shot, dict) and shot.get("enabled", True)]
    if not shots:
        raise ValueError("导演台至少需要一个已启用的分镜")
    if len(shots) > legacy.MAX_SLOTS:
        raise ValueError("导演台一次最多运行 %d 个分镜" % legacy.MAX_SLOTS)
    return shots


def _normalize_assets(raw_assets, label: str, allow_action: bool = True) -> list[dict]:
    """Keep only portable ComfyUI/input references from an asset bucket."""

    if not isinstance(raw_assets, list):
        raise ValueError("%s的素材数据必须是列表" % label)
    limits = {"image": 9, "video": 3, "audio": 3}
    counts = {kind: 0 for kind in limits}
    assets = []
    action_count = 0
    for raw in raw_assets:
        if not isinstance(raw, dict):
            continue
        kind = str(raw.get("kind") or raw.get("media_type") or "").lower()
        if kind == "picture":
            kind = "image"
        if kind not in limits:
            continue
        file_ref = raw.get("file") if isinstance(raw.get("file"), dict) else raw
        name = str(file_ref.get("name") or "").strip()
        if not name:
            continue
        counts[kind] += 1
        if counts[kind] > limits[kind]:
            raise ValueError("%s最多添加图片9个、视频3个、音频3个" % label)
        role = ("action" if allow_action and kind == "video"
                and str(raw.get("role")) == "action" else "reference")
        if role == "action":
            action_count += 1
        asset = {
            "id": str(raw.get("id") or "%s_%d" % (kind, counts[kind])),
            "kind": kind,
            "role": role,
            "label": str(raw.get("label") or name),
            "file": {
                "name": name,
                "subfolder": str(file_ref.get("subfolder") or ""),
                "type": "input",
            },
        }
        weight_mode = str(raw.get("reference_weight_mode") or "off").lower()
        if weight_mode in {"auto", "manual"}:
            try:
                weight = max(0.25, min(3.0, float(
                    raw.get("reference_weight") or 1.0)))
            except (TypeError, ValueError):
                weight = 1.0
            asset.update(reference_weight_mode=weight_mode,
                         reference_weight=weight)
        assets.append(asset)
    if action_count > 1:
        raise ValueError("%s只能指定一个动作源视频" % label)
    return assets


def _shot_assets(shot: dict, shot_index: int) -> tuple[str, list[dict]]:
    assets = _normalize_assets(shot.get("assets") or [], "分镜 %d " % shot_index)
    mode = "叠加全局素材" if str(shot.get("asset_mode")) == "叠加全局素材" else "仅本镜头"
    return mode, assets


def _timeline_globals(raw, mode: str | None = None) -> list[dict]:
    """Materials authored on the Director itself and shared by every segment.

    These land in the same bundle an external Media Agent would produce, so the
    splitter's LLM sees them in its manifest and can hand each segment the tags
    it actually needs.
    """

    # Action transfer has one public asset bucket on its only shot.  Do not
    # resurrect the former hidden global bucket if an old workflow still has
    # entries there; the frontend folds those entries into shot.assets.
    if str(mode or "") == legacy.TASK_TRANSFER:
        return []
    try:
        data = _timeline_payload(raw, mode)
    except ValueError:
        raise
    if not isinstance(data, dict):
        return []
    return _normalize_assets(
        data.get("global_assets") or [], "公共素材", allow_action=False)


def _timeline_template_contract(raw, mode: str | None = None) -> dict:
    """Return an applied template's runtime duration contract, if present."""
    data = _timeline_payload(raw, mode)
    if not isinstance(data, dict):
        return {}
    contract = data.get("template_contract")
    if not isinstance(contract, dict) or contract.get("active") is not True:
        return {}
    try:
        total = max(0.0, float(contract.get("total_seconds") or 0.0))
        segment = max(0.0, float(contract.get("segment_seconds") or 0.0))
    except (TypeError, ValueError):
        return {}
    # The action video's decoded length is authoritative.  Older action
    # templates stored a total_seconds contract and could therefore turn a
    # 14-second source with a 6-second ceiling into a one-segment plan.
    if mode == legacy.TASK_TRANSFER:
        total = 0.0
    return {"active": True, "total_seconds": total,
            "segment_seconds": segment}


def _timeline_plan(raw, overlap: int, fallback_prompt: str = "",
                   mode: str | None = None) -> dict:
    shots = _timeline_shots(raw, mode)

    segments = []
    frame_counts = []
    fallback_prompt = str(fallback_prompt or "").strip()
    for index, shot in enumerate(shots, 1):
        seconds = float(
            shot.get("duration_seconds", shot.get("seconds", 5.0)) or 5.0)
        if not 0.2 <= seconds <= 30.0:
            raise ValueError("分镜 %d 时长必须在 0.2～30 秒" % index)
        frames = core.length_for(seconds, 24.0)
        if overlap >= frames:
            raise ValueError(
                "分镜 %d 只有 %d 帧，短于段间锚点 %d 帧" %
                (index, frames, overlap))
        prompt = str(shot.get("prompt") or fallback_prompt).strip()
        if not prompt:
            raise ValueError("分镜 %d 的提示词为空" % index)
        brief = str(shot.get("brief") or prompt[:60]).strip()
        transition = "开场" if index == 1 else (
            "切镜" if str(shot.get("transition") or "").strip() == "切镜"
            else "承接")
        asset_mode, assets = _shot_assets(shot, index)
        segments.append({
            "index": index,
            "id": str(shot.get("id") or "shot_%d" % index),
            "brief": brief,
            "prompt": prompt,
            "transition": transition,
            "duration_seconds": frames / 24.0,
            "frames": frames,
            "asset_mode": asset_mode,
            "assets": assets,
        })
        frame_counts.append(frames)

    connected_boundaries = sum(
        str(segment.get("transition") or "承接") == "承接"
        for segment in segments[1:])
    total_frames = sum(frame_counts) - overlap * connected_boundaries
    uniform = len(set(frame_counts)) == 1
    return {
        "source": "myang_director_timeline",
        "segment_count": len(segments),
        "frames_per_segment": frame_counts[0] if uniform else 0,
        "segment_seconds_snapped": frame_counts[0] / 24.0 if uniform else 0,
        "overlap_frames": overlap,
        "fps": 24.0,
        "total_seconds_actual": total_frames / 24.0,
        "ref_frames_needed": total_frames,
        "style_header": "",
        "segments": segments,
    }


def _slice_plan_from_segment(plan: dict, start_segment: int,
                             overlap: int, has_context: bool = False) -> dict:
    """Keep the absolute tail of a Director plan for an interrupted rerun.

    The splitter produces its JSON at execution time, so this operation must be
    available as a node as well as a helper.  Segment indices are deliberately
    not renumbered: progress events, reference-video windows and saved filenames
    must continue to say ``第05段`` when a rerun starts at segment five.
    """

    if not isinstance(plan, dict):
        raise ValueError("导演台分段计划必须是 JSON 对象")
    segments = [item for item in (plan.get("segments") or [])
                if isinstance(item, dict)]
    if not segments:
        raise ValueError("导演台分段计划里没有可生成的分段")
    start = max(1, int(start_segment or 1))
    total = len(segments)
    if start > total:
        raise ValueError(
            "从第 %d 段开始生成，但本次分镜计划只有 %d 段" %
            (start, total))
    if start == 1:
        result = dict(plan)
        result["resume_start_segment"] = 1
        result["total_segments_planned"] = total
        result["resume_context_required"] = False
        return result

    kept = [dict(item) for item in segments[start - 1:]]
    first_transition = (
        "切镜" if str(kept[0].get("transition") or "").strip() == "切镜"
        else "承接")
    context_required = first_transition == "承接"
    if context_required and not has_context:
        raise ValueError(
            "从第 %d 段开始生成，而这一段设置为『承接』；请把第 %d 段成片"
            "接到『前段视频』。如果它本来就是独立镜头，可把该分镜前的衔接改成『切镜』" %
            (start, start - 1))

    fps = float(plan.get("fps") or 24.0)
    default_frames = int(plan.get("frames_per_segment") or 0)
    default_seconds = float(plan.get("segment_seconds_snapped") or 0.0)
    frames = []
    for item in kept:
        if item.get("frames") is not None:
            frame_count = int(item["frames"])
        elif item.get("duration_seconds") is not None:
            frame_count = core.length_for(float(item["duration_seconds"]), fps)
        elif default_frames:
            frame_count = default_frames
        else:
            frame_count = core.length_for(default_seconds or 5.0, fps)
        frames.append(frame_count)

    connected = sum(
        str(item.get("transition") or "承接").strip() != "切镜"
        for item in kept[1:])
    trim_count = connected + (1 if context_required else 0)
    total_frames = sum(frames) - int(overlap) * trim_count
    uniform = len(set(frames)) == 1
    result = dict(plan)
    result.update({
        "segment_count": len(kept),
        "frames_per_segment": frames[0] if uniform else 0,
        "segment_seconds_snapped": frames[0] / fps if uniform else 0,
        "total_seconds_actual": total_frames / fps,
        "ref_frames_needed": total_frames,
        "resume_start_segment": start,
        "total_segments_planned": total,
        "resume_context_required": context_required,
        "resume_source_transition": first_transition,
        "segments": kept,
    })
    return result


def _aligned_frames_up(frame_count: int) -> int:
    frame_count = max(5, int(frame_count))
    return max(5, math.ceil((frame_count - 5) / 17) * 17 + 5)


def _aligned_frames_down(frame_count: int) -> int:
    frame_count = max(5, int(frame_count))
    return max(5, math.floor((frame_count - 5) / 17) * 17 + 5)


def _limit_plan_to_visible_frames(plan: dict, target_frames: int,
                                  overlap: int) -> dict:
    """Fit a plan to a rough-cut I/O duration without changing its prompts.

    Every H3 segment stays on the legal 17k+5 grid.  The final aligned surplus
    is deliberately left for ``H3RoughCutSave`` to crop, so the written clip is
    still exactly the user's I/O duration while generation never comes up short.
    """

    if not isinstance(plan, dict):
        raise ValueError("导演台分段计划必须是 JSON 对象")
    segments = [dict(item) for item in (plan.get("segments") or [])
                if isinstance(item, dict)]
    if not segments:
        raise ValueError("导演台分段计划没有可用于粗剪区间的镜头")
    target = max(1, int(target_frames))
    overlap = max(0, int(overlap))
    kept = []
    visible = 0
    completed = False
    for item in segments:
        raw_frames = _aligned_frames_up(int(item.get("frames") or 5))
        # Match H3LongVideo: legacy/empty transition means seamless carry;
        # only an explicit cut disables overlap trimming.
        connected = bool(kept) and str(
            item.get("transition") or "").strip() != "切镜"
        hidden_head = overlap if connected else 0
        # H3LongVideo validates every segment against the configured context
        # size even when the first shot does not consume it.  A tiny I/O range
        # therefore needs one legal over-generated segment, cropped at save.
        if raw_frames <= overlap:
            raw_frames = _aligned_frames_up(overlap + 1)
        contribution = max(1, raw_frames - hidden_head)
        remaining = target - visible
        fitted = dict(item)
        if remaining <= contribution:
            minimum = hidden_head + max(1, remaining)
            raw_frames = _aligned_frames_up(minimum)
            if raw_frames <= overlap:
                raw_frames = _aligned_frames_up(overlap + 1)
            completed = True
        fitted["frames"] = raw_frames
        fitted["duration_seconds"] = raw_frames / 24.0
        kept.append(fitted)
        visible += max(1, raw_frames - hidden_head)
        if completed:
            break

    if visible < target:
        # The selected I/O range is longer than the authored plan.  Continue
        # the final shot instead of freezing the saved tail.
        extra = target - visible
        last = kept[-1]
        previous_frames = int(last["frames"])
        last["frames"] = _aligned_frames_up(previous_frames + extra)
        last["duration_seconds"] = int(last["frames"]) / 24.0

    actual = 0
    for index, item in enumerate(kept):
        connected = index > 0 and str(
            item.get("transition") or "").strip() != "切镜"
        actual += int(item["frames"]) - (overlap if connected else 0)
    fitted_plan = dict(plan)
    fitted_plan["segments"] = kept
    fitted_plan["segment_count"] = len(kept)
    fitted_plan["frames_per_segment"] = (
        int(kept[0]["frames"])
        if len({int(item["frames"]) for item in kept}) == 1 else 0)
    fitted_plan["segment_seconds_snapped"] = (
        float(fitted_plan["frames_per_segment"]) / 24.0
        if fitted_plan["frames_per_segment"] else 0)
    fitted_plan["total_seconds_actual"] = actual / 24.0
    fitted_plan["ref_frames_needed"] = actual
    fitted_plan["roughcut_target_frames"] = target
    return fitted_plan


def _single_prompt_transfer_plan(raw, overlap: int, segment_seconds: float,
                                 ref_frames: int, fallback_prompt: str = "",
                                 allow_embedded_video: bool = False,
                                 start_segment: int = 1,
                                 auto_segment: bool = True) -> dict:
    """Split one action source while keeping one prompt and one material set.

    ``start_segment`` drops the already produced head of a run.  The full split
    is still computed first, so every kept segment carries the absolute index
    and the absolute ``ref_start_frame`` it would have had in a full run.
    """

    shot = _timeline_shots(raw, legacy.TASK_TRANSFER)[0]
    prompt = str(shot.get("prompt") or fallback_prompt).strip()
    if not prompt:
        raise ValueError("动作迁移需要填写一个全片统一提示词")
    _, assets = _shot_assets(shot, 1)
    videos = [asset for asset in assets if asset["kind"] == "video"]
    if len(videos) > 1:
        raise ValueError("动作迁移导演台最多只能上传一个动作参考视频")
    if videos and not allow_embedded_video:
        raise ValueError(
            "动作迁移只允许一个『动作参考视频』输入；"
            "请删除导演台素材卡里的视频，把完整视频接到左侧 ref_video")
    # The action video is removed from the per-shot Media Agent bundle and is
    # wired through H3LongVideo's direct ``ref_video`` socket.  Preserve the
    # card's weight choice before removing it; otherwise the backend silently
    # falls back to the old hard-coded ``off`` policy and the log never shows
    # that automatic weighting was requested.  External ref_video inputs have
    # no card metadata, so they intentionally use the same automatic default.
    action_weight_mode = "auto"
    action_weight = 1.0
    if videos:
        action_weight_mode = str(
            videos[0].get("reference_weight_mode") or "auto").strip().lower()
        if action_weight_mode not in {"auto", "manual"}:
            action_weight_mode = "auto"
        try:
            action_weight = max(0.25, min(
                3.0, float(videos[0].get("reference_weight") or 1.0)))
        except (TypeError, ValueError):
            action_weight = 1.0
    assets = [asset for asset in assets if asset["kind"] != "video"]
    ref_frames = int(ref_frames)
    if ref_frames <= overlap:
        raise ValueError(
            "动作参考视频只有 %d 帧，必须长于段间锚点 %d 帧" %
            (ref_frames, overlap))
    # A short action clip often needs no artificial segmentation. In that
    # mode the source length is authoritative and the whole clip becomes one
    # segment; the normal segmented path remains unchanged.
    frames = (core.length_for(float(segment_seconds), 24.0)
              if auto_segment else _aligned_frames_up(ref_frames))
    if frames <= overlap:
        raise ValueError("动作迁移分段时长太短：每段帧数必须大于段间锚点")
    hop = frames - overlap
    # Start from the user's visible-duration count. The overlap still consumes
    # real sampler frames, however, so the fitted final H3 window may exceed
    # the ceiling even when ``ceil(source / ceiling)`` does not. On 16GB cards
    # that exact 175 -> 192 jump is enough to make only segment 2 fail at 832P.
    count = max(1, math.ceil(ref_frames / frames))
    if count > legacy.MAX_SLOTS:
        raise ValueError(
            "动作参考视频按 %.2f 秒切分会产生 %d 段，超过导演台上限 %d；"
            "请增大单段时长" % (float(segment_seconds), count, legacy.MAX_SLOTS))

    frame_counts = [frames] * count
    if count > 1:
        last_start = (count - 1) * hop
        frame_counts[-1] = _aligned_frames_up(ref_frames - last_start)
    else:
        frame_counts[0] = _aligned_frames_up(ref_frames)
    if frame_counts[-1] <= overlap:
        frame_counts[-1] = _aligned_frames_up(overlap + 1)

    # Treat segment_seconds as a hard sampler-window ceiling, not merely a
    # visible-story estimate. Preserve the first full window and distribute an
    # overflowing tail across the minimum number of additional H3-grid clips.
    # This avoids both an oversized final pass and a nearly empty tail shot.
    rebalanced = False
    if auto_segment and any(value > frames for value in frame_counts):
        while (count * frames - overlap * max(0, count - 1)
               < ref_frames):
            count += 1
        if count > legacy.MAX_SLOTS:
            raise ValueError(
                "动作参考视频计入段间承接后需要 %d 段，超过导演台上限 %d；"
                "请增大单段上限" % (count, legacy.MAX_SLOTS))
        raw_required = ref_frames + overlap * max(0, count - 1)
        minimum = _aligned_frames_up(overlap + 1)
        remaining_slots = max(0, count - 1)
        if remaining_slots:
            target = math.ceil((raw_required - frames) / remaining_slots)
            balanced = max(minimum, min(frames, _aligned_frames_down(target)))
            frame_counts = [frames] + [balanced] * remaining_slots
            cursor = count - 1
            while sum(frame_counts) < raw_required:
                if frame_counts[cursor] + 17 <= frames:
                    frame_counts[cursor] += 17
                cursor -= 1
                if cursor <= 0:
                    cursor = count - 1
        else:
            frame_counts = [frames]
        rebalanced = True

    ref_starts = []
    source_cursor = 0
    for frame_count in frame_counts:
        ref_starts.append(source_cursor)
        source_cursor += int(frame_count) - overlap
    if rebalanced:
        logger.info(
            "H3-Myang: 动作迁移尾段原本超过单段上限，已按承接窗口均衡为 %s 帧",
            "/".join(str(value) for value in frame_counts))

    brief = str(shot.get("brief") or "动作迁移").strip()
    segments = [{
        "index": index,
        "id": "transfer_%d" % index,
        "brief": "%s · 第%d段" % (brief, index),
        "prompt": prompt,
        "transition": "开场" if index == 1 else "承接",
        "duration_seconds": frame_count / 24.0,
        "frames": frame_count,
        "ref_start_frame": ref_starts[index - 1],
        "asset_mode": "叠加全局素材",
        "assets": assets,
    } for index, frame_count in enumerate(frame_counts, 1)]
    total_frames = sum(frame_counts) - overlap * max(0, count - 1)

    start = int(start_segment or 1)
    if not 1 <= start <= count:
        raise ValueError(
            "起始段必须在 1～%d 之间；这条动作参考视频按 %.2f 秒只能切成 %d 段" %
            (count, float(segment_seconds), count))
    kept = segments[start - 1:]
    kept_frames = [segment["frames"] for segment in kept]
    return {
        "source": "myang_director_action_transfer",
        "segment_count": len(kept),
        "frames_per_segment": frames if len(set(kept_frames)) == 1 else 0,
        "segment_seconds_snapped": frames / 24.0,
        "overlap_frames": overlap,
        "fps": 24.0,
        "total_seconds_actual": total_frames / 24.0,
        "ref_frames_needed": total_frames,
        "ref_frames_available": ref_frames,
        "reference_tail_pad": True,
        "resume_start_segment": start,
        "total_segments_planned": count,
        "style_header": "",
        "action_reference_weight_mode": action_weight_mode,
        "action_reference_weight": action_weight,
        "segments": kept,
    }


def _action_video_assets(raw, mode: str | None = legacy.TASK_TRANSFER) -> list[dict]:
    _, assets = _shot_assets(_timeline_shots(raw, mode)[0], 1)
    return [asset for asset in assets if asset["kind"] == "video"]


def _load_director_action_video(asset: dict, resolution=None,
                                width=1920, height=1080):
    """Load the uploaded action clip already capped to the wanted canvas.

    The downstream ``H3ReferenceResize`` cannot save this one: by the time it
    runs, the clip has already been materialised at whatever the phone filmed,
    and that is where a 4K/60fps source takes the run down. Handing the same
    resolution policy to the decoder is what makes the cap actually bound peak
    memory instead of only trimming what survived.
    """
    path, _ = director_media._safe_input_path(asset)
    choice = str(resolution or director_media.REFERENCE_VIDEO_ORIGINAL)

    def canvas(shown_width, shown_height):
        return director_media.reference_video_size(
            shown_width, shown_height, choice, int(width), int(height))

    frames, soundtrack, source_fps = core.decode_video_bounded(
        path, size_policy=canvas)
    frames = core._to_24fps(frames, source_fps)
    if int(frames.shape[0]) < 1:
        raise ValueError("导演台上传的动作参考视频没有可用帧")
    return frames, soundtrack


def _reject_director_video_references(plan: dict, task_label: str):
    for segment in plan.get("segments") or []:
        if any(str(asset.get("kind")) == "video"
               for asset in (segment.get("assets") or [])
               if isinstance(asset, dict)):
            raise ValueError(
                "%s里的前文/动作视频必须接左侧 ref_video；"
                "导演台镜头素材只保留图片和音频" % task_label)


def _reject_media_videos(media, task_label: str):
    videos = [item for item in (getattr(media, "items", ()) or ())
              if str(getattr(item, "media_type", "")).lower() == "video"]
    if videos:
        raise ValueError(
            "%s只允许左侧 ref_video 这一条视频输入；"
            "Media Agent 里可以保留图片和音频，但请移除视频" % task_label)


def _send_plan_ready(owner: str, plan: dict) -> None:
    """Push the finished segment list to the Director panel.

    The Agent / long-script path only learns its per-segment prompts once the
    LLM has run, so without this the operator cannot read what will actually be
    generated until each segment starts — and then only one truncated line of
    whichever segment is running.
    """
    try:
        from server import PromptServer
        instance = getattr(PromptServer, "instance", None)
        if instance is None or not hasattr(instance, "send_sync"):
            return
        default_frames = int(plan.get("frames_per_segment") or 0)
        default_seconds = float(plan.get("segment_seconds_snapped") or 0.0)
        segments = []
        for offset, segment in enumerate(plan.get("segments") or [], 1):
            if not isinstance(segment, dict):
                continue
            segments.append({
                "index": int(segment.get("index") or offset),
                "title": legacy._sanitize_storyboard_title(
                    segment.get("title"), segment.get("brief") or segment.get("prompt")),
                "brief": str(segment.get("brief") or ""),
                "prompt": str(segment.get("prompt") or ""),
                "transition": str(segment.get("transition") or ""),
                "frames": int(segment.get("frames") or default_frames),
                "duration_seconds": float(
                    segment.get("duration_seconds") or default_seconds),
                "subjects": list(segment.get("subjects") or []),
                "skills": list(segment.get("skills") or []),
                "skill_source": str(segment.get("skill_source") or ""),
            })
        instance.send_sync("myh3_director_plan", {
            "owner_id": owner,
            "source": str(plan.get("source") or ""),
            "segment_count": len(segments),
            "style_header": str(plan.get("style_header") or ""),
            "skill_source": str(plan.get("skill_source") or ""),
            "skill_strategy": str(plan.get("skill_strategy") or ""),
            "skill_plan": list(plan.get("skill_plan") or []),
            "segments": segments,
        })
    except Exception as error:  # noqa: BLE001 - a preview must never fail a run
        logger.debug("H3-Myang: 分段提示词预览推送失败：%s", error)


class H3DirectorPlanValue:
    CATEGORY = "沐阳 H3/导演台"
    FUNCTION = "emit"
    RETURN_TYPES = ("STRING", "FLOAT")
    RETURN_NAMES = ("plan_json", "fps")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "plan_json": ("STRING", {"multiline": True, "default": ""}),
            "progress_owner": ("STRING", {"default": ""}),
        }}

    def emit(self, plan_json, progress_owner=""):
        text = str(plan_json)
        owner = str(progress_owner or "").strip()
        if owner:
            try:
                plan = json.loads(text)
            except json.JSONDecodeError as error:
                raise ValueError("导演台分段计划不是有效 JSON：%s" % error) from error
            if not isinstance(plan, dict):
                raise ValueError("导演台分段计划必须是 JSON 对象")
            plan["progress_owner"] = owner
            text = json.dumps(plan, ensure_ascii=False)
            # 这是全流程里唯一同时握有完整分段计划和面板 id 的地方：
            # 手动分镜、智能切分、动作迁移三条路都汇到这里。
            _send_plan_ready(owner, plan)
        return (text, 24.0)


class H3DirectorPlanSlice:
    CATEGORY = "沐阳 H3/导演台/内部"
    FUNCTION = "slice"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("plan_json",)
    DESCRIPTION = "导演台内部节点：在 LLM/分镜计划完成后按绝对段号裁切断点续跑范围。"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "plan_json": ("STRING", {"forceInput": True}),
                "start_segment": ("INT", {
                    "default": 2, "min": 1, "max": legacy.MAX_SLOTS}),
                "overlap_frames": ("INT", {
                    "default": 22, "min": 5, "max": 56}),
            },
            "optional": {
                "context_video": ("IMAGE",),
            },
        }

    def slice(self, plan_json, start_segment, overlap_frames,
              context_video=None):
        try:
            plan = json.loads(plan_json) if isinstance(plan_json, str) else plan_json
        except json.JSONDecodeError as error:
            raise ValueError("导演台分段计划不是有效 JSON：%s" % error) from error
        sliced = _slice_plan_from_segment(
            plan, int(start_segment), int(overlap_frames),
            has_context=context_video is not None)
        return (json.dumps(sliced, ensure_ascii=False),)


class H3DirectorPlanLimit:
    CATEGORY = "沐阳 H3/导演台/内部"
    FUNCTION = "fit"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("plan_json",)
    DESCRIPTION = "导演台内部节点：把任意分镜计划拟合到粗剪时间轴 I/O 时长。"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "plan_json": ("STRING", {"forceInput": True}),
            "target_frames": ("INT", {"default": 120, "min": 1, "max": 1000000}),
            "overlap_frames": ("INT", {"default": 22, "min": 0, "max": 56}),
        }}

    def fit(self, plan_json, target_frames, overlap_frames):
        try:
            plan = json.loads(plan_json) if isinstance(plan_json, str) else plan_json
        except json.JSONDecodeError as error:
            raise ValueError("导演台分段计划不是有效 JSON：%s" % error) from error
        fitted = _limit_plan_to_visible_frames(
            plan, int(target_frames), int(overlap_frames))
        return (json.dumps(fitted, ensure_ascii=False),)


class H3DirectorActionSource:
    CATEGORY = "沐阳 H3/导演台/内部"
    FUNCTION = "load"
    RETURN_TYPES = ("STRING", "IMAGE", "AUDIO")
    RETURN_NAMES = ("plan_json", "动作参考视频", "动作参考音频")
    DESCRIPTION = "导演台内部节点：加载卡片中唯一的动作视频，并按指定时长生成自动分段计划。"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "timeline_json": ("STRING", {"multiline": True, "default": ""}),
            "fallback_prompt": ("STRING", {"multiline": True, "default": ""}),
            "segment_seconds": ("FLOAT", {
                "default": 10.0, "min": legacy.MIN_SECONDS,
                "max": legacy.MAX_SECONDS, "step": 0.5}),
            "auto_segment": ("BOOLEAN", {
                "default": True,
                "label_on": "按时长自动分段",
                "label_off": "整段生成",
                "tooltip": "关闭后按动作视频完整长度只生成一段"}),
            "overlap_frames": ("INT", {"default": 22, "min": 5, "max": 56}),
            "start_segment": ("INT", {
                "default": 1, "min": 1, "max": legacy.MAX_SLOTS,
                "tooltip": "从第几段开始生成；1 表示整条重跑"}),
        }, "optional": {
            "target_frames": ("INT", {
                "default": 0, "min": 0, "max": 1000000,
                "tooltip": "粗剪 I/O 指定的生成帧数；0=沿用完整动作视频"}),
            "minimum_source_frame": ("INT", {
                "default": 0, "min": 0, "max": 1000000,
                "tooltip": "断点动作迁移至少需要到达的动作源帧"}),
            "resolution": (director_media.REFERENCE_VIDEO_RESOLUTIONS, {
                "default": director_media.REFERENCE_VIDEO_ORIGINAL,
                "tooltip": "解码时就套用的画布上限。4K / 60fps 的源必须在这里收窄，"
                           "下游的缩放节点已经来不及了"}),
            "width": ("INT", {
                "default": 1920, "min": 32, "max": 1920, "step": 32}),
            "height": ("INT", {
                "default": 1080, "min": 32, "max": 1920, "step": 32}),
        }}

    def load(self, timeline_json, fallback_prompt, segment_seconds, overlap_frames,
             auto_segment=True, start_segment=1, target_frames=0,
             minimum_source_frame=0, resolution=None, width=1920, height=1080):
        videos = _action_video_assets(timeline_json)
        if len(videos) != 1:
            raise ValueError("动作迁移导演台必须上传且只能上传一个动作参考视频")
        frames, soundtrack = _load_director_action_video(
            videos[0], resolution=resolution, width=width, height=height)
        required = int(target_frames or 0)
        if required > 0 and int(frames.shape[0]) < required:
            raise ValueError(
                "粗剪 I/O / 模板时长需要 %d 帧动作参考，但上传的动作视频只有 %d 帧；"
                "请缩短目标时长或换更长的动作视频" %
                (required, int(frames.shape[0])))
        if int(minimum_source_frame or 0) >= int(frames.shape[0]):
            raise ValueError(
                "动作迁移断点位于参考视频第 %d 帧，但动作源只有 %d 帧" %
                (int(minimum_source_frame), int(frames.shape[0])))
        planning_frames = (int(target_frames) if int(target_frames or 0) > 0
                           else int(frames.shape[0]))
        plan = _single_prompt_transfer_plan(
            timeline_json, int(overlap_frames), float(segment_seconds),
            planning_frames, fallback_prompt, allow_embedded_video=True,
            start_segment=int(start_segment),
            auto_segment=_input_enabled(auto_segment))
        logger.info(
            "H3-Myang: 动作迁移参考视频已解析 | %d 帧（%.2f 秒）| 单段上限 %.2f 秒 | 计划 %d 段",
            int(frames.shape[0]), int(frames.shape[0]) / 24.0,
            float(segment_seconds), int(plan.get("total_segments_planned") or
                                       plan.get("segment_count") or 1))
        return (json.dumps(plan, ensure_ascii=False), frames,
                soundtrack if soundtrack is not None else director_media._empty_audio())


class H3DirectorMediaImage:
    CATEGORY = "沐阳 H3/导演台/内部"
    FUNCTION = "pick"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("角色参考图",)
    DESCRIPTION = "导演台内部节点：从公共素材包选择一张角色图。"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "media": ("MINIMAX_H3_MEDIA",),
            "image_ordinal": ("INT", {"default": 1, "min": 1, "max": 9}),
        }}

    def pick(self, media, image_ordinal):
        wanted = int(image_ordinal)
        seen = 0
        for item in (getattr(media, "items", ()) or ()):
            if str(getattr(item, "media_type", "")).lower() != "image":
                continue
            seen += 1
            if seen != wanted:
                continue
            frames = core._frames(getattr(item, "value", None))
            if frames is None or int(frames.shape[0]) < 1:
                break
            return (frames[:1],)
        raise ValueError(
            "多视角分镜选择了 @图片%d，但公共素材实际只有 %d 张图片；"
            "请在公共素材或 Media Agent 中添加角色图，或修改图片序号" %
            (wanted, seen))


class H3DirectorTurnaroundMedia:
    CATEGORY = "沐阳 H3/导演台/内部"
    FUNCTION = "append"
    RETURN_TYPES = ("MINIMAX_H3_MEDIA", "STRING")
    RETURN_NAMES = ("增强素材", "增强分镜计划")
    DESCRIPTION = "导演台内部节点：把五视图加入公共素材并绑定到每个分镜。"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "media": ("MINIMAX_H3_MEDIA",),
            "sheet": ("IMAGE",),
            "plan_json": ("STRING", {"multiline": True, "forceInput": True}),
            "owner_id": ("STRING", {"default": ""}),
        }}

    def append(self, media, sheet, plan_json, owner_id=""):
        if not hasattr(media, "items"):
            raise ValueError("多视角分镜需要有效的 Media Agent / 导演台公共素材包")
        try:
            plan = json.loads(str(plan_json))
        except json.JSONDecodeError as error:
            raise ValueError("多视角分镜收到的导演台计划不是有效 JSON") from error
        if not isinstance(plan, dict):
            raise ValueError("多视角分镜收到的导演台计划必须是 JSON 对象")

        items = list(media.items)
        links = list(getattr(media, "links", ()) or ())
        ordinal = 1 + sum(
            str(getattr(item, "media_type", "")).lower() == "image"
            for item in items)
        input_index = max(
            [int(getattr(item, "input_index", 0)) for item in items] + [0]) + 1
        items.append(media_catalog.MyangMediaAsset(
            input_index=input_index, media_type="image", value=sheet[:1]))
        links.append({
            "order": input_index,
            "media_type": "image",
            "filename": "generated_turnaround_sheet.png",
            "subject": "角色五视图分镜参考",
            "source": "director_turnaround",
        })

        suffix = (
            "\nMulti-view identity reference: @图片%d; keep the same face, hair, "
            "body proportions and wardrobe across camera angles; do not render "
            "the contact-sheet layout or panel borders." % ordinal)
        segments = plan.get("segments") or []
        for segment in segments:
            if not isinstance(segment, dict):
                continue
            prompt = str(segment.get("prompt") or "").rstrip()
            if "@图片%d" % ordinal not in prompt:
                segment["prompt"] = prompt + suffix
            # The generated sheet is a global reference. Keep its ordinal stable
            # by retaining the global bundle ahead of shot-local materials.
            segment["asset_mode"] = "叠加全局素材"
        plan["turnaround_reference"] = {
            "image_ordinal": ordinal,
            "subject": "角色五视图分镜参考",
        }

        owner = str(owner_id or "")
        if owner:
            try:
                from server import PromptServer
                from .progress import _save_preview_frame
                filename = _save_preview_frame(
                    sheet[:1], "turnaround_%s" % owner, 0, "sheet")
                instance = getattr(PromptServer, "instance", None)
                if filename and instance is not None and hasattr(instance, "send_sync"):
                    instance.send_sync("myh3_director_turnaround", {
                        "owner_id": owner,
                        "preview_file": filename,
                        "preview_ts": int(time.time() * 1000),
                        "image_ordinal": ordinal,
                    })
            except Exception as error:  # preview must never stop generation
                logger.debug("H3-Myang: 多视角分镜预览推送失败：%s", error)

        return (media_catalog.MyangMediaCatalog(
            items=tuple(items), links=tuple(links)),
            json.dumps(plan, ensure_ascii=False))


class H3DirectorEnhancementSettings:
    CATEGORY = "沐阳 H3/导演台/内部"
    FUNCTION = "build"
    RETURN_TYPES = ("MYANG_H3_ENHANCE",)
    RETURN_NAMES = ("增强设置",)
    DESCRIPTION = "导演台内部节点：FaceRefine 与 MAINodes 的可选分段增强设置。"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "face_enabled": ("BOOLEAN", {"default": False}),
            "face_detector": ("STRING", {"default": FACE_DETECTOR}),
            "face_steps": ("INT", {"default": 4, "min": 1, "max": 50}),
            "face_denoise": ("FLOAT", {"default": 0.45, "min": 0.01,
                                       "max": 1.0, "step": 0.01}),
            "face_crop_factor": ("FLOAT", {"default": 2.5, "min": 1.2,
                                           "max": 8.0, "step": 0.1}),
            "face_identity_ordinal": ("INT", {"default": 0, "min": 0, "max": 9}),
            "motion_enabled": ("BOOLEAN", {"default": False}),
            "motion_preset": (["balanced (default)",
                               "max quality (wide plateau)",
                               "economy (tight spans)"],
                              {"default": "balanced (default)"}),
            "motion_steps": ("INT", {"default": 6, "min": 4, "max": 50}),
            "motion_inject": ("FLOAT", {"default": 0.70, "min": 0.05,
                                        "max": 1.0, "step": 0.05}),
        }, "optional": {
            "model": ("MODEL",),
        }}

    def build(self, face_enabled, face_detector, face_steps, face_denoise,
              face_crop_factor, face_identity_ordinal, motion_enabled,
              motion_preset, motion_steps, motion_inject, model=None):
        return ({
            "model": model,
            "face": {
                "enabled": bool(face_enabled),
                "detector": str(face_detector),
                "steps": int(face_steps),
                "denoise": float(face_denoise),
                "crop_factor": float(face_crop_factor),
                "identity_ordinal": int(face_identity_ordinal),
            },
            "motion": {
                "enabled": bool(motion_enabled),
                "preset": str(motion_preset),
                "steps": int(motion_steps),
                "inject": float(motion_inject),
            },
        },)


class H3Director:
    CATEGORY = "沐阳 H3/导演台"
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE", "AUDIO", "STRING", "FLOAT")
    RETURN_NAMES = ("images", "audio", "plan_json", "fps")
    DESCRIPTION = (
        "把 Media Agent、分镜时间线、原生锚点长视频、Turbo LoRA 联合音画调度与二采统一到一个导演台。"
        "底层仍调用现有 Myang 节点，旧工作流和单节点高级用法不会被替换。")

    @classmethod
    def INPUT_TYPES(cls):
        director_services = list(legacy.llm_service_options())
        if "未配置 LLM 服务" not in director_services:
            director_services.insert(0, "未配置 LLM 服务")
        return {
            "required": {
                "h3": ("MYANG_H3",),
                "model": ("MODEL", {"tooltip": "一采模型；基础模型或 Turbo 联合模型都直接接这里"}),
                "sampler": ("SAMPLER",),
                "source_mode": (DIRECTOR_SOURCES, {"default": DIRECTOR_TIMELINE}),
                "timeline_json": ("STRING", {
                    "multiline": True,
                    "default": json.dumps(DEFAULT_TIMELINE, ensure_ascii=False),
                }),
                "script_fallback": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "智能切分时作为剧本；可把此控件转换为输入后连接 Agent easy_prompt",
                }),
                "total_seconds": ("FLOAT", {
                    "default": 60.0, "min": 1.0, "max": 3600.0, "step": 1.0}),
                "segment_seconds": ("FLOAT", {
                    "default": 10.0, "min": legacy.MIN_SECONDS,
                    "max": legacy.MAX_SECONDS, "step": 0.5}),
                "llm_enabled": ("BOOLEAN", {
                    "default": True, "label_on": "LLM 自动拆镜头",
                    "label_off": "同一提示词直通"}),
                "llm_service": (director_services,),
                "task_mode": (legacy.TASK_MODES, {"default": legacy.TASK_FRESH}),
                "resolution": (core.RESOLUTION_PRESETS, {"default": "480P"}),
                "aspect_ratio": (core.ASPECT_RATIOS, {"default": "16:9"}),
                "width": ("INT", {"default": 864, "min": 32, "max": 16384, "step": 32}),
                "height": ("INT", {"default": 480, "min": 32, "max": 16384, "step": 32}),
                "steps": ("INT", {"default": 25, "min": 1, "max": 200}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 1.0, "step": 0.01}),
                "scheduler": (["simple", "beta", "normal"], {"default": "simple"}),
                "noise_seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xffffffffffffffff,
                    "control_after_generate": True}),
                "context_length": (legacy.CONTEXT_LENGTHS, {"default": "22"}),
                "ref_image_size": (list(core.REF_IMAGE_SIZES), {
                    "default": core.REF_AREA_MATCH}),
                "二采开启": ("BOOLEAN", {
                    "default": False, "label_on": "开启导演台二采",
                    "label_off": "关闭导演台二采"}),
                "二采模式": (detail.DETAIL_MODES, {
                    "default": detail.DETAIL_MODE_UPSCALE_REFINE}),
                "二采分辨率": (detail.DETAIL_RESOLUTIONS[1:], {
                    "default": "832P"}),
                "二采自定义宽": ("INT", {
                    "default": 1664, "min": 32, "max": 8192, "step": 32}),
                "二采自定义高": ("INT", {
                    "default": 928, "min": 32, "max": 8192, "step": 32}),
                "二采步数": ("INT", {"default": 4, "min": 1, "max": 100}),
                "二采重绘幅度": ("FLOAT", {
                    "default": 0.2, "min": 0.01, "max": 1.0, "step": 0.01}),
                "二采调度器": (detail.DETAIL_SCHEDULERS, {"default": "beta"}),
                "二采采样器": (detail.DETAIL_SAMPLERS, {
                    "default": "res_multistep"}),
                "二采放大方式": (detail.DETAIL_UPSCALE_METHODS, {
                    "default": "neural_3d (神经3D Latent放大·推荐)"}),
                "二采分块帧数": ("INT", {
                    "default": 4, "min": 1, "max": 64}),
                "二采Latent模型": (detail.latent_model_names(), {
                    "tooltip": "神经3D放大权重；其他放大方式会自动隐藏"}),
                "二采精度": (detail.LATENT_PRECISIONS, {
                    "default": detail.LATENT_PRECISIONS[0]}),
                "二采时间分块": ("INT", {
                    "default": 0, "min": 0, "max": 256,
                    "tooltip": "神经3D放大的时间分块；0=全上下文单次推理（无接缝，"
                               "推荐）。显存不够再往上调，8 最省显存"}),
                "二采轮数": ("INT", {"default": 1, "min": 1, "max": 8}),
                "二采种子策略": (detail.DETAIL_SEED_MODES, {
                    "default": detail.DETAIL_SEED_INHERIT}),
                "save_segments": ("BOOLEAN", {"default": True}),
                "segment_prefix": ("STRING", {"default": "video/H3_导演台"}),
                "save_raw_segments": ("BOOLEAN", {"default": False}),
                "参考视频分辨率": (director_media.REFERENCE_VIDEO_RESOLUTIONS, {
                    "default": director_media.REFERENCE_VIDEO_ORIGINAL,
                    "tooltip": "动作迁移/视频续写的参考视频预处理；默认保持原尺寸，最高 1080P"}),
                "参考视频自定义宽": ("INT", {
                    "default": 1920, "min": 32, "max": 1920, "step": 32}),
                "参考视频自定义高": ("INT", {
                    "default": 1080, "min": 32, "max": 1920, "step": 32}),
                "起始段": ("INT", {
                    "default": 1, "min": 1, "max": legacy.MAX_SLOTS,
                    "tooltip": "勾选『从指定段开始』后生效；支持动作迁移和导演台"
                               "手动分镜卡。Agent 智能切分始终从第 1 段生成"}),
                "skill_preset": (legacy.skill_preset_options(), {
                    "default": legacy.SKILL_PRESET_AUTO,
                    "tooltip": "智能切分的写作技能：决定每段提示词的输出结构、分镜格式"
                               "和素材标签写法。auto 先用一次很短的调用按剧本选技能"}),
                "skill_text": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "自定义写作规则，排在所选技能之前，优先级最高"}),
                "vlm_service": (legacy.vlm_service_options(), {
                    "default": "off",
                    "tooltip": "开启后先让 VLM 看一遍每个公共素材，把画面内容写进清单，"
                               "LLM 才能按内容判断每段该引用哪个素材"}),
                # Append-only widget block: keeping these after all existing
                # Director widgets preserves positional values in saved flows.
                "脸部精修开启": ("BOOLEAN", {
                    "default": False, "label_on": "开启小脸精修",
                    "label_off": "关闭小脸精修"}),
                "脸部检测器": (_face_detectors(), {"default": FACE_DETECTOR}),
                "脸部精修步数": ("INT", {"default": 4, "min": 1, "max": 50}),
                "脸部精修重绘": ("FLOAT", {
                    "default": 0.45, "min": 0.01, "max": 1.0, "step": 0.01}),
                "脸部裁剪倍率": ("FLOAT", {
                    "default": 2.5, "min": 1.2, "max": 8.0, "step": 0.1}),
                "脸部身份图序号": ("INT", {
                    "default": 0, "min": 0, "max": 9,
                    "tooltip": "0=自动跟踪最大脸；1～9=用对应 @图片N 辅助锁定人物"}),
                "动作修复开启": ("BOOLEAN", {
                    "default": False, "label_on": "开启高速动作修复",
                    "label_off": "关闭高速动作修复"}),
                "动作修复档位": (["balanced (default)",
                                  "max quality (wide plateau)",
                                  "economy (tight spans)"],
                                 {"default": "balanced (default)"}),
                "动作修复步数": ("INT", {"default": 6, "min": 4, "max": 50}),
                "动作修复注入": ("FLOAT", {
                    "default": 0.70, "min": 0.05, "max": 1.0, "step": 0.05}),
                "多视角分镜开启": ("BOOLEAN", {
                    "default": False, "label_on": "生成角色五视图",
                    "label_off": "不生成角色五视图"}),
                "多视角角色图片序号": ("INT", {
                    "default": 1, "min": 1, "max": 9}),
                "多视角尺寸": (["512", "1024"], {"default": "512"}),
                "多视角步数": ("INT", {"default": 28, "min": 4, "max": 50}),
                "多视角LoRA": (_turnaround_loras(), {"default": TURNAROUND_LORA}),
                "多视角LoRA强度": ("FLOAT", {
                    "default": 0.75, "min": 0.0, "max": 2.0, "step": 0.05}),
                "二采复用一采条件": ("BOOLEAN", {
                    "default": True,
                    "label_on": "复用文本/素材条件（推荐）",
                    "label_off": "按二采分辨率重建条件",
                    "tooltip": "开：只复用一采的文本 token 与已编码参考素材，"
                               "不包含、不复制640P一采成片；二采目标latent始终独立。"
                               "还省掉每段一次文本编码和参考素材VAE编码。"
                               "关：按二采分辨率重跑条件，参考图会被重采样到更大面积，"
                               "token 全变，低降噪几步收不过去，容易涂抹和轻微身份漂移"}),
                # Append-only audio block.  Never insert these above existing
                # widgets: ComfyUI restores saved widget values positionally.
                "音频精修开启": ("BOOLEAN", {
                    "default": False,
                    "label_on": "开启 H3 音频精修",
                    "label_off": "关闭 H3 音频精修",
                    "tooltip": "冻结视频 latent，仅用未挂 Turbo LoRA 的基模附加去噪音频。"
                               "每一步仍接近一次完整 H3 前向计算"}),
                "音频精修步数": ("INT", {
                    "default": 4, "min": 1, "max": 100}),
                "音频去噪强度": ("FLOAT", {
                    "default": 0.5, "min": 0.01, "max": 1.0, "step": 0.01,
                    "tooltip": "音频重新加噪后再修复的深度。数值越大，改动越明显；"
                               "不是画面重绘，也不会开放视频 latent"}),
                "音频精修采样器": (["euler", "res_multistep"], {
                    "default": "euler"}),
                "音频精修调度器": (["simple", "beta", "normal"], {
                    "default": "simple"}),
                "音频接缝平滑": ("BOOLEAN", {
                    "default": True,
                    "label_on": "平滑段间声音（推荐）",
                    "label_off": "保留硬切",
                    "tooltip": "用下一段被裁掉的重叠锚点音频融合上一段尾部；"
                               "不重复声音，不改变成片时长"}),
                "音频接缝时长": ("FLOAT", {
                    "default": 80.0, "min": 0.0, "max": 500.0, "step": 5.0,
                    "tooltip": "建议 60～120ms；对白密集可缩短，环境声可适当加长"}),
                # Append-only resume gate.  The old integer remains in its
                # original position so saved workflows keep every value aligned.
                "从指定段开始": ("BOOLEAN", {
                    "default": False,
                    "label_on": "启用断点续跑",
                    "label_off": "从头开始生成",
                    "tooltip": "关闭时无条件从第 1 段开始；开启后才读取『起始段』"}),
                # Append-only detail-memory controls.  Never move them above
                # existing widgets: ComfyUI restores old workflows positionally.
                "二采显存策略": (detail.DETAIL_MEMORY_PROFILES, {
                    "default": detail.DETAIL_MEMORY_AUTO,
                    "tooltip": "16GB建议自动平衡；DynamicVRAM下数值表示留给二采激活"
                               "的容量，并非永久空置显存；避免Windows共享显存换页，"
                               "不降低832P最终输出质量"}),
                "二采自定义显存预留": ("FLOAT", {
                    "default": 1.25, "min": 0.0, "max": 8.0, "step": 0.05,
                    "tooltip": "仅自定义档生效，单位GB；DynamicVRAM下是激活空间，"
                               "非动态加载器下是传统显存预留；0=沿用ComfyUI启动设置"}),
                "二采自定义预览间隔": ("INT", {
                    "default": 2, "min": 0, "max": 100,
                    "tooltip": "仅自定义档生效；0=关闭二采逐步清晰预览，"
                               "1=每步，2=每2步；最终预览始终保留"}),
                # Append-only experimental option; older workflow widget
                # positions must remain unchanged.
                "二采连续Sigma": ("BOOLEAN", {
                    "default": False,
                    "label_on": "连续 Sigma（实验）",
                    "label_off": "独立二采（默认）",
                    "tooltip": "把一采和二采步数组成一条连续噪声轨迹；"
                               "不增加总步数。实验模式复用一采模型/采样器/调度器，"
                               "只支持 neural_3d 放大或同分辨率二采。"}),
            },
            "optional": {
                "media": ("MINIMAX_H3_MEDIA",),
                "ref_video": ("IMAGE",),
                "ref_audio": ("AUDIO",),
                "前段视频": ("IMAGE", {
                    "tooltip": "断点续跑时上一段已生成的成片。若起始分镜为『承接』，"
                               "只用其结尾做上下文锚点，不会作为 ref2va 参考视频；"
                               "起始分镜为『切镜』时不需要连接"}),
                "前段音频": ("AUDIO", {
                    "tooltip": "可选：上一段成片的音轨，用于声音接缝"}),
                "二采模型": ("MODEL", {
                    "tooltip": "导演台二采使用的 Ref2VA 基模；接 Turbo LoRA 之前的模型"}),
                # Append-only rough-cut block.  Keeping it after every legacy
                # socket/widget preserves the positional values of old saved
                # workflows.  The editor owns this JSON and does not rebuild
                # the Director form while its playhead moves.
                "粗剪时间轴开启": ("BOOLEAN", {
                    "default": False,
                    "label_on": "生成后覆盖写入 I/O 区间",
                    "label_off": "不使用粗剪时间轴",
                    "tooltip": "开启后按粗剪工程的时长模式生成；可沿用导演台时长自动补齐另一端，"
                               "也可由时间轴 I/O 覆盖本次生成时长；成片精确覆盖该区间"}),
                "粗剪工程": ("STRING", {
                    "multiline": True,
                    "default": roughcut_timeline.project_json(None),
                    "tooltip": "粗剪工作区的结构化工程数据；请用导演台里的『打开时间轴』编辑"}),
                "动作迁移自动分段": ("BOOLEAN", {
                    "default": True,
                    "label_on": "按时长自动分段",
                    "label_off": "整段生成",
                    "tooltip": "仅动作迁移生效；关闭后自动匹配动作视频长度，只生成一段"}),
                "一采显存策略": (legacy.FIRST_MEMORY_PROFILES, {
                    "default": legacy.FIRST_MEMORY_AUTO,
                    "tooltip": "16GB推荐自动平衡：条件编码完成后卸载CLIP/VAE，"
                               "采样时不逐步调用视频VAE；每段采样完成仍显示清晰预览。"
                               "兼容模式保留原来的每步清晰预览。"}),
                "一采断点模式": (legacy.PASS1_CHECKPOINT_MODES, {
                    "default": legacy.PASS1_CHECKPOINT_OFF,
                    "tooltip": "保存模式会按分段前缀保存完整音画latent；读取模式按"
                               "原段号直接进入二采。成片兼容入口只支持单段。"}),
                "一采成片": ("IMAGE", {
                    "tooltip": "兼容入口：加载一段已保存的一采视频，内部VAE重编码后直接二采"}),
                "一采成片音频": ("AUDIO", {
                    "tooltip": "可选：与一采成片配套的原始音频"}),
                # Append-only: post-decode 1:1 enhancement. It is not part of
                # 二采放大方式 on purpose -- that widget picks how to resize,
                # and 同分辨率二采 has no resize step for it to act on.
                "二采后VSR增强": ("BOOLEAN", {
                    "default": False,
                    "label_on": "二采后 VSR 增强（原尺寸）",
                    "label_off": "关闭 VSR 增强",
                    "tooltip": "仅『同分辨率二采』生效：解码后按原尺寸跑一遍 NVIDIA RTX "
                               "VSR，只去噪锐化，不改分辨率。需要 nvvfx 与 NVIDIA 显卡；"
                               "每段一次隔离进程加一整段磁盘往返、逐帧推理。"
                               "『放大 + 二采』和『仅放大』请用放大方式里的 VSR"}),
                # Append-only: layered prompt authoring for the Agent path.
                "分层提示词": ("BOOLEAN", {
                    "default": False,
                    "label_on": "分层生成（主体/时间轴/声音/台词）",
                    "label_off": "整段生成（传统）",
                    "tooltip": "仅『Agent / 长剧本智能切分』生效：每段的声音与台词各自"
                               "单独生成，篇幅短、更少报错、也更好逐层审查；台词按本段"
                               "真实秒数换算成硬性字数上限，放不下的顺延到下一段，"
                               "不会再把一整段对话塞进一个 8 秒镜头"}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    def run(self, h3, model, sampler, source_mode, timeline_json,
            script_fallback, total_seconds, segment_seconds, llm_enabled,
            llm_service, task_mode, resolution, aspect_ratio, width, height,
            steps, denoise, scheduler, noise_seed, context_length,
            ref_image_size, save_segments, segment_prefix, save_raw_segments,
            media=None, ref_video=None, ref_audio=None, **kwargs):
        from comfy_execution.graph_utils import GraphBuilder

        graph = GraphBuilder()
        progress_owner = str(kwargs.get("unique_id") or "")
        _send_director_preparation(
            progress_owner, "已进入导演台 · 校验模式、分镜卡与运行参数")
        # These kwargs keep already-queued API prompts usable after the redundant
        # sockets disappeared from INPUT_TYPES. New Director nodes have one model
        # input and one steps control.
        recommended_steps_input = kwargs.get("Turbo推荐一采步数")
        if recommended_steps_input is not None:
            steps = int(recommended_steps_input)
        turbo_model = kwargs.get("Turbo联合模型")
        active_model = turbo_model if turbo_model is not None else model
        turbo_spec = turbo.turbo_metadata(active_model)
        if turbo_model is not None and turbo_spec is None:
            raise ValueError(
                "导演台的『Turbo联合模型』没有 Myang Turbo 调度标记；"
                "请连接『沐阳 H3 · Turbo LoRA 联合音画加载调度』的模型输出")
        if turbo_spec is not None:
            # Turbo MODEL 只负责 LoRA/Shift，NFE 来自导演台自己的 steps。
            # 上面的旧 kwargs 只用于接住更新前已经排队的提示。
            steps = int(steps)
            denoise = 1.0
            scheduler = "simple"

        source_text = str(kwargs.get("script") or script_fallback or "").strip()
        overlap = int(context_length)
        task = str(task_mode)
        roughcut_enabled = _input_enabled(
            kwargs.get("粗剪时间轴开启", False))
        roughcut_json = str(kwargs.get("粗剪工程") or "")
        roughcut_project = None
        roughcut_contract = None
        roughcut_target_frames = 0
        if roughcut_enabled:
            try:
                roughcut_project = roughcut_timeline.normalize_project(
                    roughcut_json)
            except (TypeError, ValueError) as error:
                # A stale browser schema can shift one of the newer optional
                # widget values into this slot.  Losing a whole generation is
                # worse than skipping an editor feature that has no usable
                # project.  Keep the saved text untouched so opening the
                # timeline can still repair it on the client.
                logger.warning(
                    "H3-Myang: 粗剪时间轴已请求开启，但工程数据无效；"
                    "本次安全跳过粗剪写回，不阻断视频生成 | %s", error)
                roughcut_enabled = False
                roughcut_project = None
        if (roughcut_enabled and roughcut_project is not None
                and roughcut_project.get("settings", {}).get(
                    "writeback_enabled") is not True):
            logger.warning(
                "H3-Myang: 粗剪隐藏开关仍为开启，但当前时间轴工程未授权写回；"
                "本次安全跳过粗剪截断与成片写回。请在导演台重新勾选粗剪时间轴后再运行")
            roughcut_enabled = False
            roughcut_project = None
        if roughcut_enabled and roughcut_project is not None:
            roughcut_json = roughcut_timeline.project_json(roughcut_project)
            roughcut_contract = roughcut_timeline.selection_contract(roughcut_project)
            if not bool(roughcut_contract.get("ready", False)):
                if roughcut_contract.get("duration_mode") == "timeline":
                    raise ValueError(
                        "粗剪时间轴使用『时间轴 I/O 时长』，请分别设置入点 I 和出点 O")
                raise ValueError(
                    "粗剪时间轴使用『导演台时长』，请设置入点 I 或出点 O；另一端会自动匹配")
            total_seconds = float(roughcut_contract["seconds"])
            roughcut_target_frames = max(1, round(total_seconds * 24.0))

        # 导演台自带的公共素材并进素材包，走的就是 Media Agent 那条路：
        # 拆分器因此能在清单里看到它们并逐段分配标签，运行时也按同一套编号解析。
        global_assets = _timeline_globals(timeline_json, task)
        # A connected Media Agent is a single shared bundle, while the
        # Director timeline is mode-scoped.  Feeding that bundle into action
        # transfer or continuation reintroduces old assets from another mode.
        # Pure generation keeps the legacy Media Agent integration; the other
        # modes use only their selected Director bucket.
        media_link = media if task == legacy.TASK_FRESH else None
        if media is not None and task != legacy.TASK_FRESH:
            logger.info(
                "H3-Myang: %s模式隔离，忽略外接 Media Agent 素材包；"
                "仅使用当前模式导演台素材", task)
        if global_assets:
            if task in (legacy.TASK_TRANSFER, legacy.TASK_CONTINUE) and any(
                    asset["kind"] == "video" for asset in global_assets):
                raise ValueError(
                    "%s只允许左侧 ref_video 这一条视频输入；"
                    "请移除公共素材里的视频，图片和音频可以保留" %
                    ("动作迁移" if task == legacy.TASK_TRANSFER else "视频续写"))
            global_inputs = {
                "assets_json": json.dumps(global_assets, ensure_ascii=False),
                "required_frames": core.length_for(float(segment_seconds), 24.0),
                "asset_mode": "叠加全局素材",
            }
            if media is not None and task == legacy.TASK_FRESH:
                global_inputs["media"] = media
            media_link = graph.node("H3ShotMedia", **global_inputs).out(0)

        mode_ref_video = ref_video
        mode_ref_audio = ref_audio
        resume_supported = (
            task == legacy.TASK_TRANSFER or
            str(source_mode) != DIRECTOR_SCRIPT)
        resume_enabled = (
            resume_supported and bool(kwargs.get("从指定段开始", False)))
        requested_start = max(1, int(kwargs.get("起始段", 1) or 1))
        start_segment = requested_start if resume_enabled else 1
        resume_video = kwargs.get("前段视频")
        resume_audio = kwargs.get("前段音频")
        if not resume_enabled:
            if requested_start > 1 or resume_video is not None:
                logger.info(
                    "H3-Myang: %s，忽略保存的起始段 %d%s，"
                    "本次从第 1 段生成",
                    "Agent 智能切分固定从头生成" if not resume_supported
                    else "未勾选『从指定段开始』",
                    requested_start,
                    "和前段视频" if resume_video is not None else "")
            resume_video = resume_audio = None
        elif start_segment == 1:
            # 勾选后仍选择 1 与完整生成等价；不要让旧接线意外改变首段。
            resume_video = None
            resume_audio = None
        action_resume_offset = 0
        action_planning_frames = roughcut_target_frames
        template_contract = _timeline_template_contract(timeline_json, task)
        if (task == legacy.TASK_TRANSFER and action_planning_frames <= 0
                and template_contract.get("total_seconds", 0) > 0):
            action_planning_frames = max(
                1, round(float(template_contract["total_seconds"]) * 24.0))
        if roughcut_target_frames > 0 and start_segment > 1:
            action_segment_frames = core.length_for(float(segment_seconds), 24.0)
            action_hop = action_segment_frames - overlap
            action_resume_offset = (start_segment - 1) * action_hop
            action_planning_frames += action_resume_offset
        if resume_video is not None and int(resume_video.shape[0]) < overlap:
            raise ValueError(
                "『前段视频』只有 %d 帧，短于段间锚点 %d 帧；"
                "请接完整的上一段成片" % (int(resume_video.shape[0]), overlap))
        if task == legacy.TASK_TRANSFER:
            # The external Media Agent bundle is intentionally ignored for
            # this mode, so stale videos in that bundle must not trigger a
            # validation error either.
            auto_segment = _input_enabled(
                kwargs.get("动作迁移自动分段", True))
            uploaded_videos = _action_video_assets(timeline_json, task)
            if len(uploaded_videos) > 1:
                raise ValueError("动作迁移导演台最多只能上传一个动作参考视频")
            if ref_video is not None and uploaded_videos:
                raise ValueError(
                    "动作参考视频同时存在『导演台上传』和『左侧 ref_video』两路；"
                    "请只保留其中一个")
            if ref_video is None:
                if not uploaded_videos:
                    raise ValueError("动作迁移需要在导演台上传一个动作参考视频")
                action_source = graph.node(
                    "H3DirectorActionSource", timeline_json=timeline_json,
                    fallback_prompt=source_text,
                    segment_seconds=float(segment_seconds),
                    overlap_frames=overlap, start_segment=1,
                    target_frames=action_planning_frames,
                    auto_segment=auto_segment,
                    minimum_source_frame=action_resume_offset,
                    resolution=str(kwargs.get(
                        "参考视频分辨率", director_media.REFERENCE_VIDEO_ORIGINAL)),
                    width=int(kwargs.get("参考视频自定义宽", 1920)),
                    height=int(kwargs.get("参考视频自定义高", 1080)))
                plan_source = action_source.out(0)
                mode_ref_video = action_source.out(1)
                if ref_audio is None:
                    mode_ref_audio = action_source.out(2)
            else:
                if (action_planning_frames > 0
                        and int(ref_video.shape[0]) < action_planning_frames):
                    raise ValueError(
                        "粗剪 I/O / 模板时长与断点共需要 %d 帧动作参考，"
                        "但左侧动作视频只有 %d 帧；请缩短目标时长、提前断点或换更长的视频" %
                        (action_planning_frames, int(ref_video.shape[0])))
                if (action_resume_offset > 0
                        and int(ref_video.shape[0]) <= action_resume_offset):
                    raise ValueError(
                        "动作迁移断点位于参考视频第 %d 帧，但动作源只有 %d 帧" %
                        (action_resume_offset, int(ref_video.shape[0])))
                plan = _single_prompt_transfer_plan(
                    timeline_json, overlap, float(segment_seconds),
                    action_planning_frames or int(ref_video.shape[0]), source_text,
                    start_segment=1, auto_segment=auto_segment)
                logger.info(
                    "H3-Myang: 外接动作参考视频已解析 | %d 帧（%.2f 秒）| "
                    "单段上限 %.2f 秒 | 自动分段=%s | 计划 %d 段",
                    int(ref_video.shape[0]), int(ref_video.shape[0]) / 24.0,
                    float(segment_seconds), "开启" if auto_segment else "关闭",
                    int(plan.get("total_segments_planned") or
                        plan.get("segment_count") or 1))
                plan_source = json.dumps(plan, ensure_ascii=False)
        elif str(source_mode) == DIRECTOR_SCRIPT:
            if task == legacy.TASK_CONTINUE:
                if ref_video is None:
                    raise ValueError("视频续写需要把前文视频接到左侧 ref_video")
            splitter_inputs = dict(
                script=source_text, total_seconds=float(total_seconds),
                length_source=legacy.LENGTH_MANUAL,
                segment_seconds=float(segment_seconds),
                overlap_frames=overlap, fps=24.0,
                llm_service=llm_service, max_segments=legacy.MAX_SLOTS,
                ollama_auto_unload=True, use_cache=True,
                seed=int(noise_seed), llm_enabled=bool(llm_enabled),
                skill_preset=str(kwargs.get(
                    "skill_preset", legacy.SKILL_PRESET_AUTO)),
                skill_text=str(kwargs.get("skill_text", "")),
                vlm_service=str(kwargs.get("vlm_service", "off")),
                **{"分层提示词": _input_enabled(
                    kwargs.get("分层提示词", False))})
            if media_link is not None:
                splitter_inputs["media"] = media_link
            splitter = graph.node("H3ScriptSplitter", **splitter_inputs)
            plan_source = splitter.out(0)
        else:
            plan = _timeline_plan(timeline_json, overlap, source_text, task)
            if task == legacy.TASK_CONTINUE:
                if ref_video is None:
                    raise ValueError("视频续写需要把前文视频接到左侧 ref_video")
                _reject_director_video_references(plan, "视频续写")
            plan_text = json.dumps(plan, ensure_ascii=False)
            plan_source = plan_text
        if task in (legacy.TASK_TRANSFER, legacy.TASK_CONTINUE) and mode_ref_video is not None:
            reference_resize = graph.node(
                "H3ReferenceResize", image=mode_ref_video,
                resolution=str(kwargs.get(
                    "参考视频分辨率", director_media.REFERENCE_VIDEO_ORIGINAL)),
                width=int(kwargs.get("参考视频自定义宽", 1920)),
                height=int(kwargs.get("参考视频自定义高", 1080)))
            mode_ref_video = reference_resize.out(0)

        turnaround_enabled = bool(kwargs.get("多视角分镜开启", False))
        if turnaround_enabled:
            if media_link is None:
                raise ValueError(
                    "已开启角色五视图，但没有公共图片素材。"
                    "请在导演台公共素材或 Media Agent 中添加一张清晰角色图")
            try:
                import nodes as comfy_nodes
                available_nodes = comfy_nodes.NODE_CLASS_MAPPINGS
            except Exception:
                available_nodes = {}
            required_nodes = {
                "H3ContactSheet", "H3ContactSheetDecode", "LoraLoaderModelOnly",
            }
            missing = sorted(required_nodes.difference(available_nodes))
            if missing:
                raise ValueError(
                    "角色五视图需要 ComfyUI-MAINodes；当前缺少节点：%s" %
                    "、".join(missing))
            turnaround_base = kwargs.get("二采模型") or active_model
            source_image = graph.node(
                "H3DirectorMediaImage", media=media_link,
                image_ordinal=int(kwargs.get("多视角角色图片序号", 1)))
            turnaround_model = graph.node(
                "LoraLoaderModelOnly", model=turnaround_base,
                lora_name=str(kwargs.get("多视角LoRA", TURNAROUND_LORA)),
                strength_model=float(kwargs.get("多视角LoRA强度", 0.75)))
            contact = graph.node(
                "H3ContactSheet", clip=h3.clip, vae=h3.video_vae,
                prompt=("the camera orbits the subject of <Picture 1> "
                        "ninety degrees clockwise"),
                ref_image=source_image.out(0),
                size=int(kwargs.get("多视角尺寸", 512)))
            contact_noise = graph.node(
                "RandomNoise", noise_seed=int(noise_seed))
            contact_sampler = graph.node(
                "KSamplerSelect", sampler_name="res_multistep")
            contact_sigmas = graph.node(
                "BasicScheduler", model=turnaround_model.out(0),
                scheduler="simple", steps=int(kwargs.get("多视角步数", 28)),
                denoise=1.0)
            contact_guider = graph.node(
                "BasicGuider", model=turnaround_model.out(0),
                conditioning=contact.out(0))
            contact_sample = graph.node(
                "SamplerCustomAdvanced", noise=contact_noise.out(0),
                guider=contact_guider.out(0), sampler=contact_sampler.out(0),
                sigmas=contact_sigmas.out(0), latent_image=contact.out(1))
            contact_decode = graph.node(
                "H3ContactSheetDecode", vae=h3.video_vae,
                samples=contact_sample.out(0))
            augmented = graph.node(
                "H3DirectorTurnaroundMedia", media=media_link,
                sheet=contact_decode.out(1), plan_json=plan_source,
                owner_id=str(kwargs.get("unique_id") or ""))
            media_link = augmented.out(0)
            plan_source = augmented.out(1)

        # Manual storyboard cards and action transfer converge here.  Agent
        # splitting deliberately never enables this branch because a newly
        # planned set of shots has no stable relationship to an older run's
        # segment numbers.  The checkbox is the only gate, so a stale integer
        # in an older workflow is harmless while it remains unchecked.
        if resume_enabled and start_segment > 1:
            slice_inputs = {
                "plan_json": plan_source,
                "start_segment": start_segment,
                "overlap_frames": overlap,
            }
            if resume_video is not None:
                slice_inputs["context_video"] = resume_video
            plan_source = graph.node(
                "H3DirectorPlanSlice", **slice_inputs).out(0)

        # Enforce the I/O duration after every planning/resume path converges.
        # This prevents a long manual storyboard or action source from wasting
        # minutes of generation only to be truncated by the final writer.
        if roughcut_target_frames > 0:
            plan_source = graph.node(
                "H3DirectorPlanLimit", plan_json=plan_source,
                target_frames=roughcut_target_frames,
                overlap_frames=overlap).out(0)

        literal = graph.node(
            "H3DirectorPlanValue", plan_json=plan_source,
            progress_owner=str(kwargs.get("unique_id") or ""))
        plan_link = literal.out(0)

        legacy_detail = kwargs.get("二采设置")
        detail_input = legacy_detail
        integrated_detail = bool(kwargs.get("二采开启", False))
        if legacy_detail is None and integrated_detail:
            detail_model = kwargs.get("二采模型")
            mode = str(kwargs.get("二采模式", detail.DETAIL_MODE_UPSCALE_REFINE))
            continuous_sigma = bool(kwargs.get("二采连续Sigma", False))
            if (mode != detail.DETAIL_MODE_UPSCALE_ONLY
                    and not continuous_sigma and detail_model is None):
                raise ValueError(
                    "导演台已开启二采，但『二采模型』没接。请连接 Turbo LoRA 之前的 Ref2VA 基模")
            detail_node_inputs = {
                "enabled": True,
                "mode": mode,
                "resolution": str(kwargs.get("二采分辨率", "832P")),
                "width": int(kwargs.get("二采自定义宽", 1664)),
                "height": int(kwargs.get("二采自定义高", 928)),
                "steps": int(kwargs.get("二采步数", 4)),
                "denoise": float(kwargs.get("二采重绘幅度", 0.2)),
                "scheduler": str(kwargs.get("二采调度器", "beta")),
                "sampler_name": str(kwargs.get("二采采样器", "res_multistep")),
                "upscale_method": str(kwargs.get(
                    "二采放大方式", "neural_3d (神经3D Latent放大·推荐)")),
                "chunk_frames": int(kwargs.get("二采分块帧数", 4)),
                "latent_upscale_model": str(kwargs.get("二采Latent模型", "")),
                "latent_precision": str(kwargs.get(
                    "二采精度", detail.LATENT_PRECISIONS[0])),
                "latent_chunk_steps": int(kwargs.get("二采时间分块", 0)),
                "passes": int(kwargs.get("二采轮数", 1)),
                "seed_mode": str(kwargs.get(
                    "二采种子策略", detail.DETAIL_SEED_INHERIT)),
                "reuse_condition": bool(
                    kwargs.get("二采复用一采条件", True)),
                "memory_profile": str(kwargs.get(
                    "二采显存策略", detail.DETAIL_MEMORY_AUTO)),
                "custom_reserve_gb": float(kwargs.get(
                    "二采自定义显存预留", 1.25)),
                "custom_preview_interval": int(kwargs.get(
                    "二采自定义预览间隔", 2)),
                "continuous_sigma": continuous_sigma,
                "vsr_enhance": _input_enabled(
                    kwargs.get("二采后VSR增强", False)),
            }
            if detail_model is not None:
                detail_node_inputs["二采模型"] = detail_model
            detail_builder = graph.node("H3DetailSettings", **detail_node_inputs)
            detail_input = detail_builder.out(0)

        face_enabled = bool(kwargs.get("脸部精修开启", False))
        motion_enabled = bool(kwargs.get("动作修复开启", False))
        enhancement_input = None
        if face_enabled or motion_enabled:
            enhancement_model = kwargs.get("二采模型") or active_model
            enhancement_builder = graph.node(
                "H3DirectorEnhancementSettings",
                face_enabled=face_enabled,
                face_detector=str(kwargs.get("脸部检测器", FACE_DETECTOR)),
                face_steps=int(kwargs.get("脸部精修步数", 4)),
                face_denoise=float(kwargs.get("脸部精修重绘", 0.45)),
                face_crop_factor=float(kwargs.get("脸部裁剪倍率", 2.5)),
                face_identity_ordinal=int(kwargs.get("脸部身份图序号", 0)),
                motion_enabled=motion_enabled,
                motion_preset=str(kwargs.get(
                    "动作修复档位", "balanced (default)")),
                motion_steps=int(kwargs.get("动作修复步数", 6)),
                motion_inject=float(kwargs.get("动作修复注入", 0.70)),
                model=enhancement_model)
            enhancement_input = enhancement_builder.out(0)

        audio_refine_enabled = bool(kwargs.get("音频精修开启", False))
        audio_model = kwargs.get("二采模型")
        if audio_model is None and turbo_spec is None:
            audio_model = active_model
        if audio_refine_enabled and audio_model is None:
            raise ValueError(
                "导演台已开启音频精修，但一采使用 Turbo，且『二采 Ref2VA 基模』"
                "没有连接。请把 Turbo LoRA 之前的基模接到该输入；音频精修会复用它")
        if (audio_refine_enabled and audio_model is not None and
                turbo.turbo_metadata(audio_model) is not None):
            raise ValueError("音频精修必须使用 Turbo LoRA 之前的基模")
        audio_node_inputs = {
            "refine_enabled": audio_refine_enabled,
            "refine_steps": int(kwargs.get("音频精修步数", 4)),
            # Accept the short-lived old name from queued API prompts while
            # exposing the accurate audio-specific term in new workflows.
            "refine_denoise": float(kwargs.get(
                "音频去噪强度", kwargs.get("音频精修重绘", 0.5))),
            "refine_sampler": str(kwargs.get("音频精修采样器", "euler")),
            "refine_scheduler": str(kwargs.get("音频精修调度器", "simple")),
            "seam_enabled": bool(kwargs.get("音频接缝平滑", True)),
            "seam_ms": float(kwargs.get("音频接缝时长", 80.0)),
        }
        if audio_model is not None:
            audio_node_inputs["model"] = audio_model
        audio_builder = graph.node("H3AudioSettings", **audio_node_inputs)
        audio_input = audio_builder.out(0)

        long_inputs = dict(
            h3=h3, model=active_model, sampler=sampler, plan_json=plan_link,
            task_mode=task_mode, resolution=resolution,
            aspect_ratio=aspect_ratio, width=int(width), height=int(height),
            steps=int(steps), denoise=float(denoise), scheduler=scheduler,
            noise_seed=int(noise_seed), context_length=str(context_length),
            prompt_mode=legacy.MODE_DIRECT, media_prefix="",
            legacy_plan_padding="",
            ref_image_size=ref_image_size,
            detail_refinement=legacy.DETAIL_NATIVE,
            save_segments=bool(save_segments), segment_prefix=segment_prefix,
            save_raw_segments=bool(save_raw_segments) and
            (integrated_detail or legacy_detail is not None),
            **{
                "一采显存策略": str(kwargs.get(
                    "一采显存策略", legacy.FIRST_MEMORY_AUTO)),
                "一采断点模式": str(kwargs.get(
                    "一采断点模式", legacy.PASS1_CHECKPOINT_OFF)),
            })
        if roughcut_contract is not None:
            long_inputs["roughcut_exact_duration"] = True
        mode_inputs = [("media", media_link), ("二采设置", detail_input),
                       ("增强设置", enhancement_input),
                       ("音频设置", audio_input)]
        if task in (legacy.TASK_TRANSFER, legacy.TASK_CONTINUE):
            mode_inputs.extend((("ref_video", mode_ref_video),
                                ("ref_audio", mode_ref_audio)))
        # 断点续跑的上一段成片只做段间锚点上下文，绝不进 ref2va 参考视频通道。
        mode_inputs.extend((("context_video", resume_video),
                            ("context_audio", resume_audio)))
        mode_inputs.extend((("一采成片", kwargs.get("一采成片")),
                            ("一采成片音频", kwargs.get("一采成片音频"))))
        if roughcut_contract is not None:
            start_mode = str(roughcut_contract.get("start_mode") or "off")
            end_mode = str(roughcut_contract.get("end_mode") or "off")
            use_first = start_mode == "frame"
            use_last = end_mode == "frame"
            use_head_motion = start_mode == "motion"
            use_tail_motion = end_mode == "motion"
            if use_first or use_last or use_head_motion or use_tail_motion:
                boundary = graph.node(
                    "H3RoughCutBoundaryFrames", project_json=roughcut_json,
                    load_first=use_first, load_last=use_last,
                    load_head_motion=use_head_motion,
                    load_tail_motion=use_tail_motion,
                    context_frames=overlap)
                if use_first:
                    mode_inputs.append(("timeline_first_keyframe", boundary.out(0)))
                if use_last:
                    mode_inputs.append(("timeline_last_keyframe", boundary.out(1)))
                if use_head_motion:
                    mode_inputs.extend((
                        ("timeline_head_context", boundary.out(2)),
                        ("timeline_head_audio", boundary.out(3)),
                    ))
                if use_tail_motion:
                    mode_inputs.extend((
                        ("timeline_tail_context", boundary.out(4)),
                        ("timeline_tail_audio", boundary.out(5)),
                    ))
        for name, value in mode_inputs:
            if value is not None:
                long_inputs[name] = value
        long_video = graph.node("H3LongVideo", **long_inputs)
        _send_director_preparation(
            progress_owner,
            "导演台执行链已建立 · 等待分镜计划、素材绑定与条件编码")
        output_images, output_audio = long_video.out(0), long_video.out(1)
        if roughcut_contract is not None:
            roughcut_save = graph.node(
                "H3RoughCutSave", images=output_images, audio=output_audio,
                fps=24.0, project_json=roughcut_json,
                filename_prefix=f"{segment_prefix}_粗剪成片",
                owner_id=str(kwargs.get("unique_id") or ""),
                writeback_enabled=True)
            output_images, output_audio = roughcut_save.out(0), roughcut_save.out(1)
        return {
            "expand": graph.finalize(),
            "result": (output_images, output_audio,
                       plan_link, literal.out(1)),
        }


NODE_CLASS_MAPPINGS = {
    "H3Director": H3Director,
    "H3DirectorPlanValue": H3DirectorPlanValue,
    "H3DirectorPlanSlice": H3DirectorPlanSlice,
    "H3DirectorPlanLimit": H3DirectorPlanLimit,
    "H3DirectorActionSource": H3DirectorActionSource,
    "H3DirectorMediaImage": H3DirectorMediaImage,
    "H3DirectorTurnaroundMedia": H3DirectorTurnaroundMedia,
    "H3DirectorEnhancementSettings": H3DirectorEnhancementSettings,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3Director": "沐阳 H3 · 导演台（全功能）",
    "H3DirectorPlanValue": "沐阳 H3 · 导演台计划（内部）",
    "H3DirectorPlanSlice": "沐阳 H3 · 断点分段裁切（内部）",
    "H3DirectorPlanLimit": "沐阳 H3 · 粗剪时长拟合（内部）",
    "H3DirectorActionSource": "沐阳 H3 · 导演台动作源（内部）",
    "H3DirectorMediaImage": "沐阳 H3 · 选择角色参考图（内部）",
    "H3DirectorTurnaroundMedia": "沐阳 H3 · 五视图素材绑定（内部）",
    "H3DirectorEnhancementSettings": "沐阳 H3 · 画质与分镜增强（内部）",
}
