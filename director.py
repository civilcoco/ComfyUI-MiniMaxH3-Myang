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

from comfy_api.latest import InputImpl

from . import core
from . import detail
from . import media as director_media
from . import nodes as legacy
from . import nodes_v2
from . import turbo


logger = logging.getLogger(__name__)


DIRECTOR_TIMELINE = "导演台分镜卡（手动逐镜头）"
DIRECTOR_SCRIPT = "Agent / 长剧本智能切分"
DIRECTOR_SOURCES = [DIRECTOR_TIMELINE, DIRECTOR_SCRIPT]

DEFAULT_TIMELINE = {
    "version": 2,
    "shots": [{
        "id": "shot_1",
        "enabled": True,
        "duration_seconds": 5.0,
        "brief": "镜头 1",
        "prompt": "",
        "asset_mode": "仅本镜头",
        "assets": [],
    }],
}


def _timeline_shots(raw) -> list[dict]:
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
    except json.JSONDecodeError as error:
        raise ValueError("导演台分镜数据不是有效 JSON：%s" % error) from error
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
        assets.append({
            "id": str(raw.get("id") or "%s_%d" % (kind, counts[kind])),
            "kind": kind,
            "role": role,
            "label": str(raw.get("label") or name),
            "file": {
                "name": name,
                "subfolder": str(file_ref.get("subfolder") or ""),
                "type": "input",
            },
        })
    if action_count > 1:
        raise ValueError("%s只能指定一个动作源视频" % label)
    return assets


def _shot_assets(shot: dict, shot_index: int) -> tuple[str, list[dict]]:
    assets = _normalize_assets(shot.get("assets") or [], "分镜 %d " % shot_index)
    mode = "叠加全局素材" if str(shot.get("asset_mode")) == "叠加全局素材" else "仅本镜头"
    return mode, assets


def _timeline_globals(raw) -> list[dict]:
    """Materials authored on the Director itself and shared by every segment.

    These land in the same bundle an external Media Agent would produce, so the
    splitter's LLM sees them in its manifest and can hand each segment the tags
    it actually needs.
    """

    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
    except json.JSONDecodeError as error:
        raise ValueError("导演台分镜数据不是有效 JSON：%s" % error) from error
    if not isinstance(data, dict):
        return []
    return _normalize_assets(
        data.get("global_assets") or [], "公共素材", allow_action=False)


def _timeline_plan(raw, overlap: int, fallback_prompt: str = "") -> dict:
    shots = _timeline_shots(raw)

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
        asset_mode, assets = _shot_assets(shot, index)
        segments.append({
            "index": index,
            "id": str(shot.get("id") or "shot_%d" % index),
            "brief": brief,
            "prompt": prompt,
            "duration_seconds": frames / 24.0,
            "frames": frames,
            "asset_mode": asset_mode,
            "assets": assets,
        })
        frame_counts.append(frames)

    total_frames = sum(frame_counts) - overlap * max(0, len(frame_counts) - 1)
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


def _aligned_frames_up(frame_count: int) -> int:
    frame_count = max(5, int(frame_count))
    return max(5, math.ceil((frame_count - 5) / 17) * 17 + 5)


def _single_prompt_transfer_plan(raw, overlap: int, segment_seconds: float,
                                 ref_frames: int, fallback_prompt: str = "",
                                 allow_embedded_video: bool = False,
                                 start_segment: int = 1) -> dict:
    """Split one action source while keeping one prompt and one material set.

    ``start_segment`` drops the already produced head of a run.  The full split
    is still computed first, so every kept segment carries the absolute index
    and the absolute ``ref_start_frame`` it would have had in a full run.
    """

    shot = _timeline_shots(raw)[0]
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
    assets = [asset for asset in assets if asset["kind"] != "video"]
    ref_frames = int(ref_frames)
    if ref_frames <= overlap:
        raise ValueError(
            "动作参考视频只有 %d 帧，必须长于段间锚点 %d 帧" %
            (ref_frames, overlap))
    frames = core.length_for(float(segment_seconds), 24.0)
    if frames <= overlap:
        raise ValueError("动作迁移分段时长太短：每段帧数必须大于段间锚点")
    hop = frames - overlap
    count = 1 if ref_frames <= frames else 1 + math.ceil((ref_frames - frames) / hop)
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

    brief = str(shot.get("brief") or "动作迁移").strip()
    segments = [{
        "index": index,
        "id": "transfer_%d" % index,
        "brief": "%s · 第%d段" % (brief, index),
        "prompt": prompt,
        "duration_seconds": frame_count / 24.0,
        "frames": frame_count,
        "ref_start_frame": (index - 1) * hop,
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
        "segments": kept,
    }


def _action_video_assets(raw) -> list[dict]:
    _, assets = _shot_assets(_timeline_shots(raw)[0], 1)
    return [asset for asset in assets if asset["kind"] == "video"]


def _load_director_action_video(asset: dict):
    path, _ = director_media._safe_input_path(asset)
    value = InputImpl.VideoFromFile(path)
    frames, soundtrack, source_fps = core.video_stream(value)
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
                "brief": str(segment.get("brief") or ""),
                "prompt": str(segment.get("prompt") or ""),
                "transition": str(segment.get("transition") or ""),
                "frames": int(segment.get("frames") or default_frames),
                "duration_seconds": float(
                    segment.get("duration_seconds") or default_seconds),
            })
        instance.send_sync("myh3_director_plan", {
            "owner_id": owner,
            "source": str(plan.get("source") or ""),
            "segment_count": len(segments),
            "style_header": str(plan.get("style_header") or ""),
            "skill_source": str(plan.get("skill_source") or ""),
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
            "overlap_frames": ("INT", {"default": 22, "min": 5, "max": 56}),
            "start_segment": ("INT", {
                "default": 1, "min": 1, "max": legacy.MAX_SLOTS,
                "tooltip": "从第几段开始生成；1 表示整条重跑"}),
        }}

    def load(self, timeline_json, fallback_prompt, segment_seconds, overlap_frames,
             start_segment=1):
        videos = _action_video_assets(timeline_json)
        if len(videos) != 1:
            raise ValueError("动作迁移导演台必须上传且只能上传一个动作参考视频")
        frames, soundtrack = _load_director_action_video(videos[0])
        plan = _single_prompt_transfer_plan(
            timeline_json, int(overlap_frames), float(segment_seconds),
            int(frames.shape[0]), fallback_prompt, allow_embedded_video=True,
            start_segment=int(start_segment))
        return (json.dumps(plan, ensure_ascii=False), frames,
                soundtrack if soundtrack is not None else director_media._empty_audio())


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
                    "tooltip": "智能切分时作为剧本；可把此控件转换为输入后连接 Agent myang_prompt",
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
                    "default": 16, "min": 1, "max": 128}),
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
                    "tooltip": "仅动作迁移：从第几段开始生成。1=整条重跑；"
                               "断点续跑时填未生成的那一段，并把上一段成片接到"
                               "『前段视频』"}),
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
            },
            "optional": {
                "media": ("MINIMAX_H3_MEDIA",),
                "ref_video": ("IMAGE",),
                "ref_audio": ("AUDIO",),
                "前段视频": ("IMAGE", {
                    "tooltip": "动作迁移断点续跑：上一段已生成的成片。只用它的结尾做"
                               "段间锚点上下文，不会作为 ref2va 参考视频"}),
                "前段音频": ("AUDIO", {
                    "tooltip": "可选：上一段成片的音轨，用于声音接缝"}),
                "二采模型": ("MODEL", {
                    "tooltip": "导演台二采使用的 Ref2VA 基模；接 Turbo LoRA 之前的模型"}),
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

        # 导演台自带的公共素材并进素材包，走的就是 Media Agent 那条路：
        # 拆分器因此能在清单里看到它们并逐段分配标签，运行时也按同一套编号解析。
        global_assets = _timeline_globals(timeline_json)
        media_link = media
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
            if media is not None:
                global_inputs["media"] = media
            media_link = graph.node("H3ShotMedia", **global_inputs).out(0)

        mode_ref_video = ref_video
        mode_ref_audio = ref_audio
        start_segment = max(1, int(kwargs.get("起始段", 1) or 1))
        resume_video = kwargs.get("前段视频")
        resume_audio = kwargs.get("前段音频")
        if task != legacy.TASK_TRANSFER:
            # 断点续跑是动作迁移专属的，面板也只在动作迁移下显示这两个控件。
            # 换任务模式后它们是上一次留下的状态，不是用户在这个模式下的意图；
            # 一个看不见、也改不回去的控件绝不能把执行卡死，所以这里只忽略并记一行。
            if start_segment > 1 or resume_video is not None:
                logger.info(
                    "H3-Myang: 当前任务不是动作迁移，忽略断点续跑设置"
                    "（起始段=%d%s）", start_segment,
                    "，前段视频已接线" if resume_video is not None else "")
            start_segment = 1
            resume_video = resume_audio = None
        elif start_segment == 1:
            if resume_video is not None:
                raise ValueError(
                    "『前段视频』已接线，但『起始段』还是 1。"
                    "断点续跑请把起始段改成未生成的那一段；整条重跑请断开前段视频")
            resume_audio = None
        elif resume_video is None:
            raise ValueError(
                "从第 %d 段开始生成，必须把上一段（第 %d 段）已生成的成片接到"
                "『前段视频』；它只作为段间上下文锚点，不会当成参考视频" %
                (start_segment, start_segment - 1))
        if resume_video is not None and int(resume_video.shape[0]) < overlap:
            raise ValueError(
                "『前段视频』只有 %d 帧，短于段间锚点 %d 帧；"
                "请接完整的上一段成片" % (int(resume_video.shape[0]), overlap))
        if task == legacy.TASK_TRANSFER:
            _reject_media_videos(media, "动作迁移")
            uploaded_videos = _action_video_assets(timeline_json)
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
                    overlap_frames=overlap, start_segment=start_segment)
                plan_source = action_source.out(0)
                mode_ref_video = action_source.out(1)
                if ref_audio is None:
                    mode_ref_audio = action_source.out(2)
            else:
                plan = _single_prompt_transfer_plan(
                    timeline_json, overlap, float(segment_seconds),
                    int(ref_video.shape[0]), source_text,
                    start_segment=start_segment)
                plan_source = json.dumps(plan, ensure_ascii=False)
        elif str(source_mode) == DIRECTOR_SCRIPT:
            if task == legacy.TASK_CONTINUE:
                if ref_video is None:
                    raise ValueError("视频续写需要把前文视频接到左侧 ref_video")
                _reject_media_videos(media, "视频续写")
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
                vlm_service=str(kwargs.get("vlm_service", "off")))
            if media_link is not None:
                splitter_inputs["media"] = media_link
            splitter = graph.node("H3ScriptSplitter", **splitter_inputs)
            plan_source = splitter.out(0)
        else:
            plan = _timeline_plan(timeline_json, overlap, source_text)
            if task == legacy.TASK_CONTINUE:
                if ref_video is None:
                    raise ValueError("视频续写需要把前文视频接到左侧 ref_video")
                _reject_director_video_references(plan, "视频续写")
                _reject_media_videos(media, "视频续写")
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
            if mode != detail.DETAIL_MODE_UPSCALE_ONLY and detail_model is None:
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
                "latent_chunk_steps": int(kwargs.get("二采时间分块", 16)),
                "passes": int(kwargs.get("二采轮数", 1)),
                "seed_mode": str(kwargs.get(
                    "二采种子策略", detail.DETAIL_SEED_INHERIT)),
            }
            if detail_model is not None:
                detail_node_inputs["二采模型"] = detail_model
            detail_builder = graph.node("H3DetailSettings", **detail_node_inputs)
            detail_input = detail_builder.out(0)

        long_inputs = dict(
            h3=h3, model=active_model, sampler=sampler, plan_json=plan_link,
            task_mode=task_mode, resolution=resolution,
            aspect_ratio=aspect_ratio, width=int(width), height=int(height),
            steps=int(steps), denoise=float(denoise), scheduler=scheduler,
            noise_seed=int(noise_seed), context_length=str(context_length),
            prompt_mode=legacy.MODE_DIRECT, media_prefix="",
            legacy_plan_padding="",
            ref_image_size=ref_image_size,
            detail_refinement=nodes_v2.DETAIL_NATIVE,
            save_segments=bool(save_segments), segment_prefix=segment_prefix,
            save_raw_segments=bool(save_raw_segments) and
            (integrated_detail or legacy_detail is not None))
        mode_inputs = [("media", media_link), ("二采设置", detail_input)]
        if task in (legacy.TASK_TRANSFER, legacy.TASK_CONTINUE):
            mode_inputs.extend((("ref_video", mode_ref_video),
                                ("ref_audio", mode_ref_audio)))
        # 断点续跑的上一段成片只做段间锚点上下文，绝不进 ref2va 参考视频通道。
        mode_inputs.extend((("context_video", resume_video),
                            ("context_audio", resume_audio)))
        for name, value in mode_inputs:
            if value is not None:
                long_inputs[name] = value
        long_video = graph.node("H3LongVideo", **long_inputs)
        return {
            "expand": graph.finalize(),
            "result": (long_video.out(0), long_video.out(1),
                       plan_link, literal.out(1)),
        }


NODE_CLASS_MAPPINGS = {
    "H3Director": H3Director,
    "H3DirectorPlanValue": H3DirectorPlanValue,
    "H3DirectorActionSource": H3DirectorActionSource,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3Director": "沐阳 H3 · 导演台（全功能）",
    "H3DirectorPlanValue": "沐阳 H3 · 导演台计划（内部）",
    "H3DirectorActionSource": "沐阳 H3 · 导演台动作源（内部）",
}
