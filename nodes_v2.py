"""Native long-video expansion for ComfyUI-MiniMaxH3-Myang.

Public node IDs and widget positions stay identical to ``nodes.py``.  Only the
runtime continuation chain changes: Myang's own temporal anchors replace the
third-party Motion-Context nodes.
"""

import json
import logging
import time

from . import core
from . import detail
from . import nodes as legacy
from .compat_v2 import ensure_anchors


logger = logging.getLogger(__name__)

DETAIL_NATIVE = "关闭（H3原生轨迹）"
DETAIL_BALANCED = "低Sigma精修（均衡·实验）"
DETAIL_STRONG = "低Sigma精修（强化·更慢）"
DETAIL_REFINEMENTS = [DETAIL_NATIVE, DETAIL_BALANCED, DETAIL_STRONG]


def detail_refinement_params(profile):
    """Extra low-noise ODE points; keep H3's trained 12/3 AV shift intact."""
    if str(profile) == DETAIL_BALANCED:
        return {"steps": 2, "start_at_sigma": 0.8,
                "end_at_sigma": 0.0, "spacing": "cosine"}
    if str(profile) == DETAIL_STRONG:
        return {"steps": 3, "start_at_sigma": 0.8,
                "end_at_sigma": 0.0, "spacing": "cosine"}
    return None


def _native_frame_length(seconds, fps):
    return core.length_for(seconds, 24.0)


# legacy.plan_segments looks this global up at call time.  Point it at the same
# official helper used by H3Condition so planning and latent allocation cannot
# disagree on the 17k+5 frame grid.
legacy.frame_length = _native_frame_length


class H3ScriptSplitter(legacy.H3ScriptSplitter):
    DESCRIPTION = "按 H3 官方 17k+5 帧网格分段；MiniMax H3 固定使用 24fps。"

    @classmethod
    def INPUT_TYPES(cls):
        schema = super().INPUT_TYPES()
        schema["required"]["overlap_frames"] = (
            "INT", {
                "default": 22, "min": 0, "max": 240,
                "tooltip": "必须与长视频 context_length 一致；5 是实验速度锚点",
            })
        schema["required"]["fps"] = (
            "FLOAT", {"default": 24.0, "min": 24.0, "max": 24.0, "step": 1.0})
        return schema

    def split(self, script, total_seconds, length_source, segment_seconds,
              overlap_frames, fps, llm_service, max_segments,
              ollama_auto_unload, use_cache, seed, llm_enabled=True,
              detail_boost=legacy.DETAIL_BOOST_NONE, media=None, **kwargs):
        if abs(float(fps) - 24.0) > 1e-6:
            raise ValueError("MiniMax H3 固定按 24fps 建模；分段 fps 必须是 24")
        return super().split(
            script, total_seconds, length_source, segment_seconds,
            overlap_frames, 24.0, llm_service, max_segments,
            ollama_auto_unload, use_cache, seed, llm_enabled=llm_enabled,
            detail_boost=detail_boost, media=media, **kwargs)


class H3ModelFromBundle:
    CATEGORY = "沐阳 H3"
    FUNCTION = "get"
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    DESCRIPTION = "从沐阳 H3 加载器取出当前模型，供注意力或显存补丁链使用。"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "h3": ("MYANG_H3", {"tooltip": "接『沐阳 H3 加载器』"}),
            # Kept only for positional compatibility with early workflows.
            "kind": (["ref2va", "fl2va"], {"default": "ref2va"}),
        }}

    def get(self, h3, kind="ref2va"):
        if hasattr(h3, "model_for"):
            return (h3.model_for(kind),)
        # Backward compatibility for bundles saved before the dual lazy loader.
        if hasattr(h3, "model"):
            return (h3.model,)
        raise ValueError("H3-Myang: 输入不是可识别的 H3 模型包")


class H3LongVideo(legacy.H3LongVideo):
    DESCRIPTION = (
        "原生 H3 多关键帧长视频。22 帧连续窗是稳定默认；"
        "5 帧是两个 temporal latent block 的实验速度锚点。")

    @classmethod
    def INPUT_TYPES(cls):
        schema = super().INPUT_TYPES()
        # H3LongVideo only consumes an already prepared plan and must never own
        # or call an LLM service. Keep one inert STRING at the old positional
        # widget index so legacy ``widgets_values`` arrays do not shift every
        # setting after the removed COMBO.
        schema["required"].pop("llm_service", None)
        required = schema["required"]
        rebuilt = {}
        for name, spec in required.items():
            rebuilt[name] = spec
            if name == "media_prefix":
                rebuilt["legacy_plan_padding"] = (
                    "STRING", {
                        "default": "",
                        "tooltip": "旧工作流迁移占位；不参与提示词、LLM 或采样。",
                    })
        schema["required"] = rebuilt
        # 漂移校正与锚点调度已移除：回归 motion-context 的 latent 无损续接 + 全钉，
        # 段间连续性靠 latent，不靠色彩校正，也不靠选择性地钉部分步。
        schema["required"].pop("drift_method", None)
        schema["required"].pop("drift_strength", None)
        schema["required"]["context_length"] = (
            legacy.CONTEXT_LENGTHS, {
                "default": "22",
                "tooltip": (
                    "必须与分段 overlap_frames 一致。22=稳定基线；"
                    "5=实验速度锚点；39/56=更长连续窗"),
            })
        schema["required"]["detail_refinement"] = (
            DETAIL_REFINEMENTS, {
                "default": DETAIL_NATIVE,
                "tooltip": "在 H3 原生 sigma 轨迹的低噪声末段插入额外积分点。"
                           "均衡约增加 25% 采样步；强化约增加 50%。"
                           "不改变官方视频/音频 shift，也不拆成两次采样",
            })
        schema["required"]["save_raw_segments"] = (
            "BOOLEAN", {
                "default": False,
                "tooltip": "开启二采时，额外保存二采前的原始分段（已裁掉锚点帧），"
                           "文件名追加 _原始，便于和二采后的成片对比。未开二采时此项无效。",
            })
        schema.setdefault("optional", {})["二采设置"] = (
            "MYANG_H3_DETAIL", {
                "tooltip": "接『沐阳 H3 · 二采放大设置』；二采开关和所有参数都由它管理",
            })
        schema["optional"]["context_video"] = (
            "IMAGE", {
                "tooltip": "断点续跑：上一段已生成的成片。只取它的结尾做首段锚点，"
                           "不进入 ref2va 参考视频通道",
            })
        schema["optional"]["context_audio"] = (
            "AUDIO", {
                "tooltip": "可选：与 context_video 配套的音轨，用于声音接缝",
            })
        return schema

    def run(self, h3, model, sampler, plan_json, task_mode, resolution,
            aspect_ratio, width, height, steps, denoise, scheduler,
            noise_seed, context_length, prompt_mode, media_prefix,
            ref_image_size,
            detail_refinement=DETAIL_NATIVE,
            save_segments=True, segment_prefix="video/H3_长视频",
            save_raw_segments=False,
            ref_video=None, ref_audio=None, media=None, **kwargs):
        from comfy_execution.graph_utils import GraphBuilder

        context_video = kwargs.get("context_video")
        context_audio = kwargs.get("context_audio")
        if isinstance(context_audio, dict):
            waveform = context_audio.get("waveform")
            if waveform is not None and int(waveform.shape[-1]) == 0:
                context_audio = None
        resuming = context_video is not None
        prompt = kwargs.get("prompt", "")
        fixed_prompt = str(prompt or "").strip() or None
        if isinstance(ref_audio, dict):
            waveform = ref_audio.get("waveform")
            if waveform is not None and int(waveform.shape[-1]) == 0:
                ref_audio = None
        if not plan_json or str(plan_json).strip() == "":
            prompt_str = fixed_prompt or ""
            plan = {
                "segment_count": 1,
                "frames_per_segment": 125,
                "segment_seconds_snapped": 5.0,
                "overlap_frames": int(context_length),
                "fps": 24.0,
                "style_header": "",
                "segments": [{"index": 1, "brief": prompt_str[:50], "prompt": prompt_str}],
            }
        elif isinstance(plan_json, str):
            plan = json.loads(plan_json)
        else:
            plan = plan_json
        progress_owner = str(plan.get("progress_owner") or "")
        segment_entries = list(plan.get("segments") or [])
        count = int(plan.get("segment_count") or len(segment_entries) or 1)
        default_seconds = float(plan.get("segment_seconds_snapped") or 5.0)
        default_frames = int(
            plan.get("frames_per_segment") or
            core.length_for(default_seconds, 24.0))
        fps = float(plan.get("fps") or 24.0)
        overlap = int(context_length)

        segment_frames = []
        for offset in range(count):
            entry = segment_entries[offset] if offset < len(segment_entries) else {}
            if entry.get("frames") is not None:
                frame_count = int(entry["frames"])
            elif entry.get("duration_seconds") is not None:
                frame_count = core.length_for(
                    float(entry["duration_seconds"]), 24.0)
            elif entry.get("seconds") is not None:
                frame_count = core.length_for(float(entry["seconds"]), 24.0)
            else:
                frame_count = default_frames
            segment_frames.append(frame_count)

        # 分段计划可以显式给出每段在参考视频里的绝对起点（断点续跑时首段不再从
        # 第 0 帧取）。没写就退回连续 hop 累加，和旧计划完全一致。
        ref_starts = []
        cursor = 0
        for offset in range(count):
            entry = segment_entries[offset] if offset < len(segment_entries) else {}
            planned_start = entry.get("ref_start_frame")
            if planned_start is not None:
                cursor = max(0, int(planned_start))
            ref_starts.append(cursor)
            cursor += max(1, segment_frames[offset] - int(context_length))

        from .turbo import sampler_function_name, turbo_metadata
        turbo = turbo_metadata(model)
        if turbo is not None:
            if str(scheduler) != "simple":
                raise ValueError("LightX2V H3 Turbo 官方调度要求 scheduler=simple")
            if abs(float(denoise) - 1.0) > 1e-6:
                raise ValueError("LightX2V H3 Turbo 官方调度要求 denoise=1.0")
            sampler_name = sampler_function_name(sampler)
            if sampler_name and sampler_name != "sample_euler":
                raise ValueError(
                    "LightX2V H3 Turbo 官方 ComfyUI 工作流使用 Euler；"
                    "当前采样器函数是 %s" % sampler_name)
            if detail_refinement_params(detail_refinement) is not None:
                logger.warning(
                    "H3-Myang: Turbo LoRA 已启用，自动忽略『%s』，避免额外插入低 Sigma 精修步",
                    detail_refinement)
                detail_refinement = DETAIL_NATIVE

        if abs(fps - 24.0) > 1e-6:
            raise ValueError("MiniMax H3 固定按 24fps 建模；请重新运行分段计划")
        for offset, frame_count in enumerate(segment_frames):
            expected_frames = core.length_for(frame_count / 24.0, 24.0)
            if frame_count != expected_frames:
                raise ValueError(
                    "分镜 %d 写的是 %d 帧，但官方 H3 网格会生成 %d 帧。" %
                    (offset + 1, frame_count, expected_frames))
        planned_overlap = int(plan.get("overlap_frames", overlap))
        if overlap != planned_overlap:
            raise ValueError(
                "context_length(%d) 与分段 overlap_frames(%d) 不一致；"
                "两边必须相同。" % (overlap, planned_overlap))
        if overlap not in (5, 22, 39, 56):
            raise ValueError("H3-Myang: context_length 必须是 5/22/39/56")
        too_short = [index + 1 for index, frame_count in enumerate(segment_frames)
                     if overlap >= frame_count]
        if too_short:
            raise ValueError(
                "H3-Myang: 锚点窗必须短于每个分镜；过短分镜：%s" %
                ", ".join(str(index) for index in too_short))

        def has_shot_action(entry):
            return any(
                isinstance(asset, dict)
                and str(asset.get("kind") or asset.get("media_type")).lower() == "video"
                and str(asset.get("role")) == "action"
                for asset in (entry.get("assets") or []))

        shot_actions = [
            has_shot_action(segment_entries[offset])
            if offset < len(segment_entries) else False
            for offset in range(count)
        ]
        task = str(task_mode)
        continuing = task == legacy.TASK_CONTINUE
        transferring = task == legacy.TASK_TRANSFER
        if continuing and ref_video is None and not shot_actions[0]:
            raise ValueError(
                "视频续写需要 ref_video，或在第一个镜头素材中指定『动作源』视频")
        if transferring and ref_video is None and not all(shot_actions):
            missing = [str(i + 1) for i, present in enumerate(shot_actions) if not present]
            raise ValueError(
                "动作迁移的第 %s 个镜头没有『动作源』视频，且 ref_video 没接" %
                "、".join(missing))
        if turbo is not None:
            task_family = str(turbo.get("task_family", "fl2va")).lower()
            if task_family == "fl2va" and (transferring or continuing):
                logger.warning(
                    "H3-Myang: 当前 Turbo LoRA 是 FL2VA/T2VA 档，『%s』会进入"
                    "参考/续写兼容路径；动作迁移建议换 LightX2V Ref2VA Turbo。", task)
            elif task_family == "ref2va" and task == legacy.TASK_FRESH and media is None:
                logger.warning(
                    "H3-Myang: 当前是 Ref2VA Turbo，但纯生成没有 Media Agent 参考素材；"
                    "若只做 T2VA，建议换 FL2VA Turbo 档。")
        detail_settings = kwargs.pop("二采设置", None)
        # Accept old API prompts during the transition, but the visible node
        # exposes only the Chinese combined settings input.
        if detail_settings is None:
            detail_settings = kwargs.pop("detail_settings", None)
        refine_model = None
        second_pass = detail.DETAIL_OFF
        second_width, second_height = 1664, 928
        second_steps, second_denoise = 4, 0.2
        second_scheduler, second_sampler = "beta", "res_multistep"
        second_upscale_method, second_chunk_frames = "bicubic", 4
        detail_mode = detail.DETAIL_MODE_UPSCALE_REFINE
        second_passes = 1
        second_seed_mode = detail.DETAIL_SEED_INHERIT
        latent_upscale_model = ""
        latent_precision = detail.LATENT_PRECISIONS[0]
        latent_chunk_steps = 16
        if detail_settings is not None:
            if not isinstance(detail_settings, dict):
                raise ValueError("二采设置输入不是『沐阳 H3 · 二采放大设置』的输出")
            refine_model = detail_settings.get("model")
            second_pass = (detail_settings.get("resolution", "832P")
                           if detail_settings.get("enabled") else detail.DETAIL_OFF)
            second_width = detail_settings.get("width", 1664)
            second_height = detail_settings.get("height", 928)
            second_steps = detail_settings.get("steps", 4)
            second_denoise = detail_settings.get("denoise", 0.2)
            second_scheduler = detail_settings.get(
                "scheduler", "beta")
            second_sampler = detail_settings.get("sampler_name", "res_multistep")
            second_upscale_method = detail_settings.get(
                "upscale_method", "bicubic")
            second_chunk_frames = detail_settings.get(
                "chunk_frames", 4)
            detail_mode = detail_settings.get(
                "mode", detail.DETAIL_MODE_UPSCALE_REFINE)
            second_passes = max(1, min(8, int(detail_settings.get("passes", 1))))
            second_seed_mode = detail_settings.get(
                "seed_mode", detail.DETAIL_SEED_INHERIT)
            latent_upscale_model = detail_settings.get(
                "latent_upscale_model", "")
            latent_precision = detail_settings.get(
                "latent_precision", detail.LATENT_PRECISIONS[0])
            latent_chunk_steps = detail_settings.get(
                "latent_chunk_steps", 16)
        refining = str(second_pass) != detail.DETAIL_OFF
        sampling_second_pass = (
            refining and detail_mode != detail.DETAIL_MODE_UPSCALE_ONLY)
        if sampling_second_pass and refine_model is None:
            raise ValueError(
                "已开启二采放大，但『二采模型』没接。请接 LoRA 之前的 Ref2VA 基模")
        if sampling_second_pass and turbo_metadata(refine_model) is not None:
            raise ValueError(
                "『二采模型』必须接 Turbo LoRA 之前的基模，不能接 Turbo 输出")

        loaded_model = str(
            getattr(h3, "names", {}).get("model", "")).lower()
        if transferring and loaded_model and "ref2va" not in loaded_model:
            raise ValueError(
                "动作迁移需要 Ref2VA 模型；当前 H3Loader 加载的是 %s" %
                loaded_model)

        if count > 1 or continuing or resuming:
            ensure_anchors()

        available = int(ref_video.shape[0]) if ref_video is not None else 0
        allow_tail_pad = bool(plan.get("reference_tail_pad", False))
        if transferring and not all(shot_actions):
            needed = max(start + frames
                         for start, frames in zip(ref_starts, segment_frames))
            if available < needed and not allow_tail_pad:
                raise ValueError(
                    "动作迁移参考视频只有 %d 帧，但 %d 段需要 %d 帧（%.2fs @24fps）。"
                    "请接够长的素材，或把视频加载器 frame_load_cap 接到"
                    "分段节点的 ref_frames_needed。" %
                    (available, count, needed, needed / 24.0))
            if available <= ref_starts[0]:
                raise ValueError(
                    "起始段要从参考视频第 %d 帧取素材，但参考视频只有 %d 帧" %
                    (ref_starts[0], available))
        elif continuing and not shot_actions[0] and available < overlap:
            raise ValueError(
                "视频续写至少需要 %d 帧参考视频，实际只有 %d 帧" %
                (overlap, available))

        graph = GraphBuilder()
        images, audios = {}, {}
        previous_sample = None
        previous_context_pixels = None
        previous_output_pixels = None
        previous_detail_sample = None

        # 一次 run 一个 id，子图里的 signal 节点带上它，前端才能把进度事件
        # 关联回这次长视频任务（而不是别的节点或上一次的残留）。
        run_id = "r%d_%d" % (int(noise_seed), int(time.time() * 1000))
        try:
            from server import PromptServer
            if hasattr(PromptServer, "instance") and PromptServer.instance is not None:
                PromptServer.instance.send_sync("myh3_longvideo_start", {
                    "run_id": run_id,
                    "owner_id": progress_owner,
                    "total_segments": count,
                    "refining": refining,
                    "correcting": False,
                    "save_segments": bool(save_segments),
                    "segment_prefix": str(segment_prefix),
                })
        except Exception:
            pass

        for index in range(1, count + 1):
            frames = segment_frames[index - 1]
            seconds = frames / 24.0
            batch_index = ref_starts[index - 1]
            seg_entry = (segment_entries[index - 1]
                         if index <= len(segment_entries) else {})
            # 断点续跑时分段计划保留绝对段号，落盘文件名才会写第 06 段而不是第 01 段。
            label_index = int(seg_entry.get("index") or index)

            seg_brief = ""
            if fixed_prompt is not None:
                segment_prompt = fixed_prompt
            else:
                pre_sliced_prompt = str(seg_entry.get("prompt") or "").strip()
                seg_brief = str(seg_entry.get("brief") or "").strip()
                segment_prompt = pre_sliced_prompt or seg_brief
                if not segment_prompt:
                    segment_prompt = str(plan.get("style_header") or "").strip()

            # 提示词与进度的实时预览改由子图里的 H3ProgressSignal 节点在执行时
            # 发送——这里只是构图，瞬间就会跑完整段循环，在这发事件前端只能看到
            # 最后一段一闪而过。signal 节点由依赖关系驱动，执行到哪才发到哪。

            shot_assets = seg_entry.get("assets") or []
            shot_media = None
            shot_action = None
            shot_action_audio = None
            if shot_assets:
                shot_inputs = {
                    "assets_json": json.dumps(shot_assets, ensure_ascii=False),
                    "required_frames": frames,
                    "asset_mode": str(seg_entry.get("asset_mode") or "仅本镜头"),
                }
                if media is not None:
                    shot_inputs["media"] = media
                shot_media = graph.node("H3ShotMedia", **shot_inputs)
                if shot_actions[index - 1]:
                    shot_action = shot_media.out(1)
                    shot_action_audio = shot_media.out(2)

            clip = shot_action
            segment_ref_audio = None
            if transferring and clip is None:
                if allow_tail_pad:
                    clip = graph.node(
                        "H3ReferenceClip", image=ref_video,
                        start_frame=batch_index, frame_count=frames).out(0)
                else:
                    clip = graph.node(
                        "ImageFromBatch", image=ref_video,
                        batch_index=batch_index, length=frames).out(0)
                if ref_audio is not None:
                    segment_ref_audio = graph.node(
                        "H3ReferenceAudioClip", audio=ref_audio,
                        start_frame=batch_index, frame_count=frames,
                        fps=24.0).out(0)

            if allow_tail_pad and clip is not None:
                if shot_media is not None:
                    media_inputs = {"media": shot_media.out(0)}
                elif media is not None:
                    media_inputs = {"media": media}
                else:
                    media_inputs = {}
                media_inputs["ref_video"] = clip
            elif shot_media is not None:
                if clip is not None and shot_action is None:
                    swapped = graph.node(
                        "H3MediaSwapClip", media=shot_media.out(0), clip=clip,
                        video_ordinal=1)
                    media_inputs = {"media": swapped.out(0)}
                else:
                    media_inputs = {"media": shot_media.out(0)}
            elif media is not None and clip is not None:
                swapped = graph.node(
                    "H3MediaSwapClip", media=media, clip=clip,
                    video_ordinal=1)
                media_inputs = {"media": swapped.out(0)}
            elif media is not None:
                media_inputs = {"media": media}
            elif clip is not None:
                # Without Agent media, the per-segment slice still reaches the
                # official Ref2VA path as <Video 1>.
                media_inputs = {"ref_video": clip}
            else:
                media_inputs = {}
            if segment_ref_audio is not None:
                media_inputs["ref_audio"] = segment_ref_audio

            condition = graph.node(
                "H3Condition", h3=h3, prompt=segment_prompt,
                resolution=resolution, aspect_ratio=aspect_ratio,
                width=width, height=height, seconds=seconds,
                ref_image_size=ref_image_size, **media_inputs)

            video_vae, audio_vae = h3.video_vae, h3.audio_vae
            if (previous_sample is None and previous_context_pixels is None
                    and not continuing and not resuming):
                positive, anchor = condition.out(0), None
            else:
                context = {}
                if previous_sample is None:
                    # 续写 / 断点续跑首段：钉到已有视频的尾部
                    # （motion-context 的 context_frames 路径）。
                    if shot_action is not None:
                        continuation_video = shot_action
                        continuation_audio = shot_action_audio
                        tail_start = max(0, frames - overlap)
                    elif resuming:
                        # 上一段成片只在这里出现：取尾部 overlap 帧做锚点，
                        # 它不参与 ref2va 参考视频通道。
                        continuation_video = context_video
                        continuation_audio = context_audio
                        tail_start = max(0, int(context_video.shape[0]) - overlap)
                    else:
                        continuation_video = ref_video
                        continuation_audio = ref_audio
                        tail_start = max(0, available - overlap)
                    tail = graph.node(
                        "ImageFromBatch", image=continuation_video,
                        batch_index=tail_start, length=overlap)
                    context["context_frames"] = tail.out(0)
                    if continuation_audio is not None:
                        context["context_audio"] = continuation_audio
                else:
                    # motion-context 风格：上一段一采 latent 直接钉入新段开头，全钉。
                    # latent 无损续接，跳过 VAE decode→encode round-trip，段间不累积损失、
                    # 无色彩偏移；ref_video 仍作为 clip 提供 ref2va 动作参考。
                    context["context_latent"] = previous_sample.out(0)
                    if previous_context_pixels is not None:
                        context["context_audio"] = previous_context_pixels[1]

                anchor = graph.node(
                    "H3AnchorContext", conditioning=condition.out(0),
                    vae=video_vae, latent=condition.out(1),
                    context_length=str(overlap),
                    audio_vae=audio_vae, **context)
                positive = anchor.out(0)

            first_latent = (anchor.out(2) if anchor is not None
                            else condition.out(1))

            guider = graph.node(
                "BasicGuider", model=model, conditioning=positive)
            sigmas = graph.node(
                "BasicScheduler", model=model, scheduler=scheduler,
                steps=steps, denoise=denoise)
            sigma_link = sigmas.out(0)
            refinement = detail_refinement_params(detail_refinement)
            if refinement is not None:
                sigma_link = graph.node(
                    "ExtendIntermediateSigmas", sigmas=sigma_link,
                    **refinement).out(0)
            noise = graph.node(
                "RandomNoise", noise_seed=noise_seed + index - 1)
            sample = graph.node(
                "H3SamplerAdvanced", noise=noise.out(0),
                guider=guider.out(0), sampler=sampler,
                sigmas=sigma_link, latent_image=first_latent,
                vae=video_vae, run_id=run_id,
                owner_id=progress_owner,
                segment_index=index, total_segments=count,
                pass_label="sample1")
            decoded_video = graph.node(
                "VAEDecode", samples=sample.out(0), vae=video_vae)
            decoded_audio = graph.node(
                "VAEDecodeAudio", samples=sample.out(0), vae=audio_vae)

            # 采样完成：把这一帧预览和「第 N 段采样完成」推给前端。signal 透传，
            # 不改画面，只保证它一定在 VAEDecode 之后、下游之前执行。
            sampled_sig = graph.node(
                "H3ProgressSignal", images=decoded_video.out(0),
                audio=decoded_audio.out(0), segment_index=index,
                total_segments=count, stage="sampled", run_id=run_id,
                owner_id=progress_owner,
                prompt=segment_prompt, brief=seg_brief, save_preview=True)

            # Drift is fitted on the original one-pass resolution.  When the
            # detail pass is enabled that low-resolution stream remains the
            # continuation state; feeding a 928P latent into the next 720P
            # segment would make the anchor rows spatially incompatible.
            joined = sampled_sig.out(0)
            joined_audio = sampled_sig.out(1)

            trim_frames = 0 if anchor is None else anchor.out(1)

            output_joined, output_audio = joined, joined_audio
            if refining:
                # 二采准备信号：latent 放大+VAE 投影 或 像素放大+VAE 编码可能耗时，
                # 在此发信号让前端显示"二采准备中"，避免采样前长时间无反馈。
                refine_start_sig = graph.node(
                    "H3ProgressSignal", images=joined, audio=joined_audio,
                    segment_index=index, total_segments=count,
                    stage="refine_start", run_id=run_id,
                    owner_id=progress_owner,
                    prompt=segment_prompt, brief=seg_brief, save_preview=False)
                joined = refine_start_sig.out(0)
                joined_audio = refine_start_sig.out(1)

                # 同分辨率精修不做放大；另外两种模式只在第一轮放大一次。
                detail_latent = sample.out(0)
                if detail_mode != detail.DETAIL_MODE_REFINE:
                    use_latent = any(
                        token in str(second_upscale_method).lower()
                        for token in ("latent", "neural_3d"))
                    latent_kwargs = dict(
                        samples=detail_latent,
                        resolution=second_pass, aspect_ratio=aspect_ratio,
                        width=second_width, height=second_height,
                        upscale_method=second_upscale_method,
                        chunk_frames=second_chunk_frames,
                        latent_upscale_model=latent_upscale_model,
                        latent_precision=latent_precision,
                        latent_chunk_steps=latent_chunk_steps)
                    if not use_latent:
                        latent_kwargs["vae"] = video_vae
                    upscaled_latent = graph.node("H3LatentUpscale", **latent_kwargs)
                    detail_latent = upscaled_latent.out(0)

                if sampling_second_pass:
                    condition_resolution = (
                        resolution if detail_mode == detail.DETAIL_MODE_REFINE
                        else second_pass)
                    condition_width = (
                        width if detail_mode == detail.DETAIL_MODE_REFINE
                        else second_width)
                    condition_height = (
                        height if detail_mode == detail.DETAIL_MODE_REFINE
                        else second_height)
                    condition_2nd = graph.node(
                        "H3Condition", h3=h3, prompt=segment_prompt,
                        resolution=condition_resolution,
                        aspect_ratio=aspect_ratio,
                        width=condition_width, height=condition_height,
                        seconds=seconds, ref_image_size=ref_image_size,
                        **media_inputs)
                    positive_2nd = condition_2nd.out(0)
                    if previous_detail_sample is not None:
                        detail_anchor = graph.node(
                            "H3AnchorContext", conditioning=positive_2nd,
                            # 保留当前段由一采结果放大/投影得到的二采底图，只把
                            # 上一段二采尾部写入开头锚点区。这里若使用
                            # condition_2nd.out(1)，会把当前段底图替换成空条件 latent，
                            # 从第二段开始即使低降噪也会采出乱码。
                            vae=video_vae, latent=detail_latent,
                            context_length=str(overlap), audio_vae=audio_vae,
                            context_latent=previous_detail_sample.out(0))
                        positive_2nd = detail_anchor.out(0)
                        detail_latent = detail_anchor.out(2)
                    guider_2nd = graph.node(
                        "BasicGuider", model=refine_model,
                        conditioning=positive_2nd)
                    sigmas_2nd = graph.node(
                        "BasicScheduler", model=refine_model,
                        scheduler=second_scheduler,
                        steps=second_steps, denoise=second_denoise)
                    sampler_2nd = graph.node(
                        "KSamplerSelect", sampler_name=second_sampler)
                    for pass_index in range(second_passes):
                        pass_seed = noise_seed + index - 1
                        if second_seed_mode == detail.DETAIL_SEED_OFFSET:
                            pass_seed += pass_index
                        noise_2nd = graph.node(
                            "RandomNoise", noise_seed=pass_seed)
                        sample_2nd = graph.node(
                            "H3SamplerAdvanced", noise=noise_2nd.out(0),
                            guider=guider_2nd.out(0),
                            sampler=sampler_2nd.out(0),
                            sigmas=sigmas_2nd.out(0),
                            latent_image=detail_latent,
                            vae=video_vae, run_id=run_id,
                            owner_id=progress_owner,
                            segment_index=index, total_segments=count,
                            pass_label="sample2" if pass_index == 0
                            else "sample2.%d" % (pass_index + 1))
                        detail_latent = sample_2nd.out(0)

                decoded_refine_video = graph.node(
                    "VAEDecode", samples=detail_latent, vae=video_vae)
                refined_out = decoded_refine_video.out(0)

                # 二采完成：推「第 N 段二采完成」和高分辨率预览帧。
                refined_sig = graph.node(
                    "H3ProgressSignal", images=refined_out,
                    audio=joined_audio, segment_index=index,
                    total_segments=count, stage="refined", run_id=run_id,
                    owner_id=progress_owner,
                    prompt=segment_prompt, brief=seg_brief, save_preview=True)
                output_joined = refined_sig.out(0)
                output_audio = refined_sig.out(1)

            # 二采前的原始分段（含漂移校正、已裁掉锚点帧），与最终分段帧数对齐，
            # 文件名追加 _原始，方便和二采后成片对比。仅在二采开启且要求保存时落盘。
            if refining and save_raw_segments and save_segments:
                raw_trim = graph.node(
                    "H3AnchorTrim", images=joined, audio=joined_audio,
                    trim_frames=trim_frames, fps=fps)
                raw_video = graph.node(
                    "CreateVideo", images=raw_trim.out(0),
                    audio=raw_trim.out(1), fps=fps, bit_depth=8)
                graph.node(
                    "SaveVideo", video=raw_video.out(0),
                    filename_prefix="%s_第%02d段_原始" %
                                    (segment_prefix, label_index),
                    format="mp4", codec="h264")

            trim = graph.node(
                "H3AnchorTrim", images=output_joined,
                audio=output_audio, trim_frames=trim_frames, fps=fps)

            # 分段完成：推最终预览帧（二采后/漂移后）和「第 N 段完成」。
            done_sig = graph.node(
                "H3ProgressSignal", images=trim.out(0),
                audio=trim.out(1), segment_index=index,
                total_segments=count, stage="done", run_id=run_id,
                owner_id=progress_owner,
                prompt=segment_prompt, brief=seg_brief, save_preview=True)
            segment_images, segment_audio = done_sig.out(0), done_sig.out(1)

            if save_segments:
                video = graph.node(
                    "CreateVideo", images=segment_images,
                    audio=segment_audio, fps=fps, bit_depth=8)
                graph.node(
                    "SaveVideo", video=video.out(0),
                    filename_prefix="%s_第%02d段" %
                                    (segment_prefix, label_index),
                    format="mp4", codec="h264")

            images["images_%d" % index] = segment_images
            audios["audios_%d" % index] = segment_audio
            previous_sample = sample
            previous_context_pixels = (segment_images, segment_audio)
            previous_output_pixels = (segment_images, segment_audio)
            if sampling_second_pass:
                previous_detail_sample = sample_2nd

        collector = graph.node(
            "H3SegmentCollector", active_count=count, **images, **audios)
        frame_summary = (str(segment_frames[0]) if len(set(segment_frames)) == 1
                         else "/".join(str(value) for value in segment_frames))
        logger.info(
            "H3-Myang: 原生锚点展开 %d段 × %s帧 | context=%d%s%s%s",
            count, frame_summary, overlap,
            "（实验速度锚点）" if overlap == 5 else "",
            " | 二采=" + str(second_pass) if refining else "",
            " | 断点续跑：从第%d段起，参考视频第%d帧" %
            (int(plan.get("resume_start_segment") or 1), ref_starts[0])
            if resuming else "")
        return {"expand": graph.finalize(),
                "result": (collector.out(0), collector.out(1))}


NODE_CLASS_MAPPINGS = dict(legacy.NODE_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update({
    "H3ScriptSplitter": H3ScriptSplitter,
    "H3ModelFromBundle": H3ModelFromBundle,
    "H3LongVideo": H3LongVideo,
})

NODE_DISPLAY_NAME_MAPPINGS = dict(legacy.NODE_DISPLAY_NAME_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update({
    "H3ScriptSplitter": "沐阳 H3 · 分段计划",
    "H3ModelFromBundle": "沐阳 H3 · 取模型（挂补丁用）",
    "H3LongVideo": "沐阳 H3 · 长视频（原生多关键帧）",
})
