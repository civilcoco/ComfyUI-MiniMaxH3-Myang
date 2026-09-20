"""LightX2V MiniMax-H3 Turbo model-only LoRA and AV schedule adapter.

The node deliberately delegates the two sensitive operations to ComfyUI's
public primitives: ``LoraLoaderModelOnly`` loads the model patch and
``MiniMaxH3SigmaShift`` applies the jointly trained video/audio flow schedule.
Old graphs that already load LoRA upstream remain valid and do not double-load.
"""

import logging

import folder_paths


logger = logging.getLogger(__name__)

PROFILE_AUTO = "自动匹配 LoRA 文件（推荐）"
# These three serialized labels intentionally remain byte-for-byte compatible
# with existing Myang workflows. Their task family is explicit in _SPECS.
PROFILE_8_V1 = "LightX2V v1.0 · 8步（12/3·通用）"
PROFILE_4_V1_768 = "LightX2V v1.0 · 4步768P（6/3）"
PROFILE_4_V01 = "LightX2V v0.1 · 4步（12/3）"
PROFILE_REF_4_V01 = "LightX2V Ref2VA v0.1 · 4步（12/3）"
PROFILE_MANUAL = "手动（高级）"
PROFILES = [PROFILE_AUTO, PROFILE_8_V1, PROFILE_4_V1_768,
            PROFILE_REF_4_V01, PROFILE_4_V01, PROFILE_MANUAL]

LORA_EXTERNAL = "不在本节点加载（兼容旧工作流）"
LORA_NONE = "不使用"
ADDITIONAL_LORA_SLOTS = 3

TURBO_MARKER = "myang_h3_turbo_schedule"

_SPECS = {
    PROFILE_8_V1: {
        "shift_video": 12.0,
        "shift_audio": 3.0,
        "recommended_steps": 8,
        "allowed_steps": (8, 4),
        "training_resolution": "544p mixed aspect ratio",
        "task_family": "fl2va",
    },
    PROFILE_4_V1_768: {
        "shift_video": 6.0,
        "shift_audio": 3.0,
        "recommended_steps": 4,
        "allowed_steps": (4,),
        "training_resolution": "1344x768",
        "task_family": "fl2va",
    },
    PROFILE_4_V01: {
        "shift_video": 12.0,
        "shift_audio": 3.0,
        "recommended_steps": 4,
        "allowed_steps": (4,),
        "training_resolution": "544p mixed aspect ratio",
        "task_family": "fl2va",
    },
    PROFILE_REF_4_V01: {
        "shift_video": 12.0,
        "shift_audio": 3.0,
        "recommended_steps": 4,
        "allowed_steps": (4,),
        "training_resolution": "544p mixed aspect ratio",
        "task_family": "ref2va",
    },
}


def infer_profile(lora_name, strict=True):
    """Infer one official schedule from the published checkpoint filename."""
    name = str(lora_name or "").lower().replace("-", "_")
    if "ref2va" in name or "ref2v" in name:
        if "4step" in name:
            return PROFILE_REF_4_V01
    elif "8step" in name:
        return PROFILE_8_V1
    elif "4step" in name and "768p" in name:
        return PROFILE_4_V1_768
    elif "4step" in name:
        return PROFILE_4_V01
    if strict:
        raise ValueError(
            "H3-Myang: 无法从 LoRA 文件名识别 LightX2V 档位；"
            "请选择明确档位，或使用『手动（高级）』")
    return None


def resolve_profile(profile, shift_video=12.0, shift_audio=3.0,
                    recommended_steps=8, lora_name=None):
    """Return one validated, self-contained schedule contract."""
    profile = str(profile)
    if profile == PROFILE_AUTO:
        profile = infer_profile(lora_name)
    if profile in _SPECS:
        return {"profile": profile, **_SPECS[profile]}
    if profile != PROFILE_MANUAL:
        raise ValueError("H3-Myang: 未知 Turbo 调度档位：%s" % profile)
    video = float(shift_video)
    audio = float(shift_audio)
    steps = int(recommended_steps)
    if video <= 0.0 or audio <= 0.0:
        raise ValueError("H3-Myang: video/audio shift 必须大于 0")
    if steps < 1:
        raise ValueError("H3-Myang: 建议步数必须大于 0")
    return {
        "profile": profile,
        "shift_video": video,
        "shift_audio": audio,
        "recommended_steps": steps,
        "allowed_steps": (steps,),
        "training_resolution": "manual",
        "task_family": "custom",
    }


def _lora_choices():
    try:
        names = list(folder_paths.get_filename_list("loras"))
    except Exception:
        names = []
    return [LORA_EXTERNAL, *[name for name in names if name != LORA_EXTERNAL]]


def _additional_lora_choices():
    return [LORA_NONE, *[
        name for name in _lora_choices()
        if name not in {LORA_EXTERNAL, LORA_NONE}
    ]]


def _enabled(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value == 1
    if isinstance(value, str):
        return value.strip().casefold() in {"true", "1", "yes", "on", "开启", "启用"}
    return False


def turbo_metadata(model):
    """Read this package's non-invasive schedule contract from a MODEL."""
    options = getattr(model, "model_options", None)
    if not isinstance(options, dict):
        return None
    value = options.get(TURBO_MARKER)
    return dict(value) if isinstance(value, dict) else None


def sampler_function_name(sampler):
    function = getattr(sampler, "sampler_function", None)
    return str(getattr(function, "__name__", ""))


SPEED_CACHE_OFF = "关闭"
# Kept only so workflows serialized by older releases still pass ComfyUI's
# combo validation.  These values are hidden by the frontend and are no longer
# mounted onto the model: both integrations retained opaque/history state and
# were unsafe to compose with the current H3 attention/memory patch chain.
SPEED_CACHE_TESPEED = "TE-Speed 时步缓存 (提速40%)"
SPEED_CACHE_SPECTRUM = "Spectrum 频谱加速 (提速35%)"
SPEED_CACHES = [SPEED_CACHE_OFF, SPEED_CACHE_TESPEED, SPEED_CACHE_SPECTRUM]


class H3TurboSchedule:
    CATEGORY = "沐阳 H3/模型"
    FUNCTION = "apply"
    RETURN_TYPES = ("MODEL", "INT", "FLOAT", "FLOAT")
    # Serialized output names stay stable; the frontend supplies Chinese labels.
    RETURN_NAMES = ("model", "recommended_steps", "shift_video", "shift_audio")
    DESCRIPTION = (
        "合并 ComfyUI『LoRA加载器（仅模型）』与官方 H3 Sigma Shift。"
        "既可在本节点选择 LightX2V LoRA，也能沿用旧图上游已加载的 LoRA；"
        "官方 Turbo 后可按顺序叠加最多 3 个普通 H3 效果 LoRA，且不改变官方调度；"
        "自动匹配视频/音频联合轨迹，可查看实际 Shift 并按需手动覆盖。"
        "旧版内置的 TE-Speed / Spectrum 挂载已停用，避免额外占用显存/内存及补丁冲突。")

    def __init__(self):
        self._lora_loader = None

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "model": ("MODEL", {
                "tooltip": "新图接基础 MODEL 并在下方选择 LoRA；旧图也可接上游已加载 LoRA 的 MODEL"}),
            "profile": (PROFILES, {
                "default": PROFILE_AUTO,
                "tooltip": "自动档会按官方 ComfyUI LoRA 文件名识别 FL2VA/Ref2VA、4/8步和768P"}),
            "speed_cache": (SPEED_CACHES, {
                "default": SPEED_CACHE_OFF,
                "advanced": True,
                "tooltip": "旧工作流位置兼容占位；TE-Speed / Spectrum 内置挂载已移除，始终按关闭处理。"}),
            "shift_video": ("FLOAT", {
                "default": 12.0, "min": 0.01, "max": 100.0, "step": 0.01,
                "advanced": True, "tooltip": "在『手动』档或开启『手动覆盖Shift』时生效"}),
            "shift_audio": ("FLOAT", {
                "default": 3.0, "min": 0.01, "max": 100.0, "step": 0.01,
                "advanced": True, "tooltip": "在『手动』档或开启『手动覆盖Shift』时生效"}),
            "recommended_steps": ("INT", {
                "default": 8, "min": 1, "max": 100, "step": 1,
                "advanced": True, "tooltip": "只在『手动』档生效"}),
            "LoRA文件": (_lora_choices(), {
                "default": LORA_EXTERNAL,
                "tooltip": "新工作流可在这里直接加载；旧工作流保持第一项，沿用上游 LoraLoaderModelOnly，避免重复加载"}),
            "LoRA强度": ("FLOAT", {
                "default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01,
                "tooltip": "等同 ComfyUI『LoRA加载器（仅模型）』的 strength_model；LightX2V 示例通常为 1.0"}),
            "手动覆盖Shift": ("BOOLEAN", {
                "default": False,
                "label_on": "使用自定义 Shift",
                "label_off": "使用档位官方 Shift",
                "tooltip": "关闭时严格采用所选 LightX2V 档位；开启后使用上方视频/音频 Shift，属于高级实验设置"}),
            "附加LoRA开启": ("BOOLEAN", {
                "default": False,
                "label_on": "启用附加效果 LoRA",
                "label_off": "不叠加其他 LoRA",
                "tooltip": "在官方 Turbo LoRA 之后最多叠加 3 个自定义 H3 LoRA；它们不参与 Turbo 档位、步数或 Shift 推断"}),
        }
        choices = _additional_lora_choices()
        for index in range(1, ADDITIONAL_LORA_SLOTS + 1):
            required["附加LoRA%d启用" % index] = ("BOOLEAN", {
                "default": False,
                "label_on": "启用槽位 %d" % index,
                "label_off": "关闭槽位 %d" % index,
            })
            required["附加LoRA%d文件" % index] = (choices, {
                "default": LORA_NONE,
                "tooltip": "选择角色、画风、动作等兼容 MiniMax H3 的普通 LoRA；不要在这里重复选择官方 Turbo LoRA",
            })
            required["附加LoRA%d强度" % index] = ("FLOAT", {
                "default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01,
                "tooltip": "该附加 LoRA 的模型强度；允许负值，0 等同跳过",
            })
        return {"required": required}

    def apply(self, model, profile, speed_cache=SPEED_CACHE_OFF, shift_video=12.0, shift_audio=3.0,
              recommended_steps=8, **kwargs):
        lora_name = str(kwargs.get("LoRA文件", LORA_EXTERNAL) or LORA_EXTERNAL)
        lora_strength = float(kwargs.get("LoRA强度", 1.0))
        loads_here = lora_name != LORA_EXTERNAL
        additional_loras = []
        if _enabled(kwargs.get("附加LoRA开启", False)):
            for index in range(1, ADDITIONAL_LORA_SLOTS + 1):
                if not _enabled(kwargs.get("附加LoRA%d启用" % index, False)):
                    continue
                name = str(kwargs.get(
                    "附加LoRA%d文件" % index, LORA_NONE) or LORA_NONE)
                strength = float(kwargs.get("附加LoRA%d强度" % index, 1.0))
                if name == LORA_NONE or strength == 0.0:
                    continue
                additional_loras.append({
                    "slot": index, "name": name, "strength": strength})
        selected_names = ([lora_name] if loads_here else []) + [
            item["name"] for item in additional_loras]
        duplicates = sorted({
            name for name in selected_names if selected_names.count(name) > 1})
        if duplicates:
            raise ValueError(
                "H3-Myang: 同一个 LoRA 不能重复叠加：%s" % "、".join(duplicates))
        existing = turbo_metadata(model)
        if (loads_here or additional_loras) and existing is not None:
            raise ValueError(
                "H3-Myang: 输入模型已经带 Turbo 调度标记，不能再次叠加 LoRA；"
                "请把本节点接到 LoRA 之前的基础模型")

        if str(profile) not in (PROFILE_AUTO, PROFILE_MANUAL) and loads_here:
            inferred = infer_profile(lora_name, strict=False)
            if inferred is not None and inferred != str(profile):
                raise ValueError(
                    "H3-Myang: 所选 LoRA 文件看起来属于『%s』，但调度选的是『%s』" %
                    (inferred, profile))
        spec = resolve_profile(
            profile, shift_video, shift_audio, recommended_steps,
            lora_name=lora_name if loads_here else None)
        manual_profile = str(spec.get("profile")) == PROFILE_MANUAL
        override_shift = bool(kwargs.get("手动覆盖Shift", False)) and not manual_profile
        official_video = float(spec["shift_video"])
        official_audio = float(spec["shift_audio"])
        if override_shift:
            custom_video = float(shift_video)
            custom_audio = float(shift_audio)
            if custom_video <= 0.0 or custom_audio <= 0.0:
                raise ValueError("H3-Myang: 自定义 video/audio Shift 必须大于 0")
            spec = dict(spec)
            spec["shift_video"] = custom_video
            spec["shift_audio"] = custom_audio
            logger.warning(
                "H3-Myang: 已手动覆盖 LightX2V 官方 Shift %.2f/%.2f -> %.2f/%.2f",
                official_video, official_audio, custom_video, custom_audio)
        spec = dict(spec)
        spec.update({
            "shift_overridden": bool(override_shift or manual_profile),
            "official_shift_video": None if manual_profile else official_video,
            "official_shift_audio": None if manual_profile else official_audio,
        })

        working_model = model
        if loads_here or additional_loras:
            if self._lora_loader is None:
                from nodes import LoraLoaderModelOnly
                self._lora_loader = LoraLoaderModelOnly()
        if loads_here:
            loaded = self._lora_loader.load_lora_model_only(
                model, lora_name, lora_strength)
            if not isinstance(loaded, (tuple, list)) or not loaded:
                raise RuntimeError("H3-Myang: ComfyUI LoRA加载器（仅模型）没有返回 MODEL")
            working_model = loaded[0]
        for item in additional_loras:
            loaded = self._lora_loader.load_lora_model_only(
                working_model, item["name"], item["strength"])
            if not isinstance(loaded, (tuple, list)) or not loaded:
                raise RuntimeError(
                    "H3-Myang: 附加 LoRA %d 没有返回 MODEL" % item["slot"])
            working_model = loaded[0]

        from comfy_extras.nodes_minimax_h3 import MiniMaxH3SigmaShift

        result = MiniMaxH3SigmaShift.execute(
            model=working_model,
            shift_video=spec["shift_video"],
            shift_audio=spec["shift_audio"],
        )
        values = result.result if hasattr(result, "result") else result
        if not isinstance(values, (tuple, list)) or not values:
            raise RuntimeError("H3-Myang: 官方 MiniMaxH3SigmaShift 没有返回 MODEL")
        patched = values[0]

        # `speed_cache` remains in the signature solely to preserve the widget
        # positions of existing workflows.  Older builds dynamically imported
        # TE-Speed/Spectrum here.  That bypassed ComfyUI's registered-node
        # contract, retained additional cache/history memory, and made patch
        # ordering depend on opaque third-party implementations.  Never mount
        # either cache from this combined loader.
        requested_speed_cache = str(speed_cache or SPEED_CACHE_OFF)
        if requested_speed_cache != SPEED_CACHE_OFF:
            logger.warning(
                "H3-Myang: 旧工作流请求的加速缓存『%s』已停用并忽略；"
                "Turbo LoRA/Sigma Shift 将按原生方式运行",
                requested_speed_cache)

        options = getattr(patched, "model_options", None)
        if not isinstance(options, dict):
            raise RuntimeError("H3-Myang: Sigma Shift 返回了不可识别的 MODEL")
        marker = dict(spec)
        marker.update({
            "lora_name": lora_name if loads_here else None,
            "lora_strength": lora_strength if loads_here else None,
            "lora_loaded_here": loads_here,
            "additional_loras": [dict(item) for item in additional_loras],
            "speed_cache": SPEED_CACHE_OFF,
        })
        options[TURBO_MARKER] = marker
        logger.info(
            "H3-Myang Turbo: %s | lora=%s | strength=%s | additional=%s | shift video/audio=%.2f/%.2f | steps=%s | cache=%s",
            spec["profile"], lora_name if loads_here else "upstream",
            lora_strength if loads_here else "upstream",
            ", ".join("%s@%s" % (item["name"], item["strength"])
                      for item in additional_loras) or "none",
            spec["shift_video"], spec["shift_audio"],
            "/".join(str(x) for x in spec["allowed_steps"]), SPEED_CACHE_OFF)
        return (patched, spec["recommended_steps"],
                spec["shift_video"], spec["shift_audio"])


NODE_CLASS_MAPPINGS = {"H3TurboSchedule": H3TurboSchedule}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3TurboSchedule": "沐阳 H3 · Turbo LoRA 联合音画加载调度",
}
