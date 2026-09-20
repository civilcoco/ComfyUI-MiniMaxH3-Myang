"""Long-script director for MiniMax H3 — ComfyUI-MiniMaxH3-Myang.

SPDX-License-Identifier: GPL-3.0-only

Split a full script once, then expand one short brief per segment. The point is
token economy: a six-segment minute costs one pass over the whole script plus
six small calls, instead of six passes over the whole script.

This module contains the public planning, segment, and native long-video
nodes. Public node IDs and serialized widget positions remain stable.

Portions of the media normalization and reference-editor compatibility layer
are adapted from nkxx188/ComfyUI-MiniMaxH3-Easy under the MIT License.  The
Media Agent and LLM bridge are maintained by Myang.  See THIRD_PARTY_NOTICES.md.
"""

import hashlib
import json
import logging
import math
import os
import re
import time
from pathlib import Path

import torch

from . import core, detail
from .anchor_compat import ensure_anchors
from .core import REF_IMAGE_SIZES as CORE_REF_SIZES

logger = logging.getLogger(__name__)

CUSTOM_NODES_DIR = Path(__file__).resolve().parent.parent

# H3 supports this duration range; reject out-of-range input instead of
# silently clamping it into a different length.
MIN_SECONDS = 4.0
MAX_SECONDS = 20.0

# Segment slots wired into the generated workflow. Only `segment_count` of them
# actually run; the rest are never evaluated thanks to lazy inputs.
MAX_SLOTS = 12

MODE_DIRECT = "直接用分段稿"
MODE_REFINE = "LLM细化"
# Action transfer off a reference video does not need a per-segment script at
# all: one sentence naming the media covers the whole film, and rewriting it
# per segment only invites the model to drift off the reference.
MODE_FIXED = "全片同一提示词"
PROMPT_MODES = [MODE_FIXED, MODE_DIRECT, MODE_REFINE]

# What the reference video is *for*. Transfer reads a different slice of it each
# segment; continuation reads only its tail, once, to start from. Neither is
# required -- with no reference video at all the prompt and stills carry it.
TASK_TRANSFER = "动作迁移（跟随参考视频）"
TASK_CONTINUE = "视频续写（接着往下演）"
TASK_FRESH = "纯生成（不用参考视频）"
TASK_MODES = [TASK_TRANSFER, TASK_CONTINUE, TASK_FRESH]

LENGTH_MANUAL = "用填写的总时长"
LENGTH_MATCH_REF = "匹配参考视频时长"
LENGTH_SOURCES = [LENGTH_MANUAL, LENGTH_MATCH_REF]

FIXED_PROMPT_DEFAULT = ("参考@视频1中的人物动作表情、镜头调度、画面风格，"
                        "并将@视频1中的人物完全替换成@图片1，"
                        "并且背景替换成美丽的大草原。保留素材中已有的版权标识，"
                        "没有背景音乐，无字幕。")

# Canonical backend values; the Chinese strings shown on the node are frontend
# labels mapped back onto these values before submission.
RESOLUTIONS = ["360P", "416P", "480P", "540P", "640P", "720P",
               "768P", "832P", "928P", "1024P", "1080P", "custom"]
ASPECTS = ["1:1", "2:3", "3:2", "3:4", "4:3", "9:16", "16:9", "21:9"]
CONTEXT_LENGTHS = ["22", "5", "39", "56"]
SCHEDULERS = ["simple", "normal", "karras", "exponential", "sgm_uniform",
              "ddim_uniform", "beta", "linear_quadratic", "kl_optimal"]

# Clear step previews require the video VAE and can evict the H3 DiT even when
# decoded infrequently.  The default now uses the official H3 latent-to-RGB
# projection on CPU; completed segments still go through the real VAE.
FIRST_MEMORY_AUTO = "自动平衡（16GB推荐）"
FIRST_MEMORY_STANDARD = "兼容模式（每步清晰预览）"
FIRST_MEMORY_LOW = "显存优先（更大预留）"
# Keep the former value in the schema so already-saved workflows validate.
# It means the old unoptimized path: no extra cleanup, no forced reservation,
# and no step preview decode.
FIRST_MEMORY_LEGACY_OFF = "关闭"
FIRST_MEMORY_PROFILES = [FIRST_MEMORY_AUTO, FIRST_MEMORY_STANDARD,
                         FIRST_MEMORY_LOW, FIRST_MEMORY_LEGACY_OFF]

PASS1_CHECKPOINT_OFF = "关闭"
PASS1_CHECKPOINT_SAVE = "保存一采检查点"
PASS1_CHECKPOINT_RESUME = "恢复一采进度（已有跳过，缺失继续）"
PASS1_CHECKPOINT_REUSE = "读取检查点，直接二采"
PASS1_VIDEO_REUSE = "使用接入的一采成片，直接二采（单段）"
PASS1_CHECKPOINT_MODES = [
    PASS1_CHECKPOINT_OFF,
    PASS1_CHECKPOINT_SAVE,
    PASS1_CHECKPOINT_RESUME,
    PASS1_CHECKPOINT_REUSE,
    PASS1_VIDEO_REUSE,
]


def first_pass_memory_policy(profile):
    value = str(profile or FIRST_MEMORY_AUTO)
    if value == FIRST_MEMORY_LEGACY_OFF:
        return {"cleanup": False, "reserve_vram_gb": 0.0,
                "preview_interval": 0, "preview_mode": "latent_rgb"}
    if value == FIRST_MEMORY_STANDARD:
        return {"cleanup": False, "reserve_vram_gb": 0.0,
                "preview_interval": 1, "preview_mode": "vae"}
    if value == FIRST_MEMORY_LOW:
        return {"cleanup": True, "reserve_vram_gb": 1.5,
                "preview_interval": 1, "preview_mode": "latent_rgb"}
    # The CPU latent projection is cheap enough to update every step and does
    # not disturb CUDA model residency. The final step is still omitted because
    # normal post-sample VAE decode immediately publishes the finished segment.
    return {"cleanup": True, "reserve_vram_gb": 1.25,
            "preview_interval": 1, "preview_mode": "latent_rgb"}

# These two carry the Chinese label as the *value*, not as a frontend
# translation of it. Keeping the display values in Python also makes saved
# workflow JSON readable when the browser extension has not loaded. Mapping
# back happens once in `_canon`, immediately before sampling.
MENTION_MODES = {
    "按编号（@图片1）": "index",
    "按文件名": "filename",
}
REF_IMAGE_SIZES = {
    "匹配生成分辨率": "match",
    "最大1K面积": "1k",
    "最大1.5K面积": "1.5k",
    "最大2K面积": "2k",
    "匹配素材（原尺寸）": "original",
}


def _canon(mapping: dict, value, fallback: str) -> str:
    """Chinese label -> canonical value, accepting a canonical value as-is."""
    text = str(value)
    if text in mapping:
        return mapping[text]
    return text if text in set(mapping.values()) else fallback


# --------------------------------------------------------------------------
# Myang LLM service glue
# --------------------------------------------------------------------------

def llm_service_options() -> list[str]:
    from . import llm_service
    return llm_service.llm_service_options()


def vlm_service_options() -> list[str]:
    from . import llm_service
    try:
        return llm_service.vlm_service_options()
    except Exception as exc:  # noqa: BLE001 - a missing VLM must not hide the node
        logger.warning("H3-Myang: 无法读取 VLM 服务列表：%s", exc)
        return ["off"]


SKILL_PRESET_AUTO = "auto"
SKILL_PRESET_NONE = "none"


def skill_preset_options() -> list[str]:
    """Skill dropdown for prompt-writing nodes, shared with the Media Agent."""
    try:
        from . import agent_nodes
        return agent_nodes.skill_preset_options()
    except Exception as exc:  # noqa: BLE001 - skills are optional
        logger.warning("H3-Myang: 无法读取技能列表：%s", exc)
        return [SKILL_PRESET_AUTO, SKILL_PRESET_NONE]


def resolve_skill(skill_preset, skill_text="", llm_service="",
                  ollama_auto_unload=False, routing_prompt="") -> tuple[str, str]:
    """Resolve a Skill to its writing rules; never let a bad Skill break a run."""
    preset = str(skill_preset or SKILL_PRESET_NONE).strip() or SKILL_PRESET_NONE
    if preset == SKILL_PRESET_NONE and not str(skill_text or "").strip():
        return "", ""
    try:
        from . import agent_nodes
        return agent_nodes.resolve_skill(
            preset, skill_text, llm_service=llm_service,
            ollama_auto_unload=ollama_auto_unload, routing_prompt=routing_prompt)
    except Exception as exc:  # noqa: BLE001 - degrade to the default strategy
        logger.warning("H3-Myang: 技能加载失败，改用默认写法：%s", exc)
        return "", ""


def call_llm(llm_service: str, user_text: str, system_prompt: str,
             ollama_auto_unload: bool, seed: int, max_tokens: int | None = None) -> str:
    """One LLM round trip through Myang's own stable service registry."""
    from . import llm_service as service_client
    return service_client.call_llm(
        llm_service, system_prompt, user_text,
        ollama_auto_unload=bool(ollama_auto_unload), seed=int(seed),
        max_tokens=max_tokens)


# --------------------------------------------------------------------------
# frame grid / segment maths
# --------------------------------------------------------------------------

def frame_length(seconds: float, fps: float) -> int:
    """Snap a requested duration to the H3 temporal 17k+5 frame grid."""
    target = max(5.0, float(seconds) * float(fps))
    blocks = max(0, round((target - 5) / 17))
    return blocks * 17 + 5


def frame_length_at_most(seconds: float, fps: float) -> int:
    """Largest legal H3 frame count that does not exceed a duration cap."""
    target = max(5.0, float(seconds) * float(fps))
    blocks = max(0, math.floor((target - 5.0) / 17.0))
    return blocks * 17 + 5


def _normalize_storyboard_durations(storyboard, total_seconds, maximum_seconds,
                                    overlap_frames, fps):
    """Fit AI-chosen relative durations onto H3's grid without crossing the cap.

    The planner owns pacing; this function only performs deterministic frame-grid
    projection and a small total-length correction.  It never turns the maximum
    back into a fixed per-segment duration.
    """
    items = [dict(item) for item in (storyboard or []) if isinstance(item, dict)]
    if not items:
        return []
    count = len(items)
    max_frames = frame_length_at_most(maximum_seconds, fps)
    min_frames = min(max_frames, frame_length(MIN_SECONDS, fps))
    connected = sum(
        index > 0 and str(item.get("transition") or "承接") != "切镜"
        for index, item in enumerate(items))
    target_sum = int(round(float(total_seconds) * float(fps))) + (
        int(overlap_frames) * connected)

    min_blocks = max(0, int(math.ceil((min_frames - 5) / 17)))
    max_blocks = max(min_blocks, int(math.floor((max_frames - 5) / 17)))
    target_blocks = int(round((target_sum - 5 * count) / 17.0))
    target_blocks = max(min_blocks * count,
                        min(max_blocks * count, target_blocks))

    requested = []
    for item in items:
        try:
            value = float(item.get("duration_seconds") or item.get("seconds") or 0)
        except (TypeError, ValueError):
            value = 0.0
        requested.append(max(MIN_SECONDS, min(float(maximum_seconds), value))
                         if value > 0 else float(maximum_seconds))
    available = target_blocks - min_blocks * count
    capacities = [max_blocks - min_blocks] * count
    weights = [max(0.001, value - MIN_SECONDS + 0.25) for value in requested]
    allocated = [0] * count
    while available > 0 and any(
            allocated[index] < capacities[index] for index in range(count)):
        candidates = [index for index in range(count)
                      if allocated[index] < capacities[index]]
        pick = max(candidates, key=lambda index: (
            weights[index] / (allocated[index] + 1), -index))
        allocated[pick] += 1
        available -= 1

    for index, item in enumerate(items):
        frames = (min_blocks + allocated[index]) * 17 + 5
        item["frames"] = int(frames)
        item["duration_seconds"] = float(frames) / max(float(fps), 1.0)
    return items


def plan_segments(total_seconds: float, segment_seconds: float, overlap_frames: int,
                  fps: float, max_segments: int, cover_with_maximum: bool = False) -> dict:
    """How many segments cover `total_seconds`, accounting for the pinned overlap.

    Segment 1 contributes its whole length. Every later segment gives back
    `overlap_frames` to Motion Context's trim, so it only advances the film by
    `(frames - overlap) / fps`.
    """
    if not (MIN_SECONDS <= segment_seconds <= MAX_SECONDS):
        raise ValueError(
            f"segment_seconds 必须在 {MIN_SECONDS}~{MAX_SECONDS} 之间"
            f"（H3 会把范围外的值静默钳位，长度就对不上了），当前是 {segment_seconds}"
        )
    # `segment_seconds` is a ceiling.  Never let temporal-grid rounding create
    # a segment that is longer than the value shown to the user.
    frames = frame_length_at_most(segment_seconds, fps)
    if overlap_frames >= frames:
        raise ValueError(f"overlap_frames({overlap_frames}) 必须小于每段帧数({frames})")

    first = frames / fps
    tail = (frames - overlap_frames) / fps
    if total_seconds <= first:
        count = 1
    else:
        # Intelligent splitting treats the value as a hard upper bound and
        # therefore needs enough slots to cover the film. Legacy/direct mode
        # keeps its historical nearest-count behaviour for old workflows.
        allocator = math.ceil if cover_with_maximum else round
        count = 1 + max(1, allocator((float(total_seconds) - first) / tail))
    count = max(1, min(int(count), int(max_segments), MAX_SLOTS))
    return {
        "segment_count": count,
        "frames_per_segment": frames,
        "segment_seconds_snapped": frames / fps,
        "advance_seconds": tail,
        "total_seconds_actual": first + (count - 1) * tail,
        "fps": float(fps),
        "overlap_frames": int(overlap_frames),
        # Segment i slices the reference video from (i-1)*(frames-overlap); the
        # last one still needs a full `frames` window, so the loader has to be
        # allowed to read this many frames or the tail segments come up empty.
        "ref_frames_needed": (count - 1) * (frames - int(overlap_frames)) + frames,
    }


# --------------------------------------------------------------------------
# cache
# --------------------------------------------------------------------------

def _cache_dir() -> Path:
    try:
        import folder_paths
        root = Path(folder_paths.get_user_directory())
    except Exception:
        root = CUSTOM_NODES_DIR.parent / "user"
    path = root / "h3_longscript_cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _cache_key(*parts) -> str:
    return hashlib.sha256(repr(parts).encode("utf-8")).hexdigest()[:32]


AGENT_CONTEXT_VERSION = 1
AGENT_CONTEXT_PREFIX = "h3_agent_context_"


def _stable_agent_media_manifest(media_manifest):
    """Keep checkpoint identity stable when VLM descriptions vary slightly."""
    lines = []
    for raw_line in str(media_manifest or "").splitlines():
        line = raw_line.strip()
        if not line or line.startswith(("画面内容：", "语音内容：", "（这是")):
            continue
        lines.append(line)
    return "\n".join(lines)


def _agent_context_key(
    text, count, media_manifest, skill_rules, skill_preset, skill_text,
    total_seconds, segment_seconds, overlap_frames, fps,
):
    """Identify a screenplay context without binding it to a model or seed."""
    return _cache_key(
        "agent-context", AGENT_CONTEXT_VERSION, SPLIT_PROMPT_VERSION,
        str(text or ""), int(count), _stable_agent_media_manifest(media_manifest),
        str(skill_rules or ""), str(skill_preset or ""), str(skill_text or ""),
        round(float(total_seconds), 4), round(float(segment_seconds), 4),
        int(overlap_frames), round(float(fps), 4))


def _agent_context_file(context_key):
    return _cache_dir() / (AGENT_CONTEXT_PREFIX + str(context_key) + ".json")


def _write_json_atomic(path, payload):
    """Write a small local JSON checkpoint without exposing partial JSON."""
    serialized = json.dumps(payload, ensure_ascii=False, indent=2)
    try:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(serialized, encoding="utf-8")
        replace = getattr(temporary, "replace", None)
        if callable(replace):
            replace(path)
            return
    except (AttributeError, OSError, TypeError, ValueError):
        pass
    path.write_text(serialized, encoding="utf-8")


def _load_agent_context(path, context_key, expected_count):
    """Load only a context made for this exact screenplay contract."""
    try:
        if not path.is_file():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    try:
        version = int(payload.get("version") or 0)
        saved_count = int(payload.get("expected_count") or 0)
    except (TypeError, ValueError):
        return None
    if (payload.get("format") != "h3-agent-context"
            or version != AGENT_CONTEXT_VERSION
            or str(payload.get("context_key") or "") != str(context_key)
            or saved_count != int(expected_count)):
        return None
    segments = payload.get("segments")
    if not isinstance(segments, list):
        return None
    valid = []
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        try:
            index = int(segment.get("index") or 0)
        except (TypeError, ValueError):
            continue
        prompt = str(segment.get("prompt") or "").strip()
        if 1 <= index <= int(expected_count) and prompt:
            copy = dict(segment)
            copy["index"] = index
            copy["prompt"] = prompt
            valid.append(copy)
    deduplicated = {item["index"]: item for item in valid}
    payload["segments"] = [deduplicated[index] for index in sorted(deduplicated)]
    return payload


# --------------------------------------------------------------------------
# JSON recovery
# --------------------------------------------------------------------------

def _loads_loose(text: str) -> dict:
    """Parse optional JSON without treating ordinary model prose as an error."""
    cleaned = re.sub(r"^\s*```(?:json)?|```\s*$", "", str(text).strip(), flags=re.M).strip()
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(cleaned[start:end + 1])
        except Exception:
            pass
    return {"style_header": "", "segments": []}


_GLOBAL_PREAMBLE = re.compile(
    r"^\s*[【\[]?(?:全局|总体|统一|视觉风格|画面风格|风格基调|角色设定|"
    r"人物设定|主体设定|固定场景设定|世界观设定)[】\]]?\s*[:：]",
    re.IGNORECASE)
_STRUCTURAL_HEADING = re.compile(
    r"^\s*(?:第?\s*[0-9一二三四五六七八九十]+\s*[幕场镜段]|"
    r"(?:镜头|场景|分镜|shot|scene)\s*[0-9一二三四五六七八九十]+)"
    r"\s*[:：、.．-]?\s*$",
    re.IGNORECASE)
_HARD_CUT_CUE = re.compile(
    r"^\s*(?:第?\s*[0-9一二三四五六七八九十]+\s*[幕场镜]|"
    r"(?:镜头|场景|分镜|shot|scene)\s*[0-9一二三四五六七八九十]+|"
    r"与此同时|另一边|画面切换|切换到|转场|次日|翌日|数日后|多年后)",
    re.IGNORECASE)
_CHARACTER_REFERENCE_CUE = re.compile(
    r"人物|角色|主角|女主|男主|少女|少年|女孩|男孩|女性|男性|脸|面部|"
    r"五官|发型|服装|character|person|girl|boy|woman|man|face|portrait",
    re.IGNORECASE)
_PICTURE_TAG = re.compile(r"@图片\s*(\d+)|<Picture\s+(\d+)>", re.IGNORECASE)


def _extract_global_preamble(text):
    """Separate explicit global-setting lines from the chronological body."""
    global_lines, body_lines = [], []
    for line in str(text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        if _GLOBAL_PREAMBLE.search(stripped):
            global_lines.append(stripped)
        else:
            body_lines.append(stripped)
    # A one-line prompt can begin with "角色设定：" and still contain the whole
    # action. Never remove the only timeline material in that case.
    if not body_lines:
        return "", str(text or "").strip()
    return "\n".join(global_lines), "\n".join(body_lines)


def _script_units(text):
    """Return ordered, non-empty screenplay units without requiring an LLM."""
    normalized = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return []
    # Split Chinese/English sentence endings, but keep punctuation inside a
    # <d>...</d> line attached until the closing dialogue tag.
    raw = re.split(r"(?<=[。！？!?；;])(?!</d>)|\n+", normalized,
                   flags=re.IGNORECASE)
    units, heading = [], ""
    for part in raw:
        item = str(part or "").strip()
        if not item:
            continue
        if _STRUCTURAL_HEADING.match(item):
            heading = "%s\n%s" % (heading, item) if heading else item
            continue
        if heading:
            item = "%s\n%s" % (heading, item)
            heading = ""
        units.append(item)
    if heading:
        if units:
            units[-1] = "%s\n%s" % (units[-1], heading)
        else:
            units.append(heading)
    return units


def _safe_unit_bisect(unit):
    """Split one oversized unit near its middle without cutting an XML tag."""
    text = str(unit or "").strip()
    if len(text) < 2:
        return None
    midpoint = len(text) / 2.0
    candidates = []
    for match in re.finditer(r"[，,、：:]\s*|\s+", text):
        pos = match.end()
        if pos <= 0 or pos >= len(text):
            continue
        # Do not split between '<' and '>' in Easy Prompt markup.
        if text.rfind("<", 0, pos) > text.rfind(">", 0, pos):
            continue
        candidates.append(pos)
    if candidates:
        split_at = min(candidates, key=lambda pos: abs(pos - midpoint))
    else:
        split_at = max(1, min(len(text) - 1, int(round(midpoint))))
        if text.rfind("<", 0, split_at) > text.rfind(">", 0, split_at):
            next_close = text.find(">", split_at)
            if 0 <= next_close < len(text) - 1:
                split_at = next_close + 1
    left, right = text[:split_at].strip(), text[split_at:].strip()
    return (left, right) if left and right else None


def _ensure_unit_count(units, count, original_text):
    """Create enough ordered units for N distinct prompts."""
    result = list(units)
    while len(result) < int(count):
        if not result:
            break
        index = max(range(len(result)), key=lambda i: len(result[i]))
        pieces = _safe_unit_bisect(result[index])
        if pieces is None:
            break
        result[index:index + 1] = list(pieces)
    if not result:
        result = [str(original_text or "").strip()]
    # Only extremely short prompts reach this branch. Phase labels keep the
    # prompts distinct instead of silently duplicating one identical shot.
    while len(result) < int(count):
        phase = len(result) + 1
        result.append("时间线推进到第 %d/%d 阶段。%s" % (
            phase, int(count), str(original_text or "").strip()))
    return result


def _balanced_script_chunks(units, count):
    """Partition ordered units into exactly N near-equal contiguous chunks."""
    count = max(1, int(count))
    source = list(units)
    groups, start = [], 0
    for group_index in range(count):
        groups_left = count - group_index
        if groups_left == 1:
            end = len(source)
        else:
            maximum_end = len(source) - (groups_left - 1)
            target = sum(max(1, len(item)) for item in source[start:]) / groups_left
            end, accumulated = start, 0
            while end < maximum_end:
                weight = max(1, len(source[end]))
                if end > start and abs(accumulated - target) <= abs(
                        accumulated + weight - target):
                    break
                accumulated += weight
                end += 1
            end = max(start + 1, end)
        groups.append("\n".join(source[start:end]).strip())
        start = end
    return groups


def _persistent_character_picture_tags(text, media_manifest):
    """Find image tags that are explicitly described as character references."""
    tags = set()
    source = str(text or "")
    for match in _PICTURE_TAG.finditer(source):
        window = source[max(0, match.start() - 120):match.end() + 160]
        if _CHARACTER_REFERENCE_CUE.search(window):
            tags.add(int(match.group(1) or match.group(2)))

    # The manifest heading is generic and mentions characters for every image;
    # classify only its subject/description lines, not that generic heading.
    blocks = re.split(r"(?=^- <Picture\s+\d+>)", str(media_manifest or ""),
                      flags=re.MULTILINE | re.IGNORECASE)
    for block in blocks:
        match = re.match(r"^- <Picture\s+(\d+)>[^\n]*", block,
                         flags=re.IGNORECASE)
        if not match:
            continue
        first_line = match.group(0)
        meaningful_head = first_line.split("，主体名：", 1)[-1] if "，主体名：" in first_line else ""
        detail_lines = "\n".join(block.splitlines()[1:])
        if _CHARACTER_REFERENCE_CUE.search("%s\n%s" % (meaningful_head, detail_lines)):
            tags.add(int(match.group(1)))
    return ["@图片%d" % ordinal for ordinal in sorted(tags)]


def _local_timeline_split(text, count, media_manifest="", reason="",
                          maximum_seconds=None, total_seconds=None,
                          overlap_frames=22, fps=24.0):
    """Deterministic fail-safe: distinct chronological prompts, no model call."""
    style_header, body = _extract_global_preamble(text)
    units = _ensure_unit_count(_script_units(body), count, body)
    chunks = _balanced_script_chunks(units, count)
    manifest = _agent_manifest_entries(media_manifest)
    character_keys = _character_media_keys(text, media_manifest, manifest)
    segments = []
    for index, chunk in enumerate(chunks, start=1):
        transition = "开场" if index == 1 else (
            "切镜" if _HARD_CUT_CUE.search(chunk) else "承接")
        selected, required_refs = _segment_media_bindings(
            chunk, {}, manifest, character_keys)
        segment_style = _filter_segment_preamble(
            style_header, selected, manifest)
        pieces = []
        if segment_style:
            pieces.append(segment_style)
        if index > 1 and transition == "承接":
            pieces.append("画面与动作自然承接上一段结尾。")
        pieces.append(chunk)
        prompt = "\n\n".join(piece for piece in pieces if piece).strip()
        prompt, _restored = _inject_missing_character_refs(
            prompt, required_refs)
        segments.append({
            "index": index,
            "title": _short_segment_title(chunk),
            "duration_seconds": float(maximum_seconds or 5.0),
            "transition": transition,
            "brief": re.sub(r"\s+", " ", chunk)[:120],
            "prompt": prompt,
        })
    if maximum_seconds is not None and total_seconds is not None:
        segments = _normalize_storyboard_durations(
            segments, total_seconds, maximum_seconds, overlap_frames, fps)
        segments = _apply_subject_ledger(chunks, segments, media_manifest, text)
    return {
        "style_header": style_header,
        "segments": segments,
        "split_source": "local_fallback",
        "split_fallback_reason": str(reason or "LLM 未返回可用分段")[:1000],
    }


def _build_split_source_packets(text, count, media_manifest, seconds):
    """Pre-aggregate chronology and persistent references before the LLM call."""
    style_header, body = _extract_global_preamble(text)
    units = _ensure_unit_count(_script_units(body), count, body)
    chunks = _balanced_script_chunks(units, count)
    manifest = _agent_manifest_entries(media_manifest)
    character_keys = _character_media_keys(text, media_manifest, manifest)
    lines = [
        "【程序已预聚合的分段内容包｜不要重新切分、合并或移动剧情】",
        "下面每段的“源剧情”已经按原始顺序完整分配。你只负责把各段分别展开成技能规定的完整 H3 prompt。",
        "每段都必须独立交代构图与主体位置、人物外观/姿态/视线、环境与光线、动作准备到结果、"
        "表情反应、运镜目标/幅度/速度、同步声音/台词和实际生效的素材标签。",
    ]
    if style_header:
        lines.append("全局设定（每段继承）：" + style_header)
    if manifest:
        lines.append(
            "公共素材只是候选库，不是每段必用清单；每段只保留实际出镜主体或真正生效的素材。")
    for index, chunk in enumerate(chunks, 1):
        transition = "开场" if index == 1 else ("切镜" if _HARD_CUT_CUE.search(chunk) else "承接")
        continuity = ("建立完整开场状态，不依赖前文。" if index == 1 else
                      "直接建立新场景、机位和人物状态，不写黑场/闪白。" if transition == "切镜" else
                      "从上一段结尾的机位、姿态、环境和动作惯性自然接续。")
        selected, refs = _segment_media_bindings(
            chunk, {}, manifest, character_keys)
        allowed = "、".join(_entry_output_tag(entry) for entry in selected) or "无"
        ref_text = "、".join(refs) if refs else "本段无须定义未出镜主体"
        lines.extend(["", f"=== 第 {index}/{int(count)} 段｜{transition}｜目标 {float(seconds):.2f} 秒 ===",
                      "源剧情（全部保留，不得概括掉动作或台词）：", chunk,
                      "连续性职责：" + continuity,
                      "本段允许素材：" + allowed,
                      "人物参考职责：" + ref_text,
                      "写作职责：先建立可见画面，再展开动作因果与反应；素材只在真正出现/生效处写标签；"
                      "声音事件绑定到发生时刻；不要把画面正文缩成一句剧情摘要。"])
    return "\n".join(lines), chunks


def _agent_manifest_entries(media_manifest):
    """Convert the Director's rendered whitelist back to Media Agent entries."""
    entries = []
    current = None
    kind_map = {"picture": "image", "video": "video", "audio": "audio"}
    for raw_line in str(media_manifest or "").splitlines():
        line = raw_line.strip()
        match = re.match(
            r"^-\s*<(Picture|Video|Audio)\s+(\d+)>\s*(?:或\s*@[^：:]+)?\s*[：:]?\s*(.*)$",
            line, flags=re.IGNORECASE)
        if match:
            kind, ordinal, tail = match.groups()
            current = {
                "tag": "<%s %d>" % (kind.title(), int(ordinal)),
                "type": kind_map[kind.casefold()],
                "ordinal": int(ordinal),
            }
            subject = re.search(r"主体名[：:]\s*([^，,]+)", tail)
            if subject:
                current["subject_name"] = subject.group(1).strip()
            filename = re.search(r"文件[：:]\s*([^，,]+)", tail)
            if filename:
                current["filename"] = filename.group(1).strip()
            entries.append(current)
            continue
        if current and line.startswith(("画面内容：", "语音内容：")):
            current["description"] = line.split("：", 1)[-1].strip()
    return entries


_ANY_MEDIA_TAG = re.compile(
    r"(?:@(图片|视频|音频)\s*(\d+)|<\s*(Picture|Image|Video|Audio)\s+(\d+)\s*>)",
    re.IGNORECASE)
_MEDIA_KIND_MAP = {
    "图片": "image", "picture": "image", "image": "image",
    "视频": "video", "video": "video",
    "音频": "audio", "audio": "audio",
}
_MEDIA_AT_NAME = {"image": "图片", "video": "视频", "audio": "音频"}


def _media_key(kind, ordinal):
    normalized = _MEDIA_KIND_MAP.get(str(kind or "").strip().casefold())
    try:
        number = int(ordinal)
    except (TypeError, ValueError):
        return None
    return (normalized, number) if normalized and number > 0 else None


def _entry_media_key(entry):
    return _media_key(entry.get("type"), entry.get("ordinal"))


def _media_keys_in_text(value):
    keys = set()
    for match in _ANY_MEDIA_TAG.finditer(str(value or "")):
        kind = match.group(1) or match.group(3)
        ordinal = match.group(2) or match.group(4)
        key = _media_key(kind, ordinal)
        if key:
            keys.add(key)
    return keys


def _entry_output_tag(entry):
    key = _entry_media_key(entry)
    if not key:
        return str(entry.get("tag") or "").strip()
    return "@%s%d" % (_MEDIA_AT_NAME[key[0]], key[1])


def _character_media_keys(text, media_manifest, manifest=None):
    """Return only image keys that have evidence of being character refs."""
    entries = manifest if manifest is not None else _agent_manifest_entries(media_manifest)
    by_tag = {
        _entry_output_tag(entry): _entry_media_key(entry)
        for entry in entries if _entry_media_key(entry)
    }
    return {
        by_tag[tag] for tag in _persistent_character_picture_tags(text, media_manifest)
        if tag in by_tag
    }


def _subject_aliases(subject_name):
    """Conservative aliases for labels such as '桃乐丝角色立绘/正面照'."""
    name = re.sub(r"\s+", "", str(subject_name or "").strip())
    if not name:
        return []
    aliases = [name]
    stripped = re.sub(
        r"(?:角色)?(?:正面|侧面|背面|全身|半身|头像|立绘|设定|参考|素材|照片|图片|图)+$",
        "", name)
    if len(stripped) >= 2 and stripped != name:
        aliases.append(stripped)
    return aliases


def _storyboard_context(item):
    if not isinstance(item, dict):
        return ""
    values = [str(item.get("segment_goal") or "")]
    for subject in item.get("subjects") or []:
        if isinstance(subject, dict):
            values.extend(str(subject.get(key) or "") for key in (
                "name", "media", "state_action", "declaration"))
    for shot in item.get("shots") or []:
        if not isinstance(shot, dict):
            continue
        for key in ("composition", "subjects", "environment_lighting",
                    "action", "action_progression", "camera", "sound",
                    "sound_dialogue", "media_roles"):
            values.append(str(shot.get(key) or ""))
    return "\n".join(values)


def _storyboard_media_keys(item):
    keys = set()
    if not isinstance(item, dict):
        return keys
    for subject in item.get("subjects") or []:
        if isinstance(subject, dict):
            keys.update(_media_keys_in_text(subject.get("media") or ""))
    for shot in item.get("shots") or []:
        if not isinstance(shot, dict):
            continue
        for tag in shot.get("media") or []:
            keys.update(_media_keys_in_text(tag))
    return keys


def _segment_media_bindings(chunk, plan_item, manifest, character_keys):
    """Select candidate media for one segment; an empty selection is valid.

    Explicit source references, storyboard choices and an exact subject-name
    match are the only inclusion paths. Merely being connected as a shared
    asset never makes an item mandatory for every segment.
    """
    context = "%s\n%s" % (str(chunk or ""), _storyboard_context(plan_item))
    selected_keys = _media_keys_in_text(chunk) | _storyboard_media_keys(plan_item)
    compact_context = re.sub(r"\s+", "", context).casefold()
    for entry in manifest:
        key = _entry_media_key(entry)
        if not key:
            continue
        aliases = _subject_aliases(entry.get("subject_name"))
        if any(alias.casefold() in compact_context for alias in aliases):
            selected_keys.add(key)
    selected = [entry for entry in manifest
                if _entry_media_key(entry) in selected_keys]
    required_refs = [
        _entry_output_tag(entry) for entry in selected
        if _entry_media_key(entry) in character_keys
    ]
    return selected, required_refs


def _subject_id(entry):
    key = _entry_media_key(entry)
    return "%s_%d" % key if key else ""


def _subject_for_entry(entry, planned=None, previous=None):
    """Combine immutable media identity with a segment-local mutable state."""
    planned = planned if isinstance(planned, dict) else {}
    previous = previous if isinstance(previous, dict) else {}
    name = str(planned.get("name") or entry.get("subject_name") or
               _entry_output_tag(entry)).strip()
    action = str(planned.get("state_action") or "").strip()
    if action not in ("首次", "延续", "变化"):
        action = "延续" if previous else "首次"
    declaration = str(planned.get("declaration") or "").strip()
    if action == "延续" and previous and not declaration:
        declaration = str(previous.get("declaration") or "").strip()
    if not declaration:
        declaration = str(entry.get("description") or "").strip()
    previous_revision = int(previous.get("state_revision") or previous.get("revision") or 0)
    revision = (previous_revision + 1 if action == "变化"
                else previous_revision or 1)
    return {
        "id": _subject_id(entry),
        "name": name,
        "media": [_entry_output_tag(entry)],
        "identity": str(entry.get("description") or name).strip(),
        "state_action": action,
        "state_revision": revision,
        "declaration": declaration,
    }


_SUBJECT_CONTINUATION_CUE = re.compile(
    r"(?:她|他|它|其|她们|他们|两人|二人|众人|少女|少年|女孩|男孩|"
    r"主角|主人公|角色)(?:又|仍|继续|随即|随后|此时|转身|走|跑|看|说|拿|穿|换|受|被|将|把|的|在|从|向|，|。|\s)")
_SUBJECT_ABSENCE_CUE = re.compile(
    r"(?:无人|空镜|没有人物|不出现人物|所有人物离场|角色离场|人物离场)")
_SUBJECT_STATE_CHANGE_CUE = re.compile(
    r"(?:换上|换成|更换|穿上|脱下|摘下|戴上|拿起|放下|丢下|交给|"
    r"受伤|流血|淋湿|弄脏|撕裂|恢复|变身|变成|长出|剪短|染成|老去|年轻化)")


def _apply_subject_ledger(chunks, storyboard, media_manifest, text):
    """Attach only visible subjects and carry explicit state changes forward."""
    manifest = _agent_manifest_entries(media_manifest)
    character_keys = _character_media_keys(text, media_manifest, manifest)
    entries_by_key = {_entry_media_key(entry): entry for entry in manifest
                      if _entry_media_key(entry)}
    ledger = {}
    previous_visible_character_keys = []
    normalized = []
    for chunk, raw_item in zip(chunks, storyboard):
        item = dict(raw_item)
        selected, _refs = _segment_media_bindings(
            chunk, item, manifest, character_keys)
        selected_by_key = {_entry_media_key(entry): entry for entry in selected}
        transition = str(item.get("transition") or "承接")
        explicit_character_keys = {
            key for key in selected_by_key if key in character_keys
        }
        # A planner can omit SUBJECT on a pronoun-only continuation. Inherit
        # only the immediately preceding visible character set, never on a cut
        # or an explicit empty shot, so candidate media still does not leak
        # into unrelated segments.
        if (not explicit_character_keys and transition != "切镜"
                and not _SUBJECT_ABSENCE_CUE.search(str(chunk or ""))
                and _SUBJECT_CONTINUATION_CUE.search(str(chunk or ""))):
            for key in previous_visible_character_keys:
                entry = entries_by_key.get(key)
                if entry is not None:
                    selected_by_key[key] = entry
        requested = {}
        for subject in item.get("subjects") or []:
            if not isinstance(subject, dict):
                continue
            keys = _media_keys_in_text(subject.get("media") or "")
            key = next((key for key in keys if key in selected_by_key), None)
            if key is None:
                compact_name = re.sub(r"\s+", "", str(subject.get("name") or "")).casefold()
                key = next((candidate for candidate, entry in selected_by_key.items()
                            if compact_name and any(alias.casefold() == compact_name
                                for alias in _subject_aliases(entry.get("subject_name")))), None)
            if key is not None:
                requested[key] = subject
        subjects = []
        for key, entry in selected_by_key.items():
            # Subject declarations are for identifiable image-backed entities.
            # Scene/video/audio material remains a shot-level binding.
            if key not in character_keys and not entry.get("subject_name"):
                continue
            planned = requested.get(key)
            previous = ledger.get(_subject_id(entry))
            if (not planned and previous
                    and _SUBJECT_STATE_CHANGE_CUE.search(str(chunk or ""))):
                planned = {
                    "name": previous.get("name") or entry.get("subject_name"),
                    "state_action": "变化",
                    "declaration": re.sub(r"\s+", " ", str(chunk or "")).strip()[:160],
                }
            subject = _subject_for_entry(entry, planned, previous)
            ledger[subject["id"]] = subject
            subjects.append(subject)
        item["subjects"] = subjects
        current_visible = [
            _entry_media_key(entry) for entry in selected_by_key.values()
            if _entry_media_key(entry) in character_keys
        ]
        if current_visible:
            previous_visible_character_keys = current_visible
        elif transition == "切镜":
            previous_visible_character_keys = []
        normalized.append(item)
    return normalized


def _filter_segment_preamble(style_header, selected_entries, manifest):
    """Drop global character-setting lines for subjects absent from a segment."""
    source = str(style_header or "").strip()
    if not source:
        return ""
    selected_keys = {_entry_media_key(entry) for entry in selected_entries}
    lines = []
    for line in source.splitlines():
        line_keys = _media_keys_in_text(line)
        if line_keys and not line_keys.issubset(selected_keys):
            continue
        mentions_unselected_subject = False
        compact = re.sub(r"\s+", "", line).casefold()
        for entry in manifest:
            if _entry_media_key(entry) in selected_keys:
                continue
            if any(alias.casefold() in compact
                   for alias in _subject_aliases(entry.get("subject_name"))):
                mentions_unselected_subject = True
                break
        if not mentions_unselected_subject:
            lines.append(line)
    return "\n".join(lines).strip()


DIRECTOR_STORYBOARD_PLANNER_SYSTEM = """你是长视频分镜规划师，只规划，不写最终视频提示词，也不执行写作技能。
程序按“单段时长上限”算出了足够覆盖全片的段位，并给出有序源剧情包。段数与顺序不可改变，
但每段时长不是固定值：你要根据动作完整性、台词长度、镜头节奏和剧情密度，为每段选择不同的合理时长。
每段 DURATION 必须大于 0 且不得超过用户给出的上限；全部段落可见时长之和应尽量接近目标总时长。
段间衔接由你根据源剧情逐段判断：同一时空中动作、机位或人物状态连续时写“承接”；
明确换了场景、时间、主体或叙事视角时写“切镜”；第 1 段固定写“开场”。
“切镜”只表示下一段直接建立新画面，不要规划黑场、闪白、淡入淡出或转场特效。
主体必须按段管理。公共素材只是候选库：本段没出现的角色、场景或物体，不得写 SUBJECT，也不得引用其素材。
同一主体跨段出现时，名字与不可变身份特征保持一致；服装、年龄状态、伤痕、湿润程度、持有物等可变状态
只有在剧情明确变化时才能更新，并从变化发生的段落起写出新的状态。不要把所有主体复制到每一段。
不要输出 JSON。每段严格使用下面这种易读文本块，SEGMENT 编号必须连续：
[SEGMENT 1]
TITLE: 雨夜相遇
DURATION: 6.50
TRANSITION: 开场
GOAL: 本段叙事目标
SUBJECT: 桃乐丝 || @图片1 || 首次 || 不可变身份与本段服装、姿态、状态的可见声明
SHOT: 0.00-3.00 || 构图与主体位置 || 可见动作过程 || 运镜 || 声音/台词 || @图片1、@视频1

TITLE 是镜头卡片的中文短标题，必须概括本段关键动作或事件且不超过 8 个汉字。
每段可写 0-6 行 SUBJECT、1-4 行 SHOT。SUBJECT 第三栏只能写“首次”“延续”或“变化”；
没有素材时 SHOT 最后一栏写“无”。媒体标签只能从用户提供的清单中选择。
公共素材清单是候选库，不是必用清单。逐镜判断实际出现的人物、场景、动作和声音，只选择真正生效的素材；
不要为了使用素材而让未出镜人物进入画面，也不要给未出镜人物编写主体定义。允许某个镜头或整段完全不引用素材。"""


def _parse_storyboard_response(raw, expected_count):
    """Accept either legacy JSON or the planner's forgiving text blocks."""
    payload = _loads_loose(raw)
    planned = payload.get("segments") if isinstance(payload, dict) else None
    if isinstance(planned, list) and len(planned) == int(expected_count):
        return planned

    source = re.sub(r"^\s*```(?:text|markdown)?|```\s*$", "",
                    str(raw or "").strip(), flags=re.MULTILINE).strip()
    heading = re.compile(
        r"(?im)^\s*(?:#{1,6}\s*)?(?:\[\s*SEGMENT\s+(\d+)\s*\]|"
        r"=+\s*SEGMENT\s+(\d+)\s*=+|第\s*(\d+)\s*段\s*[:：]?)\s*$")
    matches = list(heading.finditer(source))
    if len(matches) != int(expected_count):
        return []
    result = []
    for offset, match in enumerate(matches):
        number = next((int(value) for value in match.groups() if value), 0)
        if number != offset + 1:
            return []
        end = matches[offset + 1].start() if offset + 1 < len(matches) else len(source)
        body = source[match.end():end].strip()
        transition_match = re.search(
            r"(?im)^\s*(?:TRANSITION|转场|衔接)\s*[:：]\s*(开场|承接|切镜)\s*$",
            body)
        goal_match = re.search(
            r"(?im)^\s*(?:GOAL|SEGMENT_GOAL|叙事目标|本段目标)\s*[:：]\s*(.+?)\s*$",
            body)
        title_match = re.search(
            r"(?im)^\s*(?:TITLE|标题|分镜标题)\s*[:：]\s*(.+?)\s*$", body)
        duration_match = re.search(
            r"(?im)^\s*(?:DURATION|时长|持续时间)\s*[:：]\s*([0-9]+(?:\.[0-9]+)?)\s*(?:s|秒)?\s*$",
            body)
        subjects = []
        for subject_match in re.finditer(
                r"(?im)^\s*(?:SUBJECT|主体)\s*\d*\s*[:：]\s*(.+?)\s*$", body):
            fields = [field.strip() for field in re.split(
                r"\s*\|\|\s*|\s+\|\s+", subject_match.group(1))]
            fields += [""] * (4 - len(fields))
            subjects.append({
                "name": fields[0], "media": fields[1],
                "state_action": fields[2], "declaration": fields[3],
            })
        shots = []
        for shot_match in re.finditer(
                r"(?im)^\s*(?:SHOT|镜头)\s*\d*\s*[:：]\s*(.+?)\s*$", body):
            fields = [field.strip() for field in re.split(
                r"\s*\|\|\s*|\s+\|\s+", shot_match.group(1))]
            fields += [""] * (6 - len(fields))
            media = [] if not fields[5] or fields[5] in ("无", "none", "N/A") else [
                item.strip() for item in re.split(r"[、,，]", fields[5]) if item.strip()]
            shots.append({
                "time": fields[0], "composition": fields[1],
                "action": fields[2], "camera": fields[3],
                "sound": fields[4], "media": media,
            })
        result.append({
            "index": number,
            "title": _sanitize_storyboard_title(
                title_match.group(1) if title_match else "", body),
            "duration_seconds": (
                float(duration_match.group(1)) if duration_match else 0.0),
            "transition": transition_match.group(1) if transition_match else "",
            "segment_goal": goal_match.group(1).strip() if goal_match else body,
            "subjects": subjects[:6],
            "shots": shots[:4],
        })
    return result


def _plan_director_storyboard(
    chunks, count, maximum_seconds, total_seconds, overlap_frames, fps,
    media_manifest, llm_service, ollama_auto_unload, seed,
):
    """Plan fixed slots with AI-paced durations; fall back locally on mismatch."""
    source = [
        "段落数：%d；目标总时长：%.2f 秒；单段时长上限：%.2f 秒；"
        "MotionContext 重叠：%d 帧；帧率：%.2f。"
        "以下源剧情已按顺序预聚合，段数和顺序不可改变，但每段时长必须由你按内容决定。" % (
            int(count), float(total_seconds), float(maximum_seconds),
            int(overlap_frames), float(fps))]
    for index, chunk in enumerate(chunks, 1):
        source.extend(["", "=== SEGMENT %d/%d ===" % (index, int(count)), chunk])
    if media_manifest:
        source.extend(["", "可用素材标签：", _compact_media_manifest(media_manifest)])
    notes = []
    try:
        raw = call_llm(
            llm_service, "\n".join(source), DIRECTOR_STORYBOARD_PLANNER_SYSTEM,
            ollama_auto_unload, int(seed) + 53, max_tokens=None)
        planned = _parse_storyboard_response(raw, count)
    except Exception as error:  # noqa: BLE001 - local plan is always available
        if type(error).__name__ == "InterruptProcessingException":
            raise
        notes.append("分镜规划调用失败，采用本地自适应规划：%s" % str(error)[:160])
        planned = []
    if not isinstance(planned, list) or len(planned) != int(count) or not all(
            isinstance(item, dict) for item in planned):
        returned = len(planned) if isinstance(planned, list) else 0
        notes.append(
            "分镜规划返回 %d 段而目标为 %d 段，采用本地自适应规划；不重试" % (
                returned, int(count)))
        planned = []
        for index, chunk in enumerate(chunks, 1):
            planned.append({
                "index": index,
                "title": _sanitize_storyboard_title("", chunk),
                "duration_seconds": float(maximum_seconds),
                "transition": "开场" if index == 1 else (
                    "切镜" if _HARD_CUT_CUE.search(chunk) else "承接"),
                "segment_goal": re.sub(r"\s+", " ", chunk).strip(),
                "subjects": [],
                "shots": [],
            })
        planned = _normalize_storyboard_durations(
            planned, total_seconds, maximum_seconds, overlap_frames, fps)
        return planned, "local_adaptive_timing", notes
    normalized = []
    for index, (item, chunk) in enumerate(zip(planned, chunks), 1):
        transition = str(item.get("transition") or "").strip()
        if index == 1:
            transition = "开场"
        elif transition not in ("承接", "切镜"):
            transition = "切镜" if _HARD_CUT_CUE.search(chunk) else "承接"
        shots = item.get("shots") if isinstance(item.get("shots"), list) else []
        normalized.append({
            "index": index,
            "title": _sanitize_storyboard_title(item.get("title"), chunk),
            "duration_seconds": item.get("duration_seconds") or maximum_seconds,
            "transition": transition,
            "segment_goal": str(item.get("segment_goal") or chunk).strip(),
            "subjects": [subject for subject in (item.get("subjects") or [])
                         if isinstance(subject, dict)][:6],
            "shots": [shot for shot in shots if isinstance(shot, dict)][:4],
        })
    normalized = _normalize_storyboard_durations(
        normalized, total_seconds, maximum_seconds, overlap_frames, fps)
    return normalized, "llm_adaptive_timing", notes


_TITLE_METADATA_CUE = re.compile(
    r"(?:人物|角色)?外观参考|画面基准|整体视听|声音与配乐|详细分镜|"
    r"电影级提示词|生成提示词|脚本与|剧本|纯中文版|全局设定|输出格式|"
    r"integrated_multimodal_description|overall_soundscape|non_diegetic_music",
    re.IGNORECASE)


def _sanitize_storyboard_title(title, chunk=""):
    """Return card-only event metadata, never a slice of a script header."""
    sources = [str(title or ""), str(chunk or "")]
    for source_index, source in enumerate(sources):
        for raw_line in source.splitlines() or [source]:
            line = str(raw_line or "").strip()
            if not line or re.fullmatch(r"[\s=_*#~\-—·|]+", line):
                continue
            line = re.sub(r"^\s*(?:#{1,6}\s*)?", "", line)
            line = re.sub(
                r"^\s*(?:(?:TITLE|标题|分镜标题|镜头标题)\s*[:：]|"
                r"(?:第?\s*\d+\s*(?:段|镜|镜头|分镜))\s*[:：、.\-]*)\s*",
                "", line, flags=re.IGNORECASE)
            if not line:
                continue
            if re.match(
                    r"^(?:DURATION|TRANSITION|GOAL|SEGMENT_GOAL|SUBJECT\s*\d*|SHOT\s*\d*|"
                    r"时长|持续时间|转场|衔接|叙事目标|本段目标|主体\s*\d*|镜头\s*\d*)\s*[:：]",
                    line, flags=re.IGNORECASE):
                continue
            # Decorative programme/script headings are not shot titles. A
            # planner-provided clean TITLE is still accepted before this gate.
            if _TITLE_METADATA_CUE.search(line):
                continue
            if "《" in line and "》" in line and (
                    source_index > 0 or re.search(r"\d+\s*秒|[上下中]篇|篇\s*\d+", line)):
                continue
            line = re.sub(r"<[^>]+>|@[\u4e00-\u9fffA-Za-z]+\s*\d+", "", line)
            line = re.sub(
                r"\[(?:Shot\s*\d+|Chinese|English|Japanese|Korean)\]",
                "", line, flags=re.IGNORECASE)
            line = re.sub(r"[\[\]【】]", "", line)
            line = re.sub(r"\([^)]*(?:秒|帧|s\b)[^)]*\)|（[^）]*(?:秒|帧|s\b)[^）]*）", "", line,
                          flags=re.IGNORECASE)
            line = re.split(r"\s*(?:\|\||[|]|={2,}|—{2,})\s*", line, maxsplit=1)[0]
            line = re.split(r"[，,。！？!?；;\n]", line, maxsplit=1)[0]
            line = re.sub(r"^[\s:=：,，.。\-—·]+|[\s:=：,，.。\-—·]+$", "", line)
            compact = re.sub(r"[^\u3400-\u9fffA-Za-z0-9]+", "", line)
            if len(compact) >= 2:
                return compact[:8]
    return "剧情推进"


def _short_segment_title(value):
    """Backward-compatible title helper used by local planning fallbacks."""
    return _sanitize_storyboard_title("", value)


def _storyboard_text(item):
    lines = [
        "本段分镜规划：",
        "本段实际时长：%.3f 秒（%d 帧）" % (
            float(item.get("duration_seconds") or 0), int(item.get("frames") or 0)),
        "叙事目标：" + str(item.get("segment_goal") or ""),
    ]
    subjects = item.get("subjects") or []
    if subjects:
        lines.append("本段主体声明（只声明实际出现的主体）：")
        for subject in subjects:
            lines.append("- %s｜%s｜状态版本 %s｜%s｜参考 %s" % (
                str(subject.get("name") or subject.get("id") or "主体"),
                str(subject.get("state_action") or "延续"),
                str(subject.get("state_revision") or 1),
                str(subject.get("declaration") or subject.get("identity") or "保持参考素材身份"),
                "、".join(subject.get("media") or []) or "无"))
    else:
        lines.append("本段没有需要声明的素材主体；禁止带入其他段人物。")
    shots = item.get("shots") or []
    if not shots:
        lines.append("镜头安排：根据本段源剧情按实际时长自然规划，完整覆盖动作起因、过程和结果。")
        return "\n".join(lines)
    for index, shot in enumerate(shots, 1):
        media = "、".join(str(tag) for tag in (shot.get("media") or [])) or "无指定素材"
        lines.append(
            "镜头%d｜时间 %s｜构图 %s｜动作 %s｜运镜 %s｜声音 %s｜素材 %s" % (
                index, str(shot.get("time") or "按本段时长安排"),
                str(shot.get("composition") or "按剧情建立"),
                str(shot.get("action") or "按源剧情展开"),
                str(shot.get("camera") or "按动作目标安排"),
                str(shot.get("sound") or "按源剧情安排"), media))
    return "\n".join(lines)


def _segment_writer_briefs(text, chunks, media_manifest, seconds, storyboard=None):
    """Build one chronology-owned brief per segment for the shared Agent writer."""
    style_header, _body = _extract_global_preamble(text)
    manifest = _agent_manifest_entries(media_manifest)
    character_keys = _character_media_keys(text, media_manifest, manifest)
    briefs = []
    for index, chunk in enumerate(chunks, 1):
        plan_item = storyboard[index - 1] if storyboard and index <= len(storyboard) else {}
        transition = str(plan_item.get("transition") or "") or (
            "开场" if index == 1 else (
                "切镜" if _HARD_CUT_CUE.search(chunk) else "承接"))
        continuity = (
            "建立完整开场状态。" if index == 1 else
            "直接建立新场景、机位和人物状态。" if transition == "切镜" else
            "承接上一段结尾的机位、人物姿态、环境状态和动作惯性。")
        selected_manifest, character_refs = _segment_media_bindings(
            chunk, plan_item, manifest, character_keys)
        segment_style = _filter_segment_preamble(
            style_header, selected_manifest, manifest)
        selected_tags = [_entry_output_tag(entry) for entry in selected_manifest]
        segment_seconds = float(
            plan_item.get("duration_seconds") or seconds)
        parts = [
            "这是长片第 %d/%d 段，只写这一段约 %.2f 秒的最终视频提示词。"
            "不要输出 JSON，不要生成其他段，不要改变段数。" % (
                index, len(chunks), segment_seconds),
        ]
        if segment_style:
            parts.append("本段适用的全片共同设定：" + segment_style)
        parts.extend([
            "本段源剧情（动作和台词全部保留）：\n" + chunk,
            "段间关系：" + transition + "；" + continuity,
            _storyboard_text(plan_item),
        ])
        if selected_tags:
            parts.append(
                "本段素材白名单（仅限这些，按实际生效位置引用）：" + "、".join(selected_tags) +
                "。没有出镜或没有实际作用的素材不要写。")
        else:
            parts.append(
                "本段不需要任何公共素材：不要引用素材标签，也不要定义未出镜主体。")
        if character_refs:
            parts.append("本段实际出镜人物的外观参考：" + "、".join(character_refs))
        subjects = plan_item.get("subjects") or []
        if subjects:
            parts.append(
                "主体一致性账本：不可变身份以素材和 identity 为准；可变服装/伤痕/持有物/情绪"
                "只采用本段 state declaration。不要沿用已经被本段明确改变的旧状态。\n" +
                "\n".join("- %s：identity=%s；state_v%s(%s)=%s；media=%s" % (
                    subject.get("name") or subject.get("id"),
                    subject.get("identity") or "按参考素材",
                    subject.get("state_revision") or 1,
                    subject.get("state_action") or "延续",
                    subject.get("declaration") or "保持上一状态",
                    "、".join(subject.get("media") or []))
                    for subject in subjects))
        parts.append(
            "请像 Media Agent 一样直接输出这一段的最终提示词正文，"
            "根据写作技能展开构图、主体、环境、动作、运镜和声音。"
            "subject_definitions/主体定义只写本段实际出现且已选中参考素材的主体。"
            "分镜标题只属于导演台卡片元数据，绝对不要在提示词正文中输出 TITLE、标题或‘分镜N：标题’行。")
        briefs.append({
            "index": index,
            "transition": transition,
            "brief": re.sub(r"\s+", " ", chunk).strip()[:120],
            "writer_input": "\n\n".join(parts),
            "manifest": selected_manifest,
            "selected_media_tags": selected_tags,
            "required_character_tags": character_refs,
            "duration_seconds": segment_seconds,
            "frames": int(plan_item.get("frames") or 0),
            "subjects": subjects,
            "fallback_prompt": "\n\n".join(
                part for part in (
                    segment_style,
                    chunk,
                ) if part),
        })
    return style_header, briefs


_LEADING_CHARACTER_REFERENCE = re.compile(
    r"\A\s*(?:(?:人物|角色)外观参考|Character\s+appearance\s+reference)"
    r"\s*[:：][^\r\n]*(?:\r?\n+|\Z)",
    re.IGNORECASE)

_LEADING_CARD_TITLE = re.compile(
    r"\A\s*(?:(?:#{1,6}\s*)?(?:TITLE|标题|分镜标题|镜头标题)\s*[:：][^\r\n]*"
    r"|(?:分镜|镜头)\s*\d+\s*[:：][^\r\n]*)(?:\r?\n+|\Z)",
    re.IGNORECASE)


def _strip_legacy_character_reference_heading(value):
    """Remove obsolete prompt-leading metadata labels from model output."""
    cleaned = str(value or "").strip()
    changed = True
    while changed:
        changed = False
        for pattern in (_LEADING_CARD_TITLE, _LEADING_CHARACTER_REFERENCE):
            if pattern.match(cleaned):
                cleaned = pattern.sub("", cleaned, count=1).lstrip()
                changed = True
    return cleaned


def _inject_missing_character_refs(prompt, required_tags):
    """Embed segment-required character refs in visual prose, never a title line."""
    value = _strip_legacy_character_reference_heading(prompt)
    missing = []
    for tag in required_tags or []:
        match = re.search(r"(\d+)$", str(tag))
        ordinal = int(match.group(1)) if match else 0
        if tag not in value and (not ordinal or f"<Picture {ordinal}>" not in value):
            missing.append(str(tag))
    if not missing:
        return value, []
    tags = "、".join(missing)
    chinese = len(re.findall(r"[\u3400-\u9fff]", value)) >= 4
    sentence = (
        "本段实际出镜人物的可见外观与 %s 保持一致。" % tags
        if chinese else
        "The visible character in this segment matches %s in appearance. " % tags)
    field = re.search(
        r"(?mi)^(?:integrated_multimodal_description|subject_definitions|detailed_description)\s*:\s*",
        value)
    if field:
        value = value[:field.end()] + sentence + value[field.end():]
    else:
        # Put the reference after the first visible clause so it remains part
        # of the shot description instead of becoming an unrelated heading.
        first_line_end = value.find("\n")
        search_end = first_line_end if first_line_end >= 0 else min(len(value), 320)
        punctuation = list(re.finditer(r"[。.!！？；;]", value[:search_end]))
        if punctuation:
            insert_at = punctuation[0].end()
            value = (value[:insert_at] + " " + sentence + " "
                     + value[insert_at:].lstrip())
        elif value:
            value = "%s %s" % (value.rstrip(), sentence.strip())
        else:
            value = sentence.strip()
    return value.strip(), missing


_BATCH_WRITER_HEADING = re.compile(
    r"(?im)^\s*(?:<<<\s*H3_SEGMENT_(\d+)_BEGIN\s*>>>|"
    r"\[\s*SEGMENT\s+(\d+)\s*\]|={3,}\s*H3_SEGMENT\s+(\d+)\s*=+)\s*$")
_BATCH_WRITER_END = re.compile(
    r"(?im)^\s*<<<\s*H3_SEGMENT_\d+_END\s*>>>\s*$")


def _parse_batch_writer_response(raw, expected_count):
    """Recover every complete prompt from JSON or forgiving text envelopes."""
    payload = _loads_loose(raw)
    listed = payload.get("segments") if isinstance(payload, dict) else None
    recovered = {}
    if isinstance(listed, list):
        for offset, item in enumerate(listed, 1):
            if not isinstance(item, dict):
                continue
            try:
                index = int(item.get("index") or offset)
            except (TypeError, ValueError):
                continue
            prompt = str(item.get("prompt") or "").strip()
            if 1 <= index <= int(expected_count) and prompt:
                recovered[index] = prompt
        if recovered:
            return recovered

    source = re.sub(r"^\s*```(?:text|markdown)?|```\s*$", "",
                    str(raw or "").strip(), flags=re.MULTILINE).strip()
    matches = list(_BATCH_WRITER_HEADING.finditer(source))
    for offset, match in enumerate(matches):
        index = next((int(value) for value in match.groups() if value), 0)
        if not 1 <= index <= int(expected_count):
            continue
        end = matches[offset + 1].start() if offset + 1 < len(matches) else len(source)
        prompt = _BATCH_WRITER_END.sub("", source[match.end():end]).strip()
        if prompt:
            recovered[index] = prompt
    return recovered


LAYER_DIALOGUE_SYSTEM = """你只负责一段视频的台词层，不写画面、不写运镜、不写音效、不写风格。

用户会给你这一段已经写好的正文。请把其中真正说出口的话取出来，按下面的预算重写成可念完的台词。

只输出 JSON，不要解释、不要 markdown 代码块：
{{"dialogue":[{{"speaker":"说话人","tone":"calm","text":"这句台词"}}]}}

硬规则：
- 本段只有 {seconds:.2f} 秒。所有台词加起来，按最快语速也必须念得完。
- tone 只能是 excited / calm / casual，分别对应 3.5-6.0 / 2.5-4.0 / 1.5-3.0 字每秒。
- 本段全部 text 合计不得超过 {budget} 字。宁可删句、宁可缩短，也绝对不要超。
- text 里只放真正说出口的字：不要写说话人名字、不要写括号里的动作神态、不要写 <d> 标签。
- 正文里本来就没人说话时输出 {{"dialogue":[]}}，不要为了填满时长硬凑。
- 不要新增正文里没有的情节或人物。"""

LAYER_SOUND_SYSTEM = """你只负责一段视频的声音层，不写画面、不写台词、不写运镜。
用户会给你这一段已经写好的正文。请提炼出它的声音设计。

只输出 JSON，不要解释、不要 markdown 代码块：
{{"ambient":"环境音","bgm":"背景音乐","sfx":["音效1","音效2"]}}

硬规则：
- 三个字段都可以是空字符串或空数组。正文里没有就留空，不要编。
- 不要写台词，不要复述画面和人物动作。
- sfx 最多 4 条，每条不超过 12 字。"""

LAYER_STRUCTURE_SYSTEM = """你只负责把一段已经写好的视频正文拆成「主体」和「时间轴」两层，不写台词、不写音效。

只输出 JSON，不要解释、不要 markdown 代码块：
{"subjects":[{"name":"人物或主体名","appearance":"外观","wardrobe":"服装","ref_tag":"@图片1"}],
 "timeline":[{"beat":"0-3s","camera":"镜头运动","action":"这段时间发生什么"}]}

硬规则：
- 只做拆分，不新增、不删改剧情。所有内容都必须能在正文里找到出处。
- subjects 只写出镜的人物/主体。正文里带 @图片N 的，把那个标签原样填进 ref_tag；没有就留空字符串。
- appearance 和 wardrobe 各不超过 30 字；正文没写就留空字符串，不要编。
- timeline 按时间顺序，最多 6 条，覆盖整段时长。beat 用「0-3s」这种区间；正文没有明确时间就按顺序均分。
- camera 写景别与运镜（如「中景缓慢推近」），action 写画面里发生的事，各不超过 40 字。
- 正文里说出口的台词不要出现在 action 里。
- 两个数组都可以为空。"""


def _write_segment_layers(visual, seconds, llm_service, ollama_auto_unload,
                          seed, index):
    """Write the sound and dialogue layers for one segment as separate calls.

    Split out from the visual writer deliberately. Each answer is a handful of
    tokens, which is what stops a flash model rejecting an oversized request and
    a reasoning model spending its whole budget in the thinking channel. The
    dialogue call is also the only point where the shot's real seconds can be
    turned into a hard character ceiling *before* anything is written, instead
    of being audited after the fact.

    These calls rewrite from the finished prose rather than replacing part of
    the visual writer's job, so the Skill contract still governs a complete
    prompt body and a failed layer costs nothing: the caller keeps the original
    prose exactly as it is today. Nothing can silently lose its dialogue.
    """
    try:
        from . import dialogue_audit
    except ImportError:  # pragma: no cover - flat-import fallback
        import dialogue_audit
    body = str(visual or "").strip()
    if not body:
        return {}, []
    window = max(0.5, float(seconds or 0.0))
    layers, notes = {}, []
    requests = (
        ("dialogue", LAYER_DIALOGUE_SYSTEM.format(
            seconds=window,
            budget=dialogue_audit.budget_units(window)), 4409),
        ("sound", LAYER_SOUND_SYSTEM, 7717),
    )
    for name, system_prompt, salt in requests:
        try:
            raw = call_llm(
                llm_service, "本段正文：\n" + body, system_prompt,
                ollama_auto_unload, int(seed) + int(index) * 313 + salt,
                max_tokens=None)
        except Exception as error:  # noqa: BLE001 - a layer is never fatal
            if type(error).__name__ == "InterruptProcessingException":
                raise
            notes.append("第%d段%s层调用失败：%s" % (index, name, str(error)[:120]))
            continue
        payload = _loads_loose(raw)
        if not isinstance(payload, dict):
            notes.append("第%d段%s层没有返回可用 JSON" % (index, name))
            continue
        if name == "dialogue":
            lines = payload.get("dialogue")
            if isinstance(lines, list):
                layers["dialogue"] = [entry for entry in lines
                                      if isinstance(entry, (dict, str))]
        elif any(payload.get(key) for key in ("ambient", "bgm", "sfx")):
            layers["sound"] = {
                "ambient": payload.get("ambient") or "",
                "bgm": payload.get("bgm") or "",
                "sfx": payload.get("sfx") if isinstance(payload.get("sfx"), list) else [],
            }
    return layers, notes


def _write_segments_with_media_agent(
    text, chunks, count, maximum_seconds, total_seconds, overlap_frames, fps,
    media_manifest, skill_rules, llm_service, ollama_auto_unload, seed,
    skill_preset=SKILL_PRESET_NONE, skill_text="", skill_source="",
    checkpoint_writer=None, resume_context=None, layered_prompts=False,
):
    """Write N plain prompts with the package's existing Media Agent method."""
    from . import agent_nodes

    manifest = _agent_manifest_entries(media_manifest)
    saved_context = resume_context if isinstance(resume_context, dict) else {}
    saved_storyboard = saved_context.get("storyboard")
    saved_briefs = saved_context.get("briefs")
    context_has_writer_state = (
        isinstance(saved_storyboard, list)
        and len(saved_storyboard) == int(count)
        and all(isinstance(item, dict) for item in saved_storyboard)
        and isinstance(saved_briefs, list)
        and len(saved_briefs) == int(count)
        and all(isinstance(item, dict) and str(item.get("writer_input") or "").strip()
                for item in saved_briefs))
    if context_has_writer_state:
        storyboard = [dict(item) for item in saved_storyboard]
        storyboard_source = str(
            saved_context.get("storyboard_source") or "saved_agent_context")
        plan_notes = [
            "已载入 Agent 上下文：复用分镜规划和逐段写作输入，"
            "只继续未完成分段"
        ]
        style_header = str(saved_context.get("style_header") or "")
        briefs = [dict(item) for item in saved_briefs]
    else:
        storyboard, storyboard_source, plan_notes = _plan_director_storyboard(
            chunks, count, maximum_seconds, total_seconds, overlap_frames, fps,
            media_manifest, llm_service, ollama_auto_unload, seed)
        storyboard = _apply_subject_ledger(
            chunks, storyboard, media_manifest, text)
        style_header, briefs = _segment_writer_briefs(
            text, chunks, media_manifest, maximum_seconds, storyboard=storyboard)
    saved_fallback_indices = {
        int(index) for index in (saved_context.get("writer_fallback_segments") or [])
        if str(index).isdigit() and 1 <= int(index) <= int(count)
    }
    resumed_records = {}
    saved_segments = (saved_context.get("segments") or []) \
        if context_has_writer_state else []
    for saved_segment in saved_segments:
        if not isinstance(saved_segment, dict):
            continue
        try:
            saved_index = int(saved_segment.get("index") or 0)
        except (TypeError, ValueError):
            continue
        if (1 <= saved_index <= int(count)
                and saved_index not in saved_fallback_indices
                and str(saved_segment.get("prompt") or "").strip()):
            resumed_records[saved_index] = dict(saved_segment)
    # Reused prompts are fed through the normal ordered loop below so the
    # returned plan never contains duplicate or out-of-order cards.
    segments = []
    notes = list(plan_notes)
    fallback_segments = sorted(saved_fallback_indices)
    repaired_character_segments = []
    resolved_skill_plan = saved_context.get("skill_plan") if context_has_writer_state else None
    context_has_skill_state = (
        isinstance(resolved_skill_plan, list)
        and len(resolved_skill_plan) == int(count)
        and all(isinstance(route, dict) for route in resolved_skill_plan)
        and all("skill_rules" in item for item in briefs))
    skill_strategy = str(saved_context.get("skill_strategy") or "固定单技能") \
        if context_has_skill_state else "固定单技能"
    checkpoint_path = ""
    if checkpoint_writer is not None:
        try:
            written = checkpoint_writer({
                "storyboard": storyboard,
                "storyboard_source": storyboard_source,
                "style_header": style_header,
                "briefs": briefs,
                "skill_strategy": skill_strategy,
                "skill_plan": list(resolved_skill_plan or [])
                if isinstance(resolved_skill_plan, list) else [],
                "skill_source": str(saved_context.get("skill_source")
                                    or skill_source or ""),
                "writer_fallback_segments": list(fallback_segments),
                "segments": [dict(record) for record in resumed_records.values()],
                "model_last_used": str(llm_service or ""),
                "status": "planning",
                "last_error": "",
                "notes": list(notes[-12:]),
            })
            if written:
                checkpoint_path = str(written)
        except Exception as exc:  # noqa: BLE001 - checkpoint is best effort
            logger.warning("H3-Myang: Agent 基础上下文保存失败：%s", exc)
    if not context_has_skill_state:
        if str(skill_preset or "").strip() == SKILL_PRESET_AUTO:
            route_inputs = [{
                "index": item["index"],
                "duration_seconds": item.get("duration_seconds"),
                "transition": item.get("transition"),
                "brief": item.get("brief"),
                "segment_goal": storyboard[item["index"] - 1].get("segment_goal"),
                "selected_media_tags": item.get("selected_media_tags") or [],
                "subjects": item.get("subjects") or [],
            } for item in briefs]
            resolved_skill_plan, skill_strategy = agent_nodes.select_skill_plan_auto(
                llm_service, route_inputs, bool(ollama_auto_unload))
            notes.append(skill_strategy)
        else:
            fixed_name = str(skill_preset or SKILL_PRESET_NONE).strip() or SKILL_PRESET_NONE
            resolved_skill_plan = [{
                "index": item["index"],
                "primary": fixed_name,
                "overlays": [],
                "reviewers": [],
                "planners": [],
                "reason": "用户固定选择",
            } for item in briefs]

        for item, route in zip(briefs, resolved_skill_plan):
            if str(skill_preset or "").strip() == SKILL_PRESET_AUTO:
                item_rules, item_source, item_skills = agent_nodes.resolve_skill_bundle(
                    route.get("primary"), route.get("overlays"), route.get("reviewers"),
                    skill_text=skill_text)
            else:
                item_rules = skill_rules
                item_source = str(skill_source or route.get("primary") or "默认写法")
                item_skills = ([route.get("primary")]
                               if route.get("primary") not in (None, "", SKILL_PRESET_NONE)
                               else [])
            item["skill_rules"] = item_rules
            item["skill_source"] = item_source
            item["skills"] = [str(name) for name in item_skills if str(name or "").strip()]
            item["skill_plan"] = {
                "index": int(item["index"]),
                "primary": str(route.get("primary") or SKILL_PRESET_NONE),
                "overlays": list(route.get("overlays") or []),
                "reviewers": list(route.get("reviewers") or []),
                "reason": str(route.get("reason") or "")[:120],
                "source": item_source,
            }
    else:
        notes.append("已复用上次的写作技能方案，不重新调用技能选择模型")

    def _checkpoint_segments():
        merged = {}
        for record in resumed_records.values():
            try:
                index = int(record.get("index") or 0)
            except (TypeError, ValueError):
                continue
            if 1 <= index <= int(count):
                merged[index] = dict(record)
        for record in segments:
            try:
                index = int(record.get("index") or 0)
            except (TypeError, ValueError):
                continue
            if 1 <= index <= int(count):
                merged[index] = dict(record)
        return [merged[index] for index in sorted(merged)]

    def _checkpoint_note():
        if not checkpoint_path:
            return ""
        return "Agent上下文已保存：%d/%d 段 → %s" % (
            len(_checkpoint_segments()), int(count), checkpoint_path)

    def _segment_record(item, prompt):
        """Build the persisted shape for one generated segment."""
        return {
            "index": item["index"],
            "title": _sanitize_storyboard_title(
                storyboard[item["index"] - 1].get("title"), item.get("brief")),
            "transition": item["transition"],
            "duration_seconds": item.get("duration_seconds"),
            "frames": item.get("frames"),
            "subjects": item.get("subjects") or [],
            "skills": item.get("skills") or [],
            "skill_source": item.get("skill_source") or "",
            "brief": item["brief"],
            "prompt": str(prompt or "").strip(),
        }

    def _save_checkpoint(status="partial", error_text=""):
        nonlocal checkpoint_path
        if checkpoint_writer is None:
            return
        state = {
            "storyboard": storyboard,
            "storyboard_source": storyboard_source,
            "style_header": style_header,
            "briefs": briefs,
            "skill_strategy": skill_strategy,
            "skill_plan": list(resolved_skill_plan or [])
            if isinstance(resolved_skill_plan, list) else [],
            "skill_source": (skill_strategy
                             if str(skill_preset or "").strip() == SKILL_PRESET_AUTO
                             else str(skill_source or "")),
            "writer_fallback_segments": list(fallback_segments),
            "segments": _checkpoint_segments(),
            "model_last_used": str(llm_service or ""),
            "status": str(status or "partial"),
            "last_error": str(error_text or "")[:500],
            "notes": list(notes[-12:]),
        }
        try:
            written = checkpoint_writer(state)
            if written:
                checkpoint_path = str(written)
        except Exception as exc:  # noqa: BLE001 - a checkpoint must not break generation
            logger.warning("H3-Myang: Agent 上下文保存失败：%s", exc)

    def _writer_input(item):
        """Give a missing segment bounded read-only context from prior output."""
        current_index = int(item.get("index") or 0)
        previous = [record for record in _checkpoint_segments()
                    if int(record.get("index") or 0) < current_index]
        if not previous:
            return str(item.get("writer_input") or "")
        context_parts = []
        for record in previous[-2:]:
            prompt = str(record.get("prompt") or "").strip()
            if prompt:
                context_parts.append(
                    "第%d段已完成提示词（只读）：\n%s" %
                    (int(record.get("index") or 0), prompt[:2400]))
        if not context_parts:
            return str(item.get("writer_input") or "")
        return (str(item.get("writer_input") or "")
                + "\n\n【Agent 已完成上下文｜只用于保持连续性，不要重写已完成段】\n"
                + "\n\n".join(context_parts))

    _save_checkpoint()
    batch_parts = [
        "一次完成下面 %d 段提示词。段数、编号和顺序固定；每段独立执行对应输入，"
        "不要把不同段的剧情合并。" % int(count)]
    for item in briefs:
        batch_parts.extend([
            "", "=== INPUT SEGMENT %d/%d ===" % (item["index"], int(count)),
            _writer_input(item),
        ])
    batch_keys = {
        _entry_media_key(entry)
        for item in briefs for entry in item.get("manifest") or []
    }
    batch_manifest = [entry for entry in manifest
                      if _entry_media_key(entry) in batch_keys]
    batch_skill_rules = briefs[0].get("skill_rules", skill_rules) if briefs else skill_rules
    batch_user, batch_system = agent_nodes.build_media_agent_writer_request(
        "\n".join(batch_parts), batch_manifest, batch_skill_rules,
        maximum_seconds, expand=False)
    batch_system += (
        "\n\n【批量输出协议】\n"
        "必须输出 %d 个分段正文。不要输出 JSON、解释或 Markdown 代码块。"
        "每段只用以下边界包住，边界编号必须连续：\n"
        "<<<H3_SEGMENT_1_BEGIN>>>\n第1段最终提示词正文\n"
        "<<<H3_SEGMENT_1_END>>>\n"
        "后续段按相同格式编号到 %d。段内继续遵守写作技能。"
        "分镜标题只写入卡片元数据，段内正文禁止输出 TITLE、标题或‘分镜N：标题’行。" % (
            int(count), int(count)))
    generated_by_index = {
        int(index): str(record.get("prompt") or "").strip()
        for index, record in resumed_records.items()
        if str(record.get("prompt") or "").strip()
    }
    # Long jobs use a map-style writer: one compact request per segment.  A
    # single batch request makes the model hold every segment, the storyboard,
    # media manifest and Skill contract in one context, which is exactly where
    # reasoning models spend their budget or hit provider context limits.
    per_segment_media = [tuple(item.get("selected_media_tags") or []) for item in briefs]
    per_segment_durations = [round(float(item.get("duration_seconds") or 0), 3)
                             for item in briefs]
    per_segment_skills = [(
        tuple(item.get("skills") or []), hashlib.sha256(
            str(item.get("skill_rules") or "").encode("utf-8")).hexdigest()[:12]
    ) for item in briefs]
    use_batch_writer = int(count) <= 3 and (
        len(batch_user) + len(batch_system) <= 16000) and (
        not per_segment_media or len(set(per_segment_media)) == 1) and (
        len(set(per_segment_durations)) <= 1) and (
        len(set(per_segment_skills)) <= 1) and not generated_by_index
    if not use_batch_writer:
        notes.append(
            "长上下文采用逐段 Agent 聚合（每次只写一段，保持固定段数）")
    else:
        try:
            raw = call_llm(
                llm_service, batch_user, batch_system, ollama_auto_unload,
                int(seed) + 7919, max_tokens=None)
            for index, prompt in _parse_batch_writer_response(raw, count).items():
                cleaned = agent_nodes._sanitize_llm_output(prompt)
                if cleaned:
                    generated_by_index[index] = cleaned
            # Persist partial batch output before entering the ordered repair
            # loop.  If the first missing segment fails, later batch results
            # must still be available to the next run.
            if generated_by_index:
                segments[:] = [
                    _segment_record(item, generated_by_index[item["index"]])
                    for item in briefs if item["index"] in generated_by_index
                ]
                segments.sort(key=lambda segment: int(segment.get("index") or 0))
                _save_checkpoint()
            if len(generated_by_index) != int(count):
                notes.append("批量写作返回 %d/%d 段；仅补写缺失段" % (
                    len(generated_by_index), int(count)))
        except Exception as error:  # noqa: BLE001 - missing segments are repaired below
            if type(error).__name__ == "InterruptProcessingException":
                raise
            notes.append("批量写作调用失败：%s" % str(error)[:160])
            if _is_llm_service_unavailable(error):
                _save_checkpoint("partial", str(error))
                checkpoint_note = _checkpoint_note()
                raise RuntimeError(
                    "分段提示词生成已停止：LLM 服务当前没有可用线路。"
                    "路由层已按错误类型执行可中断重试；"
                    "不会再逐段调用，也不会生成整批兜底提示词。\n"
                    "请在 Myang_node → LLM 服务设置中查看线路状态，"
                    "等待冷却、修复额度或切换服务后重新运行。"
                    + ("\n" + checkpoint_note if checkpoint_note else "")
                ) from error

    remote_writer_confirmed = bool(generated_by_index)
    # Parallel pre-warm. By the point a segment is being written every brief is
    # already fully specified (storyboard, media whitelist, subject ledger),
    # and the batch writer above already writes up to three segments without
    # seeing each other's output, so the map writers do not depend on each
    # other either -- the read-back lookback in _writer_input is a soft
    # continuity aid. Once one segment has proven the provider actually
    # answers, the remaining segments fire concurrently, turning N sequential
    # 60-90s calls into two rounds. Anything that comes back empty or failed
    # simply falls through to the original sequential path, which keeps the
    # lookback context and every repair rule intact. The first segment stays
    # sequential on purpose: a silent model must still stop after its bounded
    # probe instead of fanning N doomed calls at the provider.
    prewarm: dict[int, str] = {}
    prewarm_errors: dict[int, str] = {}
    prewarm_fired = False

    def _fire_prewarm(items):
        nonlocal prewarm_fired
        prewarm_fired = True
        from concurrent.futures import ThreadPoolExecutor

        def _prewarm_one(item):
            user_prompt, system_prompt = agent_nodes.build_media_agent_writer_request(
                _writer_input(item), item.get("manifest") or [],
                item.get("skill_rules", skill_rules),
                item.get("duration_seconds") or maximum_seconds,
                expand=False)
            return call_llm(
                llm_service, user_prompt, system_prompt, ollama_auto_unload,
                int(seed) + int(item["index"]) * 131, max_tokens=None)

        worker_count = min(3, len(items))
        logger.info(
            "H3-Myang: 并行预写 %d 段（%d 线程）；空回复或失败的段回退逐段补写",
            len(items), worker_count)
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            futures = {int(item["index"]): pool.submit(_prewarm_one, item)
                       for item in items}
            for index, future in futures.items():
                try:
                    prewarm[index] = str(future.result() or "")
                except Exception as error:  # noqa: BLE001 - repaired sequentially below
                    if type(error).__name__ == "InterruptProcessingException":
                        raise
                    prewarm_errors[index] = str(error)[:160]
        if prewarm_errors:
            notes.append("并行预写 %d 段失败，已回退逐段补写：%s" % (
                len(prewarm_errors), "；".join(
                    "第%d段 %s" % (index, reason)
                    for index, reason in sorted(prewarm_errors.items()))))

    def _maybe_fire_prewarm(current_index):
        if prewarm_fired or use_batch_writer:
            return
        remaining = [later for later in briefs
                     if int(later["index"]) > int(current_index)
                     and int(later["index"]) not in generated_by_index]
        if len(remaining) >= 2:
            _fire_prewarm(remaining)

    for item in briefs:
        generated = generated_by_index.get(item["index"], "")
        if not generated:
            prewarm_raw = prewarm.pop(int(item["index"]), None)
            if prewarm_raw is not None:
                # Same seed formula as the sequential call, so a pre-warmed
                # segment that came back empty does not waste its retry budget.
                generated = agent_nodes._sanitize_llm_output(prewarm_raw)
                if not generated:
                    notes.append("第%d段并行预写返回空正文" % int(item["index"]))
            elif int(item["index"]) in prewarm_errors:
                notes.append("第%d段并行预写失败：%s" % (
                    int(item["index"]), prewarm_errors[int(item["index"])]))
        if not generated:
            user_prompt, system_prompt = agent_nodes.build_media_agent_writer_request(
                _writer_input(item), item.get("manifest") or [],
                item.get("skill_rules", skill_rules),
                item.get("duration_seconds") or maximum_seconds,
                expand=False)
            label = ("第%d段生成" if not use_batch_writer else "第%d段补写") % item["index"]
            try:
                raw = call_llm(
                    llm_service, user_prompt, system_prompt,
                    ollama_auto_unload,
                    int(seed) + item["index"] * 131,
                    max_tokens=None)
            except Exception as error:  # noqa: BLE001 - classified locally
                if type(error).__name__ == "InterruptProcessingException":
                    raise
                notes.append("%s 调用失败：%s" % (label, str(error)[:160]))
                if _is_llm_service_unavailable(error):
                    _save_checkpoint("partial", str(error))
                    checkpoint_note = _checkpoint_note()
                    raise RuntimeError(
                        "分段提示词生成已停止：补写时 LLM 服务失去全部可用线路；"
                        "已停止剩余分段，不会生成兜底提示词。"
                        + ("\n" + checkpoint_note if checkpoint_note else "")
                    ) from error
            else:
                generated = agent_nodes._sanitize_llm_output(raw)
                if not generated:
                    notes.append("%s 返回空正文" % label)
            # A reasoning model can legally finish with an empty visible
            # channel once. Give the first map item one fresh-seed probe, then
            # stop; this avoids the old retry storm while handling a transient
            # empty response before declaring the whole job unusable.
            if not generated and not remote_writer_confirmed:
                try:
                    retry_raw = call_llm(
                        llm_service, user_prompt, system_prompt,
                        ollama_auto_unload,
                        int(seed) + item["index"] * 131 + 100003,
                        max_tokens=None)
                except Exception as error:  # noqa: BLE001 - classified locally
                    if type(error).__name__ == "InterruptProcessingException":
                        raise
                    notes.append("%s 首次空回复后的单次探测失败：%s" % (
                        label, str(error)[:160]))
                    if _is_llm_service_unavailable(error):
                        _save_checkpoint("partial", str(error))
                        checkpoint_note = _checkpoint_note()
                        raise RuntimeError(
                            "分段提示词生成已停止：LLM 服务在空回复探测期间失去可用线路；"
                            "不会继续重试或生成兜底提示词。"
                            + ("\n" + checkpoint_note if checkpoint_note else "")
                        ) from error
                else:
                    generated = agent_nodes._sanitize_llm_output(retry_raw)
                    if generated:
                        notes.append("%s 单次空回复探测成功" % label)
                    else:
                        notes.append("%s 单次空回复探测仍为空" % label)
        if generated:
            remote_writer_confirmed = True
            _maybe_fire_prewarm(int(item["index"]))
        elif not remote_writer_confirmed:
            _save_checkpoint("partial", "首段没有返回正文")
            checkpoint_note = _checkpoint_note()
            raise ValueError(
                "智能切分失败：LLM 的批量写作和首段小请求探测都没有返回正文。"
                "已停止剩余分段，不会拿整段剧本去凑，也不会生成十段兜底提示词。\n"
                "如果本来就不需要 LLM 改写，请关掉『智能切片』；"
                "否则请检查当前 LLM 线路后重试。"
                + ("\n" + checkpoint_note if checkpoint_note else "")
            )
        if not generated:
            _save_checkpoint("partial", "第%d段没有返回正文" % item["index"])
            checkpoint_note = _checkpoint_note()
            raise ValueError(
                "智能切分失败：第%d段补写没有返回正文。"
                "已停止，不会用原始剧情冒充 LLM 分段提示词。"
                % item["index"]
                + ("\n" + checkpoint_note if checkpoint_note else "")
            )
        generated, restored_refs = _inject_missing_character_refs(
            generated, item.get("required_character_tags") or [])
        if restored_refs:
            repaired_character_segments.append(item["index"])
        record = _segment_record(item, generated)
        if layered_prompts:
            seg_layers, layer_call_notes = _write_segment_layers(
                generated, item.get("duration_seconds") or maximum_seconds,
                llm_service, ollama_auto_unload, seed, int(item["index"]))
            notes.extend(layer_call_notes)
            if seg_layers:
                # The prose stays the visual layer verbatim, so a segment whose
                # layer calls both failed is byte-identical to the unlayered
                # path and nothing can lose its dialogue.
                seg_layers["visual"] = generated
                record["layers"] = seg_layers
        segments[:] = [segment for segment in segments
                       if int(segment.get("index") or 0) != int(item["index"])]
        segments.append(record)
        segments.sort(key=lambda segment: int(segment.get("index") or 0))
        if int(item["index"]) in fallback_segments:
            fallback_segments.remove(int(item["index"]))
        _save_checkpoint()
    payload = {
        "style_header": style_header,
        "segments": segments,
        "split_source": (
            "media_agent_writer_with_local_fallback"
            if fallback_segments else "media_agent_writer"),
        "writer_mode": (
            "batch_plain_prompt_with_missing_segment_repair_v2"
            if use_batch_writer else
            "sequential_agent_map_with_missing_segment_repair_v1"),
        "storyboard_source": storyboard_source,
        "storyboard": storyboard,
        "skill_strategy": skill_strategy,
        "skill_plan": [item.get("skill_plan") for item in briefs],
        "skill_source": skill_strategy if str(skill_preset or "").strip() == SKILL_PRESET_AUTO
                        else str(skill_source or ""),
        "writer_fallback_segments": fallback_segments,
        "media_reference_compliance": {
            "passed": True,
            "required_character_tags": sorted({
                tag for item in briefs
                for tag in item.get("required_character_tags") or []
            }),
            "required_character_tags_by_segment": {
                str(item["index"]): list(item.get("required_character_tags") or [])
                for item in briefs
            },
            "selected_media_tags_by_segment": {
                str(item["index"]): list(item.get("selected_media_tags") or [])
                for item in briefs
            },
            "repaired_segments": repaired_character_segments,
        },
    }
    _save_checkpoint("complete")
    return payload, notes


def _usable_split_payload(payload, expected_count):
    segments = payload.get("segments") if isinstance(payload, dict) else None
    if not isinstance(segments, list) or len(segments) != int(expected_count):
        return False
    return all(isinstance(segment, dict) and (
        str(segment.get("prompt") or "").strip()
        or str(segment.get("brief") or "").strip()) for segment in segments)


def _cacheable_split_payload(payload, expected_count):
    return (_usable_split_payload(payload, expected_count)
            and not payload.get("writer_fallback_segments")
            and str(payload.get("split_source") or "") not in {
                "local_fallback", "media_agent_writer_with_local_fallback"})


def _is_timeout_error(error):
    message = str(error or "").lower()
    return isinstance(error, TimeoutError) or "timed out" in message or "timeout" in message


def _is_rate_limit_error(error):
    message = str(error or "").lower()
    return "429" in message or "rate limit" in message or "tpm" in message


def _is_llm_service_unavailable(error):
    try:
        from . import llm_service as service_client
        return service_client.is_service_unavailable_error(error)
    except Exception:
        message = str(error or "").lower()
        return any(token in message for token in (
            "所有 api 线路都在冷却", "整次调用已达到", "timed out",
            "connection failed", "api error 401", "api error 403",
            "api error 429", "api error 500", "api error 502",
            "api error 503", "api error 504",
        ))


MANIFEST_ROLE = {
    "image": "静态图像（角色外观/服装/场景/构图参考）",
    "video": "动态视频（动作/运镜参考）",
    "audio": "音频（配乐/音效/台词对齐）",
}
MANIFEST_TAG = {"image": "图片", "video": "视频", "audio": "音频"}
MANIFEST_LABEL = {"image": "Picture", "video": "Video", "audio": "Audio"}


def _render_media_manifest(manifest) -> str:
    """Render the agent's media whitelist in the tag style segment prompts use.

    Both ``@图片1`` and ``<Picture 1>`` resolve for the generator, so the model
    is shown the pair it may paste.  The ordinal is the agent's per-type count,
    which is the one ``H3Condition`` also uses.
    """
    lines = []
    has_subject = False
    for entry in manifest:
        kind = str(entry.get("type") or "")
        if kind not in MANIFEST_TAG:
            continue
        ordinal = int(entry.get("ordinal") or 0)
        details = []
        if entry.get("resolution"):
            details.append(str(entry["resolution"]))
        if entry.get("duration") is not None:
            details.append("%ss" % entry["duration"])
        head = "- <%s %d> 或 @%s%d：%s%s" % (
            MANIFEST_LABEL[kind], ordinal, MANIFEST_TAG[kind], ordinal,
            MANIFEST_ROLE[kind],
            "（%s）" % "，".join(details) if details else "")
        subject = str(entry.get("subject_name") or "").strip()
        if subject:
            has_subject = True
            head += "，主体名：%s" % subject
        elif entry.get("filename"):
            head += "，文件：%s" % entry["filename"]
        lines.append(head)
        if entry.get("description"):
            lines.append("  画面内容：%s" % entry["description"])
        if entry.get("transcript"):
            lines.append("  语音内容：%s" % entry["transcript"])
            lines.append("  （这是该音频的真实语音，台词必须与它一致，不要另编。）")
        if entry.get("subject"):
            lines.append("  归属：%s" % entry["subject"])
    if has_subject:
        lines.append(
            "（主体名是用户给素材起的名字，可以在正文里直接用它指代人物或物体；"
            "同一个主体在各段必须用同一个名字。文件名只用于辨认素材，不要写进提示词。）")
    return "\n".join(lines)


def _compact_media_manifest(media_manifest: str) -> str:
    """Keep media tags and short descriptions for the storyboard planner."""
    source = str(media_manifest or "").strip()
    if not source:
        return ""
    lines = []
    for raw in source.splitlines():
        line = raw.strip()
        if not line:
            continue
        if re.match(r"^-\s*<(?:Picture|Video|Audio)\s+\d+>", line, re.I):
            lines.append(line[:420])
        elif line.startswith(("画面内容：", "语音内容：", "归属：")):
            lines.append("  " + line[:280])
    return "\n".join(lines) if lines else source[:1800]


def _format_media_manifest(media, vlm_service: str = "off",
                           ollama_auto_unload: bool = False) -> str:
    """Extract a clean human-and-LLM-readable manifest from a media bundle.

    Delegates to the Media Agent's own whitelist builder so the splitter sees
    what the agent sees: real resolutions and durations, the user's subject
    names, and -- when a VLM is connected -- what each image or clip actually
    shows.  A bare filename cannot tell the model which segment needs which
    asset; a description can.
    """
    if media is None:
        return ""
    if isinstance(media, str):
        return media.strip()
    if getattr(media, "items", None):
        try:
            from . import agent_nodes
        except ImportError:  # pragma: no cover - standalone import fallback
            import agent_nodes
        manifest, errors = agent_nodes.media_whitelist(
            media, vlm_service=vlm_service, ollama_auto_unload=ollama_auto_unload)
        for error in errors:
            logger.warning("H3-Myang: 素材描述失败 %s", error)
        return _render_media_manifest(manifest)
    manifest_lines = []
    if isinstance(media, dict):
        for k, v in media.items():
            manifest_lines.append(f"- {k}: {type(v).__name__}")
    elif isinstance(media, (list, tuple)):
        for i, itm in enumerate(media, 1):
            manifest_lines.append(f"- 素材{i}: {type(itm).__name__}")
    return "\n".join(manifest_lines)


# The authored layers of one shot. A segment carries these instead of a single
# prompt blob so that each LLM answer stays short, every element can be reviewed
# on its own, and the dialogue can be billed against the shot's real seconds
# before anything is rendered. ``subjects`` lives on the plan (one cast for the
# whole film); ``subjects_override`` is how a single segment expresses a costume
# change or a transformation without every other segment restating the wardrobe.
LAYER_KEYS = ("subjects", "subjects_override", "visual", "timeline", "sound",
              "dialogue")


def _layering_enabled(value):
    """Read a BOOLEAN widget without treating every non-empty string as true.

    Saved workflows can queue optional widget values against the previous
    positional layout while the browser still holds a cached node definition,
    and ``bool("False")`` is True in Python.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value == 1
    if isinstance(value, str):
        return value.strip().casefold() in {"true", "1", "yes", "on", "开启", "启用"}
    return False


def _clean_layer_text(value):
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _strip_dialogue_tags(text):
    """Drop any <d> the writer already emitted so wrapping cannot nest."""
    return re.sub(r"</?d>", "", str(text or "")).strip()


def _strip_dialogue_prose(text):
    """Remove inline <d>…</d> from a prose body, tidying what it leaves behind.

    Used when a budgeted dialogue layer replaces the writer's inline lines. Any
    line that held nothing but dialogue disappears entirely rather than being
    left as a dangling speaker name or a lone quotation mark.
    """
    stripped = re.sub(r"<d>.*?</d>", "", str(text or ""), flags=re.DOTALL)
    lines = []
    for line in stripped.splitlines():
        cleaned = re.sub(r"\s+", " ", line).strip()
        if cleaned and re.search(r"[\w一-鿿]", cleaned):
            lines.append(cleaned)
    return "\n".join(lines)


def _subject_lines(subjects):
    """One line per subject: who they are, how they look, and their @ tag."""
    lines = []
    for entry in subjects or []:
        if isinstance(entry, str):
            text = _clean_layer_text(entry)
            if text:
                lines.append(text)
            continue
        if not isinstance(entry, dict):
            continue
        name = _clean_layer_text(entry.get("name"))
        body = "，".join(part for part in (
            _clean_layer_text(entry.get("appearance")),
            _clean_layer_text(entry.get("wardrobe"))) if part)
        head = "：".join(part for part in (name, body) if part)
        tag = _clean_layer_text(entry.get("ref_tag"))
        if tag:
            head = ("%s %s" % (head, tag)).strip() if head else tag
        if head:
            lines.append(head)
    return lines


def _timeline_lines(timeline):
    """One line per beat, camera before action so the framing reads first.

    Accepts the ``time_range``/``camera_movement``/``content`` spelling too,
    because that is the vocabulary the storyboard summary agent already emits
    and imported cards carry it.
    """
    lines = []
    for entry in timeline or []:
        if isinstance(entry, str):
            text = _clean_layer_text(entry)
            if text:
                lines.append(text)
            continue
        if not isinstance(entry, dict):
            continue
        beat = _clean_layer_text(entry.get("beat") or entry.get("time_range"))
        body = "，".join(part for part in (
            _clean_layer_text(entry.get("camera") or entry.get("camera_movement")),
            _clean_layer_text(entry.get("action") or entry.get("content"))) if part)
        if not body:
            continue
        lines.append(("%s %s" % (beat, body)).strip() if beat else body)
    return lines


_EMPTY_SOUND = {"", "无", "none", "null", "n/a", "无台词", "无音效"}


def _sound_lines(sound):
    """Ambient / BGM / SFX, one labelled line each, placeholders dropped."""
    if isinstance(sound, str):
        text = _clean_layer_text(sound)
        return [text] if text.lower() not in _EMPTY_SOUND else []
    if not isinstance(sound, dict):
        return []
    lines = []
    for key, label in (("ambient", "环境音"), ("bgm", "背景音乐")):
        text = _clean_layer_text(sound.get(key))
        if text and text.lower() not in _EMPTY_SOUND:
            lines.append("%s：%s" % (label, text))
    effects = sound.get("sfx")
    if isinstance(effects, str):
        effects = [effects]
    named = [_clean_layer_text(item) for item in (effects or [])
             if _clean_layer_text(item).lower() not in _EMPTY_SOUND]
    if named:
        lines.append("音效：" + "、".join(named))
    return lines


def _dialogue_lines(dialogue):
    """Each spoken line wrapped in <d>…</d>, speaker and tone left outside it.

    The tag must hold only the words actually said: ``dialogue_audit`` bills
    whatever sits inside it against the shot's seconds, so a speaker name or a
    tone marker inside the tag would be charged as if it were spoken aloud.
    """
    lines = []
    for entry in dialogue or []:
        if isinstance(entry, str):
            text = _strip_dialogue_tags(_clean_layer_text(entry))
            if text:
                lines.append("<d>%s</d>" % text)
            continue
        if not isinstance(entry, dict):
            continue
        text = _strip_dialogue_tags(_clean_layer_text(entry.get("text")))
        if not text:
            continue
        speaker = _clean_layer_text(entry.get("speaker"))
        tone = _clean_layer_text(entry.get("tone"))
        prefix = speaker + ("（%s）" % tone if tone else "")
        lines.append("%s<d>%s</d>" % (prefix, text))
    return lines


def segment_layers(segment):
    """The layer dict for a segment, or None when it predates layering."""
    layers = segment.get("layers") if isinstance(segment, dict) else None
    if not isinstance(layers, dict):
        return None
    return layers if any(layers.get(key) for key in LAYER_KEYS) else None


def compose_segment_prompt(global_layers, layers):
    """Assemble one Easy Prompt from authored layers, deterministically.

    The model never writes the final string; it only fills in layers. Order is
    fixed because the tokenizer sees the concatenation: style and scene, then who
    is in frame, then what happens, then what is heard, then what is said.

    A ``visual`` layer holds the writer's finished prose, which is already a
    complete prompt body carrying its own style and cast. When it is present it
    replaces the structured head rather than being appended to it, and its
    inline ``<d>`` lines are dropped -- the dialogue layer is the budgeted
    rewrite of exactly those lines, so keeping both would say everything twice.
    """
    shared = global_layers if isinstance(global_layers, dict) else {}
    layers = layers if isinstance(layers, dict) else {}
    spoken = _dialogue_lines(layers.get("dialogue"))
    visual = str(layers.get("visual") or "").strip()
    if visual:
        blocks = [_strip_dialogue_prose(visual) if spoken else visual]
    else:
        blocks = [_clean_layer_text(shared.get("style")),
                  _clean_layer_text(shared.get("scene"))]
        blocks.extend(_subject_lines(layers.get("subjects_override")
                                     or shared.get("subjects")))
        blocks.extend(_timeline_lines(layers.get("timeline")))
    blocks.extend(_sound_lines(layers.get("sound")))
    blocks.extend(spoken)
    return "\n".join(block for block in blocks if block)


def _spoken_seconds(entry, rates, default_tone):
    """Fastest time a line can be delivered, and the tone it was billed at."""
    text = _strip_dialogue_tags(_clean_layer_text(
        entry.get("text") if isinstance(entry, dict) else entry))
    if not text:
        return 0.0, default_tone
    tone = _clean_layer_text(entry.get("tone")) if isinstance(entry, dict) else ""
    tone = tone if tone in rates else default_tone
    try:
        from . import dialogue_audit
    except ImportError:  # pragma: no cover - flat-import fallback
        import dialogue_audit
    return dialogue_audit.speech_units(text) / rates[tone][1], tone


def enforce_dialogue_budget(segments, window_for):
    """Keep every spoken line inside the shot it sits in, spilling the excess.

    ``dialogue_audit`` already knows whether a line can physically be said in a
    given window. What was missing is applying it to the *Director's* real
    segment seconds instead of the writer agent's own shot plan, which is how a
    whole conversation ended up inside one 8 second shot.

    Deliberately deterministic: no LLM call. Compression already gets one
    attempt upstream in the Media Agent's dialogue sub-agent; this is the net
    that runs after the real segmentation is known, and a net that can time out
    or answer with silence is not a net. Overflow moves to the next segment,
    because a line arriving one shot late is better than a shot that cannot
    finish its own sentence.

    A line longer than an entire window is left where it is -- the next window
    is the same size, so spilling it would only defer the same problem and
    eventually drop it.

    Mutates each segment's dialogue layer in place. Returns operator notes.
    """
    try:
        from . import dialogue_audit
    except ImportError:  # pragma: no cover - flat-import fallback
        import dialogue_audit
    rates = dialogue_audit.SPEECH_RATES
    default_tone = dialogue_audit.DEFAULT_TONE
    notes = []
    carried = []
    for segment in segments:
        layers = segment.get("layers")
        index = int(segment.get("index") or 0)
        if not isinstance(layers, dict):
            if carried:
                notes.append("第%d段没有分层，%d 句顺延台词已丢弃"
                             % (index, len(carried)))
                carried = []
            continue
        window = max(0.1, float(window_for(segment)))
        pending = carried + list(layers.get("dialogue") or [])
        carried, kept, used = [], [], 0.0
        for entry in pending:
            need, _tone = _spoken_seconds(entry, rates, default_tone)
            if need <= 0.0:
                continue
            if used + need <= window + 0.05:
                kept.append(entry)
                used += need
            elif not kept and need > window:
                kept.append(entry)
                used += need
                notes.append(
                    "第%d段有一句台词单独就需要 %.2f 秒，超过本段 %.2f 秒；"
                    "顺延也无法解决，请手动压缩" % (index, need, window))
            else:
                carried.append(entry)
        if carried:
            notes.append("第%d段台词超出 %.2f 秒可用时长，%d 句顺延到下一段"
                         % (index, window, len(carried)))
        layers["dialogue"] = kept
    if carried:
        notes.append("最后一段仍有 %d 句台词放不下，已截断" % len(carried))
    return notes


SPLIT_PROMPT_VERSION = 16


SPLIT_SYSTEM = """你是一个专业的 AI 视频分镜切片与 Easy Prompt 生成器（Shot / Segment Slicer & Prompt Author）。
用户会给你一段完整的视频剧本或提示词（通常由上游 Agent 生成，包含分镜设定、人物动作、镜头运镜、@图片1/@视频1等素材引用、以及<d>台词</d>或#台词）。{media_section}{skill_section}

你的任务是：将这段完整提示词/剧本，严格按时间线精准切分成恰好 {count} 个连续的视频分段，每段对应约 {seconds:.2f} 秒的生成片段。

【Easy Prompt 规范要求】：
1. 忠实切片，不要重新编写或杜撰新的剧情故事，只做时间轴上的逻辑切分与镜头分配。
2. 保持素材与台词完整（严格遵循 Easy Prompt 语法）：
   - 画面主体与外观：如使用了角色/通用参考（如 @图片1），各分段提示词中必须自然保留 `@图片1` 标签；
   - 动作与运镜参考：如使用了动作视频参考（如 @视频1），在对应的动作分段中保留 `@视频1` 标签；
   - 人物台词：台词必须严格用 `<d>台词文本</d>` 标签包裹（例如 `<d>“我们出发吧！”</d>`）；
   - 音频配乐：如有对齐音频，可标注 `@音频1`；
3. 段间衔接由你根据剧本自己判断，不要一刀切地全部写成平滑延续：
   - 同一场景内动作连续时用「承接」：上一段结尾的镜头位置、人物姿态和环境状态延续到本段开头；
   - 剧本在这里换了场景、换了时间、换了视角或换了主体时用「切镜」：本段直接进新镜头，
     并把新镜头的机位、景别、环境和人物状态完整交代一遍，不要依赖上一段的描述；
   - 用 transition 字段标出本段相对上一段是「承接」还是「切镜」，第 1 段固定写「开场」；
   - 注意：无论哪种，系统都会用上一段结尾做约 {overlap_seconds:.2f} 秒的接缝锚点（成片里会裁掉），
     所以「切镜」要靠本段正文的画面内容切过去，不要写黑场、闪白或转场特效指令。
4. 每段 prompt 是一个完整可独立渲染的 MiniMax H3 Easy Prompt 提示词（包含视觉风格、人物外观、本段动作、镜头运动、素材标签、音效与台词）。
5. 若上面给出了【写作技能】，技能里的输出结构、分镜格式、素材标签写法和禁止事项优先级高于本节的默认写法；
   但分段数量必须是 {count} 段、每段时长以上面给的秒数为准，技能文档里的示例镜头数和示例秒数一律不作数。

只输出 JSON，不要解释、不要 markdown 代码块、不要在 JSON 之外写任何字。必须输出恰好 {count} 个 segments。格式如下：
{{
  "style_header": "全局通用的画面风格、主体外观、镜头基调或场景设定（提炼自原文本，50~150字）",
  "segments": [
    {{
      "index": 1,
      "transition": "开场",
      "brief": "第1段动作剧情简要（用于界面预览，30~80字）",
      "prompt": "第1段完整的生成提示词（包含全局风格+本段动作镜头+素材标签+台词等，可直接输入视频模型）"
    }}
  ]
}}"""

SPLIT_SYSTEM_LEAN = """你是视频分镜切片器。把用户给的剧本按时间顺序切成恰好 {count} 段，每段约 {seconds:.2f} 秒。
只输出 JSON，不要解释、不要代码块、不要思考过程。每段 prompt 要能独立渲染，保留原文里的 @图片N/@视频N 标签和 <d>台词</d>。
transition 写「承接」或「切镜」，第 1 段写「开场」。
{{"style_header":"全局风格","segments":[{{"index":1,"transition":"开场","brief":"简要","prompt":"完整提示词"}}]}}"""

def _call_split_ladder(llm_service, variants, ollama_auto_unload, seed,
                       expected_count):
    """Run the split call through a fallback ladder instead of accepting silence.

    Two failure modes make a single call unreliable, and neither is a bug in the
    prompt -- the Media Agent hit both first and solves them the same way:

    * A reasoning model can spend its whole budget in the thinking channel and
      return success with empty content. Replaying the same seed reproduces the
      same silence, so each retry varies it.
    * A small "flash/lite" model rejects an oversized request with HTTP 400.
      That needs a *shorter* request, so every later variant drops another block
      (the Skill first, then the material rules) and lowers the ceiling.

    Returns (payload, notes). An empty payload means every rung came back
    without segments, which the caller must report rather than paper over.
    """
    notes = []
    # Let each provider choose its own output ceiling. Shortening the input
    # prompt is the safe fallback; imposing 3000/1500-token caps caused empty
    # answers on reasoning models.
    ladder = (None, None, None)
    for level, (user_prompt, system_prompt) in enumerate(variants):
        max_tokens = ladder[min(level, len(ladder) - 1)]
        for attempt in range(2):
            label = "L%d#%d" % (level + 1, attempt + 1)
            try:
                raw = call_llm(
                    llm_service, user_prompt, system_prompt, ollama_auto_unload,
                    int(seed) + level * 17 + attempt * 101, max_tokens=max_tokens)
            except Exception as error:  # noqa: BLE001 - classified right here
                if type(error).__name__ == "InterruptProcessingException":
                    raise
                notes.append("拆分 %s 调用失败：%s" % (label, str(error)[:140]))
                if _is_timeout_error(error):
                    notes.append("自适应等待后仍读取超时，停止远程重试并切换本地时间线切分")
                    return {}, notes
                continue
            payload = _loads_loose(raw)
            if _usable_split_payload(payload, expected_count):
                if notes:
                    notes.append("拆分在 %s 成功" % label)
                    logger.warning("H3-Myang: 拆分重试记录 | %s", " | ".join(notes))
                return payload, notes
            returned = len(payload.get("segments") or []) if isinstance(payload, dict) else 0
            notes.append("拆分 %s 没拿到恰好 %d 段（返回 %d 段、空回复或非 JSON）" % (
                label, int(expected_count), returned))
    return {}, notes


REFINE_SYSTEM = """你是 MiniMax H3 的提示词作者。根据给定的全局风格和本段剧情，写出这一段视频的完整提示词。
只输出提示词正文，不要解释、不要标题、不要 markdown。
必须覆盖：画面内容、人物动作与神情、镜头运动、光线质感、音效与台词。
不要写分镜编号，不要写"本段"之类的元叙述。严格只描述这一段，不要延伸到后面的剧情。"""


# --------------------------------------------------------------------------
# nodes
# --------------------------------------------------------------------------

DETAIL_BOOST_NONE = "标准自然（默认）"
DETAIL_BOOST_SMALL_OBJECTS = "远景小物体与五官强化（推荐·兼容加速）"
DETAIL_BOOST_CINEMATIC = "电影级胶片光影与微观材质"
DETAIL_BOOST_ANIME = "二次元/动漫超清线条与质感"
DETAIL_BOOSTS = [
    DETAIL_BOOST_NONE,
    DETAIL_BOOST_SMALL_OBJECTS,
    DETAIL_BOOST_CINEMATIC,
    DETAIL_BOOST_ANIME,
]

BOOST_PROMPTS = {
    DETAIL_BOOST_SMALL_OBJECTS: "清晰锐利的远景面部五官轮廓，细腻的发丝光影，微观皮肤纹理与小物体反光细节，高保真微距对比度，极致清晰度，ultra-detailed micro features, crisp distant facial contours and delicate textures",
    DETAIL_BOOST_CINEMATIC: "电影级通透光影，丁达尔光线，细腻胶片颗粒质感，浅景深自然虚化，丰富暗部细节，cinematic lighting, film grain, anamorphic bokeh, rich shadow details",
    DETAIL_BOOST_ANIME: "清爽利落的二次元线条，通透赛璐璐上色，鲜明高光反光，动感二次元构图，high quality anime visual, clean lineart, vibrant celluloid shading, dynamic anime cinematography",
}


class _H3ScriptSplitterBase:
    CATEGORY = "沐阳 H3"
    FUNCTION = "split"
    RETURN_TYPES = ("STRING", "INT", "FLOAT", "INT", "STRING", "INT")
    RETURN_NAMES = ("plan_json", "segment_count", "segment_seconds", "frames_per_segment",
                    "plan_preview", "ref_frames_needed")
    DESCRIPTION = "把长剧本/提示词智能切分成 N 段分镜，全流程只调一次 LLM。提示词内嵌于 plan_json。"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "script": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "完整提示词/剧本：可接入 Agent 节点的 easy_prompt，"
                               "或手动粘贴剧本。切片节点会用 LLM 将其按时间轴切分成各段提示词。"
                               "留空则只算时间与帧数，不调用 LLM。"}),
                "total_seconds": ("FLOAT", {"default": 60.0, "min": 1.0, "max": 3600.0, "step": 1.0}),
                "length_source": (LENGTH_SOURCES, {
                    "default": LENGTH_MANUAL,
                    "tooltip": "匹配参考视频时长：接上 ref_video 后按它的帧数算总时长，"
                               "上面填的数就不用管了"}),
                "segment_seconds": ("FLOAT", {
                    "default": 10.0, "min": MIN_SECONDS, "max": MAX_SECONDS, "step": 0.5,
                    "tooltip": "智能切分的单段时长上限，不是固定时长。AI 会按动作、台词与镜头节奏"
                               "为每段选择不同的实际时长，且不会超过该值。"}),
                "overlap_frames": ("INT", {"default": 22, "min": 0, "max": 240,
                                           "tooltip": "必须与 MiniMaxH3MotionContext 的 context_length 一致"}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 1.0}),
                "llm_service": (llm_service_options(),),
                "max_segments": ("INT", {"default": MAX_SLOTS, "min": 1, "max": MAX_SLOTS}),
                "ollama_auto_unload": ("BOOLEAN", {"default": True}),
                "use_cache": ("BOOLEAN", {"default": True,
                                          "tooltip": "剧本和段数没变就复用上次结果；中断时保存 Agent 上下文，"
                                                     "下次可切换模型续接缺失分段"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                                 "control_after_generate": True}),
                "llm_enabled": ("BOOLEAN", {
                    "default": True,
                    "label_on": "调用LLM切片",
                    "label_off": "不调用LLM(全片共用/直通)",
                    "tooltip": "手动选择是否调用 LLM 进行分段切片。"
                               "开启：优先用 LLM 切片；连续超时、格式错误或段数不对时，"
                               "自动使用本地时间线算法生成各段不同的提示词；"
                               "关闭：不调用 LLM，直接将完整提示词作为各段通用提示词（0 token 消耗）。"}),
                # NOTE: new widgets must be appended at the END. ComfyUI restores
                # saved widget values positionally, so inserting one in the middle
                # shifts every later value in already-saved workflows.
                "skill_preset": (skill_preset_options(), {
                    "default": SKILL_PRESET_AUTO,
                    "tooltip": "写作技能：决定每段提示词的输出结构、分镜格式和素材标签写法。"
                               "auto 会先用一次很短的调用按剧本选技能；none 用默认写法"}),
                "skill_text": ("STRING", {
                    "multiline": True, "dynamicPrompts": False, "default": "",
                    "tooltip": "自定义写作规则，排在所选技能之前，优先级最高"}),
                "vlm_service": (vlm_service_options(), {
                    "default": "off",
                    "tooltip": "开启后先让 VLM 看一遍每张图片/视频，把画面内容写进素材清单，"
                               "LLM 才能按内容判断每段该引用哪个素材；off 时只给文件名和主体名"}),
            },
            "optional": {
                "media": ("MINIMAX_H3_MEDIA", {
                    "tooltip": "连接 Media Agent 或素材包。切片器会自动感知可用素材清单（图片、视频、音频），"
                               "在各分段提示词中精准分配 @图片N/@视频N 等素材，并在「匹配参考视频时长」时直接提取视频素材总时长。"}),
                # Append-only: layered authoring. Off keeps the single-blob
                # prompt every saved workflow was written against.
                "分层提示词": ("BOOLEAN", {
                    "default": False,
                    "label_on": "分层生成（主体/时间轴/声音/台词）",
                    "label_off": "整段生成（传统）",
                    "tooltip": "开启后每段的声音与台词各自单独生成，篇幅短、更少报错，"
                               "并且台词会按本段真实秒数换算成硬性字数上限；"
                               "超出的台词顺延到下一段而不是硬塞进本段。"
                               "画面正文仍由原来的写作流程产出，分层失败也不会丢内容"}),
            },
        }

    def split(self, script, total_seconds, length_source, segment_seconds, overlap_frames, fps,
              llm_service, max_segments, ollama_auto_unload, use_cache, seed, llm_enabled=True,
              media=None, **kwargs):
        script = kwargs.get("prompt", script)
        media = kwargs.get("media", media)
        ref_video = kwargs.get("ref_video", None)
        if str(length_source) == LENGTH_MATCH_REF:
            video_frames = None
            if media is not None:
                items = getattr(media, "items", None) or ()
                for itm in items:
                    if getattr(itm, "media_type", "") == "video":
                        val = getattr(itm, "value", None)
                        if hasattr(val, "shape") and len(val.shape) > 0:
                            video_frames = int(val.shape[0])
                            break
            if video_frames is None and ref_video is not None and hasattr(ref_video, "shape") and len(ref_video.shape) > 0:
                video_frames = int(ref_video.shape[0])
            if video_frames is not None and video_frames > 0:
                total_seconds = video_frames / max(float(fps), 1.0)
                logger.info("H3-Myang: 按 media 视频素材取总时长 %d 帧 / %.0ffps = %.2fs",
                            video_frames, fps, total_seconds)
            else:
                raise ValueError("length_source 选了「匹配参考视频时长」，但未接 media 素材包（或 media 中未包含视频素材）")
        text = str(script or "").strip()
        plan = plan_segments(
            total_seconds, segment_seconds, overlap_frames, fps, max_segments,
            cover_with_maximum=bool(llm_enabled) and bool(text))
        count = plan["segment_count"]
        # Only pay for vision when the LLM is going to read the manifest.
        media_manifest = _format_media_manifest(
            media,
            vlm_service=str(kwargs.get("vlm_service", "off"))
            if bool(llm_enabled) and text else "off",
            ollama_auto_unload=bool(ollama_auto_unload))

        # detail_boost removed

        # Handle manual LLM toggle: if llm_enabled is False
        if not bool(llm_enabled):
            header = ""
            seg_prompts = []
            for i in range(1, count + 1):
                seg_prompts.append({"index": i, "brief": text[:100] if text else "", "prompt": text})

            plan.update({
                "style_header": header,
                "full_prompt": text,
                "segments": seg_prompts,
            })
            if media_manifest:
                plan["media_manifest"] = media_manifest
            preview = [
                f"共 {count} 段 × {plan['segment_seconds_snapped']:.3f}s"
                f"（每段 {plan['frames_per_segment']} 帧，段间重叠 {overlap_frames} 帧）",
                f"成片约 {plan['total_seconds_actual']:.2f}s（目标 {total_seconds:.1f}s）",
                f"参考视频至少要 {plan['ref_frames_needed']} 帧"
                f"（约 {plan['ref_frames_needed'] / max(fps, 1.0):.1f}s @ {fps:.0f}fps）",
                "",
                "【LLM 切片已手动关闭】未调用 LLM（0 token 消耗）。",
            ]
            if media_manifest:
                preview.append("【已绑定可用素材清单】")
                for mline in media_manifest.split("\n"):
                    if mline.strip():
                        preview.append(f"  {mline.strip()}")
            if text:
                preview.append(f"每段直接采用通用提示词：{text[:100]}…")
            else:
                preview.append("提示词为空，仅计算了分段与时间线。")

            logger.info("H3-Myang: LLM 切片已关闭，跳过 LLM，只做分段（%d 段）", count)
            return (json.dumps(plan, ensure_ascii=False), count,
                    plan["segment_seconds_snapped"], plan["frames_per_segment"],
                    "\n".join(preview), plan["ref_frames_needed"])

        if not text:
            # Script is empty while LLM enabled: skip LLM and emit empty segment math
            plan.update({"style_header": "",
                         "full_prompt": "",
                         "segments": [{"index": i, "brief": "", "prompt": ""} for i in range(1, count + 1)]})
            preview = [
                f"共 {count} 段 × {plan['segment_seconds_snapped']:.3f}s"
                f"（每段 {plan['frames_per_segment']} 帧，段间重叠 {overlap_frames} 帧）",
                f"成片约 {plan['total_seconds_actual']:.2f}s（目标 {total_seconds:.1f}s）",
                f"参考视频至少要 {plan['ref_frames_needed']} 帧"
                f"（约 {plan['ref_frames_needed'] / max(fps, 1.0):.1f}s @ {fps:.0f}fps）",
                "",
                "提示词/剧本为空 → 没有调用 LLM，只计算了分段帧数。",
            ]
            logger.info("H3-Myang: 剧本为空，跳过 LLM，只做分段（%d 段）", count)
            try:
                from server import PromptServer
                inst = getattr(PromptServer, "instance", None)
                if inst is not None and hasattr(inst, "send_sync"):
                    first_seg = plan["segments"][0] if plan.get("segments") else {}
                    inst.send_sync("myh3_plan_ready", {
                        "total_segments": count,
                        "first_prompt": str(first_seg.get("prompt") or ""),
                        "first_brief": str(first_seg.get("brief") or ""),
                    })
            except Exception:
                pass

            return (json.dumps(plan, ensure_ascii=False), count,
                    plan["segment_seconds_snapped"], plan["frames_per_segment"],
                    "\n".join(preview), plan["ref_frames_needed"])
        # Fixed presets keep their historical single-Skill behaviour. Auto is
        # resolved only after the storyboard exists, so each segment can route
        # to a different primary/overlay bundle without a second planning pass.
        selected_skill_preset = str(
            kwargs.get("skill_preset", SKILL_PRESET_NONE) or SKILL_PRESET_NONE).strip()
        selected_skill_text = str(kwargs.get("skill_text", "") or "")
        # Opt-in: layer the shot prompt so dialogue and sound each get their own
        # short call, and the dialogue gets a character ceiling from the segment's
        # real seconds. Off by default because it costs two extra small calls per
        # segment and the unlayered path is byte-identical to before.
        layered_prompts = _layering_enabled(kwargs.get("分层提示词", False))
        if selected_skill_preset == SKILL_PRESET_AUTO:
            try:
                from . import agent_nodes
                custom_digest = hashlib.sha256(
                    selected_skill_text.encode("utf-8")).hexdigest()[:16]
                skill_rules = ("auto-segment-plan:" + agent_nodes.skill_catalog_signature()
                               + ":" + custom_digest)
            except Exception:
                skill_rules = "auto-segment-plan"
            skill_source = "逐镜头自动技能组合"
        else:
            skill_rules, skill_source = resolve_skill(
                selected_skill_preset, selected_skill_text, llm_service=llm_service,
                ollama_auto_unload=bool(ollama_auto_unload), routing_prompt=text)
        if skill_source:
            logger.info("H3-Myang: 分段写作技能 %s", skill_source)
        seconds_each = plan["segment_seconds_snapped"]
        _source_packets, source_chunks = _build_split_source_packets(
            text, count, media_manifest, seconds_each)
        cache_root = _cache_dir()
        cache_key = _cache_key(
            SPLIT_PROMPT_VERSION, text, count, llm_service, seed,
            media_manifest, skill_rules, round(float(total_seconds), 4),
            round(float(segment_seconds), 4), int(overlap_frames), float(fps))
        cache_file = cache_root / ("%s.json" % cache_key)
        context_key = _agent_context_key(
            text, count, media_manifest, skill_rules, selected_skill_preset,
            selected_skill_text, total_seconds, segment_seconds, overlap_frames, fps)
        context_file = _agent_context_file(context_key)
        resume_context = None
        if use_cache:
            resume_context = _load_agent_context(context_file, context_key, count)
            if resume_context:
                logger.info(
                    "H3-Myang: 命中 Agent 上下文 %s（已保存 %d/%d 段）",
                    context_file.name, len(resume_context.get("segments") or []), count)

        def checkpoint_writer(state):
            envelope = {
                "format": "h3-agent-context",
                "version": AGENT_CONTEXT_VERSION,
                "context_key": context_key,
                "expected_count": int(count),
                "source": {
                    "text": text,
                    "chunks": list(source_chunks),
                    "media_manifest": media_manifest,
                    "total_seconds": float(total_seconds),
                    "segment_seconds": float(segment_seconds),
                    "overlap_frames": int(overlap_frames),
                    "fps": float(fps),
                },
                "updated_at": time.time(),
            }
            envelope.update(state or {})
            _write_json_atomic(context_file, envelope)
            return str(context_file)
        payload = None
        if use_cache and cache_file.is_file():
            try:
                candidate = json.loads(cache_file.read_text(encoding="utf-8"))
                if _cacheable_split_payload(candidate, count):
                    payload = candidate
                    logger.info("H3-Myang: 命中拆分缓存 %s", cache_file.name)
            except Exception:
                payload = None

        if payload is None:
            payload, writer_notes = _write_segments_with_media_agent(
                text, source_chunks, count, segment_seconds, total_seconds,
                overlap_frames, fps, media_manifest, skill_rules, llm_service,
                ollama_auto_unload, seed, skill_preset=selected_skill_preset,
                skill_text=selected_skill_text, skill_source=skill_source,
                checkpoint_writer=checkpoint_writer if use_cache else None,
                resume_context=resume_context if use_cache else None,
                layered_prompts=layered_prompts)
            if not _usable_split_payload(payload, count):
                reason = "；".join(writer_notes[-6:]) or "Media Agent 未返回完整分段正文"
                payload = _local_timeline_split(
                    text, count, media_manifest=media_manifest, reason=reason,
                    maximum_seconds=segment_seconds, total_seconds=total_seconds,
                    overlap_frames=overlap_frames, fps=fps)
                payload["storyboard_source"] = "local_adaptive_timing"
                payload["writer_fallback_segments"] = list(range(1, count + 1))
                logger.warning("H3-Myang: Media Agent 写作不可用，采用本地自适应时长兜底 | %s", reason)
            if use_cache and _cacheable_split_payload(payload, count):
                try:
                    cache_file.write_text(
                        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                except Exception as exc:
                    logger.warning("H3-Myang: 缓存写入失败: %s", exc)
            elif payload.get("writer_fallback_segments"):
                logger.warning(
                    "H3-Myang: 本次含写作兜底段 %s，不写入长期缓存",
                    ",".join(str(value) for value in payload["writer_fallback_segments"]))

        # Final defensive gate: no cache or future caller may reintroduce the
        # old "copy the last segment until N" behavior.
        if not _usable_split_payload(payload, count):
            payload = _local_timeline_split(
                text, count, media_manifest=media_manifest,
                reason="缓存或响应的段数/内容无效",
                maximum_seconds=segment_seconds, total_seconds=total_seconds,
                overlap_frames=overlap_frames, fps=fps)
        style_header = str(payload.get("style_header") or "").strip()
        segments = [seg for seg in (payload.get("segments") or [])
                    if isinstance(seg, dict)]
        for i, seg in enumerate(segments, start=1):
            seg["index"] = i
            seg["brief"] = str(seg.get("brief") or "").strip()
            seg["title"] = _sanitize_storyboard_title(
                seg.get("title"), seg.get("brief") or seg.get("prompt"))
            transition = str(seg.get("transition") or "").strip()
            seg["transition"] = ("开场" if i == 1
                                 else transition if transition in ("承接", "切镜")
                                 else "承接")

        # Dialogue is budgeted against the real segment seconds before any
        # prompt text is assembled, because spilling a line moves it into a
        # different segment's prompt. Segments that carry no layers are left
        # exactly as they arrive, so nothing about the pre-layer path changes.
        layer_notes = []
        global_layers = (dict(payload["layers"])
                         if isinstance(payload.get("layers"), dict) else {})
        if style_header and not _clean_layer_text(global_layers.get("style")):
            global_layers["style"] = style_header
        if any(segment_layers(seg) for seg in segments):
            layer_notes = enforce_dialogue_budget(
                segments,
                lambda seg: float(seg.get("duration_seconds")
                                  or plan["segment_seconds_snapped"]))
            for note in layer_notes:
                logger.warning("H3-Myang: 分层台词预算 | %s", note)

        for seg in segments:
            layers = segment_layers(seg)
            if layers is not None:
                composed = compose_segment_prompt(global_layers, layers)
                if composed:
                    seg["prompt"] = composed
            prompt_val = str(seg.get("prompt") or "").strip()
            if not prompt_val:
                prompt_val = "\n\n".join(p for p in (style_header, seg["brief"]) if p)
            seg["prompt"] = _strip_legacy_character_reference_heading(prompt_val)

        # A seamless boundary duplicates and then trims the overlap window;
        # a hard cut generates the incoming segment independently and keeps
        # all of its frames. Recompute the displayed/runtime length after the
        # LLM has chosen per-boundary transitions.
        connected_boundaries = sum(
            str(seg.get("transition") or "承接") == "承接"
            for seg in segments[1:])
        actual_frames = (
            sum(int(seg.get("frames") or plan["frames_per_segment"])
                for seg in segments)
            - int(overlap_frames) * connected_boundaries)
        plan["total_seconds_actual"] = actual_frames / max(float(fps), 1.0)
        plan["ref_frames_needed"] = actual_frames

        plan.update({"style_header": style_header,
                     "full_prompt": text,
                     "segments": segments})
        if global_layers:
            plan["layers"] = global_layers
        if layer_notes:
            plan["layer_notes"] = layer_notes
        plan["split_source"] = str(payload.get("split_source") or "llm")
        if payload.get("writer_mode"):
            plan["writer_mode"] = str(payload["writer_mode"])
        if payload.get("storyboard_source"):
            plan["storyboard_source"] = str(payload["storyboard_source"])
        if isinstance(payload.get("storyboard"), list):
            plan["storyboard"] = payload["storyboard"]
        if isinstance(payload.get("skill_plan"), list):
            plan["skill_plan"] = payload["skill_plan"]
        if payload.get("skill_strategy"):
            plan["skill_strategy"] = str(payload["skill_strategy"])
        if isinstance(payload.get("writer_fallback_segments"), list):
            plan["writer_fallback_segments"] = list(payload["writer_fallback_segments"])
        if isinstance(payload.get("media_reference_compliance"), dict):
            plan["media_reference_compliance"] = dict(payload["media_reference_compliance"])
        if payload.get("split_fallback_reason"):
            plan["split_fallback_reason"] = str(payload["split_fallback_reason"])
        if use_cache:
            completed_context_segments = len(payload.get("segments") or [])
            plan["agent_context"] = {
                "key": context_key,
                "path": str(context_file),
                "resumed": bool(resume_context),
                "completed_segments": completed_context_segments,
                "pending_segments": max(0, int(count) - completed_context_segments),
            }
        if media_manifest:
            plan["media_manifest"] = media_manifest
        effective_skill_source = str(payload.get("skill_source") or skill_source or "")
        if effective_skill_source:
            plan["skill_source"] = effective_skill_source

        actual_durations = [float(segment.get("duration_seconds") or
                                   plan["segment_seconds_snapped"])
                            for segment in segments]
        duration_range = ("%.2f~%.2fs" % (min(actual_durations), max(actual_durations))) \
            if actual_durations else "--"
        preview = [
            f"共 {count} 段 · 单段上限 {float(segment_seconds):.2f}s · AI实际时长 {duration_range}"
            f"（段间重叠 {overlap_frames} 帧）",
            f"成片约 {plan['total_seconds_actual']:.2f}s（目标 {total_seconds:.1f}s）",
            f"参考视频至少要 {plan['ref_frames_needed']} 帧"
            f"（约 {plan['ref_frames_needed'] / max(fps, 1.0):.1f}s @ {fps:.0f}fps）",
            "",
            (f"[分镜规划] {payload.get('storyboard_source')}"
             if payload.get("storyboard_source") else "[分镜规划] 未单独启用"),
            ("[写作兜底] Media Agent 未返回的段落已用本地时间线补齐"
             if payload.get("writer_fallback_segments") else ""),
            f"[写作技能] {effective_skill_source}" if effective_skill_source else "[写作技能] 默认写法",
            f"[全局设定] {style_header[:120]}…" if style_header else "[全局设定] 无",
            "",
        ]
        if use_cache:
            context_info = plan["agent_context"]
            if context_info["resumed"]:
                preview.append(
                    "[Agent上下文] 已续接 %d/%d 段；未完成段会在本次模型设置下继续。"
                    % (context_info["completed_segments"], count))
            else:
                preview.append(
                    "[Agent上下文] 已启用；每完成一段都会保存，可切换模型后续接。")
        if payload.get("split_source") == "local_fallback" and payload.get(
                "split_fallback_reason"):
            preview.insert(5, "[兜底原因] %s" % str(
                payload["split_fallback_reason"])[:240])
        if media_manifest:
            preview.append("【LLM 可选用的公共素材】")
            for mline in media_manifest.split("\n"):
                if mline.strip():
                    preview.append(f"  {mline.strip()}")
            preview.append("")
        for s in segments:
            idx = s["index"]
            brf = s.get("brief", "")
            pmt = s.get("prompt", "")
            disp = brf if brf else pmt
            title = _sanitize_storyboard_title(
                s.get("title"), s.get("brief") or s.get("prompt"))
            duration = float(s.get("duration_seconds") or 0)
            title_text = f"分镜{idx}：{title}（时长 {duration:.2f}s）"
            preview.append(f"[{title_text}·{s.get('transition', '承接')}] {disp[:90]}…")

        try:
            from server import PromptServer
            inst = getattr(PromptServer, "instance", None)
            if inst is not None and hasattr(inst, "send_sync"):
                first_seg = plan["segments"][0] if plan.get("segments") else {}
                inst.send_sync("myh3_plan_ready", {
                    "total_segments": count,
                    "first_prompt": str(first_seg.get("prompt") or ""),
                    "first_brief": str(first_seg.get("brief") or ""),
                    "segments": plan.get("segments") or [],
                })
        except Exception:
            pass

        return (json.dumps(plan, ensure_ascii=False),
                count,
                plan["segment_seconds_snapped"],
                plan["frames_per_segment"],
                "\n".join(preview),
                plan["ref_frames_needed"])


class H3SegmentPrompt:
    CATEGORY = "沐阳 H3"
    FUNCTION = "build"
    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("prompt", "batch_index")
    DESCRIPTION = "取出第 N 段的提示词。直通模式 0 token；细化模式只送这一段的梗概。"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "plan_json": ("STRING", {"forceInput": True}),
                "segment_index": ("INT", {"default": 1, "min": 1, "max": MAX_SLOTS}),
                "mode": (PROMPT_MODES, {"default": MODE_FIXED}),
                "media_prefix": ("STRING", {
                    "multiline": True,
                    "default": FIXED_PROMPT_DEFAULT,
                    "tooltip": "每段都原样前置的媒体引用句。放在这里而不是交给 LLM，"
                               "是因为模型经常把 @视频1 这类标记改写坏。"
                               "『全片同一提示词』模式下，这里就是全片唯一的提示词。",
                }),
                "llm_service": (llm_service_options(),),
                "carry_prev_tail": ("BOOLEAN", {"default": True,
                                                "tooltip": "细化时附上一段梗概的结尾，让动作衔接更稳"}),
                "ollama_auto_unload": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                                 "control_after_generate": True}),
                "llm_enabled": ("BOOLEAN", {
                    "default": True,
                    "label_on": "调用LLM切片",
                    "label_off": "不调用LLM(全片共用/直通)",
                    "tooltip": "手动选择是否调用 LLM 进行分段切片。"
                               "开启：使用 LLM 将提示词按时间轴切分成各段不同的分镜提示词；"
                               "关闭：不调用 LLM，直接将完整提示词作为各段通用提示词（0 token 消耗）。"}),
            },
        }

    def build(self, plan_json, segment_index, mode, media_prefix, llm_service,
              carry_prev_tail, ollama_auto_unload, seed):
        plan = json.loads(plan_json)
        segments = plan.get("segments") or []
        count = int(plan.get("segment_count") or len(segments))
        index = max(1, min(int(segment_index), max(1, count)))
        header = str(plan.get("style_header") or "").strip()
        seg_entry = segments[index - 1] if index <= len(segments) else {}
        seg_prompt = str(seg_entry.get("prompt") or "").strip()
        brief = str(seg_entry.get("brief") or "").strip()
        prefix = str(media_prefix or "").strip()

        # Reference-video slicing advances by the trimmed length, not the full
        # one, so slice i starts where the previous segment actually ended.
        advance = int(plan.get("frames_per_segment", 0)) - int(plan.get("overlap_frames", 0))
        batch_index = (index - 1) * max(1, advance)

        m_str = str(mode)
        if m_str in (MODE_FIXED, "fixed"):
            # Every segment gets exactly the operator's sentence. The reference
            # video already carries the choreography, so there is nothing per
            # segment to say and no LLM to call.
            return (prefix, batch_index)

        if m_str in (MODE_DIRECT, "直接使用分段稿", "direct") or not m_str:
            if seg_prompt:
                body = seg_prompt
            else:
                body = "\n\n".join(p for p in (header, brief) if p)
        else:
            parts = [f"全局风格：\n{header}", f"本段（第 {index}/{count} 段）剧情：\n{brief or seg_prompt}"]
            if carry_prev_tail and index > 1 and index - 2 < len(segments):
                prev = str(segments[index - 2].get("brief") or "").strip()
                if prev:
                    parts.insert(1, f"上一段的结尾（只用于衔接，不要重复描写）：\n{prev[-160:]}")
            body = call_llm(llm_service, "\n\n".join(parts), REFINE_SYSTEM,
                            ollama_auto_unload, seed)

        if prefix and prefix not in body and ("<Video" not in body and "@视频" not in body and "<Picture" not in body and "@图片" not in body):
            final_prompt = f"{prefix}\n\n{body}" if body else prefix
        else:
            final_prompt = body or prefix
        return (final_prompt, batch_index)


def _broadcast_director_progress(payload):
    """Send a non-blocking progress event without cross-module hot-reload state.

    ``nodes.py`` and ``progress.py`` can be reloaded independently by ComfyUI.
    Keeping final-assembly transport here prevents a newly loaded collector
    from importing an older, already-cached progress module at execution time.
    """
    try:
        from server import PromptServer
        inst = getattr(PromptServer, "instance", None)
        if inst is not None and hasattr(inst, "send_sync"):
            inst.send_sync("myh3_progress", dict(payload or {}))
            return True
    except Exception as exc:  # pragma: no cover - UI transport is optional
        logger.debug("H3-Myang: 合并进度广播失败: %s", exc)
    return False


class H3SegmentCollector:
    CATEGORY = "沐阳 H3"
    FUNCTION = "collect"
    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("images", "audio")
    DESCRIPTION = "按顺序拼接各段画面与声音。未启用的段声明为惰性输入，整条上游链不会执行。"

    @classmethod
    def INPUT_TYPES(cls):
        optional = {
            "run_id": ("STRING", {"default": ""}),
            "owner_id": ("STRING", {"default": ""}),
            "total_segments": ("INT", {"default": 1, "min": 1, "max": MAX_SLOTS}),
        }
        for i in range(1, MAX_SLOTS + 1):
            optional[f"images_{i}"] = ("IMAGE", {"lazy": True})
            optional[f"audios_{i}"] = ("AUDIO", {"lazy": True})
        return {
            "required": {
                "active_count": ("INT", {"default": 1, "min": 1, "max": MAX_SLOTS, "forceInput": True}),
            },
            "optional": optional,
        }

    def check_lazy_status(self, active_count, **kwargs):
        """Only ask for the slots this run actually needs.

        Everything upstream of an unrequested slot is never executed, which is
        what lets one graph serve any segment count without ExecutionBlocker
        poisoning the join.
        """
        needed = []
        for i in range(1, max(1, min(int(active_count), MAX_SLOTS)) + 1):
            for name in (f"images_{i}", f"audios_{i}"):
                if name in kwargs and kwargs[name] is None:
                    needed.append(name)
        return needed

    def collect(self, active_count, run_id="", owner_id="", total_segments=1,
                **kwargs):
        count = max(1, min(int(active_count), MAX_SLOTS))
        total = max(count, int(total_segments or count))
        frames, waves, rate = [], [], None
        for i in range(1, count + 1):
            img = kwargs.get(f"images_{i}")
            if img is not None:
                frames.append(img)
            aud = kwargs.get(f"audios_{i}")
            if aud is not None and aud.get("waveform") is not None:
                waves.append(aud["waveform"])
                rate = rate or aud.get("sample_rate", 44100)
        if not frames:
            raise ValueError("H3SegmentCollector 没有拿到任何画面，请检查各段是否已连线")

        height, width = frames[0].shape[1], frames[0].shape[2]
        for i, f in enumerate(frames[1:], start=2):
            if f.shape[1] != height or f.shape[2] != width:
                raise ValueError(
                    f"第 {i} 段分辨率 {f.shape[2]}x{f.shape[1]} 与第 1 段 {width}x{height} 不一致，"
                    "无法拼接。各段必须共用同一个分辨率来源。"
                )
        if str(run_id or ""):
            logger.info(
                "H3-Myang: 开始合并分段 | %d 段画面与音频 | run=%s",
                count, str(run_id)[:16])
            _broadcast_director_progress({
                "run_id": str(run_id), "owner_id": str(owner_id or ""),
                "segment_index": count, "total_segments": total,
                "stage": "assembling", "prompt": "", "brief": "",
            })

        images = torch.cat(frames, dim=0)

        if waves:
            channels = max(w.shape[1] for w in waves)
            fixed = [w.repeat(1, channels, 1) if w.shape[1] == 1 and channels > 1 else w for w in waves]
            audio = {"waveform": torch.cat(fixed, dim=-1), "sample_rate": rate or 44100}
        else:
            audio = {"waveform": torch.zeros(1, 2, 1), "sample_rate": 44100}
        if str(run_id or ""):
            logger.info(
                "H3-Myang: 分段合并完成 | %d 帧 | run=%s",
                int(images.shape[0]), str(run_id)[:16])
            _broadcast_director_progress({
                "run_id": str(run_id), "owner_id": str(owner_id or ""),
                "segment_index": count, "total_segments": total,
                "stage": "assembled", "prompt": "", "brief": "",
            })
        return (images, audio)


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
        if hasattr(h3, "model"):
            return (h3.model,)
        raise ValueError("H3-Myang: 输入不是可识别的 H3 模型包")


def _pass1_checkpoint_path(prefix, segment_index):
    """Return a stable checkpoint path confined to ComfyUI/output."""
    import folder_paths

    root = os.path.realpath(folder_paths.get_output_directory())
    relative = str(prefix or "video/H3_长视频").strip().replace("\\", "/")
    relative = relative.lstrip("/")
    if not relative:
        relative = "video/H3_长视频"
    candidate = os.path.realpath(os.path.join(
        root, "%s_一采检查点_第%02d段.h3pass1" %
        (relative, max(1, int(segment_index)))))
    try:
        inside = os.path.commonpath((root, candidate)) == root
    except ValueError:
        inside = False
    if not inside:
        raise ValueError("一采检查点路径不能超出 ComfyUI/output")
    return candidate


def _h3_latent_streams(samples):
    payload = samples.get("samples") if isinstance(samples, dict) else samples
    if hasattr(payload, "unbind"):
        streams = list(payload.unbind())
    elif isinstance(payload, (tuple, list)):
        streams = list(payload)
    else:
        streams = [payload]
    if len(streams) < 2:
        raise ValueError("一采检查点必须同时包含 H3 视频与音频 latent")
    if not all(torch.is_tensor(stream) for stream in streams[:2]):
        raise ValueError("一采检查点收到无效的 latent 数据")
    return streams[0], streams[1]


class H3Pass1CheckpointSave:
    """Persist the exact packed H3 result so pass 1 can be skipped later."""

    CATEGORY = "沐阳 H3/内部"
    FUNCTION = "save"
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("samples",)
    DESCRIPTION = "保存完整一采视频/音频 latent；内部节点。"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "samples": ("LATENT",),
            "filename_prefix": ("STRING", {"default": "video/H3_长视频"}),
            "segment_index": ("INT", {"default": 1, "min": 1, "max": 9999}),
            "frames": ("INT", {"default": 125, "min": 1, "max": 100000}),
        }}

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Saving is an intentional side effect and must survive graph caching.
        return float("NaN")

    def save(self, samples, filename_prefix, segment_index, frames):
        import safetensors.torch

        video, audio = _h3_latent_streams(samples)
        path = _pass1_checkpoint_path(filename_prefix, segment_index)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        temporary = path + ".tmp"
        payload = {
            "video_latent": video.detach().to("cpu").contiguous(),
            "audio_latent": audio.detach().to("cpu").contiguous(),
            "h3_pass1_version": torch.tensor([1], dtype=torch.int32),
            "segment_index": torch.tensor([int(segment_index)], dtype=torch.int32),
            "frames": torch.tensor([int(frames)], dtype=torch.int32),
        }
        try:
            # Use safetensors directly. ComfyUI's generic saver also collects
            # prompt metadata and can wait on the global output-save path;
            # this internal pass-through node needs a small deterministic AV
            # checkpoint without holding up the expanded sampling graph.
            safetensors.torch.save_file(payload, temporary)
            os.replace(temporary, path)
        finally:
            if os.path.isfile(temporary):
                try:
                    os.remove(temporary)
                except OSError:
                    pass
        logger.info("H3-Myang: 已保存一采检查点 | 第%d段 | %s",
                    int(segment_index), path)
        return (samples,)


class H3Pass1CheckpointLoad:
    """Load one stable per-segment H3 pass-1 checkpoint."""

    CATEGORY = "沐阳 H3/内部"
    FUNCTION = "load"
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("samples",)
    DESCRIPTION = "读取完整一采检查点并直接交给二采；内部节点。"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "filename_prefix": ("STRING", {"default": "video/H3_长视频"}),
            "segment_index": ("INT", {"default": 1, "min": 1, "max": 9999}),
            "expected_frames": ("INT", {"default": 125, "min": 1, "max": 100000}),
        }}

    @classmethod
    def IS_CHANGED(cls, filename_prefix, segment_index, expected_frames):
        path = _pass1_checkpoint_path(filename_prefix, segment_index)
        try:
            stat = os.stat(path)
            return "%s:%d:%d" % (path, stat.st_size, stat.st_mtime_ns)
        except OSError:
            return "%s:missing" % path

    def load(self, filename_prefix, segment_index, expected_frames):
        import comfy.nested_tensor
        import safetensors.torch

        path = _pass1_checkpoint_path(filename_prefix, segment_index)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                "找不到第%d段一采检查点：%s。请先选择『保存一采检查点』跑完一采，"
                "并保持相同的分段文件名前缀。" % (int(segment_index), path))
        payload = safetensors.torch.load_file(path, device="cpu")
        if int(payload.get("h3_pass1_version", torch.tensor([0]))[0]) != 1:
            raise ValueError("一采检查点版本不受支持：%s" % path)
        saved_segment = int(payload.get("segment_index", torch.tensor([-1]))[0])
        saved_frames = int(payload.get("frames", torch.tensor([-1]))[0])
        if saved_segment != int(segment_index):
            raise ValueError(
                "一采检查点段号不匹配：文件是第%d段，当前需要第%d段" %
                (saved_segment, int(segment_index)))
        if saved_frames != int(expected_frames):
            raise ValueError(
                "第%d段一采检查点帧数为%d，当前分镜需要%d；请恢复原分镜时长/"
                "段间锚点，或重新生成该段一采。" %
                (int(segment_index), saved_frames, int(expected_frames)))
        video = payload.get("video_latent")
        audio = payload.get("audio_latent")
        if video is None or audio is None:
            raise ValueError("一采检查点缺少视频或音频 latent：%s" % path)
        # safetensors exposes CPU tensors backed by its mapped file. Returning
        # those storages keeps the checkpoint locked on Windows for as long as
        # the latent remains alive, blocking overwrite, cleanup and reruns.
        video = video.clone()
        audio = audio.clone()
        logger.info("H3-Myang: 复用一采检查点，跳过一采 | 第%d段 | %s",
                    int(segment_index), path)
        return ({"samples": comfy.nested_tensor.NestedTensor((video, audio))},)


class H3Pass1VideoEncode:
    """Fallback: reconstruct an H3 AV latent from one finished pass-1 clip."""

    CATEGORY = "沐阳 H3/内部"
    FUNCTION = "encode"
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("samples",)
    DESCRIPTION = "把单段一采成片重新编码成 H3 latent 后直接二采；内部节点。"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "base_latent": ("LATENT",),
            "video_vae": ("VAE",),
            "audio_vae": ("VAE",),
            "expected_frames": ("INT", {"default": 125, "min": 1}),
        }, "optional": {"audio": ("AUDIO",)}}

    def encode(self, images, base_latent, video_vae, audio_vae,
               expected_frames, audio=None):
        import comfy.nested_tensor
        import torchaudio

        if int(images.shape[0]) != int(expected_frames):
            raise ValueError(
                "接入的一采成片有%d帧，当前单段分镜需要%d帧；请让分镜时长与"
                "原一采文件一致。" % (int(images.shape[0]), int(expected_frames)))
        video = video_vae.encode(images)
        _, base_audio = _h3_latent_streams(base_latent)
        encoded_audio = base_audio
        if isinstance(audio, dict) and audio.get("waveform") is not None:
            waveform = audio["waveform"]
            source_rate = int(audio.get("sample_rate", 44100))
            target_rate = int(getattr(audio_vae, "audio_sample_rate", 44100))
            if source_rate != target_rate:
                waveform = torchaudio.functional.resample(
                    waveform, source_rate, target_rate)
            encoded_audio = audio_vae.encode(waveform.movedim(1, -1))
        logger.info("H3-Myang: 已把单段一采成片重新编码，跳过一采采样")
        return ({"samples": comfy.nested_tensor.NestedTensor(
            (video, encoded_audio))},)


def _vram_usage_breakdown(device):
    """One-line VRAM accounting for the phase-barrier log.

    The barrier's own "净释放" only compares free memory before/after. When
    evicted pages land back in a DLL-side pool, free memory stays flat and
    several GB go unaccounted -- append torch's allocator numbers and AIMDO's
    own usage so the next log pinpoints the holder instead of guessing.
    """
    parts = []
    try:
        if (device is not None and getattr(device, "type", "") == "cuda"
                and torch.cuda.is_available()):
            stats = torch.cuda.memory_stats(device)
            active = int(stats.get("active_bytes.all.current", 0)) or 0
            reserved = int(stats.get("reserved_bytes.all.current", 0)) or 0
            parts.append("torch已用 %.2fGB·缓存 %.2fGB" % (
                active / 1024 ** 3,
                max(0, reserved - active) / 1024 ** 3))
    except Exception:  # pragma: no cover - reporting only
        pass
    try:
        import comfy_aimdo.control as aimdo_control
        usage = int(aimdo_control.get_total_vram_usage() or 0)
        if usage > 0:
            parts.append("AIMDO占用 %.2fGB" % (usage / 1024 ** 3))
    except Exception:  # pragma: no cover - optional aimdo build
        pass
    return ("｜" + "·".join(parts)) if parts else ""


def _loaded_entries_for(model, include_clones=True):
    """LoadedModel entries backed by *model* and optionally related clones."""
    import comfy.model_management as model_management

    if model is None:
        return []
    base_uuid = getattr(model, "clone_base_uuid", None)
    real = getattr(model, "model", None)
    keep = []
    for loaded in list(model_management.current_loaded_models):
        patcher = getattr(loaded, "model", None)
        if patcher is None:
            continue
        if patcher is model:
            keep.append(loaded)
        elif not include_clones:
            continue
        elif (base_uuid is not None
                and getattr(patcher, "clone_base_uuid", None) == base_uuid):
            keep.append(loaded)
        elif real is not None and getattr(patcher, "model", None) is real:
            keep.append(loaded)
    return keep


def _clear_dynamic_transient_pins(patcher):
    """Release VBAR pins left on a model after an interrupted/last block.

    AIMDO only reclaims pages that are no longer pinned.  The normal prefetch
    cleanup covers queues that are still registered, but a sampler exception
    or a final block can leave ``_prefetch``/``_v_block_faulted`` directly on a
    module after its queue has already been discarded.  Clear only those
    transient markers; the model's staged host buffers and reusable signatures
    stay intact.
    """
    if patcher is None:
        return 0
    model = getattr(patcher, "model", None)
    if model is None:
        return 0
    try:
        import comfy_aimdo.model_vbar as aimdo_vbar
    except Exception:
        return 0
    released = 0
    for module in model.modules():
        prefetch = getattr(module, "_prefetch", None)
        if prefetch is not None:
            try:
                if prefetch.get("signature") is not None:
                    allocation = getattr(module, "_v", None)
                    if allocation is not None:
                        aimdo_vbar.vbar_unpin(allocation)
                        released += 1
            except Exception:
                pass
            try:
                delattr(module, "_prefetch")
            except AttributeError:
                pass
        if getattr(module, "_v_block_faulted", False):
            try:
                allocation = getattr(module, "_v_block", None)
                if allocation is not None:
                    aimdo_vbar.vbar_unpin(allocation)
                    released += 1
            except Exception:
                pass
            try:
                delattr(module, "_v_block_faulted")
            except AttributeError:
                pass
    return released


def _release_comfy_models(stage, keep_model=None, keep_clones=True,
                          preserve_dynamic_host_cache=False):
    """Offload registered patchers and return unused CUDA blocks to the pool.

    ``unload_all_models`` also runs ``partially_unload_ram`` on every dynamic
    patcher, which truncates AIMDO's pinned host buffer back to zero. The next
    pass then re-stages the whole DiT from disk instead of from RAM, which is
    what leaves the second pass at ~1GB pinned while tens of GB of RAM sit idle.
    When the caller names a model that is about to be used again, keep its
    LoadedModel entry so only the VAE/text-encoder/upscaler residency is freed.
    """
    import gc
    import comfy.model_prefetch
    import comfy.model_management as model_management

    device = None
    free_before = None
    try:
        device = model_management.get_torch_device()
        free_before = int(model_management.get_free_memory(device))
    except Exception:
        # Memory reporting is diagnostic only; never make the barrier fail on
        # CPU-only/test environments or third-party device implementations.
        pass

    # A model can be absent from current_loaded_models yet still own prefetched
    # VBAR pages, CUDA graph streams or reusable cast buffers.  Merely calling
    # free_memory/empty_cache leaves those allocations pinned, which showed up
    # as several GB of "other" memory at pass-2 step zero.  Release the transient
    # execution machinery first while retaining the weight host buffers.
    comfy.model_prefetch.cleanup_prefetch_queues()
    model_management.reset_cast_buffers()
    gc.collect()
    keep = _loaded_entries_for(keep_model, include_clones=keep_clones)
    dynamic_keep = []
    dynamic_freed = 0
    if preserve_dynamic_host_cache:
        logger.info(
            "H3-Myang: AIMDO 动态显存疏通 | %s | "
            "仅释放 AIMDO GPU 页，保留模型 RAM 缓存", stage)
        # AIMDO's normal ``unload_all_models`` eventually calls
        # ModelPatcherDynamic.detach(), which also truncates every pinned host
        # buffer.  With two H3 inputs (Turbo first pass + Ref2VA second pass)
        # that turns this tiny boundary into a many-GB RAM teardown/reload and
        # can sit at 100% GPU while the VAE is waiting.  Evict only VBAR GPU
        # pages here and retain the 64GB-system-RAM cache for the later sampler.
        # Non-dynamic stale models (old VAE/CLIP/upscaler) are still unloaded.
        seen = set()
        candidates = []
        for loaded in list(model_management.current_loaded_models):
            patcher = getattr(loaded, "model", None)
            if patcher is None:
                continue
            try:
                is_dynamic = bool(patcher.is_dynamic())
            except Exception:
                is_dynamic = False
            if not is_dynamic:
                continue
            dynamic_keep.append(loaded)
            candidates.append(patcher)
            seen.add(id(patcher))
            loaded.currently_used = False
        if keep_model is not None and id(keep_model) not in seen:
            try:
                if bool(keep_model.is_dynamic()):
                    candidates.append(keep_model)
            except Exception:
                pass
        # A previous sampler can have promoted its VBAR to the highest AIMDO
        # priority (``ModelPatcherDynamic.load`` calls ``prioritize``).  That
        # priority is intentionally sticky across nodes, so a direct
        # ``free_memory`` may leave the just-finished pass resident even after
        # prefetch buffers have been cleaned.  Clear stale limits and lower
        # each VBAR's priority at this phase boundary.  The next sampler calls
        # ``prioritize`` again when it actually needs the model.
        try:
            import comfy_aimdo.model_vbar as aimdo_vbar
            aimdo_vbar.vbars_reset_watermark_limits()
        except Exception:
            pass
        failed_dynamic_ids = set()
        deprioritized = 0
        transient_pins = 0
        for patcher in candidates:
            try:
                transient_pins += _clear_dynamic_transient_pins(patcher)
                get_vbar = getattr(patcher, "_vbar_get", None)
                if callable(get_vbar):
                    vbar = get_vbar()
                    lower_priority = getattr(vbar, "deprioritize", None)
                    if callable(lower_priority):
                        lower_priority()
                        deprioritized += 1
                before = int(patcher.loaded_size())
                patcher.partially_unload(
                    getattr(patcher, "offload_device", None), 1e32)
                after = int(patcher.loaded_size())
                dynamic_freed += max(0, before - after)
            except Exception as exc:
                failed_dynamic_ids.add(id(patcher))
                logger.warning(
                    "H3-Myang: 条件编码前释放 AIMDO GPU 页失败，将继续使用"
                    "ComfyUI 清场 | %s", exc)
        if failed_dynamic_ids:
            # Do not protect a patcher whose targeted eviction failed; the
            # regular ComfyUI path may be slower, but it must be allowed to
            # detach that model rather than leave it beside the VAE.
            dynamic_keep = [
                loaded for loaded in dynamic_keep
                if id(getattr(loaded, "model", None)) not in failed_dynamic_ids]
        for target_device in model_management.get_all_torch_devices():
            model_management.free_memory(
                1e30, target_device, keep_loaded=dynamic_keep)
        keep = dynamic_keep
        if deprioritized:
            logger.info(
                "H3-Myang: AIMDO 已解除 %d 个 VBAR 的阶段优先级；"
                "下一阶段加载时会自动恢复优先级", deprioritized)
        if transient_pins:
            logger.info(
                "H3-Myang: AIMDO 清理 %d 个遗留临时页固定，"
                "允许阶段屏障回收对应 GPU 页", transient_pins)
    elif keep:
        for target_device in model_management.get_all_torch_devices():
            model_management.free_memory(1e30, target_device, keep_loaded=keep)
    else:
        model_management.unload_all_models()
    model_management.soft_empty_cache()
    if preserve_dynamic_host_cache:
        suffix = "｜释放动态GPU页 %.2fGB，保留 %d 个模型的RAM缓存" % (
            dynamic_freed / 1024 ** 3, len(dynamic_keep))
    else:
        suffix = "｜保留 %d 个模型的宿主缓存" % len(keep) if keep else ""
    try:
        free_after = int(model_management.get_free_memory(device))
    except Exception:
        free_after = None
    if free_before is not None and free_after is not None:
        logger.info(
            "H3-Myang: 显存阶段疏通完成 | %s%s | 可用显存 %.2fGB → %.2fGB"
            "（净释放 %.2fGB）%s",
            stage, suffix, free_before / 1024 ** 3, free_after / 1024 ** 3,
            (free_after - free_before) / 1024 ** 3,
            _vram_usage_breakdown(device))
    else:
        logger.info("H3-Myang: 显存阶段疏通完成 | %s%s", stage, suffix)


class H3PreConditionMemoryBarrier:
    """Evict stale DiT residency before reference/CLIP/VAE conditioning.

    H3LongVideo receives a MODEL input, so ComfyUI may keep that model from a
    previous queue or stage it while the dynamic graph is being expanded. The
    reference VAE then has to fit beside several GB of DiT weights and can OOM
    before pass 1 even starts. Carry the MODEL as an ordering dependency, clear
    all registered model residency, and return only the lightweight H3 bundle.
    The same ModelPatcher object is loaded normally when BasicGuider samples.
    """

    CATEGORY = "沐阳 H3/内部"
    FUNCTION = "release"
    RETURN_TYPES = ("MYANG_H3",)
    RETURN_NAMES = ("h3",)
    DESCRIPTION = "内部显存阶段屏障：条件编码前卸载残留 DiT，避免与 VAE 同驻显存。"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "h3": ("MYANG_H3",),
            "stage": ("STRING", {"default": "sampling model -> conditioning"}),
        }, "optional": {
            # The value is intentionally not returned. It exists so graph
            # scheduling cannot run the cleanup before the upstream MODEL has
            # finished and registered any retained GPU residency.
            "loaded_model": ("MODEL",),
        }}

    def release(self, h3, stage="sampling model -> conditioning",
                loaded_model=None):
        _release_comfy_models(
            stage, keep_model=loaded_model,
            preserve_dynamic_host_cache=True)
        return (h3,)


class H3ConditionMemoryBarrier:
    """Run after all CLIP/reference/VAE conditioning and before the DiT."""

    CATEGORY = "沐阳 H3/内部"
    FUNCTION = "release"
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    DESCRIPTION = "内部显存阶段屏障：条件编码后卸载 CLIP/VAE，再进入一采。"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # The conditioning itself is frequently cached across seed changes;
        # cleanup is a side effect and therefore must still run every queue.
        return float("NaN")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "conditioning": ("CONDITIONING",),
            "stage": ("STRING", {"default": "conditioning -> sampling"}),
        }, "optional": {
            "keep_model": ("MODEL", {
                "tooltip": "本段马上要用的 DiT；保留它的宿主缓存，只清 CLIP/VAE",
            }),
        }}

    def release(self, conditioning, stage="conditioning -> sampling",
                keep_model=None):
        _release_comfy_models(stage, keep_model=keep_model)
        return (conditioning,)


class H3RefineMemoryBarrier:
    """Wait for pass-2 inputs, then evict every dynamic GPU page.

    ``keep_model`` is an ordering/host-cache hint, not a request to leave the
    model resident on the device.  In a multi-segment run the first-pass and
    second-pass patchers can share the same underlying dynamic VBAR.  Keeping
    that entry in ComfyUI's ``keep_loaded`` list used to preserve the pages
    touched by the preceding first pass, so segment 2 entered refinement with
    less headroom than segment 1.  Retain the pinned host cache and rematerialize
    pages on demand instead; this is both deterministic and much cheaper than
    spilling the activation into Windows shared VRAM.
    """

    CATEGORY = "沐阳 H3/内部"
    FUNCTION = "release"
    RETURN_TYPES = ("CONDITIONING", "LATENT")
    RETURN_NAMES = ("conditioning", "latent")
    DESCRIPTION = "内部显存阶段屏障：二采 latent/条件就绪后清场，再加载二采模型。"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "conditioning": ("CONDITIONING",),
            "latent": ("LATENT",),
            "stage": ("STRING", {"default": "pass-1/VAE/upscaler -> pass-2"}),
        }, "optional": {
            "keep_model": ("MODEL", {
                "tooltip": "二采基模；保留它的宿主缓存，只清一采/VAE/放大器",
            }),
        }}

    def release(self, conditioning, latent,
                stage="pass-1/VAE/upscaler -> pass-2", keep_model=None):
        _release_comfy_models(
            stage, keep_model=keep_model,
            preserve_dynamic_host_cache=True)
        return (conditioning, latent)


class H3OutputMemoryRelease:
    """Release model residency once the final IMAGE/AUDIO output exists."""

    CATEGORY = "沐阳 H3/内部"
    FUNCTION = "release"
    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("images", "audio")
    DESCRIPTION = "内部显存阶段屏障：成片合并后卸载模型，不复制输出张量。"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "audio": ("AUDIO",),
            "stage": ("STRING", {"default": "output complete"}),
        }}

    def release(self, images, audio, stage="output complete"):
        _release_comfy_models(stage)
        return (images, audio)


def _compact_h3_video_latent_tail(latent, context_length, label):
    """Replace a cached joint latent with the CPU video tail it still needs."""
    if not isinstance(latent, dict) or latent.get("samples") is None:
        return None

    from .anchors import steps_for_frames

    payload = latent["samples"]
    if torch.is_tensor(payload):
        parts = [payload]
    elif hasattr(payload, "unbind"):
        parts = list(payload.unbind())
    elif isinstance(payload, (tuple, list)):
        parts = list(payload)
    else:
        parts = [payload]
    if not parts or not torch.is_tensor(parts[0]):
        raise ValueError("H3-Myang: %s缺少有效的视频 latent" % label)
    video = parts[0]
    if video.ndim == 4:
        video = video.unsqueeze(0)
    if video.ndim != 5:
        raise ValueError(
            "H3-Myang: %s视频 latent 应为 [B,C,T,H,W]，实际 %s" %
            (label, tuple(video.shape)))
    steps = steps_for_frames(int(context_length))
    if steps is None or steps > int(video.shape[2]):
        raise ValueError(
            "H3-Myang: %s承接 %s 帧无法从 %d 个 latent step 提取" %
            (label, context_length, int(video.shape[2])))
    full_bytes = sum(
        int(part.numel()) * int(part.element_size())
        for part in parts if torch.is_tensor(part))
    tail = video[:1, :, -steps:].detach().to("cpu").contiguous()
    compact = {"samples": tail}
    # This used to also do ``latent["samples"] = tail`` so the full tensor's
    # storage could die before the next segment began. That mutates a LATENT
    # dict the finished segment is still reachable through: under the RAM
    # pressure cache (this build's default) the segment's decoded pixels get
    # evicted, the collector's late demand for ``images_%d`` re-runs VAEDecode
    # on the truncated latent, and the segment comes back as the 22-frame
    # anchor tail. A 226+119 frame film then assembled as 141 frames with the
    # whole first segment missing. The full latent is 18-30MB here, which is
    # not worth trading a silently truncated film for, so it now stays intact
    # and only the returned tail is compacted.
    logger.info(
        "H3-Myang: 段间%s压缩 | 完整AV %.1fMB -> CPU尾部 %d帧/%.1fMB",
        label, full_bytes / 1024 ** 2, int(context_length),
        int(tail.numel()) * int(tail.element_size()) / 1024 ** 2)
    del payload, parts, video
    return compact


class H3SegmentMemoryBarrier:
    """Clear pass-2/high-resolution residency before the next segment starts."""

    CATEGORY = "沐阳 H3/内部"
    FUNCTION = "release"
    RETURN_TYPES = ("IMAGE", "AUDIO", "MYANG_H3", "LATENT", "LATENT")
    RETURN_NAMES = ("images", "audio", "h3", "detail_context",
                    "pass1_context")
    DESCRIPTION = "内部段间显存屏障：二采完成后清场，再进入下一段一采。"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "audio": ("AUDIO",),
            "stage": ("STRING", {"default": "pass-2 -> next pass-1"}),
        }, "optional": {
            # Returning this lightweight bundle creates a real dependency from
            # the next segment to this cleanup.  Older expanded graphs remain
            # valid because the input is optional and output indices 0/1 stay
            # unchanged.
            "h3": ("MYANG_H3",),
            "keep_model": ("MODEL", {
                "tooltip": "下一段继续使用的一采模型；仅保留其宿主缓存",
            }),
            "detail_latent": ("LATENT", {
                "tooltip": "本段最终二采 latent；屏障只保留 CPU 尾部供下一段承接",
            }),
            "pass1_latent": ("LATENT", {
                "tooltip": "本段最终一采 latent；屏障只保留 CPU 尾部供下一段一采承接",
            }),
            "context_length": (CONTEXT_LENGTHS, {"default": "22"}),
        }}

    def release(self, images, audio, stage="pass-2 -> next pass-1",
                h3=None, keep_model=None, detail_latent=None,
                pass1_latent=None, context_length="22"):
        detail_context = _compact_h3_video_latent_tail(
            detail_latent, context_length, "二采 latent")
        pass1_context = _compact_h3_video_latent_tail(
            pass1_latent, context_length, "一采 latent")
        # Drop every dynamic model's VBAR pages, including the just-finished
        # Ref2VA pass, but retain their staged host buffers. A normal full
        # unload also truncates the RAM cache, so segment 2 has to restage the
        # same multi-GB weights and can hit a transient WDDM residency spike.
        _release_comfy_models(
            stage, keep_model=keep_model,
            preserve_dynamic_host_cache=True)
        return (images, audio, h3, detail_context, pass1_context)


class _H3LongVideoInputs:
    CATEGORY = "沐阳 H3"
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("images", "audio")
    DESCRIPTION = ("一个节点跑完整条长视频。运行时按分段计划的段数展开成 N 条采样链，"
                   "段间自动衔接并做漂移校正，最后拼成成片。图里只有这一个节点。")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "h3": ("MYANG_H3", {"tooltip": "接「沐阳 H3 加载器」"}),
                "model": ("MODEL",),
                "sampler": ("SAMPLER",),
                # Segmentation maths only -- how many segments, how many frames
                # each, how much they overlap. It carries per-segment briefs too
                # when a script was written, but that is a fallback for an empty
                # prompt box, not a competing prompt input.
                "plan_json": ("STRING", {
                    "forceInput": True,
                    "tooltip": "分段计划：段数 / 每段帧数 / 重叠。不是提示词。"}),
                "task_mode": (TASK_MODES, {
                    "default": TASK_TRANSFER,
                    "tooltip": "动作迁移：每段跟随参考视频对应的那一片。"
                               "视频续写：只取参考视频的结尾作为起点，接着往下演，"
                               "画面和声音都从那里无缝接上。"
                               "纯生成：不用参考视频，全靠提示词和图片。"}),
                "resolution": (RESOLUTIONS, {"default": "480P"}),
                "aspect_ratio": (ASPECTS, {"default": "16:9"}),
                "width": ("INT", {"default": 1344, "min": 32, "max": 16384, "step": 32}),
                "height": ("INT", {"default": 768, "min": 32, "max": 16384, "step": 32}),
                "steps": ("INT", {"default": 8, "min": 1, "max": 200}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "scheduler": (SCHEDULERS, {"default": "simple"}),
                "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                                       "control_after_generate": True}),
                "context_length": (CONTEXT_LENGTHS, {
                    "default": "22",
                    "tooltip": "必须与 H3ScriptSplitter 的 overlap_frames 一致，否则段数和切片会错位"}),
                # No "whole film" option here: that case is simply a non-empty
                # prompt box, which takes precedence and hides these two.
                "prompt_mode": ([MODE_DIRECT, MODE_REFINE], {
                    "default": MODE_DIRECT,
                    "tooltip": "只在 prompt 框留空时生效：逐段用 plan_json 的分段稿"}),
                # Deliberately single-line. A multiline widget renders as an
                # anonymous textarea on this frontend -- no label above it -- so
                # as a multiline field it read as a second prompt box sitting
                # next to the real editor. It is one sentence; one line is enough.
                "media_prefix": ("STRING", {
                    "default": "参考@视频1中的人物动作表情、镜头角度、画面风格，仅将人物替换成@图片1。",
                    "tooltip": "只在走分段稿时用：每段提示词前面原样加上这句媒体引用"}),
                "llm_service": (llm_service_options(),),
                # Default off. Correction only pays for itself on long chains,
                # and getting it wrong is more visible than the drift it fixes.
                "drift_method": (["off", "mean_std (逐通道)", "mkl (推荐·全协方差)"], {
                    "default": "off",
                    "tooltip": "抵消链式续写的累积偏色。只比对切口两侧的少量帧，"
                               "所以镜头和光线的真实变化不会被压掉。"
                               "段数多（4 段以上）再开；mean_std 更温和，mkl 更彻底。"
                               "开启时段间改走像素路径（多一次尾帧编码）"}),
                "drift_strength": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.05,
                                             "tooltip": "0.6 起步。1.0 是完全对齐上一段"}),
                "ref_image_size": (list(CORE_REF_SIZES), {
                    "default": "匹配生成分辨率",
                    "tooltip": "参考素材怎么缩放。匹配生成分辨率最省显存；"
                               "最大保真走 2048 短边，画面更像但每一步都要带着它算，慢好几倍"}),
                "save_segments": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "每段单独存一份。长片跑到一半崩了还能接着用，"
                               "也方便挑出想重跑的那一段"}),
                "segment_prefix": ("STRING", {"default": "video/H3_长视频"}),
            },
            "optional": {
                # One media input. The image_* slots duplicated what the bundle
                # already carries, and the media_*/media_links_json transport
                # slots were only ever filled by the other pack's frontend --
                # here they showed up as an empty text field that did nothing.
                "ref_video": ("IMAGE", {
                    "tooltip": "动作迁移要它够长（每段切一片）；续写只用它的结尾；"
                               "纯生成不用接"}),
                "ref_audio": ("AUDIO", {
                    "tooltip": "续写时接上，声音也从参考视频的结尾接着走"}),
                "media": ("MINIMAX_H3_MEDIA", {
                    "tooltip": "接 MiniMaxH3MediaAgent 的 media 输出。全片素材都从这里来；"
                               "循环只把包里那条参考视频换成本段切片，编号不变。"}),
            },
        }

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


# plan_segments looks this global up at call time.  Point it at the same
# official helper used by H3Condition so planning and latent allocation cannot
# disagree on the 17k+5 frame grid.
frame_length = _native_frame_length


class H3ScriptSplitter(_H3ScriptSplitterBase):
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
              detail_boost=DETAIL_BOOST_NONE, media=None, **kwargs):
        if abs(float(fps) - 24.0) > 1e-6:
            raise ValueError("MiniMax H3 固定按 24fps 建模；分段 fps 必须是 24")
        return super().split(
            script, total_seconds, length_source, segment_seconds,
            overlap_frames, 24.0, llm_service, max_segments,
            ollama_auto_unload, use_cache, seed, llm_enabled=llm_enabled,
            detail_boost=detail_boost, media=media, **kwargs)

class H3FramesToSeconds:
    """Keep dynamic MAINodes frame counts type-safe for H3Condition."""

    CATEGORY = "沐阳 H3/导演台/内部"
    FUNCTION = "convert"
    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("seconds",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "frames": ("INT", {"forceInput": True}),
            "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0}),
        }}

    def convert(self, frames, fps=24.0):
        return (float(frames) / max(float(fps), 1.0),)


class H3LatentIdentity:
    """Expose a linked latent as a normal graph node output.

    Continuous-Sigma sampling needs both the noisy split-state and the clean
    x0 preview returned by one sampler.  The rest of the Director deliberately
    treats a segment latent as a node with ``out(0)``; this tiny internal node
    preserves that stable contract without branching the whole pipeline.
    """

    CATEGORY = "沐阳 H3/导演台/内部"
    FUNCTION = "forward"
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("samples",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"samples": ("LATENT",)}}

    def forward(self, samples):
        return (samples,)


def _roughcut_hidden_layout(frames, overlap, hidden_head=False,
                            hidden_tail=False):
    """Return exact visible/head/tail frame accounting for one H3 segment."""
    visible = int(frames)
    context = int(overlap)
    head = context if hidden_head else 0
    requested = visible + head + (context if hidden_tail else 0)
    conditioned = (core.length_for(requested / 24.0, 24.0)
                   if head or hidden_tail else visible)
    tail_start = visible + head
    tail_trim = max(0, conditioned - tail_start)
    return {
        "visible_frames": visible,
        "head_trim_frames": head,
        "tail_start_frame": tail_start,
        "tail_trim_frames": tail_trim,
        "condition_frames": conditioned,
    }

class H3LongVideo(_H3LongVideoInputs):
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
            CONTEXT_LENGTHS, {
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
        schema["optional"]["增强设置"] = (
            "MYANG_H3_ENHANCE", {
                "tooltip": "导演台可选的 H3 FaceRefine / MAINodes 分段增强设置",
            })
        schema["optional"]["音频设置"] = (
            "MYANG_H3_AUDIO", {
                "tooltip": "导演台集成的音频精修与段间接缝平滑设置",
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
        schema["optional"]["first_frame"] = (
            "IMAGE", {"tooltip": "只约束整条生成区间的第一帧；该段自动使用 FL2VA"})
        schema["optional"]["last_frame"] = (
            "IMAGE", {"tooltip": "只约束整条生成区间的最后一帧；该段自动使用 FL2VA"})
        schema["optional"]["fl2va_model"] = (
            "MODEL", {"tooltip": "首尾帧约束段使用的 FL2VA 模型（导演台自动提供）"})
        schema["optional"]["timeline_head_context"] = (
            "IMAGE", {"tooltip": "粗剪入点之前的 MotionContext 画面窗口"})
        schema["optional"]["timeline_head_audio"] = (
            "AUDIO", {"tooltip": "粗剪入点之前的 MotionContext 声音窗口"})
        schema["optional"]["timeline_tail_context"] = (
            "IMAGE", {"tooltip": "粗剪出点之后的隐藏 MotionContext 画面窗口"})
        schema["optional"]["timeline_tail_audio"] = (
            "AUDIO", {"tooltip": "粗剪出点之后的隐藏 MotionContext 声音窗口"})
        schema["optional"]["timeline_first_keyframe"] = (
            "IMAGE", {"tooltip": "粗剪入点单帧关键图；与 Ref2VA 素材条件可组合"})
        schema["optional"]["timeline_last_keyframe"] = (
            "IMAGE", {"tooltip": "粗剪出点单帧关键图；与 Ref2VA 素材条件可组合"})
        schema["optional"]["roughcut_exact_duration"] = (
            "BOOLEAN", {
                "default": False,
                "tooltip": "导演台内部：为 I/O 外部上下文额外生成隐藏帧，裁剪后保持精确时长"})
        schema["optional"]["一采显存策略"] = (
            FIRST_MEMORY_PROFILES, {
                "default": FIRST_MEMORY_AUTO,
                "tooltip": "自动档在条件编码后卸载 CLIP/VAE，并关闭采样中的逐步"
                           "VAE解码；采样完成后的清晰分段预览仍会保留。"})
        schema["optional"]["一采断点模式"] = (
            PASS1_CHECKPOINT_MODES, {
                "default": PASS1_CHECKPOINT_OFF,
                "tooltip": "保存模式把每段完整音画 latent 写入 output；读取模式按"
                           "分段文件名前缀和绝对段号直接进入二采。成片模式仅支持单段。"})
        schema["optional"]["一采成片"] = (
            "IMAGE", {"tooltip": "兼容入口：接一段已保存的一采视频，重新编码后直接二采"})
        schema["optional"]["一采成片音频"] = (
            "AUDIO", {"tooltip": "可选：与一采成片配套的音频"})
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
        first_frame = kwargs.get("first_frame")
        last_frame = kwargs.get("last_frame")
        fl2va_model = kwargs.get("fl2va_model")
        timeline_head_context = kwargs.get("timeline_head_context")
        timeline_head_audio = kwargs.get("timeline_head_audio")
        timeline_tail_context = kwargs.get("timeline_tail_context")
        timeline_tail_audio = kwargs.get("timeline_tail_audio")
        timeline_first_keyframe = kwargs.get("timeline_first_keyframe")
        timeline_last_keyframe = kwargs.get("timeline_last_keyframe")
        roughcut_exact_duration = bool(kwargs.get("roughcut_exact_duration", False))
        first_memory_profile = str(kwargs.get(
            "一采显存策略", FIRST_MEMORY_AUTO))
        first_memory = first_pass_memory_policy(first_memory_profile)
        pass1_checkpoint_mode = str(kwargs.get(
            "一采断点模式", PASS1_CHECKPOINT_OFF))
        pass1_video = kwargs.get("一采成片")
        pass1_audio = kwargs.get("一采成片音频")
        if isinstance(timeline_head_audio, dict):
            waveform = timeline_head_audio.get("waveform")
            if waveform is not None and int(waveform.shape[-1]) == 0:
                timeline_head_audio = None
        if isinstance(timeline_tail_audio, dict):
            waveform = timeline_tail_audio.get("waveform")
            if waveform is not None and int(waveform.shape[-1]) == 0:
                timeline_tail_audio = None
        if (first_frame is not None or last_frame is not None) and fl2va_model is None:
            raise ValueError("首尾帧约束已启用，但没有可用的 FL2VA 模型")
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
        # A resumed manual storyboard can begin on an explicit cut.  In that
        # case the previous render may still be wired on the Director, but the
        # plan is authoritative: do not inject its tail into an independent
        # shot.  Action-transfer and seamless-card resumes set this flag true.
        if (int(plan.get("resume_start_segment") or 1) > 1 and
                not bool(plan.get("resume_context_required", True))):
            context_video = None
            context_audio = None
        resuming = context_video is not None
        progress_owner = str(plan.get("progress_owner") or "")
        segment_entries = list(plan.get("segments") or [])
        count = int(plan.get("segment_count") or len(segment_entries) or 1)
        default_seconds = float(plan.get("segment_seconds_snapped") or 5.0)
        default_frames = int(
            plan.get("frames_per_segment") or
            core.length_for(default_seconds, 24.0))
        fps = float(plan.get("fps") or 24.0)
        overlap = int(context_length)

        # `transition` belongs to the incoming segment and controls the
        # boundary immediately before it.  Legacy plans had no field, so keep
        # their historical seamless-continuation behaviour by default.
        segment_transitions = []
        for offset in range(count):
            entry = segment_entries[offset] if offset < len(segment_entries) else {}
            if offset == 0:
                segment_transitions.append("开场")
            else:
                segment_transitions.append(
                    "切镜" if str(entry.get("transition") or "").strip() == "切镜"
                    else "承接")

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
        continuing = task == TASK_CONTINUE
        transferring = task == TASK_TRANSFER
        # Action-transfer clips are deliberately passed through the direct
        # ``ref_video`` socket (the per-shot media bundle only contains stills
        # and audio).  Carry the policy saved by H3DirectorActionSource across
        # every condition rebuild, including the independent second-pass
        # condition.  Old plans have no field; their Director UI already
        # presents the action clip as "自动", so that remains the safe default.
        action_reference_weight_mode = "off"
        action_reference_weight = 1.0
        if transferring:
            action_reference_weight_mode = str(
                plan.get("action_reference_weight_mode") or "auto").strip().lower()
            if action_reference_weight_mode not in {"auto", "manual"}:
                action_reference_weight_mode = "auto"
            try:
                action_reference_weight = max(0.25, min(
                    3.0, float(plan.get("action_reference_weight") or 1.0)))
            except (TypeError, ValueError):
                action_reference_weight = 1.0
            logger.info(
                "H3-Myang: 动作参考视频权重策略=%s%s",
                "自动" if action_reference_weight_mode == "auto" else "手动",
                "（%.2f）" % action_reference_weight
                if action_reference_weight_mode == "manual" else "")

        def _condition_media_inputs(inputs):
            """Add the action clip policy without mutating shared graph inputs."""
            result = dict(inputs or {})
            if transferring and result.get("ref_video") is not None:
                result["ref_video_weight_mode"] = action_reference_weight_mode
                result["ref_video_weight"] = action_reference_weight
            return result

        if transferring and first_memory_profile != FIRST_MEMORY_STANDARD:
            # Action transfer already carries a dense reference-video stream.
            # Even a one-token latent preview adds a CUDA->CPU sync and image
            # encode inside every sampler callback, which can stall cards at
            # the memory limit. Auto/VRAM-first therefore publish step numbers
            # only; users can explicitly choose the compatibility profile when
            # they accept the memory cost of a clear per-step preview.
            first_memory = dict(first_memory)
            first_memory["preview_interval"] = 0
            first_memory["preview_mode"] = "latent_rgb"
            logger.info(
                "H3-Myang: 动作迁移的%s档关闭一采中途图像预览；"
                "仅保留采样步数与段落完成清晰帧", first_memory_profile)
        elif transferring:
            logger.warning(
                "H3-Myang: 动作迁移已手动选择兼容模式，将执行每步清晰VAE预览；"
                "16GB显卡可能显著变慢或触发换页")
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
                    "H3-Myang: 当前 Turbo LoRA 是 FL2VA/T2VA 档，『%s』不会"
                    "自动变成 Ref2VA；若同时连接二采 Ref2VA 基模，队列会解析两套"
                    "H3 权重。动作迁移请换 LightX2V Ref2VA Turbo，既避免模型"
                    "家族不匹配，也减少准备阶段的模型驻留压力。", task)
            elif task_family == "ref2va" and task == TASK_FRESH and media is None:
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
        reuse_condition = True
        detail_memory_profile = detail.DETAIL_MEMORY_AUTO
        detail_reserve_vram_gb = 1.25
        detail_preview_interval = 2
        continuous_sigma_requested = False
        detail_vsr_enhance = False
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
            reuse_condition = bool(
                detail_settings.get("reuse_condition", True))
            detail_memory_profile = str(detail_settings.get(
                "memory_profile", detail.DETAIL_MEMORY_AUTO))
            continuous_sigma_requested = bool(
                detail_settings.get("continuous_sigma", False))
            detail_vsr_enhance = bool(
                detail_settings.get("vsr_enhance", False))
            detail_reserve_vram_gb, detail_preview_interval = (
                detail.detail_memory_policy(
                    detail_memory_profile,
                    detail_settings.get("reserve_vram_gb", 1.25),
                    detail_settings.get("preview_interval", 2)))
        refining = str(second_pass) != detail.DETAIL_OFF
        sampling_second_pass = (
            refining and detail_mode != detail.DETAIL_MODE_UPSCALE_ONLY)
        continuous_sigma = bool(
            continuous_sigma_requested and sampling_second_pass)
        reusing_pass1 = pass1_checkpoint_mode in (
            PASS1_CHECKPOINT_REUSE, PASS1_VIDEO_REUSE)
        if reusing_pass1 and not refining:
            raise ValueError(
                "已选择跳过一采，但二采没有开启。请连接/开启二采设置，或把"
                "『一采断点模式』改回关闭")
        if pass1_checkpoint_mode == PASS1_VIDEO_REUSE:
            if pass1_video is None:
                raise ValueError(
                    "已选择『使用接入的一采成片，直接二采』，但没有连接『一采成片』")
            if count != 1:
                raise ValueError(
                    "一采成片兼容入口只支持单段，当前计划有%d段。多段请先用"
                    "『保存一采检查点』，再选择『读取检查点，直接二采』" % count)
        elif pass1_video is not None:
            # Silently ignoring a connected input reads as "the feature does not
            # work": the run just does a normal first pass and the video never
            # participates. Fail loudly and name the switch instead.
            raise ValueError(
                "已连接『一采成片』，但『一采断点模式』当前是『%s』，接入的视频不会被"
                "使用。请把『一采断点模式』改成『%s』（导演台『一采显存与预览』面板里），"
                "或断开『一采成片』" % (pass1_checkpoint_mode, PASS1_VIDEO_REUSE))
        if continuous_sigma and pass1_checkpoint_mode != PASS1_CHECKPOINT_OFF:
            raise ValueError(
                "连续 Sigma 二采必须从同一条噪声轨迹完成一采与二采，暂不兼容"
                "一采检查点、恢复进度或接入成片直入二采")
        if continuous_sigma and abs(float(denoise) - 1.0) > 1e-6:
            raise ValueError(
                "连续 Sigma 二采要求一采重绘幅度为 1.0；否则没有完整的起始 Sigma 轨迹")
        if continuous_sigma and second_passes != 1:
            raise ValueError("连续 Sigma 二采目前只支持 1 轮二采")
        continuous_latent_upscale = any(
            token in str(second_upscale_method).lower()
            for token in ("latent", "neural_3d"))
        if (continuous_sigma
                and detail_mode != detail.DETAIL_MODE_REFINE
                and not continuous_latent_upscale):
            raise ValueError(
                "连续 Sigma 二采只能使用 neural_3d 放大，或选择同分辨率二采；"
                "像素/VSR 需要先解码干净画面，会中断连续噪声轨迹")
        if (continuous_sigma
                and detail_refinement_params(detail_refinement) is not None):
            raise ValueError(
                "连续 Sigma 二采不能同时开启一采『低Sigma精修』；两者都会改写同一条 Sigma 轨迹")
        if sampling_second_pass and not continuous_sigma and refine_model is None:
            raise ValueError(
                "已开启二采放大，但『二采模型』没接。请接 LoRA 之前的 Ref2VA 基模")
        if (sampling_second_pass and not continuous_sigma
                and turbo_metadata(refine_model) is not None):
            raise ValueError(
                "『二采模型』必须接 Turbo LoRA 之前的基模，不能接 Turbo 输出")

        audio_settings = kwargs.get("音频设置")
        if audio_settings is None:
            audio_settings = kwargs.get("audio_settings")
        if audio_settings is not None and not isinstance(audio_settings, dict):
            raise ValueError("音频设置输入不是导演台『音频精修与接缝』的输出")
        audio_settings = audio_settings or {}
        audio_refine_enabled = bool(
            audio_settings.get("refine_enabled", False))
        audio_refine_steps = max(1, int(audio_settings.get("steps", 4)))
        audio_refine_denoise = float(audio_settings.get("denoise", 0.5))
        audio_refine_sampler = str(
            audio_settings.get("sampler_name", "euler"))
        audio_refine_scheduler = str(
            audio_settings.get("scheduler", "simple"))
        audio_seam_enabled = bool(audio_settings.get("seam_enabled", False))
        audio_seam_ms = max(0.0, float(audio_settings.get("seam_ms", 80.0)))
        audio_refine_model = audio_settings.get("model")
        if audio_refine_model is None and turbo is None:
            # A non-Turbo first pass is already the undistilled base model.
            audio_refine_model = model
        if audio_refine_enabled and audio_refine_model is None:
            raise ValueError(
                "已开启音频精修，但当前一采使用 Turbo。请把 Turbo LoRA 之前的"
                "基模接到导演台『二采 Ref2VA 基模』；音频精修会复用这一路基模")
        if (audio_refine_enabled and
                turbo_metadata(audio_refine_model) is not None):
            raise ValueError(
                "音频精修模型必须是 Turbo LoRA 之前的基模，不能接 Turbo 输出")

        enhancement_settings = kwargs.get("增强设置")
        if enhancement_settings is None:
            enhancement_settings = kwargs.get("enhancement_settings")
        if enhancement_settings is not None and not isinstance(enhancement_settings, dict):
            raise ValueError("增强设置输入不是导演台『画质与分镜增强』的输出")
        enhancement_settings = enhancement_settings or {}
        face_settings = enhancement_settings.get("face") or {}
        motion_settings = enhancement_settings.get("motion") or {}
        face_enabled = bool(face_settings.get("enabled", False))
        motion_enabled = bool(motion_settings.get("enabled", False))
        if continuous_sigma and audio_refine_enabled:
            raise ValueError(
                "连续 Sigma 二采暂不兼容音频精修；请关闭音频精修或关闭连续 Sigma")
        if continuous_sigma and (face_enabled or motion_enabled):
            raise ValueError(
                "连续 Sigma 二采暂不兼容小脸精修或动作修复；这些步骤需要先完成干净的一采成片")
        enhancement_model = enhancement_settings.get("model") or model
        if face_enabled or motion_enabled:
            try:
                import nodes as comfy_nodes
                registered = comfy_nodes.NODE_CLASS_MAPPINGS
            except Exception:
                registered = {}
            required = set()
            if face_enabled:
                required.update({
                    "H3FaceTrackCrop", "H3InjectVideoLatent",
                    "H3PerFrameDenoise", "H3FaceStitch",
                })
            if motion_enabled:
                required.update({
                    "H3JerkOracle", "H3TimeSmear", "H3V2VInit",
                    "H3InjectSchedule", "H3ExactRecover", "H3AudioRecover",
                })
            missing = sorted(required.difference(registered))
            if missing:
                package = "ComfyUI-H3-FaceRefine" if face_enabled else "ComfyUI-MAINodes"
                raise ValueError(
                    "导演台增强已开启，但 %s 没有完整加载；缺少节点：%s" %
                    (package, "、".join(missing)))
            if face_enabled:
                validator = getattr(
                    registered.get("H3FaceTrackCrop"),
                    "validate_detector", None)
                if callable(validator):
                    try:
                        validator(str(face_settings.get(
                            "detector", "bbox\\face_yolov8m.pt")))
                    except Exception as error:
                        raise ValueError(
                            "小脸精修检测器预检失败，已在视频采样前停止：%s" %
                            error) from error

        loaded_model = str(
            getattr(h3, "names", {}).get("model", "")).lower()
        if transferring and loaded_model and "ref2va" not in loaded_model:
            raise ValueError(
                "动作迁移需要 Ref2VA 模型；当前 H3Loader 加载的是 %s" %
                loaded_model)

        if (continuing or resuming
                or any(value == "承接" for value in segment_transitions[1:])):
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
        previous_detail_context = None
        # Each inter-segment memory barrier returns this lightweight token.
        # Feeding it into the next prepare node makes cleanup part of the real
        # execution path instead of a side branch that the scheduler may defer
        # until final collection.
        segment_h3 = h3

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
                    "motion_repair": motion_enabled,
                    "face_refine": face_enabled,
                    "audio_refine": audio_refine_enabled,
                    "audio_seam": audio_seam_enabled,
                    "correcting": False,
                    "save_segments": bool(save_segments),
                    "segment_prefix": str(segment_prefix),
                })
        except Exception:
            pass

        for index in range(1, count + 1):
            frames = segment_frames[index - 1]
            # A rough-cut I-point MotionContext is outside the requested
            # selection.  Generate an extra hidden head window so trimming the
            # copied context does not shorten the visible I/O duration.  Normal
            # inter-segment continuation already includes its overlap in
            # ``frames`` and therefore must not receive this extension.
            hidden_head = bool(
                index == 1 and roughcut_exact_duration
                and (timeline_head_context is not None or continuing or resuming)
                and first_frame is None and timeline_first_keyframe is None)
            tail_motion_active = index == count and timeline_tail_context is not None
            hidden_layout = _roughcut_hidden_layout(
                frames, overlap, hidden_head=hidden_head,
                hidden_tail=tail_motion_active)
            head_motion_extension = hidden_layout["head_trim_frames"]
            condition_frames = hidden_layout["condition_frames"]
            seconds = condition_frames / 24.0
            batch_index = ref_starts[index - 1]
            seg_entry = (segment_entries[index - 1]
                         if index <= len(segment_entries) else {})
            transition = segment_transitions[index - 1]
            continue_from_previous = index > 1 and transition == "承接"
            # An explicit rough-cut first-frame constraint is authoritative.
            # Do not also inject a continuation tail at frame zero: two
            # independent anchors at the same boundary fight each other.
            use_external_first_context = (
                index == 1
                and (timeline_head_context is not None or continuing or resuming)
                and first_frame is None and timeline_first_keyframe is None)
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

            boundary_inputs = dict(media_inputs)
            if index == 1 and first_frame is not None:
                boundary_inputs["first_frame"] = first_frame
            if index == count and last_frame is not None:
                boundary_inputs["last_frame"] = last_frame
            boundary_segment = "first_frame" in boundary_inputs or "last_frame" in boundary_inputs
            segment_model = fl2va_model if boundary_segment else model
            preparation_items = ["提示词"]
            if shot_assets:
                preparation_items.append("%d项镜头素材" % len(shot_assets))
            if clip is not None:
                preparation_items.append("动作/参考视频")
            if continue_from_previous or use_external_first_context:
                preparation_items.append("MotionContext承接锚点")
            if boundary_segment:
                preparation_items.append("首尾帧约束")
            prepared_h3 = graph.node(
                "H3PrepareSignal", h3=segment_h3, segment_index=label_index,
                total_segments=count, run_id=run_id, owner_id=progress_owner,
                activity="第%d/%d段准备 · 整理%s并进行条件编码" % (
                    label_index, count, "、".join(preparation_items)))
            condition_h3 = prepared_h3.out(0)
            if first_memory["cleanup"]:
                # The Director's MODEL input is resolved before this dynamic
                # graph executes and may still occupy VRAM. Evict it *before*
                # reference-video VAE encoding; the post-condition barrier
                # below then removes CLIP/VAE before the DiT is loaded again.
                condition_h3 = graph.node(
                    "H3PreConditionMemoryBarrier",
                    h3=condition_h3, loaded_model=segment_model,
                    stage="第%d段 一采模型 -> 条件编码" % label_index).out(0)
            condition = graph.node(
                "H3Condition", h3=condition_h3, prompt=segment_prompt,
                resolution=resolution, aspect_ratio=aspect_ratio,
                width=width, height=height, seconds=seconds,
                ref_image_size=ref_image_size,
                **_condition_media_inputs(boundary_inputs))

            video_vae, audio_vae = h3.video_vae, h3.audio_vae
            base_positive, base_latent = condition.out(0), condition.out(1)
            if index == 1 and timeline_first_keyframe is not None:
                first_keyframe = graph.node(
                    "H3AnchorKeyframe", conditioning=base_positive,
                    vae=video_vae, latent=base_latent,
                    image=timeline_first_keyframe, frame_index=0)
                base_positive = first_keyframe.out(0)
            if index == count and timeline_last_keyframe is not None:
                last_keyframe = graph.node(
                    "H3AnchorKeyframe", conditioning=base_positive,
                    vae=video_vae, latent=base_latent,
                    image=timeline_last_keyframe,
                    # The keyframe belongs to the visible O point.  Any H3-grid
                    # alignment frames after it are hidden and cropped below.
                    frame_index=frames + head_motion_extension - 1)
                base_positive = last_keyframe.out(0)
            if not continue_from_previous and not use_external_first_context:
                positive, anchor = base_positive, None
            else:
                context = {}
                if use_external_first_context:
                    # 续写 / 断点续跑首段：钉到已有视频的尾部
                    # （motion-context 的 context_frames 路径）。
                    if timeline_head_context is not None:
                        continuation_video = timeline_head_context
                        continuation_audio = timeline_head_audio
                        tail_start = max(
                            0, int(timeline_head_context.shape[0]) - overlap)
                    elif shot_action is not None:
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
                    "H3AnchorContext", conditioning=base_positive,
                    vae=video_vae, latent=base_latent,
                    context_length=str(overlap),
                    audio_vae=audio_vae, **context)
                positive = anchor.out(0)

            first_latent = (anchor.out(2) if anchor is not None
                            else base_latent)
            # H3's 17k+5 alignment may allocate a few extra frames after the
            # visible selection when the hidden I-point window is present.
            # Crop those even when no O-point MotionContext was requested.
            tail_trim_frames = hidden_layout["tail_trim_frames"]
            if tail_motion_active:
                tail_inputs = {
                    "conditioning": positive,
                    "vae": video_vae,
                    "latent": first_latent,
                    "context_frames": timeline_tail_context,
                    "context_length": str(overlap),
                    # Future context begins immediately after the visible
                    # selection, which is shifted by the hidden I-point window
                    # when both sides use MotionContext in a single segment.
                    "anchor_start_frame": hidden_layout["tail_start_frame"],
                    "audio_vae": audio_vae,
                }
                if timeline_tail_audio is not None:
                    tail_inputs["context_audio"] = timeline_tail_audio
                tail_anchor = graph.node("H3TailAnchorContext", **tail_inputs)
                positive = tail_anchor.out(0)
                tail_trim_frames = tail_anchor.out(1)
                first_latent = tail_anchor.out(2)

            checkpoint_exists = (
                pass1_checkpoint_mode == PASS1_CHECKPOINT_RESUME
                and os.path.isfile(_pass1_checkpoint_path(
                    segment_prefix, label_index)))
            continuous_partial_sample = None
            continuous_tail_sigmas = None
            continuous_conditioning = None
            if (pass1_checkpoint_mode == PASS1_CHECKPOINT_REUSE
                    or checkpoint_exists):
                segment_sample = graph.node(
                    "H3Pass1CheckpointLoad",
                    filename_prefix=segment_prefix,
                    segment_index=label_index,
                    expected_frames=condition_frames)
            elif pass1_checkpoint_mode == PASS1_VIDEO_REUSE:
                if condition_frames != frames:
                    raise ValueError(
                        "一采成片直接二采暂不支持粗剪首尾隐藏上下文；请关闭粗剪"
                        "MotionContext，或改用一采检查点")
                video_inputs = {
                    "images": pass1_video,
                    "base_latent": base_latent,
                    "video_vae": video_vae,
                    "audio_vae": audio_vae,
                    "expected_frames": frames,
                }
                if pass1_audio is not None:
                    video_inputs["audio"] = pass1_audio
                segment_sample = graph.node(
                    "H3Pass1VideoEncode", **video_inputs)
            else:
                # H3Condition may leave the 32B text encoder and video VAE
                # resident. The barrier runs only when a real first pass follows.
                if first_memory["cleanup"]:
                    positive = graph.node(
                        "H3ConditionMemoryBarrier", conditioning=positive,
                        keep_model=segment_model,
                        stage="第%d段 条件编码 -> 一采" % label_index).out(0)

                guider = graph.node(
                    "BasicGuider", model=segment_model, conditioning=positive)
                sigmas = graph.node(
                    "BasicScheduler", model=segment_model, scheduler=scheduler,
                    steps=(int(steps) + int(second_steps)
                           if continuous_sigma else steps),
                    denoise=denoise)
                if continuous_sigma:
                    split_sigmas = graph.node(
                        "SplitSigmas", sigmas=sigmas.out(0), step=int(steps))
                    sigma_link = split_sigmas.out(0)
                    continuous_tail_sigmas = split_sigmas.out(1)
                    continuous_conditioning = positive
                else:
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
                    pass_label="sample1",
                    reserve_vram_gb=float(first_memory["reserve_vram_gb"]),
                    preview_interval=int(first_memory["preview_interval"]),
                    preview_mode=str(first_memory["preview_mode"]),
                    return_denoised=continuous_sigma)
                if continuous_sigma:
                    continuous_partial_sample = sample
                    segment_sample = graph.node(
                        "H3LatentIdentity", samples=sample.out(1))
                else:
                    segment_sample = sample
                if audio_refine_enabled:
                    # The base model re-denoises only the packed audio stream.
                    segment_sample = graph.node(
                        "H3AudioRefineSampler", model=audio_refine_model,
                        positive=positive, negative=positive,
                        latent=sample.out(0),
                        seed=noise_seed + 300000 + index - 1,
                        steps=audio_refine_steps, cfg=1.0,
                        sampler_name=audio_refine_sampler,
                        scheduler=audio_refine_scheduler,
                        audio_denoise=audio_refine_denoise,
                        video_denoise=0.0,
                        run_id=run_id, owner_id=progress_owner,
                        segment_index=index, total_segments=count)
                if pass1_checkpoint_mode in (
                        PASS1_CHECKPOINT_SAVE, PASS1_CHECKPOINT_RESUME):
                    segment_sample = graph.node(
                        "H3Pass1CheckpointSave",
                        samples=segment_sample.out(0),
                        filename_prefix=segment_prefix,
                        segment_index=label_index,
                        frames=condition_frames)
            decoded_video = graph.node(
                "VAEDecode", samples=segment_sample.out(0), vae=video_vae)
            decoded_audio = graph.node(
                "VAEDecodeAudio", samples=segment_sample.out(0), vae=audio_vae)

            # 采样完成：把这一帧预览和「第 N 段采样完成」推给前端。signal 透传，
            # 不改画面，只保证它一定在 VAEDecode 之后、下游之前执行。
            sampled_sig = graph.node(
                "H3ProgressSignal", images=decoded_video.out(0),
                audio=decoded_audio.out(0), segment_index=index,
                total_segments=count, stage="sampled", run_id=run_id,
                owner_id=progress_owner,
                prompt=segment_prompt, brief=seg_brief,
                # This is the compact progress-card frame, not ComfyUI's
                # canvas-level node.imgs preview. Keep it for every source,
                # including an externally supplied pass-1 movie, so the user
                # can see the actual input while pass 2 prepares. The frontend
                # independently suppresses node.imgs before canvas painting.
                save_preview=True)

            # Drift is fitted on the original one-pass resolution.  When the
            # detail pass is enabled that low-resolution stream remains the
            # continuation state; feeding a 928P latent into the next 720P
            # segment would make the anchor rows spatially incompatible.
            joined = sampled_sig.out(0)
            joined_audio = sampled_sig.out(1)
            motion_context_sample = None
            face_context_sample = None

            if motion_enabled:
                motion_start = graph.node(
                    "H3ProgressSignal", images=joined, audio=joined_audio,
                    segment_index=index, total_segments=count,
                    stage="motion_start", run_id=run_id,
                    owner_id=progress_owner, prompt=segment_prompt,
                    brief=seg_brief, save_preview=False)
                motion_oracle = graph.node(
                    "H3JerkOracle", samples=segment_sample.out(0), length=frames,
                    q=0.75, d_max=4, ramp=True,
                    preset=str(motion_settings.get(
                        "preset", "balanced (default)")),
                    bridge=8, protect_tail=min(17, frames))
                smeared = graph.node(
                    "H3TimeSmear", images=motion_start.out(0), dilation=4,
                    hold_map=motion_oracle.out(0), expand_to_end=True)
                smeared_latent = graph.node(
                    "VAEEncode", pixels=smeared.out(0), vae=video_vae)
                motion_init = graph.node(
                    "H3V2VInit", samples=smeared_latent.out(0),
                    length=smeared.out(2))
                motion_seconds = graph.node(
                    "H3FramesToSeconds", frames=smeared.out(2), fps=24.0)
                motion_condition = graph.node(
                    "H3Condition", h3=condition_h3, prompt=segment_prompt,
                    resolution=resolution, aspect_ratio=aspect_ratio,
                    width=width, height=height,
                    seconds=motion_seconds.out(0),
                    ref_image_size=ref_image_size,
                    **_condition_media_inputs(media_inputs))
                motion_guider = graph.node(
                    "BasicGuider", model=enhancement_model,
                    conditioning=motion_condition.out(0))
                motion_sigmas = graph.node(
                    "H3InjectSchedule", model=enhancement_model,
                    scheduler="simple",
                    total_steps=int(motion_settings.get("steps", 6)),
                    inject=float(motion_settings.get("inject", 0.70)),
                    preset="custom")
                motion_noise = graph.node(
                    "RandomNoise", noise_seed=noise_seed + 100000 + index - 1)
                motion_sample = graph.node(
                    "H3SamplerAdvanced", noise=motion_noise.out(0),
                    guider=motion_guider.out(0), sampler=sampler,
                    sigmas=motion_sigmas.out(0),
                    latent_image=motion_init.out(0), vae=video_vae,
                    run_id=run_id, owner_id=progress_owner,
                    segment_index=index, total_segments=count,
                    pass_label="motion")
                motion_video = graph.node(
                    "VAEDecode", samples=motion_sample.out(0), vae=video_vae)
                motion_audio = graph.node(
                    "VAEDecodeAudio", samples=motion_sample.out(0), vae=audio_vae)
                recovered_video = graph.node(
                    "H3ExactRecover", images=motion_video.out(0),
                    hold_map=smeared.out(1))
                recovered_audio = graph.node(
                    "H3AudioRecover", audio=motion_audio.out(0),
                    hold_map=smeared.out(1), fps=24,
                    reference=motion_start.out(1), reference_mix=1.0,
                    audio_source="keep the original performance (safe default)")
                motion_done = graph.node(
                    "H3ProgressSignal", images=recovered_video.out(0),
                    audio=recovered_audio.out(0), segment_index=index,
                    total_segments=count, stage="motion_refined",
                    run_id=run_id, owner_id=progress_owner,
                    prompt=segment_prompt, brief=seg_brief,
                    save_preview=True)
                joined, joined_audio = motion_done.out(0), motion_done.out(1)
                # Enhanced mode deliberately pays one VAE round trip so the
                # repaired final motion, not the unrepaired first pass, anchors
                # the beginning of the next segment.
                recovered_latent = graph.node(
                    "VAEEncode", pixels=joined, vae=video_vae)
                motion_context_sample = graph.node(
                    "H3V2VInit", samples=recovered_latent.out(0), length=frames)

            if face_enabled:
                face_start = graph.node(
                    "H3ProgressSignal", images=joined, audio=joined_audio,
                    segment_index=index, total_segments=count,
                    stage="face_start", run_id=run_id,
                    owner_id=progress_owner, prompt=segment_prompt,
                    brief=seg_brief, save_preview=False)
                track_inputs = {
                    "images": face_start.out(0),
                    "detector": str(face_settings.get(
                        "detector", "bbox\\face_yolov8m.pt")),
                    "confidence": 0.35,
                    "crop_factor": float(face_settings.get(
                        "crop_factor", 2.5)),
                    "canvas_width": 512,
                    "canvas_height": 512,
                    "canvas_mode": "auto_capped_768",
                    "smooth_window": 21,
                    "size_smooth_window": 51,
                    "smooth_method": "gaussian",
                    "size_mode": "per_frame",
                    "identity_track": True,
                    "identity_threshold": 0.28,
                    "select": "largest",
                    "fallback_detector": "none",
                    "fallback_head_frac": 0.5,
                }
                identity_ordinal = int(
                    face_settings.get("identity_ordinal", 0) or 0)
                identity_media = media_inputs.get("media")
                if identity_ordinal > 0:
                    if identity_media is None:
                        raise ValueError(
                            "小脸精修指定了身份图 @图片%d，但本镜头没有可用素材包" %
                            identity_ordinal)
                    identity_image = graph.node(
                        "H3DirectorMediaImage", media=identity_media,
                        image_ordinal=identity_ordinal)
                    track_inputs["identity_reference"] = identity_image.out(0)
                face_track = graph.node("H3FaceTrackCrop", **track_inputs)

                # The tracked crop fixes the target canvas dynamically. Seed
                # that joint AV latent with the real crop and denoise only the
                # video rows; the already generated segment audio is carried
                # through unchanged outside this visual repair pass.
                face_condition_inputs = {}
                if "media" in media_inputs:
                    face_condition_inputs["media"] = media_inputs["media"]
                face_condition = graph.node(
                    "H3Condition", h3=condition_h3, prompt=segment_prompt,
                    resolution="自定义", aspect_ratio="1:1",
                    width=face_track.out(4), height=face_track.out(5),
                    seconds=seconds, ref_image_size=ref_image_size,
                    **face_condition_inputs)
                face_seed = graph.node(
                    "H3InjectVideoLatent", av_latent=face_condition.out(1),
                    images=face_track.out(0), vae=video_vae)
                face_denoise = graph.node(
                    "H3PerFrameDenoise", av_latent=face_seed.out(0),
                    transform=face_track.out(1),
                    strength_small_face=1.0,
                    strength_large_face=0.35,
                    scale_mode="absolute_px",
                    face_px_small=30.0, face_px_large=120.0,
                    gamma=1.0, smooth_frames=9)
                face_guider = graph.node(
                    "BasicGuider", model=enhancement_model,
                    conditioning=face_condition.out(0))
                face_sigmas = graph.node(
                    "BasicScheduler", model=enhancement_model,
                    scheduler="simple",
                    steps=max(1, int(face_settings.get("steps", 4))),
                    denoise=float(face_settings.get("denoise", 0.45)))
                face_noise = graph.node(
                    "RandomNoise", noise_seed=noise_seed + 200000 + index - 1)
                face_sample = graph.node(
                    "H3SamplerAdvanced", noise=face_noise.out(0),
                    guider=face_guider.out(0), sampler=sampler,
                    sigmas=face_sigmas.out(0),
                    latent_image=face_denoise.out(0), vae=video_vae,
                    run_id=run_id, owner_id=progress_owner,
                    segment_index=index, total_segments=count,
                    pass_label="face")
                face_video = graph.node(
                    "VAEDecode", samples=face_sample.out(0), vae=video_vae)
                face_stitch = graph.node(
                    "H3FaceStitch", base_images=face_start.out(0),
                    refined_crops=face_video.out(0),
                    transform=face_track.out(1),
                    paste_region="face_only", mask_dilation=16,
                    feather=6, colour_match=1.0, blend=1.0,
                    undetected_frames="fade_out")
                face_done = graph.node(
                    "H3ProgressSignal", images=face_stitch.out(0),
                    audio=face_start.out(1), segment_index=index,
                    total_segments=count, stage="face_refined",
                    run_id=run_id, owner_id=progress_owner,
                    prompt=segment_prompt, brief=seg_brief,
                    save_preview=True)
                joined, joined_audio = face_done.out(0), face_done.out(1)
                face_latent = graph.node(
                    "VAEEncode", pixels=joined, vae=video_vae)
                face_context_sample = graph.node(
                    "H3V2VInit", samples=face_latent.out(0), length=frames)

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

                use_latent = any(
                    token in str(second_upscale_method).lower()
                    for token in ("latent", "neural_3d"))
                # 仅放大 + 像素/VSR：一采已经解码成像素了，直接在像素上放大出图。
                # 走 latent 通路会是 decode→放大→encode→decode，多一次完整的
                # VAE 往返，把 RTX VSR 刚算出来的锐利边缘又抹回去。
                pixel_only = (detail_mode == detail.DETAIL_MODE_UPSCALE_ONLY
                              and not use_latent)
                if pixel_only:
                    refined_out = graph.node(
                        "H3PixelUpscale", images=joined,
                        resolution=second_pass, aspect_ratio=aspect_ratio,
                        width=second_width, height=second_height,
                        upscale_method=second_upscale_method,
                        chunk_frames=second_chunk_frames).out(0)
                else:
                    # 同分辨率精修不做放大；另外两种模式只在第一轮放大一次。
                    detail_latent = (
                        continuous_partial_sample.out(0)
                        if continuous_sigma else
                        face_context_sample.out(0)
                        if face_context_sample is not None else
                        motion_context_sample.out(0)
                        if motion_context_sample is not None else
                        segment_sample.out(0))
                    if detail_mode != detail.DETAIL_MODE_REFINE:
                        latent_kwargs = dict(
                            samples=detail_latent,
                            resolution=second_pass, aspect_ratio=aspect_ratio,
                            width=second_width, height=second_height,
                            upscale_method=second_upscale_method,
                            chunk_frames=second_chunk_frames,
                            latent_upscale_model=latent_upscale_model,
                            latent_precision=latent_precision,
                            latent_chunk_steps=latent_chunk_steps,
                            reserve_vram_gb=detail_reserve_vram_gb)
                        if not use_latent:
                            latent_kwargs["vae"] = video_vae
                        upscaled_latent = graph.node(
                            "H3LatentUpscale", **latent_kwargs)
                        detail_latent = upscaled_latent.out(0)

                    if sampling_second_pass:
                        if continuous_sigma:
                            # The low-Sigma continuation must use the exact
                            # same conditioning (including any MotionContext
                            # anchor) as the high-Sigma half. Re-encoding or
                            # substituting the base condition would no longer
                            # be one continuous diffusion trajectory.
                            positive_2nd = continuous_conditioning
                        elif reuse_condition:
                            # 复用一采那份条件，锚点之前的原始版本。H3 的
                            # conditioning 只带提示词 token 和 minimax_refs，目标
                            # 画布尺寸完全来自 latent，所以放大后的 latent 配一采
                            # 条件是合法的。
                            #
                            # 按二采分辨率重建反而有害：ref_image_size 默认「匹配
                            # 生成分辨率」，画布变大后参考图被重采样到更大面积，
                            # minimax_refs 的 latent_h/latent_w 和 token 数全变，
                            # 于是低降噪的几步被拉向一个不同的解又收不过去，出来
                            # 就是涂抹加轻微身份漂移。顺带省掉每段一次文本编码和
                            # 全部参考图的 VAE 编码。
                            #
                            # 必须取 condition.out(0) 而不是 positive：后者可能已
                            # 经过 H3AnchorContext，而锚点是 append 的，再锚一次会
                            # 把一采的低分辨率锚点块和二采的高分辨率块混在一起。
                            positive_2nd = condition.out(0)
                        else:
                            condition_resolution = (
                                resolution
                                if detail_mode == detail.DETAIL_MODE_REFINE
                                else second_pass)
                            condition_width = (
                                width
                                if detail_mode == detail.DETAIL_MODE_REFINE
                                else second_width)
                            condition_height = (
                                height
                                if detail_mode == detail.DETAIL_MODE_REFINE
                                else second_height)
                            condition_2nd = graph.node(
                                "H3Condition", h3=condition_h3,
                                prompt=segment_prompt,
                                resolution=condition_resolution,
                                aspect_ratio=aspect_ratio,
                                width=condition_width, height=condition_height,
                                seconds=seconds,
                                ref_image_size=ref_image_size,
                                **_condition_media_inputs(media_inputs))
                            positive_2nd = condition_2nd.out(0)
                        if (not continuous_sigma and continue_from_previous
                                and previous_detail_context is not None):
                            detail_anchor = graph.node(
                                "H3LatentOverlapSeed",
                                # 保留当前段由一采结果放大/投影得到的二采底图，只把
                                # 上一段二采尾部写入开头零噪声区。二采不再把同一批
                                # 高分辨率块重复注册成条件关键帧，避免第二段开始增加
                                # packed attention token 并推高激活显存。这里若使用
                                # condition_2nd.out(1)，会把当前段底图替换成空条件
                                # latent，从第二段开始即使低降噪也会采出乱码。
                                latent=detail_latent,
                                context_length=str(overlap),
                                context_latent=previous_detail_context)
                            detail_latent = detail_anchor.out(1)
                        # Pixel/VAE projection and neural 3D upscale can leave
                        # several GB resident after producing the high-res
                        # latent. A conditioning-only barrier is insufficient
                        # because graph scheduling may run the upscaler after
                        # it. Carry both dependencies through one side-effect
                        # node so cleanup is guaranteed to be the last action
                        # before the pass-2 DiT sampler loads its weights.
                        refine_memory = graph.node(
                            "H3RefineMemoryBarrier",
                            conditioning=positive_2nd,
                            latent=detail_latent,
                            keep_model=(segment_model if continuous_sigma
                                        else refine_model),
                            stage="第%d段 一采/VAE/放大器 -> 二采" % label_index)
                        positive_2nd = refine_memory.out(0)
                        detail_latent = refine_memory.out(1)
                        guider_2nd = graph.node(
                            "BasicGuider",
                            model=(segment_model if continuous_sigma
                                   else refine_model),
                            conditioning=positive_2nd)
                        if continuous_sigma:
                            sigma_2nd_link = continuous_tail_sigmas
                            sampler_2nd_link = sampler
                        else:
                            sigmas_2nd = graph.node(
                                "BasicScheduler", model=refine_model,
                                scheduler=second_scheduler,
                                steps=second_steps, denoise=second_denoise)
                            sampler_2nd = graph.node(
                                "KSamplerSelect", sampler_name=second_sampler)
                            sigma_2nd_link = sigmas_2nd.out(0)
                            sampler_2nd_link = sampler_2nd.out(0)
                        for pass_index in range(
                                1 if continuous_sigma else second_passes):
                            pass_seed = noise_seed + index - 1
                            if second_seed_mode == detail.DETAIL_SEED_OFFSET:
                                pass_seed += pass_index
                            noise_2nd = (graph.node("DisableNoise")
                                         if continuous_sigma else graph.node(
                                             "RandomNoise",
                                             noise_seed=pass_seed))
                            sample_2nd = graph.node(
                                "H3SamplerAdvanced", noise=noise_2nd.out(0),
                                guider=guider_2nd.out(0),
                                sampler=sampler_2nd_link,
                                sigmas=sigma_2nd_link,
                                latent_image=detail_latent,
                                vae=video_vae, run_id=run_id,
                                owner_id=progress_owner,
                                segment_index=index, total_segments=count,
                                reserve_vram_gb=detail_reserve_vram_gb,
                                preview_interval=detail_preview_interval,
                                preview_mode="vae",
                                pass_label="sample2" if pass_index == 0
                                else "sample2.%d" % (pass_index + 1))
                            detail_latent = sample_2nd.out(0)

                    decoded_refine_video = graph.node(
                        "VAEDecode", samples=detail_latent, vae=video_vae)
                    refined_out = decoded_refine_video.out(0)

                if (detail_vsr_enhance
                        and detail_mode == detail.DETAIL_MODE_REFINE):
                    # Only 同分辨率二采 has no resize step, so this is the one
                    # mode where VSR has to appear as a separate 1:1 pass. The
                    # other two modes reach VSR through 放大方式, and the panel
                    # hides this checkbox there -- but a saved workflow can
                    # still carry a stale ``true``, so gate on the mode rather
                    # than on "does this mode sample", or those frames go
                    # through VSR twice.
                    #
                    # Placed before the trim so the anchor window is enhanced
                    # with the body and then cut, keeping both sides of a seam
                    # processed alike; nothing downstream re-encodes through the
                    # VAE to smear the edges back out.
                    refined_out = graph.node(
                        "H3VsrEnhance", images=refined_out,
                        chunk_frames=second_chunk_frames).out(0)

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
                    trim_frames=trim_frames, tail_trim_frames=tail_trim_frames,
                    fps=fps)
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
                audio=output_audio, trim_frames=trim_frames,
                tail_trim_frames=tail_trim_frames, fps=fps)

            trimmed_audio = trim.out(1)
            if (audio_seam_enabled and previous_output_pixels is not None
                    and anchor is not None):
                # The new segment still contains its pinned, duplicated head at
                # this point.  Use that otherwise-discarded audio to rewrite the
                # previous tail, then return the normally trimmed current track.
                audio_seam = graph.node(
                    "H3AudioSeam",
                    previous_audio=previous_output_pixels[1],
                    next_audio=output_audio, next_images=output_joined,
                    trim_frames=trim_frames, fade_ms=audio_seam_ms, fps=fps)
                previous_key = "audios_%d" % (index - 1)
                if previous_key in audios:
                    audios[previous_key] = audio_seam.out(0)
                trimmed_audio = audio_seam.out(1)

            # 分段完成：推最终预览帧（二采后/漂移后）和「第 N 段完成」。
            done_sig = graph.node(
                "H3ProgressSignal", images=trim.out(0),
                audio=trimmed_audio, segment_index=index,
                total_segments=count, stage="done", run_id=run_id,
                owner_id=progress_owner,
                prompt=segment_prompt, brief=seg_brief, save_preview=True)
            segment_images, segment_audio = done_sig.out(0), done_sig.out(1)
            current_pass1_sample = (face_context_sample or motion_context_sample
                                    or segment_sample)
            next_pass1_sample = current_pass1_sample

            if refining and index < count:
                # Do not carry Ref2VA/high-resolution VAE/upscaler residency
                # into the next segment's text/reference conditioning. Clear
                # both H3 models' dynamic GPU pages while keeping their RAM
                # staging buffers, so neither pass restages from disk and no
                # previous-pass pages overlap the next segment's activations.
                barrier_inputs = {
                    "images": segment_images,
                    "audio": segment_audio,
                    "h3": condition_h3,
                    "keep_model": segment_model,
                    "pass1_latent": current_pass1_sample.out(0),
                    "stage": "第%d段 二采/高分辨率解码 -> 第%d段 一采" %
                             (label_index, label_index + 1),
                }
                if sampling_second_pass:
                    barrier_inputs.update({
                        "detail_latent": detail_latent,
                        "context_length": str(overlap),
                    })
                segment_barrier = graph.node(
                    "H3SegmentMemoryBarrier", **barrier_inputs)
                segment_images = segment_barrier.out(0)
                segment_audio = segment_barrier.out(1)
                segment_h3 = segment_barrier.out(2)
                previous_detail_context = (
                    segment_barrier.out(3) if sampling_second_pass else None)
                # Keep the existing ``previous_sample.out(0)`` contract while
                # making it point at the compact CPU tail returned by the
                # segment barrier, not the full first-pass result.
                next_pass1_sample = graph.node(
                    "H3LatentIdentity", samples=segment_barrier.out(4))

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
            previous_sample = next_pass1_sample
            previous_context_pixels = (segment_images, segment_audio)
            previous_output_pixels = (segment_images, segment_audio)
        collector = graph.node(
            "H3SegmentCollector", active_count=count,
            run_id=run_id, owner_id=progress_owner,
            total_segments=count, **images, **audios)
        final_images, final_audio = collector.out(0), collector.out(1)
        if first_memory["cleanup"]:
            released = graph.node(
                "H3OutputMemoryRelease", images=final_images,
                audio=final_audio, stage="长视频输出完成")
            final_images, final_audio = released.out(0), released.out(1)
        frame_summary = (str(segment_frames[0]) if len(set(segment_frames)) == 1
                         else "/".join(str(value) for value in segment_frames))
        logger.info(
            "H3-Myang: 原生锚点展开 %d段 × %s帧 | context=%d%s%s%s%s | 一采显存=%s",
            count, frame_summary, overlap,
            "（实验速度锚点）" if overlap == 5 else "",
            " | 二采=" + str(second_pass) if refining else "",
            " | 切镜=%d（不使用前段 MotionContext）" %
            segment_transitions[1:].count("切镜")
            if "切镜" in segment_transitions[1:] else "",
            " | 断点续跑：从第%d段起，参考视频第%d帧" %
            (int(plan.get("resume_start_segment") or 1), ref_starts[0])
            if resuming else "", first_memory_profile)
        return {"expand": graph.finalize(),
                "result": (final_images, final_audio)}



NODE_CLASS_MAPPINGS = {
    "H3ScriptSplitter": H3ScriptSplitter,
    "H3SegmentPrompt": H3SegmentPrompt,
    "H3SegmentCollector": H3SegmentCollector,
    "H3ModelFromBundle": H3ModelFromBundle,
    "H3Pass1CheckpointSave": H3Pass1CheckpointSave,
    "H3Pass1CheckpointLoad": H3Pass1CheckpointLoad,
    "H3Pass1VideoEncode": H3Pass1VideoEncode,
    "H3PreConditionMemoryBarrier": H3PreConditionMemoryBarrier,
    "H3ConditionMemoryBarrier": H3ConditionMemoryBarrier,
    "H3RefineMemoryBarrier": H3RefineMemoryBarrier,
    "H3OutputMemoryRelease": H3OutputMemoryRelease,
    "H3SegmentMemoryBarrier": H3SegmentMemoryBarrier,
    "H3FramesToSeconds": H3FramesToSeconds,
    "H3LatentIdentity": H3LatentIdentity,
    "H3LongVideo": H3LongVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3ScriptSplitter": "沐阳 H3 · 分段计划",
    "H3SegmentPrompt": "沐阳 H3 · 分段提示词",
    "H3SegmentCollector": "沐阳 H3 · 分段合成",
    "H3ModelFromBundle": "沐阳 H3 · 取模型（挂补丁用）",
    "H3Pass1CheckpointSave": "沐阳 H3 · 保存一采检查点（内部）",
    "H3Pass1CheckpointLoad": "沐阳 H3 · 读取一采检查点（内部）",
    "H3Pass1VideoEncode": "沐阳 H3 · 一采成片转 Latent（内部）",
    "H3PreConditionMemoryBarrier": "沐阳 H3 · 条件前显存屏障（内部）",
    "H3ConditionMemoryBarrier": "沐阳 H3 · 条件显存屏障（内部）",
    "H3RefineMemoryBarrier": "沐阳 H3 · 二采显存屏障（内部）",
    "H3OutputMemoryRelease": "沐阳 H3 · 输出显存释放（内部）",
    "H3SegmentMemoryBarrier": "沐阳 H3 · 段间显存屏障（内部）",
    "H3FramesToSeconds": "沐阳 H3 · 帧数转秒（内部）",
    "H3LatentIdentity": "沐阳 H3 · Latent 透传（内部）",
    "H3LongVideo": "沐阳 H3 · 长视频（原生多关键帧）",
}
