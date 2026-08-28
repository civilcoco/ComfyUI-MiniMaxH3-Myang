"""Long-script director for MiniMax H3 — ComfyUI-MiniMaxH3-Myang.

SPDX-License-Identifier: GPL-3.0-only

Copyright (C) 2026 Myang

Split a full script once, then expand one short brief per segment. The point is
token economy: a six-segment minute costs one pass over the whole script plus
six small calls, instead of six passes over the whole script.

This module contains the public planning, segment, and native long-video
nodes. Public node IDs and serialized widget positions remain stable.

"""

import hashlib
import json
import logging
import math
import re
import time
from pathlib import Path

import torch

from . import core, detail
from .anchor_compat import ensure_anchors
from .core import REF_IMAGE_SIZES as CORE_REF_SIZES

logger = logging.getLogger(__name__)

CUSTOM_NODES_DIR = Path(__file__).resolve().parent.parent

# Reject values outside H3's supported duration window instead of allowing a
# silent clamp that would make the planned and generated lengths disagree.
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

# Canonical backend values used by the local nodes. The Chinese strings shown
# by the frontend map back to these before submission.
RESOLUTIONS = ["360P", "416P", "480P", "540P", "640P", "720P",
               "768P", "832P", "928P", "1024P", "1080P", "custom"]
ASPECTS = ["1:1", "2:3", "3:2", "3:4", "4:3", "9:16", "16:9", "21:9"]
CONTEXT_LENGTHS = ["22", "5", "39", "56"]
SCHEDULERS = ["simple", "normal", "karras", "exponential", "sgm_uniform",
              "ddim_uniform", "beta", "linear_quadratic", "kl_optimal"]

# These two carry the Chinese label as the value so the workflow remains
# readable even when frontend localization has not loaded. Mapping back happens
# once, in `_canon`, right before the value reaches the sampler.
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
    """Snap a duration to the MiniMax H3 model's 17k+5 frame grid."""
    target = max(5.0, float(seconds) * float(fps))
    blocks = max(0, round((target - 5) / 17))
    return blocks * 17 + 5


def plan_segments(total_seconds: float, segment_seconds: float, overlap_frames: int,
                  fps: float, max_segments: int) -> dict:
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
    frames = frame_length(segment_seconds, fps)
    if overlap_frames >= frames:
        raise ValueError(f"overlap_frames({overlap_frames}) 必须小于每段帧数({frames})")

    first = frames / fps
    tail = (frames - overlap_frames) / fps
    if total_seconds <= first:
        count = 1
    else:
        # Round, not ceil. Each join costs `overlap_frames`, so a nominal
        # "16s at 8s each" only reaches 15.08s with two segments -- and ceil
        # would add a whole third segment, overshooting by 6.2s to cover a
        # 0.9s shortfall. Landing on the nearer of the two is what someone
        # asking for 16 seconds actually wants.
        count = 1 + max(1, round((float(total_seconds) - first) / tail))
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
        # Do not split between '<' and '>' in H3 reference markup.
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


def _local_timeline_split(text, count, media_manifest="", reason=""):
    """Deterministic fail-safe: distinct chronological prompts, no model call."""
    style_header, body = _extract_global_preamble(text)
    units = _ensure_unit_count(_script_units(body), count, body)
    chunks = _balanced_script_chunks(units, count)
    character_refs = _persistent_character_picture_tags(text, media_manifest)
    segments = []
    for index, chunk in enumerate(chunks, start=1):
        transition = "开场" if index == 1 else (
            "切镜" if _HARD_CUT_CUE.search(chunk) else "承接")
        pieces = []
        if style_header:
            pieces.append(style_header)
        missing_refs = [tag for tag in character_refs if tag not in chunk]
        if missing_refs:
            pieces.append("人物外观参考：%s。" % "、".join(missing_refs))
        if index > 1 and transition == "承接":
            pieces.append("画面与动作自然承接上一段结尾。")
        pieces.append(chunk)
        prompt = "\n\n".join(piece for piece in pieces if piece).strip()
        segments.append({
            "index": index,
            "transition": transition,
            "brief": re.sub(r"\s+", " ", chunk)[:120],
            "prompt": prompt,
        })
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
    character_refs = _persistent_character_picture_tags(text, media_manifest)
    lines = [
        "【程序已预聚合的分段内容包｜不要重新切分、合并或移动剧情】",
        "下面每段的“源剧情”已经按原始顺序完整分配。你只负责把各段分别展开成技能规定的完整 H3 prompt。",
        "每段都必须独立交代构图与主体位置、人物外观/姿态/视线、环境与光线、动作准备到结果、"
        "表情反应、运镜目标/幅度/速度、同步声音/台词和实际生效的素材标签。",
    ]
    if style_header:
        lines.append("全局设定（每段继承）：" + style_header)
    if character_refs:
        lines.append("贯穿人物外观参考（人物出现的每段必须保留）：" + "、".join(character_refs))
    for index, chunk in enumerate(chunks, 1):
        transition = "开场" if index == 1 else ("切镜" if _HARD_CUT_CUE.search(chunk) else "承接")
        continuity = ("建立完整开场状态，不依赖前文。" if index == 1 else
                      "直接建立新场景、机位和人物状态，不写黑场/闪白。" if transition == "切镜" else
                      "从上一段结尾的机位、姿态、环境和动作惯性自然接续。")
        refs = "、".join(character_refs) if character_refs else "按本段源剧情与公共素材清单匹配"
        lines.extend(["", f"=== 第 {index}/{int(count)} 段｜{transition}｜目标 {float(seconds):.2f} 秒 ===",
                      "源剧情（全部保留，不得概括掉动作或台词）：", chunk,
                      "连续性职责：" + continuity, "人物参考职责：" + refs,
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


DIRECTOR_STORYBOARD_PLANNER_SYSTEM = """你是长视频分镜规划师，只规划，不写最终视频提示词，也不执行写作技能。
用户已经给出固定数量的有序源剧情段。必须一一对应，不能新增、删除、合并或移动段落。
不要输出 JSON。每段严格使用下面这种易读文本块，SEGMENT 编号必须连续：
[SEGMENT 1]
TRANSITION: 开场
GOAL: 本段叙事目标
SHOT: 0.00-3.00 || 构图与主体位置 || 可见动作过程 || 运镜 || 声音/台词 || @图片1、@视频1

每段可写 1-4 行 SHOT；没有素材时最后一栏写“无”。媒体标签只能从用户提供的清单中选择。"""


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
            "transition": transition_match.group(1) if transition_match else "",
            "segment_goal": goal_match.group(1).strip() if goal_match else body,
            "shots": shots[:4],
        })
    return result


def _plan_director_storyboard(
    chunks, count, seconds, media_manifest, llm_service,
    ollama_auto_unload, seed,
):
    """Plan fixed-count segment storyboards once; fall back locally on mismatch."""
    source = [
        "固定段落数：%d；每段约 %.2f 秒。以下源剧情已经分好，段数和顺序不可改变。" % (
            int(count), float(seconds))]
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
        notes.append("分镜规划调用失败，采用固定段落本地规划：%s" % str(error)[:160])
        planned = []
    if not isinstance(planned, list) or len(planned) != int(count) or not all(
            isinstance(item, dict) for item in planned):
        returned = len(planned) if isinstance(planned, list) else 0
        notes.append(
            "分镜规划返回 %d 段而目标为 %d 段，采用固定段落本地规划；不重试" % (
                returned, int(count)))
        planned = []
        for index, chunk in enumerate(chunks, 1):
            planned.append({
                "index": index,
                "transition": "开场" if index == 1 else (
                    "切镜" if _HARD_CUT_CUE.search(chunk) else "承接"),
                "segment_goal": re.sub(r"\s+", " ", chunk).strip(),
                "shots": [],
            })
        return planned, "local_fixed_count", notes
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
            "transition": transition,
            "segment_goal": str(item.get("segment_goal") or chunk).strip(),
            "shots": [shot for shot in shots if isinstance(shot, dict)][:4],
        })
    return normalized, "llm_fixed_count", notes


def _storyboard_text(item):
    lines = ["本段分镜规划：", "叙事目标：" + str(item.get("segment_goal") or "")]
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
    character_refs = _persistent_character_picture_tags(text, media_manifest)
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
        parts = [
            "这是长片第 %d/%d 段，只写这一段约 %.2f 秒的最终视频提示词。"
            "不要输出 JSON，不要生成其他段，不要改变段数。" % (
                index, len(chunks), float(seconds)),
        ]
        if style_header:
            parts.append("全片共同设定：" + style_header)
        parts.extend([
            "本段源剧情（动作和台词全部保留）：\n" + chunk,
            "段间关系：" + transition + "；" + continuity,
            _storyboard_text(plan_item),
        ])
        if character_refs:
            parts.append(
                "人物出现时保留这些外观参考标签：" + "、".join(character_refs))
        parts.append(
            "请像 Media Agent 一样直接输出这一段的最终提示词正文，"
            "根据写作技能展开构图、主体、环境、动作、运镜和声音。")
        briefs.append({
            "index": index,
            "transition": transition,
            "brief": re.sub(r"\s+", " ", chunk).strip()[:120],
            "writer_input": "\n\n".join(parts),
            "fallback_prompt": "\n\n".join(
                part for part in (
                    style_header,
                    "人物外观参考：%s。" % "、".join(character_refs)
                    if character_refs else "",
                    chunk,
                ) if part),
        })
    return style_header, briefs


def _write_segments_with_media_agent(
    text, chunks, count, seconds, media_manifest, skill_rules,
    llm_service, ollama_auto_unload, seed,
):
    """Write N plain prompts with the package's existing Media Agent method."""
    from . import agent_nodes

    manifest = _agent_manifest_entries(media_manifest)
    storyboard, storyboard_source, plan_notes = _plan_director_storyboard(
        chunks, count, seconds, media_manifest, llm_service,
        ollama_auto_unload, seed)
    style_header, briefs = _segment_writer_briefs(
        text, chunks, media_manifest, seconds, storyboard=storyboard)
    segments = []
    notes = list(plan_notes)
    fallback_segments = []
    writer_disabled = any(
        str(note).startswith("分镜规划调用失败") for note in plan_notes)
    for item in briefs:
        user_prompt, system_prompt = agent_nodes.build_media_agent_writer_request(
            item["writer_input"], manifest, skill_rules, seconds, expand=False)
        generated = ""
        # One remote writer call per segment. A provider response that already
        # ended with empty content/length will not improve by replaying the
        # same large Skill request, and that duplicate used to add another
        # several-minute socket window. The segment has a deterministic local
        # prompt below, so fail fast instead.
        for attempt in (range(1) if not writer_disabled else range(0)):
            label = "第%d段#%d" % (item["index"], attempt + 1)
            try:
                raw = call_llm(
                    llm_service, user_prompt, system_prompt,
                    ollama_auto_unload,
                    int(seed) + item["index"] * 131 + attempt * 1009,
                    max_tokens=None)
            except Exception as error:  # noqa: BLE001 - classified locally
                if type(error).__name__ == "InterruptProcessingException":
                    raise
                notes.append("%s 调用失败：%s" % (label, str(error)[:160]))
                if _is_rate_limit_error(error) or _is_timeout_error(error):
                    writer_disabled = True
                    break
                continue
            generated = agent_nodes._sanitize_llm_output(raw)
            if generated:
                break
            notes.append("%s 返回空正文" % label)
        if not generated:
            generated = item["fallback_prompt"]
            fallback_segments.append(item["index"])
            notes.append("第%d段 Media Agent 写作不可用，保留固定分镜源剧情" % item["index"])
            writer_disabled = True
        segments.append({
            "index": item["index"],
            "transition": item["transition"],
            "brief": item["brief"],
            "prompt": generated,
        })
    payload = {
        "style_header": style_header,
        "segments": segments,
        "split_source": (
            "media_agent_writer_with_local_fallback"
            if fallback_segments else "media_agent_writer"),
        "writer_mode": "one_plain_prompt_per_segment_v1",
        "storyboard_source": storyboard_source,
        "storyboard": storyboard,
        "writer_fallback_segments": fallback_segments,
    }
    return payload, notes


def _usable_split_payload(payload, expected_count):
    segments = payload.get("segments") if isinstance(payload, dict) else None
    if not isinstance(segments, list) or len(segments) != int(expected_count):
        return False
    return all(isinstance(segment, dict) and (
        str(segment.get("prompt") or "").strip()
        or str(segment.get("brief") or "").strip()) for segment in segments)


def _is_timeout_error(error):
    message = str(error or "").lower()
    return isinstance(error, TimeoutError) or "timed out" in message or "timeout" in message


def _is_rate_limit_error(error):
    message = str(error or "").lower()
    return "429" in message or "rate limit" in message or "tpm" in message


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
    if getattr(media, "assets", None):
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


SPLIT_PROMPT_VERSION = 10


SPLIT_SYSTEM = """你是一个专业的 AI 视频分镜切片与 H3 提示词生成器（Shot / Segment Slicer & Prompt Author）。
用户会给你一段完整的视频剧本或提示词（通常由上游 Agent 生成，包含分镜设定、人物动作、镜头运镜、@图片1/@视频1等素材引用、以及<d>台词</d>或#台词）。{media_section}{skill_section}

你的任务是：将这段完整提示词/剧本，严格按时间线精准切分成恰好 {count} 个连续的视频分段，每段对应约 {seconds:.2f} 秒的生成片段。

【H3 提示词规范要求】：
1. 忠实切片，不要重新编写或杜撰新的剧情故事，只做时间轴上的逻辑切分与镜头分配。
2. 保持素材与台词完整（严格遵循本包 H3 引用语法）：
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
4. 每段 prompt 是一个完整可独立渲染的 MiniMax H3 提示词（包含视觉风格、人物外观、本段动作、镜头运动、素材标签、音效与台词）。
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

def _native_frame_length(seconds, fps):
    return core.length_for(seconds, 24.0)


# plan_segments looks this global up at call time.  Point it at the same
# official helper used by H3Condition so planning and latent allocation cannot
# disagree on the 17k+5 frame grid.
frame_length = _native_frame_length

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
                    "tooltip": "完整提示词/剧本：可接入 Agent 节点的 myang_prompt，"
                               "或手动粘贴剧本。切片节点会用 LLM 将其按时间轴切分成各段提示词。"
                               "留空则只算时间与帧数，不调用 LLM。"}),
                "total_seconds": ("FLOAT", {"default": 60.0, "min": 1.0, "max": 3600.0, "step": 1.0}),
                "length_source": (LENGTH_SOURCES, {
                    "default": LENGTH_MANUAL,
                    "tooltip": "匹配参考视频时长：接上 ref_video 后按它的帧数算总时长，"
                               "上面填的数就不用管了"}),
                "segment_seconds": ("FLOAT", {"default": 10.0, "min": MIN_SECONDS, "max": MAX_SECONDS, "step": 0.5}),
                "overlap_frames": ("INT", {"default": 22, "min": 0, "max": 240,
                                           "tooltip": "必须与 MiniMaxH3MotionContext 的 context_length 一致"}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 1.0}),
                "llm_service": (llm_service_options(),),
                "max_segments": ("INT", {"default": MAX_SLOTS, "min": 1, "max": MAX_SLOTS}),
                "ollama_auto_unload": ("BOOLEAN", {"default": True}),
                "use_cache": ("BOOLEAN", {"default": True,
                                          "tooltip": "剧本和段数没变就复用上次的拆分结果，不再调用 LLM"}),
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
                assets = getattr(media, "assets", None) or ()
                for asset in assets:
                    if getattr(asset, "kind", "") == "video":
                        val = getattr(asset, "payload", None)
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
        plan = plan_segments(total_seconds, segment_seconds, overlap_frames, fps, max_segments)
        count = plan["segment_count"]
        text = str(script or "").strip()
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
        # 技能决定每段提示词的写法，所以要在拆分之前解析出来，并进缓存键。
        skill_rules, skill_source = resolve_skill(
            kwargs.get("skill_preset", SKILL_PRESET_NONE),
            kwargs.get("skill_text", ""), llm_service=llm_service,
            ollama_auto_unload=bool(ollama_auto_unload), routing_prompt=text)
        if skill_source:
            logger.info("H3-Myang: 分段写作技能 %s", skill_source)
        seconds_each = plan["segment_seconds_snapped"]
        _source_packets, source_chunks = _build_split_source_packets(
            text, count, media_manifest, seconds_each)
        cache_file = _cache_dir() / (
            "%s.json" % _cache_key(
                SPLIT_PROMPT_VERSION, text, count, llm_service, seed,
                media_manifest, skill_rules))
        payload = None
        if use_cache and cache_file.is_file():
            try:
                candidate = json.loads(cache_file.read_text(encoding="utf-8"))
                if _usable_split_payload(candidate, count):
                    payload = candidate
                    logger.info("H3-Myang: 命中拆分缓存 %s", cache_file.name)
            except Exception:
                payload = None

        if payload is None:
            payload, writer_notes = _write_segments_with_media_agent(
                text, source_chunks, count, seconds_each,
                media_manifest, skill_rules, llm_service,
                ollama_auto_unload, seed)
            if not _usable_split_payload(payload, count):
                reason = "；".join(writer_notes[-6:]) or "Media Agent 未返回完整分段正文"
                payload = _local_timeline_split(
                    text, count, media_manifest=media_manifest, reason=reason)
                payload["storyboard_source"] = "local_fixed_count"
                payload["writer_fallback_segments"] = list(range(1, count + 1))
                logger.warning("H3-Myang: Media Agent 写作不可用，采用本地固定段落兜底 | %s", reason)
            try:
                cache_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception as exc:
                logger.warning("H3-Myang: 缓存写入失败: %s", exc)

        # Final defensive gate: no cache or future caller may reintroduce the
        # old "copy the last segment until N" behavior.
        if not _usable_split_payload(payload, count):
            payload = _local_timeline_split(
                text, count, media_manifest=media_manifest,
                reason="缓存或响应的段数/内容无效")
        style_header = str(payload.get("style_header") or "").strip()
        segments = [seg for seg in (payload.get("segments") or [])
                    if isinstance(seg, dict)]
        for i, seg in enumerate(segments, start=1):
            seg["index"] = i
            seg["brief"] = str(seg.get("brief") or "").strip()
            transition = str(seg.get("transition") or "").strip()
            seg["transition"] = ("开场" if i == 1
                                 else transition if transition in ("承接", "切镜")
                                 else "承接")
            prompt_val = str(seg.get("prompt") or "").strip()
            if not prompt_val:
                prompt_val = "\n\n".join(p for p in (style_header, seg["brief"]) if p)
            seg["prompt"] = prompt_val

        plan.update({"style_header": style_header,
                     "full_prompt": text,
                     "segments": segments})
        plan["split_source"] = str(payload.get("split_source") or "llm")
        if payload.get("writer_mode"):
            plan["writer_mode"] = str(payload["writer_mode"])
        if payload.get("storyboard_source"):
            plan["storyboard_source"] = str(payload["storyboard_source"])
        if isinstance(payload.get("storyboard"), list):
            plan["storyboard"] = payload["storyboard"]
        if payload.get("split_fallback_reason"):
            plan["split_fallback_reason"] = str(payload["split_fallback_reason"])
        if media_manifest:
            plan["media_manifest"] = media_manifest
        if skill_source:
            plan["skill_source"] = skill_source

        preview = [
            f"共 {count} 段 × {plan['segment_seconds_snapped']:.3f}s"
            f"（每段 {plan['frames_per_segment']} 帧，段间重叠 {overlap_frames} 帧）",
            f"成片约 {plan['total_seconds_actual']:.2f}s（目标 {total_seconds:.1f}s）",
            f"参考视频至少要 {plan['ref_frames_needed']} 帧"
            f"（约 {plan['ref_frames_needed'] / max(fps, 1.0):.1f}s @ {fps:.0f}fps）",
            "",
            (f"[分镜规划] {payload.get('storyboard_source')}"
             if payload.get("storyboard_source") else "[分镜规划] 未单独启用"),
            ("[写作兜底] Media Agent 未返回的段落已用本地时间线补齐"
             if payload.get("writer_fallback_segments") else ""),
            f"[写作技能] {skill_source}" if skill_source else "[写作技能] 默认写法",
            f"[全局设定] {style_header[:120]}…" if style_header else "[全局设定] 无",
            "",
        ]
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
            preview.append(f"[第{idx}段·{s.get('transition', '承接')}] {disp[:90]}…")

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

        return (json.dumps(plan, ensure_ascii=False),
                count,
                plan["segment_seconds_snapped"],
                plan["frames_per_segment"],
                "\n".join(preview),
                plan["ref_frames_needed"])

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
            from .progress import broadcast_progress
            broadcast_progress({
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
            broadcast_progress({
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
        # Backward compatibility for bundles saved before the dual lazy loader.
        if hasattr(h3, "model"):
            return (h3.model,)
        raise ValueError("H3-Myang: 输入不是可识别的 H3 模型包")

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
                # already carries, and the asset_*/asset_manifest_json transport
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
            noise = graph.node(
                "RandomNoise", noise_seed=noise_seed + index - 1)
            sample = graph.node(
                "H3SamplerAdvanced", noise=noise.out(0),
                guider=guider.out(0), sampler=sampler,
                sigmas=sigmas.out(0), latent_image=first_latent,
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

NODE_CLASS_MAPPINGS = {
    "H3ScriptSplitter": H3ScriptSplitter,
    "H3SegmentPrompt": H3SegmentPrompt,
    "H3SegmentCollector": H3SegmentCollector,
    "H3ModelFromBundle": H3ModelFromBundle,
    "H3LongVideo": H3LongVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3ScriptSplitter": "沐阳 H3 · 分段计划",
    "H3SegmentPrompt": "沐阳 H3 · 分段提示词",
    "H3SegmentCollector": "沐阳 H3 · 分段合成",
    "H3ModelFromBundle": "沐阳 H3 · 取模型（挂补丁用）",
    "H3LongVideo": "沐阳 H3 · 长视频（原生多关键帧）",
}
