"""Dialogue audit for the MiniMax H3 agent.

SPDX-License-Identifier: GPL-3.0-only

A prompt can be perfectly written and still be unusable: if the dialogue takes
longer to say than the shot it sits in, the generated video either talks over
its own cut or trails into silence. This module checks two things that plain
prose review keeps missing -- that the dialogue is in the language the user
actually asked for, and that it can physically be spoken in the time available.

Speaking rates are expressed in 字/s (characters per second), which is the
natural unit for Chinese: Han characters are monosyllabic, so one character is
one spoken beat. For alphabetic scripts the equivalent beat is the syllable, so
those are counted by vowel groups rather than by letters or words -- comparing
English word counts against a per-character budget would flag every line.
"""

from __future__ import annotations

import re
from typing import Any

DIALOGUE_RE = re.compile(r"<d>(.*?)</d>", re.IGNORECASE | re.DOTALL)

# (min, max) 字/s. Below min the line drags; above max it cannot be articulated.
SPEECH_RATES: dict[str, tuple[float, float]] = {
    "excited": (3.5, 6.0),
    "calm": (2.5, 4.0),
    "casual": (1.5, 3.0),
}
DEFAULT_TONE = "calm"

# Tone is inferred from the surrounding prose, so the cues are the words a
# writer actually uses around a line rather than a formal emotion taxonomy.
TONE_CUES: dict[str, tuple[str, ...]] = {
    "excited": (
        "兴奋", "激动", "急促", "大喊", "喊道", "尖叫", "怒吼", "咆哮", "惊呼", "急切",
        "慌张", "愤怒", "狂喜", "呐喊", "嘶吼", "抢白", "厉声",
        "excited", "shout", "yell", "scream", "urgent", "frantic", "angry",
    ),
    "casual": (
        "撒娇", "慵懒", "轻声", "低语", "呢喃", "喃喃", "叹息", "犹豫", "迟疑", "温柔",
        "悠闲", "休闲", "困倦", "懒洋洋", "娇嗔", "小声", "耳语",
        "casual", "whisper", "murmur", "sigh", "lazy", "gentle", "hesitant",
    ),
    "calm": (
        "平静", "冷静", "沉稳", "淡淡", "平淡", "认真", "严肃", "陈述", "解释",
        "calm", "steady", "flat", "explain", "state",
    ),
}

_HAN_RE = re.compile(r"[一-鿿㐀-䶿]")
_KANA_RE = re.compile(r"[぀-ゟ゠-ヿ]")
_HANGUL_RE = re.compile(r"[가-힯ᄀ-ᇿ]")
_LATIN_RE = re.compile(r"[A-Za-z]")
_CYRILLIC_RE = re.compile(r"[Ѐ-ӿ]")

SCRIPT_NAMES = {
    "han": "中文",
    "japanese": "日语",
    "korean": "韩语",
    "latin": "英文/拉丁语系",
    "cyrillic": "西里尔语系",
    "unknown": "未知",
}


def detect_script(text: str) -> str:
    """Identify the dominant writing system of a fragment.

    Japanese is checked before Han on purpose: Japanese mixes kanji with kana,
    so any kana at all means the line is Japanese rather than Chinese.

    Han is weighed against Latin *words*, not Latin letters. A Chinese line
    containing an English proper noun ("PLAYER 1 准备好了吗") has more letters
    than characters, and counting letters would misfile it as English.
    """
    value = str(text or "")
    if not value.strip():
        return "unknown"
    if _KANA_RE.search(value):
        return "japanese"
    if _HANGUL_RE.search(value):
        return "korean"
    counts = {
        "han": len(_HAN_RE.findall(value)),
        "latin": len(re.findall(r"[A-Za-z]+", value)),
        "cyrillic": len(re.findall(r"[Ѐ-ӿ]+", value)),
    }
    best = max(counts, key=counts.get)
    return best if counts[best] > 0 else "unknown"


def speech_units(text: str, script: str | None = None) -> int:
    """Count spoken beats: Han/kana/hangul characters, or Latin syllables.

    Punctuation and whitespace carry no duration and are excluded.
    """
    value = str(text or "")
    script = script or detect_script(value)
    if script in {"han", "japanese", "korean"}:
        # Count CJK characters plus any embedded alphanumerics, which are
        # normally read out (e.g. "PLAYER 1" inside a Chinese line).
        cjk = len(_HAN_RE.findall(value)) + len(_KANA_RE.findall(value)) + len(_HANGUL_RE.findall(value))
        latin_syllables = _latin_syllables(value)
        return cjk + latin_syllables
    if script in {"latin", "cyrillic"}:
        return _latin_syllables(value) if script == "latin" else len(re.findall(r"[Ѐ-ӿ]", value))
    return len(re.sub(r"\s+", "", value))


def _latin_syllables(text: str) -> int:
    """Approximate English syllable count by counting vowel groups.

    Deliberately simple: a silent trailing "e" is dropped and every word is
    worth at least one syllable. This is an estimate for pacing, not a
    pronunciation dictionary.
    """
    total = 0
    for word in re.findall(r"[A-Za-z]+", text):
        lowered = word.lower()
        groups = len(re.findall(r"[aeiouy]+", lowered))
        if lowered.endswith("e") and groups > 1 and not lowered.endswith(("le", "ee", "ye")):
            groups -= 1
        total += max(1, groups)
    total += len(re.findall(r"\d", text))
    return total


def infer_tone(context: str) -> str:
    """Pick a speaking rate bracket from the prose around a line.

    Scans the narration next to the dialogue rather than the dialogue itself:
    "他兴奋地喊道" is what tells you the delivery, not the words being said.
    """
    lowered = str(context or "").lower()
    best_tone, best_hit = DEFAULT_TONE, -1
    for tone in ("excited", "casual", "calm"):
        for cue in TONE_CUES[tone]:
            index = lowered.find(cue.lower())
            if index >= 0 and index > best_hit:
                best_tone, best_hit = tone, index
    return best_tone


def dialogue_blocks(text: str) -> list[dict[str, Any]]:
    """Every <d>…</d> line with the narration that surrounds it."""
    value = str(text or "")
    blocks: list[dict[str, Any]] = []
    for index, match in enumerate(DIALOGUE_RE.finditer(value), 1):
        inner = match.group(1).strip()
        if not inner:
            continue
        # Keep the context inside this shot: a newline or a shot marker ends it.
        # Without that boundary a calm line picks up "兴奋" from the next shot
        # and gets billed at the wrong speaking rate.
        before = value[max(0, match.start() - 160): match.start()]
        before = re.split(r"\n|(?:\[\s*)?\bShot\s*\d+", before)[-1]
        after = value[match.end(): match.end() + 60]
        after = re.split(r"\n|(?:\[\s*)?\bShot\s*\d+", after)[0]
        context = before + " " + after
        script = detect_script(inner)
        tone = infer_tone(context)
        units = speech_units(inner, script)
        low, high = SPEECH_RATES[tone]
        blocks.append({
            "index": index,
            "text": inner,
            "script": script,
            "script_name": SCRIPT_NAMES.get(script, script),
            "tone": tone,
            "units": units,
            "rate_min": low,
            "rate_max": high,
            # Fastest delivery is the shortest time the line can occupy.
            "min_seconds": round(units / high, 2) if high else 0.0,
            "max_seconds": round(units / low, 2) if low else 0.0,
        })
    return blocks


def audit(
    prompt: str,
    seconds: float,
    expected_script: str | None = None,
    plan: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Check dialogue language and speakability against the timeline.

    When a shot plan is available each line is billed against the shot it falls
    in, which is stricter and more useful than billing everything against the
    whole clip: four lines can each fit in 10 seconds while being impossible
    inside their own 2.5-second shots.
    """
    total = max(1.0, float(seconds))
    blocks = dialogue_blocks(prompt)
    issues: list[dict[str, Any]] = []

    if not blocks:
        return {"blocks": [], "issues": [], "spoken_seconds": 0.0, "target_seconds": total, "ok": True}

    windows = _shot_windows(prompt, plan, total, len(blocks))

    for block in blocks:
        window = windows.get(block["index"], total)
        block["window_seconds"] = round(window, 2)

        if expected_script and expected_script != "unknown" and block["script"] not in {"unknown", expected_script}:
            issues.append({
                "kind": "language",
                "index": block["index"],
                "text": block["text"],
                "detail": (
                    f"台词 {block['index']} 是{block['script_name']}，"
                    f"但用户要求的是{SCRIPT_NAMES.get(expected_script, expected_script)}"
                ),
            })

        if block["min_seconds"] > window + 0.05:
            issues.append({
                "kind": "too_long",
                "index": block["index"],
                "text": block["text"],
                "detail": (
                    f"台词 {block['index']}（{block['tone']}，{block['units']}字）最快也要 "
                    f"{block['min_seconds']:.2f} 秒，超过可用的 {window:.2f} 秒。"
                    f"建议压缩到 {max(1, int(window * block['rate_max']))} 字以内"
                ),
                "suggest_units": max(1, int(window * block["rate_max"])),
            })
        elif block["max_seconds"] < window * 0.35 and window >= 2.0:
            issues.append({
                "kind": "too_short",
                "index": block["index"],
                "text": block["text"],
                "detail": (
                    f"台词 {block['index']}（{block['tone']}，{block['units']}字）只需 "
                    f"{block['max_seconds']:.2f} 秒，而该镜头有 {window:.2f} 秒，画面会长时间无声。"
                    f"可扩充到约 {max(1, int(window * block['rate_min']))} 字，或补充动作/环境音"
                ),
                "suggest_units": max(1, int(window * block["rate_min"])),
            })

    spoken = round(sum(block["min_seconds"] for block in blocks), 2)
    return {
        "blocks": blocks,
        "issues": issues,
        "spoken_seconds": spoken,
        "target_seconds": total,
        "ok": not issues,
    }


def _shot_windows(
    prompt: str,
    plan: list[dict[str, Any]] | None,
    total: float,
    block_count: int,
) -> dict[int, float]:
    """Map each dialogue line to the seconds available to speak it."""
    if not plan:
        # No plan: assume the lines share the clip evenly.
        share = total / max(1, block_count)
        return {index: share for index in range(1, block_count + 1)}

    durations = [max(0.1, float(shot.get("end", 0)) - float(shot.get("start", 0))) for shot in plan]
    marker_re = re.compile(r"(?:\[\s*)?\bShot\s*(\d+)", re.IGNORECASE)
    windows: dict[int, float] = {}
    per_shot: dict[int, int] = {}
    value = str(prompt or "")

    for index, match in enumerate(DIALOGUE_RE.finditer(value), 1):
        markers = marker_re.findall(value[: match.start()])
        shot_no = int(markers[-1]) if markers else 0
        if 1 <= shot_no <= len(durations):
            per_shot[shot_no] = per_shot.get(shot_no, 0) + 1
            windows[index] = (shot_no, durations[shot_no - 1])
        else:
            windows[index] = (0, total / max(1, block_count))

    # Two lines inside one shot each get half of it.
    resolved: dict[int, float] = {}
    for index, (shot_no, window) in windows.items():
        share = per_shot.get(shot_no, 1) if shot_no else 1
        resolved[index] = window / max(1, share)
    return resolved


def report_text(result: dict[str, Any]) -> str:
    """Human-readable audit summary for the manifest report."""
    if not result.get("blocks"):
        return "无台词，跳过对话检查。"
    lines = [
        f"台词 {len(result['blocks'])} 句，最快念完需 {result['spoken_seconds']:.2f} 秒 / 目标 {result['target_seconds']:.2f} 秒。"
    ]
    for block in result["blocks"]:
        lines.append(
            f"  · 台词{block['index']}[{block['script_name']}/{block['tone']} "
            f"{block['rate_min']}-{block['rate_max']}字/s] {block['units']}字 → "
            f"{block['min_seconds']:.2f}-{block['max_seconds']:.2f}秒 / 可用 {block.get('window_seconds', 0):.2f}秒"
        )
    if result["issues"]:
        lines.append("发现问题：")
        lines.extend(f"  ! {issue['detail']}" for issue in result["issues"])
    else:
        lines.append("语言与时长均匹配。")
    return "\n".join(lines)
