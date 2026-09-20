"""Revise an existing Director storyboard without regenerating it.

SPDX-License-Identifier: GPL-3.0-only

Once a script exists, the expensive operations are the ones that throw it away.
Three things an operator actually needs are all cheaper than a rewrite, and all
three are served here as interactive HTTP calls rather than as graph nodes,
because they happen while editing rather than while sampling:

* **Layering an existing shot.** The card editor can hold subjects, timeline,
  sound and dialogue layers, but until now the operator had to type all four.
  Splitting prose that already exists is a mechanical job, and it is the same
  job ``_write_segment_layers`` already does for freshly written segments -- so
  this reuses those exact system prompts instead of inventing new ones.

* **Rebinding materials without touching the story.** Uploading a voice clip for
  a character, or removing a reference image, should not put a word of the script
  at risk. So in this mode the model never sees a rewrite instruction and never
  returns prose: it returns a *binding* only, and the binding is applied
  deterministically. The visual layer comes back byte-identical by construction,
  not by hoping the model behaved.

* **Rewriting one shot on purpose.** The story-changing mode is scoped to a
  single shot and told the duration it has to fit, so a local fix cannot
  cascade into the rest of the timeline.
"""

import asyncio
import json
import logging
import re

from . import dialogue_audit
from . import llm_service
from . import nodes as director_nodes


logger = logging.getLogger(__name__)

LAYER_ROUTE = "/minimax-h3-myang/director/layer-shot"
REBIND_ROUTE = "/minimax-h3-myang/director/rebind-materials"
REWRITE_ROUTE = "/minimax-h3-myang/director/rewrite-shot"

MAX_PROMPT_CHARS = 8000
MAX_INSTRUCTION_CHARS = 2000
MAX_MATERIALS = 32
_ROUTES_REGISTERED = False


def _text(value, limit):
    return str(value or "").strip()[:limit]


def _materials(raw):
    """Normalise the card's material list into the manifest the LLM is shown."""
    items = []
    for entry in (raw or [])[:MAX_MATERIALS]:
        if not isinstance(entry, dict):
            continue
        kind = str(entry.get("kind") or "image")
        if kind not in {"image", "video", "audio"}:
            kind = "image"
        name = _text(entry.get("name") or (entry.get("file") or {}).get("name"), 160)
        if not name:
            continue
        items.append({
            "kind": kind,
            "ordinal": max(1, int(entry.get("ordinal") or len(items) + 1)),
            "label": _text(entry.get("label"), 120),
            "name": name,
            "subject": _text(entry.get("subject"), 120),
        })
    return items


def _manifest_text(materials):
    tag = {"image": "图片", "video": "视频", "audio": "音频"}
    if not materials:
        return "（本次没有可用素材）"
    lines = []
    for item in materials:
        label = item["label"] or item["name"]
        subject = f"，对应主体：{item['subject']}" if item["subject"] else ""
        lines.append(
            f"- @{tag[item['kind']]}{item['ordinal']}：{label}（文件 {item['name']}）{subject}")
    return "\n".join(lines)


def _loads(raw):
    """Parse a model answer that may still be wrapped in prose or a fence."""
    text = re.sub(r"<think\b[^>]*>.*?</think>", "", str(raw or ""),
                  flags=re.IGNORECASE | re.DOTALL).strip()
    fence = re.match(r"^```(?:json)?\s*\n?(.*?)\n?```$", text,
                     re.IGNORECASE | re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        value = json.loads(text)
    except ValueError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return {}
        try:
            value = json.loads(text[start:end + 1])
        except ValueError:
            return {}
    return value if isinstance(value, dict) else {}


def _safe_error(error):
    return re.sub(
        r"(?i)(bearer\s+|api[_-]?key\s*[=:]\s*)[^\s,;]+",
        r"\1[redacted]", str(error or "LLM 请求失败"))[:800]


def _call(service, system_prompt, user_prompt, seed):
    return llm_service.call_llm_interactive(
        str(service or ""), system_prompt, user_prompt,
        seed=int(seed or 0), max_tokens=None)


def layer_shot(prompt, seconds, service, seed=0, materials=None):
    """Split one existing shot into subjects / timeline / sound / dialogue.

    Three small calls rather than one big one, for the same reason the segment
    writer splits them: each answer is a handful of tokens, which is what keeps a
    flash model from rejecting the request and a reasoning model from spending
    its budget in the thinking channel. A call that fails leaves that layer empty
    instead of failing the whole split, and the prose is never touched -- it
    becomes the visual layer verbatim, so nothing can be lost by layering.
    """
    body = _text(prompt, MAX_PROMPT_CHARS)
    if not body:
        raise ValueError("这个镜头还没有提示词，没有可以拆分的内容")
    window = max(0.5, float(seconds or 0.0))
    budget = dialogue_audit.budget_units(window)
    manifest = _manifest_text(_materials(materials))
    layers = {"visual": body}
    notes = []
    requests = (
        ("structure", director_nodes.LAYER_STRUCTURE_SYSTEM, 101),
        ("sound", director_nodes.LAYER_SOUND_SYSTEM, 211),
        ("dialogue", director_nodes.LAYER_DIALOGUE_SYSTEM.format(
            seconds=window, budget=budget), 331),
    )
    user_prompt = (
        f"本段时长：{window:.2f} 秒\n"
        f"可用素材：\n{manifest}\n\n"
        "本段正文：\n" + body
    )
    for name, system_prompt, salt in requests:
        try:
            payload = _loads(_call(service, system_prompt, user_prompt,
                                   int(seed or 0) + salt))
        except Exception as error:  # noqa: BLE001 - one layer is never fatal
            if type(error).__name__ == "InterruptProcessingException":
                raise
            notes.append("%s层失败：%s" % (name, _safe_error(error)[:120]))
            continue
        if not payload:
            notes.append("%s层没有返回可用 JSON" % name)
            continue
        if name == "structure":
            subjects = payload.get("subjects")
            timeline = payload.get("timeline")
            if isinstance(subjects, list):
                layers["subjects_override"] = [item for item in subjects
                                               if isinstance(item, (dict, str))]
            if isinstance(timeline, list):
                layers["timeline"] = [item for item in timeline
                                      if isinstance(item, (dict, str))]
        elif name == "sound":
            if any(payload.get(key) for key in ("ambient", "bgm", "sfx")):
                layers["sound"] = {
                    "ambient": _text(payload.get("ambient"), 120),
                    "bgm": _text(payload.get("bgm"), 120),
                    "sfx": [_text(item, 40) for item
                            in (payload.get("sfx") or [])][:4],
                }
        else:
            lines = payload.get("dialogue")
            if isinstance(lines, list):
                layers["dialogue"] = [item for item in lines
                                      if isinstance(item, (dict, str))]
    logger.info(
        "H3-Myang: 镜头分层 | %.2fs | 台词预算 %d 字 | 主体%d 节拍%d 台词%d%s",
        window, budget, len(layers.get("subjects_override") or []),
        len(layers.get("timeline") or []), len(layers.get("dialogue") or []),
        " | " + "；".join(notes) if notes else "")
    return {"layers": layers, "notes": notes, "budget_units": budget}


REBIND_SYSTEM = """你负责把素材绑定到已有分镜上。你绝对不能改动任何剧情文字。

用户会给你一份分镜清单（每段的编号、标题、出场主体、说话人）和一份素材清单。
请判断每个素材应该挂到哪些分镜上，例如新上传的人声音频应该挂到该角色说话的那几段。

只输出 JSON，不要解释、不要 markdown 代码块：
{"bindings":[{"kind":"audio","ordinal":1,"shots":[1,3],"subject":"角色名","reason":"为什么挂这里"}],
 "unbound":[{"kind":"image","ordinal":2,"reason":"清单里找不到对应主体"}]}

硬规则：
- 你只输出绑定关系。不要输出提示词、不要输出改写建议、不要复述剧情。
- kind 只能是 image / video / audio，ordinal 必须是素材清单里真实存在的编号。
- shots 里只能出现分镜清单里真实存在的编号；同一个素材可以挂多段。
- 人声音频要按说话人匹配：清单里没有这个人说话的段落，就不要硬挂。
- 找不到合理归属的素材放进 unbound 并说明原因，不要为了用掉素材而乱挂。
- 拿不准就少挂：漏挂可以人工补，错挂会污染画面。"""

REWRITE_SYSTEM = """你负责按用户要求改写单独一个分镜的提示词，其他分镜一个字都不许动。

只输出 JSON，不要解释、不要 markdown 代码块：
{{"prompt":"改写后的本段完整提示词","brief":"本段新标题，30字内"}}

硬规则：
- 只改这一个分镜。不要输出别的分镜，也不要在正文里交代别段该怎么改。
- 本段时长固定为 {seconds:.2f} 秒，改写后的内容必须能在这个时长内演完。
- 台词用 <d>台词</d> 包裹，本段台词合计不得超过 {budget} 字。
- 只能引用素材清单里真实存在的 @图片N / @视频N / @音频N，编号含义不得改变。
- 保持与上一段的衔接关系（承接就接着上一段的机位和人物状态，切镜就把新机位交代清楚）。
- 除了用户明确要求改的地方，其余画面风格、人物外观和镜头语言保持原样。"""


def rebind_materials(shots, materials, service, seed=0):
    """Map materials onto existing shots without letting prose be rewritten.

    The story safety here is structural, not a promise extracted from a prompt:
    the model is asked for a binding table and nothing else, so there is no
    channel through which a word of the script could come back changed.
    """
    catalogue = _materials(materials)
    if not catalogue:
        raise ValueError("没有可绑定的素材")
    listing = []
    valid_shots = set()
    for entry in (shots or [])[:director_nodes.MAX_SLOTS]:
        if not isinstance(entry, dict):
            continue
        index = int(entry.get("index") or len(listing) + 1)
        valid_shots.add(index)
        speakers = [_text(name, 60) for name in (entry.get("speakers") or [])]
        listing.append(
            f"- 第{index}段「{_text(entry.get('brief'), 80) or '未命名'}」"
            f"{'，说话人：' + '、'.join(filter(None, speakers)) if any(speakers) else ''}"
            f"{'，出场：' + _text(entry.get('subjects'), 120) if entry.get('subjects') else ''}")
    if not listing:
        raise ValueError("没有可绑定的分镜")
    user_prompt = ("分镜清单：\n" + "\n".join(listing)
                   + "\n\n素材清单：\n" + _manifest_text(catalogue))
    payload = _loads(_call(service, REBIND_SYSTEM, user_prompt, seed))
    known = {(item["kind"], item["ordinal"]) for item in catalogue}
    bindings = []
    for entry in (payload.get("bindings") or []):
        if not isinstance(entry, dict):
            continue
        kind = str(entry.get("kind") or "")
        try:
            ordinal = int(entry.get("ordinal") or 0)
        except (TypeError, ValueError):
            continue
        if (kind, ordinal) not in known:
            continue
        targets = sorted({int(value) for value in (entry.get("shots") or [])
                          if isinstance(value, (int, float))
                          and int(value) in valid_shots})
        if not targets:
            continue
        bindings.append({
            "kind": kind, "ordinal": ordinal, "shots": targets,
            "subject": _text(entry.get("subject"), 120),
            "reason": _text(entry.get("reason"), 200),
        })
    unbound = [{
        "kind": str(entry.get("kind") or ""),
        "ordinal": int(entry.get("ordinal") or 0),
        "reason": _text(entry.get("reason"), 200),
    } for entry in (payload.get("unbound") or []) if isinstance(entry, dict)]
    if not bindings and not unbound:
        raise ValueError("LLM 没有返回可用的绑定结果，请重试或更换模型")
    logger.info("H3-Myang: 素材绑定 | 素材%d | 绑定%d 条 | 未绑定%d 条",
                len(catalogue), len(bindings), len(unbound))
    return {"bindings": bindings, "unbound": unbound}


_TAG_RE = re.compile(r"@(图片|视频|音频)\s*(\d+)")


def rewrite_shot(prompt, seconds, instruction, service, seed=0,
                 materials=None, previous_brief="", next_brief="",
                 transition=""):
    """Rewrite exactly one shot, bounded by its own duration and manifest."""
    body = _text(prompt, MAX_PROMPT_CHARS)
    want = _text(instruction, MAX_INSTRUCTION_CHARS)
    if not want:
        raise ValueError("请先写清楚这个镜头要怎么改")
    window = max(0.5, float(seconds or 0.0))
    budget = dialogue_audit.budget_units(window)
    catalogue = _materials(materials)
    context = []
    if previous_brief:
        context.append(f"上一段：{_text(previous_brief, 120)}")
    if transition:
        context.append(f"本段与上一段的关系：{_text(transition, 20)}")
    if next_brief:
        context.append(f"下一段：{_text(next_brief, 120)}")
    user_prompt = (
        (("\n".join(context) + "\n\n") if context else "")
        + f"可用素材：\n{_manifest_text(catalogue)}\n\n"
        + "本段现有提示词：\n" + (body or "（本段还是空的）")
        + "\n\n用户要求的改动：\n" + want
    )
    payload = _loads(_call(
        service,
        REWRITE_SYSTEM.format(seconds=window, budget=budget),
        user_prompt, seed))
    revised = _text(payload.get("prompt"), MAX_PROMPT_CHARS)
    if not revised:
        raise ValueError("LLM 没有返回改写后的提示词，请重试或更换模型")
    allowed = {(item["kind"], item["ordinal"]) for item in catalogue}
    kinds = {"图片": "image", "视频": "video", "音频": "audio"}
    invented = sorted({
        f"@{label}{number}" for label, number in _TAG_RE.findall(revised)
        if (kinds[label], int(number)) not in allowed})
    if invented:
        raise ValueError(
            "改写结果引用了不存在的素材：%s。本段可用素材只有 %d 项"
            % ("、".join(invented), len(catalogue)))
    audit = dialogue_audit.audit(revised, window)
    # Only an unspeakably *long* line is a correctness problem: that is what
    # enforce_dialogue_budget defers to the next segment at run time. A line that
    # leaves the shot quiet is advisory, so it belongs in the report the operator
    # reads rather than in the flag the editor colours red.
    too_long = [issue for issue in (audit.get("issues") or [])
                if issue.get("kind") == "too_long"]
    logger.info(
        "H3-Myang: 单镜头改写 | %.2fs | %d字 -> %d字 | 台词 %d 句%s",
        window, len(body), len(revised), len(audit.get("blocks") or []),
        " | 台词超时，将在运行时顺延" if too_long else "")
    return {
        "prompt": revised,
        "brief": _text(payload.get("brief"), 60),
        "dialogue_ok": not too_long,
        "dialogue_report": dialogue_audit.report_text(audit),
        "budget_units": budget,
    }


def _status_for(error):
    """Map a provider failure onto HTTP, matching the image-prompt service."""
    reason = llm_service.error_reason(error)
    if type(error).__name__ == "InterruptProcessingException" \
            or str(error) == "LLM request cancelled":
        return 409, reason
    if reason == "timeout":
        return 504, reason
    if reason in {"rate_limit", "quota_exhausted", "quota_retry", "cooling"}:
        return 429, reason
    return 502, reason


def register_routes():
    global _ROUTES_REGISTERED
    if _ROUTES_REGISTERED:
        return
    from aiohttp import web
    from server import PromptServer

    prompt_server = getattr(PromptServer, "instance", None)
    if prompt_server is None:
        return

    async def _run(request, worker):
        try:
            data = await request.json()
        except Exception:
            return web.json_response({"error": "请求必须是 JSON"}, status=400)
        if not isinstance(data, dict):
            return web.json_response({"error": "请求 JSON 必须是对象"}, status=400)
        try:
            result = await asyncio.to_thread(worker, data)
            return web.json_response({"success": True, **result})
        except ValueError as error:
            return web.json_response({"error": _safe_error(error)}, status=400)
        except Exception as error:  # noqa: BLE001 - classified into HTTP here
            status, reason = _status_for(error)
            response = {"error": _safe_error(error),
                        "reason": reason or "request_error"}
            retry_after = getattr(error, "retry_after", 0.0)
            if retry_after:
                response["retry_after"] = round(float(retry_after), 1)
            return web.json_response(response, status=status)

    @prompt_server.routes.post(LAYER_ROUTE)
    async def layer_shot_route(request):
        return await _run(request, lambda data: layer_shot(
            prompt=data.get("prompt"),
            seconds=data.get("seconds"),
            service=data.get("llm_service"),
            seed=data.get("seed") or 0,
            materials=data.get("materials")))

    @prompt_server.routes.post(REBIND_ROUTE)
    async def rebind_materials_route(request):
        return await _run(request, lambda data: rebind_materials(
            shots=data.get("shots"),
            materials=data.get("materials"),
            service=data.get("llm_service"),
            seed=data.get("seed") or 0))

    @prompt_server.routes.post(REWRITE_ROUTE)
    async def rewrite_shot_route(request):
        return await _run(request, lambda data: rewrite_shot(
            prompt=data.get("prompt"),
            seconds=data.get("seconds"),
            instruction=data.get("instruction"),
            service=data.get("llm_service"),
            seed=data.get("seed") or 0,
            materials=data.get("materials"),
            previous_brief=data.get("previous_brief"),
            next_brief=data.get("next_brief"),
            transition=data.get("transition")))

    _ROUTES_REGISTERED = True
