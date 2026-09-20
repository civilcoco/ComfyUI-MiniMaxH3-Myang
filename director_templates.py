"""Reusable Director video templates with explicit fixed/input slots.

Templates are permanent JSON documents stored under the ComfyUI user directory.
The browser may export a portable document with embedded fixed media; during
import those bytes are restored into ComfyUI/input before this module receives
and persists the normalized file references.  The server-side catalogue itself
therefore stays small and never stores base64 media payloads.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import folder_paths


FORMAT = "minimax-h3-myang-director-template"
VERSION = 2
MAX_TEMPLATES = 128
MAX_CARDS = 256
MAX_MATERIALS = 64
KINDS = frozenset({"image", "video", "audio"})
TASK_MODES = frozenset({
    "纯生成（不用参考视频）",
    "动作迁移（跟随参考视频）",
    "视频续写（接着往下演）",
})
DEFAULT_TASK_MODE = "纯生成（不用参考视频）"
_ROUTES_REGISTERED = False


def templates_path() -> Path:
    return (Path(folder_paths.get_user_directory()) / "default" / "Myang_node"
            / "config" / "director_templates.json")


def _text(value: Any, maximum: int = 500) -> str:
    return str(value or "").strip()[:maximum]


def _slot_id(value: Any, fallback: str) -> str:
    clean = _text(value, 100)
    if clean:
        return clean
    return "slot_" + hashlib.sha256(fallback.encode("utf-8")).hexdigest()[:16]


def _material(record: Any, fallback: str) -> dict[str, Any] | None:
    if not isinstance(record, dict):
        return None
    kind = _text(record.get("kind"), 16).lower()
    if kind not in KINDS:
        return None
    mode = "input" if _text(record.get("mode"), 16).lower() == "input" else "fixed"
    file_info = record.get("file") if isinstance(record.get("file"), dict) else {}
    filename = _text(file_info.get("name"), 260)
    if mode == "fixed" and not filename:
        return None
    result = {
        "slot_id": _slot_id(record.get("slot_id"), fallback),
        "mode": mode,
        "kind": kind,
        "role": "action" if kind == "video" and record.get("role") == "action" else "reference",
        "label": _text(record.get("label") or filename or "参考素材", 120),
        "input_label": _text(
            record.get("input_label") or record.get("label") or filename or "参考素材", 120),
        "required": record.get("required") is not False,
    }
    weight_mode = _text(record.get("reference_weight_mode"), 16).lower()
    if weight_mode in {"auto", "manual"}:
        try:
            weight = float(record.get("reference_weight") or 1.0)
        except (TypeError, ValueError):
            weight = 1.0
        result["reference_weight_mode"] = weight_mode
        result["reference_weight"] = max(0.25, min(3.0, weight))
    if mode == "fixed":
        result["file"] = {
            "name": filename,
            "subfolder": _text(file_info.get("subfolder"), 260),
            "type": "input",
        }
    return result


def _materials(source: Any, scope: str) -> list[dict[str, Any]]:
    if not isinstance(source, list):
        return []
    result = []
    for index, raw in enumerate(source[:MAX_MATERIALS]):
        item = _material(raw, "%s|%d" % (scope, index + 1))
        if item:
            result.append(item)
    return result


def _card(record: Any, index: int) -> dict[str, Any] | None:
    if not isinstance(record, dict):
        return None
    prompt_mode = (
        "input" if _text(record.get("prompt_mode"), 16).lower() == "input"
        else "fixed")
    prompt = _text(record.get("prompt"), 120_000) if prompt_mode == "fixed" else ""
    try:
        duration = max(0.2, min(30.0, float(record.get("duration_seconds") or 5.0)))
    except (TypeError, ValueError):
        duration = 5.0
    transition = _text(record.get("transition"), 16)
    if index == 0:
        transition = "开场"
    elif transition not in {"承接", "切镜"}:
        transition = "承接"
    return {
        "order": index + 1,
        "enabled": record.get("enabled") is not False,
        "title": _text(record.get("title") or "分镜%d" % (index + 1), 120),
        "duration_seconds": duration,
        "transition": transition,
        "prompt_mode": prompt_mode,
        "prompt": prompt,
        "prompt_label": _text(record.get("prompt_label") or "分镜%d提示词" % (index + 1), 120),
        "material_policy": (
            "叠加全局素材" if record.get("material_policy") == "叠加全局素材"
            else "仅本镜头"),
        "materials": _materials(record.get("materials"), "card:%d" % (index + 1)),
    }


def _settings(source: Any) -> dict[str, str | int | float | bool]:
    if not isinstance(source, dict):
        return {}
    result = {}
    for raw_name, value in list(source.items())[:160]:
        name = _text(raw_name, 100)
        if not name or not isinstance(value, (str, int, float, bool)):
            continue
        result[name] = _text(value, 500) if isinstance(value, str) else value
    return result


def _settings_policy(source: Any) -> dict[str, str]:
    if not isinstance(source, dict):
        return {}
    result = {}
    for raw_name, raw_mode in list(source.items())[:160]:
        name = _text(raw_name, 100)
        if not name:
            continue
        result[name] = "fixed" if _text(raw_mode, 16).lower() == "fixed" else "local"
    return result


def _duration_rule(source: Any, fallback: float) -> dict[str, str | float]:
    record = source if isinstance(source, dict) else {}
    mode = (_text(record.get("mode"), 16).lower()
            if _text(record.get("mode"), 16).lower()
            in {"fixed", "input", "reference"} else "input")
    if mode == "reference":
        return {"mode": "reference", "value": 0.0}
    try:
        value = max(0.2, min(3600.0, float(record.get("value") or fallback)))
    except (TypeError, ValueError):
        value = fallback
    return {"mode": mode, "value": value}


def _normalise(record: Any) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ValueError("模板数据不是有效对象")
    raw_cards = record.get("cards")
    if not isinstance(raw_cards, list) or not raw_cards:
        raise ValueError("模板至少需要一张分镜卡")
    cards = [item for index, raw in enumerate(raw_cards[:MAX_CARDS])
             if (item := _card(raw, index)) is not None]
    if not cards:
        raise ValueError("模板没有有效分镜卡")
    name = _text(record.get("name"), 120)
    if not name:
        raise ValueError("模板名称不能为空")
    now = time.time()
    stable = "%s|%s" % (name, _text(record.get("created"), 40) or str(now))
    settings = _settings(record.get("settings"))
    settings_fixed = record.get("settings_fixed") is True
    settings_policy = _settings_policy(record.get("settings_policy"))
    if not settings_policy:
        settings_policy = {
            name: ("fixed" if settings_fixed else "local") for name in settings
        }
    settings_fixed = any(mode == "fixed" for mode in settings_policy.values())
    task_mode = (_text(record.get("task_mode"), 80)
                 if _text(record.get("task_mode"), 80) in TASK_MODES
                 else DEFAULT_TASK_MODE)
    if task_mode == "动作迁移（跟随参考视频）":
        # Action transfer has one full-film prompt/material payload. Older UI
        # versions copied the first line of that prompt into a storyboard
        # title, sometimes persisting separator bars and an entire script name.
        # Clear it during catalogue reads without touching the prompt itself.
        cards = cards[:1]
        legacy_title = cards[0]["title"]
        if legacy_title and legacy_title in cards[0]["prompt_label"]:
            cards[0]["prompt_label"] = "动作迁移提示词"
        cards[0]["title"] = ""
    raw_duration = (record.get("duration_policy")
                    if isinstance(record.get("duration_policy"), dict) else {})
    legacy = int(record.get("version") or 1) < 2
    try:
        total_fallback = float(settings.get("total_seconds") or sum(
            card["duration_seconds"] for card in cards) or 5.0)
    except (TypeError, ValueError):
        total_fallback = sum(card["duration_seconds"] for card in cards) or 5.0
    try:
        segment_fallback = float(settings.get("segment_seconds") or 5.0)
    except (TypeError, ValueError):
        segment_fallback = 5.0
    duration_policy = {
        "total": _duration_rule(raw_duration.get("total"), total_fallback),
        "segment": _duration_rule(raw_duration.get("segment"), segment_fallback),
    }
    if legacy and not raw_duration:
        # Version-1 templates had no duration input contract. Preserve their
        # old behavior instead of unexpectedly changing Director time values.
        duration_policy = {}
    if task_mode == "动作迁移（跟随参考视频）":
        # Migrate old action templates too: the decoded action video, not a
        # stale saved number, owns total runtime and automatic segment count.
        duration_policy["total"] = {"mode": "reference", "value": 0.0}
    return {
        "id": _text(record.get("id"), 100) or (
            "template_" + hashlib.sha256(stable.encode("utf-8")).hexdigest()[:18]),
        "format": FORMAT,
        "version": VERSION,
        "name": name,
        "description": _text(record.get("description"), 1200),
        "task_mode": task_mode,
        "created": float(record.get("created") or now),
        "updated": float(record.get("updated") or now),
        "cards": cards,
        "global_materials": _materials(record.get("global_materials"), "global"),
        "settings": settings,
        "settings_fixed": settings_fixed,
        "settings_policy": settings_policy,
        "duration_policy": duration_policy,
        "interface_locked": record.get("interface_locked") is True,
    }


def load_templates(path: Path | None = None) -> list[dict[str, Any]]:
    target = path or templates_path()
    if not target.is_file():
        return []
    try:
        payload = json.loads(target.read_text("utf-8"))
    except json.JSONDecodeError as error:
        # Never turn a damaged permanent catalogue into an apparently empty
        # one: the next save would otherwise overwrite every recoverable
        # template. Surface the problem and leave the original file untouched.
        raise ValueError("本地模板仓库 JSON 已损坏，请先备份并修复 director_templates.json") from error
    except OSError as error:
        raise ValueError("本地模板仓库无法读取，请检查文件权限或磁盘状态") from error
    raw_templates = payload.get("templates") if isinstance(payload, dict) else []
    result, seen = [], set()
    for raw in raw_templates if isinstance(raw_templates, list) else []:
        try:
            item = _normalise(raw)
        except ValueError:
            continue
        if item["id"] in seen:
            continue
        seen.add(item["id"])
        result.append(item)
    return result


def _save(items: list[dict[str, Any]], path: Path | None = None) -> None:
    target = path or templates_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps({
        "version": VERSION, "templates": items[:MAX_TEMPLATES],
    }, ensure_ascii=False, indent=2), "utf-8")
    temporary.replace(target)


def add_template(record: dict[str, Any], path: Path | None = None) -> dict[str, Any]:
    item = _normalise(record)
    items = load_templates(path)
    existing = next((value for value in items if value["id"] == item["id"]), None)
    if existing is not None:
        item["created"] = existing["created"]
        item["updated"] = time.time()
        items[items.index(existing)] = item
    else:
        if len(items) >= MAX_TEMPLATES:
            raise ValueError("导演台模板数量已达到上限 %d" % MAX_TEMPLATES)
        items.append(item)
    _save(items, path)
    return item


def update_template(template_id: str, changes: dict[str, Any],
                    path: Path | None = None) -> dict[str, Any]:
    items = load_templates(path)
    current = next((item for item in items if item["id"] == str(template_id)), None)
    if current is None:
        raise ValueError("导演台模板不存在")
    merged = dict(current)
    for key in ("name", "description", "task_mode", "cards", "global_materials",
                "settings", "settings_fixed", "settings_policy", "duration_policy",
                "interface_locked"):
        if key in changes:
            merged[key] = changes[key]
    merged["id"] = current["id"]
    merged["created"] = current["created"]
    merged["updated"] = time.time()
    item = _normalise(merged)
    items[items.index(current)] = item
    _save(items, path)
    return item


def remove_template(template_id: str, path: Path | None = None) -> bool:
    items = load_templates(path)
    kept = [item for item in items if item["id"] != str(template_id)]
    if len(kept) == len(items):
        return False
    _save(kept, path)
    return True


def _require_loopback(request) -> None:
    remote = str(getattr(request, "remote", "") or "")
    if remote not in {"127.0.0.1", "::1", "localhost"}:
        from aiohttp import web
        raise web.HTTPForbidden(text="导演台模板只允许本机管理")


def register_routes() -> None:
    global _ROUTES_REGISTERED
    if _ROUTES_REGISTERED:
        return
    from aiohttp import web
    from server import PromptServer

    prompt_server = getattr(PromptServer, "instance", None)
    if prompt_server is None:
        return

    @prompt_server.routes.get("/minimax-h3-myang/director-templates")
    async def list_director_templates(request):
        _require_loopback(request)
        return web.json_response({"templates": load_templates()})

    @prompt_server.routes.post("/minimax-h3-myang/director-templates")
    async def create_director_template(request):
        _require_loopback(request)
        try:
            item = add_template(await request.json())
        except ValueError as error:
            raise web.HTTPBadRequest(text=str(error)) from error
        return web.json_response({"template": item})

    @prompt_server.routes.patch("/minimax-h3-myang/director-templates/{template_id}")
    async def patch_director_template(request):
        _require_loopback(request)
        try:
            item = update_template(
                request.match_info["template_id"], await request.json())
        except ValueError as error:
            raise web.HTTPBadRequest(text=str(error)) from error
        return web.json_response({"template": item})

    @prompt_server.routes.delete("/minimax-h3-myang/director-templates/{template_id}")
    async def delete_director_template(request):
        _require_loopback(request)
        return web.json_response({
            "removed": remove_template(request.match_info["template_id"]),
        })

    _ROUTES_REGISTERED = True
