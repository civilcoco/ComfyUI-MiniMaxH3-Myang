"""Categorised reusable asset catalogue for the Myang Director.

The catalogue stores references and subject metadata only.  It never copies or
deletes the user's media files, so removing an entry is always recoverable by
adding the existing ComfyUI/input material again.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import folder_paths


CATEGORIES = frozenset({
    "character", "scene", "voice", "music", "video", "image", "other",
})
KINDS = frozenset({"image", "video", "audio"})
_ROUTES_REGISTERED = False


def catalogue_path() -> Path:
    return (Path(folder_paths.get_user_directory()) / "default" / "Myang_node"
            / "config" / "asset_catalogue.json")


def _clean_text(value: Any, maximum: int = 500) -> str:
    return str(value or "").strip()[:maximum]


def _normalise(record: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(record, dict):
        return None
    kind = _clean_text(record.get("kind"), 16).lower()
    category = _clean_text(record.get("category"), 24).lower()
    if kind not in KINDS:
        return None
    if category not in CATEGORIES:
        category = {"image": "image", "video": "video", "audio": "music"}[kind]
    file_info = record.get("file") if isinstance(record.get("file"), dict) else {}
    filename = _clean_text(file_info.get("name") or record.get("filename"), 260)
    if not filename:
        return None
    stable = "|".join((
        kind, filename, _clean_text(file_info.get("subfolder"), 260),
        _clean_text(file_info.get("type") or "input", 24),
        _clean_text(record.get("subject_id"), 80),
    ))
    return {
        "id": _clean_text(record.get("id"), 80) or (
            "asset_" + hashlib.sha256(stable.encode("utf-8")).hexdigest()[:18]),
        "name": _clean_text(record.get("name") or record.get("subject_name") or filename, 120),
        "category": category,
        "kind": kind,
        "subject_id": _clean_text(record.get("subject_id"), 80),
        "subject_name": _clean_text(record.get("subject_name"), 120),
        "identity": _clean_text(record.get("identity"), 1200),
        "file": {
            "name": filename,
            "subfolder": _clean_text(file_info.get("subfolder"), 260),
            "type": _clean_text(file_info.get("type") or "input", 24),
        },
        "created": float(record.get("created") or time.time()),
        "updated": float(record.get("updated") or time.time()),
    }


def load_catalogue(path: Path | None = None) -> list[dict[str, Any]]:
    target = path or catalogue_path()
    if not target.is_file():
        return []
    try:
        payload = json.loads(target.read_text("utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    raw_assets = payload.get("assets") if isinstance(payload, dict) else []
    result, seen = [], set()
    for raw in raw_assets if isinstance(raw_assets, list) else []:
        item = _normalise(raw)
        if item and item["id"] not in seen:
            seen.add(item["id"])
            result.append(item)
    return result


def _save(assets: list[dict[str, Any]], path: Path | None = None) -> None:
    target = path or catalogue_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps({
        "version": 1, "assets": assets,
    }, ensure_ascii=False, indent=2), "utf-8")
    temporary.replace(target)


def add_asset(record: dict[str, Any], path: Path | None = None) -> dict[str, Any]:
    item = _normalise(record)
    if item is None:
        raise ValueError("素材记录缺少有效的文件、类型或名称")
    assets = load_catalogue(path)
    existing = next((asset for asset in assets if asset["id"] == item["id"]), None)
    if existing is not None:
        created = existing["created"]
        existing.update(item)
        existing["created"] = created
        existing["updated"] = time.time()
        item = existing
    else:
        assets.append(item)
    _save(assets, path)
    return item


def update_asset(asset_id: str, changes: dict[str, Any],
                 path: Path | None = None) -> dict[str, Any]:
    assets = load_catalogue(path)
    current = next((asset for asset in assets if asset["id"] == str(asset_id)), None)
    if current is None:
        raise ValueError("素材库条目不存在")
    merged = dict(current)
    for key in ("name", "category", "subject_name", "identity"):
        if key in changes:
            merged[key] = changes[key]
    merged["id"] = current["id"]
    merged["created"] = current["created"]
    merged["updated"] = time.time()
    normalised = _normalise(merged)
    if normalised is None:
        raise ValueError("素材库修改无效")
    assets[assets.index(current)] = normalised
    _save(assets, path)
    return normalised


def remove_asset(asset_id: str, path: Path | None = None) -> bool:
    assets = load_catalogue(path)
    kept = [asset for asset in assets if asset["id"] != str(asset_id)]
    if len(kept) == len(assets):
        return False
    _save(kept, path)
    return True


def _require_loopback(request) -> None:
    remote = str(getattr(request, "remote", "") or "")
    if remote not in {"127.0.0.1", "::1", "localhost"}:
        from aiohttp import web
        raise web.HTTPForbidden(text="导演台素材库只允许本机管理")


def register_routes() -> None:
    global _ROUTES_REGISTERED
    if _ROUTES_REGISTERED:
        return
    from aiohttp import web
    from server import PromptServer

    prompt_server = getattr(PromptServer, "instance", None)
    if prompt_server is None:
        return

    @prompt_server.routes.get("/minimax-h3-myang/assets")
    async def list_assets(request):
        _require_loopback(request)
        return web.json_response({"assets": load_catalogue()})

    @prompt_server.routes.post("/minimax-h3-myang/assets")
    async def create_asset(request):
        _require_loopback(request)
        try:
            item = add_asset(await request.json())
        except ValueError as error:
            raise web.HTTPBadRequest(text=str(error)) from error
        return web.json_response({"asset": item})

    @prompt_server.routes.patch("/minimax-h3-myang/assets/{asset_id}")
    async def patch_asset(request):
        _require_loopback(request)
        try:
            item = update_asset(request.match_info["asset_id"], await request.json())
        except ValueError as error:
            raise web.HTTPBadRequest(text=str(error)) from error
        return web.json_response({"asset": item})

    @prompt_server.routes.delete("/minimax-h3-myang/assets/{asset_id}")
    async def delete_asset(request):
        _require_loopback(request)
        return web.json_response({
            "removed": remove_asset(request.match_info["asset_id"]),
        })

    _ROUTES_REGISTERED = True
