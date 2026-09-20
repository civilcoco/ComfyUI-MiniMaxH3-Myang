"""Local media-library registry and HTTP boundary for the rough-cut editor."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
from pathlib import Path, PurePosixPath
from typing import Any

import av
import folder_paths


MEDIA_EXTENSIONS = {
    "image": frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}),
    "video": frozenset({".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".wmv"}),
    "audio": frozenset({".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".opus"}),
}
ALL_MEDIA_EXTENSIONS = frozenset().union(*MEDIA_EXTENSIONS.values())
MAX_LIBRARY_ASSETS = 10000
_ROUTES_REGISTERED = False


def config_path() -> Path:
    return (Path(folder_paths.get_user_directory()) / "default" / "Myang_node"
            / "config" / "roughcut_libraries.json")


def _kind(path: Path) -> str:
    suffix = path.suffix.lower()
    return next((kind for kind, extensions in MEDIA_EXTENSIONS.items()
                 if suffix in extensions), "")


def _library_id(path: Path) -> str:
    canonical = os.path.normcase(str(path.resolve()))
    return "lib_" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def load_libraries(path: Path | None = None) -> list[dict[str, Any]]:
    target = path or config_path()
    if not target.is_file():
        return []
    try:
        payload = json.loads(target.read_text("utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    records = payload.get("libraries") if isinstance(payload, dict) else None
    if not isinstance(records, list):
        return []
    normalized = []
    seen = set()
    for record in records:
        if not isinstance(record, dict):
            continue
        raw_path = str(record.get("path") or "").strip()
        if not raw_path:
            continue
        root = Path(raw_path).expanduser().resolve()
        library_id = _library_id(root)
        if library_id in seen:
            continue
        seen.add(library_id)
        normalized.append({
            "id": library_id,
            "name": str(record.get("name") or root.name or root),
            "path": str(root),
            "available": root.is_dir(),
        })
    return normalized


def _save_libraries(libraries: list[dict[str, Any]], path: Path | None = None) -> None:
    target = path or config_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": 1, "libraries": [
        {"name": item["name"], "path": item["path"]} for item in libraries
    ]}
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), "utf-8")
    temporary.replace(target)


def add_library(raw_path: str, name: str = "", path: Path | None = None) -> dict[str, Any]:
    if not str(raw_path or "").strip():
        raise ValueError("请填写素材文件夹路径")
    root = Path(str(raw_path)).expanduser().resolve()
    if not root.is_dir():
        raise ValueError("素材文件夹不存在或无法访问")
    libraries = load_libraries(path)
    library_id = _library_id(root)
    existing = next((item for item in libraries if item["id"] == library_id), None)
    if existing is not None:
        return existing
    record = {
        "id": library_id,
        "name": str(name or root.name or root),
        "path": str(root),
        "available": True,
    }
    libraries.append(record)
    _save_libraries(libraries, path)
    return record


def remove_library(library_id: str, path: Path | None = None) -> bool:
    libraries = load_libraries(path)
    kept = [item for item in libraries if item["id"] != str(library_id)]
    if len(kept) == len(libraries):
        return False
    _save_libraries(kept, path)
    return True


def _library(library_id: str, path: Path | None = None) -> dict[str, Any]:
    record = next((item for item in load_libraries(path)
                   if item["id"] == str(library_id)), None)
    if record is None:
        raise ValueError("素材库不存在或已移除")
    root = Path(record["path"]).resolve()
    if not root.is_dir():
        raise ValueError("素材库文件夹当前不可用")
    return record


def resolve_asset(library_id: str, relative_path: str,
                  path: Path | None = None) -> Path:
    record = _library(library_id, path)
    relative = PurePosixPath(str(relative_path or "").replace("\\", "/"))
    if not relative.parts or relative.is_absolute() or ".." in relative.parts:
        raise ValueError("素材相对路径无效")
    root = Path(record["path"]).resolve()
    target = root.joinpath(*relative.parts).resolve()
    try:
        target.relative_to(root)
    except ValueError as error:
        raise ValueError("素材路径越出了已登记文件夹") from error
    if not target.is_file() or target.suffix.lower() not in ALL_MEDIA_EXTENSIONS:
        raise ValueError("素材不存在或格式不受支持")
    return target


def resolve_output(filename: str, subfolder: str = "") -> Path:
    name = str(filename or "").strip()
    relative_folder = PurePosixPath(str(subfolder or "").replace("\\", "/"))
    if (not name or PurePosixPath(name).name != name or relative_folder.is_absolute()
            or ".." in relative_folder.parts):
        raise ValueError("输出素材引用无效")
    root = Path(folder_paths.get_output_directory()).resolve()
    target = root.joinpath(*relative_folder.parts, name).resolve()
    try:
        target.relative_to(root)
    except ValueError as error:
        raise ValueError("输出素材路径越出了 ComfyUI/output") from error
    if not target.is_file() or target.suffix.lower() not in ALL_MEDIA_EXTENSIONS:
        raise ValueError("输出素材不存在或格式不受支持")
    return target


def resolve_input(filename: str, subfolder: str = "") -> Path:
    """Resolve a Director-catalogue reference inside ComfyUI/input.

    The reusable Director catalogue stores input-relative file records rather
    than rough-cut folder-library ids.  Keep the same containment guarantees as
    folder and output sources so catalogue media can safely be used on the
    timeline without copying it again.
    """
    name = str(filename or "").strip()
    relative_folder = PurePosixPath(str(subfolder or "").replace("\\", "/"))
    if (not name or PurePosixPath(name).name != name or relative_folder.is_absolute()
            or ".." in relative_folder.parts):
        raise ValueError("导演台素材库引用无效")
    root = Path(folder_paths.get_input_directory()).resolve()
    target = root.joinpath(*relative_folder.parts, name).resolve()
    try:
        target.relative_to(root)
    except ValueError as error:
        raise ValueError("导演台素材路径越出了 ComfyUI/input") from error
    if not target.is_file() or target.suffix.lower() not in ALL_MEDIA_EXTENSIONS:
        raise ValueError("导演台素材不存在或格式不受支持")
    return target


def scan_library(library_id: str, path: Path | None = None,
                 maximum: int = MAX_LIBRARY_ASSETS) -> list[dict[str, Any]]:
    record = _library(library_id, path)
    root = Path(record["path"]).resolve()
    assets = []
    for directory, folders, files in os.walk(root, followlinks=False):
        folders[:] = sorted(folders, key=str.casefold)
        for filename in sorted(files, key=str.casefold):
            candidate = Path(directory, filename)
            if candidate.suffix.lower() not in ALL_MEDIA_EXTENSIONS:
                continue
            try:
                resolved = candidate.resolve()
                relative = resolved.relative_to(root).as_posix()
            except (OSError, ValueError):
                continue
            stat = resolved.stat()
            assets.append({
                "library_id": record["id"],
                "relative_path": relative,
                "name": resolved.name,
                "kind": _kind(resolved),
                "size": int(stat.st_size),
                "modified": float(stat.st_mtime),
            })
            if len(assets) >= int(maximum):
                return assets
    return assets


def _probe_path(target: Path) -> dict[str, Any]:
    result = {"kind": _kind(target), "duration": 0.0, "fps": 0.0,
              "width": 0, "height": 0}
    with av.open(str(target), mode="r") as container:
        if container.duration is not None:
            result["duration"] = float(container.duration / av.time_base)
        video = next((stream for stream in container.streams if stream.type == "video"), None)
        audio = next((stream for stream in container.streams if stream.type == "audio"), None)
        if video is not None:
            result["width"], result["height"] = int(video.width), int(video.height)
            if video.average_rate:
                result["fps"] = float(video.average_rate)
            if result["duration"] <= 0 and video.duration is not None and video.time_base is not None:
                result["duration"] = float(video.duration * video.time_base)
        elif audio is not None and result["duration"] <= 0:
            if audio.duration is not None and audio.time_base is not None:
                result["duration"] = float(audio.duration * audio.time_base)
    if result["kind"] == "image":
        result["duration"] = 0.0
        result["fps"] = 0.0
    return result


def probe_asset(library_id: str, relative_path: str,
                path: Path | None = None) -> dict[str, Any]:
    return _probe_path(resolve_asset(library_id, relative_path, path))


def probe_input(filename: str, subfolder: str = "") -> dict[str, Any]:
    return _probe_path(resolve_input(filename, subfolder))


def choose_library_folder() -> str:
    """Open the native folder picker on the local ComfyUI host."""
    root = None
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        root.update()
        return str(filedialog.askdirectory(
            parent=root, title="选择沐阳素材库文件夹", mustexist=True) or "")
    except Exception as error:
        raise RuntimeError("无法打开系统文件夹选择器：%s" % error) from error
    finally:
        if root is not None:
            root.destroy()


def import_asset(library_id: str, relative_path: str) -> dict[str, str]:
    """Copy a selected library asset into ComfyUI/input for Director use."""
    source = resolve_asset(library_id, relative_path)
    library = _library(library_id)
    input_root = Path(folder_paths.get_input_directory()).resolve()
    destination_root = (input_root / "Myang_node" / "library_imports"
                        / str(library["id"])).resolve()
    destination_root.relative_to(input_root)
    destination_root.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(
        (str(source.resolve()) + "|" + str(source.stat().st_mtime_ns)).encode("utf-8")
    ).hexdigest()[:12]
    safe_stem = "".join(
        character if character.isalnum() or character in "-_." else "_"
        for character in source.stem)[:120].strip("._") or "asset"
    destination = destination_root / f"{safe_stem}_{digest}{source.suffix.lower()}"
    if not destination.is_file() or destination.stat().st_size != source.stat().st_size:
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        try:
            shutil.copy2(source, temporary)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
    relative = destination.relative_to(input_root)
    return {
        "name": destination.name,
        "subfolder": relative.parent.as_posix(),
        "type": "input",
    }


def _require_loopback(request) -> None:
    remote = str(getattr(request, "remote", "") or "")
    if remote not in {"127.0.0.1", "::1", "localhost"}:
        from aiohttp import web
        raise web.HTTPForbidden(text="粗剪素材库只允许本机管理")


def register_routes() -> None:
    global _ROUTES_REGISTERED
    if _ROUTES_REGISTERED:
        return
    from aiohttp import web
    from server import PromptServer

    prompt_server = getattr(PromptServer, "instance", None)
    if prompt_server is None:
        return

    @prompt_server.routes.get("/minimax-h3-myang/roughcut/libraries")
    async def list_roughcut_libraries(request):
        _require_loopback(request)
        return web.json_response({"libraries": load_libraries()})

    @prompt_server.routes.post("/minimax-h3-myang/roughcut/libraries")
    async def add_roughcut_library(request):
        _require_loopback(request)
        payload = await request.json()
        try:
            record = add_library(payload.get("path", ""), payload.get("name", ""))
        except ValueError as error:
            raise web.HTTPBadRequest(text=str(error)) from error
        return web.json_response({"library": record})

    @prompt_server.routes.post("/minimax-h3-myang/roughcut/browse-folder")
    async def browse_roughcut_library(request):
        _require_loopback(request)
        try:
            payload = await request.json()
        except (json.JSONDecodeError, TypeError):
            payload = {}
        try:
            selected = await asyncio.to_thread(choose_library_folder)
            if not selected:
                return web.json_response({"cancelled": True})
            record = add_library(selected, payload.get("name", ""))
        except (ValueError, RuntimeError, OSError) as error:
            raise web.HTTPBadRequest(text=str(error)) from error
        return web.json_response({"cancelled": False, "library": record})

    @prompt_server.routes.post("/minimax-h3-myang/roughcut/import")
    async def import_roughcut_asset(request):
        _require_loopback(request)
        payload = await request.json()
        try:
            file_info = await asyncio.to_thread(
                import_asset, payload.get("library_id", ""), payload.get("path", ""))
        except (ValueError, OSError) as error:
            raise web.HTTPBadRequest(text=str(error)) from error
        return web.json_response({"file": file_info})

    @prompt_server.routes.delete("/minimax-h3-myang/roughcut/libraries/{library_id}")
    async def delete_roughcut_library(request):
        _require_loopback(request)
        removed = remove_library(request.match_info["library_id"])
        return web.json_response({"removed": removed})

    @prompt_server.routes.get("/minimax-h3-myang/roughcut/assets")
    async def list_roughcut_assets(request):
        _require_loopback(request)
        try:
            assets = scan_library(request.query.get("library_id", ""))
        except ValueError as error:
            raise web.HTTPBadRequest(text=str(error)) from error
        return web.json_response({"assets": assets})

    @prompt_server.routes.get("/minimax-h3-myang/roughcut/probe")
    async def probe_roughcut_asset(request):
        _require_loopback(request)
        try:
            if request.query.get("source_type", "").strip().lower() == "input":
                result = probe_input(
                    request.query.get("filename", ""),
                    request.query.get("subfolder", ""))
            else:
                result = probe_asset(
                    request.query.get("library_id", ""), request.query.get("path", ""))
        except (ValueError, av.error.FFmpegError) as error:
            raise web.HTTPBadRequest(text=str(error)) from error
        return web.json_response(result)

    @prompt_server.routes.get("/minimax-h3-myang/roughcut/media")
    async def serve_roughcut_asset(request):
        _require_loopback(request)
        try:
            target = resolve_asset(
                request.query.get("library_id", ""), request.query.get("path", ""))
        except ValueError as error:
            raise web.HTTPBadRequest(text=str(error)) from error
        return web.FileResponse(target)

    _ROUTES_REGISTERED = True
