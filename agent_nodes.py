"""Media-aware prompt agent for ComfyUI-MiniMaxH3-Myang.

SPDX-License-Identifier: GPL-3.0-only

Copyright (C) 2026 Myang

This Media Agent is maintained as part of the Myang package.

The node deliberately treats connected media as the only source of truth. The
LLM is given a strict whitelist and the returned prompt is validated again in
Python before it can reach the MiniMax H3 sampler chain.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import importlib
import json
import logging
import os
import re
import sys
import asyncio
from pathlib import Path
from typing import Any

from aiohttp import web

import folder_paths
from server import PromptServer

try:
    from .media_catalog import (
        MAX_ASSETS,
        MyangMediaAsset,
        MyangMediaCatalog,
        asset_from_input,
        audio_track,
        classify_payload,
        image_batch,
        parse_asset_manifest,
        video_stream,
    )
except ImportError:
    from media_catalog import (
        MAX_ASSETS,
        MyangMediaAsset,
        MyangMediaCatalog,
        asset_from_input,
        audio_track,
        classify_payload,
        image_batch,
        parse_asset_manifest,
        video_stream,
    )

try:
    from . import dialogue_audit
except ImportError:
    import dialogue_audit

logger = logging.getLogger(__name__)

MAX_SKILL_BYTES = 512 * 1024
MAX_SKILL_IMPORT_BYTES = 8 * 1024 * 1024
MAX_VIDEO_FRAMES = 4
MAX_DESCRIPTION_CHARS = 600
SKILL_DIR = Path(__file__).resolve().parent / "skills"
SKILL_TEXT_EXTENSIONS = {".md", ".txt", ".json", ".yaml", ".yml"}
MEDIA_TAG_RE = re.compile(r"<\s*(Picture|Image|Video|Audio)\s+(\d+)\s*>", re.IGNORECASE)
GENERIC_FILE_RE = re.compile(r"[\w\-.\u4e00-\u9fff]+\.(?:png|jpe?g|webp|bmp|gif|mp4|mov|webm|avi|mkv|mp3|wav|m4a|flac|aac|ogg)", re.IGNORECASE)
AT_TOKEN_RE = re.compile(r"(?<![\w.])@([^\s@#<>，。；;,.!！?？:：()（）\[\]{}\"']+)")
EDITOR_AT_RE = re.compile(r"(?<![\w])@([^\s@#<>，。；;!！?？:：()（）\[\]{}\"']+(?:[\t ]+\d+)?)")
EDITOR_INDEX_RE = re.compile(r"@(图片|图|视频|音频|声音)[\t _]*(\d+)")
DIALOGUE_LINE_RE = re.compile(r"(?m)(^|\n)[ \t]*#[ \t]*([^\n]+)")

DEFAULT_AGENT_RULE = """你是 MiniMax H3 的媒体感知提示词 Agent。你的任务是改写成可直接用于 MiniMax H3 参考生视频的高质量提示词。

硬性规则：
1. 只能引用 AVAILABLE MEDIA 中列出的媒体标签，标签必须逐字使用，例如 <Picture 1>、<Video 2>、<Audio 1>。
2. 不得引用、推测或描述任何不在 AVAILABLE MEDIA 中的素材；没有上传的内容必须明确忽略，不能假设它存在。
3. 不得输出 Markdown 代码块、解释说明、思考过程、JSON 或"最终提示词："之类的前言。若 Skill 规定了输出结构（例如 integrated_multimodal_description / overall_soundscape / non_diegetic_music 这类字段名，或 [Shot N] 分镜标记），必须严格按 Skill 的结构输出，这些字段名和标记属于正文的一部分，不算额外标题。
4. 普通画面描述保持自然语言；需要人物说出的台词必须使用 <d>台词</d> 包装，可保留输入中的台词块。
5. 保留用户原始意图、镜头顺序、风格、情绪和动作约束；如果原始输入已经引用了素材，必须保持该引用语义。
6. 若 AVAILABLE MEDIA 中某个标签附带了 content 描述，该描述就是素材的真实内容，可以放心引用其中明确出现的人物外观、服装、场景、动作、风格等细节；没有 content 描述的素材内容未知，不得声称它包含任何无法确认的具体细节。
7. 输出只包含最终提示词正文。

写作策略（Skill 未规定结构时适用）：
- 开头给一句话总览，明确镜头主体、环境、动作和情绪。
- 再补充镜头运动、光线、材质、节奏、色彩和转场。
- 对每一个被引用的媒体，结合其 content 描述（如有）写清楚它用于角色外观、构图、动作、场景、声音或情绪中的哪一方面。
- 台词应尽量简短、有动作动机，台词之外补一句说话状态或语气。
"""

IMAGE_DESCRIBE_PROMPT = """请客观描述这张图片，供视频生成提示词撰写使用。依次说明：主体（人物/物体的外观、服装、姿态）、场景环境、光线与色彩、画面风格、构图。只描述画面中确定可见的内容，不要推测，120字以内，直接输出描述正文，不要标题、不要列表符号。"""

VIDEO_DESCRIBE_PROMPT = """这些图片是同一个视频按时间顺序抽取的关键帧。请客观描述该视频的内容，供视频生成提示词撰写使用。说明：主体外观与动作变化、镜头运动、场景环境、光线与色彩、整体风格与氛围。只描述确定可见的内容，不要推测，150字以内，直接输出描述正文，不要标题、不要分段编号。"""


_EXTRA_SKILLS_VALUE = os.environ.get("MINIMAX_H3_SKILLS_DIR", "").strip()
EXTRA_SKILLS_DIR = (
    Path(_EXTRA_SKILLS_VALUE).expanduser() if _EXTRA_SKILLS_VALUE else None
)


def _skill_dir() -> Path:
    SKILL_DIR.mkdir(parents=True, exist_ok=True)
    return SKILL_DIR


def _skill_directories() -> list[Path]:
    dirs = [_skill_dir()]
    if (EXTRA_SKILLS_DIR is not None and EXTRA_SKILLS_DIR.is_dir()
            and EXTRA_SKILLS_DIR.resolve() != SKILL_DIR.resolve()):
        dirs.append(EXTRA_SKILLS_DIR)
    return dirs


def _path_is_within(path: Path, root: Path) -> bool:
    """Return whether a resolved path is contained by a resolved root."""
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _safe_skill_leaf(name: str) -> str | None:
    """Accept one opaque filename/package id, never a filesystem path."""
    value = str(name or "").strip()
    if not value or value in {".", ".."}:
        return None
    candidate = Path(value)
    if candidate.is_absolute() or candidate.name != value:
        return None
    if "/" in value or "\\" in value or "\x00" in value:
        return None
    return value


def _contained_skill_candidate(root: Path, name: str) -> Path | None:
    leaf = _safe_skill_leaf(name)
    if leaf is None:
        return None
    resolved_root = root.resolve()
    candidate = (resolved_root / leaf).resolve()
    return candidate if _path_is_within(candidate, resolved_root) else None


def _resolve_skill_import_source(dir_path: str) -> Path:
    """Resolve an import directory inside the administrator-approved root."""
    configured = os.environ.get("MINIMAX_H3_SKILLS_IMPORT_DIR", "").strip()
    if not configured:
        raise ValueError(
            "Local directory import is disabled. Set "
            "MINIMAX_H3_SKILLS_IMPORT_DIR to a dedicated import directory first.")
    import_root = Path(configured).expanduser().resolve()
    if not import_root.is_dir():
        raise ValueError("MINIMAX_H3_SKILLS_IMPORT_DIR is not an existing directory")
    requested = Path(str(dir_path or "").strip()).expanduser()
    source = (requested if requested.is_absolute() else import_root / requested).resolve()
    if not _path_is_within(source, import_root):
        raise ValueError("Import path must stay inside MINIMAX_H3_SKILLS_IMPORT_DIR")
    if not source.is_dir():
        raise ValueError("Import path is not an existing directory")
    return source


def _read_import_package(entry: Path) -> list[tuple[Path, bytes]]:
    """Validate a text-only Skill package before copying any of it."""
    if entry.is_symlink():
        raise ValueError("symbolic links are not allowed")
    files: list[tuple[Path, bytes]] = []
    total = 0
    for item in sorted(entry.rglob("*")):
        if item.is_symlink():
            raise ValueError(f"symbolic link is not allowed: {item.name}")
        if item.is_dir():
            continue
        if item.suffix.lower() not in SKILL_TEXT_EXTENSIONS:
            raise ValueError(f"unsupported file type: {item.name}")
        raw = item.read_bytes()
        if len(raw) > MAX_SKILL_BYTES * 4:
            raise ValueError(f"file is too large: {item.name}")
        try:
            raw.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValueError(f"file is not UTF-8: {item.name}") from exc
        total += len(raw)
        if total > MAX_SKILL_IMPORT_BYTES:
            raise ValueError("Skill package exceeds the total import size limit")
        files.append((item.relative_to(entry), raw))
    return files


def _skill_names() -> list[str]:
    names: list[str] = ["none"]
    seen = set(["none"])

    # Phase 1: Scan all Skill package directories across all search paths
    for s_dir in _skill_directories():
        if not s_dir.is_dir():
            continue
        for entry in sorted(s_dir.iterdir()):
            if entry.name.startswith((".", "_")):
                continue
            if entry.is_symlink():
                continue
            if entry.is_dir() and ((entry / "SKILL.md").is_file() or (entry / "SKILL.cn.md").is_file()):
                name = entry.name
                if name not in seen:
                    names.append(name)
                    seen.add(name)
                    seen.add(f"{name}.md")

    # Phase 2: Scan all standalone files, skipping any shadowed by package directories
    for s_dir in _skill_directories():
        if not s_dir.is_dir():
            continue
        for entry in sorted(s_dir.iterdir()):
            # "_" is reserved for our own bookkeeping (the cached skill index).
            if entry.name.startswith((".", "_")) or entry.name in seen:
                continue
            if entry.is_symlink():
                continue
            if entry.is_file() and entry.suffix.lower() in SKILL_TEXT_EXTENSIONS:
                if entry.stem in seen or entry.name in seen:
                    continue
                names.append(entry.name)
                seen.add(entry.name)
    return names


def _get_skill_target(name: str) -> Path | None:
    if not name or name == "none":
        return None
    for s_dir in _skill_directories():
        candidate = _contained_skill_candidate(s_dir, name)
        if candidate is None or candidate.is_symlink():
            continue
        if candidate.is_file() and candidate.suffix.lower() in SKILL_TEXT_EXTENSIONS:
            return candidate
        if candidate.is_dir() and ((candidate / "SKILL.md").is_file()
                                   or (candidate / "SKILL.cn.md").is_file()):
            return candidate
    return None


_prompt_server = getattr(PromptServer, "instance", None)
if _prompt_server is not None:

    @_prompt_server.routes.get("/minimax-h3-agent/skills")
    async def list_agent_skills(request):
        skills_info = []
        for name in _skill_names():
            if name == "none":
                skills_info.append({"name": "none", "content": "", "deletable": False})
                continue
            try:
                content, source = _read_skill(name)
                target = _get_skill_target(name)
                deletable = False
                if target and target.is_file() and target.parent == _skill_dir():
                    deletable = name not in {"default.md", "none"}
                skills_info.append({
                    "name": name,
                    "content": content,
                    "source": source,
                    "deletable": deletable,
                })
            except Exception:
                skills_info.append({"name": name, "content": "", "source": "error", "deletable": False})
        return web.json_response({"skills": _skill_names(), "details": skills_info})

    @_prompt_server.routes.get("/minimax-h3-agent/skill")
    async def get_agent_skill(request):
        name = str(request.query.get("name") or "").strip()
        if not name or name == "none":
            return web.json_response({"name": "none", "content": "", "deletable": False})
        try:
            content, source = _read_skill(name)
            target = _get_skill_target(name)
            deletable = False
            if target and target.is_file() and target.parent == _skill_dir():
                deletable = name not in {"default.md", "none"}
            return web.json_response({"name": name, "content": content, "source": source, "deletable": deletable})
        except Exception as exc:
            return web.json_response({"success": False, "error": str(exc)}, status=404)

    @_prompt_server.routes.post("/minimax-h3-agent/skills")
    async def upload_or_save_agent_skill(request):
        try:
            data = await request.json()
        except Exception:
            return web.json_response({"success": False, "error": "Expected JSON payload"}, status=400)
        requested_name = str(data.get("filename") or data.get("name") or "").strip()
        filename = _safe_skill_leaf(requested_name)
        if filename is None:
            return web.json_response({"success": False, "error": "A plain filename is required"}, status=400)

        if Path(filename).suffix.lower() not in SKILL_TEXT_EXTENSIONS:
            filename += ".md"

        content_raw = data.get("content")
        if content_raw is not None:
            raw = str(content_raw).encode("utf-8")
        else:
            encoded = str(data.get("content_base64") or "")
            try:
                raw = base64.b64decode(encoded, validate=True)
            except (binascii.Error, ValueError):
                return web.json_response({"success": False, "error": "Invalid Skill content"}, status=400)

        if len(raw) > MAX_SKILL_BYTES * 4:
            return web.json_response({"success": False, "error": "Skill file is too large"}, status=400)

        try:
            raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            return web.json_response({"success": False, "error": "Skill file must be UTF-8 text"}, status=400)

        path = _contained_skill_candidate(_skill_dir(), filename)
        if path is None:
            return web.json_response({"success": False, "error": "Invalid Skill filename"}, status=400)
        path.write_bytes(raw)
        return web.json_response({"success": True, "filename": filename, "skills": _skill_names()})

    @_prompt_server.routes.delete("/minimax-h3-agent/skills")
    async def delete_agent_skill(request):
        filename = ""
        try:
            data = await request.json()
            filename = _safe_skill_leaf(str(data.get("filename") or data.get("name") or "")) or ""
        except Exception:
            filename = _safe_skill_leaf(
                str(request.query.get("filename") or request.query.get("name") or "")) or ""

        if not filename or filename in {"none", "default.md"}:
            return web.json_response({"success": False, "error": "Cannot delete default skill"}, status=400)

        path = _contained_skill_candidate(_skill_dir(), filename)
        if path is None:
            return web.json_response({"success": False, "error": "Invalid Skill filename"}, status=400)
        if path.is_file():
            try:
                path.unlink()
                return web.json_response({"success": True, "filename": filename, "skills": _skill_names()})
            except Exception as exc:
                return web.json_response({"success": False, "error": f"Failed to delete file: {exc}"}, status=500)
        return web.json_response({"success": False, "error": "Skill file not found"}, status=404)

    @_prompt_server.routes.post("/minimax-h3-agent/skills/upload-dir")
    async def upload_skill_directory(request):
        """Import Skills from an administrator-approved local directory.

        The source must be contained by ``MINIMAX_H3_SKILLS_IMPORT_DIR``.
        Packages are copied as validated UTF-8 text only; symlinks, binary files
        and over-sized packages are rejected.
        """
        try:
            data = await request.json()
        except Exception:
            return web.json_response({"success": False, "error": "Expected JSON payload"}, status=400)

        dir_path_str = str(data.get("dir_path") or "").strip()
        if not dir_path_str:
            return web.json_response({"success": False, "error": "dir_path is required"}, status=400)

        try:
            src_dir = _resolve_skill_import_source(dir_path_str)
        except ValueError as exc:
            return web.json_response({"success": False, "error": str(exc)}, status=403)

        dest = _skill_dir()
        imported = []
        skipped = []

        for entry in sorted(src_dir.iterdir()):
            name = entry.name
            if name.startswith(".") or name.startswith("_") or entry.is_symlink():
                continue

            if entry.is_file():
                if entry.suffix.lower() not in SKILL_TEXT_EXTENSIONS:
                    continue
                try:
                    raw = entry.read_bytes()
                    if len(raw) > MAX_SKILL_BYTES * 4:
                        skipped.append(f"{name} (too large)")
                        continue
                    try:
                        raw.decode("utf-8-sig")
                    except UnicodeDecodeError:
                        skipped.append(f"{name} (not UTF-8)")
                        continue
                    target = _contained_skill_candidate(dest, name)
                    if target is None:
                        skipped.append(f"{name} (invalid name)")
                        continue
                    target.write_bytes(raw)
                    imported.append(name)
                except Exception as exc:
                    skipped.append(f"{name} ({exc})")

            elif entry.is_dir():
                has_skill = any(
                    (entry / sf).is_file()
                    for sf in ("SKILL.md", "SKILL.cn.md")
                )
                if not has_skill:
                    continue
                target = _contained_skill_candidate(dest, name)
                if target is None:
                    skipped.append(f"{name}/ (invalid name)")
                    continue
                if target.exists():
                    skipped.append(f"{name}/ (already exists)")
                    continue
                try:
                    package_files = _read_import_package(entry)
                    target.mkdir()
                    for relative, raw in package_files:
                        output = (target / relative).resolve()
                        if not _path_is_within(output, target.resolve()):
                            raise ValueError("package path escaped its destination")
                        output.parent.mkdir(parents=True, exist_ok=True)
                        output.write_bytes(raw)
                    imported.append(f"{name}/")
                except Exception as exc:
                    skipped.append(f"{name}/ ({exc})")

        # Invalidate skill index cache
        _skill_index(force=True)

        return web.json_response({
            "success": True,
            "imported": imported,
            "skipped": skipped,
            "imported_count": len(imported),
            "skills": _skill_names(),
        })

    @_prompt_server.routes.get("/minimax-h3-agent/llm-config")
    async def get_llm_config(request):
        """Return editable LLM configuration without API-key material."""
        try:
            from . import llm_service as _llm
            return web.json_response(_llm.public_config())
        except Exception as exc:
            return web.json_response({"error": str(exc)}, status=500)

    @_prompt_server.routes.post("/minimax-h3-agent/llm-config")
    async def save_llm_config(request):
        """Validate and atomically replace Myang's complete service list."""
        try:
            data = await request.json()
        except Exception:
            return web.json_response({"error": "Expected JSON"}, status=400)

        try:
            from . import llm_service as _llm
            result = _llm.save_public_services(data.get("services"))
            result.update({
                "success": True,
                "service_count": len(result["services"]),
                "llm_options": _llm.llm_service_options(),
                "vlm_options": _llm.vlm_service_options(),
            })
            return web.json_response(result)
        except ValueError as exc:
            return web.json_response({"error": str(exc)}, status=400)
        except Exception as exc:
            return web.json_response({"error": str(exc)}, status=500)

    @_prompt_server.routes.post("/minimax-h3-agent/llm-routes/reset")
    async def reset_llm_route_state(request):
        """Clear local route cooldowns without changing saved API keys."""
        try:
            data = await request.json()
        except Exception:
            data = {}
        try:
            from . import llm_service as _llm
            result = _llm.reset_route_runtime(
                str(data.get("service_id") or data.get("service") or ""),
                str(data.get("route_id") or ""))
            return web.json_response(result)
        except ValueError as exc:
            return web.json_response({"error": str(exc)}, status=400)
        except Exception as exc:
            return web.json_response({"error": str(exc)}, status=500)

    @_prompt_server.routes.post("/minimax-h3-agent/learn")
    async def learn_agent_skills(request):
        """Read Skills and distil them into reusable writing specs.

        With an llm_service the model actually reads each Skill and writes the
        spec that gets injected at runtime. Without one this degrades to
        caching the verbatim text, which still helps but is not learning.
        """
        try:
            data = await request.json()
        except Exception:
            data = {}

        llm_service = str(data.get("llm_service") or "").strip()
        ollama_auto_unload = bool(data.get("ollama_auto_unload", False))

        if data.get("all"):
            targets = [name for name in _skill_names() if name != "none"]
        else:
            name = str(data.get("name") or "").strip()
            if not name or name == "none":
                return web.json_response({"success": False, "error": "No skill selected"}, status=400)
            targets = [name]

        memory = _skill_memory()
        results = []
        for name in targets:
            try:
                learned = await asyncio.to_thread(_learn_skill, name, llm_service, ollama_auto_unload)
            except Exception as exc:  # noqa: BLE001 - report per-skill, keep going
                results.append({"name": name, "success": False, "error": str(exc)})
                continue
            if not learned.get("full_text"):
                results.append({"name": name, "success": False, "error": "empty skill"})
                continue
            memory[name] = learned
            results.append({
                "name": name,
                "success": True,
                "chars": learned["chars"],
                "digest_chars": learned.get("digest_chars", 0),
                "learned_by": learned.get("learned_by", "file"),
                "notes": learned.get("notes", []),
            })
        _write_skill_memory(memory)

        learned_count = sum(1 for item in results if item.get("success"))
        return web.json_response({
            "success": learned_count > 0,
            "learned": learned_count,
            "total": len(targets),
            "llm": bool(llm_service),
            "results": results,
        })

    @_prompt_server.routes.get("/minimax-h3-agent/learn")
    async def learned_agent_skills(request):
        memory = _skill_memory()
        return web.json_response({
            "skills": {
                name: {
                    "chars": entry.get("chars", 0),
                    "digest_chars": entry.get("digest_chars", 0),
                    "learned_by": entry.get("learned_by", "file"),
                }
                for name, entry in memory.items()
            },
        })

    @_prompt_server.routes.get("/minimax-h3-agent/digest")
    async def get_skill_digest(request):
        name = str(request.query.get("name") or "").strip()
        entry = _skill_memory().get(name) or {}
        return web.json_response({
            "name": name,
            "digest": entry.get("digest", ""),
            "learned_by": entry.get("learned_by", ""),
            "chars": entry.get("chars", 0),
            "digest_chars": entry.get("digest_chars", 0),
        })


def _read_skill(skill_name_or_path: Any, pasted_text: str = "", budget: int = 0) -> tuple[str, str]:
    """Load a Skill's text.

    ``budget`` caps how many characters of Skill body may be injected into the
    system prompt; pass 0 (the default) to get the raw text, which is what the
    editor routes want. When a budget is set, reference manuals are only
    included with whatever room is left over -- they are background material and
    routinely dwarf the instructions they support.
    """
    chunks: list[str] = []
    sources: list[str] = []

    pasted = str(pasted_text or "").strip()
    if pasted:
        if len(pasted.encode("utf-8")) > MAX_SKILL_BYTES * 4:
            raise ValueError("Pasted Skill content exceeds limit")
        chunks.append("--- User Custom Paste Rules ---\n" + pasted)
        sources.append("pasted_text")

    name_str = str(skill_name_or_path or "").strip()
    if isinstance(skill_name_or_path, Path):
        target_path = skill_name_or_path
    else:
        target_path = _get_skill_target(name_str)

    if target_path:
        if target_path.is_file():
            text = target_path.read_text(encoding="utf-8-sig", errors="replace").strip()
            if budget:
                text = _trim_skill_text(text, budget)
            if text:
                chunks.append(f"--- Skill File: {target_path.name} ---\n" + text)
                sources.append(target_path.name)
        elif target_path.is_dir():
            skill_md = target_path / "SKILL.cn.md"
            if not skill_md.is_file():
                skill_md = target_path / "SKILL.md"

            main_len = 0
            if skill_md.is_file():
                main_text = skill_md.read_text(encoding="utf-8-sig", errors="replace").strip()
                if budget:
                    main_text = _trim_skill_text(main_text, budget)
                main_len = len(main_text)
                chunks.append(f"--- Official Skill Package: {target_path.name} ({skill_md.name}) ---\n" + main_text)
                sources.append(f"{target_path.name}/{skill_md.name}")

            ref_dir = target_path / "references"
            ref_budget = MAX_SKILL_REFERENCE_CHARS if budget else 0
            if budget and main_len >= budget:
                # The instructions alone already filled the budget.
                ref_budget = 0
            if ref_dir.is_dir() and (not budget or ref_budget):
                ref_texts = []
                used = 0
                for ref_file in sorted(ref_dir.iterdir()):
                    if not ref_file.is_file() or ref_file.suffix.lower() not in {".txt", ".md", ".json"}:
                        continue
                    content = ref_file.read_text(encoding="utf-8-sig", errors="replace").strip()
                    if not content:
                        continue
                    if ref_budget:
                        remaining = ref_budget - used
                        if remaining <= 200:
                            break
                        content = _trim_skill_text(content, remaining)
                        used += len(content)
                    ref_texts.append(f"=== Reference Document: {ref_file.name} ===\n{content}")
                if ref_texts:
                    chunks.append("--- Skill Reference Manuals & Guides ---\n" + "\n\n".join(ref_texts))
                    sources.append(f"{target_path.name}/references ({len(ref_texts)} files)")

    full_skill_text = "\n\n".join(chunks)
    source_str = ", ".join(sources) or "default"
    return full_skill_text, source_str


SKILL_AUTO = "auto"
SKILL_INDEX_FILENAME = "_skill_index.json"
# The official packages run to ~40k characters of SKILL.cn.md plus another ~40k
# of reference manuals. Injecting that whole wall of text into every rewrite is
# what made the system prompt feel "off": it costs a fortune and buries the few
# sections that actually say how to write a prompt.
MAX_SKILL_INJECT_CHARS = 8000
MAX_SKILL_REFERENCE_CHARS = 2000
SKILL_KEEP_HINTS = (
    "提示词", "prompt", "写作", "撰写", "文案", "台词", "对白", "字幕",
    "镜头", "shot", "结构", "structure", "规则", "rule", "要求", "格式",
    "format", "模板", "template", "示例", "example", "风格", "style",
    "视觉", "visual", "音频", "audio", "画面",
)


def _strip_frontmatter(text: str) -> str:
    stripped = str(text or "").lstrip("﻿ \t\r\n")
    if not stripped.startswith("---"):
        return str(text or "").strip()
    end = stripped.find("\n---", 3)
    if end == -1:
        return stripped.strip()
    return stripped[end + 4:].strip()


def _split_sections(text: str) -> list[tuple[str, str]]:
    """Split markdown into (heading, body) pairs, preamble first."""
    parts: list[tuple[str, str]] = []
    heading = ""
    buffer: list[str] = []
    for line in text.splitlines():
        if line.startswith("## "):
            parts.append((heading, "\n".join(buffer).strip()))
            heading = line.strip()
            buffer = []
        else:
            buffer.append(line)
    parts.append((heading, "\n".join(buffer).strip()))
    return [(head, body) for head, body in parts if head or body]


def _trim_skill_text(text: str, budget: int) -> str:
    """Fit a Skill into a character budget without shredding its instructions.

    Sections whose heading looks like prompt-writing guidance are kept first,
    then whatever else still fits, so an over-long Skill degrades by dropping
    pipeline/tooling chapters rather than by truncating mid-sentence.
    """
    body = _strip_frontmatter(text)
    if budget <= 0 or len(body) <= budget:
        return body

    sections = _split_sections(body)
    if len(sections) <= 1:
        return body[:budget].rstrip() + "\n\n[... 已截断以节省 token ...]"

    preferred: list[int] = []
    rest: list[int] = []
    for index, (heading, _) in enumerate(sections):
        low = heading.lower()
        is_writing = index == 0 or any(hint in low for hint in SKILL_KEEP_HINTS)
        (preferred if is_writing else rest).append(index)

    chosen: set[int] = set()
    used = 0
    for index in preferred + rest:
        heading, chunk = sections[index]
        size = len(heading) + len(chunk) + 2
        if used + size > budget:
            continue
        chosen.add(index)
        used += size

    if not chosen:
        return body[:budget].rstrip() + "\n\n[... 已截断以节省 token ...]"

    out = "\n\n".join(f"{sections[i][0]}\n{sections[i][1]}".strip() for i in sorted(chosen))
    dropped = len(sections) - len(chosen)
    if dropped:
        out += f"\n\n[... 已省略 {dropped} 个非写作章节以节省 token ...]"
    return out


def _read_meta_scalars(path: Path) -> dict[str, str]:
    """Read the flat `key: value` pairs from a Skill meta.yaml.

    Only single-line scalars are needed (display name, tag, summary), so this
    stays a few lines instead of taking a hard dependency on a YAML parser.
    """
    data: dict[str, str] = {}
    try:
        raw_text = path.read_text(encoding="utf-8-sig", errors="replace")
    except OSError:
        return data
    for raw in raw_text.splitlines():
        if not raw[:1] or raw[:1] in {" ", "\t", "-", "#"}:
            continue
        key, sep, value = raw.partition(":")
        if not sep:
            continue
        value = value.strip().strip('"').strip("'")
        if value:
            data[key.strip()] = value
    return data


def _frontmatter_description(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8-sig", errors="replace")
    except OSError:
        return ""
    stripped = text.lstrip("﻿ \t\r\n")
    if not stripped.startswith("---"):
        return ""
    end = stripped.find("\n---", 3)
    lines = stripped[3:end if end != -1 else len(stripped)].splitlines()
    for index, line in enumerate(lines):
        key, sep, value = line.partition(":")
        if not sep or key.strip() != "description":
            continue
        value = value.strip()
        if value and value not in {"|", ">", "|-", ">-", "|+", ">+"}:
            return value
        folded = []
        for follow in lines[index + 1:]:
            if follow.strip() and follow[:1] not in {" ", "\t"}:
                break
            folded.append(follow.strip())
        return " ".join(part for part in folded if part).strip()
    return ""


def _skill_signature() -> str:
    parts: list[str] = []
    for name in _skill_names():
        if name == "none":
            continue
        target = _get_skill_target(name)
        if target is None:
            continue
        try:
            parts.append(f"{name}:{int(target.stat().st_mtime)}")
        except OSError:
            parts.append(f"{name}:0")
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def _build_skill_index() -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    for name in _skill_names():
        if name == "none":
            continue
        target = _get_skill_target(name)
        if target is None:
            continue
        display, tag, summary = name, "", ""
        if target.is_dir():
            meta = _read_meta_scalars(target / "meta.yaml") or _read_meta_scalars(target / "meta.yml")
            display = meta.get("display-name-zh") or meta.get("display-name-en") or name
            tag = meta.get("tag-cn") or meta.get("tag-en") or ""
            summary = meta.get("summary-cn") or meta.get("summary-en") or ""
            if not summary:
                for candidate in ("SKILL.cn.md", "SKILL.md"):
                    summary = _frontmatter_description(target / candidate)
                    if summary:
                        break
        else:
            display = target.stem
            summary = _frontmatter_description(target)
        entries.append({
            "name": name,
            "display": " ".join(str(display).split())[:60],
            "tag": " ".join(str(tag).split())[:40],
            "summary": " ".join(str(summary).split())[:200],
        })
    return entries


def _skill_index(force: bool = False) -> list[dict[str, str]]:
    """Cached one-line-per-Skill catalogue that drives `auto` selection.

    Persisting it is what keeps auto-select cheap: the router sees a few hundred
    tokens of names, tags and summaries instead of re-reading tens of thousands
    of characters of Skill text. Rebuilt whenever a Skill file's mtime changes.
    """
    cache_path = _skill_dir() / SKILL_INDEX_FILENAME
    signature = _skill_signature()
    if not force and cache_path.is_file():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8-sig"))
            if isinstance(cached, dict) and cached.get("signature") == signature:
                entries = cached.get("entries")
                if isinstance(entries, list):
                    return entries
        except (OSError, ValueError):
            pass
    entries = _build_skill_index()
    try:
        cache_path.write_text(
            json.dumps({"signature": signature, "entries": entries}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError:
        pass
    return entries


SKILL_MEMORY_FILENAME = "_skill_memory.json"
SKILL_DIGEST_CHUNK = 18000
SKILL_PROFILE_VERSION = 2
SKILL_PROFILE_KEYS = (
    "output_structure",
    "shot_format",
    "duration_rules",
    "media_rules",
    "shot_detail_dimensions",
    "dialogue_rules",
    "audio_rules",
    "density_rules",
    "prohibitions",
)
SKILL_PROFILE_HEADINGS = {
    "output_structure": "输出结构",
    "shot_format": "分镜格式",
    "duration_rules": "时长规则",
    "media_rules": "素材标签与引用",
    "shot_detail_dimensions": "每镜必写维度",
    "dialogue_rules": "语言、台词与说话人",
    "audio_rules": "环境声与配乐分工",
    "density_rules": "细节密度",
    "prohibitions": "禁止事项",
}

SKILL_DIGEST_SYSTEM = """你是技能分析师。你会读到一份视频提示词写作技能文档（可能是其中一段），请把它压缩成一份可直接执行的写作规范。

只输出规范正文，不要 Markdown 代码块、不要“以下是”之类的开场白。按这个顺序写，没有对应内容的小节直接省略：

【输出结构】输出必须包含哪些部分、字段名逐字写出、先后顺序。
【分镜格式】镜头标记怎么写、时间怎么标注（几位小数、什么分隔符）。
【时长规则】镜头数与时长的关系。文档里出现的具体秒数如果只是示例，必须标注“（示例，按实际时长重排）”。
【素材标签】<Picture N>/<Video N>/<Audio N> 的用法和对齐声明写法。
【必写要素】每个镜头必须交代的内容。
【语言与台词】用什么语言写、台词怎么包装。
【禁止事项】明确不能做的事。

要求：
- 保留原文的字段名、标记、术语原样，不要翻译或改写它们。
- 写成可直接照做的祈使句，不要复述文档结构，不要写“本文档介绍了……”。
- 如果这段文档只是索引/指路（例如“详见 references/xxx”），就写出它指向什么，不要虚构内容。
- 控制在 1200 字以内。
"""

SKILL_MERGE_SYSTEM = """你是技能分析师。你会读到同一个技能文档拆分学习后得到的多份规范草稿，请合并成一份最终写作规范。

只输出规范正文，不要 Markdown 代码块、不要开场白。保持这个小节顺序，没有内容的小节省略：
【输出结构】【分镜格式】【时长规则】【素材标签】【必写要素】【语言与台词】【禁止事项】

要求：
- 去重合并，冲突时以更具体、更可执行的写法为准。
- 字段名、镜头标记、术语一律保留原样。
- 具体秒数若是文档示例，标注“（示例，按实际时长重排）”。
- 控制在 1800 字以内。
"""


def _parse_skill_profile(raw: str) -> dict[str, list[str]] | None:
    value = _sanitize_llm_output(raw)
    match = re.search(r"\{[\s\S]*\}", value)
    if match:
        value = match.group(0)
    try:
        data = json.loads(value)
    except (TypeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    profile: dict[str, list[str]] = {}
    for key in SKILL_PROFILE_KEYS:
        source = data.get(key) or []
        if isinstance(source, str):
            source = [source]
        if not isinstance(source, list):
            source = []
        cleaned = []
        for item in source:
            text = re.sub(r"\s+", " ", str(item or "")).strip(" -•\t\r\n")
            if text:
                cleaned.append(text[:600])
        profile[key] = cleaned[:12]
    return profile if any(profile.values()) else None


def _merge_skill_profiles(profiles: list[dict[str, list[str]]]) -> dict[str, list[str]]:
    """Loss-aware local merge: union executable rules without another LLM pass."""
    merged = {key: [] for key in SKILL_PROFILE_KEYS}
    seen = {key: set() for key in SKILL_PROFILE_KEYS}
    for profile in profiles:
        for key in SKILL_PROFILE_KEYS:
            for item in profile.get(key) or []:
                normalized = re.sub(r"[\s，。；;,.]+", "", item).casefold()
                if not normalized or normalized in seen[key]:
                    continue
                seen[key].add(normalized)
                merged[key].append(item)
    return merged


def _render_skill_profile(profile: dict[str, list[str]]) -> str:
    lines = [f"【结构化技能画像 v{SKILL_PROFILE_VERSION}｜逐条执行，不得再摘要】"]
    for key in SKILL_PROFILE_KEYS:
        items = profile.get(key) or []
        if not items:
            continue
        lines.append(f"【{SKILL_PROFILE_HEADINGS[key]}】")
        lines.extend("- " + item for item in items)
    return "\n".join(lines)


def _h3_source_skill_profile(full_text: str) -> dict[str, list[str]] | None:
    """Compile the official H3 guide deterministically, without lossy relearning."""
    source = str(full_text or "")
    lowered = source.casefold()
    required = (
        "integrated_multimodal_description",
        "overall_soundscape",
        "non_diegetic_music",
        "detailed_description",
        "retention_analysis",
    )
    if not all(token in lowered for token in required):
        return None
    return {
        "output_structure": [
            "基础 T2VA/I2VA/FL2VA/L2VA 使用 integrated_multimodal_description → overall_soundscape → non_diegetic_music，字段名和顺序不可改变。",
            "仅在 Ref2VA/full-reference 模式使用 subject_definitions → summary → retention_analysis → detailed_description → overall_soundscape → non_diegetic_music。",
            "rewrite sections 使用英文；只让台词、歌词和屏幕可见文字保留原语言。",
        ],
        "shot_format": [
            "第一镜写 [Shot 1] 且不加时间戳。",
            "第二镜起严格写 [Shot N] At MM:SS.mmm, ...，镜头编号连续、时间严格递增且小于目标时长。",
            "普通切镜使用 the camera cuts to / the shot cuts to 等自然句式；仅在用户明确要求时使用淡入淡出或擦除。",
        ],
        "duration_rules": [
            "以本次真实目标时长重新分配镜头；技能示例中的镜头数和秒数只示范格式，不得照抄。",
            "完整覆盖从 0 秒到目标结尾，动作、台词和切点必须能在对应镜头时长内完成。",
        ],
        "media_rules": [
            "I2VA/FL2VA/L2VA 只在图片确实承担首帧或尾帧关键帧职责时使用官方对齐声明。",
            "普通人物外观参考在人物实际出现的镜头正文中保留对应 <Picture N>/@图片N，不把它误写成 0 秒构图锚点。",
            "Ref2VA 先在 subject_definitions 定义 Subject/Picture/Video/Audio 的职责，再在实际生效镜头和 retention_analysis 中复用同一标签含义。",
            "只引用实际可用的素材标签，不创建未定义标签，不用纯文字替代已指定的角色参考标签。",
        ],
        "shot_detail_dimensions": [
            "每个镜头分别写清当前 composition、shot size、camera angle，以及主体在前景/中景/背景中的位置和空间关系。",
            "写清每个主体可见的身份外观、服装、姿态、视线目标和进入镜头时的状态，不用剧情概述替代。",
            "写清 environment、关键道具、光源方向、软硬、色温、阴影、高光和材质响应，并维持跨镜连续性。",
            "把主要动作展开为准备与重心变化 → 动作启动 → 接触/受力 → 惯性与次级运动 → 减速、结果和人物反应。",
            "把 camera movement 写成自然英文动作，明确运动类型、目标，并在有意义时给出幅度和速度。",
            "写清表情触发、眼神移动、眉眼/嘴角/下颌变化、呼吸和身体反应，避免突然换表情。",
            "把同步台词、动作声和参考素材真正出现或生效的时点写进对应镜头，不只写到总结栏。",
        ],
        "dialogue_rules": [
            "真实发声者按首次发声顺序分配稳定 (S1)/(S2)，跨镜保持相同 ID；无发声角色不分配 ID。",
            "台词写成 <d>[Chinese] 原文</d> 等格式，逐字保留用户原文和标点，不翻译、不改写。",
            "画外音使用 says in an off-screen voiceover，并明确画面人物嘴唇保持闭合。",
            "跨切镜台词使用 <scenetrans> 并说明声音连续；结尾被截断时使用 <cutoff>。",
            "直接复用且仅归属 <Audio N> 的完整音轨不虚构额外 (Sx)；具体人物实际发声时必须使用 (Sx)。",
        ],
        "audio_rules": [
            "把与动作同步的脚步、碰撞、布料、呼吸和台词写在 integrated_multimodal_description/detailed_description 的对应镜头。",
            "overall_soundscape 只用连续英文段落总结环境声、物理动作声和非语言人声，不重复台词，也不混入观众配乐。",
            "non_diegetic_music 只描述观众能听到的配乐，写明乐器、速度/节拍和动态变化；没有配乐写 N/A。",
        ],
        "density_rules": [
            "基础模式按时长提供足够的逐镜可见细节，不能压缩成剧情提纲、关键词清单或每镜一句话。",
            "Ref2VA 生成任务的 detailed_description 通常为 350–500 个英文词；短分段按实际承载能力缩放，但不可删除每镜必写维度。",
            "需要压缩时先删重复画质同义词和次要装饰，最后才压缩环境细节；标签定义、参考生效点、关键动作、台词和声音关系不可删除。",
        ],
        "prohibitions": [
            "不得输出剧情摘要、未解析标签、与目标时长不匹配的时间线或堆叠式 camera 标签。",
            "不得把 dialogue、singing 或 diegetic music 重复写入 overall_soundscape/non_diegetic_music。",
            "不得凭空改变人物身份、五官、发型、服装、饰品、产品几何、Logo 或场景结构；用户明确要求的变化必须执行。",
            "不得把环境声写入 non_diegetic_music，也不得只用 cheerful/healing/cinematic 等抽象情绪词代替乐器、节拍和动态。",
        ],
    }


def _local_skill_chunk_digest(chunk: str, index: int, total: int) -> str:
    """Keep failed chunks useful and compact until the remote learner resumes."""
    important = re.compile(
        r"(?:必须|不得|禁止|需要|应该|输出|格式|结构|镜头|时长|素材|标签|"
        r"台词|声音|动作|运镜|光线|连续|保留|must|should|never|format|shot)",
        re.IGNORECASE,
    )
    selected: list[str] = []
    chars = 0
    for raw_line in str(chunk or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        is_rule = (
            line.startswith(("#", "-", "*", ">", "【"))
            or bool(re.match(r"^(?:\d+[.)、]|[一二三四五六七八九十]+[、.])", line))
            or bool(important.search(line))
        )
        if not is_rule:
            continue
        line = line[:600]
        if chars + len(line) + 1 > 5000:
            break
        selected.append(line)
        chars += len(line) + 1
    if chars < 500:
        selected.append(str(chunk or "").strip()[: max(0, 1800 - chars)])
    body = "\n".join(item for item in selected if item).strip()
    return f"【第 {index}/{total} 段本地规则保留·待续学】\n{body}".strip()


def _compact_learning_error(exc: BaseException) -> str:
    message = str(exc or "").split("【解决建议】", 1)[0].strip()
    return re.sub(r"\s+", " ", message)[:360]


def _digest_skill_text(
    name: str,
    full_text: str,
    llm_service: str,
    ollama_auto_unload: bool,
    prior_progress: dict[str, Any] | None = None,
    progress_out: dict[str, Any] | None = None,
) -> tuple[str, list[str]]:
    """Have the LLM read a Skill and write a compact, executable spec.

    This is the part that makes "learning" mean something. A raw Skill package
    is 20k-56k characters of prose, worked examples and pipeline chapters, and
    injecting all of it every run both costs a fortune and buries the few rules
    that actually govern the output. Reading it once and keeping the distilled
    rules gives the writer a denser, shorter brief.

    Long Skills are distilled chunk by chunk and then merged, so a document
    larger than the model's context still gets covered end to end instead of
    being silently truncated.
    """
    text = str(full_text or "").strip()
    if not text:
        return "", ["技能内容为空"]

    notes: list[str] = []
    chunks = [text[i:i + SKILL_DIGEST_CHUNK]
              for i in range(0, len(text), SKILL_DIGEST_CHUNK)]
    source_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    prior = prior_progress if isinstance(prior_progress, dict) else {}
    can_resume = (
        prior.get("source_hash") == source_hash
        and int(prior.get("chunk_size") or 0) == SKILL_DIGEST_CHUNK
        and int(prior.get("total_chunks") or 0) == len(chunks)
        and str(prior.get("llm_service") or "") == str(llm_service or "")
    )
    old_chunks = prior.get("chunks") if can_resume else []
    old_chunks = old_chunks if isinstance(old_chunks, list) else []
    records: list[dict[str, Any]] = []
    drafts: list[str] = []
    remote_halted = ""
    for index, chunk in enumerate(chunks, 1):
        label = f"{name} 第 {index}/{len(chunks)} 段"
        old = old_chunks[index - 1] if index <= len(old_chunks) else {}
        if isinstance(old, dict) and old.get("status") == "complete" and old.get("digest"):
            draft = str(old.get("digest") or "").strip()
            drafts.append(draft)
            records.append({"index": index, "status": "complete", "digest": draft})
            notes.append(f"{label} 已复用上次成功结果")
            continue

        if remote_halted:
            fallback = _local_skill_chunk_digest(chunk, index, len(chunks))
            drafts.append(fallback)
            records.append({
                "index": index,
                "status": "pending",
                "fallback": fallback,
                "reason": remote_halted,
            })
            continue

        question = f"技能文档（{label}）：\n\n{chunk}\n\n请输出这一段的写作规范。"
        try:
            draft = _expand_with_prompt_assistant(
                llm_service, question, SKILL_DIGEST_SYSTEM,
                ollama_auto_unload, 0)
        except Exception as exc:  # noqa: BLE001 - partial learning beats none
            if _is_interrupt(exc):
                raise
            kind = (
                "quota" if _is_quota_error(exc)
                else "rate_limit" if _is_rate_limit_error(exc)
                else "empty" if _is_empty_response_error(exc)
                else "error"
            )
            notes.append(f"{label} 远程学习未完成: {_compact_learning_error(exc)}")
            fallback = _local_skill_chunk_digest(chunk, index, len(chunks))
            drafts.append(fallback)
            records.append({
                "index": index,
                "status": "pending",
                "fallback": fallback,
                "reason": kind,
            })
            if kind in {"quota", "rate_limit", "empty"}:
                remote_halted = kind
            continue
        draft = _sanitize_llm_output(draft)
        if draft:
            drafts.append(draft)
            records.append({"index": index, "status": "complete", "digest": draft})
        else:
            fallback = _local_skill_chunk_digest(chunk, index, len(chunks))
            drafts.append(fallback)
            records.append({
                "index": index,
                "status": "pending",
                "fallback": fallback,
                "reason": "empty",
            })
            remote_halted = "empty"

    if not drafts:
        return "", notes + ["技能没有可保留的内容"]

    completed = sum(1 for record in records if record.get("status") == "complete")
    pending = max(0, len(chunks) - completed)
    if pending:
        notes.append(
            f"已保留学习进度 {completed}/{len(chunks)}；待续学 {pending} 段。"
            "下次点击学习只重试未完成分块，不会重学已成功部分")

    merge_complete = len(drafts) == 1 and pending == 0
    final_digest = drafts[0] if len(drafts) == 1 else ""
    if pending:
        final_digest = "\n\n".join(drafts)
    elif len(drafts) > 1:
        merged_input = "\n\n---\n\n".join(
            f"草稿 {i}:\n{draft}" for i, draft in enumerate(drafts, 1))
        try:
            merged = _expand_with_prompt_assistant(
                llm_service, merged_input, SKILL_MERGE_SYSTEM,
                ollama_auto_unload, 0)
        except Exception as exc:  # noqa: BLE001 - concatenation is still usable
            notes.append(f"合并未完成，已保留分块草稿: {_compact_learning_error(exc)}")
            final_digest = "\n\n".join(drafts)
        else:
            merged = _sanitize_llm_output(merged)
            if merged:
                final_digest = merged
                merge_complete = True
            else:
                notes.append("合并返回空结果，已保留分块草稿")
                final_digest = "\n\n".join(drafts)

    if progress_out is not None:
        progress_out.update({
            "version": 1,
            "source_hash": source_hash,
            "chunk_size": SKILL_DIGEST_CHUNK,
            "total_chunks": len(chunks),
            "completed_chunks": completed,
            "pending_chunks": pending,
            "llm_service": str(llm_service or ""),
            "merge_complete": bool(merge_complete),
            "chunks": records,
        })
    return final_digest, notes


def _learn_skill(name: str, llm_service: str = "", ollama_auto_unload: bool = False) -> dict[str, Any]:
    """Read one Skill and record it in the memory file.

    Always caches the verbatim text. When an llm_service is supplied it also
    stores an LLM-written digest, which is what the writer actually receives at
    runtime; the full text stays as the fallback.
    """
    text, source = _read_skill(name, budget=0)
    prior_entry = _skill_memory().get(name) or {}
    entry: dict[str, Any] = {
        "name": name,
        "full_text": text,
        "chars": len(text),
        "source": source,
        "digest": "",
        "digest_chars": 0,
        "profile_version": 0,
        "learned_by": "file",
        "notes": [],
        "learning_progress": {},
        "completed_chunks": 0,
        "pending_chunks": 0,
    }
    if text and llm_service:
        progress: dict[str, Any] = {}
        digest, notes = _digest_skill_text(
            name,
            text,
            llm_service,
            ollama_auto_unload,
            prior_progress=(prior_entry.get("learning_progress")
                            if isinstance(prior_entry, dict) else None),
            progress_out=progress,
        )
        entry["notes"] = notes
        entry["learning_progress"] = progress
        entry["completed_chunks"] = int(progress.get("completed_chunks") or 0)
        entry["pending_chunks"] = int(progress.get("pending_chunks") or 0)
        if digest:
            entry["digest"] = digest
            entry["digest_chars"] = len(digest)
            entry["profile_version"] = 0
            entry["learned_by"] = (
                "llm" if (not progress or (
                    not entry["pending_chunks"] and progress.get("merge_complete")))
                else "llm_partial"
            )
    return entry


def _skill_memory(force: bool = False) -> dict[str, dict[str, Any]]:
    """Learned Skill texts, keyed by Skill name.

    Shares _skill_signature() with the index cache, so editing any Skill file
    invalidates both: a stale full-text memory would silently keep feeding the
    old writing rules to the LLM.
    """
    cache_path = _skill_dir() / SKILL_MEMORY_FILENAME
    signature = _skill_signature()
    if not force and cache_path.is_file():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8-sig"))
            if isinstance(cached, dict) and cached.get("signature") == signature:
                skills = cached.get("skills")
                if isinstance(skills, dict):
                    # Older builds marked a one-chunk digest as fully learned
                    # even when a later chunk failed. Do not keep injecting an
                    # incomplete half-Skill; the next manual learn will create
                    # resumable chunk state.
                    for entry in skills.values():
                        if not isinstance(entry, dict) or entry.get("learning_progress"):
                            continue
                        old_notes = " ".join(str(item) for item in (entry.get("notes") or []))
                        if entry.get("learned_by") == "llm" and "学习失败" in old_notes:
                            entry["digest"] = ""
                            entry["digest_chars"] = 0
                            entry["learned_by"] = "file"
                            entry["notes"] = list(entry.get("notes") or []) + [
                                "检测到旧版不完整分块摘要，运行时已回退全文；重新学习后将启用断点续学"
                            ]
                    return skills
        except (OSError, ValueError):
            pass
    return {}


def _write_skill_memory(skills: dict[str, dict[str, Any]]) -> None:
    cache_path = _skill_dir() / SKILL_MEMORY_FILENAME
    try:
        cache_path.write_text(
            json.dumps(
                {"signature": _skill_signature(), "skills": skills},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    except OSError as exc:
        logger.warning("[MiniMax H3 Agent] could not persist skill memory: %s", exc)


def _learned_skill_text(
    name: str,
    auto_learn: bool = True,
    budget: int = 0,
) -> tuple[str, str]:
    """Return a Skill's remembered rules and how they were obtained.

    Prefers the LLM-written digest: it is an order of magnitude smaller than the
    package and carries only the rules that shape the output, so the writer's
    attention is not spread across pipeline chapters it cannot act on. Falls
    back to the verbatim text, then to "" so the caller can use the budgeted
    read.
    """
    if not name or name == "none":
        return "", ""
    memory = _skill_memory()
    entry = memory.get(name)
    if not isinstance(entry, dict) and auto_learn:
        try:
            # File-only learning: an automatic fallback must not silently spend
            # LLM calls the user did not ask for.
            learned = _learn_skill(name)
        except Exception as exc:  # noqa: BLE001 - a bad Skill must not break generation
            logger.warning("[MiniMax H3 Agent] auto-learn failed for %r: %s", name, exc)
            return "", ""
        if learned.get("full_text"):
            memory[name] = learned
            _write_skill_memory(memory)
            entry = learned

    if not isinstance(entry, dict):
        return "", ""
    digest = str(entry.get("digest") or "")
    full_text = str(entry.get("full_text") or "")
    if digest:
        progress = entry.get("learning_progress") or {}
        pending = int(progress.get("pending_chunks") or 0)
        if pending:
            completed = int(progress.get("completed_chunks") or 0)
            total = int(progress.get("total_chunks") or completed + pending)
            return digest, f"{name} [Media Agent续学中 {completed}/{total} · {len(digest)}字]"
        return digest, f"{name} [Media Agent学习 {len(digest)}字]"
    if full_text:
        runtime_text = _trim_skill_text(full_text, budget) if budget else full_text
        label = (
            f"{name} [运行时节选 {len(runtime_text)}/{len(full_text)}字]"
            if len(runtime_text) < len(full_text)
            else f"{name} [全文 {len(full_text)}字]"
        )
        return runtime_text, label
    return "", ""


SKILL_ROUTER_SYSTEM = """你是 MiniMax H3 的技能路由器。根据用户的视频提示词，从候选技能中挑出最合适的一个。
依据技能的名称、分类标签和用途摘要判断。若没有明显匹配的技能，输出 none。
只输出一个 name 字段的原文（或 none），不要输出解释、标点或引号。"""


def _select_skill_auto(llm_service: str, user_prompt: str, ollama_auto_unload: bool) -> tuple[str, str]:
    """Pick a Skill from the cached index with one short LLM call."""
    index = _skill_index()
    if not index:
        return "none", "no skills available"
    listing = "\n".join(
        f"- {entry['name']} | {entry.get('display', '')} | {entry.get('tag', '')} | {entry.get('summary', '')}"
        for entry in index
    )
    question = (
        f"候选技能：\n{listing}\n\n"
        f"用户提示词：\n{str(user_prompt or '')[:600]}\n\n"
        "输出最合适的一个 name："
    )
    try:
        raw, _notes = _call_llm_resilient(
            llm_service, [(question, SKILL_ROUTER_SYSTEM)], ollama_auto_unload, 0, "路由"
        )
        if not raw:
            return "none", "auto: 路由无响应"
    except Exception as exc:  # noqa: BLE001 - routing must never break generation
        if _is_interrupt(exc):
            raise
        logger.warning("[MiniMax H3 Agent] auto skill selection failed: %s", exc)
        return "none", f"auto failed ({exc})"

    answer = str(raw or "").strip().strip("`").strip().strip('"\'').strip()
    answer = answer.splitlines()[-1].strip() if answer else ""
    if not answer or answer.lower() == "none":
        return "none", "auto: no match"

    names = [entry["name"] for entry in index]
    for name in names:
        if answer == name:
            return name, f"auto: {name}"
    lowered = answer.lower()
    for name in names:
        if name.lower() == lowered or name.lower() in lowered:
            return name, f"auto: {name}"
    logger.warning("[MiniMax H3 Agent] auto skill selection returned unknown name: %r", answer)
    return "none", f"auto: unrecognised reply ({answer[:40]})"


def _custom_nodes_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def _llm_service_options() -> list[str]:
    try:
        from . import llm_service as _llm
        return _llm.llm_service_options()
    except Exception as exc:
        logger.warning("Unable to read LLM services: %s", exc)
        return ["zhipu/glm-4-flash"]


def _vlm_service_options() -> list[str]:
    try:
        from . import llm_service as _llm
        return _llm.vlm_service_options()
    except Exception as exc:
        logger.warning("Unable to read VLM services: %s", exc)
        return ["off"]


def _describe_media_items(
    manifest: list[dict[str, Any]],
    assets: list[MyangMediaAsset],
    vlm_service: str,
    ollama_auto_unload: bool,
) -> list[str]:
    """Look at each connected image/video with a VLM and attach descriptions.

    Returns a list of human-readable errors for media that could not be
    described; failures degrade gracefully so the agent can still run with a
    metadata-only whitelist.
    """
    from . import llm_service as _llm

    asset_by_slot = {asset.slot: asset for asset in assets}
    errors: list[str] = []
    for entry in manifest:
        asset = asset_by_slot.get(int(entry.get("slot") or -1))
        if asset is None:
            continue
        tag = str(entry["tag"])
        kind = str(entry["type"])
        try:
            if kind == "image":
                img_tensor = image_batch(asset.payload)
                description = _llm.call_vlm(
                    vlm_service,
                    [_llm.tensor_to_base64(img_tensor[:1])],
                    IMAGE_DESCRIBE_PROMPT,
                    ollama_auto_unload,
                )
            elif kind == "video":
                frames, _soundtrack, _fps = video_stream(asset.payload)
                total = int(frames.shape[0])
                if total <= 0:
                    raise ValueError("Reference video has no frames")
                if total <= MAX_VIDEO_FRAMES:
                    indexes = list(range(total))
                else:
                    step = (total - 1) / (MAX_VIDEO_FRAMES - 1)
                    indexes = sorted({int(round(i * step)) for i in range(MAX_VIDEO_FRAMES)})
                img_b64_list = [_llm.tensor_to_base64(frames[idx:idx + 1]) for idx in indexes]
                description = _llm.call_vlm(
                    vlm_service,
                    img_b64_list,
                    VIDEO_DESCRIBE_PROMPT,
                    ollama_auto_unload,
                )
            else:
                continue
        except Exception as exc:
            logger.warning("MiniMax H3 Media Agent: VLM description failed for %s: %s", tag, exc)
            errors.append(f"{tag}: {exc}")
            continue
        if description:
            entry["description"] = description[:MAX_DESCRIPTION_CHARS]
        else:
            errors.append(f"{tag}: empty description")
    return errors


def _media_manifest(assets: list[MyangMediaAsset]) -> list[dict[str, Any]]:
    """Describe the same deterministic order consumed by H3Condition."""
    catalog = MyangMediaCatalog(tuple(assets))
    manifest: list[dict[str, Any]] = []
    counts = {"image": 0, "video": 0, "audio": 0}
    tags = {"image": "Picture", "video": "Video", "audio": "Audio"}
    for asset in catalog.ordered():
        counts[asset.kind] += 1
        ordinal = counts[asset.kind]
        entry = {
            "tag": f"<{tags[asset.kind]} {ordinal}>",
            "type": asset.kind,
            "ordinal": ordinal,
            "slot": asset.slot,
        }
        if asset.filename:
            entry["filename"] = asset.filename
        if asset.label:
            entry["label"] = asset.label
            entry["subject_name"] = asset.label[:40]
        if asset.origin:
            entry["origin"] = asset.origin
        try:
            if asset.kind == "image":
                frames = image_batch(asset.payload)
                entry["shape"] = list(frames.shape)
                entry["resolution"] = f"{frames.shape[2]}x{frames.shape[1]}"
            elif asset.kind == "video":
                frames, soundtrack, fps = video_stream(asset.payload)
                entry["shape"] = list(frames.shape)
                entry["resolution"] = f"{frames.shape[2]}x{frames.shape[1]}"
                entry["frame_count"] = int(frames.shape[0])
                entry["fps"] = fps
                entry["duration"] = round(frames.shape[0] / max(1.0, fps), 1)
                entry["has_audio"] = soundtrack is not None
            else:
                audio = audio_track(asset.payload)
                waveform = audio.get("waveform") if audio else None
                sample_rate = int(audio.get("sample_rate") or 32000) if audio else 32000
                if waveform is not None and sample_rate > 0:
                    entry["duration"] = round(float(waveform.shape[-1]) / sample_rate, 2)
                    entry["sample_rate"] = sample_rate
        except Exception:
            pass
        manifest.append(entry)
    return manifest


def _manifest_text(manifest: list[dict[str, Any]]) -> str:
    if not manifest:
        return "AVAILABLE MEDIA:\n(none)\n"
    lines = ["AVAILABLE MEDIA (complete whitelist):"]
    for item in manifest:
        details = []
        if item.get("filename"):
            details.append(f"filename={item['filename']}")
        elif item.get("label"):
            details.append(f"label={item['label']}")
        if item.get("resolution"):
            details.append(f"resolution={item['resolution']}")
        if item.get("frame_count") is not None:
            details.append(f"frames={item['frame_count']}")
        if item.get("duration") is not None:
            details.append(f"duration={item['duration']}s")
        suffix = f" ({'; '.join(details)})" if details else ""
        lines.append(f"- {item['tag']}: {item['type']}{suffix}")
        if item.get("subject_name"):
            lines.append(f"  subject_name: {item['subject_name']}")
        if item.get("description"):
            lines.append(f"  content: {item['description']}")
        if item.get("transcript"):
            lines.append(f"  speech: {item['transcript']}")
            lines.append("  (这是该音频的真实语音内容，台词必须与它一致，不要另编台词。)")
        if item.get("subject"):
            lines.append(f"  subject: {item['subject']}")
    # filename/label only became visible to the model once the media metadata
    # actually reached the backend, so spell out that they are identifiers -
    # echoing one back would fail the strict media check.
    lines.append("(filename/label 仅用于辨认素材，禁止出现在输出中；引用素材只能逐字使用上面的标签。)")
    if any(item.get("subject_name") for item in manifest):
        lines.append(
            "(subject_name 是用户为该素材指定的主体名称，可以在正文中直接使用该名称指代人物或物体，"
            "首次出现时建议写成「主体名称（<标签>）」的形式，之后可只用名称；"
            "不同镜头引用同一主体时必须使用同一个名称。)"
        )
    return "\n".join(lines) + "\n"


def _alias_key(value: str) -> str:
    return re.sub(r"[\s_\-]+", "", str(value or "").strip().lstrip("@").casefold())


def _editor_aliases(manifest: list[dict[str, Any]]) -> dict[str, str]:
    aliases: dict[str, str] = {}

    def put(alias: str, tag: str):
        key = _alias_key(alias)
        if key and key not in aliases:
            aliases[key] = tag

    for item in manifest:
        tag = str(item["tag"])
        kind = str(item["type"])
        ordinal = int(item["ordinal"])
        slot = int(item.get("slot") or 0)
        bare_tag = tag.strip("<>")
        put(bare_tag, tag)
        put(bare_tag.replace(" ", ""), tag)
        names = {
            "image": ("Picture", "Pic", "Image", "Img", "图片", "图"),
            "video": ("Video", "Vid", "视频", "影片", "视"),
            "audio": ("Audio", "Aud", "音频", "声音", "音"),
        }.get(kind, ("Picture", "Video", "Audio"))
        for name in names:
            put(f"{name}{ordinal}", tag)
            put(f"{name} {ordinal}", tag)
            if slot > 0:
                put(f"{name}{slot}", tag)
                put(f"{name} {slot}", tag)
        if slot > 0:
            put(f"asset{slot}", tag)
            put(f"asset {slot}", tag)

        for key in ("label", "filename"):
            value = str(item.get(key) or "").strip()
            if not value:
                continue
            put(value, tag)
            put(Path(value).stem, tag)
    return aliases


def _normalize_editor_syntax(text: str, manifest: list[dict[str, Any]]) -> str:
    aliases = _editor_aliases(manifest)

    def replace_mention(match):
        return aliases.get(_alias_key(match.group(1)), match.group(0))

    normalized = EDITOR_INDEX_RE.sub(
        lambda match: aliases.get(_alias_key(match.group(1) + match.group(2)), match.group(0)),
        str(text or ""),
    )
    normalized = EDITOR_AT_RE.sub(replace_mention, normalized)
    normalized = DIALOGUE_LINE_RE.sub(
        lambda match: f"{match.group(1)}<d>{match.group(2).strip()}</d>",
        normalized,
    )
    return normalized


SHOT_PLANNER_SYSTEM = """你是 MiniMax H3 的分镜规划师。你只做一件事：把用户的创意拆成一份镜头计划表。

只输出一个 JSON 对象，不要 Markdown 代码块、不要解释：
{
  "shots": [
    {"index": 1, "start": 0.0, "end": 2.5,
     "beat": "忠实保留的剧情节拍",
     "composition": "景别、机位、构图和主体空间位置",
     "subjects": "本镜出现的主体外观、姿态、视线和起始状态",
     "environment_lighting": "场景、道具、光源、色温、阴影和材质",
     "action_progression": "准备→启动→接触/受力→惯性/次级运动→结果/反应",
     "camera": "运镜类型、目标、幅度和速度",
     "media": ["<Picture 1>"], "media_roles": "每个标签在本镜的具体职责",
     "sound_dialogue": "同步动作声、环境声、说话人和原始台词",
     "continuity": "承接上一镜时必须保持的机位、姿态、环境和身份状态"}
  ]
}

规则：
- 第一个镜头必须从 0 开始，最后一个镜头必须结束在给定的总时长，中间不能有空档或重叠。
- media 只能填写 AVAILABLE MEDIA 里给出的标签原文；这个镜头不引用素材就填空数组 []。
- 先把写作器需要的事实聚合完整，但不要写最终英文文案；每个字段必须给出具体内容，不能写“自定”“保持一致”“同上”。
- composition、subjects、environment_lighting、action_progression、camera、sound_dialogue 分工不同，不能合并成一句剧情摘要。
- 人物动作必须有可见过程和落点；表情必须有触发和变化；运镜必须说明目标；声音必须绑定到实际动作或说话事件。
- media_roles 必须说明标签用于人物外观、场景、构图、动作、声音还是其他明确职责，不能只列标签。
- 忠实于用户创意，不要自行加入用户没提到的情节。
"""


def _is_interrupt(exc: Exception) -> bool:
    return type(exc).__name__ == "InterruptProcessingException"


def _is_rate_limit_error(exc: BaseException) -> bool:
    try:
        from . import llm_service as _llm
        return _llm.is_rate_limit_error(exc)
    except Exception:
        message = str(exc or "").casefold()
        return "api error 429" in message or "tpm exhausted" in message


def _is_quota_error(exc: BaseException) -> bool:
    try:
        from . import llm_service as _llm
        return _llm.is_quota_error(exc)
    except Exception:
        message = str(exc or "").casefold()
        return any(token in message for token in (
            "insufficient_quota", "allocated quota exceeded",
            "额度不足", "额度耗尽", "配额耗尽", "余额不足", "欠费",
        ))


def _is_empty_response_error(exc: BaseException) -> bool:
    """Recognise an empty completed response through local wrapper layers."""
    current: BaseException | None = exc
    for _depth in range(4):
        if current is None:
            break
        if type(current).__name__ == "LLMEmptyResponseError":
            return True
        message = str(current or "").casefold()
        if "returned empty content" in message or "returned an empty result" in message:
            return True
        current = current.__cause__ or current.__context__
    return False


def _call_llm_resilient(
    llm_service: str,
    variants: list[tuple[str, str]],
    ollama_auto_unload: bool,
    seed: int,
    label: str,
    max_tokens_ladder: tuple[int | None, ...] = (None, 2000, 1000),
    validator=None,
    max_validator_retries: int = 1,
) -> tuple[str, list[str]]:
    """Call the LLM through a fallback ladder instead of failing on first miss.

    Two failure modes make a single call unreliable in practice, and neither is
    a bug in the prompt:

    * A reasoning model can spend its whole budget in the thinking channel and
      return an empty answer. The service reports success with no content, so
      the only cure is to ask again -- with a different seed, since replaying
      the same one reproduces the same silence.
    * Small "flash/lite" models reject an oversized request with HTTP 400. That
      needs a *smaller* request, not another identical one, so each variant in
      the ladder is a shorter prompt than the one before and the token ceiling
      steps down alongside it.

    Returns (text, notes). An empty text means every rung failed; callers are
    expected to degrade rather than propagate, so that one flaky response
    cannot take down a whole run. User interrupts are re-raised untouched.
    """
    notes: list[str] = []
    validator_failures = 0
    validator_feedback = ""
    for level, (user_prompt, system_prompt) in enumerate(variants):
        max_tokens = max_tokens_ladder[min(level, len(max_tokens_ladder) - 1)]
        for attempt in range(2):
            try:
                raw = _expand_with_prompt_assistant(
                    llm_service,
                    user_prompt,
                    system_prompt + validator_feedback,
                    ollama_auto_unload,
                    # Vary the seed: an identical request to a reasoning model
                    # reproduces the same empty answer.
                    int(seed) + level * 17 + attempt * 101,
                    max_tokens=max_tokens,
                )
            except Exception as exc:  # noqa: BLE001 - classified just below
                if _is_interrupt(exc):
                    raise
                notes.append(f"{label} L{level + 1}#{attempt + 1} 失败: {str(exc)[:120]}")
                if _is_quota_error(exc):
                    notes.append(
                        f"{label} 路由层已尝试可用线路，但剩余线路额度/配额不可用，停止本轮重复请求")
                    return "", notes
                if _is_rate_limit_error(exc):
                    notes.append(
                        f"{label} 已执行 TPM 冷却重试但仍被限流，停止本轮重复请求")
                    return "", notes
                if _is_empty_response_error(exc):
                    notes.append(
                        f"{label} 服务端已结束但正文为空，不原样重试")
                    return "", notes
                continue
            text = _sanitize_llm_output(raw)
            if text and validator is not None:
                try:
                    violations = list(validator(text) or [])
                except Exception as exc:  # noqa: BLE001 - a broken guard must be visible
                    violations = ["格式校验异常: %s" % str(exc)[:120]]
                if violations:
                    validator_failures += 1
                    notes.append(
                        f"{label} L{level + 1}#{attempt + 1} 技能格式不合格："
                        + "；".join(violations[:4]))
                    if validator_failures > max(0, int(max_validator_retries)):
                        notes.append(
                            f"{label} 定向纠正已执行 {max_validator_retries} 次，"
                            "停止继续改写以避免重试风暴")
                        return "", notes
                    validator_feedback = (
                        "\n\n【上一版未通过技能验收，只允许再纠正一次】\n"
                        + "；".join(violations[:8])[:1200]
                        + "\n请保留原任务与素材关系，按上述缺项重新输出完整最终提示词。")
                    continue
            if text:
                if notes:
                    notes.append(f"{label} 在 L{level + 1}#{attempt + 1} 成功")
                return text, notes
            notes.append(f"{label} L{level + 1}#{attempt + 1} 返回空内容（可能只输出了思考过程）")
    return "", notes


def _plan_shots(
    user_prompt: str,
    manifest: list[dict[str, Any]],
    seconds: float,
    llm_service: str,
    ollama_auto_unload: bool,
    seed: int,
) -> list[dict[str, Any]] | None:
    """First sub-agent: decide the shot breakdown before any prose is written.

    Splitting planning from writing is what stops the writer from anchoring on a
    Skill's worked example. Several official Skills embed a fixed timeline (the
    co-op template is a hardcoded 15s / 6 shots), and a single-call writer reads
    that as the answer and stops after its first segment. The planner never sees
    the Skill text, so it can only work from the real duration, and the writer
    then has a concrete per-shot contract to fill in.
    """
    total = max(1.0, float(seconds))
    shots = max(1, min(8, round(total / 2.5)))
    question = (
        _manifest_text(manifest)
        + f"\n目标视频总时长：{total:g} 秒，请规划约 {shots} 个镜头。\n"
        + "用户创意：\n"
        + (str(user_prompt or "").strip() or "(未提供，请根据素材设计一个合理的短片)")
        + "\n\n请输出镜头计划 JSON。"
    )
    try:
        raw, _notes = _call_llm_resilient(
            llm_service, [(question, SHOT_PLANNER_SYSTEM)], ollama_auto_unload, seed, "规划"
        )
    except Exception as exc:  # noqa: BLE001 - planning is an optimisation, never fatal
        if _is_interrupt(exc):
            raise
        logger.warning("[MiniMax H3 Agent] shot planner failed: %s", exc)
        return None
    if not raw:
        return None

    value = _sanitize_llm_output(raw)
    match = re.search(r"\{[\s\S]*\}", value)
    if match:
        value = match.group(0)
    try:
        data = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        logger.warning("[MiniMax H3 Agent] shot planner returned non-JSON")
        return None
    raw_shots = data.get("shots") if isinstance(data, dict) else None
    if not isinstance(raw_shots, list) or not raw_shots:
        return None

    allowed = {str(item.get("tag") or "") for item in manifest}
    planned: list[dict[str, Any]] = []
    for entry in raw_shots:
        if not isinstance(entry, dict):
            continue
        try:
            start = float(entry.get("start", 0.0))
            end = float(entry.get("end", 0.0))
        except (TypeError, ValueError):
            continue
        if end <= start:
            continue
        media = [tag for tag in (entry.get("media") or []) if str(tag) in allowed]
        planned.append({
            "index": len(planned) + 1,
            "start": start,
            "end": end,
            "beat": str(entry.get("beat") or ""),
            "shot_size": str(entry.get("shot_size") or entry.get("composition") or ""),
            "composition": str(entry.get("composition") or entry.get("shot_size") or ""),
            "subjects": str(entry.get("subjects") or ""),
            "environment_lighting": str(entry.get("environment_lighting") or ""),
            "action_progression": str(
                entry.get("action_progression") or entry.get("beat") or ""),
            "camera": str(entry.get("camera") or ""),
            "media": media,
            "media_roles": str(entry.get("media_roles") or ""),
            "sound": str(entry.get("sound") or entry.get("sound_dialogue") or ""),
            "sound_dialogue": str(entry.get("sound_dialogue") or entry.get("sound") or ""),
            "continuity": str(entry.get("continuity") or ""),
        })
    if not planned:
        return None

    # Snap the plan onto the requested duration. The planner usually gets this
    # right, but a plan that quietly ends early would hand the writer the very
    # bug this stage exists to prevent.
    planned[0]["start"] = 0.0
    planned[-1]["end"] = total
    for previous, current in zip(planned, planned[1:]):
        if current["start"] < previous["end"]:
            current["start"] = previous["end"]
    return [shot for shot in planned if shot["end"] > shot["start"]]


def _plan_text(plan: list[dict[str, Any]], seconds: float) -> str:
    lines = [
        f"镜头写作内容包（总时长 {max(1.0, float(seconds)):g} 秒；事实已经聚合完成，"
        "写作器只负责展开成技能规定的最终格式，不得合并、删项或重新规划）："]
    for shot in plan:
        media = "、".join(shot["media"]) if shot["media"] else "不引用素材"
        lines.append(
            f"[Shot {shot['index']}] {shot['start']:.2f}秒–{shot['end']:.2f}秒"
            f"\n  剧情节拍：{shot.get('beat') or '按用户原文'}"
            f"\n  构图与位置：{shot.get('composition') or shot.get('shot_size') or '依据剧情明确填写'}"
            f"\n  主体状态：{shot.get('subjects') or '依据用户原文和素材清单明确填写'}"
            f"\n  环境与光线：{shot.get('environment_lighting') or '依据场景连续性明确填写'}"
            f"\n  动作过程：{shot.get('action_progression') or shot.get('beat') or '展开可见动作过程'}"
            f"\n  运镜：{shot.get('camera') or '依据动作目标明确填写'}"
            f" | 素材：{media}"
            f"\n  素材职责：{shot.get('media_roles') or '按素材内容和剧情明确分配'}"
            f"\n  声音与台词：{shot.get('sound_dialogue') or shot.get('sound') or '依据用户原文明确填写'}"
            f"\n  连续性：{shot.get('continuity') or ('开场建立完整状态' if int(shot['index']) == 1 else '承接上一镜可见状态')}"
        )
    return "\n".join(lines)


COVERAGE_FIXER_SYSTEM = """你是 MiniMax H3 的提示词补全助手。给你一份镜头计划和一版只写了一部分的提示词，请补全缺失的镜头。

规则：
- 保留已经写好的镜头原文，不要改写它们。
- 按镜头计划补齐所有缺失的镜头，保持与已有镜头相同的格式和写作密度。
- 只能引用计划中列出的素材标签，逐字使用。
- 只输出补全后的完整提示词正文，不要解释、不要 Markdown 代码块。
"""


def _shot_coverage(text: str) -> set[int]:
    return {int(n) for n in SHOT_MARKER_RE.findall(str(text or ""))}


def _ensure_shot_coverage(
    generated: str,
    plan: list[dict[str, Any]],
    seconds: float,
    llm_service: str,
    ollama_auto_unload: bool,
    seed: int,
) -> tuple[str, list[str]]:
    """Verify the writer covered the plan, and repair it with one more pass.

    This is the check that would have caught the reported bug: a 10s request
    that came back as a single 3s shot. Returns the (possibly repaired) text
    plus notes describing what happened, which surface in media_manifest.
    """
    notes: list[str] = []
    if not plan:
        return generated, notes
    expected = {shot["index"] for shot in plan}
    covered = _shot_coverage(generated)
    missing = sorted(expected - covered)
    if not missing:
        return generated, notes

    notes.append(f"writer covered {sorted(covered) or '无'} of {sorted(expected)}; repairing {missing}")
    question = (
        _plan_text(plan, seconds)
        + "\n\n当前提示词（缺少 "
        + "、".join(f"Shot {n}" for n in missing)
        + "）：\n"
        + generated
        + "\n\n请补全缺失的镜头，输出完整提示词。"
    )
    try:
        repaired, _notes = _call_llm_resilient(
            llm_service, [(question, COVERAGE_FIXER_SYSTEM)], ollama_auto_unload, seed, "补全"
        )
    except Exception as exc:  # noqa: BLE001 - keep the partial prompt on failure
        if _is_interrupt(exc):
            raise
        notes.append(f"repair failed: {exc}")
        return generated, notes
    if not repaired:
        notes.append("repair returned empty output")
        return generated, notes

    repaired = _sanitize_llm_output(repaired)
    if not repaired:
        notes.append("repair returned empty output")
        return generated, notes
    # A repair only adds shots, so it must keep every shot and tag it started
    # with. min_ratio is 0.9 because the output should strictly grow here.
    damage = _body_preserved(generated, repaired, min_ratio=0.9)
    if damage:
        notes.append(f"repair rejected ({damage})")
        return generated, notes
    still_missing = sorted(expected - _shot_coverage(repaired))
    if len(still_missing) >= len(missing):
        notes.append(f"repair did not improve coverage (still missing {still_missing})")
        return generated, notes
    notes.append(f"repaired to cover {sorted(_shot_coverage(repaired))}")
    return repaired, notes


def _body_preserved(original: str, candidate: str, min_ratio: float = 0.7) -> str:
    """Check a repair pass kept the prompt it was supposed to edit.

    Both repair sub-agents are told to return the whole prompt with a small
    part changed, but an LLM can truncate, summarise, or answer with a stray
    remark instead. Judging those results only by whether they fixed the
    reported problem is backwards: a prompt cut down to one line trivially has
    no dialogue-timing errors left, so the "did it improve?" check accepts it
    and the real prompt is lost. Returns "" when the candidate is safe, or a
    reason string when it must be rejected.
    """
    source = str(original or "")
    result = str(candidate or "")
    if not result.strip():
        return "返回空结果"
    if len(result) < len(source) * min_ratio:
        return f"正文从 {len(source)} 字符缩到 {len(result)} 字符（疑似截断）"
    missing_shots = sorted(_shot_coverage(source) - _shot_coverage(result))
    if missing_shots:
        return f"丢失镜头 {missing_shots}"
    source_tags = {match.group(0).lower() for match in MEDIA_TAG_RE.finditer(source)}
    result_tags = {match.group(0).lower() for match in MEDIA_TAG_RE.finditer(result)}
    lost_tags = sorted(source_tags - result_tags)
    if lost_tags:
        return f"丢失素材标签 {lost_tags}"
    return ""


DIALOGUE_FIXER_SYSTEM = """你是 MiniMax H3 的台词校正助手。给你若干条台词和各自的约束，请逐条改写。

语速标准：
- 兴奋/激动/大喊：3.5-6 字/秒
- 平静/陈述/认真：2.5-4 字/秒
- 休闲/撒娇/轻声/慵懒：1.5-3 字/秒

只输出一个 JSON 对象，键是台词编号，值是改写后的台词文本，不要 Markdown 代码块、不要解释：
{"1": "改写后的台词", "2": "改写后的台词"}

规则：
- 台词过长：精简到能在给定秒数内说完，保留原意和情绪，宁可断句也不要加快到不自然。
- 台词过短：适度扩充到匹配时长，不要硬凑废话。
- 语言不符：改写成要求的语言，保持语气一致。
- 值只写台词本身，不要带 <d></d> 标签，不要带编号前缀，不要写画面描述。
- 不需要改动的台词可以省略不输出。
"""


def _fix_dialogue_via_agent(
    generated: str,
    audit_result: dict[str, Any],
    seconds: float,
    llm_service: str,
    ollama_auto_unload: bool,
    seed: int,
) -> tuple[str, list[str]]:
    """Repair dialogue that is mistimed or in the wrong language.

    Only the dialogue lines are sent and only the dialogue lines come back; the
    surrounding prose is spliced by code and never leaves the process. Asking
    the model to re-emit the whole prompt to change a few lines was both
    wasteful and unsafe -- a single truncated reply replaced a 4000-character
    prompt with 157 characters, and because the shortened text no longer had
    any dialogue to be wrong about, the "did it improve?" check waved it
    through.

    Only runs when the deterministic audit found something, so a clean prompt
    costs nothing. Any failure keeps the original text.
    """
    issues = audit_result.get("issues") or []
    blocks = audit_result.get("blocks") or []
    if not issues or not blocks:
        return generated, []

    notes = [issue["detail"] for issue in issues]
    flagged = {issue["index"] for issue in issues}
    by_index = {block["index"]: block for block in blocks}
    expected_script = audit_result.get("expected_script")
    expected_name = dialogue_audit.SCRIPT_NAMES.get(expected_script, expected_script or "原语言")

    lines = []
    for index in sorted(flagged):
        block = by_index.get(index)
        if not block:
            continue
        window = float(block.get("window_seconds") or 0.0)
        lines.append(
            f"台词 {index}：{block['text']}\n"
            f"  语气：{block['tone']}（{block['rate_min']}-{block['rate_max']} 字/秒）\n"
            f"  可用时长：{window:.2f} 秒 → 建议 {max(1, int(window * block['rate_min']))}-{max(1, int(window * block['rate_max']))} 字\n"
            f"  当前：{block['units']} 字（{block['script_name']}）\n"
            f"  要求语言：{expected_name}"
        )
    if not lines:
        return generated, notes

    question = "\n\n".join(lines) + "\n\n请输出改写后的台词 JSON。"
    try:
        raw, _notes = _call_llm_resilient(
            llm_service, [(question, DIALOGUE_FIXER_SYSTEM)], ollama_auto_unload, seed, "台词"
        )
        if not raw:
            return generated, notes + ["台词修正未返回内容，保留原文"]
    except Exception as exc:  # noqa: BLE001 - dialogue polish must never be fatal
        if _is_interrupt(exc):
            raise
        return generated, notes + [f"台词修正失败: {exc}"]

    value = _sanitize_llm_output(raw)
    match = re.search(r"\{[\s\S]*\}", value)
    if match:
        value = match.group(0)
    try:
        replacements = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return generated, notes + ["台词修正返回非 JSON，保留原文"]
    if not isinstance(replacements, dict) or not replacements:
        return generated, notes + ["台词修正未返回任何改写，保留原文"]

    # Splice by position so only the <d> payloads change.
    clean: dict[int, str] = {}
    for key, text in replacements.items():
        try:
            index = int(str(key).strip())
        except (TypeError, ValueError):
            continue
        new_text = re.sub(r"</?d>", "", str(text or "")).strip()
        if index in by_index and new_text:
            clean[index] = new_text
    if not clean:
        return generated, notes + ["台词修正结果无效，保留原文"]

    counter = {"n": 0}

    def swap(match):
        counter["n"] += 1
        return f"<d>{clean[counter['n']]}</d>" if counter["n"] in clean else match.group(0)

    fixed = dialogue_audit.DIALOGUE_RE.sub(swap, generated)

    damage = _body_preserved(generated, fixed, min_ratio=0.5)
    if damage:
        return generated, notes + [f"台词修正被拒绝（{damage}），保留原文"]

    recheck = dialogue_audit.audit(
        fixed,
        seconds,
        expected_script=expected_script,
        plan=audit_result.get("plan"),
    )
    if len(recheck.get("issues") or []) >= len(issues):
        return generated, notes + [f"台词修正未改善（仍有 {len(recheck.get('issues') or [])} 处），保留原文"]
    return fixed, notes + [f"台词已修正 {len(clean)} 句：{len(issues)} 处 → {len(recheck.get('issues') or [])} 处"]


NO_EXPAND_RULE = """
【严格模式 — 不扩写】
用户已关闭扩写。你只能细化用户原始指令中已经存在的内容，使其达到目标时长所需的描述密度：
- 允许：把"两人打架"细化为具体的动作、景别、运镜、光线、材质、节奏。
- 禁止：新增用户没有提到的情节、角色、场景、道具、台词或转折。
- 禁止：自行编造故事线或补充前因后果。
若用户指令过于简短、不足以填满时长，请通过放慢节奏、增加镜头内的细节描写来达到时长，而不是增加新情节。
"""

EXPAND_RULE = """
【扩写模式】
用户已开启扩写。你可以在忠于用户原始意图的前提下，合理展开情节、补充符合语境的动作、环境细节与情绪层次，
让 {seconds:g} 秒的视频内容饱满。新增内容必须服务于用户原始指令，不得偏离主题或改变结局。
"""


_H3_BASE_FIELDS = (
    "integrated_multimodal_description",
    "overall_soundscape",
    "non_diegetic_music",
)
_H3_REF_FIELDS = (
    "subject_definitions",
    "summary",
    "retention_analysis",
    "detailed_description",
    "overall_soundscape",
    "non_diegetic_music",
)
_STRICT_SHOT_RE = re.compile(r"\[\s*Shot\s+\d+\s*\]", re.IGNORECASE)
_STRICT_CUT_TIME_RE = re.compile(r"\bAt\s+\d{2}:\d{2}\.\d{3}\b", re.IGNORECASE)


def skill_output_contract(skill_text: str) -> dict[str, Any]:
    """Extract the small enforceable part of a learned writing Skill.

    Learning remains useful prose for the writer, but prose alone cannot stop
    a fallback model from returning a generic paragraph.  This contract keeps
    only rules Python can verify without trying to judge creative quality.
    """
    rules = str(skill_text or "")
    lowered = rules.casefold()
    alternatives: list[tuple[str, ...]] = []
    if all(field.casefold() in lowered for field in _H3_BASE_FIELDS):
        alternatives.append(_H3_BASE_FIELDS)
    if all(field.casefold() in lowered for field in _H3_REF_FIELDS):
        alternatives.append(_H3_REF_FIELDS)
    require_shots = bool(
        re.search(r"\[\s*Shot\s+N\s*\]", rules, re.IGNORECASE)
        or ("镜头标记" in rules and "shot" in lowered)
    )
    require_cut_times = bool(
        re.search(r"At\s+MM:SS\.mmm", rules, re.IGNORECASE)
        or ("三位小数" in rules and "at " in lowered)
    )
    english_sections = bool(
        ("rewrite sections" in lowered and "english" in lowered)
        or "所有 rewrite sections 用英文" in rules
        or "rewrite sections in english" in lowered
    )
    return {
        "alternatives": alternatives,
        "require_shots": require_shots,
        "require_cut_times": require_cut_times,
        "english_sections": english_sections,
        "active": bool(alternatives or require_shots or require_cut_times),
    }


def skill_contract_instruction(skill_text: str) -> str:
    """Render a compact source-writing contract; post-checks remain advisory."""
    contract = skill_output_contract(skill_text)
    if not contract["active"]:
        return ""
    lines = ["【技能格式写作清单（请在首轮输出中完整落实）】"]
    alternatives = contract["alternatives"]
    if alternatives:
        rendered = [" → ".join(fields) for fields in alternatives]
        if len(rendered) > 1:
            lines.append(
                "输出必须完整采用以下一套字段并保持顺序；除非任务明确是 Ref2VA/full-reference，"
                "否则采用第一套：" + "；或 ".join(rendered))
        else:
            lines.append("输出必须完整包含这些字段并保持顺序：" + rendered[0])
        lines.append("字段名必须单独位于行首并紧跟冒号，不得省略、翻译或改名。")
    if contract["require_shots"]:
        lines.append("正文必须使用严格的 [Shot 1]、[Shot 2]……镜头标记。")
    if contract["require_cut_times"]:
        lines.append("从第二个镜头起必须使用 At MM:SS.mmm 格式的严格递增切点时间。")
    if contract["english_sections"]:
        lines.append("rewrite sections 必须用英文；台词、歌词和屏幕可见文字保留原语言。")
    lines.append("请直接按上述结构完成首轮输出，不要依赖后续改写。")
    return "\n".join(lines)


def skill_output_issues(output: str, skill_text: str) -> list[str]:
    """Return deterministic Skill-format violations for one generated prompt."""
    contract = skill_output_contract(skill_text)
    if not contract["active"]:
        return []
    value = str(output or "").strip()
    if not value:
        return ["返回为空"]
    issues: list[str] = []
    alternatives = contract["alternatives"]
    if alternatives:
        valid_structure = False
        for fields in alternatives:
            positions = []
            for field in fields:
                match = re.search(
                    rf"(?mi)^\s*{re.escape(field)}\s*:\s*", value)
                positions.append(match.start() if match else -1)
            if all(position >= 0 for position in positions) and positions == sorted(positions):
                valid_structure = True
                break
        if not valid_structure:
            expected = " 或 ".join(" → ".join(fields) for fields in alternatives)
            issues.append("缺少必需字段或字段顺序错误（应为 %s）" % expected)

    shots = _STRICT_SHOT_RE.findall(value)
    if contract["require_shots"] and not shots:
        issues.append("缺少严格的 [Shot N] 镜头标记")
    if contract["require_cut_times"] and len(shots) > 1:
        cut_times = _STRICT_CUT_TIME_RE.findall(value)
        if len(cut_times) < len(shots) - 1:
            issues.append("第二镜头以后缺少 At MM:SS.mmm 切点时间")

    if contract["english_sections"] and not issues:
        # Dialogue and visible quoted text are explicitly allowed to retain the
        # user's language.  Only reject a gross all-Chinese rewrite; this is a
        # guardrail, not a language classifier.
        prose = re.sub(r"<d>[\s\S]*?</d>", "", value, flags=re.IGNORECASE)
        prose = re.sub(r'["“][^"”]*["”]', "", prose)
        cjk_count = len(re.findall(r"[\u3400-\u9fff]", prose))
        latin_words = len(re.findall(r"\b[A-Za-z]{3,}\b", prose))
        if cjk_count >= 80 and latin_words < 20:
            issues.append("rewrite sections 没有按技能要求使用英文")
    return issues


def _normalize_dialogue_language_tags(output: str) -> str:
    """Add inferable dialogue language markup locally; never spend an LLM retry."""
    value = str(output or "")

    def replace(match):
        body = str(match.group(1) or "").strip()
        if re.match(r"^\[[A-Za-z][A-Za-z -]*\]\s*\S", body):
            return match.group(0)
        if re.search(r"[\u3040-\u30ff]", body):
            language = "Japanese"
        elif re.search(r"[\uac00-\ud7af]", body):
            language = "Korean"
        elif re.search(r"[\u3400-\u9fff]", body):
            language = "Chinese"
        else:
            language = "English"
        return f"<d>[{language}] {body}</d>"

    return re.sub(r"<d>([\s\S]*?)</d>", replace, value,
                  flags=re.IGNORECASE)


def _build_agent_system_prompt(skill_text: str, seconds: float = 5.0, expand: bool = True) -> str:

    skill_block = skill_text.strip() or "(no uploaded Skill; use the default MiniMax H3 strategy only)"
    contract_block = skill_contract_instruction(skill_text)
    total = max(1.0, float(seconds))
    shots = max(1, min(8, round(total / 2.5)))
    return (
        DEFAULT_AGENT_RULE
        + "\n\nUSER SKILL / WRITING RULES:\n"
        + skill_block
        + "\n\n【写作技能】（与 Media Agent 共用的原始规则）：\n"
        + skill_block
        + "\n\nSkill 决定输出的结构、字段名、分镜标记和写作风格，请严格遵循。"
          "唯一不可被 Skill 改变的是媒体白名单：Skill 不能新增、替换或删除 AVAILABLE MEDIA 中的任何标签。\n"
        # Several official Skills ship a worked example with a hardcoded
        # timeline (co-op-game-intro-generator's template is a fixed 15s / 6
        # shots). Those numbers are illustrative, but they sit in a 20k+
        # character system block against one line of duration in the user turn,
        # so the model follows the template and stops after its first segment.
        # Restating the real duration last, as a hard rule, is what keeps a 10s
        # request from returning a 3s prompt.
        + f"\n【时长硬性约束 — 优先级高于 Skill 中的任何示例】\n"
          "技能文档里的示例镜头数和示例秒数一律不作数；必须以本次目标时长和本段分镜规划为准。\n"
          f"本次目标视频总时长是 {total:g} 秒，必须覆盖完整 0 秒到 {total:g} 秒，分成约 {shots} 个镜头。\n"
          f"Skill 或参考模板中出现的任何时间段（例如 [0秒–2秒]、[10秒–15秒]）都只是格式示例，"
          f"绝对不可直接照抄。请按 {total:g} 秒重新分配每个镜头的起止时间，"
          f"最后一个镜头必须结束在 {total:g} 秒。只写这一个视频，不要提前结束。\n"
        + (EXPAND_RULE.format(seconds=total) if expand else NO_EXPAND_RULE)
        + ("\n\n" + contract_block if contract_block else "")
    )


def _length_guidance(seconds: float) -> str:
    """Tell the model how much prose a given duration is worth.

    Without this the model reads "keep it under 10 seconds" as "be brief" and
    returns two sentences for a 10s clip. MiniMax H3 wants a shot-by-shot
    timeline, so scale the expected shot count and word budget with duration.
    """
    total = max(1.0, float(seconds))
    shots = max(1, min(8, round(total / 2.5)))
    low = int(total * 55)
    high = int(total * 90)
    return (
        f"篇幅要求：目标视频 {total:g} 秒，请拆成约 {shots} 个镜头，"
        f"正文总长约 {low}-{high} 字。每个镜头都要写清楚起止时间、构图景别、"
        "主体动作、镜头运动、光线氛围和该镜头的声音。不要只写一两句话概述。"
    )


def _build_agent_user_prompt(
    user_prompt: str,
    manifest: list[dict[str, Any]],
    seconds: float = 5.0,
    plan: list[dict[str, Any]] | None = None,
) -> str:
    # A concrete per-shot contract beats a word count: the writer fills in the
    # plan instead of inventing a structure, so a Skill's example timeline can
    # no longer decide how long the video is.
    if plan:
        brief = "\n" + _plan_text(plan, seconds) + "\n请严格按上面的镜头计划逐个写出完整画面描述。\n"
    else:
        brief = "\n" + _length_guidance(seconds) + "\n"
    return (
        _manifest_text(manifest)
        + f"\nTARGET VIDEO DURATION: {seconds} seconds.\n"
        + "ORIGINAL USER PROMPT:\n"
        + (user_prompt.strip() or "(empty; create a concise MiniMax H3 reference-video prompt from the goal and media whitelist)")
        + f"\n\n请基于上述白名单改良提示词，视频总时长控制在 {seconds} 秒以内。\n"
        + brief
        + "只输出最终提示词正文。"
    )


def build_media_agent_writer_request(
    user_prompt: str,
    manifest: list[dict[str, Any]],
    skill_text: str,
    seconds: float,
    expand: bool = False,
) -> tuple[str, str]:
    """Shared Media Agent writer request used by both Agent and Director.

    The caller owns chronology and calls this once for one target prompt.  This
    deliberately returns plain-text writer prompts rather than asking a model
    to invent a variable-length JSON list of multiple videos.
    """
    system_prompt = _build_agent_system_prompt(
        skill_text, seconds=seconds, expand=expand)
    user_text = _build_agent_user_prompt(
        user_prompt, manifest, seconds=seconds, plan=None)
    return user_text, system_prompt


def _allowed_tags(manifest: list[dict[str, Any]]) -> dict[str, set[int]]:
    allowed: dict[str, set[int]] = {"picture": set(), "video": set(), "audio": set()}
    for item in manifest:
        kind = "picture" if item["type"] == "image" else item["type"]
        allowed[kind].add(int(item["ordinal"]))
    return allowed


def _invalid_references(text: str, manifest: list[dict[str, Any]]) -> list[str]:
    allowed = _allowed_tags(manifest)
    invalid: list[str] = []
    for match in MEDIA_TAG_RE.finditer(text):
        raw_kind, number_text = match.groups()
        kind = raw_kind.lower()
        kind = "picture" if kind == "image" else kind
        number = int(number_text)
        if number < 1 or number not in allowed.get(kind, set()):
            invalid.append(match.group(0))
    for match in GENERIC_FILE_RE.finditer(text):
        invalid.append(match.group(0))
    for match in AT_TOKEN_RE.finditer(text):
        invalid.append("@" + match.group(1))
    return list(dict.fromkeys(invalid))


REASONING_BLOCK_RE = re.compile(
    r"<\s*(think|thinking|reasoning|thought|analysis|scratchpad)\s*>[\s\S]*?<\s*/\s*\1\s*>",
    re.IGNORECASE,
)
UNCLOSED_REASONING_RE = re.compile(
    r"^[\s\S]*?<\s*/\s*(?:think|thinking|reasoning|thought|analysis|scratchpad)\s*>",
    re.IGNORECASE,
)


def _strip_reasoning(text: str) -> str:
    """Drop a reasoning model's visible thinking, keeping the answer.

    Reasoning models leak their scratchpad in several shapes: properly paired
    <think>…</think>, or a stream that opens one and only ever emits the close
    tag. Both are noise here -- the node's output is a prompt, not a transcript
    of how it was written.
    """
    value = str(text or "")
    value = REASONING_BLOCK_RE.sub("", value)
    if re.search(r"<\s*/\s*(?:think|thinking|reasoning|thought|analysis|scratchpad)\s*>", value, re.IGNORECASE):
        value = UNCLOSED_REASONING_RE.sub("", value)
    return value.strip()


def _sanitize_llm_output(text: str) -> str:
    value = _strip_reasoning(text)
    fenced = re.fullmatch(r"```(?:\w+)?\s*([\s\S]*?)\s*```", value)
    if fenced:
        value = fenced.group(1).strip()
    prefixes = ("最终提示词：", "最终提示词:", "提示词：", "提示词:", "Prompt:", "prompt:")
    for prefix in prefixes:
        if value.startswith(prefix):
            value = value[len(prefix):].lstrip()
            break
    return value


def _validate_output(text: str, manifest: list[dict[str, Any]]) -> str:
    value = _sanitize_llm_output(text)
    if not value:
        raise ValueError("Agent returned an empty prompt")
    invalid = _invalid_references(value, manifest)
    if invalid:
        raise ValueError(
            "Agent referenced media that is not connected or uses a forbidden format: "
            + ", ".join(invalid)
        )
    return value


EDITOR_LABELS = {"image": "图片", "video": "视频", "audio": "音频"}


def _editor_prompt(text: str, manifest: list[dict[str, Any]]) -> str:
    """Render the prompt in the Myang prompt-editor syntax.

    Media become @图片1 / @视频1 / @音频1 mentions and dialogue stays wrapped in
    <d></d> blocks, which is exactly what that editor round-trips. The Myang
    backend resolves the same syntax, so this output can be wired straight into
    its prompt input as well as read by a human.
    """
    label_by_tag: dict[str, str] = {}
    for item in manifest or []:
        if not isinstance(item, dict):
            continue
        tag = str(item.get("tag") or "").strip().lower()
        kind = str(item.get("type") or "")
        ordinal = item.get("ordinal")
        if not tag or kind not in EDITOR_LABELS or ordinal is None:
            continue
        try:
            label_by_tag[tag] = f"@{EDITOR_LABELS[kind]}{int(ordinal)}"
        except (TypeError, ValueError):
            continue

    def replace(match):
        kind = match.group(1).lower()
        kind = "picture" if kind == "image" else kind
        return label_by_tag.get(f"<{kind} {match.group(2)}>", match.group(0))

    return MEDIA_TAG_RE.sub(replace, str(text or ""))


def _expand_with_prompt_assistant(
    llm_service: str,
    combined_text: str,
    system_prompt: str,
    ollama_auto_unload: bool,
    seed: int,
    max_tokens: int | None = None,
) -> str:
    from . import llm_service as _llm
    try:
        expanded = _llm.call_llm(
            service_str=llm_service,
            system_prompt=system_prompt,
            user_prompt=combined_text,
            ollama_auto_unload=ollama_auto_unload,
            seed=seed,
            max_tokens=max_tokens,
        )
    except Exception as error:
        if "interrupt" in str(error).lower():
            from comfy.model_management import InterruptProcessingException
            raise InterruptProcessingException()
        if isinstance(error, _llm.LLMEmptyResponseError):
            raise
        if _llm.is_quota_error(error):
            advice = (
                "路由层已尝试其余可用线路；若仍返回此错误，说明其他线路也在冷却/熔断，"
                "或多个 Key 共享同一个已耗尽的 Workspace 配额。请补充额度或换用独立工作区 Key，"
                "然后在 Agent 技能面板重置线路状态。"
            )
        elif _llm.is_rate_limit_error(error):
            advice = (
                "路由层已尝试其余可用线路；当前所有剩余线路都处于 TPM 冷却。"
                "本轮不会原地等待或连续轰炸，稍后可直接续学未完成分块。"
            )
        else:
            advice = "请在 Agent 技能面板查看各线路状态，修复异常线路后可续学未完成分块。"
        raise RuntimeError(
            f"MiniMax H3 Media Agent 运行失败 (LLM服务 [{llm_service}] 报错: {error})\n"
            f"【路由诊断】：{advice}"
        ) from error
    if not expanded:
        raise RuntimeError("MiniMax H3 Media Agent returned an empty result")
    return expanded


FIELD_PREFIX_RE = re.compile(r"(?mi)^[ \t]*(?:integrated_multimodal_description|overall_soundscape|non_diegetic_music|subject_definitions|summary|retention_analysis|detailed_description)[ \t]*:[ \t]*")
# Writers label shots in several ways: a bare "[Shot 1]", a decorated
# "[Shot 1 — 0秒–3秒]", or a bare-line "Shot 1:". Match the number in all of
# them, otherwise coverage checking silently passes on a truncated prompt.
SHOT_MARKER_RE = re.compile(r"(?:\[\s*)?\bShot\s*(\d+)\s*(?:\]|[—–\-:|，,]|$)", re.IGNORECASE | re.MULTILINE)



def _strip_structure_markers(text: str) -> str:
    """Drop Skill field names and [Shot N] markers for prose-only consumers.

    The rewrite itself keeps them -- MiniMax H3 wants that structure -- but the
    storyboard summary treats the text as prose, and without this the first
    "sentence" of a Skill-formatted prompt is literally
    "integrated_multimodal_description: ...", which then shows up as the scene
    description in the preview panel.
    """
    value = FIELD_PREFIX_RE.sub("", str(text or ""))
    value = SHOT_MARKER_RE.sub("", value)
    return value.strip()


SUMMARY_AGENT_SYSTEM = """你是 MiniMax H3 的分镜总结助手。你会收到一段已经写好的视频提示词，请把它整理成结构化的剧情总结。

只输出一个 JSON 对象，不要 Markdown 代码块、不要解释。字段：
{
  "style": "画面风格，一句话",
  "scene": "场景设定，一句话",
  "timeline": [{"time_range": "0.0s - 2.5s", "camera_movement": "景别或运镜", "content": "该镜头发生了什么"}],
  "ambient_sound": "环境音",
  "bgm": "背景音乐",
  "dialogue": "所有台词，用；分隔；没有则写 无台词"
}

规则：
- 时间轴必须覆盖完整时长，分段依据提示词里的镜头划分；提示词若带 [Shot N] 或时间标记，按它来分。
- 不要把 integrated_multimodal_description 之类的字段名写进任何值里。
- 忠实于原文，不要新增原文没有的情节、素材或台词。
"""


def _summary_via_agent(
    prompt: str,
    manifest: list[dict[str, Any]],
    seconds: float,
    llm_service: str,
    ollama_auto_unload: bool,
    seed: int,
    plan: list[dict[str, Any]] | None = None,
) -> str | None:
    """Ask a dedicated sub-agent to build the storyboard summary.

    Kept separate from the rewrite call so the two never contaminate each other:
    the rewrite agent is under strict "prompt body only" rules and would refuse
    to emit JSON, while this one never sees the media whitelist rules and so
    cannot be tempted to edit the prompt. Returns None to fall back to the
    heuristic builder.
    """
    question = (
        f"目标时长：{max(1.0, float(seconds)):g} 秒\n\n"
        + (f"该提示词依据的镜头计划：\n{_plan_text(plan, seconds)}\n\n" if plan else "")
        + "提示词正文：\n"
        + str(prompt or "").strip()
        + "\n\n请输出 JSON。"
    )
    try:
        raw, _notes = _call_llm_resilient(
            llm_service, [(question, SUMMARY_AGENT_SYSTEM)], ollama_auto_unload, seed, "摘要"
        )
        if not raw:
            return None
    except Exception as exc:  # noqa: BLE001 - the summary is a preview, never fatal
        if _is_interrupt(exc):
            raise
        logger.warning("[MiniMax H3 Agent] summary sub-agent failed: %s", exc)
        return None

    value = _sanitize_llm_output(raw)
    fenced = re.search(r"\{[\s\S]*\}", value)
    if fenced:
        value = fenced.group(0)
    try:
        data = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        logger.warning("[MiniMax H3 Agent] summary sub-agent returned non-JSON")
        return None
    if not isinstance(data, dict):
        return None

    timeline = data.get("timeline")
    if not isinstance(timeline, list) or not timeline:
        return None
    clean_timeline = []
    for entry in timeline:
        if not isinstance(entry, dict):
            continue
        clean_timeline.append({
            "time_range": str(entry.get("time_range") or ""),
            "camera_movement": str(entry.get("camera_movement") or "自然跟随运镜"),
            "content": _strip_structure_markers(str(entry.get("content") or "")),
        })
    if not clean_timeline:
        return None

    media_lines = _media_summary_lines(manifest)
    summary_obj = {
        "basic_setup": {
            "style": _strip_structure_markers(str(data.get("style") or "")) or "高清写实，影院级电影质感与光影",
            "scene": _strip_structure_markers(str(data.get("scene") or "")) or "画面主体与背景环境",
            "characters": media_lines,
            "media_manifest": manifest,
            "target_seconds": max(1.0, float(seconds)),
        },
        "timeline": clean_timeline,
        "multimodal": {
            "ambient_sound": str(data.get("ambient_sound") or "根据画面匹配的自然环境音效"),
            "bgm": str(data.get("bgm") or "契合画面情绪氛围的背景音乐"),
            "dialogue": str(data.get("dialogue") or "无台词"),
        },
    }
    return json.dumps(summary_obj, ensure_ascii=False, indent=2)


def _media_summary_lines(manifest: list[dict[str, Any]]) -> str:
    type_names = {"image": "图片", "video": "视频", "audio": "音频"}
    media_items_summary = []
    for item in manifest:
        tag = item.get("tag", "")
        kind = type_names.get(item.get("type"), "素材")
        filename = item.get("filename") or item.get("label") or item.get("description") or "未命名素材"
        info_parts = []
        if item.get("resolution"):
            info_parts.append(f"分辨率 {item['resolution']}")
        if item.get("frame_count"):
            info_parts.append(f"{item['frame_count']}帧")
        if item.get("duration"):
            info_parts.append(f"时长 {item['duration']}秒")
        if item.get("sample_rate"):
            info_parts.append(f"采样率 {item['sample_rate']}Hz")
        info_str = f" ({', '.join(info_parts)})" if info_parts else ""
        media_items_summary.append(f"{tag} ➔ {kind}：{filename}{info_str}")
    return "\n  ".join(f"• {m}" for m in media_items_summary) if media_items_summary else "• 未连接外部参考素材"


def _generate_summary_json(prompt: str, manifest: list[dict[str, Any]], seconds: float = 5.0) -> str:
    characters_str = _media_summary_lines(manifest)

    dialogues = re.findall(r"<d>(.*?)</d>", prompt)
    dialogue_str = "；".join(d.strip() for d in dialogues) if dialogues else "无台词"

    camera_keywords = ["推镜头", "拉镜头", "摇镜头", "移镜头", "俯拍", "仰拍", "特写", "全景", "跟拍", "环绕", "固定镜头", "慢镜头", "变焦", "升降", "切至", "转场", "pan", "tilt", "zoom", "orbit", "tracking", "close-up", "wide shot"]
    found_cameras = [kw for kw in camera_keywords if kw in prompt.lower()]
    camera_str = "、".join(found_cameras) if found_cameras else "自然跟随运镜"

    clean_text = re.sub(r"<[^>]+>", "", _strip_structure_markers(prompt)).strip()
    sentences = [s.strip() for s in re.split(r"[。！!？?\n]+", clean_text) if s.strip()]

    style_str = "高清写实，影院级电影质感与光影"
    scene_str = sentences[0] if sentences else "画面主体与背景环境"

    total_sec = max(1.0, float(seconds))
    timeline = []
    if len(sentences) <= 1:
        timeline.append({
            "time_range": f"0.0s - {total_sec:.1f}s",
            "camera_movement": camera_str,
            "content": sentences[0] if sentences else prompt.strip(),
        })
    else:
        num_blocks = min(len(sentences), 4)
        step = total_sec / num_blocks
        for idx in range(num_blocks):
            start_t = round(idx * step, 1)
            end_t = round((idx + 1) * step, 1)
            sent_cam = [kw for kw in camera_keywords if kw in sentences[idx].lower()]
            cam_desc = "、".join(sent_cam) if sent_cam else camera_str
            timeline.append({
                "time_range": f"{start_t:.1f}s - {end_t:.1f}s",
                "camera_movement": cam_desc,
                "content": sentences[idx],
            })

    ambient_sound = "根据画面匹配的自然环境音效"
    bgm_sound = "契合画面情绪氛围的背景音乐"

    summary_obj = {
        "basic_setup": {
            "style": style_str,
            "scene": scene_str,
            "characters": characters_str,
            "media_manifest": manifest,
            "target_seconds": total_sec,
        },
        "timeline": timeline,
        "multimodal": {
            "ambient_sound": ambient_sound,
            "bgm": bgm_sound,
            "dialogue": dialogue_str,
        }
    }
    return json.dumps(summary_obj, ensure_ascii=False, indent=2)


WHISPER_MODELS = ("off", "tiny", "base", "small", "medium")
AUDIO_SUBJECTS = ("自动", "图片1", "图片2", "图片3", "图片4", "视频1", "视频2", "纯配乐(无主体)")
_whisper_cache: dict[str, Any] = {}


def _transcribe_audio(audio: Any, model_name: str) -> tuple[str, str]:
    """Run Whisper over one connected audio clip.

    Knowing what the audio actually says changes what the agent can write: it
    can align the dialogue in the prompt with the real recording instead of
    inventing lines over it. Returns (text, error) and never raises -- a failed
    transcription should cost the clip its subtitles, not the whole run.
    """
    if model_name == "off":
        return "", ""
    try:
        import whisper  # noqa: PLC0415 - optional, loaded only when asked for
    except ImportError:
        return "", "whisper 未安装（pip install openai-whisper）"

    # audio_track raises on malformed input; a bad clip must cost only
    # its own transcript, not the run.
    try:
        data = audio_track(audio)
    except Exception as exc:  # noqa: BLE001
        return "", f"无法解析音频数据: {exc}"
    if not data:
        return "", "无法解析音频数据"
    waveform = data.get("waveform")
    sample_rate = int(data.get("sample_rate") or 0)
    if waveform is None or not sample_rate:
        return "", "音频缺少波形或采样率"

    try:
        import torch  # noqa: PLC0415

        samples = waveform
        while hasattr(samples, "dim") and samples.dim() > 2:
            samples = samples[0]
        if hasattr(samples, "dim") and samples.dim() == 2:
            samples = samples.mean(dim=0)
        samples = samples.detach().to("cpu", dtype=torch.float32)

        # Whisper expects 16 kHz mono.
        if sample_rate != 16000:
            import torchaudio  # noqa: PLC0415

            samples = torchaudio.functional.resample(samples, sample_rate, 16000)

        peak = float(samples.abs().max()) if samples.numel() else 0.0
        if peak > 1.0:
            samples = samples / peak

        model = _whisper_cache.get(model_name)
        if model is None:
            logger.info("[MiniMax H3 Agent] loading whisper model %r (first run downloads it)", model_name)
            model = whisper.load_model(model_name)
            _whisper_cache[model_name] = model

        result = model.transcribe(samples.numpy(), fp16=False)
        text = " ".join(str(result.get("text") or "").split())
        language = str(result.get("language") or "")
        if not text:
            return "", "未识别到语音内容"
        return (f"{text}（识别语言: {language}）" if language else text), ""
    except Exception as exc:  # noqa: BLE001 - transcription is an enhancement
        logger.warning("[MiniMax H3 Agent] whisper transcription failed: %s", exc)
        return "", f"识别失败: {exc}"


def _annotate_audio_items(
    manifest: list[dict[str, Any]],
    assets: list[MyangMediaAsset],
    model_name: str,
    subject: str,
) -> list[str]:
    """Attach transcripts and subject binding to the audio entries."""
    errors: list[str] = []
    by_slot = {asset.slot: asset for asset in assets}
    for entry in manifest:
        if entry.get("type") != "audio":
            continue
        if subject and subject != "自动":
            entry["subject"] = "纯配乐，无对应人物主体" if subject.startswith("纯配乐") else f"该音频对应 {subject} 中的人物"
        asset = by_slot.get(entry.get("slot"))
        if model_name != "off" and asset is not None:
            text, error = _transcribe_audio(asset.payload, model_name)
            if text:
                entry["transcript"] = text
            if error:
                errors.append(f"{entry.get('tag', '音频')}: {error}")
    return errors


# ---------------------------------------------------------------------------
# Shared with the Director / script splitter
#
# The splitter authors segment prompts, which is the same job the agent does,
# so it needs the same two things: a Skill that says how a prompt is written,
# and a media whitelist grounded in what the media actually shows.  Both are
# exposed here rather than reimplemented so the two nodes cannot drift apart.
# ---------------------------------------------------------------------------

def skill_preset_options() -> list[str]:
    """Skill names for a node dropdown, with ``auto`` routing first."""
    return [SKILL_AUTO] + _skill_names()


def resolve_skill(skill_preset: str, skill_text: str = "", llm_service: str = "",
                  ollama_auto_unload: bool = False, routing_prompt: str = "",
                  budget: int = MAX_SKILL_INJECT_CHARS) -> tuple[str, str]:
    """Return (skill_rules, source_label) exactly the way the agent resolves them.

    ``auto`` routes through the cached one-line index, so choosing a Skill costs
    a few hundred tokens instead of reading every package.  The LLM-written
    digest wins over the raw package: it carries the rules that shape output
    without the pipeline chapters the writer cannot act on.  Pasted rules always
    come first so a user override outranks the packaged Skill.
    """
    resolved = str(skill_preset or "none").strip() or "none"
    choice = resolved
    if resolved == SKILL_AUTO:
        if str(llm_service or "").strip():
            resolved, choice = _select_skill_auto(
                llm_service, routing_prompt, bool(ollama_auto_unload))
        else:
            resolved, choice = "none", "auto: 未配置 LLM 服务"
    pasted = str(skill_text or "").strip()
    learned, learned_label = _learned_skill_text(resolved, budget=budget)
    if learned:
        content, source = learned, learned_label
        if pasted:
            content = "--- User Custom Paste Rules ---\n" + pasted + "\n\n" + content
            source = f"pasted_text, {source}"
    else:
        content, source = _read_skill(resolved, pasted, budget=budget)
    if choice != resolved:
        source = f"{source} [{choice}]"
    return content, source


def media_whitelist(media, vlm_service: str = "off", ollama_auto_unload: bool = False,
                    whisper_model: str = "off",
                    audio_subject: str = "自动") -> tuple[list[dict[str, Any]], list[str]]:
    """Describe a MINIMAX_H3_MEDIA bundle the way the agent describes its own.

    Metadata alone (a filename, a resolution) does not tell a writer whether
    ``<Picture 2>`` is the heroine or a street; with a VLM connected each entry
    also carries what the media actually shows, and audio carries what is
    actually said.  Returns (manifest, errors); errors degrade gracefully so a
    failed description never blocks generation.
    """
    assets = list(media.assets) if isinstance(media, MyangMediaCatalog) else []
    if not assets:
        return [], []
    manifest = _media_manifest(assets)
    errors: list[str] = []
    if str(vlm_service or "off") != "off":
        errors += _describe_media_items(
            manifest, assets, str(vlm_service), bool(ollama_auto_unload))
    errors += _annotate_audio_items(
        manifest, assets, str(whisper_model or "off"), str(audio_subject or "自动"))
    return manifest, errors


class MiniMaxH3MediaAgent:

    CATEGORY = "沐阳 H3"
    FUNCTION = "plan"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "MINIMAX_H3_MEDIA", "STRING")
    RETURN_NAMES = ("agent_prompt", "summary_json", "media_manifest", "media", "myang_prompt")
    DESCRIPTION = "用 LLM 重写 MiniMax H3 提示词，并严格校验每个素材引用都真实连接。"

    @classmethod
    def INPUT_TYPES(cls):
        # asset_* and asset_manifest_json are transport inputs. The browser
        # extension fills them in graphToPrompt and strips them from the node
        # definition, so the media type is always detected from the upstream
        # slot and never typed by hand. They must be declared as real optional
        # inputs: ComfyUI only forwards prompt values for required and optional
        # inputs, and populates "hidden" ones exclusively for its own magic
        # types (PROMPT, UNIQUE_ID, ...). Declaring asset_manifest_json as hidden
        # made the backend drop it, leaving the agent with no filenames, no
        # labels and no media-type hints. asset_manifest_json is the sole metadata
        # channel: filenames cannot be recovered from a decoded tensor.
        optional = {"catalog": ("MINIMAX_H3_MEDIA",)}
        for index in range(1, MAX_ASSETS + 1):
            optional[f"asset_{index}"] = ("*",)
        optional["asset_manifest_json"] = ("STRING", {"default": "[]"})
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": ""}),
                "llm_service": (_llm_service_options(),),
                "skill_preset": ([SKILL_AUTO] + _skill_names(), {"default": SKILL_AUTO}),
                "skill_text": ("STRING", {"multiline": True, "dynamicPrompts": False, "default": ""}),
                "agent_enabled": ("BOOLEAN", {"default": True}),
                "strict_media_check": ("BOOLEAN", {"default": True}),
                "ollama_auto_unload": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                # NOTE: new widgets must be appended at the END. ComfyUI
                # restores saved widget values positionally, so inserting a
                # widget in the middle shifts every later value (including the
                # seed and its control widget) in previously saved workflows.
                "vlm_service": (_vlm_service_options(), {"default": "off"}),
                "时长": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 30.0, "step": 0.5}),
                "扩写": ("BOOLEAN", {"default": True, "label_on": "扩写(展开情节)", "label_off": "严格(只细化)"}),
                "对话修正": ("BOOLEAN", {"default": True, "label_on": "开启台词校正", "label_off": "关闭"}),
                # Audio-only; the frontend hides both until an audio clip is
                # actually connected.
                "音频识别": (list(WHISPER_MODELS), {"default": "off"}),
                "音频主体": (list(AUDIO_SUBJECTS), {"default": "自动"}),
            },
            "optional": optional,
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        seed = kwargs.get("seed", 0)
        seconds = kwargs.get("时长", kwargs.get("seconds", 5.0))
        prompt_hash = hashlib.sha256(str(kwargs.get("prompt", "")).encode("utf-8")).hexdigest()
        skill_hash = hashlib.sha256(str(kwargs.get("skill_text", "")).encode("utf-8")).hexdigest()
        media_state = []
        for index in range(1, MAX_ASSETS + 1):
            value = kwargs.get(f"asset_{index}")
            if value is None:
                continue
            media_state.append((index, classify_payload(value), id(value)))
        return hashlib.sha256(repr((seed, seconds, prompt_hash, skill_hash, kwargs.get("skill_preset"), kwargs.get("vlm_service"), kwargs.get("asset_manifest_json"), kwargs.get("扩写"), kwargs.get("对话修正"), kwargs.get("音频识别"), kwargs.get("音频主体"), media_state)).encode()).hexdigest()

    @staticmethod
    def _collect_media(kwargs: dict, metadata: dict[int, dict[str, Any]] | None = None) -> list[MyangMediaAsset]:
        assets: list[MyangMediaAsset] = []
        catalog = kwargs.get("catalog")
        if isinstance(catalog, MyangMediaCatalog):
            assets.extend(catalog.assets)
        occupied = {asset.slot for asset in assets}
        for index in range(1, MAX_ASSETS + 1):
            value = kwargs.get(f"asset_{index}")
            if value is None:
                continue
            slot = index
            while slot in occupied:
                slot += MAX_ASSETS
            assets.append(asset_from_input(slot, value, (metadata or {}).get(index)))
            occupied.add(slot)
        return assets

    @classmethod
    def plan(
        cls,
        prompt,
        llm_service,
        skill_preset,
        skill_text,
        agent_enabled,
        strict_media_check,
        ollama_auto_unload,
        seed,
        vlm_service="off",
        **kwargs,
    ):
        seconds = kwargs.get("时长", kwargs.get("seconds", 5.0))
        expand = bool(kwargs.get("扩写", True))
        dialogue_check = bool(kwargs.get("对话修正", True))
        asset_manifest_json = str(kwargs.get("asset_manifest_json") or "[]")
        metadata = parse_asset_manifest(asset_manifest_json)
        assets = cls._collect_media(kwargs, metadata=metadata)
        manifest = _media_manifest(assets)
        # Hand the resolved media straight to Myang conditioning so both ends agree
        # on the ordering the <Picture n> tags were written against.
        media_bundle = MyangMediaCatalog(tuple(assets))
        original_prompt = _normalize_editor_syntax(str(prompt or ""), manifest)

        if not agent_enabled:
            checked_prompt = _validate_output(original_prompt, manifest) if strict_media_check else _sanitize_llm_output(original_prompt)
            summary_json = _generate_summary_json(checked_prompt, manifest, seconds=seconds)
            editor_prompt = _editor_prompt(checked_prompt, manifest)
            return {
                # Broadcast the editor-syntax prompt so downstream Myang nodes
                # can render it with media chips. ComfyUI sends "executed" for
                # any node that returns a ui payload, not just output nodes.
                "ui": {"myang_prompt": [editor_prompt]},
                "result": (
                    checked_prompt,
                    summary_json,
                    json.dumps(manifest, ensure_ascii=False, indent=2),
                    media_bundle,
                    editor_prompt,
                ),
            }

        # "auto" routes through the cached Skill index: one short call that only
        # sees names, tags and summaries, so picking a Skill costs a few hundred
        # tokens instead of reading every package.
        resolved_preset = str(skill_preset or "none")
        skill_choice = resolved_preset
        if resolved_preset == SKILL_AUTO:
            resolved_preset, skill_choice = _select_skill_auto(llm_service, original_prompt, bool(ollama_auto_unload))

        # Prefer a learned digest. File-only memory is still capped to the same
        # runtime budget: feeding a 20k Skill into every segment multiplies TPM
        # cost and inflates each socket timeout without adding useful rules.
        learned, learned_label = _learned_skill_text(
            resolved_preset, budget=MAX_SKILL_INJECT_CHARS)
        if learned:
            skill_content, skill_source = learned, learned_label
            if skill_text and str(skill_text).strip():
                skill_content = "--- User Custom Paste Rules ---\n" + str(skill_text).strip() + "\n\n" + skill_content
                skill_source = f"pasted_text, {skill_source}"
        else:
            skill_content, skill_source = _read_skill(resolved_preset, skill_text, budget=MAX_SKILL_INJECT_CHARS)
        if skill_choice != resolved_preset:
            skill_source = f"{skill_source} [{skill_choice}]"

        # Vision stage: let a VLM actually look at the connected images/videos
        # so the agent writes prompts grounded in real media content instead of
        # a bare tag whitelist. Audio has no visual channel; its duration is
        # already part of the manifest metadata.
        vision_errors: list[str] = []
        if str(vlm_service) != "off" and manifest:
            vision_errors = _describe_media_items(manifest, items, str(vlm_service), bool(ollama_auto_unload))

        # Audio stage: transcribe speech and bind the clip to a subject so the
        # writer references what is actually said instead of inventing lines.
        audio_errors = _annotate_audio_items(
            manifest, items, str(kwargs.get("音频识别", "off")), str(kwargs.get("音频主体", "自动"))
        )

        # Sub-agent pipeline: plan -> write -> verify -> summarise. Each stage
        # sees only what it needs, so the Skill text cannot leak into the
        # storyboard and a Skill's example timeline cannot decide the duration.
        shot_plan = _plan_shots(
            original_prompt, manifest, seconds, str(llm_service), bool(ollama_auto_unload), int(seed)
        )

        system_prompt = _build_agent_system_prompt(skill_content, seconds=seconds, expand=expand)
        user_prompt = _build_agent_user_prompt(original_prompt, manifest, seconds=seconds, plan=shot_plan)
        # Ladder: full Skill memory, then a budgeted Skill, then its compact
        # executable contract.  A previous fallback removed the Skill entirely;
        # that made the call look successful while returning a generic paragraph.
        # Every retry now preserves the selected Skill's format.
        variants = [(user_prompt, system_prompt)]
        if len(skill_content) > MAX_SKILL_INJECT_CHARS:
            trimmed, _ = _read_skill(resolved_preset, skill_text, budget=MAX_SKILL_INJECT_CHARS)
            if trimmed and len(trimmed) < len(skill_content):
                variants.append((
                    user_prompt,
                    _build_agent_system_prompt(trimmed, seconds=seconds, expand=expand),
                ))
        variants.append((
            user_prompt,
            _build_agent_system_prompt(
                compact_contract if skill_contract["active"] else "",
                seconds=seconds,
                expand=expand,
            ),
        ))

        generated, writer_notes = _call_llm_resilient(
            str(llm_service), variants, bool(ollama_auto_unload), int(seed), "改写",
            validator=(
                (lambda value: skill_output_issues(value, skill_content))
                if skill_contract["active"] else None
            ),
        )
        writer_failed = not generated
        if writer_failed:
            if skill_contract["active"]:
                detail = "；".join(writer_notes[-6:]) or "LLM 未返回符合技能格式的正文"
                raise ValueError(
                    "Agent 调用失败：LLM 没有返回任何可用正文。"
                    "这不是技能格式诊断造成的阻断；系统不会删除已选择的技能。\n"
                    "调用记录：" + detail)
            # Every rung failed. Falling back to the user's own prompt keeps the
            # graph running with something usable instead of aborting the queue;
            # writer_notes carries the reason into media_manifest.
            logger.warning("[MiniMax H3 Agent] writer exhausted all fallbacks; using the original prompt")
            writer_notes.append("改写全部失败，已退回原始提示词（直通模式）")
            generated = original_prompt
        coverage_notes: list[str] = []
        if shot_plan and not writer_failed:
            generated, coverage_notes = _ensure_shot_coverage(
                generated, shot_plan, seconds, str(llm_service), bool(ollama_auto_unload), int(seed)
            )
        elif writer_failed:
            # The LLM is clearly unavailable right now; asking it to "fill in
            # the missing shots" of the user's raw prompt would just burn two
            # more doomed calls.
            coverage_notes = ["改写已失败，跳过分镜补全"]
        generated = _normalize_editor_syntax(generated, manifest)
        # Dialogue sub-agent: a deterministic audit first (language + whether the
        # line can physically be spoken in its shot), then one repair call only
        # if that audit found something.
        dialogue_report: dict[str, Any] = {}
        dialogue_notes: list[str] = []
        if dialogue_check:
            expected_script = dialogue_audit.detect_script(original_prompt)
            dialogue_report = dialogue_audit.audit(
                generated, seconds, expected_script=expected_script, plan=shot_plan
            )
            dialogue_report["expected_script"] = expected_script
            dialogue_report["plan"] = shot_plan
            if dialogue_report.get("issues"):
                generated, dialogue_notes = _fix_dialogue_via_agent(
                    generated, dialogue_report, seconds, str(llm_service), bool(ollama_auto_unload), int(seed)
                )
                generated = _normalize_editor_syntax(generated, manifest)
        final_skill_issues = skill_output_issues(generated, skill_content)
        if final_skill_issues:
            raise ValueError(
                "Agent 输出在后处理后不再符合写作技能格式，已停止："
                + "；".join(final_skill_issues))
        if strict_media_check:
            generated = _validate_output(generated, manifest)
        # Final sub-agent: it only ever sees the finished prompt (plus the plan
        # it was written against), so the storyboard follows the real shot
        # breakdown instead of regex sentence-splitting, and it cannot alter the
        # prompt itself.
        summary_json = _summary_via_agent(
            generated, manifest, seconds, str(llm_service), bool(ollama_auto_unload), int(seed), plan=shot_plan
        ) or _generate_summary_json(generated, manifest, seconds=seconds)
        manifest_report = {
            "skill_source": skill_source,
            "skill_compliance": {
                "enforced": bool(skill_contract["active"]),
                "passed": not final_skill_issues,
                "contract": compact_contract,
            },
            "strict_media_check": bool(strict_media_check),
            "vlm_service": str(vlm_service),
            "vision_errors": vision_errors,
            "audio_asr": str(kwargs.get("音频识别", "off")),
            "audio_subject": str(kwargs.get("音频主体", "自动")),
            "audio_errors": audio_errors,
            "shot_plan": shot_plan or [],
            "writer_notes": writer_notes,
            "coverage_notes": coverage_notes,
            "expand_mode": "expand" if expand else "strict",
            "dialogue_check": dialogue_check,
            "dialogue_report": dialogue_audit.report_text(dialogue_report) if dialogue_report else "未开启对话修正",
            "dialogue_notes": dialogue_notes,
            "media": manifest,
            "asset_catalog": manifest,
        }
        editor_prompt = _editor_prompt(generated, manifest)
        return {
            "ui": {"myang_prompt": [editor_prompt]},
            "result": (
                generated,
                summary_json,
                json.dumps(manifest_report, ensure_ascii=False, indent=2),
                media_bundle,
                editor_prompt,
            ),
        }


DEFAULT_VIEWER_TEMPLATE = """=========================================
 🎬 MiniMax H3 最终重写提示词 (Agent Prompt 示例)
=========================================
[Shot 1] 实拍风格，电影质感画面。中景拍摄两个人从初始构图开始打架。两人互相推搡、挥拳，侧身闪避。镜头缓慢推进，捕捉动作细节。

=========================================
 📊 MiniMax H3 剧情总结与分镜预览表 (Storyboard 示例)
=========================================

【1. 基础设置与素材对照】
• 画面风格：写实电影质感，4K 高清
• 场景设定：武操场 / 练习室
• 关联素材清单：
  • <Picture 1> ➔ 男主 (正面动作参考)
  • <Picture 2> ➔ 对手 (侧面动作参考)

【2. 时间轴动作分镜描述】
• 0.0s - 2.5s [中景推进] 两人面对面站立，开始互相推搡爆发冲突。
• 2.5s - 5.0s [特写跟拍] 拳脚碰撞细节，侧身闪避，镜头定格在最终姿势。

【3. 多模态音效与台词】
• 环境音：拳脚碰撞声、衣服摩擦声
• BGM：无背景音乐
• 台词：无台词
========================================="""


class MiniMaxH3Viewer:
    CATEGORY = "沐阳 H3"
    FUNCTION = "view"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("preview_text",)
    DESCRIPTION = "Unified visual prompt & summary viewer node for MiniMax H3."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True,
                    "default": DEFAULT_VIEWER_TEMPLATE,
                    "dynamicPrompts": False,
                }),
            },
            "optional": {
                "agent_prompt": ("STRING", {"multiline": True, "dynamicPrompts": False, "default": "", "forceInput": True}),
                "summary_json": ("STRING", {"multiline": True, "dynamicPrompts": False, "default": "", "forceInput": True}),
                "myang_prompt": ("STRING", {"multiline": True, "dynamicPrompts": False, "default": "", "forceInput": True}),
            }
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # A preview node is useless when it is cached: ComfyUI only sends the
        # "executed" message (which carries the ui payload the frontend renders)
        # for nodes that actually run, so a cache hit leaves the panel showing
        # the previous run's text.
        return float("nan")

    def view(self, text: str = "", agent_prompt: str = "", summary_json: str = "", myang_prompt: str = ""):
        raw_prompt = str(agent_prompt or "").strip()
        raw_summary = str(summary_json or "").strip()
        raw_myang = str(myang_prompt or "").strip()

        data = {}
        if raw_summary:
            try:
                data = json.loads(raw_summary)
            except (json.JSONDecodeError, TypeError):
                data = {}

        display_prompt = raw_prompt

        sections = []

        if display_prompt:
            sections.append(
                "=========================================\n"
                " 【第 1 页】🎬 MiniMax H3 最终重写提示词 (Agent Prompt)\n"
                "=========================================\n"
                f"{display_prompt}"
            )

        if raw_myang:
            sections.append(
                "=========================================\n"
                " 【第 2 页】✏️ 沐阳编辑提示词\n"
                " @图片1 / @视频1 / @音频1 可直接接入沐阳 H3 节点\n"
                "=========================================\n"
                f"{raw_myang}"
            )

        if raw_summary:
            basic = data.get("basic_setup", {}) if isinstance(data, dict) else {}
            style = str(basic.get("style") or "高清写实，影院级质感")
            scene = str(basic.get("scene") or "未设定场景")
            characters = str(basic.get("characters") or "• 未连接外部参考素材")

            timeline_items = data.get("timeline", []) if isinstance(data, dict) else []
            timeline_lines = []
            if isinstance(timeline_items, list):
                for item in timeline_items:
                    if isinstance(item, dict):
                        t_range = item.get("time_range", "")
                        cam = item.get("camera_movement", "")
                        content = item.get("content", "")
                        timeline_lines.append(f"• {t_range} [{cam}] {content}")
            timeline_text = "\n".join(timeline_lines) if timeline_lines else "暂无时间轴分段"

            multi = data.get("multimodal", {}) if isinstance(data, dict) else {}
            ambient = str(multi.get("ambient_sound") or "无")
            bgm = str(multi.get("bgm") or "无")
            dialogue = str(multi.get("dialogue") or "无台词")

            multimodal_text = f"• 环境音：{ambient}\n• BGM：{bgm}\n• 台词：{dialogue}"

            summary_formatted = (
                "=========================================\n"
                " 【第 3 页】📊 MiniMax H3 剧情总结与分镜预览表\n"
                "=========================================\n\n"
                "【1. 基础设置与素材对照】\n"
                f"• 画面风格：{style}\n"
                f"• 场景设定：{scene}\n"
                f"• 关联素材清单：\n  {characters}\n\n"
                "【2. 时间轴动作分镜描述】\n"
                f"{timeline_text}\n\n"
                "【3. 多模态音效与台词】\n"
                f"{multimodal_text}"
            )
            sections.append(summary_formatted)

        if not sections:
            formatted = str(text or DEFAULT_VIEWER_TEMPLATE)
        else:
            formatted = "\n\n".join(sections) + "\n========================================="

        return {
            "ui": {
                "text": [formatted],
                "prompt": [display_prompt],
                "summary_json": [raw_summary],
                "summary_data": [data if isinstance(data, dict) else {}],
            },
            "result": (formatted,),
        }


MiniMaxH3SummaryViewer = MiniMaxH3Viewer


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3MediaAgent": MiniMaxH3MediaAgent,
    "MiniMaxH3Viewer": MiniMaxH3Viewer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3MediaAgent": "沐阳 H3 · Media Agent",
    "MiniMaxH3Viewer": "沐阳 H3 · Agent 预览",
}
