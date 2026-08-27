"""Standard-library release hygiene checks for the public source tree."""

from __future__ import annotations

import argparse
import json
import re
import sys
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {
    ".css", ".html", ".ini", ".js", ".json", ".md", ".mjs", ".py",
    ".toml", ".txt", ".yaml", ".yml",
}
MEDIA_SUFFIXES = {".gif", ".jpeg", ".jpg", ".m4a", ".mov", ".mp3", ".mp4", ".png", ".wav", ".webm", ".webp"}
REQUIRED_FILES = {
    "LICENSE", "NOTICE", "README.md", "LEGAL.md", "SECURITY.md",
    "THIRD_PARTY_NOTICES.md", "pyproject.toml", ".comfyignore",
}
IGNORED_PARTS = {".git", "__pycache__", ".pytest_cache"}
RUNTIME_DEPENDENCY_FILES = {
    "director.py", "nodes_v2.py", "agent_nodes.py",
    "web/h3_director_ui.js", "web/h3_longvideo_ui.js",
    "web/minimax_h3_myang_agent_ui.js",
}
FORBIDDEN_EXTERNAL_NODE_IDS = re.compile(
    r"H3(?:ContactSheet(?:Decode)?|JerkOracle|TimeSmear|V2VInit|"
    r"InjectSchedule|ExactRecover|AudioRecover|FaceTrackCrop|"
    r"InjectVideoLatent|PerFrameDenoise|FaceStitch)"
)
FORBIDDEN_LEGACY_MEDIA_PROTOCOL = re.compile(
    r"ComfyUI-MiniMaxH3-Easy|nkxx188|MiniMaxH3MediaBundle|_MediaInput|"
    r"__MINIMAX_H3_REF_|media_links_json|easy_prompt|"
    r"minimax_h3_agent_media_connections",
    re.I,
)


def public_files():
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(part in IGNORED_PARTS for part in path.parts):
            continue
        yield path


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strict-metadata", action="store_true",
        help="fail while the GitHub owner or Comfy publisher placeholders remain")
    args = parser.parse_args()
    errors: list[str] = []
    warnings: list[str] = []

    for name in sorted(REQUIRED_FILES):
        if not (ROOT / name).is_file():
            errors.append(f"missing required release file: {name}")

    pyproject_path = ROOT / "pyproject.toml"
    if pyproject_path.is_file():
        try:
            metadata = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
            project = metadata["project"]
            comfy = metadata["tool"]["comfy"]
            if project.get("version") != "0.1.0":
                errors.append("pyproject version must match the planned v0.1.0 release")
            placeholder_fields = []
            if any("REPLACE_WITH_" in str(value)
                   for value in (project.get("urls") or {}).values()):
                placeholder_fields.append("GitHub owner")
            if "REPLACE_WITH_" in str(comfy.get("PublisherId", "")):
                placeholder_fields.append("Comfy PublisherId")
            if placeholder_fields:
                message = (
                    "replace " + " and ".join(placeholder_fields)
                    + " before publishing"
                )
                (errors if args.strict_metadata else warnings).append(message)
        except (KeyError, tomllib.TOMLDecodeError) as exc:
            errors.append(f"invalid Comfy pyproject metadata: {exc}")

    allowed_skill_files = {"skills/README.md"}
    actual_skill_files = {
        relative(path) for path in (ROOT / "skills").rglob("*") if path.is_file()
    } if (ROOT / "skills").is_dir() else set()
    unexpected_skills = sorted(actual_skill_files - allowed_skill_files)
    if unexpected_skills:
        errors.append("unreviewed bundled Skill content: " + ", ".join(unexpected_skills))

    forbidden_names = {"_skill_index.json", "_skill_memory.json"}
    for path in public_files():
        rel = relative(path)
        if path.name in forbidden_names or path.suffix.lower() in {".pyc", ".pyo"}:
            errors.append(f"runtime cache included: {rel}")
        if path.suffix.lower() in MEDIA_SUFFIXES:
            errors.append(f"media asset requires a separate rights review: {rel}")
        if path.suffix.lower() == ".json":
            try:
                json.loads(path.read_text(encoding="utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                errors.append(f"invalid JSON {rel}: {exc}")

        if path.suffix.lower() not in TEXT_SUFFIXES or rel == "tools/release_audit.py":
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            errors.append(f"non-UTF-8 text file: {rel}")
            continue
        checks = {
            "Windows absolute path": re.compile(r"[A-Za-z]:\\\\(?:Users|AI-PAINTING|ComfyUI)"),
            "chat attachment query": re.compile(r"(?:MsgID|skey)=|@crypt_", re.I),
            "watermark-removal instruction": re.compile(r"(?:去掉|移除|消除).{0,12}水印"),
            "known private material filename": re.compile(r"Mosi_Image|zit202|鸣潮舞蹈|彩叶5s", re.I),
        }
        for label, pattern in checks.items():
            if pattern.search(text):
                errors.append(f"{label} found in {rel}")
        if FORBIDDEN_LEGACY_MEDIA_PROTOCOL.search(text):
            errors.append(f"retired media protocol or attribution found in {rel}")
        if rel in RUNTIME_DEPENDENCY_FILES and FORBIDDEN_EXTERNAL_NODE_IDS.search(text):
            errors.append(f"external custom-node runtime call found in {rel}")

    if errors:
        print("RELEASE AUDIT FAILED")
        for item in sorted(set(errors)):
            print(f"ERROR: {item}")
    else:
        print("RELEASE AUDIT PASSED")
    for item in sorted(set(warnings)):
        print(f"WARNING: {item}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
