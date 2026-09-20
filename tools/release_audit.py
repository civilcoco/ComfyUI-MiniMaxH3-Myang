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
MEDIA_SUFFIXES = {
    ".gif", ".jpeg", ".jpg", ".m4a", ".mov", ".mp3", ".mp4", ".png",
    ".wav", ".webm", ".webp",
}
REQUIRED_FILES = {
    ".comfyignore", ".gitignore", "CHANGELOG.md", "CONTRIBUTING.md", "LEGAL.md",
    "LICENSE", "NOTICE", "README.md", "SECURITY.md", "THIRD_PARTY_NOTICES.md",
    "pyproject.toml", "tools/audit_native_media.py", "tools/run_tests.ps1",
    "tools/run_test_file.py", "tools/run_suite.py",
}
REQUIRED_COMFY_EXCLUDES = {
    ".github/", "docs/research/", "skills/", "tests/", "tools/",
}
IGNORED_PARTS = {".git", "__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache"}
LOCAL_ONLY_PREFIXES = ("docs/research/", "skills/")
ALLOWED_LOCAL_FILES = {"skills/.gitignore", "skills/README.md"}
OPTIONAL_NODE_DEPENDENCIES = {
    "ComfyUI-MAINodes": re.compile(
        r"H3(?:ContactSheet(?:Decode)?|JerkOracle|TimeSmear|V2VInit|"
        r"InjectSchedule|ExactRecover|AudioRecover)"),
    "ComfyUI-H3-FaceRefine": re.compile(
        r"H3(?:FaceTrackCrop|InjectVideoLatent|PerFrameDenoise|FaceStitch)"),
}


def source_files():
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(part in IGNORED_PARTS for part in path.parts):
            continue
        yield path


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def is_local_only(rel: str) -> bool:
    return rel not in ALLOWED_LOCAL_FILES and rel.startswith(LOCAL_ONLY_PREFIXES)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strict-metadata", action="store_true",
        help="fail while repository or Comfy publisher placeholders remain",
    )
    parser.add_argument(
        "--strict-local", action="store_true",
        help="fail when ignored local Skills or research artifacts exist (used by CI)",
    )
    parser.add_argument(
        "--strict-private", action="store_true",
        help="fail when private self-use material remains in a public staging tree",
    )
    args = parser.parse_args()
    errors: list[str] = []
    warnings: list[str] = []
    readme_text = (ROOT / "README.md").read_text(encoding="utf-8")

    for name in sorted(REQUIRED_FILES):
        if not (ROOT / name).is_file():
            errors.append(f"missing required release file: {name}")

    comfyignore_path = ROOT / ".comfyignore"
    if comfyignore_path.is_file():
        comfy_excludes = {
            line.strip() for line in comfyignore_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        for pattern in sorted(REQUIRED_COMFY_EXCLUDES - comfy_excludes):
            errors.append(f".comfyignore must exclude release-only/local path: {pattern}")

    pyproject_path = ROOT / "pyproject.toml"
    if pyproject_path.is_file():
        try:
            metadata = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
            project = metadata["project"]
            comfy = metadata["tool"]["comfy"]
            version = str(project.get("version") or "")
            if not re.fullmatch(r"0\.[0-9]+\.[0-9]+", version):
                errors.append("pyproject version must be a valid public semver")
            placeholders = []
            if any("REPLACE_WITH_" in str(value) for value in (project.get("urls") or {}).values()):
                placeholders.append("GitHub owner")
            if "REPLACE_WITH_" in str(comfy.get("PublisherId", "")):
                placeholders.append("Comfy PublisherId")
            if placeholders:
                message = "replace " + " and ".join(placeholders) + " before publishing"
                (errors if args.strict_metadata else warnings).append(message)
        except (KeyError, tomllib.TOMLDecodeError) as exc:
            errors.append(f"invalid Comfy pyproject metadata: {exc}")

    local_only = sorted(relative(path) for path in source_files() if is_local_only(relative(path)))
    if local_only:
        message = (
            "ignored local content is present and must not be force-added to a release: "
            + ", ".join(local_only)
        )
        (errors if args.strict_local else warnings).append(message)

    private_checks = {
        "Windows absolute path": re.compile(r"[A-Za-z]:\\(?:Users|AI-PAINTING|ComfyUI)"),
        "chat attachment query": re.compile(r"(?:MsgID|skey)=|@crypt_", re.I),
        "watermark-removal instruction": re.compile(r"(?:去掉|移除|消除).{0,12}水印"),
        "known private material filename": re.compile(r"Mosi_Image|zit202|鸣潮舞蹈|彩叶5s", re.I),
        "probable API secret": re.compile(r"\b(?:sk|ak)-[A-Za-z0-9_-]{24,}\b"),
    }

    for path in source_files():
        rel = relative(path)
        if is_local_only(rel):
            continue
        suffix = path.suffix.lower()
        if path.name in {"_skill_index.json", "_skill_memory.json"} or suffix in {".pyc", ".pyo"}:
            errors.append(f"runtime cache included: {rel}")
        if suffix in MEDIA_SUFFIXES:
            errors.append(f"media asset requires a separate rights review: {rel}")
        if suffix == ".json":
            try:
                json.loads(path.read_text(encoding="utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                errors.append(f"invalid JSON {rel}: {exc}")
        if suffix not in TEXT_SUFFIXES or rel == "tools/release_audit.py":
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            errors.append(f"non-UTF-8 text file: {rel}")
            continue
        for label, pattern in private_checks.items():
            if pattern.search(text):
                message = f"{label} found in {rel}"
                (errors if args.strict_private else warnings).append(message)
        for package, pattern in OPTIONAL_NODE_DEPENDENCIES.items():
            if pattern.search(text) and package not in readme_text:
                errors.append(
                    f"optional custom-node dependency {package} used but not "
                    f"documented in README.md ({rel})")

    director_text = (ROOT / "director.py").read_text(encoding="utf-8")
    if "H3ContactSheet" in director_text and "ComfyUI-MAINodes" not in readme_text:
        errors.append("optional H3ContactSheet dependency is not documented in README.md")

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
