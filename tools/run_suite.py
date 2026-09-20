"""Run a clean source snapshot, without private Skills or live ComfyUI state."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comfy-root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1]
    root = (args.output or Path(tempfile.mkdtemp(prefix="myang-suite-"))).resolve()
    root.mkdir(parents=True, exist_ok=True)
    copy = root / source.name
    if copy.exists():
        parser.error("output already contains a snapshot; choose a fresh directory")
    names = subprocess.check_output(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=source).decode("utf-8").split("\0")
    for name in sorted(set(names)):
        if not name or not (source / name).is_file():
            continue
        if name.startswith(("skills/", "docs/research/")) and name not in {
                "skills/README.md", "skills/.gitignore"}:
            continue
        dest = copy / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, dest)
    logs = root / "logs"
    logs.mkdir()
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    tasks = [(p.name, [sys.executable, "-I", "-B",
              str(copy / "tools/run_test_file.py"), str(args.comfy_root.resolve()), str(p)])
             for p in sorted((copy / "tests").glob("test_*.py"))]
    node = shutil.which("node")
    if not node:
        parser.error("Node.js is required to validate all frontend tests")
    tasks += [(p.name, [node, str(p)]) for p in sorted((copy / "tests").glob("test_*.mjs"))]
    tasks += [(name, [sys.executable, "-I", "-B", str(copy / "tools" / name)] + options)
              for name, options in [
                  ("audit_native_media.py", []),
                  ("release_audit.py", ["--strict-local", "--strict-private", "--strict-metadata"])]]
    results = []
    for name, command in tasks:
        with (logs / (name + ".log")).open("w", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=copy, env=env, stdout=log,
                                    stderr=subprocess.STDOUT)
        results.append({"name": name, "exit": result.returncode})
        print(("PASS " if result.returncode == 0 else "FAIL ") + name, flush=True)
    (root / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("Results:", root)
    return int(any(item["exit"] for item in results))


if __name__ == "__main__":
    sys.exit(main())
