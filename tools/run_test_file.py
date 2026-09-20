"""Run every test_* function in one test file; used by tools/run_tests.ps1.

Usage: python tools/run_test_file.py <ComfyUI root> <test file>

A test may raise the file's own ``SkipTest`` (if it defines one) to report an
absent optional dependency; that counts as a skip, not a failure.
"""
import asyncio
import inspect
import pathlib
import runpy
import sys
import os
import tempfile
import importlib


def main(argv):
    if len(argv) != 2:
        print("usage: run_test_file.py <comfy_root> <test_file>")
        return 2
    sys.path.insert(0, str(pathlib.Path(argv[0]).resolve()))
    path = str(pathlib.Path(argv[1]).resolve())
    # Configure CPU and private state before any custom-node imports.
    import comfy.options
    comfy.options.enable_args_parsing()
    sys.argv = [path, "--cpu"]
    # Windows Defender/sandbox can deny re-opening a freshly-created directory
    # under AppData\Local\Temp. Keep disposable state beside the clean source
    # snapshot instead; the suite snapshot itself is disposable.
    state_root = pathlib.Path(path).resolve().parents[1] / ".test-state"
    state_root.mkdir(exist_ok=True)
    directory = pathlib.Path(tempfile.mkdtemp(prefix="case-", dir=state_root))
    return run_tests(path, directory)


def run_tests(path, state):
    import folder_paths
    for name in ("input", "output", "temp", "user"):
        directory = state / name
        directory.mkdir(exist_ok=True)
        setattr(folder_paths, name + "_directory", str(directory))
    os.environ.pop("MINIMAX_H3_LEGACY_SKILL_MEMORY", None)
    os.environ.pop("MINIMAX_H3_SKILLS_DIR", None)
    sys.path.insert(0, str(pathlib.Path(path).parents[2]))
    agent = importlib.import_module("ComfyUI-MiniMaxH3-Myang.agent_nodes")
    agent.SKILL_DIR = state / "skills"
    agent.EXTRA_SKILLS_DIR = None
    # Forbid accidental writes to a real user's state, even from a regression.
    def audit(event, values):
        if event == "socket.connect":
            raise PermissionError("Regression tests must mock network requests")
        paths = []
        if event == "open":
            target, mode, flags = values
            if (isinstance(mode, str) and any(c in mode for c in "wax+")) or (
                    flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)):
                paths = [target]
        elif event in ("os.remove", "os.rmdir", "os.mkdir", "os.chmod", "os.utime"):
            paths = values[:1]
        elif event in ("os.rename", "os.link", "os.symlink"):
            paths = values[:2]
        package = pathlib.Path(path).resolve().parents[1]
        for target in paths:
            if not isinstance(target, (str, bytes, os.PathLike)):
                continue
            resolved = pathlib.Path(os.fsdecode(target)).resolve()
            if os.fsdecode(target).lower() == os.devnull.lower():
                continue
            if not (resolved.is_relative_to(state) or resolved.is_relative_to(package)):
                raise PermissionError(f"Test write outside isolation: {resolved}")
    tempfile.tempdir = str(state / "temp")
    sys.addaudithook(audit)
    namespace = runpy.run_path(path)
    skip_type = namespace.get("SkipTest")
    tests = [(name, fn) for name, fn in sorted(namespace.items())
             if name.startswith("test_") and callable(fn)]
    skipped = 0
    for name, fn in tests:
        try:
            if inspect.iscoroutinefunction(fn):
                asyncio.run(fn())
            else:
                fn()
        except Exception as error:  # noqa: BLE001 - re-raised unless a skip
            if skip_type is not None and isinstance(error, skip_type):
                skipped += 1
                print("SKIP", name, "-", error)
                continue
            raise
    print("PASS", pathlib.Path(path).name, len(tests) - skipped,
          "skipped", skipped)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
