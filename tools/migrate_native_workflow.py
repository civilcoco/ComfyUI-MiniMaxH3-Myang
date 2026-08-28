"""Safely migrate an existing Myang loop workflow to the native Ref2VA path.

This is intentionally a surgical, idempotent migration rather than a rebuild:
all user-added nodes, links, groups, positions and properties are retained.
"""

import argparse
import copy
import json
import os
import shutil
from pathlib import Path


HERE = Path(__file__).resolve().parent
COMFY = HERE.parent.parent.parent
DEFAULT_WORKFLOW = (COMFY / "user" / "default" / "workflows" / "minimax"
                    / "minimax-沐阳Myang-长视频循环版.json")
REF2VA = "minimax\\minimax_h3_ref2va_int8_convrot.safetensors"
KNOWN_OLD = {
    "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
    "minimax\\minimax_h3_fl2va_pruned_int8_convrot.safetensors",
    "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
    "minimax\\minimax_h3_ref2va_pruned_int8_convrot.safetensors",
}


def migrate(path):
    path = Path(path).resolve()
    target_model = COMFY / "models" / "diffusion_models" / Path(
        REF2VA.replace("\\", "/"))
    if not target_model.is_file():
        raise FileNotFoundError("目标 Ref2VA 不存在：%s" % target_model)
    original_text = path.read_text(encoding="utf-8-sig")
    data = json.loads(original_text)
    before = copy.deepcopy(data)

    loaders = [node for node in data.get("nodes", [])
               if node.get("type") == "H3Loader"]
    if len(loaders) != 1:
        raise RuntimeError("需要且只能有一个 H3Loader，实际 %d 个" % len(loaders))
    widgets = loaders[0].get("widgets_values")
    if not isinstance(widgets, list) or not widgets:
        raise RuntimeError("H3Loader 没有 diffusion_model widget")
    old = str(widgets[0])
    model_changed = old != REF2VA
    if model_changed and old not in KNOWN_OLD:
        raise RuntimeError("拒绝覆盖未知 H3Loader 模型：%s" % old)
    if model_changed:
        widgets[0] = REF2VA

    expected = copy.deepcopy(before)
    if model_changed:
        expected_loader = next(node for node in expected["nodes"]
                               if node.get("id") == loaders[0].get("id"))
        expected_loader["widgets_values"][0] = REF2VA
    if data != expected:
        raise RuntimeError("迁移改变了获准字段以外的 JSON 内容")

    if not model_changed:
        return (False, len(data.get("nodes", [])),
                len(data.get("links", [])), old)

    backup = path.with_suffix(path.suffix + ".before-ref2va.bak")
    if not backup.exists():
        shutil.copy2(path, backup)
    temporary = path.with_suffix(path.suffix + ".native-anchor.tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    written = json.loads(temporary.read_text(encoding="utf-8"))
    if written != expected:
        temporary.unlink(missing_ok=True)
        raise RuntimeError("写回前的 JSON 校验失败")
    os.replace(temporary, path)
    return (True, len(data.get("nodes", [])), len(data.get("links", [])), old)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("workflow", nargs="?", default=str(DEFAULT_WORKFLOW))
    args = parser.parse_args()
    changed, nodes, links, old = migrate(args.workflow)
    if changed:
        print("已迁移 H3Loader: %s -> %s" % (old, REF2VA))
    else:
        print("H3Loader 已使用目标 Ref2VA，无需修改")
    print("节点 %d、连线 %d，全部保留" % (nodes, links))


if __name__ == "__main__":
    main()
