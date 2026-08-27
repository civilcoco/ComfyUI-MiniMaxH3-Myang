"""Migrate an existing Myang workflow to the combined Turbo LoRA loader.

The migration is deliberately surgical: it preserves every node and link ID,
keeps the old LoraLoaderModelOnly node as an unconnected fallback, moves that
node's saved file/strength into H3TurboSchedule, and feeds Turbo from the same
base model that previously fed the old loader.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
from pathlib import Path


HERE = Path(__file__).resolve().parent
COMFY = HERE.parent.parent.parent
DEFAULT_WORKFLOW = (COMFY / "user" / "default" / "workflows" / "minimax"
                    / "minimax-沐阳Myang-长视频循环版.json")
PROFILE_AUTO = "自动匹配 LoRA 文件（推荐）"
WIDGET_NAMES = [
    "profile", "speed_cache", "shift_video", "shift_audio",
    "recommended_steps", "LoRA文件", "LoRA强度",
]


def _one(data, node_type):
    nodes = [node for node in data.get("nodes", [])
             if node.get("type") == node_type]
    if len(nodes) != 1:
        raise RuntimeError("需要且只能有一个 %s，实际 %d 个" %
                           (node_type, len(nodes)))
    return nodes[0]


def _link(data, link_id):
    found = [link for link in data.get("links", [])
             if isinstance(link, list) and link[0] == link_id]
    if len(found) != 1:
        raise RuntimeError("连线 %s 数量异常：%d" % (link_id, len(found)))
    return found[0]


def _output(node, slot):
    outputs = node.get("outputs") or []
    if not 0 <= int(slot) < len(outputs):
        raise RuntimeError("节点 %s 输出槽 %s 不存在" % (node.get("id"), slot))
    return outputs[int(slot)]


def _remove_link_cache(output, link_id):
    output["links"] = [value for value in (output.get("links") or [])
                       if value != link_id]


def _add_link_cache(output, link_id):
    links = list(output.get("links") or [])
    if link_id not in links:
        links.append(link_id)
    output["links"] = links


def migrate(path):
    path = Path(path).resolve()
    text = path.read_text(encoding="utf-8-sig")
    data = json.loads(text)
    before = copy.deepcopy(data)
    turbo = _one(data, "H3TurboSchedule")

    model_input = next((slot for slot in turbo.get("inputs", [])
                        if slot.get("name") == "model"), None)
    if model_input is None or model_input.get("link") is None:
        raise RuntimeError("Turbo 节点的 model 没有连线")
    turbo_link = _link(data, model_input["link"])
    nodes_by_id = {node.get("id"): node for node in data.get("nodes", [])}
    old_loader = nodes_by_id.get(turbo_link[1])
    if old_loader is None or old_loader.get("type") != "LoraLoaderModelOnly":
        # Already migrated: only heal the visible schema below.
        old_loader = None

    props = turbo.setdefault("properties", {})
    old_names = list(props.get("myang_widget_names") or [
        "profile", "speed_cache", "shift_video", "shift_audio",
        "recommended_steps",
    ])
    old_values = list(turbo.get("widgets_values") or [])
    values = dict(zip(old_names, old_values))

    if old_loader is not None:
        loader_values = old_loader.get("widgets_values") or []
        if isinstance(loader_values, dict):
            lora_name = loader_values.get("lora_name")
            strength = loader_values.get("strength_model", 1.0)
        else:
            lora_name = loader_values[0] if loader_values else None
            strength = loader_values[1] if len(loader_values) > 1 else 1.0
        if not lora_name:
            raise RuntimeError("旧 LoRA 加载器没有保存 lora_name")
        values.update({
            "profile": PROFILE_AUTO,
            "LoRA文件": str(lora_name),
            "LoRA强度": float(strength),
        })

        loader_model = next((slot for slot in old_loader.get("inputs", [])
                             if slot.get("name") == "model"), None)
        if loader_model is None or loader_model.get("link") is None:
            raise RuntimeError("旧 LoRA 加载器的基础 model 没有连线")
        base_link = _link(data, loader_model["link"])
        old_source = nodes_by_id.get(turbo_link[1])
        base_source = nodes_by_id.get(base_link[1])
        if old_source is None or base_source is None:
            raise RuntimeError("LoRA/Turbo 模型源节点不存在")
        _remove_link_cache(_output(old_source, turbo_link[2]), turbo_link[0])
        turbo_link[1] = base_link[1]
        turbo_link[2] = base_link[2]
        _add_link_cache(_output(base_source, base_link[2]), turbo_link[0])

    values.setdefault("profile", PROFILE_AUTO)
    values.setdefault("speed_cache", "关闭")
    values.setdefault("shift_video", 12.0)
    values.setdefault("shift_audio", 3.0)
    values.setdefault("recommended_steps", 8)
    values.setdefault("LoRA文件", "不在本节点加载（兼容旧工作流）")
    values.setdefault("LoRA强度", 1.0)
    turbo["widgets_values"] = [values[name] for name in WIDGET_NAMES]
    named = turbo.setdefault("widgets_values_named", {})
    named.update({name: values[name] for name in WIDGET_NAMES})
    props["myang_widget_names"] = list(WIDGET_NAMES)
    connectable = props.setdefault("ue_properties", {}).setdefault(
        "widget_ue_connectable", {})
    connectable.update({"LoRA文件": True, "LoRA强度": True})

    inputs = turbo.setdefault("inputs", [])
    if not any(slot.get("name") == "LoRA文件" for slot in inputs):
        inputs.append({
            "localized_name": "Turbo LoRA 文件", "name": "LoRA文件",
            "type": "COMBO", "widget": {"name": "LoRA文件"}, "link": None,
        })
    if not any(slot.get("name") == "LoRA强度" for slot in inputs):
        inputs.append({
            "localized_name": "LoRA 模型强度", "name": "LoRA强度",
            "type": "FLOAT", "widget": {"name": "LoRA强度"}, "link": None,
        })
    turbo["title"] = "Turbo LoRA 联合音画加载调度（LightX2V 官方档位）"
    size = list(turbo.get("size") or [444, 214])
    if len(size) == 2:
        size[1] = max(float(size[1]), 286.0)
        turbo["size"] = size

    if len(data.get("nodes", [])) != len(before.get("nodes", [])):
        raise RuntimeError("迁移意外改变了节点数量")
    if len(data.get("links", [])) != len(before.get("links", [])):
        raise RuntimeError("迁移意外改变了连线数量")
    if {node.get("id") for node in data["nodes"]} != {
            node.get("id") for node in before["nodes"]}:
        raise RuntimeError("迁移意外改变了节点 ID")

    backup = path.with_suffix(path.suffix + ".before-turbo-loader.bak")
    if not backup.exists():
        shutil.copy2(path, backup)
    path.write_text(json.dumps(
        data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return {
        "nodes": len(data["nodes"]),
        "links": len(data.get("links", [])),
        "lora": values["LoRA文件"],
        "strength": values["LoRA强度"],
        "backup": str(backup),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("workflow", nargs="?", default=str(DEFAULT_WORKFLOW))
    args = parser.parse_args()
    result = migrate(args.workflow)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
