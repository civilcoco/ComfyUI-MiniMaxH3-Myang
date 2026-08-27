"""Append the Myang Turbo schedule node to an existing loop workflow.

This is intentionally surgical: no existing node is deleted or rebuilt.  The
current LoRA strength and every user-added node remain untouched.
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
PROFILE = "LightX2V v1.0 · 8步（12/3·通用）"


def _one(data, node_type):
    found = [node for node in data.get("nodes", [])
             if node.get("type") == node_type]
    if len(found) != 1:
        raise RuntimeError("需要且只能有一个 %s，实际 %d 个" %
                           (node_type, len(found)))
    return found[0]


def migrate(path):
    path = Path(path).resolve()
    original_text = path.read_text(encoding="utf-8-sig")
    data = json.loads(original_text)
    existing = [node for node in data.get("nodes", [])
                if node.get("type") == "H3TurboSchedule"]
    if existing:
        if len(existing) != 1:
            raise RuntimeError("H3TurboSchedule 节点重复")
        return False, len(data["nodes"]), len(data.get("links", []))

    before = copy.deepcopy(data)
    lora = _one(data, "LoraLoaderModelOnly")
    long_video = _one(data, "H3LongVideo")
    sampler = _one(data, "KSamplerSelect")
    lora_values = lora.get("widgets_values") or []
    lora_name = str(lora_values[0]) if lora_values else ""
    if "minimax_h3_fl2v_turbo_8step_v1.0" not in lora_name.lower():
        raise RuntimeError("拒绝给未知 LoRA 自动套用 8step v1.0：%s" % lora_name)

    model_input = next((item for item in long_video.get("inputs", [])
                        if item.get("name") == "model"), None)
    if model_input is None or model_input.get("link") is None:
        raise RuntimeError("H3LongVideo 的 model 没有连线")
    old_link_id = model_input["link"]
    old_link = next((link for link in data.get("links", [])
                     if isinstance(link, list) and link[0] == old_link_id), None)
    if old_link is None or old_link[1] != lora.get("id"):
        raise RuntimeError("当前 LoRA → H3LongVideo 模型链与预期不一致")

    node_id = max([int(node["id"]) for node in data["nodes"]] +
                  [int(data.get("last_node_id") or 0)]) + 1
    link_id = max([int(link[0]) for link in data.get("links", [])
                   if isinstance(link, list)] +
                  [int(data.get("last_link_id") or 0)]) + 1
    old_link[3] = node_id
    old_link[4] = 0
    model_input["link"] = link_id

    node = {
        "id": node_id,
        "type": "H3TurboSchedule",
        "pos": [510, 1595],
        "size": [444, 190],
        "flags": {},
        "order": max([int(n.get("order") or 0) for n in data["nodes"]] + [0]) + 1,
        "mode": 0,
        "inputs": [
            {"localized_name": "model", "name": "model", "type": "MODEL",
             "link": old_link_id},
            {"localized_name": "profile", "name": "profile", "type": "COMBO",
             "widget": {"name": "profile"}, "link": None},
            {"localized_name": "speed_cache", "name": "speed_cache", "type": "COMBO",
             "widget": {"name": "speed_cache"}, "link": None},
            {"localized_name": "shift_video", "name": "shift_video", "type": "FLOAT",
             "widget": {"name": "shift_video"}, "link": None},
            {"localized_name": "shift_audio", "name": "shift_audio", "type": "FLOAT",
             "widget": {"name": "shift_audio"}, "link": None},
            {"localized_name": "recommended_steps", "name": "recommended_steps",
             "type": "INT", "widget": {"name": "recommended_steps"}, "link": None},
            {"localized_name": "Turbo LoRA 文件", "name": "LoRA文件", "type": "COMBO",
             "widget": {"name": "LoRA文件"}, "link": None},
            {"localized_name": "LoRA 模型强度", "name": "LoRA强度", "type": "FLOAT",
             "widget": {"name": "LoRA强度"}, "link": None},
        ],
        "outputs": [
            {"localized_name": "model", "name": "model", "type": "MODEL",
             "links": [link_id]},
            {"localized_name": "recommended_steps", "name": "recommended_steps",
             "type": "INT", "links": None},
            {"localized_name": "shift_video", "name": "shift_video",
             "type": "FLOAT", "links": None},
            {"localized_name": "shift_audio", "name": "shift_audio",
             "type": "FLOAT", "links": None},
        ],
        "title": "Turbo LoRA 联合音画调度（LightX2V 官方档位）",
        "properties": {
            "Node name for S&R": "H3TurboSchedule",
            "myang_widget_names": [
                "profile", "speed_cache", "shift_video", "shift_audio",
                "recommended_steps", "LoRA文件", "LoRA强度"],
        },
        "widgets_values": [PROFILE, "关闭", 12.0, 3.0, 8,
                           "不在本节点加载（兼容旧工作流）", 1.0],
        "color": "#332922",
        "bgcolor": "#593930",
    }
    data["nodes"].append(node)
    data["links"].append(
        [link_id, node_id, 0, long_video["id"], 1, "MODEL"])
    sampler_values = sampler.get("widgets_values") or []
    if not sampler_values:
        raise RuntimeError("KSamplerSelect 没有 sampler_name")
    sampler_values[0] = "euler"
    for group in data.get("groups", []):
        if str(group.get("title", "")).startswith("补丁链"):
            box = group.get("bounding")
            if isinstance(box, list) and len(box) == 4:
                box[3] = max(float(box[3]), 1800.0)
            break
    data["last_node_id"] = node_id
    data["last_link_id"] = link_id

    if len(data["nodes"]) != len(before["nodes"]) + 1:
        raise RuntimeError("迁移意外改变了节点数量")
    if len(data.get("links", [])) != len(before.get("links", [])) + 1:
        raise RuntimeError("迁移意外改变了连线数量")
    old_ids = {node["id"] for node in before["nodes"]}
    if {node["id"] for node in data["nodes"] if node["id"] in old_ids} != old_ids:
        raise RuntimeError("迁移删除了已有节点")

    backup = path.with_suffix(path.suffix + ".before-turbo-schedule.bak")
    if not backup.exists():
        shutil.copy2(path, backup)
    temporary = path.with_suffix(path.suffix + ".turbo-schedule.tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8")
    written = json.loads(temporary.read_text(encoding="utf-8"))
    if written != data:
        temporary.unlink(missing_ok=True)
        raise RuntimeError("工作流写回校验失败")
    os.replace(temporary, path)
    return True, len(data["nodes"]), len(data.get("links", []))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("workflow", nargs="?", default=str(DEFAULT_WORKFLOW))
    args = parser.parse_args()
    changed, nodes, links = migrate(args.workflow)
    print("已追加 Turbo 调度节点" if changed else "Turbo 调度节点已存在")
    print("节点 %d、连线 %d；已有节点全部保留" % (nodes, links))


if __name__ == "__main__":
    main()
