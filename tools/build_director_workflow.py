"""Build the compact Myang Director example workflow.

The generated graph deliberately contains only the shared H3 loader/model,
sampler, Director and final video writer. Shot prompts and shot-local materials
are authored inside the Director card UI; the optional detail pass is also
configured there.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent
COMFY = PACKAGE.parent.parent
SOURCE = PACKAGE / "example_workflows" / "Minimax_H3_Myang_LongVideo_CN.json"
EXAMPLE = PACKAGE / "example_workflows" / "Minimax_H3_Myang_Director_CN.json"
USER_COPY = COMFY / "user" / "default" / "workflows" / "minimax" / EXAMPLE.name


def widget_input(name, kind="COMBO"):
    return {"name": name, "type": kind, "link": None, "widget": {"name": name}}


def socket_input(name, kind, shape=None):
    value = {"name": name, "type": kind, "link": None}
    if shape is not None:
        value["shape"] = shape
    return value


def clean_template(node, node_id, pos, title=None):
    value = copy.deepcopy(node)
    value["id"] = node_id
    value["pos"] = list(pos)
    value["order"] = node_id - 1
    for item in value.get("inputs", []) or []:
        item["link"] = None
    for item in value.get("outputs", []) or []:
        item["links"] = []
    if title:
        value["title"] = title
    return value


def director_node():
    timeline = {
        "version": 2,
        "shots": [{
            "id": "shot_1", "enabled": True, "duration_seconds": 5,
            "brief": "镜头 1",
            "prompt": "电影感镜头，主体自然运动，光影细腻，画面稳定，无字幕。",
            "asset_mode": "仅本镜头", "assets": [],
        }],
    }
    widgets = [
        "导演台分镜卡（手动逐镜头）",
        json.dumps(timeline, ensure_ascii=False),
        "", 60.0, 10.0, False, "未配置 LLM 服务",
        "纯生成（不用参考视频）", "480P", "16:9", 864, 480,
        25, 1.0, "simple", 0, "fixed", "22", "匹配生成分辨率",
        False, "放大 + 二采（推荐）", "832P", 1664, 928,
        4, 0.2, "beta", "res_multistep",
        "neural_3d (神经3D Latent放大·推荐)", 4,
        "minimax_h3_latent_upscaler_3d_fp16.safetensors",
        "fp16（推荐·省显存）", 16, 1, "每轮沿用同一种子",
        True, "video/H3_导演台", False,
    ]
    widget_names = [
        "source_mode", "timeline_json", "script_fallback", "total_seconds",
        "segment_seconds", "llm_enabled", "llm_service", "task_mode",
        "resolution", "aspect_ratio", "width", "height", "steps", "denoise",
        "scheduler", "noise_seed", "__control_after_generate", "context_length",
        "ref_image_size", "二采开启", "二采模式", "二采分辨率", "二采自定义宽",
        "二采自定义高", "二采步数", "二采重绘幅度", "二采调度器", "二采采样器",
        "二采放大方式", "二采分块帧数", "二采Latent模型", "二采精度",
        "二采时间分块", "二采轮数", "二采种子策略", "save_segments",
        "segment_prefix", "save_raw_segments",
    ]
    widget_types = {
        "timeline_json": "STRING", "script_fallback": "STRING",
        "total_seconds": "FLOAT", "segment_seconds": "FLOAT",
        "llm_enabled": "BOOLEAN", "width": "INT", "height": "INT",
        "steps": "INT", "denoise": "FLOAT", "noise_seed": "INT",
        "二采开启": "BOOLEAN", "二采自定义宽": "INT", "二采自定义高": "INT",
        "二采步数": "INT", "二采重绘幅度": "FLOAT", "二采分块帧数": "INT",
        "二采时间分块": "INT", "二采轮数": "INT", "save_segments": "BOOLEAN",
        "segment_prefix": "STRING", "save_raw_segments": "BOOLEAN",
    }
    inputs = [
        socket_input("h3", "MYANG_H3"), socket_input("model", "MODEL"),
        socket_input("sampler", "SAMPLER"),
    ]
    for name in widget_names:
        if name == "__control_after_generate":
            continue
        inputs.append(widget_input(name, widget_types.get(name, "COMBO")))
    inputs.extend([
        socket_input("script", "STRING", 7),
        socket_input("media", "MINIMAX_H3_MEDIA", 7),
        socket_input("ref_video", "IMAGE", 7),
        socket_input("ref_audio", "AUDIO", 7),
        socket_input("二采模型", "MODEL", 7),
        socket_input("二采设置", "MYANG_H3_DETAIL", 7),
        socket_input("Turbo联合模型", "MODEL", 7),
    ])
    return {
        "id": 4, "type": "H3Director", "pos": [970, 30], "size": [760, 980],
        "flags": {}, "order": 3, "mode": 0, "inputs": inputs,
        "outputs": [
            {"name": "images", "type": "IMAGE", "links": []},
            {"name": "audio", "type": "AUDIO", "links": []},
            {"name": "plan_json", "type": "STRING", "links": []},
            {"name": "fps", "type": "FLOAT", "links": []},
        ],
        "title": "② Myang H3 导演台 · 分镜 / 素材 / 二采",
        "properties": {"Node name for S&R": "H3Director", "myang_widget_names": widget_names},
        "widgets_values": widgets,
        "widgets_values_named": dict(zip(widget_names, widgets)),
        "color": "#17324d", "bgcolor": "#1d4263",
    }


def build():
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    by_type = {}
    for node in source.get("nodes", []):
        by_type.setdefault(node.get("type"), node)

    loader = clean_template(by_type["H3Loader"], 1, [30, 30], "① H3 双模型加载器")
    loader["widgets_values"] = [
        "minimax\\minimax_h3_ref2va_int8_convrot.safetensors",
        "minimax\\minimax_h3_ref2va_pruned_int8_convrot.safetensors",
        "MiniMax-H3\\qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
        "MiniMax-H3\\minimax_h3_video_vae_fp16.safetensors",
        "MiniMax-H3\\minimax_h3_audio_vae_fp32.safetensors", "default",
    ]
    model = clean_template(by_type["H3Model"], 2, [500, 30], "① Ref2VA 基模（一采 + 二采）")
    model["widgets_values"] = ["ref2va"]
    sampler = clean_template(by_type["KSamplerSelect"], 3, [500, 150], "① 一采采样器")
    sampler["widgets_values"] = ["euler"]
    director = director_node()
    create = clean_template(by_type["CreateVideo"], 5, [1790, 80], "③ 合成导演台成片")
    create["widgets_values"] = [24, 8]
    save = clean_template(by_type["SaveVideo"], 6, [1790, 240], "③ 保存导演台成片")
    save["widgets_values"] = ["video/H3_导演台_成片", "mp4", "h264", "auto"]

    note_template = by_type.get("MarkdownNote")
    note = clean_template(note_template, 7, [30, 330], "使用说明")
    note["size"] = [870, 590]
    note["widgets_values"] = [
        "## Myang H3 导演台工作流\n\n"
        "1. 在导演台每张镜头卡中直接添加图片、视频和音频。\n"
        "2. 二采默认关闭；展开导演台里的二采面板即可开启。\n"
        "3. 『二采模型』已接 Ref2VA 基模；如果接 Turbo，仍保留这里的基模线。\n"
        "4. 开启『同时保存二采前分段』后，会同时输出原始一采分段与二采分段。\n"
        "5. 示例安装没有独立 fl2va 时，加载器的 fl2va 槽暂用已安装模型占位；需要纯 FL2VA/T2VA 时请换成对应模型。"
    ]

    nodes = [loader, model, sampler, director, create, save, note]
    links = []
    next_link = 1

    def link(src, src_slot, dst, input_name, kind):
        nonlocal next_link
        target_index = next(i for i, item in enumerate(dst["inputs"])
                            if item["name"] == input_name)
        links.append([next_link, src["id"], src_slot, dst["id"], target_index, kind])
        src["outputs"][src_slot].setdefault("links", []).append(next_link)
        dst["inputs"][target_index]["link"] = next_link
        next_link += 1

    link(loader, 0, model, "h3", "MYANG_H3")
    link(loader, 0, director, "h3", "MYANG_H3")
    link(model, 0, director, "model", "MODEL")
    link(model, 0, director, "二采模型", "MODEL")
    link(sampler, 0, director, "sampler", "SAMPLER")
    link(director, 0, create, "images", "IMAGE")
    link(director, 1, create, "audio", "AUDIO")
    link(director, 3, create, "fps", "FLOAT")
    link(create, 0, save, "video", "VIDEO")

    workflow = {
        "id": "myang-h3-director-example", "revision": 0,
        "last_node_id": 7, "last_link_id": next_link - 1,
        "nodes": nodes, "links": links,
        "groups": [
            {"id": 1, "title": "模型与采样器", "bounding": [10, 10, 900, 930],
             "color": "#3f789e", "flags": {}},
            {"id": 2, "title": "导演台", "bounding": [940, 10, 820, 1030],
             "color": "#285d65", "flags": {}},
            {"id": 3, "title": "成片", "bounding": [1770, 10, 420, 440],
             "color": "#4f7d47", "flags": {}},
        ],
        "config": {},
        "extra": {"frontendVersion": "1.47.12", "ds": {"scale": 0.72, "offset": [20, 30]}},
        "version": 0.4,
    }
    text = json.dumps(workflow, ensure_ascii=False, indent=2) + "\n"
    EXAMPLE.parent.mkdir(parents=True, exist_ok=True)
    USER_COPY.parent.mkdir(parents=True, exist_ok=True)
    EXAMPLE.write_text(text, encoding="utf-8")
    USER_COPY.write_text(text, encoding="utf-8")
    return EXAMPLE, USER_COPY, len(nodes), len(links)


if __name__ == "__main__":
    example, user_copy, nodes, links = build()
    print("built", example)
    print("copied", user_copy)
    print("nodes", nodes, "links", links)
