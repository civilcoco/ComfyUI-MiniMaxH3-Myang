
"""Generate the loop-based MiniMax H3 long-video workflow.

The visible graph holds one H3LongVideo node. At run time it expands into as
many segment chains as the plan calls for, so changing the total duration
changes the render without touching the graph.

    python tools/build_loop_workflow.py

Run it with ComfyUI's bundled python from the pack directory.

The source graph is a 150-node single-shot workflow. Carrying nodes over from
it means carrying their positions too, which left the survivors scattered over
7000px of empty canvas next to two dozen groups whose members had been stripped.
Everything kept is therefore re-laid-out from scratch and the groups are rebuilt
around the result.
"""

import argparse
import copy
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
COMFY = HERE.parent.parent.parent
WORKFLOWS = COMFY / "user" / "default" / "workflows" / "minimax"

# Carried over from the source graph.
KEEP = {
    "loader": 757,          # legacy source-workflow loader
    "model_tail": 667,      # LoraLoaderModelOnly, end of the patch chain
    "patch_head": 708,      # ReservedVRAMSetter, head of the patch chain
    "sampler_select": 123,  # KSamplerSelect
    "ref_video": 638,       # VHS_LoadVideo
    "fps": 769,             # PrimitiveFloat
    "createvideo": 753,
    "savevideo": 754,
    "model_note": 117,      # MarkdownNote with the model download links
    "agent": 683,           # MiniMaxH3MediaAgent: owns the media and the prompt
}
IMAGES = [137, 139, 151]

# Driven the reference loader's frame_load_cap off a hand-set "one segment"
# length. In a chained render that truncates the clip to the first segment, so
# every later slice runs off the end. Replaced by the splitter's own count.
DROP = [
    757,   # legacy loader: replaced by our own H3Loader
    131,   # ComfyMathExpression: single-segment frame_load_cap
    132,   # PrimitiveFloat "Video Length (seconds)": its only consumer
    150, 159, 711, 765,   # rgthree bypassers keyed to groups that no longer exist
    637, 642,             # notes about the single-shot graph
]

USAGE_NOTE = """## 长视频 · 循环版

**先跑通再跑长**：把 `①` 的 `total_seconds` 改成 **16**，只展开 2 段，几分钟就能看到接缝效果。确认没问题再改回 60。

### 提示词与素材流水线
- **提示词 Agent**：负责剧本/提示词创作与素材理解（图片、视频、音频挂在它身上，能预览、能校验 `@图片1` / `@视频1`）。
- **① 剧本/分段切片**：接收 Agent 节点的 `myang_prompt`（或手动粘贴剧本），由 LLM 按时间轴切分成各段提示词，打包进 `plan_json`。
- **② 长视频生成**：只需接入 `plan_json` 与 `media`（无需再连 `prompt` 线），运行时直接消费 `plan_json` 内嵌的各段提示词。

### 三种任务模式（`②` 的 `task_mode`）
- **动作迁移**：每段跟随参考视频对应的那一片。要求素材够长——第 i 段从第 `(i-1)×(帧数-重叠)` 帧取，
  60s / 6 段需要约 1150 帧。`frame_load_cap` 已接 `ref_frames_needed` 自动算，素材本身不够长会警告。
- **视频续写**：只取参考视频的**结尾** `context_length` 帧作为起点，接着往下演。素材多短都行。
  想让声音也接上就把音频接进 `ref_audio`。
- **纯生成**：不接参考视频，全靠提示词和 Agent 上的图片。

`①` 的 `length_source` 选「匹配参考视频时长」就不用自己填总时长，跟着素材走。

### 两个必须一致的数
`①` 的 `overlap_frames` 和 `②` 的 `context_length` 必须相同（默认都是 22）。\n`5` 是两个 temporal latent block 的实验速度锚点；要试时两处一起改成 5。

### 漂移校正（默认关闭）
链式续写每段都以上一段的结果为条件，模型偏色会累积（实测 4 段饱和度 25%→36%，亮度 108→92）。
校正只比对**切口两侧各 8 帧**——那里内容几乎相同，差异就是漂移本身，
所以镜头和光线的真实变化不会被压掉。段数少（3 段以内）不用开；
要开就从 `mean_std` + 强度 `0.6` 起步，`mkl` 更彻底但也更容易改变观感。

### 二采放大（默认关闭）
二采按段执行，不会把整部成片一次送进二采显存（最终合并仍占 CPU 内存）。当前工作流已把 LoRA 前、显存补丁后的
Ref2VA 基模接到独立节点的『二采模型』；开启时先试 `832P / 4步 / denoise 0.2 / beta /
res_multistep / bicubic / chunk 4`。最终声音仍用一采原音频，不会让二采重唱音乐尾巴。

### 段数在运行时展开
图上永远只有一个 `②`。7 段跑的时候展开成 100 多个内部节点，跑完就消失。
各段单独的文件在 `output/video/H3_长视频_第NN段`，成片在 `H3_长视频_成片`。

### 示例剧本（想试逐段写戏时粘到 `①` 里）
```
主角：白发少女，深蓝短外套配浅色短裙。场景：黄昏的天台，远处是城市天际线。
开场她背对镜头站在天台边缘，肩膀随鼓点轻晃，慢慢转身露出笑意。
鼓点变密，她原地起跳双手打开，正式进入舞蹈，镜头正面平推靠近。
副歌前她突然定住，手指点在唇边直视镜头；下一拍鼓声炸开，转身连做两个大幅摆臂。
镜头绕到侧后方跟着旋转走，天色从橙红转向紫蓝，楼宇的灯一盏盏亮起。
她跳到天台中央，动作收小成碎步和手部小动作，表情转成放松的笑。
间奏走向护栏单手撑上去望向车流，镜头低角度仰拍，把她和晚霞框在一起。
最后一段副歌回到画面中央，把所有动作串一遍，幅度最大，长发甩出完整弧线。
收尾鼓点停在最后一下，她单手指向镜头定格，城市灯光在身后连成一片。
```"""

DETAIL_USAGE_MARKER = "### 二采放大（默认关闭）"
DETAIL_USAGE_APPEND = """### 二采放大（默认关闭）
二采按段执行，不会把整部成片一次送进二采显存（最终合并仍占 CPU 内存）。当前工作流已把 LoRA 前、显存补丁后的
Ref2VA 基模接到独立节点的『二采模型』；开启时先试 `832P / 4步 / denoise 0.2 / beta /
res_multistep / bicubic / chunk 4`。最终声音仍用一采原音频，不会让二采重唱音乐尾巴。"""


WIDGET_NAMES = {
    "H3ScriptSplitter": [
        "script", "total_seconds", "length_source", "segment_seconds", "overlap_frames",
        "fps", "llm_service", "max_segments", "ollama_auto_unload", "use_cache",
        "seed", "llm_enabled", "detail_boost", "__control_after_generate",
    ],
    "H3LongVideo": [
        "task_mode", "resolution", "aspect_ratio", "width",
        "height", "steps", "denoise", "scheduler", "noise_seed", "__control_after_generate",
        "context_length", "prompt_mode", "media_prefix", "llm_service", "ref_image_size",
        "save_segments", "segment_prefix", "save_raw_segments",
    ],
    "H3Loader": [
        "ref2va_model", "fl2va_model", "text_encoder", "video_vae", "audio_vae",
        "weight_dtype",
    ],
    "H3Model": ["kind"],
    "H3TurboSchedule": [
        "profile", "speed_cache", "shift_video", "shift_audio", "recommended_steps",
        "LoRA文件", "LoRA强度",
    ],
    "H3DetailSettings": [
        "enabled", "resolution", "width", "height", "steps", "denoise",
        "scheduler", "sampler_name", "upscale_method", "chunk_frames",
    ],
}
NAMES_PROP = "myang_widget_names"
WIDGET_DEFAULTS = {
    "H3DetailSettings": {
        "upscale_method": "nvidia_rtx_vsr",
        "resolution": "768P",
    },
    "H3ScriptSplitter": {
        "llm_enabled": True,
        "detail_boost": "远景小物体与五官强化（推荐·兼容加速）",
    },
    "H3TurboSchedule": {
        "speed_cache": "TE-Speed 时步缓存 (提速40%)",
        "LoRA文件": "不在本节点加载（兼容旧工作流）",
        "LoRA强度": 1.0,
    },
}

LONG_INPUT_SPECS = {
    "二采设置": ("MYANG_H3_DETAIL", 7),
}

RETIRED_LONG_DETAIL_INPUTS = {
    "second_pass", "second_width", "second_height", "second_steps",
    "second_denoise", "second_scheduler", "second_sampler",
    "second_upscale_method", "second_chunk_frames",
    "refine_model", "detail_settings",
}

# Declared the same way as the widget order and for the same reason: a node that
# loses an output invalidates every link aimed at a later slot, and ComfyUI
# reports that as "tuple index out of range" from inside an unrelated node's
# validation. Keeping the expected shape here lets `update` heal it without
# needing ComfyUI itself loaded.
# Widget layouts these nodes used to have, so a graph written against an older
# version can be migrated by name instead of by position.
LEGACY_WIDGETS = {
    "H3Loader": [
        # single transformer, before ref2va/fl2va were split apart
        ["ref2va_model", "text_encoder", "video_vae", "audio_vae", "weight_dtype"],
    ],
}

OUTPUT_NAMES = {
    "MiniMaxH3MediaAgent": [
        ("agent_prompt", "STRING"), ("summary_json", "STRING"),
        ("media_manifest", "STRING"), ("media", "MINIMAX_H3_MEDIA"),
        ("myang_prompt", "STRING"),
    ],
    "H3Loader": [("h3", "MYANG_H3")],
    "H3Model": [("model", "MODEL")],
    "H3TurboSchedule": [
        ("model", "MODEL"), ("recommended_steps", "INT"),
        ("shift_video", "FLOAT"), ("shift_audio", "FLOAT"),
    ],
    "H3DetailSettings": [("二采设置", "MYANG_H3_DETAIL")],
    "H3LongVideo": [("images", "IMAGE"), ("audio", "AUDIO")],
    "H3ScriptSplitter": [
        ("plan_json", "STRING"), ("segment_count", "INT"), ("segment_seconds", "FLOAT"),
        ("frames_per_segment", "INT"), ("plan_preview", "STRING"), ("ref_frames_needed", "INT"),
    ],
}


def io(name, typ, shape=None, widget=None):
    slot = {"name": name, "type": typ, "link": None}
    if shape is not None:
        slot["shape"] = shape
    if widget:
        slot["widget"] = {"name": widget}
    return slot


class Builder:
    def __init__(self, src):
        self.d = json.loads(Path(src).read_text(encoding="utf-8"))
        self.nodes = {n["id"]: n for n in self.d["nodes"]}
        self.next_id = max(self.nodes) + 1000
        self.next_link = max([l[0] for l in self.d["links"] if isinstance(l, list)] + [0]) + 1

    def add(self, node):
        node["id"] = self.next_id
        self.next_id += 1
        self.d["nodes"].append(node)
        self.nodes[node["id"]] = node
        return node["id"]

    def clone(self, tpl, widgets=None, title=None):
        n = copy.deepcopy(tpl if isinstance(tpl, dict) else self.nodes[tpl])
        for i in n.get("inputs", []) or []:
            i["link"] = None
        for o in n.get("outputs", []) or []:
            o["links"] = []
        if widgets is not None:
            n["widgets_values"] = widgets
        n["title"] = title or n.get("title")
        if not n["title"]:
            n.pop("title")
        return self.add(n)

    def make(self, typ, inputs, outputs, widgets, title=None, size=None):
        return self.add({
            "id": 0, "type": typ, "pos": [0, 0], "size": size or [400, 200], "flags": {},
            "order": 0, "mode": 0, "inputs": inputs,
            "outputs": [{"name": nm, "type": tp, "links": []} for nm, tp in outputs],
            "properties": {"Node name for S&R": typ,
                           **({NAMES_PROP: WIDGET_NAMES[typ]} if typ in WIDGET_NAMES else {})},
            "widgets_values": widgets,
            **({"title": title} if title else {})})

    def link(self, src, slot, dst, input_name, typ):
        inputs = self.nodes[dst].get("inputs", []) or []
        index = next((k for k, i in enumerate(inputs) if i.get("name") == input_name), None)
        if index is None:
            raise KeyError(f"节点 {dst} ({self.nodes[dst]['type']}) 没有输入 {input_name}")
        lid = self.next_link
        self.next_link += 1
        self.d["links"].append([lid, src, slot, dst, index, typ])
        outs = self.nodes[src].get("outputs") or []
        if slot < len(outs):
            outs[slot].setdefault("links", [])
            outs[slot]["links"].append(lid)
        inputs[index]["link"] = lid
        return lid

    def unlink(self, dst, input_name):
        for i in self.nodes[dst].get("inputs", []) or []:
            if i.get("name") == input_name and i.get("link") is not None:
                self._kill_link(i["link"])
                return True
        return False

    def _kill_link(self, lid):
        self.d["links"] = [l for l in self.d["links"]
                           if not (isinstance(l, list) and l[0] == lid)]
        for n in self.d["nodes"]:
            for i in n.get("inputs", []) or []:
                if i.get("link") == lid:
                    i["link"] = None
            for o in n.get("outputs", []) or []:
                if o.get("links"):
                    o["links"] = [x for x in o["links"] if x != lid]

    def remove(self, ids):
        drop = {i for i in ids if i in self.nodes}
        self.d["nodes"] = [n for n in self.d["nodes"] if n["id"] not in drop]
        self.d["links"] = [l for l in self.d["links"] if isinstance(l, list)
                           and l[1] not in drop and l[3] not in drop]
        self.nodes = {n["id"]: n for n in self.d["nodes"]}
        self._prune_dead_links()
        return drop

    def _prune_dead_links(self):
        alive = {l[0] for l in self.d["links"] if isinstance(l, list)}
        for n in self.d["nodes"]:
            for i in n.get("inputs", []) or []:
                if i.get("link") is not None and i["link"] not in alive:
                    i["link"] = None
            for o in n.get("outputs", []) or []:
                if o.get("links"):
                    o["links"] = [x for x in o["links"] if x in alive]

    def strip(self, roots):
        """Drop everything not feeding a root.

        An earlier version also kept anything whose *type name* contained
        "Patch"/"Bypasser". That spared the second-pass branch, which has no
        consumer in loop mode and sat 6000px below the rest of the graph.
        Every patch node that matters is reachable through the model chain, so
        reachability alone is the right rule; notes are passed in as roots.
        """
        L = {l[0]: l for l in self.d["links"] if isinstance(l, list)}
        keep, stack = set(), list(roots)
        while stack:
            x = stack.pop()
            if x in keep or x not in self.nodes:
                continue
            keep.add(x)
            for i in self.nodes[x].get("inputs", []) or []:
                if i.get("link") in L:
                    stack.append(L[i["link"]][1])
        return self.remove(set(self.nodes) - keep)

    # ---- layout -----------------------------------------------------------

    def layout(self, columns, top=80, gap=36, col_gap=70):
        """Stack each column vertically using real node sizes.

        Returns one bounding box per column so groups can be drawn around them.
        """
        boxes, x = [], 0
        for title, ids in columns:
            ids = [i for i in ids if i in self.nodes]
            width = max([self.nodes[i].get("size", [300, 100])[0] for i in ids] + [200])
            y = top
            for i in ids:
                n = self.nodes[i]
                n["pos"] = [x, y]
                y += n.get("size", [300, 100])[1] + gap
            boxes.append((title, [x - 20, top - 60, width + 40, y - top + 30]))
            x += width + col_gap
        return boxes


def build(src, dst, total_seconds, segment_seconds, overlap):
    b = Builder(src)
    service = next((n["widgets_values"][1] for n in b.d["nodes"]
                    if n["type"] == "MiniMaxH3MediaAgent" and n.get("widgets_values")), "")

    ref = b.nodes[KEEP["ref_video"]].get("widgets_values")
    if isinstance(ref, dict):
        # Reference video is ~53% of the attention sequence; this is the single
        # biggest VRAM lever measured on this graph.
        ref["custom_width"], ref["custom_height"] = 864, 480
        ref["frame_load_cap"] = 0

    splitter = b.make(
        "H3ScriptSplitter", [
            io("script", "STRING", shape=7),
            io("media", "MINIMAX_H3_MEDIA", shape=7),
        ],
        [("plan_json", "STRING"), ("segment_count", "INT"), ("segment_seconds", "FLOAT"),
         ("frames_per_segment", "INT"), ("plan_preview", "STRING"), ("ref_frames_needed", "INT")],
        ["", float(total_seconds), "用填写的总时长", float(segment_seconds), int(overlap),
         24.0, service, 12, True, True, 0, True, "fixed"],
        title="① 剧本/提示词分段切片（接收 Agent 输出或手填）", size=[420, 430])

    preview = b.make("PreviewAny", [io("source", "*")], [], [],
                     title="分段结果（段数 / 帧数 / 成片时长 / 参考视频需要多少帧）",
                     size=[560, 340])
    b.link(splitter, 4, preview, "source", "*")

    note = b.make("MarkdownNote", [], [], [USAGE_NOTE], title="使用说明", size=[560, 620])
    b.nodes[note]["color"] = "#432"
    b.nodes[note]["bgcolor"] = "#653"

    # Reference loader must not depend on downstream splitter (prevents cyclic dependency: Agent -> Splitter -> Video -> Agent).
    # Setting frame_load_cap=0 lets the loader read full video; H3LongVideo extracts exact segment slices in memory.
    b.unlink(KEEP["ref_video"], "frame_load_cap")

    # Our own loader, straight onto ComfyUI's official H3 support. It names both
    # transformers and loads whichever the task needs, so the patch chain hangs
    # off a separate take-model node rather than a second loader output.
    old = b.nodes[KEEP["loader"]].get("widgets_values") or []

    def pick(i):
        return old[i] if len(old) > i else ""

    # Older source workflows listed fl2va first, then ref2va, then the encoders.
    loader = b.make(
        "H3Loader", [], [("h3", "MYANG_H3")],
        [pick(1), pick(0), pick(2), pick(3), pick(4), "default"],
        title="沐阳 H3 加载器", size=[460, 170])
    getmodel = b.make(
        "H3Model", [io("h3", "MYANG_H3")], [("model", "MODEL")], ["ref2va"],
        title="取模型 → 补丁链", size=[300, 80])
    b.link(loader, 0, getmodel, "h3", "MYANG_H3")
    b.unlink(KEEP["patch_head"], "anything")
    b.link(getmodel, 0, KEEP["patch_head"], "anything", "*")

    long_inputs = [
        io("h3", "MYANG_H3"), io("model", "MODEL"), io("sampler", "SAMPLER"),
        io("plan_json", "STRING"),
        io("width", "INT", widget="width"), io("height", "INT", widget="height"),
        io("ref_video", "IMAGE", shape=7), io("ref_audio", "AUDIO", shape=7),
        io("media", "MINIMAX_H3_MEDIA", shape=7),
        io("二采设置", "MYANG_H3_DETAIL", shape=7),
    ]
    turbo = b.make(
        "H3TurboSchedule", [io("model", "MODEL")],
        [("model", "MODEL"), ("recommended_steps", "INT"),
         ("shift_video", "FLOAT"), ("shift_audio", "FLOAT")],
        ["LightX2V v1.0 · 8步（12/3·通用）",
         "TE-Speed 时步缓存 (提速40%)", 12.0, 3.0, 8,
         "不在本节点加载（兼容旧工作流）", 1.0],
        title="Turbo LoRA 联合音画调度", size=[390, 180])
    b.link(KEEP["model_tail"], 0, turbo, "model", "MODEL")

    longvid = b.make(
        "H3LongVideo", long_inputs,
        [("images", "IMAGE"), ("audio", "AUDIO")],
        ["动作迁移（跟随参考视频）", "480P", "16:9", 864, 480, 8, 1.0, "simple", 0, "fixed",
         str(overlap), "直接用分段稿", "参考@视频1中的人物动作表情、镜头调度、画面风格。",
         service, "off", 0.6, "最大1K面积", True, "video/H3_长视频",
         "实心（每步都钉·最稳）", 8, "关闭（H3原生轨迹）"],
        title="② 长视频（运行时自动展开 N 段 + 漂移校正）", size=[520, 890])
    detail_settings = b.make(
        "H3DetailSettings", [
            io("enabled", "BOOLEAN", widget="enabled"),
            io("resolution", "COMBO", widget="resolution"),
            io("width", "INT", widget="width"),
            io("height", "INT", widget="height"),
            io("steps", "INT", widget="steps"),
            io("denoise", "FLOAT", widget="denoise"),
            io("scheduler", "COMBO", widget="scheduler"),
            io("sampler_name", "COMBO", widget="sampler_name"),
            io("upscale_method", "COMBO", widget="upscale_method"),
            io("chunk_frames", "INT", widget="chunk_frames"),
            io("二采模型", "MODEL", shape=7),
        ], [("二采设置", "MYANG_H3_DETAIL")],
        [False, "832P", 1664, 928, 4, 0.2, "beta", "res_multistep",
         "latent_bicubic (Latent极速双采·推荐)", 4], title="二采放大（独立开关）", size=[390, 410])

    b.link(loader, 0, longvid, "h3", "MYANG_H3")
    b.link(turbo, 0, longvid, "model", "MODEL")
    lora_input = next(i for i in b.nodes[KEEP["model_tail"]]["inputs"]
                      if i.get("name") == "model")
    lora_source = next(link for link in b.d["links"]
                       if link[0] == lora_input["link"])
    b.link(lora_source[1], lora_source[2], detail_settings, "二采模型", "MODEL")
    b.link(detail_settings, 0, longvid, "二采设置", "MYANG_H3_DETAIL")
    b.link(KEEP["sampler_select"], 0, longvid, "sampler", "SAMPLER")
    b.link(splitter, 0, longvid, "plan_json", "STRING")
    b.link(KEEP["ref_video"], 0, longvid, "ref_video", "IMAGE")

    # The Agent stays in charge of media and prompt: it previews every clip,
    # numbers them, and checks that each @图片N / @视频N resolves. The loop only
    # swaps its reference clip for the current segment's slice.
    agent = KEEP["agent"]
    b.unlink(agent, "时长")          # was fed by the single-shot length node
    b.link(agent, 3, longvid, "media", "MINIMAX_H3_MEDIA")
    b.link(agent, 3, splitter, "media", "MINIMAX_H3_MEDIA")
    b.link(agent, 4, splitter, "script", "STRING")

    cv = b.clone(KEEP["createvideo"], [24, 8], "③ 成片")
    sv = b.clone(KEEP["savevideo"], ["video/H3_长视频_成片", "auto", "auto"], "③ 保存成片")
    b.link(longvid, 0, cv, "images", "IMAGE")
    b.link(longvid, 1, cv, "audio", "AUDIO")
    b.link(KEEP["fps"], 0, cv, "fps", "FLOAT")
    b.link(cv, 0, sv, "video", "VIDEO")

    b.strip({sv, preview, note, KEEP["model_note"], KEEP["agent"]}
            | set(IMAGES) | {KEEP["ref_video"]})
    dropped = b.remove(DROP)

    # Whatever survived feeding the Agent's media slots is the media column.
    links = {l[0]: l for l in b.d["links"] if isinstance(l, list)}
    sources = [links[i["link"]][1]
               for i in b.nodes[KEEP["agent"]].get("inputs", []) or []
               if str(i.get("name", "")).startswith("media_") and i.get("link") in links]
    seen, media_col = set(), []
    for nid in sources + [KEEP["fps"]]:
        if nid in b.nodes and nid not in seen:
            seen.add(nid)
            media_col.append(nid)

    boxes = b.layout([
        ("模型", [loader, getmodel, KEEP["patch_head"]]),
        ("补丁链 · 采样器", [733, 734, 663, 636, 655, 156,
                            KEEP["model_tail"], turbo, KEEP["sampler_select"]]),
        ("素材（由 Agent 编号）", media_col),
        ("提示词 Agent（素材预览 + 校验）", [KEEP["agent"]]),
        ("① 剧本 → 分段", [splitter, preview]),
        ("② 长视频", [detail_settings, longvid]),
        ("③ 成片", [cv, sv]),
        ("说明", [note, KEEP["model_note"]]),
    ])
    b.d["groups"] = [
        {"id": i + 1, "title": t, "bounding": bb, "color": c, "flags": {}}
        for i, ((t, bb), c) in enumerate(zip(boxes, ["#3f789e", "#3f789e", "#b58b2a",
                                                     "#285d29", "#285d29", "#a1309b",
                                                     "#8A8", "#444"]))
    ]

    # The saved viewport still pointed at the old node cluster, which is why the
    # workflow opened onto empty canvas.
    span = max(bb[0] + bb[2] for _, bb in boxes)
    b.d.setdefault("extra", {})["ds"] = {"scale": 0.45, "offset": [120, 60]}
    b.d["extra"].pop("qgn_navigation_groups", None)
    b.d["last_node_id"] = b.next_id
    b.d["last_link_id"] = b.next_link
    fix_ids(b.d)
    Path(dst).write_text(json.dumps(b.d, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(b.d["nodes"]), len(b.d["links"]), len(dropped), span, len(b.d["groups"])


def repair_outputs(data):
    """Make our nodes' output slots match the classes, and heal what that breaks.

    A node that loses an output silently invalidates every link that pointed at
    a later slot: ComfyUI validates by reading RETURN_TYPES[slot] and raises
    "tuple index out of range" from deep inside an unrelated node's validation,
    which is a miserable thing to debug from the error alone. So reconcile the
    slots here, then re-point anything left dangling.
    """
    nodes_by_id = {n["id"]: n for n in data.get("nodes", [])}
    fixed, dropped = [], []

    for node in data.get("nodes", []):
        spec = OUTPUT_NAMES.get(node.get("type"))
        if not spec:
            continue
        names = [nm for nm, _ in spec]
        types = [tp for _, tp in spec]
        current = node.get("outputs") or []
        if [o.get("name") for o in current] == names:
            continue
        keep = {o.get("name"): o for o in current}
        node["outputs"] = [
            {"name": nm, "type": types[i] if i < len(types) else "*",
             "links": list(keep.get(nm, {}).get("links") or [])}
            for i, nm in enumerate(names)
        ]
        fixed.append((node["type"], [o.get("name") for o in current], names))

    # Any link now aiming past the end of its source's outputs has to go.
    alive = []
    for link in data.get("links", []):
        if not isinstance(link, list):
            continue
        src = nodes_by_id.get(link[1])
        if src is not None and link[2] < len(src.get("outputs") or []):
            alive.append(link)
        else:
            dropped.append(link)
    if dropped:
        gone = {l[0] for l in dropped}
        data["links"] = alive
        for node in data.get("nodes", []):
            for slot in node.get("inputs") or []:
                if slot.get("link") in gone:
                    slot["link"] = None
            for slot in node.get("outputs") or []:
                if slot.get("links"):
                    slot["links"] = [x for x in slot["links"] if x not in gone]

    # The global link table is authoritative. Rebuild every source-slot cache
    # after a rename so changing `detail_settings` to `二采设置` cannot
    # leave a valid link invisible on the node output itself.
    for node in data.get("nodes", []):
        for output in node.get("outputs") or []:
            output["links"] = []
    for link in data.get("links", []):
        if not isinstance(link, list):
            continue
        source = nodes_by_id.get(link[1])
        outputs = source.get("outputs") if source is not None else None
        if outputs is not None and 0 <= int(link[2]) < len(outputs):
            outputs[int(link[2])]["links"].append(link[0])
    return fixed, dropped


def migrate_legacy_detail_inputs(data):
    """Collapse the old long-node detail controls into one Chinese settings link."""
    long_node = next((node for node in data.get("nodes", [])
                      if node.get("type") == "H3LongVideo"), None)
    if long_node is None:
        return []
    controller = next((node for node in data.get("nodes", [])
                       if node.get("type") == "H3DetailSettings"), None)
    inputs = long_node.setdefault("inputs", [])
    setting_slot = next((slot for slot in inputs
                         if slot.get("name") == "二采设置"), None)
    if setting_slot is None:
        setting_slot = io("二采设置", "MYANG_H3_DETAIL", shape=7)
        inputs.append(setting_slot)

    model_slot = None
    if controller is not None:
        controller_inputs = controller.setdefault("inputs", [])
        model_slot = next((slot for slot in controller_inputs
                           if slot.get("name") == "二采模型"), None)
        if model_slot is None:
            model_slot = io("二采模型", "MODEL", shape=7)
            controller_inputs.append(model_slot)

    links = {link[0]: link for link in data.get("links", [])
             if isinstance(link, list)}
    old_settings = next((slot for slot in inputs
                         if slot.get("name") == "detail_settings"), None)
    if old_settings and old_settings.get("link") is not None:
        lid = old_settings["link"]
        if setting_slot.get("link") is None and lid in links:
            setting_slot["link"] = lid
            links[lid][3] = long_node["id"]
            links[lid][4] = inputs.index(setting_slot)
        old_settings["link"] = None

    old_model = next((slot for slot in inputs
                      if slot.get("name") == "refine_model"), None)
    if (old_model and old_model.get("link") is not None and
            model_slot is not None and model_slot.get("link") is None):
        lid = old_model["link"]
        if lid in links:
            model_slot["link"] = lid
            links[lid][3] = controller["id"]
            links[lid][4] = controller["inputs"].index(model_slot)
        old_model["link"] = None

    gone = set()
    mapping, kept = {}, []
    for old_index, slot in enumerate(inputs):
        if slot.get("name") in RETIRED_LONG_DETAIL_INPUTS:
            if slot.get("link") is not None:
                gone.add(slot["link"])
            continue
        mapping[old_index] = len(kept)
        kept.append(slot)
    long_node["inputs"] = kept
    for link in data.get("links", []):
        if not isinstance(link, list) or link[3] != long_node["id"]:
            continue
        if link[0] in gone or link[4] not in mapping:
            gone.add(link[0])
        else:
            link[4] = mapping[link[4]]
    if gone:
        data["links"] = [link for link in data.get("links", [])
                         if not isinstance(link, list) or link[0] not in gone]
        for node in data.get("nodes", []):
            for output in node.get("outputs") or []:
                if output.get("links"):
                    output["links"] = [lid for lid in output["links"]
                                       if lid not in gone]
    size = long_node.get("size")
    if isinstance(size, list) and len(size) > 1 and float(size[1]) > 760:
        size[1] = max(620, float(size[1]) - 9 * 24)
    return sorted(gone)


def repair_long_inputs(data):
    """Append current H3LongVideo inputs without shifting existing links."""
    added = []
    for node in data.get("nodes", []):
        if node.get("type") != "H3LongVideo":
            continue
        inputs = node.setdefault("inputs", [])
        present = {slot.get("name") for slot in inputs}
        for name, (typ, shape) in LONG_INPUT_SPECS.items():
            if name in present:
                continue
            slot = io(name, typ, shape=shape)
            inputs.append(slot)
            added.append((node["id"], name))
    return added


def ensure_detail_settings_node(data):
    """Add one visible, default-off detail controller and connect it in place."""
    target = next((node for node in data.get("nodes", [])
                   if node.get("type") == "H3LongVideo"), None)
    if target is None:
        return None
    inputs = target.get("inputs") or []
    input_index = next((i for i, slot in enumerate(inputs)
                        if slot.get("name") == "二采设置"), None)
    if input_index is None or inputs[input_index].get("link") is not None:
        return None

    controller = next((node for node in data.get("nodes", [])
                       if node.get("type") == "H3DetailSettings"), None)
    if controller is None:
        x, y = target.get("pos") or [0, 0]
        height = (target.get("size") or [520, 860])[1]
        controller = {
            "id": max([node["id"] for node in data.get("nodes", [])] + [0]) + 1,
            "type": "H3DetailSettings", "pos": [x, y + height + 70],
            "size": [390, 410], "flags": {},
            "order": max([int(node.get("order", 0))
                          for node in data.get("nodes", [])] + [0]) + 1,
            "mode": 0,
            "inputs": [
                io("enabled", "BOOLEAN", widget="enabled"),
                io("resolution", "COMBO", widget="resolution"),
                io("width", "INT", widget="width"),
                io("height", "INT", widget="height"),
                io("steps", "INT", widget="steps"),
                io("denoise", "FLOAT", widget="denoise"),
                io("scheduler", "COMBO", widget="scheduler"),
                io("sampler_name", "COMBO", widget="sampler_name"),
                io("upscale_method", "COMBO", widget="upscale_method"),
                io("chunk_frames", "INT", widget="chunk_frames"),
                io("二采模型", "MODEL", shape=7),
            ],
            "outputs": [{"name": "二采设置",
                         "type": "MYANG_H3_DETAIL", "links": []}],
            "properties": {"Node name for S&R": "H3DetailSettings",
                           NAMES_PROP: WIDGET_NAMES["H3DetailSettings"]},
            "widgets_values": [False, "832P", 1664, 928, 4, 0.2,
                               "beta", "res_multistep", "bicubic", 4],
            "title": "二采放大（独立开关）",
        }
        data.setdefault("nodes", []).append(controller)

    lid = max([link[0] for link in data.get("links", [])
               if isinstance(link, list)] + [0]) + 1
    data.setdefault("links", []).append([
        lid, controller["id"], 0, target["id"], input_index,
        "MYANG_H3_DETAIL"])
    inputs[input_index]["link"] = lid
    controller["outputs"][0].setdefault("links", []).append(lid)
    return controller["id"]


def reconnect_model(data):
    """Re-feed the patch chain from H3Model, creating that node if it is missing.

    The loader used to hand out the MODEL itself. Splitting the two transformers
    apart moved that job to H3Model, so a graph written before the split has
    nothing to feed the patch chain with once the stale link is dropped.
    """
    by_type = {}
    for node in data.get("nodes", []):
        by_type.setdefault(node.get("type"), []).append(node)
    head = next((n for n in data.get("nodes", []) if n.get("id") == KEEP["patch_head"]), None)
    loader = (by_type.get("H3Loader") or [None])[0]
    model_node = (by_type.get("H3Model") or [None])[0]
    if head is None or loader is None:
        return None

    next_link = max([l[0] for l in data.get("links", []) if isinstance(l, list)] + [0]) + 1
    if model_node is None:
        pos = list(loader.get("pos") or [0, 0])
        model_node = {
            "id": max(n["id"] for n in data["nodes"]) + 1,
            "type": "H3Model", "pos": [pos[0], pos[1] + 200], "size": [300, 80],
            "flags": {}, "order": 0, "mode": 0,
            "inputs": [io("h3", "MYANG_H3")],
            "outputs": [{"name": "model", "type": "MODEL", "links": []}],
            "properties": {"Node name for S&R": "H3Model",
                           NAMES_PROP: WIDGET_NAMES["H3Model"]},
            "widgets_values": ["ref2va"], "title": "取模型 → 补丁链",
        }
        data["nodes"].append(model_node)
        data["links"].append([next_link, loader["id"], 0, model_node["id"], 0, "MYANG_H3"])
        model_node["inputs"][0]["link"] = next_link
        outs = loader.get("outputs") or []
        if outs:
            outs[0].setdefault("links", []).append(next_link)
        next_link += 1
    slot = next((i for i in head.get("inputs") or [] if i.get("name") == "anything"), None)
    if slot is None or slot.get("link") is not None:
        return model_node["id"]
    lid = next_link
    index = [i.get("name") for i in head.get("inputs") or []].index("anything")
    data.setdefault("links", []).append([lid, model_node["id"], 0, head["id"], index, "*"])
    slot["link"] = lid
    outs = model_node.get("outputs") or []
    if outs:
        outs[0].setdefault("links", []).append(lid)
    return model_node["id"]


def connect_refine_model(data, model_id):
    """Feed the pre-LoRA Ref2VA model into the combined detail controller."""
    if model_id is None:
        return None
    nodes = {node["id"]: node for node in data.get("nodes", [])}
    source = nodes.get(model_id)
    source_slot = 0
    links = {link[0]: link for link in data.get("links", [])
             if isinstance(link, list)}
    lora = next((node for node in data.get("nodes", [])
                 if node.get("type") == "LoraLoaderModelOnly"
                 and int(node.get("mode", 0)) != 4), None)
    if lora is not None:
        model_input = next((slot for slot in lora.get("inputs") or []
                            if slot.get("name") == "model"), None)
        link = links.get(model_input.get("link")) if model_input else None
        if link is not None:
            source = nodes.get(link[1])
            source_slot = int(link[2])
    target = next((node for node in data.get("nodes", [])
                   if node.get("type") == "H3DetailSettings"), None)
    if source is None or target is None:
        return None
    inputs = target.get("inputs") or []
    index = next((i for i, slot in enumerate(inputs)
                  if slot.get("name") == "二采模型"), None)
    if index is None or inputs[index].get("link") is not None:
        return None
    lid = max([link[0] for link in data.get("links", [])
               if isinstance(link, list)] + [0]) + 1
    data.setdefault("links", []).append(
        [lid, source["id"], source_slot, target["id"], index, "MODEL"])
    inputs[index]["link"] = lid
    outputs = source.get("outputs") or []
    if source_slot < len(outputs):
        outputs[source_slot].setdefault("links", []).append(lid)
    return lid


def fix_ids(data):
    """Keep last_node_id / last_link_id ahead of what the graph actually uses.

    The frontend hands out new ids by incrementing these. Adding a node or a
    link without bumping them makes it reissue an id that is already taken, and
    the older link silently loses its source -- which surfaces much later as a
    type mismatch on a slot nobody touched.
    """
    ids = [l[0] for l in data.get("links", []) if isinstance(l, list)]
    nodes = [n["id"] for n in data.get("nodes", [])]
    before = (data.get("last_node_id"), data.get("last_link_id"))
    data["last_node_id"] = max(nodes + [int(data.get("last_node_id") or 0)])
    data["last_link_id"] = max(ids + [int(data.get("last_link_id") or 0)])
    return before != (data["last_node_id"], data["last_link_id"])


def break_cycles(data):
    """Undo the one loop this graph invites: length matching vs frame budgeting.

    Wiring the reference loader into the splitter (to take the total duration
    from the footage) while the splitter drives that loader's frame_load_cap
    asks each node for an answer only the other one has. Matching the footage
    means wanting all of it, so the budget link is the one to give up, and the
    cap goes to 0.
    """
    nodes = {n["id"]: n for n in data.get("nodes", [])}
    links = [l for l in data.get("links", []) if isinstance(l, list)]
    feeds = {}
    for link in links:
        feeds.setdefault(link[1], set()).add(link[3])

    broken = []
    for link in list(links):
        src, dst = nodes.get(link[1]), nodes.get(link[3])
        if src is None or dst is None:
            continue
        if src.get("type") != "H3ScriptSplitter":
            continue
        if link[1] not in feeds.get(link[3], ()):   # dst does not feed back
            continue
        broken.append(link)
        target = dst.get("widgets_values")
        if isinstance(target, dict) and "frame_load_cap" in target:
            target["frame_load_cap"] = 0
            params = (target.get("videopreview") or {}).get("params")
            if isinstance(params, dict):
                params["frame_load_cap"] = 0

    if broken:
        gone = {l[0] for l in broken}
        data["links"] = [l for l in links if l[0] not in gone]
        for node in data.get("nodes", []):
            for slot in node.get("inputs") or []:
                if slot.get("link") in gone:
                    slot["link"] = None
            for slot in node.get("outputs") or []:
                if slot.get("links"):
                    slot["links"] = [x for x in slot["links"] if x not in gone]
    return broken


def update(dst):
    """Re-align our nodes' widget values in place, touching nothing else.

    Rebuilding from the source graph throws away whatever was added to the
    generated workflow by hand. This exists so that a schema change need not
    cost those edits: it reads the widget order each node was written with
    (`myang_widget_names`), maps the saved values onto the current order by
    name, and leaves every other node, link, group and position alone.
    """
    path = Path(dst)
    data = json.loads(path.read_text(encoding="utf-8"))
    fixed, skipped = [], []
    for node in data.get("nodes", []):
        names = WIDGET_NAMES.get(node.get("type"))
        if not names:
            continue
        old = node.get("properties", {}).get(NAMES_PROP)
        values = node.get("widgets_values") or []
        if not old:
            for legacy in LEGACY_WIDGETS.get(node.get("type"), []):
                if len(values) == len(legacy):
                    old = legacy
                    break
        if not old:
            if len(values) == len(names):
                # Same length means the order never changed, so position is
                # trustworthy here; record the names so it stays trustworthy.
                old = names
            else:
                # Guessing positionally is exactly how values land in the wrong
                # fields, so refuse instead.
                skipped.append(node.get("type"))
                continue
        by_name = dict(zip(old, values))
        defaults = WIDGET_DEFAULTS.get(node.get("type"), {})
        missing = [n for n in names if n not in by_name]
        unknown = [n for n in missing if n not in defaults]
        if unknown:
            # A missing value without an explicit schema default is not safe to
            # guess.  Leave the whole node byte-for-byte intact instead of
            # silently writing JSON null into a live workflow.
            skipped.append(node.get("type"))
            continue
        node["widgets_values"] = [
            by_name[n] if n in by_name else defaults[n] for n in names]
        node.setdefault("properties", {})[NAMES_PROP] = names
        fixed.append((node.get("type"), missing))

    for node in data.get("nodes", []):
        if node.get("type") == "MiniMaxH3MediaAgent":
            for index, slot in enumerate(
                    [slot for slot in node.get("inputs", [])
                     if str(slot.get("name") or "") == "media"
                     or str(slot.get("name") or "").startswith("media_")], 1):
                slot["name"] = f"asset_{index}"
                slot["localized_name"] = f"asset_{index}"
        if node.get("type") != "MarkdownNote" or node.get("title") != "使用说明":
            continue
        value = node.get("widgets_values")
        text = value[0] if isinstance(value, list) and value else value
        if isinstance(text, str):
            text = text.replace(
                "二采按段执行，不会把整部成片一次塞进显存。",
                "二采按段执行，不会把整部成片一次送进二采显存（最终合并仍占 CPU 内存）。")
        if isinstance(text, str) and DETAIL_USAGE_MARKER not in text:
            text = text.rstrip() + "\n\n" + DETAIL_USAGE_APPEND
        if isinstance(text, str):
            node["widgets_values"] = [text] if isinstance(value, list) else text
            if isinstance(node.get("widgets_values_named"), dict):
                node["widgets_values_named"]["text"] = text
            if isinstance(node.get("properties", {}).get("text"), str):
                node["properties"]["text"] = text

    migrate_legacy_detail_inputs(data)
    repair_long_inputs(data)
    ensure_detail_settings_node(data)
    slots, dropped = repair_outputs(data)
    rewired = reconnect_model(data)
    connect_refine_model(data, rewired)
    cycles = break_cycles(data)
    fix_ids(data)

    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return fixed, skipped, len(data.get("nodes", [])), slots, dropped, rewired, cycles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(WORKFLOWS / "minimax-沐阳Myang自用版.json"))
    ap.add_argument("--dst", default=str(WORKFLOWS / "minimax-沐阳Myang-长视频循环版.json"))
    ap.add_argument("--total-seconds", type=float, default=60.0)
    ap.add_argument("--segment-seconds", type=float, default=10.0)
    ap.add_argument("--overlap", type=int, default=22)
    ap.add_argument("--rebuild", action="store_true",
                    help="从源工作流重新生成。会丢掉你在生成结果上手动加的节点")
    a = ap.parse_args()

    # Default to updating in place. Overwriting someone's working graph should
    # take an explicit flag, not be the thing that happens when they re-run the
    # generator to pick up a fix.
    if Path(a.dst).exists() and not a.rebuild:
        fixed, skipped, total, slots, dropped, rewired, cycles = update(a.dst)
        print(f"已就地更新 {a.dst}")
        print(f"  {total} 个节点原样保留，只对齐了本包节点的参数")
        for typ, missing in fixed:
            note = f"，新增参数取默认值：{', '.join(missing)}" if missing else ""
            print(f"  {typ} 参数已对齐{note}")
        for typ in skipped:
            print(f"  ! {typ} 没有记录参数顺序，已跳过（请手动核对，或用 --rebuild 重建）")
        for typ, was, now in slots:
            print(f"  {typ} 输出槽已对齐：{was} -> {now}")
        for link in dropped:
            print(f"  ! 断开了指向已消失输出槽的连线：节点 {link[1]} 槽 {link[2]} -> 节点 {link[3]}")
        if rewired:
            print(f"  已把补丁链重新接到 H3Model（节点 {rewired}）")
        for link in cycles:
            print(f"  ! 解开了依赖环：节点 {link[1]} -> 节点 {link[3]}，"
                  f"该加载节点的 frame_load_cap 已设为 0（整条素材都读）")
        if not fixed and not skipped:
            print("  没找到本包的节点")
        print("  要从源工作流重建（会删掉你后加的节点）：加 --rebuild")
        return

    n, l, d, span, ng = build(a.src, a.dst, a.total_seconds, a.segment_seconds, a.overlap)
    print(f"已生成 {a.dst}")
    print(f"  节点 {n}，连线 {l}，清理 {d}（段数在运行时展开，不占图）")
    print(f"  画布宽 {span:.0f}px，{ng} 个分组，视口已重置")


if __name__ == "__main__":
    main()
