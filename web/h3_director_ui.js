import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE = "H3Director";
const MANUAL = "导演台分镜卡（手动逐镜头）";
const TIMELINE_WIDGET = "timeline_json";
const TRANSFER = "动作迁移（跟随参考视频）";
const CONTINUE = "视频续写（接着往下演）";
const FRESH = "纯生成（不用参考视频）";
const AGENT_LINKS = "myang_h3_asset_sources_v2";
const KIND_OF_TYPE = {image: "图片", video: "视频", audio: "音频"};
const TYPE_OF_KIND = {图片: "image", 视频: "video", 音频: "audio"};
const GLYPH = {图片: "▣", 视频: "▶", 音频: "♪"};
const TAG_MAP = {picture: "图片", video: "视频", audio: "音频"};
const MENTION_RE = /(@(图片|视频|音频)[ \t_]*(\d+)|<(Picture|Video|Audio)[ \t_]*(\d+)>)/gi;
const DIALOGUE_RE = /<d>([\s\S]*?)<\/d>/g;
const TURBO_8 = "LightX2V v1.0 · 8步（12/3·通用）";
const TURBO_4_768 = "LightX2V v1.0 · 4步768P（6/3）";
const TURBO_4_REF = "LightX2V Ref2VA v0.1 · 4步（12/3）";
const TURBO_4 = "LightX2V v0.1 · 4步（12/3）";
const TURBO_AUTO = "自动匹配 LoRA 文件（推荐）";
const TURBO_MANUAL = "手动（高级）";
const NATIVE_VIDEO_WIDGET = "video-preview";
const OUTPUT_VIDEO_MIN_HEIGHT = 180;
const OUTPUT_VIDEO_MAX_HEIGHT = 360;
const SCRIPT_INPUT_MIN_HEIGHT = 56;
const SCRIPT_INPUT_MAX_HEIGHT = 220;
const REFERENCE_ORIGINAL = "匹配参考视频原分辨率";
const REFERENCE_RESOLUTIONS = [
    REFERENCE_ORIGINAL, "360P", "416P", "480P", "540P", "640P", "720P",
    "768P", "832P", "928P", "1024P", "1080P", "自定义",
];

const PROGRESS_PHASE = {
    sample1: {start: 0.00, span: 0.40, label: "一采采样", steps: true},
    drift: {start: 0.40, span: 0.10, label: "漂移校正", steps: false},
    refine_prep: {start: 0.50, span: 0.10, label: "二采准备", steps: false},
    sample2: {start: 0.60, span: 0.25, label: "二采采样", steps: true},
    finalizing: {start: 0.85, span: 0.15, label: "解码保存", steps: false},
};

let openDirectorMenu = null;
let directorWatcher = null;

function styleOnce() {
    if (document.getElementById("myh3-style")) return;
    const link = document.createElement("link");
    link.id = "myh3-style";
    link.rel = "stylesheet";
    link.href = new URL("./h3_prompt_editor.css", import.meta.url).href;
    document.head.appendChild(link);
}

const LABELS = {
    source_mode: "分镜来源",
    timeline_json: "分镜数据（内部）",
    script_fallback: "长剧本 / Agent 提示词",
    total_seconds: "目标总时长",
    segment_seconds: "智能切分单段时长",
    llm_enabled: "智能切片",
    llm_service: "LLM 服务",
    task_mode: "生成任务",
    resolution: "一采分辨率",
    aspect_ratio: "画面比例",
    width: "自定义宽",
    height: "自定义高",
    steps: "一采步数",
    denoise: "一采重绘幅度",
    scheduler: "一采调度器",
    noise_seed: "种子",
    context_length: "段间锚点帧",
    ref_image_size: "参考图尺寸",
    "二采开启": "导演台二采",
    "二采模式": "二采模式",
    "二采分辨率": "二采输出短边",
    "二采自定义宽": "二采自定义宽",
    "二采自定义高": "二采自定义高",
    "二采步数": "二采步数",
    "二采重绘幅度": "二采重绘幅度",
    "二采调度器": "二采调度器",
    "二采采样器": "二采采样器",
    "二采放大方式": "二采放大方式",
    "二采分块帧数": "像素 / VSR 分块帧数",
    "二采Latent模型": "神经 3D Latent 模型",
    "二采精度": "神经 3D 精度",
    "二采时间分块": "神经 3D 时间分块",
    "二采轮数": "二采轮数",
    "二采种子策略": "二采种子策略",
    save_segments: "保存每段",
    segment_prefix: "分段文件名前缀",
    save_raw_segments: "同时保存二采前分段",
    "参考视频分辨率": "参考视频分辨率",
    "参考视频自定义宽": "参考视频自定义宽",
    "参考视频自定义高": "参考视频自定义高",
    "起始段": "起始段（断点续跑）",
    skill_preset: "写作技能",
    skill_text: "自定义写作规则",
    vlm_service: "素材识图 VLM",
};

const INPUT_LABELS = {
    h3: "H3 模型包",
    model: "一采模型（基础 / Turbo）",
    sampler: "一采采样器",
    script: "旧 Agent 输入（请改连长剧本控件）",
    media: "Media Agent 素材包",
    ref_video: "动作迁移 / 续写参考视频",
    ref_audio: "续写参考音频",
    "前段视频": "前段成片（仅段间上下文）",
    "前段音频": "前段成片音轨（可选）",
    "二采模型": "二采 Ref2VA 基模（不开 Turbo LoRA）",
    "二采设置": "旧工作流二采设置（兼容入口）",
};

function widget(node, name) {
    return node.widgets?.find((item) => item.name === name);
}

function hideWidget(item) {
    if (!item || item.__myangDirectorHidden) return;
    item.__myangDirectorHidden = {
        type: item.type,
        computeSize: item.computeSize,
        ownCompute: Object.prototype.hasOwnProperty.call(item, "computeSize"),
        computedHeight: item.computedHeight,
        ownHeight: Object.prototype.hasOwnProperty.call(item, "computedHeight"),
    };
    item.hidden = true;
    item.type = "hidden";
    item.computeSize = () => [0, -4];
    item.computedHeight = 0;
    if (item.inputEl) item.inputEl.style.display = "none";
    if (item.element) item.element.style.display = "none";
}

function showWidget(item) {
    if (!item?.__myangDirectorHidden) return;
    const state = item.__myangDirectorHidden;
    delete item.__myangDirectorHidden;
    item.hidden = false;
    item.type = state.type;
    if (state.ownCompute) item.computeSize = state.computeSize;
    else delete item.computeSize;
    if (state.ownHeight) item.computedHeight = state.computedHeight;
    else delete item.computedHeight;
    if (item.inputEl) item.inputEl.style.display = "";
    if (item.element) item.element.style.display = "";
}

function setVisible(item, visible) {
    if (visible) showWidget(item);
    else hideWidget(item);
}

function scriptInputElement(item) {
    if (item?.inputEl) return item.inputEl;
    if (item?.element?.matches?.("textarea")) return item.element;
    return item?.element?.querySelector?.("textarea") || null;
}

function syncScriptInputHeight(node) {
    const item = widget(node, "script_fallback");
    const input = scriptInputElement(item);
    if (!item || !input || item.hidden || item.type === "hidden") return;
    // Converted / linked widgets are intentionally hidden by ComfyUI.  Do not
    // make them visible merely to measure them.
    if (input.isConnected && input.offsetParent === null) return;

    if (!item.__myangScriptSizerInstalled) {
        item.__myangScriptSizerInstalled = true;
        item.__myangScriptHeight = SCRIPT_INPUT_MIN_HEIGHT + 8;
        item.computeSize = (width) => [width, item.__myangScriptHeight];
        if (item.options) {
            item.options.getMinHeight = () => item.__myangScriptHeight;
            item.options.getMaxHeight = () => item.__myangScriptHeight;
            item.options.getHeight = () => item.__myangScriptHeight;
        }
        input.addEventListener("input", () => {
            requestAnimationFrame(() => syncScriptInputHeight(node));
        });
    }

    input.style.boxSizing = "border-box";
    input.style.minHeight = `${SCRIPT_INPUT_MIN_HEIGHT}px`;
    input.style.maxHeight = `${SCRIPT_INPUT_MAX_HEIGHT}px`;
    input.style.resize = "none";
    input.style.height = "auto";
    const contentHeight = Math.max(SCRIPT_INPUT_MIN_HEIGHT, Number(input.scrollHeight || 0));
    const inputHeight = Math.min(SCRIPT_INPUT_MAX_HEIGHT, contentHeight);
    input.style.height = `${inputHeight}px`;
    input.style.overflowY = contentHeight > SCRIPT_INPUT_MAX_HEIGHT ? "auto" : "hidden";
    item.__myangScriptHeight = inputHeight + 8;
    item.computedHeight = item.__myangScriptHeight;
    node.graph?.setDirtyCanvas?.(true, true);
}

function fitScriptTextArea(input) {
    if (!input) return;
    input.style.height = "auto";
    const contentHeight = Math.max(SCRIPT_INPUT_MIN_HEIGHT, Number(input.scrollHeight || 0));
    const height = Math.min(SCRIPT_INPUT_MAX_HEIGHT, contentHeight);
    input.style.height = `${height}px`;
    input.style.overflowY = contentHeight > SCRIPT_INPUT_MAX_HEIGHT ? "auto" : "hidden";
}

function freshShot(index = 1) {
    return {
        id: `shot_${Date.now().toString(36)}_${index}`,
        enabled: true,
        duration_seconds: 5,
        brief: `镜头 ${index}`,
        prompt: "",
        asset_mode: "仅本镜头",
        assets: [],
    };
}

function normalizeAsset(asset, index) {
    const file = asset?.file && typeof asset.file === "object" ? asset.file : asset || {};
    const kind = ["image", "video", "audio"].includes(asset?.kind) ? asset.kind : "image";
    return {
        id: String(asset?.id || `${kind}_${Date.now().toString(36)}_${index}`),
        kind,
        role: kind === "video" && asset?.role === "action" ? "action" : "reference",
        label: String(asset?.label || file?.name || `${kind} ${index + 1}`),
        file: {
            name: String(file?.name || ""),
            subfolder: String(file?.subfolder || ""),
            type: "input",
        },
    };
}

function parseTimeline(node) {
    const raw = String(widget(node, TIMELINE_WIDGET)?.value || "");
    try {
        const value = JSON.parse(raw);
        const shots = Array.isArray(value) ? value : value?.shots;
        if (Array.isArray(shots) && shots.length) {
            return shots.map((shot, index) => ({
                id: String(shot?.id || `shot_${index + 1}`),
                enabled: shot?.enabled !== false,
                duration_seconds: Number(shot?.duration_seconds ?? shot?.seconds ?? 5),
                brief: String(shot?.brief || `镜头 ${index + 1}`),
                prompt: String(shot?.prompt || ""),
                transition: String(shot?.transition || ""),
                fixed_from_plan: shot?.fixed_from_plan === true,
                asset_mode: shot?.asset_mode === "叠加全局素材" ? "叠加全局素材" : "仅本镜头",
                assets: Array.isArray(shot?.assets)
                    ? shot.assets.map((asset, assetIndex) => normalizeAsset(asset, assetIndex))
                        .filter((asset) => asset.file.name)
                    : [],
            }));
        }
    } catch (error) {
        console.warn("[Myang Director] 分镜 JSON 恢复失败", error);
    }
    return [freshShot(1)];
}

function parseGlobalAssets(node) {
    const raw = String(widget(node, TIMELINE_WIDGET)?.value || "");
    try {
        const value = JSON.parse(raw);
        const assets = Array.isArray(value) ? null : value?.global_assets;
        if (Array.isArray(assets)) {
            return assets.map((asset, index) => normalizeAsset(asset, index))
                .filter((asset) => asset.file.name)
                .map((asset) => ({...asset, role: "reference"}));
        }
    } catch (error) {
        console.warn("[Myang Director] 公共素材恢复失败", error);
    }
    return [];
}

function normalizePlanSnapshot(raw) {
    if (!raw || typeof raw !== "object" || !Array.isArray(raw.segments)) return null;
    const segments = raw.segments.map((segment, index) => ({
        index: Number(segment?.index || index + 1),
        brief: String(segment?.brief || `镜头 ${index + 1}`),
        prompt: String(segment?.prompt || ""),
        transition: String(segment?.transition || (index === 0 ? "开场" : "承接")),
        frames: Math.max(0, Number(segment?.frames || 0)),
        duration_seconds: Math.max(0, Number(segment?.duration_seconds || 0)),
    })).filter((segment) => segment.prompt.trim());
    if (!segments.length) return null;
    return {
        version: 1,
        saved_at: String(raw.saved_at || new Date().toISOString()),
        source: String(raw.source || "llm_split"),
        style_header: String(raw.style_header || ""),
        skill_source: String(raw.skill_source || ""),
        segment_count: segments.length,
        segments,
    };
}

function parsePlanSnapshot(node) {
    const raw = String(widget(node, TIMELINE_WIDGET)?.value || "");
    try {
        const value = JSON.parse(raw);
        if (value && !Array.isArray(value)) return normalizePlanSnapshot(value.plan_snapshot);
    } catch (error) {
        console.warn("[Myang Director] 分镜快照恢复失败", error);
    }
    return null;
}

function parseStoryboardMetadata(node) {
    const raw = String(widget(node, TIMELINE_WIDGET)?.value || "");
    try {
        const value = JSON.parse(raw);
        const metadata = !Array.isArray(value) ? value?.storyboard_metadata : null;
        if (!metadata || typeof metadata !== "object") return null;
        return {
            title: String(metadata.title || ""),
            source: String(metadata.source || ""),
            style_header: String(metadata.style_header || ""),
            skill_source: String(metadata.skill_source || ""),
        };
    } catch (error) {
        console.warn("[Myang Director] 分镜卡元数据恢复失败", error);
    }
    return null;
}

function saveTimeline(node) {
    const target = widget(node, TIMELINE_WIDGET);
    if (!target) return;
    node.__myangDirectorSaving = true;
    try {
        target.value = JSON.stringify({
            version: 3,
            shots: node.__myangDirectorShots,
            global_assets: node.__myangDirectorGlobals || [],
            plan_snapshot: normalizePlanSnapshot(node.__myangDirectorPlan),
            storyboard_metadata: node.__myangStoryboardMetadata || null,
        });
    } finally {
        node.__myangDirectorSaving = false;
    }
    node.graph?.setDirtyCanvas?.(true, true);
}

function storyboardNotice(node, message, tone = "success") {
    node.__myangStoryboardNotice = {message: String(message || ""), tone};
}

function exportStoryboardCards(node) {
    try {
        const documentData = createStoryboardCardDocument({
            shots: node.__myangDirectorShots,
            globalAssets: node.__myangDirectorGlobals,
            title: node.__myangStoryboardMetadata?.title || node.title || "H3导演台分镜卡",
            plan: node.__myangDirectorPlan || node.__myangStoryboardMetadata,
        });
        const blob = new Blob([JSON.stringify(documentData, null, 2)], {
            type: "application/json;charset=utf-8",
        });
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = url;
        link.download = storyboardCardFileName(documentData.storyboard.title);
        link.style.display = "none";
        document.body.appendChild(link);
        link.click();
        link.remove();
        setTimeout(() => URL.revokeObjectURL(url), 0);
        storyboardNotice(node, `已导出 ${documentData.storyboard.cards.length} 张结构化分镜卡`);
    } catch (error) {
        storyboardNotice(node, `导出失败：${error.message}`, "error");
    }
    renderTimeline(node);
}

function chooseStoryboardFile(node) {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = ".json,application/json";
    input.setAttribute("aria-label", "选择沐阳 H3 导演台分镜卡文件");
    input.onchange = async () => {
        const file = input.files?.[0];
        input.remove();
        if (!file) return;
        try {
            if (file.size > STORYBOARD_FILE_MAX_BYTES) {
                throw new Error("文件超过 10 MB，请确认没有把图片或视频数据写进 JSON");
            }
            const imported = parseStoryboardCardDocument(await file.text());
            const existing = (node.__myangDirectorShots || []).some(
                (shot) => String(shot?.prompt || "").trim() || (shot?.assets || []).length);
            if (existing && !window.confirm?.(
                `导入会用文件中的 ${imported.shots.length} 张分镜卡替换当前卡片，`
                + "并恢复文件内的公共素材引用。是否继续？")) return;
            const source = widget(node, "source_mode");
            if (!source) throw new Error("找不到导演台的分镜来源控件");
            node.__myangDirectorShots = imported.shots;
            node.__myangDirectorGlobals = imported.globalAssets;
            node.__myangDirectorPlan = null;
            node.__myangStoryboardMetadata = imported.metadata;
            source.value = MANUAL;
            saveTimeline(node);
            storyboardNotice(
                node,
                `已导入 ${imported.shots.length} 张分镜卡；素材引用需对应当前 ComfyUI/input 文件`);
            refresh(node);
        } catch (error) {
            storyboardNotice(node, `导入失败：${error.message}`, "error");
            renderTimeline(node);
        }
    };
    input.click();
}

function currentTask(node) {
    return String(widget(node, "task_mode")?.value || FRESH);
}

function inputLinked(node, name) {
    return (node.inputs || []).some((input) => input.name === name && input.link != null);
}

function alignedFrames(seconds) {
    const target = Math.max(5, Math.round(Number(seconds || 0) * 24));
    return Math.max(5, Math.round((target - 5) / 17) * 17 + 5);
}

function timelineStats(node) {
    const active = (node.__myangDirectorShots || []).filter((shot) => shot.enabled !== false);
    const overlap = Number(widget(node, "context_length")?.value || 22);
    const frames = active.map((shot) => alignedFrames(shot.duration_seconds));
    const outputFrames = frames.reduce((sum, value) => sum + value, 0)
        - Math.max(0, frames.length - 1) * overlap;
    return {count: active.length, frames, seconds: Math.max(0, outputFrames) / 24};
}

function button(text, title = "") {
    const element = document.createElement("button");
    element.type = "button";
    element.textContent = text;
    element.title = title;
    element.style.cssText = "border:1px solid #3c4654;background:#252d38;color:#dce5ef;border-radius:5px;padding:4px 8px;cursor:pointer;font-size:11px;";
    return element;
}

const MEDIA_META = {
    image: {label: "图片", accept: "image/*", limit: 9, token: "图片"},
    video: {label: "视频", accept: "video/*", limit: 3, token: "视频"},
    audio: {label: "音频", accept: "audio/*", limit: 3, token: "音频"},
};

function assetViewUrl(asset) {
    const query = new URLSearchParams({
        filename: asset.file.name,
        subfolder: asset.file.subfolder || "",
        type: "input",
    });
    return `/view?${query.toString()}`;
}

function upstream(node, inputName) {
    const input = (node.inputs || []).find((item) => item.name === inputName && item.link != null);
    if (!input || !node.graph) return null;
    const link = node.graph.links?.[input.link];
    return link ? node.graph.getNodeById?.(link.origin_id) : null;
}

function globalMediaList(node) {
    const agent = upstream(node, "media");
    const raw = agent?.properties?.[AGENT_LINKS];
    const counts = {图片: 0, 视频: 0, 音频: 0};
    const entries = [];
    if (Array.isArray(raw)) {
        for (const link of raw) {
            const kind = KIND_OF_TYPE[String(link?.media_type || "image").toLowerCase()] || "图片";
            const file = String(link?.filename || "");
            entries.push({
                kind,
                name: String(link?.subject || "") || file || String(link?.label || ""),
                subject: String(link?.subject || "").trim(),
                file,
                subfolder: String(link?.subfolder || ""),
                source: "global",
            });
        }
    }
    // 导演台自己的公共素材接在 Media Agent 之后，和 H3ShotMedia 的追加顺序一致。
    for (const asset of node.__myangDirectorGlobals || []) {
        const kind = KIND_OF_TYPE[asset.kind] || "图片";
        entries.push({
            kind,
            name: asset.label || asset.file?.name || "",
            subject: asset.label && asset.label !== asset.file?.name
                ? String(asset.label).trim() : "",
            file: asset.file?.name || "",
            subfolder: asset.file?.subfolder || "",
            assetId: asset.id,
            source: "director_global",
        });
    }
    return entries.map((entry) => {
        counts[entry.kind] += 1;
        return {...entry, ordinal: counts[entry.kind],
            token: `@${entry.kind}${counts[entry.kind]}`};
    });
}

/** 素材编号严格复刻 H3Condition：动作切片永远先占 @视频1，
 *  叠加模式先排 Media Agent，再排当前镜头文件；仅本镜头则从 1 开始。 */
function directorMediaList(node, shot) {
    const task = currentTask(node);
    const transferring = task === TRANSFER;
    const localAssets = (shot?.assets || []).filter((asset) =>
        !(transferring && asset.kind === "video"));
    const includeGlobal = transferring || shot?.asset_mode === "叠加全局素材"
        || localAssets.length === 0;
    const counts = {图片: 0, 视频: 0, 音频: 0};
    const result = [];

    if (transferring && (inputLinked(node, "ref_video")
        || (shot?.assets || []).some((asset) => asset.kind === "video"))) {
        const uploaded = (shot?.assets || []).find((asset) => asset.kind === "video");
        counts.视频 = 1;
        result.push({
            kind: "视频", ordinal: 1, token: "@视频1",
            name: uploaded?.label || "动作参考视频",
            subject: uploaded?.label && uploaded.label !== uploaded?.file?.name
                ? String(uploaded.label).trim() : "",
            file: uploaded?.file?.name || "",
            subfolder: uploaded?.file?.subfolder || "",
            assetId: uploaded?.id || null,
            source: uploaded ? "shot" : "direct",
        });
    }

    if (includeGlobal) {
        for (const entry of globalMediaList(node)) {
            counts[entry.kind] += 1;
            result.push({...entry, ordinal: counts[entry.kind], token: `@${entry.kind}${counts[entry.kind]}`});
        }
    }

    for (const asset of localAssets) {
        const kind = KIND_OF_TYPE[asset.kind] || "图片";
        counts[kind] += 1;
        result.push({
            kind,
            ordinal: counts[kind],
            token: `@${kind}${counts[kind]}`,
            name: asset.label || asset.file?.name || `${kind}${counts[kind]}`,
            subject: asset.label && asset.label !== asset.file?.name
                ? String(asset.label).trim() : "",
            file: asset.file?.name || "",
            subfolder: asset.file?.subfolder || "",
            assetId: asset.id,
            source: "shot",
        });
    }
    return result;
}

function mediaSignature(node) {
    return globalMediaList(node).map((entry) => `${entry.token}|${entry.file}`).join(",");
}

function promptThumb(entry) {
    if (entry?.file) {
        const type = TYPE_OF_KIND[entry.kind];
        if (type === "image") {
            const img = document.createElement("img");
            img.className = "myh3-chip-thumb";
            img.src = `/api/view?filename=${encodeURIComponent(entry.file)}&type=input&subfolder=${encodeURIComponent(entry.subfolder || "")}`;
            img.onerror = () => {
                const fallback = document.createElement("span");
                fallback.className = "myh3-chip-thumb";
                fallback.textContent = GLYPH[entry.kind] || "▣";
                img.replaceWith(fallback);
            };
            return img;
        }
    }
    const glyph = document.createElement("span");
    glyph.className = "myh3-chip-thumb";
    glyph.textContent = GLYPH[entry?.kind] || "▣";
    return glyph;
}

function promptChip(kind, ordinal, entry) {
    const chip = document.createElement("span");
    chip.className = "myh3-chip";
    chip.dataset.kind = kind;
    chip.dataset.token = `@${kind}${ordinal}`;
    chip.contentEditable = "false";
    if (!entry) chip.classList.add("is-missing");
    chip.appendChild(promptThumb(entry || {kind}));
    const label = document.createElement("span");
    label.textContent = entry?.subject
        ? `${kind}${ordinal} · ${entry.subject}` : `${kind}${ordinal}`;
    chip.appendChild(label);
    chip.title = entry
        ? `${chip.dataset.token}\n${entry.file || entry.name}`
        : `${chip.dataset.token}\n没有匹配到对应素材`;
    return chip;
}

function promptDialogue(text) {
    const block = document.createElement("span");
    block.className = "myh3-line";
    block.dataset.dialogue = "1";
    block.contentEditable = "false";
    block.textContent = text;
    block.title = "台词块（运行时发送为 <d>…</d>）";
    return block;
}

function appendPromptText(container, text) {
    for (const [index, part] of String(text).split("\n").entries()) {
        if (index) container.appendChild(document.createElement("br"));
        if (part) container.appendChild(document.createTextNode(part));
    }
}

function renderPromptMentions(container, text, byToken) {
    MENTION_RE.lastIndex = 0;
    let cursor = 0;
    for (let match = MENTION_RE.exec(text); match; match = MENTION_RE.exec(text)) {
        if (match.index > cursor) appendPromptText(container, text.slice(cursor, match.index));
        const kind = match[2] || TAG_MAP[String(match[4] || "").toLowerCase()] || "图片";
        const ordinal = Number(match[3] || match[5] || 1);
        const token = `@${kind}${ordinal}`;
        container.appendChild(promptChip(kind, ordinal, byToken.get(token)));
        cursor = match.index + match[0].length;
    }
    if (cursor < text.length) appendPromptText(container, text.slice(cursor));
}

function renderPromptInto(container, text, list) {
    container.replaceChildren();
    const byToken = new Map(list.map((entry) => [entry.token, entry]));
    const source = String(text || "");
    let cursor = 0;
    DIALOGUE_RE.lastIndex = 0;
    for (let match = DIALOGUE_RE.exec(source); match; match = DIALOGUE_RE.exec(source)) {
        if (match.index > cursor) renderPromptMentions(container, source.slice(cursor, match.index), byToken);
        container.appendChild(promptDialogue(match[1]));
        cursor = match.index + match[0].length;
    }
    if (cursor < source.length) renderPromptMentions(container, source.slice(cursor), byToken);
}

function readPromptText(container) {
    let output = "";
    const walk = (node) => {
        for (const child of node.childNodes || []) {
            if (child.nodeType === Node.TEXT_NODE) { output += child.nodeValue; continue; }
            if (child.nodeName === "BR") { output += "\n"; continue; }
            if (child.dataset?.token) { output += child.dataset.token; continue; }
            if (child.dataset?.dialogue) { output += `<d>${child.textContent}</d>`; continue; }
            if (child.nodeName === "DIV" || child.nodeName === "P") output += "\n";
            walk(child);
        }
    };
    walk(container);
    return output.replace(/ /g, " ");
}

function promptNodeLength(node) {
    if (node.nodeType === Node.TEXT_NODE) return String(node.nodeValue || "").length;
    if (node.nodeName === "BR") return 1;
    if (node.dataset?.token) return node.dataset.token.length;
    if (node.dataset?.dialogue) return node.textContent.length + 7;
    return Array.from(node.childNodes || []).reduce((sum, child) => sum + promptNodeLength(child), 0);
}

function promptCaretOffset(editor) {
    const selection = window.getSelection();
    if (!selection?.rangeCount || !editor.contains(selection.anchorNode)) return readPromptText(editor).length;
    const range = selection.getRangeAt(0).cloneRange();
    range.setStart(editor, 0);
    return promptNodeLength(range.cloneContents());
}

function restorePromptCaret(editor, wanted) {
    const selection = window.getSelection();
    if (!selection) return;
    let remaining = Math.max(0, Number(wanted) || 0);
    const range = document.createRange();
    let placed = false;
    const walk = (node) => {
        for (const child of node.childNodes || []) {
            if (placed) return;
            const length = promptNodeLength(child);
            if (child.nodeType === Node.TEXT_NODE && remaining <= length) {
                range.setStart(child, Math.min(remaining, length));
                placed = true;
                return;
            }
            if ((child.dataset?.token || child.dataset?.dialogue || child.nodeName === "BR")
                && remaining <= length) {
                range.setStartAfter(child);
                placed = true;
                return;
            }
            if (remaining <= length && child.childNodes?.length) {
                walk(child);
                return;
            }
            remaining -= length;
        }
    };
    walk(editor);
    if (!placed) range.selectNodeContents(editor), range.collapse(false);
    else range.collapse(true);
    selection.removeAllRanges();
    selection.addRange(range);
}

function hasRawMention(editor) {
    const walker = document.createTreeWalker(editor, NodeFilter.SHOW_TEXT);
    for (let node = walker.nextNode(); node; node = walker.nextNode()) {
        if (node.parentElement?.closest?.(".myh3-chip, .myh3-line")) continue;
        MENTION_RE.lastIndex = 0;
        if (MENTION_RE.test(node.nodeValue || "")) return true;
    }
    return false;
}

function insertAtCaret(editor, content) {
    const selection = window.getSelection();
    if (!selection?.rangeCount || !editor.contains(selection.anchorNode)) {
        editor.appendChild(content);
        return;
    }
    const range = selection.getRangeAt(0);
    range.deleteContents();
    range.insertNode(content);
    range.setStartAfter(content);
    range.collapse(true);
    selection.removeAllRanges();
    selection.addRange(range);
}

function closeDirectorMenu() {
    openDirectorMenu?.remove();
    openDirectorMenu = null;
}

function showDirectorMenu(editor, list, onPick) {
    closeDirectorMenu();
    const menu = document.createElement("div");
    menu.className = "myh3-menu";
    if (!list.length) {
        const empty = document.createElement("div");
        empty.className = "myh3-menu-empty";
        empty.textContent = "没有可引用素材；请先在镜头素材或 Media Agent 中添加";
        menu.appendChild(empty);
    }
    for (const entry of list) {
        const item = document.createElement("div");
        item.className = "myh3-menu-item";
        item.appendChild(promptThumb(entry));
        const token = document.createElement("span");
        token.textContent = entry.token;
        item.appendChild(token);
        const name = document.createElement("span");
        name.className = "myh3-menu-file";
        name.textContent = entry.name || entry.file || "";
        item.appendChild(name);
        item.onmousedown = (event) => {
            event.preventDefault();
            event.stopPropagation();
            onPick(entry);
            closeDirectorMenu();
        };
        menu.appendChild(item);
    }
    const selection = window.getSelection();
    const range = selection?.rangeCount ? selection.getRangeAt(0).cloneRange() : null;
    range?.collapse(true);
    const caret = range?.getBoundingClientRect?.();
    const fallback = editor.getBoundingClientRect();
    menu.style.left = `${Math.round(caret?.left || fallback.left)}px`;
    menu.style.top = `${Math.round(caret?.bottom || fallback.bottom) + 4}px`;
    document.body.appendChild(menu);
    openDirectorMenu = menu;
}

function createPromptEditor(node, shot, options = {}) {
    const editor = document.createElement("div");
    editor.className = "myh3-editor myang-director-prompt";
    editor.contentEditable = "true";
    editor.setAttribute("role", "textbox");
    editor.setAttribute("aria-multiline", "true");
    editor.dataset.placeholder = options.placeholder || "输入提示词；键入 @ 可选择素材";
    editor.style.cssText = "height:auto;min-height:86px;max-height:none;overflow:visible;resize:none;font-size:11px;line-height:1.55;";

    const materialList = () => directorMediaList(node, shot);
    renderPromptInto(editor, shot.prompt, materialList());
    const sync = () => {
        shot.prompt = readPromptText(editor);
        saveTimeline(node);
        if (hasRawMention(editor)) {
            const caret = promptCaretOffset(editor);
            renderPromptInto(editor, shot.prompt, materialList());
            restorePromptCaret(editor, caret);
            closeDirectorMenu();
        }
    };
    editor.addEventListener("input", sync);
    editor.addEventListener("blur", () => { sync(); closeDirectorMenu(); });
    editor.addEventListener("keydown", (event) => {
        if (event.key === "Escape") closeDirectorMenu();
        if (event.key !== "Backspace" && event.key !== "Delete") return;
        const anchor = window.getSelection()?.anchorNode;
        const atomic = (anchor?.nodeType === Node.ELEMENT_NODE ? anchor : anchor?.parentElement)
            ?.closest?.(".myh3-chip, .myh3-line");
        if (atomic && editor.contains(atomic)) {
            event.preventDefault();
            atomic.remove();
            sync();
        }
    });
    editor.addEventListener("keyup", (event) => {
        if (event.key !== "@") return;
        showDirectorMenu(editor, materialList(), (entry) => {
            const selection = window.getSelection();
            const range = selection?.rangeCount ? selection.getRangeAt(0) : null;
            if (range?.startContainer?.nodeType === Node.TEXT_NODE && range.startOffset > 0
                && range.startContainer.nodeValue.slice(0, range.startOffset).endsWith("@")) {
                range.setStart(range.startContainer, range.startOffset - 1);
                range.deleteContents();
            }
            insertAtCaret(editor, promptChip(entry.kind, entry.ordinal, entry));
            sync();
        });
    });
    editor.addEventListener("paste", (event) => {
        event.preventDefault();
        insertAtCaret(editor, document.createTextNode(
            event.clipboardData?.getData("text/plain") || ""));
        sync();
    });
    return editor;
}

function directorByOwner(ownerId) {
    const wanted = String(ownerId || "");
    if (!wanted) return null;
    return (app.graph?._nodes || []).find((node) =>
        node.type === NODE && String(node.id) === wanted) || null;
}

function progressState(node) {
    node.__myangDirectorProgress ||= {
        status: "idle", runId: "", total: 1, seg: 1,
        phase: "sample1", step: 0, stepMax: 0,
        previewFile: "", previewTs: 0, prompt: "", brief: "",
        refining: false, correcting: false, error: "",
    };
    return node.__myangDirectorProgress;
}

function updateDirectorProgress(node) {
    const state = progressState(node);
    const elements = node.__myangDirectorProgressEls;
    if (!elements?.panel?.isConnected) return;
    const phase = PROGRESS_PHASE[state.phase] || PROGRESS_PHASE.sample1;
    let inner = phase.start;
    if (state.status === "done" || state.phase === "done") inner = 1;
    else if (phase.steps && state.stepMax > 0) inner += (state.step / state.stepMax) * phase.span;
    else inner += phase.span * 0.5;
    const percent = state.status === "idle" ? 0
        : ((Math.max(1, state.seg) - 1 + Math.min(1, inner)) / Math.max(1, state.total)) * 100;
    elements.fill.style.width = `${Math.max(0, Math.min(100, percent)).toFixed(1)}%`;
    elements.text.classList.remove("is-running", "is-done", "is-error");

    if (state.status === "idle") {
        elements.text.textContent = "等待执行 · 运行后这里会显示分段、采样步数和当前预览";
    } else if (state.status === "error") {
        elements.text.classList.add("is-error");
        elements.text.textContent = `执行中断 · ${state.error || "未知错误"}`;
    } else if (state.status === "done" || state.phase === "done") {
        elements.text.classList.add("is-done");
        elements.text.textContent = `全部 ${state.total} 段完成`;
    } else {
        elements.text.classList.add("is-running");
        let text = `第 ${state.seg}/${state.total} 段 · ${phase.label}`;
        if (phase.steps && state.stepMax > 0) text += ` · 第 ${state.step}/${state.stepMax} 步`;
        elements.text.textContent = text;
    }
    const prompt = String(state.prompt || "").trim();
    elements.prompt.hidden = !prompt;
    if (prompt) {
        elements.prompt.textContent = `${state.brief ? `${state.brief} · ` : ""}${prompt}`;
        elements.prompt.title = prompt;
    }
    if (state.previewFile) {
        elements.preview.hidden = false;
        elements.preview.src = `/api/view?filename=${encodeURIComponent(state.previewFile)}&type=temp&subfolder=&_t=${state.previewTs || Date.now()}`;
    } else {
        elements.preview.hidden = true;
    }
    // 分段列表只在"正在跑第几段"真的变了时重画：每个采样步都重建会打断选中和滚动。
    const active = state.status === "running" ? Number(state.seg || 0) : 0;
    if (node.__myangDirectorPlanActive !== active) {
        node.__myangDirectorPlanActive = active;
        updateSegmentPlan(node);
    }
}

function renderDirectorProgressPanel(node) {
    const panel = document.createElement("div");
    panel.className = "myh3-progress";
    panel.style.cssText += ";margin-bottom:8px;background:#101923;border-color:#33465d;";
    const heading = document.createElement("div");
    heading.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:8px;color:#bfdbfe;font-size:10px;font-weight:700;";
    const title = document.createElement("span");
    title.textContent = "生成进度与当前画面";
    const live = document.createElement("span");
    live.textContent = "实时";
    live.style.cssText = "color:#60a5fa;font-size:9px;font-weight:500;";
    heading.append(title, live);
    const bar = document.createElement("div");
    bar.className = "myh3-progress-bar";
    const fill = document.createElement("div");
    fill.className = "myh3-progress-fill";
    bar.appendChild(fill);
    const text = document.createElement("div");
    text.className = "myh3-progress-text";
    const prompt = document.createElement("div");
    prompt.style.cssText = "font-size:9px;line-height:1.45;color:#93a4b8;background:#0b1118;border-radius:5px;padding:5px 7px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;";
    const preview = document.createElement("img");
    preview.className = "myh3-progress-preview";
    preview.alt = "导演台当前分段预览帧";
    preview.hidden = true;
    panel.append(heading, bar, text, prompt, preview);
    node.__myangDirectorProgressEls = {panel, fill, text, prompt, preview};
    updateDirectorProgress(node);
    return panel;
}

function outputVideoHeight(node) {
    const nodeHeight = Number(node.size?.[1] || 920);
    return Math.round(Math.max(OUTPUT_VIDEO_MIN_HEIGHT,
        Math.min(OUTPUT_VIDEO_MAX_HEIGHT, nodeHeight * 0.32)));
}

function syncOutputVideoGeometry(node) {
    const host = node.__myangDirectorVideoHost;
    if (!host) return;
    host.style.height = `${outputVideoHeight(node)}px`;
}

function clearOutputVideo(node) {
    node.__myangDirectorVideoMountToken = Number(node.__myangDirectorVideoMountToken || 0) + 1;
    const video = node.__myangDirectorVideoElement;
    try { video?.pause?.(); } catch (_error) { /* detached media can already be closed */ }
    video?.remove?.();
    node.__myangDirectorVideoElement = null;
    node.videoContainer?.replaceChildren?.();
    node.__myangDirectorVideoHost?.replaceChildren?.();
    if (node.__myangDirectorVideoPanel) node.__myangDirectorVideoPanel.hidden = true;
}

function mountNativeVideoPreview(node) {
    const host = node.__myangDirectorVideoHost;
    const panel = node.__myangDirectorVideoPanel;
    if (!host || !panel) return false;

    const container = node.videoContainer;
    const incoming = container?.querySelector?.("video") || null;
    if (incoming && incoming !== node.__myangDirectorVideoElement) {
        try { node.__myangDirectorVideoElement?.pause?.(); } catch (_error) { /* no-op */ }
        node.__myangDirectorVideoElement?.remove?.();
        node.__myangDirectorVideoElement = incoming;
    }
    const video = node.__myangDirectorVideoElement;
    if (!video) {
        panel.hidden = true;
        return false;
    }

    // Keep ComfyUI's own HTMLVideoElement and controls, but remove its separate
    // layout row.  The same element now lives inside the Director card.
    const nativeWidget = node.widgets?.find((item) => item.name === NATIVE_VIDEO_WIDGET);
    if (nativeWidget && !nativeWidget.__myangDirectorHidden) hideWidget(nativeWidget);
    if (video.parentElement !== host) host.replaceChildren(video);
    video.controls = true;
    video.playsInline = true;
    video.loop = true;
    video.setAttribute("aria-label", "导演台成片预览");
    video.style.cssText = "display:block;width:100%;height:100%;max-width:100%;max-height:100%;object-fit:contain;background:#05070a;";
    panel.hidden = false;
    syncOutputVideoGeometry(node);

    // Once the playable result is present, the last sampling still-frame is a
    // duplicate and needlessly takes another row.
    if (progressState(node).status === "done" && node.__myangDirectorProgressEls?.preview) {
        node.__myangDirectorProgressEls.preview.hidden = true;
    }
    return true;
}

function scheduleNativeVideoMount(node) {
    const token = Number(node.__myangDirectorVideoMountToken || 0) + 1;
    node.__myangDirectorVideoMountToken = token;
    let attempts = 24;
    const check = () => {
        if (node.__myangDirectorVideoMountToken !== token) return;
        if (mountNativeVideoPreview(node) || --attempts <= 0) return;
        setTimeout(check, 125);
    };
    requestAnimationFrame(check);
}

function renderOutputVideoPanel(node) {
    const panel = document.createElement("section");
    panel.hidden = true;
    panel.style.cssText = "flex:0 0 auto;margin-top:auto;padding-top:8px;";
    const card = document.createElement("div");
    card.style.cssText = "border:1px solid #33465d;background:#0b1118;border-radius:7px;padding:7px;";
    const heading = document.createElement("div");
    heading.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:8px;margin-bottom:6px;";
    const title = document.createElement("span");
    title.textContent = "成片预览";
    title.style.cssText = "color:#bfdbfe;font-size:10px;font-weight:700;";
    const hint = document.createElement("span");
    hint.textContent = "自适应播放器 · 可全屏";
    hint.style.cssText = "color:#718096;font-size:9px;";
    heading.append(title, hint);
    const host = document.createElement("div");
    host.style.cssText = "display:flex;align-items:center;justify-content:center;width:100%;overflow:hidden;border-radius:5px;background:#05070a;";
    card.append(heading, host);
    panel.appendChild(card);
    node.__myangDirectorVideoPanel = panel;
    node.__myangDirectorVideoHost = host;
    if (node.__myangDirectorVideoElement) host.appendChild(node.__myangDirectorVideoElement);
    syncOutputVideoGeometry(node);
    mountNativeVideoPreview(node);
    return panel;
}

function applyDirectorProgress(node, detail) {
    const state = progressState(node);
    if (state.runId && detail?.run_id && String(detail.run_id) !== String(state.runId)) return;
    state.status = "running";
    state.seg = Number(detail?.segment_index || state.seg || 1);
    state.total = Number(detail?.total_segments || state.total || 1);
    const stage = String(detail?.stage || "");
    if (stage === "sampling") {
        state.phase = String(detail?.pass_label || "sample1").startsWith("sample2")
            ? "sample2" : "sample1";
        state.step = Number(detail?.step || 0);
        state.stepMax = Number(detail?.step_total || 0);
    } else {
        state.step = 0;
        state.stepMax = 0;
        if (stage === "sampled") state.phase = state.correcting
            ? "drift" : state.refining ? "refine_prep" : "finalizing";
        else if (stage === "drifted") state.phase = state.refining ? "refine_prep" : "finalizing";
        else if (stage === "refine_start") state.phase = "refine_prep";
        else if (stage === "refined") state.phase = "finalizing";
        else if (stage === "done") {
            if (state.seg < state.total) {
                state.seg += 1;
                state.phase = "sample1";
            } else {
                state.phase = "done";
                state.status = "done";
            }
        }
    }
    if (detail?.preview_file) {
        state.previewFile = detail.preview_file;
        state.previewTs = detail.preview_ts || Date.now();
    }
    if (String(detail?.prompt || "").trim()) state.prompt = String(detail.prompt);
    if (String(detail?.brief || "").trim()) state.brief = String(detail.brief);
    updateDirectorProgress(node);
}

function turboStepSpecForNode(turboNode) {
    if (!turboNode) return null;
    if (turboNode.type !== "H3TurboSchedule") return null;
    let profile = String(widget(turboNode, "profile")?.value || "");
    if (profile === TURBO_AUTO) {
        const name = String(widget(turboNode, "LoRA文件")?.value || "").toLowerCase().replaceAll("-", "_");
        if ((name.includes("ref2va") || name.includes("ref2v")) && name.includes("4step")) profile = TURBO_4_REF;
        else if (name.includes("8step")) profile = TURBO_8;
        else if (name.includes("4step") && name.includes("768p")) profile = TURBO_4_768;
        else if (name.includes("4step")) profile = TURBO_4;
        else return {allowed: null, recommended: null, label: "自动档 · 运行时识别"};
    }
    if (profile === TURBO_8) return {allowed: [8, 4], recommended: 8, label: "推荐 8 步·可手调"};
    if ([TURBO_4_768, TURBO_4_REF, TURBO_4].includes(profile)) {
        return {allowed: [4], recommended: 4, label: "推荐 4 步·可手调"};
    }
    if (profile === TURBO_MANUAL) {
        const steps = Math.max(1, Number(widget(turboNode, "recommended_steps")?.value || 8));
        return {allowed: [steps], recommended: steps, label: `推荐 ${steps} 步·可手调`};
    }
    return {allowed: null, recommended: null, label: "运行时校验"};
}

function turboStepSpec(node) {
    return turboStepSpecForNode(
        upstream(node, "Turbo联合模型") || upstream(node, "model"));
}

function turboSignature(node) {
    const turboNode = upstream(node, "Turbo联合模型") || upstream(node, "model");
    if (!turboNode) return "";
    return [turboNode.id, widget(turboNode, "profile")?.value,
        widget(turboNode, "LoRA文件")?.value,
        widget(turboNode, "recommended_steps")?.value].join("|");
}

function migrateLegacyInputs(node) {
    if (node.__myangMigratingLegacyInputs) return;
    node.__myangMigratingLegacyInputs = true;
    try {
        const oldTurboIndex = (node.inputs || []).findIndex(
            (input) => input.name === "Turbo联合模型");
        const oldTurboInput = node.inputs?.[oldTurboIndex];
        if (oldTurboInput?.link != null && node.graph) {
            const link = node.graph.links?.[oldTurboInput.link];
            const origin = link ? node.graph.getNodeById?.(link.origin_id) : null;
            const modelIndex = (node.inputs || []).findIndex(
                (input) => input.name === "model");
            if (origin && modelIndex >= 0) {
                if (node.inputs[modelIndex]?.link != null) node.disconnectInput(modelIndex);
                node.disconnectInput(oldTurboIndex);
                origin.connect(link.origin_slot, node, modelIndex);
            }
        }

        const recommendedIndex = (node.inputs || []).findIndex(
            (input) => input.name === "Turbo推荐一采步数");
        const recommendedInput = node.inputs?.[recommendedIndex];
        const recommended = turboStepSpec(node)?.recommended;
        if (recommendedInput?.link != null && recommended != null) {
            if (widget(node, "steps")) widget(node, "steps").value = recommended;
            node.disconnectInput(recommendedIndex);
        }

        for (const name of ["script", "二采设置", "Turbo联合模型", "Turbo推荐一采步数"]) {
            const index = (node.inputs || []).findIndex((input) => input.name === name);
            if (index >= 0 && node.inputs[index].link == null) node.removeInput(index);
        }
    } finally {
        node.__myangMigratingLegacyInputs = false;
    }
}

async function uploadShotFile(shot, file) {
    const body = new FormData();
    body.append("image", file, file.name);
    body.append("type", "input");
    body.append("subfolder", `Myang_node/director/${String(shot.id).replace(/[^a-zA-Z0-9_-]/g, "_")}`);
    const response = await fetch("/upload/image", {method: "POST", body});
    if (!response.ok) throw new Error(`上传失败（HTTP ${response.status}）`);
    return response.json();
}

function insertAtCursor(editor, text) {
    editor.focus();
    if (editor.isContentEditable) {
        insertAtCaret(editor, document.createTextNode(text));
        editor.dispatchEvent(new Event("input", {bubbles: true}));
        return;
    }
    const start = editor.selectionStart ?? editor.value.length;
    const end = editor.selectionEnd ?? start;
    editor.value = `${editor.value.slice(0, start)}${text}${editor.value.slice(end)}`;
    editor.selectionStart = editor.selectionEnd = start + text.length;
    editor.dispatchEvent(new Event("input", {bubbles: true}));
}

function renderShotAssets(node, shot, prompt, options = {}) {
    const allowedKinds = new Set(options.allowedKinds || ["image", "video", "audio"]);
    const limits = options.limits || {};
    const showAssetMode = options.showAssetMode !== false;
    const section = document.createElement("div");
    section.style.cssText = "margin-top:7px;border:1px solid #2f3e50;background:#111923;border-radius:6px;padding:7px;";
    const head = document.createElement("div");
    head.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:8px;margin-bottom:6px;";
    const title = document.createElement("div");
    title.textContent = `${options.title || "镜头素材"} · ${shot.assets.length} 个`;
    title.style.cssText = "font-size:11px;font-weight:700;color:#7dd3fc;";
    head.appendChild(title);
    if (showAssetMode) {
        const mode = document.createElement("select");
        mode.title = "仅本镜头：编号从 @图片1 开始；叠加全局：放在 Media Agent 素材之后";
        mode.style.cssText = "background:#10151c;color:#cbd5e1;border:1px solid #344254;border-radius:4px;padding:3px 5px;font-size:10px;";
        for (const value of ["仅本镜头", "叠加全局素材"]) {
            const option = document.createElement("option");
            option.value = option.textContent = value;
            mode.appendChild(option);
        }
        mode.value = shot.asset_mode;
        mode.onchange = () => { shot.asset_mode = mode.value; saveTimeline(node); };
        head.appendChild(mode);
    }
    section.appendChild(head);

    const tools = document.createElement("div");
    tools.style.cssText = "display:flex;gap:5px;align-items:center;flex-wrap:wrap;margin-bottom:6px;";
    const status = document.createElement("span");
    status.style.cssText = "font-size:9px;color:#94a3b8;";
    for (const [kind, meta] of Object.entries(MEDIA_META)) {
        if (!allowedKinds.has(kind)) continue;
        const limit = Number(limits[kind] ?? meta.limit);
        const add = button(`＋ ${kind === "video" && options.videoRole === "action" ? "动作视频" : meta.label}`,
            `本区域最多 ${limit} 个${meta.label}`);
        const picker = document.createElement("input");
        picker.type = "file";
        picker.accept = meta.accept;
        picker.multiple = true;
        picker.style.display = "none";
        add.onclick = () => picker.click();
        picker.onchange = async () => {
            const current = shot.assets.filter((asset) => asset.kind === kind).length;
            const files = Array.from(picker.files || []).slice(0, Math.max(0, limit - current));
            if (!files.length) {
                status.textContent = current >= limit ? `${meta.label}已达到上限` : "未选择文件";
                return;
            }
            add.disabled = true;
            status.textContent = `正在上传 ${files.length} 个${meta.label}…`;
            try {
                for (const file of files) {
                    const saved = await uploadShotFile(shot, file);
                    const asset = normalizeAsset({
                        kind,
                        label: file.name,
                        file: saved,
                    }, shot.assets.length);
                    if (kind === "video" && options.videoRole) asset.role = options.videoRole;
                    shot.assets.push(asset);
                }
                saveTimeline(node);
                renderTimeline(node);
            } catch (error) {
                console.error("[Myang Director] 镜头素材上传失败", error);
                status.textContent = error?.message || "上传失败";
                status.style.color = "#fb7185";
            } finally {
                add.disabled = false;
                picker.value = "";
            }
        };
        tools.append(add, picker);
    }
    tools.appendChild(status);
    section.appendChild(tools);

    if (!shot.assets.length) {
        const empty = document.createElement("div");
        empty.textContent = options.emptyText || "直接添加本镜头专用素材；文件保存在 ComfyUI/input，工作流只记录引用。";
        empty.style.cssText = "font-size:9px;color:#64748b;padding:4px 1px;";
        section.appendChild(empty);
        return section;
    }

    const list = document.createElement("div");
    list.style.cssText = "display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:6px;";
    const resolvedMaterials = options.resolve
        ? options.resolve() : directorMediaList(node, shot);
    const typeOrdinals = {image: 0, video: 0, audio: 0};
    shot.assets.forEach((asset, assetIndex) => {
        typeOrdinals[asset.kind] += 1;
        const ordinal = typeOrdinals[asset.kind];
        const meta = MEDIA_META[asset.kind];
        const resolved = resolvedMaterials.find((entry) => entry.assetId === asset.id);
        const mention = resolved?.token || `@${meta.token}${ordinal}`;
        const item = document.createElement("div");
        item.style.cssText = "min-width:0;border:1px solid #334155;background:#0d141d;border-radius:5px;padding:5px;";
        if (!allowedKinds.has(asset.kind)) {
            item.style.borderColor = "#be5b65";
            const warning = document.createElement("div");
            warning.textContent = "当前任务不使用这里的视频，请移除后改接左侧专用视频输入";
            warning.style.cssText = "color:#fda4af;font-size:9px;line-height:1.35;margin-bottom:4px;";
            item.appendChild(warning);
        }
        const preview = document.createElement(asset.kind === "image" ? "img" : asset.kind);
        preview.src = assetViewUrl(asset);
        if (asset.kind === "image") {
            preview.alt = asset.label;
            preview.loading = "lazy";
            preview.style.cssText = "display:block;width:100%;height:72px;object-fit:cover;border-radius:3px;background:#090d12;";
        } else {
            preview.controls = true;
            preview.preload = "metadata";
            if (asset.kind === "video") preview.muted = true;
            preview.style.cssText = asset.kind === "video"
                ? "display:block;width:100%;height:72px;object-fit:cover;border-radius:3px;background:#090d12;"
                : "display:block;width:100%;height:30px;margin:21px 0;";
        }
        item.appendChild(preview);
        const row = document.createElement("div");
        row.style.cssText = "display:flex;gap:4px;align-items:center;margin-top:4px;";
        const ref = button(mention, prompt
            ? `插入 ${mention} 到提示词光标处`
            : `${mention}（LLM 会自动分配，也可手动写进剧本）`);
        ref.style.color = "#7dd3fc";
        if (prompt) ref.onclick = () => insertAtCursor(prompt, mention);
        else ref.disabled = true;
        const label = document.createElement("input");
        label.value = asset.label && asset.label !== asset.file.name ? asset.label : "";
        label.placeholder = "主体名（可选）";
        label.title = `绑定主体名；留空时标签只显示 ${mention}\n文件：${asset.file.name}`;
        label.style.cssText = "min-width:0;flex:1;background:#0b1118;color:#cbd5e1;border:1px solid #293548;border-radius:3px;padding:3px 4px;font-size:9px;";
        label.oninput = () => {
            asset.label = label.value.trim() || asset.file.name;
            saveTimeline(node);
        };
        const remove = button("×", "移除素材引用（不会删除 input 中的文件）");
        remove.style.color = "#fb7185";
        remove.onclick = () => {
            shot.assets.splice(assetIndex, 1);
            saveTimeline(node);
            renderTimeline(node);
        };
        row.append(ref, label, remove);
        item.appendChild(row);
        if (asset.kind === "video" && options.allowVideoRole === true) {
            const role = document.createElement("select");
            role.style.cssText = "width:100%;margin-top:4px;background:#10151c;color:#fbbf24;border:1px solid #344254;border-radius:3px;padding:3px;font-size:9px;";
            for (const [value, text] of [["reference", "参考视频"], ["action", "动作源（逐镜头迁移/续写）"]]) {
                const option = document.createElement("option");
                option.value = value;
                option.textContent = text;
                role.appendChild(option);
            }
            role.value = asset.role;
            role.onchange = () => {
                if (role.value === "action") {
                    shot.assets.forEach((other) => {
                        if (other.kind === "video" && other !== asset) other.role = "reference";
                    });
                }
                asset.role = role.value;
                saveTimeline(node);
                renderTimeline(node);
            };
            item.appendChild(role);
        }
        list.appendChild(item);
    });
    section.appendChild(list);
    return section;
}

function modeNotice(titleText, text, ready) {
    const box = document.createElement("div");
    box.style.cssText = `border:1px solid ${ready ? "#3d6d57" : "#7c5c2c"};background:${ready ? "#14251f" : "#2a2215"};border-radius:7px;padding:9px 10px;margin-bottom:8px;color:#c8d5df;font-size:10px;line-height:1.55;`;
    const title = document.createElement("div");
    title.textContent = titleText;
    title.style.cssText = `font-weight:700;color:${ready ? "#86efac" : "#fbbf24"};margin-bottom:2px;`;
    const body = document.createElement("div");
    body.textContent = text;
    box.append(title, body);
    return box;
}

function renderTransferPanel(node, root) {
    const shot = node.__myangDirectorShots[0] || freshShot(1);
    if (!node.__myangDirectorShots.length) node.__myangDirectorShots.push(shot);
    const linked = inputLinked(node, "ref_video");
    const uploadedVideos = shot.assets.filter((asset) => asset.kind === "video");
    const hasUpload = uploadedVideos.length === 1;
    const conflict = linked && hasUpload;
    root.appendChild(modeNotice(
        conflict ? "动作视频来源冲突" : hasUpload ? "导演台动作视频已就绪"
            : linked ? "外接动作视频已就绪（兼容模式）" : "请在下方上传动作参考视频",
        conflict
            ? "导演台上传和左侧 ref_video 同时存在，请只保留一个。"
            : "在导演台上传一个完整动作视频即可，无需外接加载节点。运行时按下方时长自动连续切段，所有段落共用同一提示词，尾段不会被裁掉。",
        (hasUpload || linked) && !conflict));

    const card = document.createElement("div");
    card.style.cssText = "border:1px solid #42546a;background:#1a222c;border-radius:7px;padding:9px;";
    const row = document.createElement("div");
    row.style.cssText = "display:grid;grid-template-columns:1fr 120px;gap:7px;align-items:end;margin-bottom:7px;";
    const label = document.createElement("label");
    label.textContent = "全片统一提示词";
    label.style.cssText = "color:#7dd3fc;font-size:10px;font-weight:700;";
    const durationLabel = document.createElement("label");
    durationLabel.textContent = "自动分段时长（秒）";
    durationLabel.style.cssText = "color:#fbbf24;font-size:10px;font-weight:700;";
    const duration = document.createElement("input");
    duration.type = "number";
    duration.min = "1";
    duration.max = "30";
    duration.step = "0.5";
    duration.value = String(widget(node, "segment_seconds")?.value || 10);
    duration.style.cssText = "width:100%;box-sizing:border-box;background:#111820;color:#ffd866;border:1px solid #44546a;border-radius:4px;padding:6px;";
    duration.onchange = () => setNativeWidget(node, "segment_seconds", Number(duration.value) || 10);
    const durationWrap = document.createElement("div");
    durationWrap.append(durationLabel, duration);
    row.append(label, durationWrap);
    card.appendChild(row);
    card.appendChild(renderReferenceVideoPanel(node));
    card.appendChild(renderResumePanel(node));

    const prompt = createPromptEditor(node, shot, {
        placeholder: "描述要迁移的动作；键入 @ 或直接输入 @图片1 / @视频1，所有自动分段共用此提示词",
    });
    card.appendChild(prompt);
    card.appendChild(renderShotAssets(node, shot, prompt, {
        title: "动作源与全片辅助素材",
        allowedKinds: ["image", "video", "audio"],
        limits: {video: 1},
        videoRole: "action",
        showAssetMode: false,
        emptyText: "上传一个完整动作视频；还可添加目标人物图片和辅助音频。",
    }));
    root.appendChild(card);
}

const DETAIL_FIELDS = [
    "二采开启", "二采模式", "二采分辨率", "二采自定义宽", "二采自定义高",
    "二采步数", "二采重绘幅度", "二采调度器", "二采采样器", "二采放大方式",
    "二采分块帧数", "二采Latent模型", "二采精度", "二采时间分块", "二采轮数",
    "二采种子策略", "save_raw_segments",
];

const REFERENCE_FIELDS = [
    "参考视频分辨率", "参考视频自定义宽", "参考视频自定义高",
];

const SKILL_FIELDS = ["skill_preset", "skill_text", "vlm_service"];

function setNativeWidget(node, name, value) {
    const target = widget(node, name);
    if (!target) return;
    target.value = value;
    target.callback?.(value);
    node.graph?.setDirtyCanvas?.(true, true);
    requestAnimationFrame(() => refresh(node));
}

function detailControl(node, name, label, options = {}) {
    const target = widget(node, name);
    if (!target) return null;
    const values = options.values === true
        ? (target.options?.values || []) : options.values;
    const wrap = document.createElement("label");
    wrap.style.cssText = "display:flex;flex-direction:column;gap:3px;min-width:0;color:#aebdcd;font-size:9px;line-height:1.35;";
    const caption = document.createElement("span");
    caption.textContent = label;
    wrap.appendChild(caption);
    let input;
    if (Array.isArray(values)) {
        input = document.createElement("select");
        for (const value of values) {
            const option = document.createElement("option");
            option.value = option.textContent = value;
            input.appendChild(option);
        }
        input.value = String(target.value ?? values[0] ?? "");
        input.onchange = () => setNativeWidget(node, name, input.value);
    } else if (options.type === "checkbox") {
        input = document.createElement("input");
        input.type = "checkbox";
        input.checked = target.value !== false;
        input.onchange = () => setNativeWidget(node, name, input.checked);
        wrap.style.flexDirection = "row";
        wrap.style.alignItems = "center";
        wrap.style.gap = "6px";
        wrap.innerHTML = "";
        wrap.append(input, caption);
        return wrap;
    } else {
        input = document.createElement("input");
        input.type = "number";
        input.value = String(target.value ?? options.default ?? 0);
        if (options.min != null) input.min = String(options.min);
        if (options.max != null) input.max = String(options.max);
        if (options.step != null) input.step = String(options.step);
        input.onchange = () => setNativeWidget(node, name, Number(input.value));
    }
    input.setAttribute("aria-label", label);
    input.style.cssText = "min-width:0;width:100%;box-sizing:border-box;background:#0c131c;color:#dbe7f3;border:1px solid #37475b;border-radius:4px;padding:5px 6px;font-size:10px;outline:none;";
    input.onfocus = () => { input.style.borderColor = "#60a5fa"; };
    input.onblur = () => { input.style.borderColor = "#37475b"; };
    wrap.appendChild(input);
    return wrap;
}

function renderResumePanel(node) {
    const start = Math.max(1, Number(widget(node, "起始段")?.value || 1));
    const linked = inputLinked(node, "前段视频");
    const panel = document.createElement("div");
    const ok = start === 1 ? !linked : linked;
    panel.style.cssText = `border:1px solid ${ok ? "#334155" : "#7c5c2c"};background:#101923;border-radius:6px;padding:7px;margin-bottom:7px;`;
    const title = document.createElement("div");
    title.textContent = "断点续跑";
    title.style.cssText = "color:#93c5fd;font-size:10px;font-weight:700;margin-bottom:5px;";
    const grid = document.createElement("div");
    grid.style.cssText = "display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:6px;";
    grid.appendChild(detailControl(node, "起始段", "从第几段开始", {
        min: 1, max: 64, step: 1,
    }));
    const help = document.createElement("div");
    help.textContent = start === 1
        ? (linked
            ? "『前段视频』已接线，但起始段还是 1。整条重跑请断开它；续跑请把起始段改成未生成的那一段。"
            : "起始段 1 = 整条重跑。已经跑出前几段时，把起始段改成断掉的那一段，只补剩下的。")
        : (linked
            ? `从第 ${start} 段开始生成。参考视频仍按原分段对齐取第 ${start} 段那一片；`
              + `『前段视频』只取结尾锚点帧做接缝，不会被当成 @视频1 参考调用。落盘文件名保持第 ${String(start).padStart(2, "0")} 段。`
            : `从第 ${start} 段开始需要把第 ${start - 1} 段的成片接到左侧『前段视频』输入。`);
    help.style.cssText = `color:${ok ? "#718096" : "#fbbf24"};font-size:9px;line-height:1.45;margin-top:5px;`;
    panel.append(title, grid, help);
    return panel;
}

function renderReferenceVideoPanel(node) {
    const resolution = String(widget(node, "参考视频分辨率")?.value || REFERENCE_ORIGINAL);
    const panel = document.createElement("div");
    panel.style.cssText = "border:1px solid #334155;background:#101923;border-radius:6px;padding:7px;margin-bottom:7px;";
    const title = document.createElement("div");
    title.textContent = "参考视频预处理";
    title.style.cssText = "color:#93c5fd;font-size:10px;font-weight:700;margin-bottom:5px;";
    const grid = document.createElement("div");
    grid.style.cssText = "display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:6px;";
    grid.appendChild(detailControl(node, "参考视频分辨率", "分辨率", {
        values: REFERENCE_RESOLUTIONS,
    }));
    if (resolution === "自定义") {
        grid.appendChild(detailControl(node, "参考视频自定义宽", "自定义宽", {
            min: 32, max: 1920, step: 32,
        }));
        grid.appendChild(detailControl(node, "参考视频自定义高", "自定义高", {
            min: 32, max: 1920, step: 32,
        }));
    }
    const help = document.createElement("div");
    help.textContent = resolution === REFERENCE_ORIGINAL
        ? "保持输入视频原始宽高，不做空间缩放。"
        : "保持参考视频原比例分块缩放；最高限制到横屏 1920×1080 / 竖屏 1080×1920，不改变一采输出分辨率。";
    help.style.cssText = "color:#718096;font-size:9px;line-height:1.45;margin-top:5px;";
    panel.append(title, grid, help);
    return panel;
}

function renderDetailPanel(node) {
    const enabled = widget(node, "二采开启")?.value === true;
    const mode = String(widget(node, "二采模式")?.value || "放大 + 二采（推荐）");
    const resolution = String(widget(node, "二采分辨率")?.value || "832P");
    const method = String(widget(node, "二采放大方式")?.value || "");
    const sampling = mode !== "仅放大（不二采·最快）";
    const upscaling = mode !== "同分辨率二采（不放大）";
    const neural = upscaling && method.includes("neural_3d");
    const pixelLike = upscaling && (method.includes("pixel") || method.includes("vsr"));
    const externalConnected = (node.inputs || []).some(
        (input) => input.name === "二采设置" && input.link != null);
    const modelConnected = (node.inputs || []).some(
        (input) => input.name === "二采模型" && input.link != null);

    const expanded = node.__myangDetailPanelExpanded ?? enabled;
    const panel = document.createElement("section");
    panel.dataset.myangCollapsible = "detail";
    panel.style.cssText = `flex:0 0 auto;min-height:38px;border:1px solid ${enabled ? "#3b82f6" : "#334155"};background:${enabled ? "#101d30" : "#111820"};border-radius:7px;margin-bottom:8px;overflow:hidden;`;
    const header = document.createElement("div");
    header.style.cssText = "display:flex;align-items:stretch;justify-content:space-between;gap:8px;min-height:38px;padding:0 8px;";
    const summary = document.createElement("button");
    summary.type = "button";
    summary.setAttribute("aria-expanded", String(expanded));
    summary.setAttribute("aria-controls", `myang-detail-body-${node.id}`);
    summary.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:8px;min-width:0;flex:1;border:0;background:transparent;color:#dbeafe;font-size:11px;font-weight:700;text-align:left;padding:8px 0;cursor:pointer;outline:none;";
    const summaryText = document.createElement("span");
    summaryText.textContent = enabled
        ? `二采已开启 · ${mode}${upscaling ? ` · ${resolution}` : ""}`
        : "二采已关闭 · 点击展开设置";
    summaryText.style.cssText = "min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;";
    const indicator = document.createElement("span");
    indicator.textContent = expanded ? "−" : "+";
    indicator.setAttribute("aria-hidden", "true");
    indicator.style.cssText = "flex:0 0 20px;text-align:center;color:#93c5fd;font-size:16px;line-height:1;";
    summary.append(indicator, summaryText);
    summary.onclick = () => {
        node.__myangDetailPanelExpanded = !expanded;
        renderTimeline(node);
    };
    summary.onfocus = () => { summary.style.boxShadow = "inset 0 0 0 2px #3b82f6"; };
    summary.onblur = () => { summary.style.boxShadow = "none"; };
    const toggleLabel = detailControl(node, "二采开启", "开启二采", {type: "checkbox"});
    toggleLabel.style.cssText += ";flex:0 0 auto;align-self:center;min-width:82px;justify-content:flex-end;white-space:nowrap;";
    header.append(summary, toggleLabel);
    panel.appendChild(header);

    const body = document.createElement("div");
    body.id = `myang-detail-body-${node.id}`;
    body.hidden = !expanded;
    body.style.cssText = "border-top:1px solid #2b3c52;padding:8px;";
    if (!expanded) body.style.display = "none";
    if (externalConnected) {
        const warning = document.createElement("div");
        warning.textContent = "已连接旧版『二采设置』：为兼容旧工作流，外部设置优先于本面板。断开后由导演台面板接管。";
        warning.style.cssText = "border:1px solid #92400e;background:#2b1d0d;color:#fbbf24;border-radius:5px;padding:6px;margin-bottom:7px;font-size:9px;line-height:1.45;";
        body.appendChild(warning);
    }
    if (enabled && sampling) {
        const modelState = document.createElement("div");
        modelState.textContent = modelConnected
            ? "二采模型已连接：将使用未挂 Turbo LoRA 的 Ref2VA 基模。"
            : "需要连接左侧『二采 Ref2VA 基模』；仅放大模式不需要模型。";
        modelState.style.cssText = `margin-bottom:7px;color:${modelConnected ? "#86efac" : "#fb7185"};font-size:9px;`;
        body.appendChild(modelState);
    }
    const grid = document.createElement("div");
    grid.style.cssText = "display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:6px;";
    const add = (control) => { if (control) grid.appendChild(control); };
    add(detailControl(node, "二采模式", "处理模式", {values: [
        "放大 + 二采（推荐）", "同分辨率二采（不放大）", "仅放大（不二采·最快）",
    ]}));
    if (upscaling) {
        add(detailControl(node, "二采分辨率", "输出短边", {values: [
            "540P", "640P", "720P", "768P", "832P", "928P", "1024P", "1080P", "自定义",
        ]}));
        add(detailControl(node, "二采放大方式", "放大算法", {values: [
            "neural_3d (神经3D Latent放大·推荐)",
            "latent (latent空间放大·jingchen573方式)",
            "pixel (像素放大·自用版工作流方式)",
            "nvidia_rtx_vsr (NVIDIA RTX 视频超分·实验)",
        ]}));
        if (resolution === "自定义") {
            add(detailControl(node, "二采自定义宽", "自定义宽", {min: 32, max: 8192, step: 32}));
            add(detailControl(node, "二采自定义高", "自定义高", {min: 32, max: 8192, step: 32}));
        }
    }
    if (sampling) {
        add(detailControl(node, "二采步数", "采样步数", {min: 1, max: 100, step: 1}));
        add(detailControl(node, "二采重绘幅度", "重绘幅度", {min: 0.01, max: 1, step: 0.01}));
        add(detailControl(node, "二采调度器", "调度器", {values: ["beta", "simple", "normal"]}));
        add(detailControl(node, "二采采样器", "采样器", {values: ["res_multistep", "euler"]}));
        add(detailControl(node, "二采轮数", "二采轮数", {min: 1, max: 8, step: 1}));
        add(detailControl(node, "二采种子策略", "种子策略", {values: [
            "每轮沿用同一种子", "每轮种子 +1",
        ]}));
    }
    if (pixelLike) add(detailControl(node, "二采分块帧数", "像素 / VSR 分块帧数", {min: 1, max: 64, step: 1}));
    if (neural) {
        const modelValues = widget(node, "二采Latent模型")?.options?.values;
        add(detailControl(node, "二采Latent模型", "神经 3D 模型", {
            values: Array.isArray(modelValues) ? modelValues : [String(widget(node, "二采Latent模型")?.value || "")],
        }));
        add(detailControl(node, "二采精度", "神经 3D 精度", {values: [
            "fp16（推荐·省显存）", "fp32（最高稳定性）", "bf16（实验）",
        ]}));
        add(detailControl(node, "二采时间分块", "神经 3D 时间分块", {min: 1, max: 128, step: 1}));
    }
    if (enabled && widget(node, "save_segments")?.value !== false) {
        add(detailControl(node, "save_raw_segments", "同时保存二采前分段", {type: "checkbox"}));
    }
    body.appendChild(grid);
    panel.appendChild(body);
    return panel;
}

function renderGlobalAssets(node, options = {}) {
    if (!Array.isArray(node.__myangDirectorGlobals)) node.__myangDirectorGlobals = [];
    const bucket = {
        id: "__global__",
        assets: node.__myangDirectorGlobals,
        asset_mode: "叠加全局素材",
    };
    const agentLinked = inputLinked(node, "media");
    const panel = document.createElement("div");
    panel.style.cssText = "border:1px solid #334155;background:#101923;border-radius:6px;padding:7px;margin-bottom:8px;";
    const title = document.createElement("div");
    title.textContent = "公共素材（全片共用）";
    title.style.cssText = "color:#93c5fd;font-size:10px;font-weight:700;margin-bottom:5px;";
    panel.appendChild(title);
    const help = document.createElement("div");
    help.textContent = options.llmPicks
        ? "在这里上传全片共用的角色图、场景图、参考视频和配乐，并给每个素材起一个主体名。"
          + "智能切分时素材清单会连同剧本一起交给 LLM，由它判断每一段该引用哪些素材并写上 @图片N 标签。"
        : "全片共用素材。镜头卡选择「叠加全局素材」时，这里的素材排在镜头专属素材之前；"
          + "选择「仅本镜头」的镜头不会拿到它们。";
    help.style.cssText = "color:#718096;font-size:9px;line-height:1.5;margin-bottom:6px;";
    panel.appendChild(help);
    if (agentLinked) {
        const note = document.createElement("div");
        note.textContent = "已接 Media Agent：这里上传的素材排在 Agent 素材之后编号。";
        note.style.cssText = "color:#fbbf24;font-size:9px;line-height:1.5;margin-bottom:6px;";
        panel.appendChild(note);
    }
    panel.appendChild(renderShotAssets(node, bucket, options.prompt || null, {
        title: "公共素材",
        allowedKinds: options.allowedKinds || ["image", "video", "audio"],
        showAssetMode: false,
        resolve: () => globalMediaList(node),
        emptyText: "还没有公共素材。文件保存在 ComfyUI/input，工作流只记录引用。",
    }));
    return panel;
}

function renderSkillPanel(node) {
    const preset = String(widget(node, "skill_preset")?.value || "auto");
    const vlm = String(widget(node, "vlm_service")?.value || "off");
    const custom = String(widget(node, "skill_text")?.value || "");
    const panel = document.createElement("div");
    panel.style.cssText = "border:1px solid #334155;background:#101923;border-radius:6px;padding:7px;margin-bottom:8px;";
    const title = document.createElement("div");
    title.textContent = "写作技能与素材识图";
    title.style.cssText = "color:#93c5fd;font-size:10px;font-weight:700;margin-bottom:5px;";
    const grid = document.createElement("div");
    grid.style.cssText = "display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:6px;";
    grid.appendChild(detailControl(node, "skill_preset", "技能", {values: true}));
    grid.appendChild(detailControl(node, "vlm_service", "素材识图 VLM", {values: true}));
    panel.append(title, grid);

    const details = document.createElement("details");
    details.open = Boolean(custom.trim());
    details.style.cssText = "margin-top:6px;";
    const summary = document.createElement("summary");
    summary.textContent = custom.trim()
        ? `自定义写作规则（已填 ${custom.trim().length} 字）` : "自定义写作规则（可选）";
    summary.style.cssText = "cursor:pointer;color:#7dd3fc;font-size:9px;font-weight:700;outline:none;";
    const area = document.createElement("textarea");
    area.value = custom;
    area.rows = 4;
    area.placeholder = "写在这里的规则排在所选技能之前，优先级最高。留空则完全按技能来。";
    area.style.cssText = "width:100%;box-sizing:border-box;margin-top:5px;background:#0c131c;color:#dbe7f3;border:1px solid #37475b;border-radius:4px;padding:5px 6px;font-size:10px;line-height:1.5;resize:vertical;outline:none;";
    area.oninput = () => {
        const target = widget(node, "skill_text");
        if (!target) return;
        target.value = area.value;
        node.graph?.setDirtyCanvas?.(true, true);
    };
    details.append(summary, area);
    panel.appendChild(details);

    const help = document.createElement("div");
    help.textContent = `${preset === "auto"
        ? "auto：拆分前先用一次很短的调用，按剧本从技能库里挑一个。"
        : preset === "none" ? "不使用技能，按默认 H3 提示词写法拆分。"
            : `固定使用「${preset}」的输出结构、分镜格式和素材标签写法。`} `
        + `${vlm === "off"
            ? "素材识图关闭：清单只给主体名和文件名。开启后 VLM 会先看一遍每个素材，把画面内容写进清单，LLM 分配素材会准得多。"
            : "素材识图已开启：拆分前先让 VLM 描述每个公共素材的画面内容，再交给 LLM 逐段分配。"}`;
    help.style.cssText = "color:#718096;font-size:9px;line-height:1.5;margin-top:6px;";
    panel.appendChild(help);
    return panel;
}

function renderSourcePanel(node, options = {}) {
    const manual = options.manual === true;
    const continuing = options.continuing === true;
    const panel = document.createElement("section");
    panel.dataset.myangDirectorSection = "source";
    panel.style.cssText = "border:1px solid #334155;background:#101923;border-radius:7px;padding:8px;margin-bottom:8px;min-width:0;";

    const sourceGrid = document.createElement("div");
    sourceGrid.style.cssText = `display:grid;grid-template-columns:${manual
        ? "minmax(0,1fr)" : "minmax(210px,1.15fr) repeat(2,minmax(110px,.55fr))"};gap:6px;align-items:end;min-width:0;`;
    const add = (control) => { if (control) sourceGrid.appendChild(control); };
    add(detailControl(node, "source_mode", "分镜来源", {values: true}));
    if (!manual) {
        add(detailControl(node, "total_seconds", "目标总时长", {
            min: 1, max: 3600, step: 1,
        }));
        add(detailControl(node, "segment_seconds", "智能切分单段时长", {
            min: 0.2, max: 30, step: 0.5,
        }));
    }
    panel.appendChild(sourceGrid);

    if (manual) {
        const note = document.createElement("div");
        note.textContent = "手动分镜卡直接用于生成；二采、画质增强和分镜增强仍在下方统一设置。";
        note.style.cssText = "color:#718096;font-size:9px;line-height:1.45;margin-top:6px;";
        panel.appendChild(note);
        return panel;
    }

    const target = widget(node, "script_fallback");
    const linked = inputLinked(node, "script_fallback");
    const labelRow = document.createElement("div");
    labelRow.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:8px;margin:8px 0 4px;";
    const label = document.createElement("label");
    label.textContent = "长剧本 / Agent 提示词";
    label.htmlFor = `myang-script-${node.id}`;
    label.style.cssText = "color:#7dd3fc;font-size:10px;font-weight:700;";
    const status = document.createElement("span");
    status.textContent = linked ? "已由外接 Agent 输入接管" : "随内容增高 · 最高 220px";
    status.style.cssText = `color:${linked ? "#fbbf24" : "#64748b"};font-size:9px;white-space:nowrap;`;
    labelRow.append(label, status);

    const area = document.createElement("textarea");
    area.id = `myang-script-${node.id}`;
    area.value = String(target?.value || "");
    area.readOnly = linked;
    area.placeholder = linked
        ? "运行时使用左侧接入的 Agent myang_prompt；断开连接后可在这里编辑。"
        : continuing
            ? "填写需要续写的后续剧情；前文视频只走 Motion Context 输入。"
            : "填写长剧本，或在左侧把 Agent myang_prompt 转换并连接到此输入。";
    area.setAttribute("aria-label", "长剧本 / Agent 提示词");
    area.style.cssText = `display:block;width:100%;min-width:0;max-width:100%;box-sizing:border-box;min-height:${SCRIPT_INPUT_MIN_HEIGHT}px;max-height:${SCRIPT_INPUT_MAX_HEIGHT}px;background:#0c131c;color:${linked ? "#8290a1" : "#dbe7f3"};border:1px solid ${linked ? "#5b4b2a" : "#37475b"};border-radius:5px;padding:7px 8px;font-size:10px;line-height:1.55;resize:none;outline:none;white-space:pre-wrap;overflow-wrap:anywhere;`;
    area.onfocus = () => { area.style.borderColor = linked ? "#a16207" : "#60a5fa"; };
    area.onblur = () => { area.style.borderColor = linked ? "#5b4b2a" : "#37475b"; };
    area.oninput = () => {
        if (!target || linked) return;
        target.value = area.value;
        fitScriptTextArea(area);
        node.graph?.setDirtyCanvas?.(true, true);
    };
    panel.append(labelRow, area);
    node.__myangDirectorScriptInput = area;
    requestAnimationFrame(() => fitScriptTextArea(area));

    const llmGrid = document.createElement("div");
    llmGrid.style.cssText = "display:grid;grid-template-columns:minmax(140px,.65fr) minmax(220px,1.35fr);gap:6px;align-items:end;margin-top:7px;min-width:0;";
    const enabledControl = detailControl(node, "llm_enabled", "智能切片", {type: "checkbox"});
    const serviceControl = detailControl(node, "llm_service", "LLM 服务", {values: true});
    if (enabledControl) llmGrid.appendChild(enabledControl);
    if (serviceControl) llmGrid.appendChild(serviceControl);
    panel.appendChild(llmGrid);
    return panel;
}

function transferPlanToStoryboard(node) {
    const snapshot = normalizePlanSnapshot(node.__myangDirectorPlan);
    if (!snapshot?.segments?.length) return;
    const source = widget(node, "source_mode");
    if (!source) {
        window.alert?.("找不到导演台的分镜来源控件，无法转入分镜卡。");
        return;
    }
    const existing = (node.__myangDirectorShots || []).some(
        (shot) => String(shot?.prompt || "").trim());
    if (existing && !window.confirm?.(
        "转入后会用最近一次 LLM 分段覆盖当前手动分镜卡。公共素材不会删除，是否继续？")) return;

    const fallbackSeconds = Number(widget(node, "segment_seconds")?.value || 5);
    node.__myangDirectorShots = snapshot.segments.map((segment, index) => {
        const duration = Number(segment.duration_seconds || 0)
            || (Number(segment.frames || 0) > 0 ? Number(segment.frames) / 24 : fallbackSeconds);
        return {
            id: `shot_plan_${Date.now().toString(36)}_${index + 1}`,
            enabled: true,
            duration_seconds: Math.max(0.2, Math.min(30, duration)),
            brief: String(segment.brief || `镜头 ${index + 1}`),
            prompt: String(segment.prompt || ""),
            transition: String(segment.transition || (index === 0 ? "开场" : "承接")),
            fixed_from_plan: true,
            asset_mode: "仅本镜头",
            assets: [],
        };
    });
    node.__myangStoryboardMetadata = {
        title: String(node.title || "H3导演台分镜卡"),
        source: snapshot.source,
        style_header: snapshot.style_header,
        skill_source: snapshot.skill_source,
    };
    source.value = MANUAL;
    saveTimeline(node);
    refresh(node);
}

function updateSegmentPlan(node) {
    const list = node.__myangDirectorPlanList;
    if (!list?.isConnected) return;
    const plan = node.__myangDirectorPlan;
    const segments = plan?.segments || [];
    if (node.__myangDirectorPlanMeta) {
        const saved = plan?.saved_at ? new Date(plan.saved_at) : null;
        const time = saved && !Number.isNaN(saved.getTime())
            ? ` · ${saved.toLocaleTimeString([], {hour: "2-digit", minute: "2-digit"})}` : "";
        node.__myangDirectorPlanMeta.textContent = segments.length
            ? `${segments.length} 段${plan.skill_source ? ` · 技能 ${plan.skill_source}` : ""}${time}`
            : "尚未运行";
    }
    if (node.__myangDirectorImportButton) {
        node.__myangDirectorImportButton.disabled = !segments.length;
        node.__myangDirectorImportButton.style.opacity = segments.length ? "1" : ".45";
        node.__myangDirectorImportButton.style.cursor = segments.length ? "pointer" : "not-allowed";
    }
    list.replaceChildren();
    if (!segments.length) {
        const empty = document.createElement("div");
        empty.textContent = "运行后这里会列出 LLM 实际拆出的每段提示词，"
            + "素材标签显示为可辨认的缩略图，方便核对 LLM 有没有分对素材。";
        empty.style.cssText = "font-size:9px;color:#64748b;line-height:1.5;padding:2px 1px;";
        list.appendChild(empty);
        return;
    }
    const materials = globalMediaList(node);
    const state = progressState(node);
    const active = state.status === "running" ? Number(state.seg || 0) : 0;
    segments.forEach((segment, offset) => {
        const running = active > 0 && offset + 1 === active;
        const card = document.createElement("div");
        card.style.cssText = `border:1px solid ${running ? "#60a5fa" : "#334155"};background:${running ? "#132133" : "#0d141d"};border-radius:5px;padding:6px 7px;`;
        const head = document.createElement("div");
        head.style.cssText = "display:flex;align-items:center;gap:6px;margin-bottom:4px;";
        const tag = document.createElement("span");
        tag.textContent = `第 ${segment.index} 段`;
        tag.style.cssText = `font-size:9px;font-weight:700;color:${running ? "#93c5fd" : "#7dd3fc"};flex:none;`;
        head.appendChild(tag);
        const transition = String(segment.transition || "");
        if (transition) {
            const cut = transition === "切镜";
            const badge = document.createElement("span");
            badge.textContent = transition;
            badge.title = cut
                ? "LLM 判断这里换场景/换视角，本段直接进新镜头并重新交代机位与环境"
                : transition === "开场" ? "全片第一段" : "承接上一段的镜头位置、人物姿态和环境";
            badge.style.cssText = `flex:none;font-size:8px;padding:1px 4px;border-radius:3px;border:1px solid ${cut ? "#a16207" : "#334155"};color:${cut ? "#fbbf24" : "#94a3b8"};background:${cut ? "#2a2215" : "#131c27"};`;
            head.appendChild(badge);
        }
        const meta = document.createElement("span");
        const seconds = Number(segment.duration_seconds || 0);
        meta.textContent = `${seconds ? `${seconds.toFixed(2)}s` : ""}${segment.frames ? ` · ${segment.frames} 帧` : ""}`;
        meta.style.cssText = "font-size:9px;color:#64748b;flex:none;";
        const brief = document.createElement("span");
        brief.textContent = String(segment.brief || "");
        brief.title = brief.textContent;
        brief.style.cssText = "font-size:9px;color:#94a3b8;flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;";
        const copy = button("复制", "复制这一段的提示词原文");
        copy.style.cssText += ";flex:none;font-size:9px;padding:1px 5px;";
        copy.onclick = () => {
            navigator.clipboard?.writeText(String(segment.prompt || ""));
            copy.textContent = "已复制";
            setTimeout(() => { copy.textContent = "复制"; }, 1200);
        };
        head.append(meta, brief, copy);
        card.appendChild(head);        const body = document.createElement("div");
        body.className = "myh3-editor";
        body.style.cssText = "height:auto;max-height:none;overflow:visible;font-size:10px;line-height:1.55;background:#0b1118;cursor:text;user-select:text;";
        renderPromptInto(body, segment.prompt, materials);
        card.appendChild(body);
        list.appendChild(card);
    });
}

function renderSegmentPlanPanel(node) {
    const plan = node.__myangDirectorPlan;
    const panel = document.createElement("div");
    panel.style.cssText = "border:1px solid #334155;background:#101923;border-radius:6px;padding:7px;margin-bottom:8px;";
    const head = document.createElement("div");
    head.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:8px;margin-bottom:5px;";
    const heading = document.createElement("div");
    heading.style.cssText = "flex:1;min-width:0;";
    const title = document.createElement("div");
    title.textContent = "最近一次分段提示词 · 已自动保留";
    title.style.cssText = "color:#93c5fd;font-size:10px;font-weight:700;";
    const meta = document.createElement("div");
    meta.textContent = plan?.segments?.length
        ? `${plan.segments.length} 段${plan.skill_source ? ` · 技能 ${plan.skill_source}` : ""}`
        : "尚未运行";
    meta.style.cssText = "font-size:9px;color:#8e9aaa;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;";
    heading.append(title, meta);
    const importButton = button("转入导演台分镜卡", "固定最近一次分段；后续运行不再调用 LLM，可逐卡编辑");
    importButton.setAttribute("aria-label", "转入导演台分镜卡并停止后续 LLM 重写");
    importButton.style.cssText += ";flex:0 0 132px;min-height:32px;background:#2563a8;border-color:#3b82c4;color:#fff;font-weight:700;transition:background .2s ease,border-color .2s ease;";
    importButton.disabled = !plan?.segments?.length;
    importButton.style.opacity = plan?.segments?.length ? "1" : ".45";
    importButton.style.cursor = plan?.segments?.length ? "pointer" : "not-allowed";
    importButton.onclick = () => transferPlanToStoryboard(node);
    head.append(heading, importButton);
    panel.appendChild(head);
    const hint = document.createElement("div");
    hint.textContent = "分段一生成就写入当前工作流，中途停止视频也不会清除。转入后会切到手动分镜卡，换 LoRA 重跑不会再次请求 LLM。";
    hint.style.cssText = "font-size:9px;color:#718096;line-height:1.45;margin:-1px 0 6px;";
    panel.appendChild(hint);
    if (plan?.style_header) {
        const header = document.createElement("div");
        header.textContent = `全局设定：${plan.style_header}`;
        header.style.cssText = "font-size:9px;color:#94a3b8;line-height:1.5;background:#0b1118;border-radius:4px;padding:5px 6px;margin-bottom:5px;";
        panel.appendChild(header);
    }
    const list = document.createElement("div");
    list.style.cssText = "display:flex;flex-direction:column;gap:5px;padding-right:3px;";
    panel.appendChild(list);
    node.__myangDirectorPlanList = list;
    node.__myangDirectorPlanMeta = meta;
    node.__myangDirectorImportButton = importButton;
    updateSegmentPlan(node);
    return panel;
}

function renderTimeline(node) {
    const root = node.__myangDirectorRoot;
    if (!root?.isConnected) return;
    root.innerHTML = "";
    const task = currentTask(node);
    const transferring = task === TRANSFER;
    const continuing = task === CONTINUE;
    const manual = !transferring && String(widget(node, "source_mode")?.value || MANUAL) === MANUAL;
    const importedStoryboard = manual && (node.__myangDirectorShots || []).some(
        (shot) => shot.imported_storyboard === true);
    const fixedStoryboard = manual && (node.__myangDirectorShots || []).some(
        (shot) => shot.fixed_from_plan === true);
    const lockedStoryboard = importedStoryboard || fixedStoryboard;
    node.__myangDirectorScriptInput = null;

    const header = document.createElement("div");
    header.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:8px;margin-bottom:8px;";
    const title = document.createElement("div");
    title.style.cssText = "font-weight:700;color:#a9dc76;font-size:13px;";
    title.textContent = transferring ? "动作迁移导演台"
        : continuing ? "Motion Context 视频续写"
            : manual ? (importedStoryboard ? "导演台分镜卡 · 已导入"
                : fixedStoryboard ? "导演台分镜卡 · 已固定" : "纯生成分镜时间线")
                : "Agent / 长剧本智能切分";
    header.appendChild(title);
    const stats = document.createElement("div");
    stats.style.cssText = "font-size:10px;color:#8e9aaa;";
    const info = timelineStats(node);
    const turboConnected = (node.inputs || []).some(
        (input) => input.name === "Turbo联合模型" && input.link != null);
    const turboSpec = turboConnected ? turboStepSpec(node) : null;
    const baseStats = transferring
        ? `单一动作源 · ${Number(widget(node, "segment_seconds")?.value || 10).toFixed(1)} 秒自动分段`
        : manual
        ? `${info.count} 个启用镜头 · 成片约 ${info.seconds.toFixed(2)} 秒${lockedStoryboard ? " · 不调用 LLM" : ""}`
        : "执行时由分段计划节点一次拆镜头";
    stats.textContent = `${baseStats}${turboConnected ? ` · Turbo ${turboSpec?.label || "官方档位"}` : ""}`;
    header.appendChild(stats);
    root.appendChild(header);
    root.appendChild(renderDirectorProgressPanel(node));
    if (!transferring) root.appendChild(renderSourcePanel(node, {manual, continuing}));
    // These are generation-wide post-processing controls.  Keep them outside
    // every source-mode branch so manual cards, Agent splitting, continuation
    // and action transfer all expose exactly the same second-pass toolchain.
    root.appendChild(renderDetailPanel(node));

    if (transferring) {
        renderTransferPanel(node, root);
        root.appendChild(renderOutputVideoPanel(node));
        return;
    }

    if (continuing) {
        const linked = inputLinked(node, "ref_video");
        root.appendChild(modeNotice(
            linked ? "前文视频已连接：仅用于 Motion Context" : "等待连接前文视频",
            "前文视频只在第一段提取末尾锚点；后续段落使用上一段生成 latent 继续，不会把前文视频当作 @视频1 参考调用。镜头卡只允许添加图片和音频。",
            linked));
        root.appendChild(renderReferenceVideoPanel(node));
    }

    root.appendChild(renderGlobalAssets(node, {
        llmPicks: !manual,
        allowedKinds: continuing ? ["image", "audio"] : ["image", "video", "audio"],
        prompt: manual ? null : node.__myangDirectorScriptInput,
    }));
    if (!manual) root.appendChild(renderSkillPanel(node));
    if (!manual) root.appendChild(renderSegmentPlanPanel(node));

    if (!manual) {
        const help = document.createElement("div");
        help.style.cssText = "border:1px solid #334155;background:#17202b;border-radius:7px;padding:12px;color:#aab7c6;font-size:11px;line-height:1.6;";
        help.textContent = continuing
            ? "把要续写的剧情接到 script 或填写长剧本。前文视频仍只走左侧 Motion Context 输入；LLM 只负责拆分未来剧情，并按上方公共素材清单逐段分配 @图片N 标签。"
            : "把 Media Agent 的 myang_prompt 接到 script，或在下方填写长剧本。LLM 开启时逐镜头拆分，并按上方公共素材清单自行判断每段引用哪些素材；关闭时各段共用原提示词和全部公共素材，不消耗 Token。";
        root.appendChild(help);
        root.appendChild(renderOutputVideoPanel(node));
        return;
    }

    const toolbar = document.createElement("div");
    toolbar.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:8px;margin-bottom:8px;";
    const toolbarStatus = document.createElement("div");
    toolbarStatus.textContent = importedStoryboard
        ? "已从结构化分镜卡文件导入；卡片可继续编辑，运行时不会调用 LLM。"
        : fixedStoryboard
        ? "已从 LLM 快照转入；下方提示词可直接修改，换模型或 LoRA 重跑不会重新拆分。"
        : "手动分镜卡会直接用于生成，不经过 LLM。";
    toolbarStatus.style.cssText = `font-size:9px;line-height:1.4;color:${lockedStoryboard ? "#86efac" : "#718096"};flex:1;min-width:0;`;
    const toolbarActions = document.createElement("div");
    toolbarActions.style.cssText = "display:flex;gap:4px;flex:0 0 auto;align-items:center;";
    const importCards = button("导入分镜卡", "从结构化 JSON 文件重建导演台分镜卡");
    importCards.setAttribute("aria-label", "导入结构化导演台分镜卡");
    importCards.style.cssText += ";flex:0 0 88px;width:88px;min-height:30px;transition:background .2s ease,border-color .2s ease;";
    importCards.onclick = () => chooseStoryboardFile(node);
    const exportCards = button("导出分镜卡", "把当前卡片、时长、转场和素材引用导出为结构化 JSON");
    exportCards.setAttribute("aria-label", "导出结构化导演台分镜卡");
    exportCards.style.cssText += ";flex:0 0 88px;width:88px;min-height:30px;transition:background .2s ease,border-color .2s ease;";
    exportCards.onclick = () => exportStoryboardCards(node);
    const add = button("＋ 添加镜头");
    add.style.cssText += ";flex:0 0 88px;width:88px;min-height:30px;background:#628f3d;";
    add.style.color = "#fff";
    add.onclick = () => {
        node.__myangDirectorShots.push(freshShot(node.__myangDirectorShots.length + 1));
        saveTimeline(node);
        renderTimeline(node);
    };
    toolbarActions.append(importCards, exportCards, add);
    toolbar.append(toolbarStatus, toolbarActions);
    root.appendChild(toolbar);
    if (node.__myangStoryboardNotice?.message) {
        const notice = document.createElement("div");
        const failed = node.__myangStoryboardNotice.tone === "error";
        notice.textContent = node.__myangStoryboardNotice.message;
        notice.setAttribute("role", failed ? "alert" : "status");
        notice.style.cssText = `border:1px solid ${failed ? "#7f3345" : "#2f6b4b"};background:${failed ? "#2a151d" : "#10251b"};color:${failed ? "#fda4af" : "#86efac"};border-radius:5px;padding:5px 7px;margin:-2px 0 8px;font-size:9px;line-height:1.45;`;
        root.appendChild(notice);
    }

    const list = document.createElement("div");
    list.style.cssText = "display:flex;flex-direction:column;gap:7px;padding-right:3px;";
    node.__myangDirectorShots.forEach((shot, index) => {
        const card = document.createElement("div");
        card.style.cssText = `border:1px solid ${shot.enabled ? "#42546a" : "#30343b"};background:${shot.enabled ? "#1a222c" : "#191b20"};border-radius:7px;padding:8px;opacity:${shot.enabled ? "1" : ".58"};`;
        const top = document.createElement("div");
        top.style.cssText = "display:grid;grid-template-columns:24px 1fr 82px auto;gap:6px;align-items:center;margin-bottom:6px;";
        const enabled = document.createElement("input");
        enabled.type = "checkbox";
        enabled.checked = shot.enabled;
        enabled.title = "是否运行这个镜头";
        enabled.onchange = () => {
            shot.enabled = enabled.checked;
            saveTimeline(node);
            renderTimeline(node);
        };
        top.appendChild(enabled);
        const brief = document.createElement("input");
        brief.value = shot.brief;
        brief.placeholder = `镜头 ${index + 1} 标题`;
        brief.style.cssText = "background:#111820;color:#e7edf4;border:1px solid #344254;border-radius:4px;padding:5px 7px;min-width:0;";
        brief.oninput = () => { shot.brief = brief.value; saveTimeline(node); };
        top.appendChild(brief);
        const duration = document.createElement("input");
        duration.type = "number";
        duration.min = "0.2";
        duration.max = "30";
        duration.step = "0.1";
        duration.value = String(shot.duration_seconds);
        duration.title = "镜头时长（会吸附到 H3 的 17k+5 帧网格）";
        duration.style.cssText = "background:#111820;color:#ffd866;border:1px solid #344254;border-radius:4px;padding:5px 6px;width:100%;box-sizing:border-box;";
        duration.onchange = () => {
            shot.duration_seconds = Math.max(0.2, Math.min(30, Number(duration.value) || 5));
            saveTimeline(node);
            renderTimeline(node);
        };
        top.appendChild(duration);
        const actions = document.createElement("div");
        actions.style.cssText = "display:flex;gap:3px;";
        const up = button("↑", "上移");
        up.disabled = index === 0;
        up.onclick = () => {
            if (index < 1) return;
            [node.__myangDirectorShots[index - 1], node.__myangDirectorShots[index]] =
                [node.__myangDirectorShots[index], node.__myangDirectorShots[index - 1]];
            saveTimeline(node); renderTimeline(node);
        };
        const down = button("↓", "下移");
        down.disabled = index === node.__myangDirectorShots.length - 1;
        down.onclick = () => {
            if (index >= node.__myangDirectorShots.length - 1) return;
            [node.__myangDirectorShots[index + 1], node.__myangDirectorShots[index]] =
                [node.__myangDirectorShots[index], node.__myangDirectorShots[index + 1]];
            saveTimeline(node); renderTimeline(node);
        };
        const duplicate = button("⧉", "复制");
        duplicate.onclick = () => {
            const copy = {
                ...shot,
                id: `shot_${Date.now().toString(36)}`,
                assets: shot.assets.map((asset) => ({...asset, file: {...asset.file}})),
            };
            node.__myangDirectorShots.splice(index + 1, 0, copy);
            saveTimeline(node); renderTimeline(node);
        };
        const remove = button("×", "删除");
        remove.style.color = "#ff6188";
        remove.onclick = () => {
            if (node.__myangDirectorShots.length <= 1) return;
            node.__myangDirectorShots.splice(index, 1);
            saveTimeline(node); renderTimeline(node);
        };
        actions.append(up, down, duplicate, remove);
        top.appendChild(actions);
        card.appendChild(top);

        const prompt = createPromptEditor(node, shot, {
            placeholder: "写完整 H3 提示词；键入 @ 选择素材，@图片1 / @视频1 会自动显示为匹配标签",
        });
        card.appendChild(prompt);
        card.appendChild(renderShotAssets(node, shot, prompt, continuing ? {
            title: "续写辅助图片 / 音频",
            allowedKinds: ["image", "audio"],
            emptyText: "前文视频不要放在镜头素材里；它只接左侧 Motion Context 视频输入。",
        } : {}));

        const grid = document.createElement("div");
        grid.style.cssText = "font-size:9px;color:#778493;margin-top:4px;";
        grid.textContent = `${Number(shot.duration_seconds).toFixed(1)}s → ${alignedFrames(shot.duration_seconds)} 帧 @24fps`;
        card.appendChild(grid);
        list.appendChild(card);
    });
    root.appendChild(list);
    root.appendChild(renderOutputVideoPanel(node));
}

function refresh(node) {
    migrateLegacyInputs(node);
    const by = {};
    for (const item of node.widgets || []) {
        by[item.name] = item;
        if (LABELS[item.name]) item.label = LABELS[item.name];
    }
    const task = String(by.task_mode?.value || FRESH);
    const transferring = task === TRANSFER;
    const manual = !transferring && String(by.source_mode?.value || MANUAL) === MANUAL;
    hideWidget(by.timeline_json);
    for (const name of DETAIL_FIELDS) hideWidget(by[name]);
    for (const name of REFERENCE_FIELDS) hideWidget(by[name]);
    for (const name of SKILL_FIELDS) hideWidget(by[name]);
    hideWidget(by["起始段"]);
    // 断点续跑只在动作迁移面板里露出。换成别的任务模式后这个值就是上一次的残留，
    // 用户看不到也改不回去，所以这里直接归位，别让它跟着工作流存下去。
    if (!transferring && by["起始段"] && Number(by["起始段"].value) !== 1) {
        by["起始段"].value = 1;
    }
    // Source controls are rendered inside the full-width Director source card.
    // Keeping the native multiline widget visible made it retain ComfyUI's
    // narrow pre-resize width and pushed the universal second-pass cards away.
    for (const name of [
        "source_mode", "script_fallback", "total_seconds", "segment_seconds",
        "llm_enabled", "llm_service",
    ]) hideWidget(by[name]);
    const custom = String(by.resolution?.value || "") === "自定义";
    setVisible(by.width, custom);
    setVisible(by.height, custom);
    setVisible(by.segment_prefix, by.save_segments?.value !== false);
    const turboSpec = turboStepSpec(node);
    const turboConnected = turboSpec !== null;
    if (by.steps) {
        if (turboConnected && turboSpec?.recommended != null) {
            by.steps.label = `一采步数（手动；Turbo 推荐 ${turboSpec.recommended}）`;
        } else {
            by.steps.label = LABELS.steps;
        }
    }
    // Turbo 只锁定完整降噪和调度器，步数始终由导演台自己的控件决定。
    setVisible(by.steps, true);
    setVisible(by.denoise, !turboConnected);
    setVisible(by.scheduler, !turboConnected);
    for (const input of node.inputs || []) {
        if (input.name === "ref_video") {
            input.label = transferring ? "动作视频兼容输入（导演台可直接上传）"
                : task === CONTINUE ? "前文视频（仅 Motion Context）"
                    : "此模式不使用专用视频输入";
        } else if (input.name === "ref_audio") {
            input.label = transferring ? "动作参考音频（同步切段）"
                : task === CONTINUE ? "前文原音频（用于接缝）"
                    : "此模式不使用专用音频输入";
        } else if (input.name === "media") {
            input.label = task === FRESH ? "Media Agent 素材包"
                : "辅助图片 / 音频素材包（不要放视频）";
        } else if (input.name === "前段视频") {
            input.label = transferring ? "前段成片（仅段间上下文，不当参考视频）"
                : "仅动作迁移断点续跑使用";
        } else if (INPUT_LABELS[input.name]) input.label = INPUT_LABELS[input.name];
        else if (LABELS[input.name]) input.label = LABELS[input.name];
    }
    syncPanelGeometry(node);
    renderTimeline(node);
    node.__myangDirectorMediaSig = mediaSignature(node);
    node.__myangDirectorTurboSig = turboSignature(node);
    node.graph?.setDirtyCanvas?.(true, true);
}

function syncPanelGeometry(node) {
    const root = node.__myangDirectorRoot;
    if (!root) return;
    const width = Math.max(240, Number(node.size?.[0] || 700) - 24);
    const cssWidth = `${width}px`;
    for (const element of [root, root.parentElement]) {
        if (!element) continue;
        element.style.boxSizing = "border-box";
        element.style.width = cssWidth;
        element.style.minWidth = cssWidth;
        element.style.maxWidth = cssWidth;
    }
    syncOutputVideoGeometry(node);
    syncScriptInputHeight(node);
}

function makePanel(node) {
    const root = document.createElement("div");
    root.className = "myh3-director-root";
    root.style.cssText = "box-sizing:border-box;width:100%;height:100%;min-width:240px;min-height:180px;max-height:none;display:flex;flex-direction:column;background:#131923;border:1px solid #2c3949;border-radius:8px;padding:10px;overflow-x:hidden;overflow-y:auto;scrollbar-gutter:stable;";
    // Form interaction belongs to the DOM widget, not the canvas underneath.
    // In particular, a pointer-down inside an input must not repeatedly select
    // the node and trigger canvas lifecycle hooks while the user is typing.
    for (const eventName of ["pointerdown", "mousedown", "click", "dblclick", "wheel"]) {
        root.addEventListener(eventName, (event) => event.stopPropagation());
    }
    for (const eventName of ["keydown", "keyup", "keypress"]) {
        root.addEventListener(eventName, (event) => event.stopPropagation());
    }
    node.__myangDirectorRoot = root;
    renderTimeline(node);
    return root;
}

app.registerExtension({
    name: "Myang_node.MiniMaxH3.Director",
    setup() {
        styleOnce();
        if (!directorWatcher) {
            directorWatcher = setInterval(() => {
                try {
                    for (const node of app.graph?._nodes || []) {
                        if (node.type !== NODE || !node.__myangDirectorRoot) continue;
                        const media = mediaSignature(node);
                        const turbo = turboSignature(node);
                        if (media !== node.__myangDirectorMediaSig
                            || turbo !== node.__myangDirectorTurboSig) refresh(node);
                        mountNativeVideoPreview(node);
                    }
                } catch (error) {
                    console.warn("[Myang Director] 素材 / Turbo 状态同步失败", error);
                }
            }, 1000);
        }
        api.addEventListener("myh3_director_plan", (event) => {
            try {
                const detail = event?.detail || {};
                const node = directorByOwner(detail.owner_id);
                if (!node) return;
                node.__myangDirectorPlan = normalizePlanSnapshot({
                    ...detail,
                    saved_at: new Date().toISOString(),
                });
                // The plan arrives before video sampling starts. Persist it in
                // the hidden timeline widget immediately so an interrupted run
                // cannot erase the prompts the LLM already produced.
                saveTimeline(node);
                // 只重填分段列表，不走 renderTimeline：那会重建整个面板，
                // 把用户正在编辑的输入框和光标一起干掉。
                if (node.__myangDirectorPlanList?.isConnected) updateSegmentPlan(node);
                else requestAnimationFrame(() => refresh(node));
            } catch (error) {
                console.warn("[Myang Director] 接收分段提示词失败", error);
            }
        });
        api.addEventListener("myh3_longvideo_start", (event) => {
            try {
                const detail = event?.detail || {};
                const node = directorByOwner(detail.owner_id);
                if (!node) return;
                clearOutputVideo(node);
                node.__myangDirectorProgress = {
                    status: "running", runId: String(detail.run_id || ""),
                    total: Number(detail.total_segments || 1), seg: 1,
                    phase: "sample1", step: 0, stepMax: 0,
                    previewFile: "", previewTs: 0, prompt: "", brief: "",
                    refining: !!detail.refining, correcting: !!detail.correcting,
                    error: "",
                };
                updateDirectorProgress(node);
            } catch (error) {
                console.warn("[Myang Director] 初始化进度失败", error);
            }
        });
        api.addEventListener("myh3_progress", (event) => {
            try {
                const detail = event?.detail || {};
                const node = directorByOwner(detail.owner_id);
                if (node) applyDirectorProgress(node, detail);
            } catch (error) {
                console.warn("[Myang Director] 更新进度失败", error);
            }
        });
        api.addEventListener("execution_success", () => {
            for (const node of app.graph?._nodes || []) {
                if (node.type !== NODE || progressState(node).status !== "running") continue;
                node.__myangDirectorProgress.status = "done";
                node.__myangDirectorProgress.phase = "done";
                updateDirectorProgress(node);
            }
        });
        const fail = (message) => {
            for (const node of app.graph?._nodes || []) {
                if (node.type !== NODE || progressState(node).status !== "running") continue;
                node.__myangDirectorProgress.status = "error";
                node.__myangDirectorProgress.error = String(message || "执行中断");
                updateDirectorProgress(node);
            }
        };
        api.addEventListener("execution_error", (event) =>
            fail(event?.detail?.exception_message || event?.detail?.error || "执行出错"));
        api.addEventListener("execution_interrupted", () => {
            fail("已中断，正在释放模型与执行缓存");
            // /interrupt in older ComfyUI builds only stops the kernel.  This
            // package endpoint also clears Myang's learned-upscaler CPU cache
            // and asks the worker to run the supported full cleanup path.
            void api.fetchApi("/minimax-h3-myang/free-memory", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: "{}",
            }).catch((error) => console.warn("[Myang Director] 中断后释放内存失败", error));
        });
        window.addEventListener("mousedown", (event) => {
            if (openDirectorMenu && !openDirectorMenu.contains(event.target)) closeDirectorMenu();
        });
    },
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== NODE) return;
        const onCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onCreated?.apply(this, arguments);
            this.__myangDirectorShots = parseTimeline(this);
            this.__myangDirectorGlobals = parseGlobalAssets(this);
            this.__myangDirectorPlan = parsePlanSnapshot(this);
            this.__myangStoryboardMetadata = parseStoryboardMetadata(this);
            const panel = this.addDOMWidget(
                "myang_director_panel", "director", makePanel(this), {
                    serialize: false,
                    hideOnZoom: false,
                    getMinHeight: () => 260,
                    getMaxHeight: () => Math.max(520, Number(this.size?.[1] || 920)),
                    getHeight: () => "100%",
                });
            panel.serialize = false;
            for (const item of this.widgets || []) {
                if (item === panel) continue;
                const previous = item.callback;
                item.callback = (...args) => {
                    const value = previous?.apply(item, args);
                    if (item.name !== TIMELINE_WIDGET && !this.__myangDirectorSaving) {
                        requestAnimationFrame(() => refresh(this));
                    }
                    return value;
                };
            }
            this.size = [Math.max(700, this.size?.[0] || 700),
                Math.max(920, this.size?.[1] || 920)];
            requestAnimationFrame(() => {
                syncPanelGeometry(this);
                refresh(this);
            });
            return result;
        };

        for (const hook of ["onConfigure", "onAdded", "onConnectionsChange"]) {
            const original = nodeType.prototype[hook];
            nodeType.prototype[hook] = function () {
                const result = original?.apply(this, arguments);
                if (hook === "onConfigure") {
                    this.__myangDirectorShots = parseTimeline(this);
                    this.__myangDirectorGlobals = parseGlobalAssets(this);
                    this.__myangDirectorPlan = parsePlanSnapshot(this);
                    this.__myangStoryboardMetadata = parseStoryboardMetadata(this);
                }
                requestAnimationFrame(() => refresh(this));
                return result;
            };
        }

        // Selection changes may require a width correction, but must never
        // call refresh(): refresh rebuilds the form DOM and destroys focus,
        // open selects and the active text caret.
        for (const hook of ["onSelected", "onDeselected"]) {
            const original = nodeType.prototype[hook];
            nodeType.prototype[hook] = function () {
                const result = original?.apply(this, arguments);
                requestAnimationFrame(() => syncPanelGeometry(this));
                return result;
            };
        }

        const onResize = nodeType.prototype.onResize;
        nodeType.prototype.onResize = function (size) {
            size[0] = Math.max(700, size[0]);
            size[1] = Math.max(520, size[1]);
            const result = onResize?.apply(this, arguments);
            requestAnimationFrame(() => syncPanelGeometry(this));
            return result;
        };

        // Keep ComfyUI's native video output intact.  Once its asynchronous
        // video-preview widget exists, move the actual HTMLVideoElement into
        // the adaptive card at the bottom of the Director panel.
        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (output) {
            const result = onExecuted?.call(this, output);
            scheduleNativeVideoMount(this);
            return result;
        };
    },
});
