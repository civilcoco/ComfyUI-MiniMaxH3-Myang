import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import {
    STORYBOARD_CARD_FORMAT,
    STORYBOARD_CARD_VERSION,
    createStoryboardCardDocument,
    parseStoryboardCardDocument,
    storyboardCardFileName,
} from "./h3_storyboard_cards.js";
import {
    SPEECH_RATES,
    budgetUnits,
    composeSegmentPrompt,
    dialogueSeconds,
    normalizeGlobalLayers,
    normalizeSegmentLayers,
    speechUnits,
} from "./h3_prompt_layers.js";
import {
    applyRoughCutCommit,
    renderRoughCutCard,
} from "./h3_roughcut_ui.js";
import {
    DIRECTOR_TEMPLATE_ENCRYPTED_FORMAT,
    createDirectorTemplateDocument,
    decryptDirectorTemplateDocument,
    directorTemplateInputSlots,
    encryptDirectorTemplateDocument,
    instantiateDirectorTemplate,
    parseDirectorTemplateDocument,
} from "./h3_director_templates.js";

const STORYBOARD_FILE_MAX_BYTES = 10 * 1024 * 1024;
const TEMPLATE_BUNDLE_MAX_BYTES = 128 * 1024 * 1024;
// A portable template stores media as base64, and an encrypted export base64
// encodes the complete JSON once more. 128 MiB of source media can therefore
// legitimately grow beyond 192 MiB without being corrupt.
const TEMPLATE_BUNDLE_MAX_FILE_BYTES = 256 * 1024 * 1024;
const TEMPLATE_BUNDLE_MAX_ASSET_BYTES = 64 * 1024 * 1024;
const TEMPLATE_BUNDLE_VERSION = 1;

const NODE = "H3Director";
const MANUAL = "导演台分镜卡（手动逐镜头）";
const TIMELINE_WIDGET = "timeline_json";
const ROUGHCUT_WIDGET = "粗剪工程";
const TRANSFER = "动作迁移（跟随参考视频）";
const CONTINUE = "视频续写（接着往下演）";
const FRESH = "纯生成（不用参考视频）";
const DIRECTOR_TASK_MODES = [FRESH, TRANSFER, CONTINUE];
const AGENT_LINKS = "minimax_h3_agent_media_connections";
const AGENT_MEDIA_PROP = "myang_h3_asset_sources_v2";
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
const ASSET_CATEGORY_LABELS = {
    character: "角色", scene: "场景", voice: "音色", music: "音乐",
    video: "视频", image: "图片", other: "其他",
};
const MEDIA_LIBRARY_ROUTE = "/minimax-h3-myang/roughcut";
const MEDIA_TOTAL_LIMIT = 12;

const PROGRESS_PHASE = {
    prepare: {start: 0.00, span: 0.02, label: "导演台准备", steps: false},
    sample1: {start: 0.02, span: 0.30, label: "一采采样", steps: true},
    audio_refine: {start: 0.32, span: 0.16, label: "音频精修", steps: true},
    drift: {start: 0.48, span: 0.08, label: "漂移校正", steps: false},
    refine_prep: {start: 0.56, span: 0.10, label: "二采准备", steps: false},
    sample2: {start: 0.66, span: 0.22, label: "二采采样", steps: true},
    finalizing: {start: 0.88, span: 0.06, label: "分段解码", steps: false},
    assembling: {start: 0.94, span: 0.04, label: "合并分段", steps: false},
    exporting: {start: 0.98, span: 0.02, label: "合成视频", steps: false},
};

let openDirectorMenu = null;
let directorWatcher = null;

async function requestDirectorMemoryRelease() {
    const response = await api.fetchApi("/minimax-h3-myang/free-memory", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: "{}",
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
}

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
    segment_seconds: "智能切分单段上限",
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
    "脸部精修开启": "H3 小脸精修",
    "脸部检测器": "脸部检测器",
    "脸部精修步数": "脸部精修步数",
    "脸部精修重绘": "脸部精修重绘",
    "脸部裁剪倍率": "脸部裁剪倍率",
    "脸部身份图序号": "身份参考图序号",
    "动作修复开启": "MAINodes 高速动作修复",
    "动作修复档位": "动作修复档位",
    "动作修复步数": "动作修复步数",
    "动作修复注入": "动作修复注入强度",
    "多视角分镜开启": "角色五视图分镜",
    "多视角角色图片序号": "角色图片序号",
    "多视角尺寸": "五视图尺寸",
    "多视角步数": "五视图步数",
    "多视角LoRA": "五视图 LoRA",
    "多视角LoRA强度": "五视图 LoRA 强度",
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
    "二采复用一采条件": "二采复用文本/素材条件（不含成片）",
    "二采显存策略": "二采显存策略",
    "二采自定义显存预留": "二采自定义显存预留（GB）",
    "二采自定义预览间隔": "二采自定义预览间隔",
    "二采连续Sigma": "连续 Sigma 二采（实验）",
    "音频精修开启": "H3 音频精修",
    "音频精修步数": "音频精修步数",
    "音频去噪强度": "音频去噪强度",
    "音频精修采样器": "音频精修采样器",
    "音频精修调度器": "音频精修调度器",
    "音频接缝平滑": "段间音频接缝平滑",
    "音频接缝时长": "音频接缝时长（ms）",
    save_segments: "保存每段",
    segment_prefix: "分段文件名前缀",
    save_raw_segments: "同时保存二采前分段",
    "参考视频分辨率": "参考视频分辨率",
    "参考视频自定义宽": "参考视频自定义宽",
    "参考视频自定义高": "参考视频自定义高",
    "起始段": "起始段（断点续跑）",
    "从指定段开始": "从指定段开始",
    "动作迁移自动分段": "动作迁移自动分段",
    "一采显存策略": "一采显存策略",
    "一采断点模式": "一采断点 / 直接二采",
    skill_preset: "写作技能",
    skill_text: "自定义写作规则",
    vlm_service: "素材识图 VLM",
    "分层提示词": "分层生成提示词",
    "粗剪时间轴开启": "粗剪时间轴",
    "粗剪工程": "粗剪工程（内部）",
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
    "一采成片": "已保存的一采成片（直接二采·单段）",
    "一采成片音频": "一采成片音轨（可选）",
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
        transition: index === 1 ? "开场" : "承接",
        asset_mode: "仅本镜头",
        assets: [],
        collapsed: false,
        layers_open: false,
    };
}

function normalizeAsset(asset, index) {
    const file = asset?.file && typeof asset.file === "object" ? asset.file : asset || {};
    const kind = ["image", "video", "audio"].includes(asset?.kind) ? asset.kind : "image";
    const result = {
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
    if (["auto", "manual"].includes(asset?.reference_weight_mode)) {
        result.reference_weight_mode = asset.reference_weight_mode;
        result.reference_weight = Math.max(.25, Math.min(3,
            Number(asset?.reference_weight) || 1));
    }
    return result;
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
                transition: index === 0 ? "开场"
                    : shot?.transition === "切镜" ? "切镜" : "承接",
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

function storyboardShortTitle(title, brief = "") {
    const metadata = /(?:人物|角色)?外观参考|画面基准|整体视听|声音与配乐|详细分镜|电影级提示词|生成提示词|脚本与|剧本|纯中文版|全局设定|输出格式/i;
    for (const [sourceIndex, source] of [title, brief].entries()) {
        for (let line of String(source || "").split(/\r?\n/)) {
            line = line.trim();
            if (!line || /^[\s=_*#~\-—·|]+$/.test(line)) continue;
            line = line.replace(/^\s*#{1,6}\s*/, "")
                .replace(/^\s*(?:(?:TITLE|标题|分镜标题|镜头标题)\s*[:：]|(?:第?\s*\d+\s*(?:段|镜|镜头|分镜))\s*[:：、.\-]*)\s*/i, "");
            if (!line || metadata.test(line)) continue;
            if (/^(?:DURATION|TRANSITION|GOAL|SEGMENT_GOAL|SUBJECT\s*\d*|SHOT\s*\d*|时长|持续时间|转场|衔接|叙事目标|本段目标|主体\s*\d*|镜头\s*\d*)\s*[:：]/i.test(line)) continue;
            if (line.includes("《") && line.includes("》")
                && (sourceIndex > 0 || /\d+\s*秒|[上下中]篇|篇\s*\d+/.test(line))) continue;
            line = line.replace(/<[^>]+>|@[\u3400-\u9fffA-Za-z]+\s*\d+/g, "")
                .replace(/\[(?:Shot\s*\d+|Chinese|English|Japanese|Korean)\]/gi, "")
                .replace(/[\[\]【】]/g, "")
                .split(/\|\||[|]|={2,}|—{2,}|[，,。！？!?；;]/, 1)[0];
            const compact = line.replace(/[^\u3400-\u9fffA-Za-z0-9]+/g, "");
            if (compact.length >= 2) return compact.slice(0, 8);
        }
    }
    return "剧情推进";
}

function normalizePlanSnapshot(raw) {
    if (!raw || typeof raw !== "object" || !Array.isArray(raw.segments)) return null;
    const segments = raw.segments.map((segment, index) => ({
        index: Number(segment?.index || index + 1),
        title: storyboardShortTitle(segment?.title, segment?.brief || segment?.prompt),
        brief: String(segment?.brief || `镜头 ${index + 1}`),
        prompt: String(segment?.prompt || ""),
        transition: String(segment?.transition || (index === 0 ? "开场" : "承接")),
        frames: Math.max(0, Number(segment?.frames || 0)),
        duration_seconds: Math.max(0, Number(segment?.duration_seconds || 0)),
        skills: Array.isArray(segment?.skills) ? segment.skills.map(String).filter(Boolean) : [],
        skill_source: String(segment?.skill_source || ""),
        subjects: Array.isArray(segment?.subjects) ? segment.subjects.map((subject) => ({
            id: String(subject?.id || ""),
            name: String(subject?.name || ""),
            media: Array.isArray(subject?.media) ? subject.media.map(String) : [],
            identity: String(subject?.identity || ""),
            state_action: String(subject?.state_action || ""),
            state_revision: Number(subject?.state_revision || 1),
            declaration: String(subject?.declaration || ""),
        })) : [],
        layers: normalizeSegmentLayers(segment?.layers),
    })).filter((segment) => segment.prompt.trim());
    if (!segments.length) return null;
    return {
        version: 1,
        saved_at: String(raw.saved_at || new Date().toISOString()),
        source: String(raw.source || "llm_split"),
        style_header: String(raw.style_header || ""),
        skill_source: String(raw.skill_source || ""),
        skill_strategy: String(raw.skill_strategy || ""),
        skill_plan: Array.isArray(raw.skill_plan) ? raw.skill_plan : [],
        // The global half of the layered prompt: style, scene and the cast every
        // shot inherits. Dropping it here would leave the card editor composing
        // against an empty head, so a layered card would silently lose its
        // subject definitions on the first snapshot round trip.
        layers: normalizeGlobalLayers(raw.layers),
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
    syncModeBucket(node);
    node.__myangDirectorSaving = true;
    try {
        (node.__myangDirectorShots || []).forEach((shot, index) => {
            shot.transition = index === 0 ? "开场"
                : shot.transition === "切镜" ? "切镜" : "承接";
        });
        const activeMode = modeBucketKey(node.__myangDirectorActiveMode);
        const active = node.__myangDirectorModeBuckets[activeMode]
            ||= normalizeModeBucket(null);
        active.shots = node.__myangDirectorShots || [freshShot(1)];
        active.global_assets = node.__myangDirectorGlobals || [];
        active.plan_snapshot = normalizePlanSnapshot(node.__myangDirectorPlan);
        active.storyboard_metadata = node.__myangStoryboardMetadata || null;
        active.template_lock = node.__myangTemplateLock?.enabled
            ? {...node.__myangTemplateLock, enabled: true} : null;
        active.template_contract = node.__myangTemplateContract?.active
            ? {...node.__myangTemplateContract, active: true} : null;
        target.value = JSON.stringify({
            version: 4,
            active_mode: activeMode,
            modes: node.__myangDirectorModeBuckets,
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
    const connectedBoundaries = active.slice(1).filter(
        (shot) => shot.transition !== "切镜").length;
    const outputFrames = frames.reduce((sum, value) => sum + value, 0)
        - connectedBoundaries * overlap;
    return {count: active.length, frames, seconds: Math.max(0, outputFrames) / 24};
}

function modeTaskValue(node) {
    return String(widget(node, "task_mode")?.value || FRESH);
}

function modeBucketKey(value) {
    const task = String(value || FRESH);
    return DIRECTOR_TASK_MODES.includes(task) ? task : FRESH;
}

function normalizeModeBucket(raw) {
    const source = raw && typeof raw === "object" ? raw : {};
    const rawShots = Array.isArray(source.shots) ? source.shots : [];
    const shots = rawShots.length ? rawShots.map((shot, index) => {
        const normalized = {
            id: String(shot?.id || `shot_${index + 1}`),
            enabled: shot?.enabled !== false,
            duration_seconds: Number(shot?.duration_seconds ?? shot?.seconds ?? 5),
            brief: String(shot?.brief || `镜头 ${index + 1}`),
            prompt: String(shot?.prompt || ""),
            transition: index === 0 ? "开场"
                : shot?.transition === "切镜" ? "切镜" : "承接",
            fixed_from_plan: shot?.fixed_from_plan === true,
            imported_storyboard: shot?.imported_storyboard === true,
            asset_mode: shot?.asset_mode === "叠加全局素材" ? "叠加全局素材" : "仅本镜头",
            assets: Array.isArray(shot?.assets)
                ? shot.assets.map((asset, assetIndex) => normalizeAsset(asset, assetIndex))
                    .filter((asset) => asset.file.name)
                : [],
            // View state, persisted on purpose: a long storyboard is unusable if
            // every reload re-expands all of it.
            collapsed: shot?.collapsed === true,
            layers_open: shot?.layers_open === true,
        };
        // This whitelist is the load path for everything a card owns, so a key
        // missing here is silently dropped on reload. Layers used to be, which
        // quietly reverted a layered card to its flat prompt.
        const layers = normalizeSegmentLayers(shot?.layers);
        if (layers) normalized.layers = layers;
        return normalized;
    }) : [freshShot(1)];
    const globalAssets = Array.isArray(source.global_assets)
        ? source.global_assets.map((asset, index) => normalizeAsset(asset, index))
            .filter((asset) => asset.file.name)
            .map((asset) => ({...asset, role: "reference"}))
        : [];
    return {
        shots,
        global_assets: globalAssets,
        plan_snapshot: normalizePlanSnapshot(source.plan_snapshot),
        storyboard_metadata: source.storyboard_metadata && typeof source.storyboard_metadata === "object"
            ? {...source.storyboard_metadata} : null,
        template_lock: source.template_lock?.enabled === true
            ? {
                enabled: true,
                template_id: String(source.template_lock.template_id || ""),
                template_name: String(source.template_lock.template_name || "接口模板"),
            } : null,
        template_contract: source.template_contract?.active === true
            ? {
                active: true,
                total_seconds: Math.max(0, Number(
                    source.template_contract.total_seconds) || 0),
                segment_seconds: Math.max(0, Number(
                    source.template_contract.segment_seconds) || 0),
            } : null,
    };
}

function parseModeBuckets(node) {
    const raw = String(widget(node, TIMELINE_WIDGET)?.value || "");
    let value = null;
    try { value = raw ? JSON.parse(raw) : null; }
    catch (error) {
        console.warn("[Myang Director] 分镜 JSON 恢复失败", error);
    }
    const buckets = {};
    if (value && !Array.isArray(value) && value.modes
        && typeof value.modes === "object") {
        for (const mode of DIRECTOR_TASK_MODES) {
            buckets[mode] = normalizeModeBucket(value.modes[mode]);
        }
        return buckets;
    }
    // Legacy timelines had one shared bucket. Keep it intact, but scope it to
    // the mode that was active when the workflow is first opened.
    const legacyMode = modeBucketKey(modeTaskValue(node));
    buckets[legacyMode] = normalizeModeBucket(
        Array.isArray(value) ? {shots: value} : (value || {}));
    return buckets;
}

function foldTransferGlobals(bucket) {
    if (!bucket || !Array.isArray(bucket.global_assets)
        || !bucket.global_assets.length) return false;
    if (!Array.isArray(bucket.shots) || !bucket.shots.length) {
        bucket.shots = [freshShot(1)];
    }
    const shot = bucket.shots[0];
    shot.assets ||= [];
    const seen = new Set(shot.assets.map((asset) =>
        `${asset?.kind || ""}|${asset?.file?.name || ""}|${asset?.file?.subfolder || ""}`));
    for (const asset of bucket.global_assets) {
        const key = `${asset?.kind || ""}|${asset?.file?.name || ""}|${asset?.file?.subfolder || ""}`;
        if (asset?.file?.name) {
            if (seen.has(key)) continue;
            seen.add(key);
            shot.assets.push({...asset, file: {...asset.file}});
        }
    }
    bucket.global_assets = [];
    return true;
}

function loadModeBucket(node, mode) {
    const key = modeBucketKey(mode);
    node.__myangDirectorActiveMode = key;
    const bucket = node.__myangDirectorModeBuckets[key] ||= normalizeModeBucket(null);
    if (key === TRANSFER && foldTransferGlobals(bucket)) {
        node.__myangDirectorModeMigrationDirty = true;
    }
    node.__myangDirectorShots = bucket.shots;
    node.__myangDirectorGlobals = bucket.global_assets;
    node.__myangDirectorPlan = bucket.plan_snapshot;
    node.__myangStoryboardMetadata = bucket.storyboard_metadata;
    node.__myangTemplateLock = bucket.template_lock;
    node.__myangTemplateContract = bucket.template_contract;
}

function syncModeBucket(node) {
    const mode = modeBucketKey(modeTaskValue(node));
    if (!node.__myangDirectorModeBuckets) {
        node.__myangDirectorModeBuckets = parseModeBuckets(node);
        loadModeBucket(node, mode);
        return false;
    }
    if (node.__myangDirectorActiveMode === mode) {
        const migrated = Boolean(node.__myangDirectorModeMigrationDirty);
        node.__myangDirectorModeMigrationDirty = false;
        return migrated;
    }
    const previous = node.__myangDirectorModeBuckets[node.__myangDirectorActiveMode]
        ||= normalizeModeBucket(null);
    previous.shots = node.__myangDirectorShots || [freshShot(1)];
    previous.global_assets = node.__myangDirectorGlobals || [];
    previous.plan_snapshot = normalizePlanSnapshot(node.__myangDirectorPlan);
    previous.storyboard_metadata = node.__myangStoryboardMetadata || null;
    previous.template_lock = node.__myangTemplateLock?.enabled
        ? {...node.__myangTemplateLock, enabled: true} : null;
    previous.template_contract = node.__myangTemplateContract?.active
        ? {...node.__myangTemplateContract, active: true} : null;
    loadModeBucket(node, mode);
    return true;
}

function button(text, title = "") {
    const element = document.createElement("button");
    element.type = "button";
    element.textContent = text;
    element.title = title;
    element.style.cssText = "border:1px solid #3c4654;background:#252d38;color:#dce5ef;border-radius:5px;padding:4px 8px;cursor:pointer;font-size:11px;";
    return element;
}

function renderShotTransition(node, shot, index) {
    if (index < 1) return null;
    const value = shot.transition === "切镜" ? "切镜" : "承接";
    const boundary = document.createElement("div");
    boundary.setAttribute("role", "group");
    boundary.setAttribute("aria-label", `镜头 ${index} 到镜头 ${index + 1} 的生成关系`);
    boundary.style.cssText = "display:flex;align-items:center;gap:8px;min-height:44px;margin:0 4px;padding:0 4px;";

    const lineBefore = document.createElement("div");
    const lineAfter = document.createElement("div");
    for (const line of [lineBefore, lineAfter]) {
        line.style.cssText = "height:1px;background:#334155;flex:1 1 24px;min-width:12px;";
    }
    const label = document.createElement("span");
    label.textContent = "段间";
    label.style.cssText = "font-size:9px;color:#94a3b8;white-space:nowrap;";
    const controls = document.createElement("div");
    controls.style.cssText = "display:grid;grid-template-columns:72px 72px;gap:3px;flex:0 0 auto;";
    for (const option of ["承接", "切镜"]) {
        const active = value === option;
        const control = button(option, option === "承接"
            ? "使用上一段末尾 latent，通过 MotionContext 无缝续接"
            : "独立生成本段，不读取上一段视频末尾");
        control.setAttribute("aria-pressed", String(active));
        control.style.cssText += active
            ? option === "承接"
                ? ";width:72px;min-height:44px;background:#173b32;border-color:#34a37a;color:#a7f3d0;font-weight:700;"
                : ";width:72px;min-height:44px;background:#3b2b17;border-color:#d6933a;color:#fde68a;font-weight:700;"
            : ";width:72px;min-height:44px;background:#18212c;border-color:#3c4654;color:#94a3b8;";
        control.onclick = () => {
            if (shot.transition === option) return;
            shot.transition = option;
            saveTimeline(node);
            renderTimeline(node);
        };
        controls.appendChild(control);
    }
    const explanation = document.createElement("span");
    explanation.textContent = value === "承接"
        ? "MotionContext 无缝续接" : "独立生成，不参考上一段末尾";
    explanation.style.cssText = `font-size:9px;white-space:nowrap;color:${value === "承接" ? "#86efac" : "#fbbf24"};`;
    boundary.append(lineBefore, label, controls, explanation, lineAfter);
    return boundary;
}

async function stopDirectorLlm(node, control, status) {
    if (control.disabled) return;
    control.disabled = true;
    control.setAttribute("aria-busy", "true");
    control.textContent = "正在停止…";
    status.textContent = "正在切断当前请求";
    status.style.color = "#fbbf24";
    try {
        const response = await api.fetchApi("/minimax-h3-agent/llm-stop", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: "{}",
        });
        const result = await response.json();
        if (!response.ok || result?.success !== true) {
            throw new Error(result?.error || `HTTP ${response.status}`);
        }
        const stopped = Number(result.stopped || 0);
        control.textContent = stopped ? "已停止 LLM" : "停止 LLM";
        status.textContent = String(result.message || (stopped
            ? "当前任务正在中断" : "当前没有 LLM 请求"));
        status.style.color = stopped ? "#86efac" : "#94a3b8";
    } catch (error) {
        control.textContent = "停止失败";
        status.textContent = error?.message || "无法连接本地停止接口";
        status.style.color = "#fb7185";
    } finally {
        control.removeAttribute("aria-busy");
        setTimeout(() => {
            if (!control.isConnected) return;
            const enabled = widget(node, "llm_enabled")?.value !== false;
            control.disabled = !enabled;
            control.textContent = "停止 LLM";
        }, 1600);
    }
}

const MEDIA_META = {
    image: {label: "图片", accept: "image/*", limit: 9, token: "图片"},
    video: {label: "视频", accept: "video/*", limit: 3, token: "视频"},
    audio: {label: "音频", accept: "audio/*", limit: 3, token: "音频"},
};

function canonicalMaterialKind(entry) {
    const value = String(entry?.kind || "");
    if (MEDIA_META[value]) return value;
    return TYPE_OF_KIND[value] || "";
}

function materialQuota(entries, allowedKinds, limits = {}) {
    const counts = {image: 0, video: 0, audio: 0};
    for (const entry of entries || []) {
        const kind = canonicalMaterialKind(entry);
        if (kind) counts[kind] += 1;
    }
    const kinds = [...allowedKinds];
    const parts = kinds.map((kind) => {
        const limit = Number(limits[kind] ?? MEDIA_META[kind]?.limit ?? 0);
        return `${MEDIA_META[kind]?.label || kind} ${counts[kind] || 0}/${limit}`;
    });
    const total = kinds.reduce((sum, kind) => sum + (counts[kind] || 0), 0);
    parts.push(`总计 ${total}/${MEDIA_TOTAL_LIMIT}`);
    return {counts, total, text: parts.join(" · ")};
}

function assetViewUrl(asset) {
    const query = new URLSearchParams({
        filename: asset.file.name,
        subfolder: asset.file.subfolder || "",
        type: "input",
    });
    return `/view?${query.toString()}`;
}

function closeAssetPreview() {
    const overlay = document.getElementById("myang-director-asset-preview");
    if (!overlay) return;
    const media = overlay.querySelector("video, audio");
    if (media) {
        media.pause?.();
        media.removeAttribute("src");
        media.load?.();
    }
    const restoreFocus = overlay.__myangRestoreFocus;
    overlay.remove();
    restoreFocus?.focus?.({preventScroll: true});
}

function openAssetPreview(asset, trigger = null) {
    closeAssetPreview();
    const overlay = document.createElement("div");
    overlay.id = "myang-director-asset-preview";
    overlay.setAttribute("role", "dialog");
    overlay.setAttribute("aria-modal", "true");
    overlay.setAttribute("aria-label", `素材预览：${asset.label || asset.file.name}`);
    guardDialogKeys(overlay);
    overlay.__myangRestoreFocus = trigger || document.activeElement;
    overlay.style.cssText = [
        "position:fixed", "inset:0", "z-index:100000",
        "display:flex", "flex-direction:column", "align-items:center",
        "justify-content:center", "gap:10px", "padding:24px",
        "box-sizing:border-box", "background:rgba(2,6,12,.94)",
        "backdrop-filter:blur(5px)",
    ].join(";");

    const header = document.createElement("div");
    header.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:12px;width:min(96vw,1600px);color:#dbeafe;font:12px/1.4 sans-serif;";
    const title = document.createElement("div");
    title.textContent = asset.label || asset.file.name;
    title.style.cssText = "min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;";
    const close = document.createElement("button");
    close.type = "button";
    close.textContent = "关闭";
    close.setAttribute("aria-label", "关闭素材预览");
    close.style.cssText = "flex:0 0 auto;min-width:64px;min-height:44px;padding:8px 14px;border:1px solid #52647b;border-radius:6px;background:#17202b;color:#f8fafc;font-weight:700;cursor:pointer;";
    close.onmouseenter = () => { close.style.background = "#263446"; };
    close.onmouseleave = () => { close.style.background = "#17202b"; };
    close.onclick = closeAssetPreview;
    header.append(title, close);

    const stage = document.createElement("div");
    stage.style.cssText = "display:flex;align-items:center;justify-content:center;width:min(96vw,1600px);height:min(84vh,1000px);min-height:180px;overflow:hidden;border:1px solid #334155;border-radius:8px;background:#03070c;box-shadow:0 24px 80px rgba(0,0,0,.55);";
    const url = assetViewUrl(asset);
    let media;
    if (asset.kind === "video") {
        media = document.createElement("video");
        media.controls = true;
        media.preload = "metadata";
        media.playsInline = true;
        media.src = url;
    } else if (asset.kind === "audio") {
        media = document.createElement("audio");
        media.controls = true;
        media.preload = "metadata";
        media.src = url;
    } else {
        media = document.createElement("img");
        media.alt = asset.label || asset.file.name || "素材图片预览";
        media.src = url;
    }
    media.style.cssText = asset.kind === "audio"
        ? "display:block;width:min(90vw,760px);height:auto;"
        : "display:block;max-width:100%;max-height:100%;width:auto;height:auto;object-fit:contain;background:#03070c;";
    stage.appendChild(media);
    overlay.append(header, stage);
    overlay.onclick = (event) => {
        if (event.target === overlay) closeAssetPreview();
    };
    overlay.onkeydown = (event) => {
        if (event.key === "Escape") {
            event.preventDefault();
            closeAssetPreview();
        }
    };
    document.body.appendChild(overlay);
    close.focus({preventScroll: true});
}

function upstream(node, inputName) {
    const input = (node.inputs || []).find((item) => item.name === inputName && item.link != null);
    if (!input || !node.graph) return null;
    const link = node.graph.links?.[input.link];
    return link ? node.graph.getNodeById?.(link.origin_id) : null;
}

function globalMediaList(node, options = {}) {
    // Media Agent is a single upstream bundle and cannot represent the three
    // Director task-mode buckets. Keep it opt-in so transfer/continuation
    // cannot accidentally reuse another mode's old materials.
    const allowAgent = options.allowAgent !== false;
    const agent = allowAgent ? upstream(node, "media") : null;
    // The Agent UI moved to a live source-link property. Prefer it even when
    // it is an empty array: falling back to the legacy property in that case
    // would resurrect deleted materials from an older workflow.
    const modern = agent?.properties?.[AGENT_MEDIA_PROP];
    const raw = Array.isArray(modern) ? modern : agent?.properties?.[AGENT_LINKS];
    const counts = {图片: 0, 视频: 0, 音频: 0};
    const entries = [];
    if (Array.isArray(raw)) {
        for (const link of raw) {
            if (Array.isArray(modern) && node.graph
                && Number.isFinite(Number(link?.source_id))) {
                const source = node.graph.getNodeById?.(Number(link.source_id));
                const slot = Number(link?.source_slot || 0);
                // A deleted or disconnected loader must not survive through
                // serialized Agent metadata into the Director prompt.
                if (!source || !source.outputs?.[slot]) continue;
            }
            const kind = KIND_OF_TYPE[String(link?.media_type || "image").toLowerCase()] || "图片";
            const file = String(link?.filename || link?.file?.name || "");
            if (!file) continue;
            const sourceId = Number(link?.source_id);
            const sourceSlot = Number(link?.source_slot || 0);
            const hasSource = Number.isFinite(sourceId) && sourceId >= 0;
            entries.push({
                kind,
                name: String(link?.subject || "") || file || String(link?.label || ""),
                subject: String(link?.subject || "").trim(),
                file,
                subfolder: String(link?.subfolder || ""),
                assetId: hasSource
                    ? `agent:${sourceId}:${sourceSlot}`
                    : `agent:${kind}:${file}:${entries.length}`,
                source_id: hasSource ? sourceId : null,
                source_slot: hasSource ? sourceSlot : null,
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

/** 素材编号严格复刻 H3Condition：动作切片永远先占 @视频1；
 *  纯生成模式才会把 Media Agent 放入当前模式清单，其余模式只排本模式素材。 */
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
        for (const entry of globalMediaList(node, {allowAgent: task === FRESH})) {
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
    return globalMediaList(node, {allowAgent: currentTask(node) === FRESH})
        .map((entry) => `${entry.token}|${entry.file}`).join(",");
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
    block.textContent = text;
    block.contentEditable = "true";
    block.spellcheck = false;
    block.title = "可直接编辑的台词块；运行时保存为 <d>[Chinese] 台词</d>";
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

function insertAtCaret(editor, content) {    const selection = window.getSelection();
    if (!selection?.rangeCount || !editor.contains(selection.anchorNode)) {
        editor.appendChild(content);
        return;
    }
    const range = selection.getRangeAt(0);
    const boundary = content.nodeType === Node.DOCUMENT_FRAGMENT_NODE
        ? content.lastChild : content;
    range.deleteContents();
    range.insertNode(content);
    if (boundary?.parentNode) {
        range.setStartAfter(boundary);
        range.collapse(true);
    } else {
        range.selectNodeContents(editor);
        range.collapse(false);
    }
    selection.removeAllRanges();
    selection.addRange(range);
}

function promptSelectionText(editor) {
    const selection = window.getSelection();
    if (!selection?.rangeCount || !editor.contains(selection.anchorNode)
        || !editor.contains(selection.focusNode)) return "";
    const dialogue = selectedDialogueBlock(editor);
    if (dialogue) return `<d>${dialogue.textContent}</d>`;
    if (selection.isCollapsed) return "";
    const range = selection.getRangeAt(0);
    const holder = document.createElement("span");
    holder.appendChild(range.cloneContents());
    return readPromptText(holder);
}

function selectedDialogueBlock(editor) {
    const selection = window.getSelection();
    if (!selection?.rangeCount || !editor.contains(selection.anchorNode)
        || !editor.contains(selection.focusNode)) return null;
    const closestLine = (node) => (node?.nodeType === Node.ELEMENT_NODE ? node : node?.parentElement)
        ?.closest?.(".myh3-line");
    const anchor = closestLine(selection.anchorNode);
    const focus = closestLine(selection.focusNode);
    return anchor && anchor === focus && editor.contains(anchor) ? anchor : null;
}

function promptFragmentFromText(text, list) {
    const holder = document.createElement("span");
    renderPromptInto(holder, String(text || ""), list || []);
    const fragment = document.createDocumentFragment();
    while (holder.firstChild) fragment.appendChild(holder.firstChild);
    return fragment;
}

function selectDialogueBody(block, selectPlaceholder = false) {
    const text = block.firstChild;
    const selection = window.getSelection();
    if (!selection || !text || text.nodeType !== Node.TEXT_NODE) return;
    const prefix = /^\[(?:Chinese|English|Japanese|Korean|Cantonese)\]\s*/i.exec(text.nodeValue || "")?.[0] || "";
    const range = document.createRange();
    const start = Math.min(prefix.length, text.nodeValue.length);
    range.setStart(text, start);
    range.setEnd(text, selectPlaceholder ? text.nodeValue.length : start);
    selection.removeAllRanges();
    selection.addRange(range);
}

function createDialogueToolbar(editor, sync) {
    const toolbar = document.createElement("div");
    toolbar.className = "myang-director-dialogue-tools";

    const add = document.createElement("button");
    add.type = "button";
    add.className = "myang-director-dialogue-add";
    add.textContent = "＋ 插入对话";
    add.title = "选中文字后点击可转成台词；未选择时会插入一个可直接编辑的中文台词块";
    add.setAttribute("aria-label", "在当前提示词光标处插入对话块");
    add.addEventListener("mousedown", (event) => {
        // Keep the contenteditable selection alive while the toolbar is clicked.
        event.preventDefault();
    });
    add.addEventListener("click", () => {
        editor.focus();
        const selection = window.getSelection();
        const validSelection = Boolean(selection?.rangeCount && editor.contains(selection.anchorNode));
        if (!validSelection) restorePromptCaret(editor, readPromptText(editor).length);
        const activeSelection = window.getSelection();
        const selected = activeSelection?.rangeCount && editor.contains(activeSelection.anchorNode)
            ? String(activeSelection.toString() || "").trim() : "";
        const hasLanguage = /^\[(?:Chinese|English|Japanese|Korean|Cantonese)\]\s*/i.test(selected);
        const placeholder = !selected;
        const dialogue = promptDialogue(
            hasLanguage ? selected : `[Chinese] ${selected || "请输入台词"}`,
        );
        insertAtCaret(editor, dialogue);
        sync();
        editor.focus();
        selectDialogueBody(dialogue, placeholder);
    });
    toolbar.appendChild(add);

    const hint = document.createElement("span");
    hint.className = "myang-director-dialogue-hint";
    hint.textContent = "可先选中文字再转换；语法：<d>[Chinese] 台词</d>";
    toolbar.appendChild(hint);
    return toolbar;
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
    editor.dataset.myangControl = options.controlKey || `shot:${shot.id}:prompt`;
    editor.contentEditable = "true";
    editor.setAttribute("role", "textbox");
    editor.setAttribute("aria-multiline", "true");
    editor.dataset.placeholder = options.placeholder || "输入提示词；键入 @ 可选择素材；上方按钮可插入对话";
    editor.style.cssText = "height:auto;min-height:86px;max-height:none;overflow:visible;resize:none;font-size:11px;line-height:1.55;";

    const materialList = () => directorMediaList(node, shot);
    // A layered shot owns its prose in `layers.visual`, so the big editor edits
    // that and the flattened `prompt` is always recomposed from the layers.
    // Without this the two would drift the moment either side was touched.
    const read = options.read || ((target) => target.prompt);
    const write = options.write || ((target, text) => { target.prompt = text; });
    renderPromptInto(editor, read(shot), materialList());
    const sync = () => {
        write(shot, readPromptText(editor));
        saveTimeline(node);
        if (hasRawMention(editor)) {
            const caret = promptCaretOffset(editor);
            renderPromptInto(editor, read(shot), materialList());
            restorePromptCaret(editor, caret);
            closeDirectorMenu();
        }
    };
    editor.addEventListener("input", sync);
    editor.addEventListener("blur", () => { sync(); closeDirectorMenu(); });
    editor.addEventListener("keydown", (event) => {
        if ((event.ctrlKey || event.metaKey)
            && ["c", "x", "v"].includes(String(event.key || "").toLowerCase())) {
            // ComfyUI's canvas listens for these shortcuts to copy a whole node.
            event.stopPropagation();
            return;
        }
        if (event.key === "Escape") closeDirectorMenu();
        if (event.key !== "Backspace" && event.key !== "Delete") return;
        const anchor = window.getSelection()?.anchorNode;
        const atomic = (anchor?.nodeType === Node.ELEMENT_NODE ? anchor : anchor?.parentElement)
            ?.closest?.(".myh3-chip");
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
        event.stopPropagation();
        event.preventDefault();
        const text = event.clipboardData?.getData("text/plain") || "";
        insertAtCaret(editor, promptFragmentFromText(text, materialList()));
        sync();
    });
    editor.addEventListener("copy", (event) => {
        // Preserve atomic media/dialogue nodes in the clipboard instead of
        // letting the canvas interpret Ctrl+C as a node-copy command.
        event.stopPropagation();
        const text = promptSelectionText(editor);
        if (!text) return;
        event.preventDefault();
        event.clipboardData?.setData("text/plain", text);
    });
    editor.addEventListener("cut", (event) => {
        event.stopPropagation();
        const text = promptSelectionText(editor);
        if (!text) return;
        event.preventDefault();
        event.clipboardData?.setData("text/plain", text);
        const selection = window.getSelection();
        const wholeDialogue = selectedDialogueBlock(editor);
        if (wholeDialogue) wholeDialogue.remove();
        else if (selection?.rangeCount) selection.getRangeAt(0).deleteContents();
        sync();
    });
    editor.__myangDialogueToolbar = createDialogueToolbar(editor, sync);
    return editor;
}

const LAYER_FIELD_CSS = "width:100%;box-sizing:border-box;background:#0b1218;"
    + "border:1px solid #334155;border-radius:4px;color:#cbd5e1;font-size:10px;"
    + "padding:3px 5px;outline:none;";
const LAYER_TITLE_CSS = "color:#93c5fd;font-size:9px;font-weight:700;"
    + "margin:6px 0 3px;";

function layerLines(value) {
    return String(value ?? "").split("\n").map((line) => line.trim())
        .filter(Boolean);
}

const KIND_TO_EN = {"图片": "image", "视频": "video", "音频": "audio"};

/** The material manifest the revision endpoints are shown, in @N numbering. */
function directorMaterialPayload(node, shot) {
    return directorMediaList(node, shot).map((entry) => ({
        kind: KIND_TO_EN[entry.kind] || "image",
        ordinal: entry.ordinal,
        label: entry.name || "",
        name: entry.file || entry.name || "",
        subject: entry.subject || "",
    }));
}

/**
 * POST to one of the Director revision routes and surface failures as notices.
 *
 * The backend already maps provider failures onto HTTP codes, so this keeps the
 * server's own message rather than inventing one: "quota exhausted" and "the
 * model returned no prompt" need different reactions from the operator.
 */
async function callDirectorRevision(route, payload) {
    const response = await api.fetchApi(route, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(payload),
    });
    let data = {};
    try {
        data = await response.json();
    } catch (error) {
        throw new Error(`HTTP ${response.status}`);
    }
    if (!response.ok || !data.success) {
        throw new Error(String(data.error || `HTTP ${response.status}`));
    }
    return data;
}

function directorLlmService(node) {
    return String(widget(node, "llm_service")?.value || "");
}

/** Let the LLM split an existing card into layers instead of typing them in. */
async function runShotLayering(node, shot, trigger) {
    const label = trigger?.textContent;
    if (trigger) { trigger.disabled = true; trigger.textContent = "分层中…"; }
    try {
        const data = await callDirectorRevision(
            "/minimax-h3-myang/director/layer-shot", {
                prompt: shot.layers ? shot.layers.visual : shot.prompt,
                seconds: Number(shot.duration_seconds) || 0,
                llm_service: directorLlmService(node),
                seed: Math.floor(Math.random() * 1e9),
                materials: directorMaterialPayload(node, shot),
            });
        const layers = normalizeSegmentLayers(data.layers);
        if (!layers) throw new Error("LLM 没有拆出任何一层");
        shot.layers = layers;
        shot.layers_open = true;
        // The prose is returned verbatim as the visual layer, so recomposing
        // here only adds the sound and dialogue lines back under it.
        shot.prompt = composeSegmentPrompt(directorGlobalLayers(node), layers);
        const notes = Array.isArray(data.notes) ? data.notes : [];
        storyboardNotice(node, notes.length
            ? `分层完成，但有 ${notes.length} 层没拿到结果：${notes.join("；")}`
            : `「${shot.brief || "本镜头"}」已分层：`
              + `主体 ${(layers.subjects_override || []).length}`
              + ` · 节拍 ${(layers.timeline || []).length}`
              + ` · 台词 ${(layers.dialogue || []).length} 句`,
            notes.length ? "error" : "success");
    } catch (error) {
        storyboardNotice(node, `分层失败：${error.message || error}`, "error");
    } finally {
        if (trigger) { trigger.disabled = false; trigger.textContent = label; }
        saveTimeline(node); renderTimeline(node);
    }
}

/** Rewrite one shot on purpose; every other card is left untouched. */
async function runShotRewrite(node, shot, index, trigger) {
    const shots = node.__myangDirectorShots || [];
    const instruction = window.prompt(
        `这个镜头要怎么改？（只改第 ${index + 1} 段，时长 ${Number(shot.duration_seconds).toFixed(1)}s 不变）`,
        "");
    if (instruction === null || !instruction.trim()) return;
    const label = trigger?.textContent;
    if (trigger) { trigger.disabled = true; trigger.textContent = "改写中…"; }
    try {
        const data = await callDirectorRevision(
            "/minimax-h3-myang/director/rewrite-shot", {
                prompt: shot.layers ? shot.layers.visual : shot.prompt,
                seconds: Number(shot.duration_seconds) || 0,
                instruction,
                llm_service: directorLlmService(node),
                seed: Math.floor(Math.random() * 1e9),
                materials: directorMaterialPayload(node, shot),
                previous_brief: shots[index - 1]?.brief || "",
                next_brief: shots[index + 1]?.brief || "",
                transition: shot.transition || "",
            });
        if (shot.layers) shot.layers.visual = String(data.prompt || "");
        shot.prompt = shot.layers
            ? composeSegmentPrompt(directorGlobalLayers(node), shot.layers)
            : String(data.prompt || "");
        if (data.brief) shot.brief = String(data.brief);
        storyboardNotice(node, data.dialogue_ok === false
            ? `第 ${index + 1} 段已改写，但台词超出本段时长，运行时会顺延到下一段`
            : `第 ${index + 1} 段已改写；其余分镜未改动`,
            data.dialogue_ok === false ? "error" : "success");
    } catch (error) {
        storyboardNotice(node, `改写失败：${error.message || error}`, "error");
    } finally {
        if (trigger) { trigger.disabled = false; trigger.textContent = label; }
        saveTimeline(node); renderTimeline(node);
    }
}

/**
 * Bind materials onto existing shots without putting a word of the script at
 * risk: the endpoint only ever returns a binding table, so there is no channel
 * through which the prose could come back rewritten.
 */
async function runMaterialRebind(node, trigger) {
    const shots = node.__myangDirectorShots || [];
    const label = trigger?.textContent;
    if (trigger) { trigger.disabled = true; trigger.textContent = "匹配中…"; }
    try {
        const data = await callDirectorRevision(
            "/minimax-h3-myang/director/rebind-materials", {
                llm_service: directorLlmService(node),
                seed: Math.floor(Math.random() * 1e9),
                materials: directorMaterialPayload(node, null),
                shots: shots.map((shot, index) => ({
                    index: index + 1,
                    brief: shot.brief || "",
                    speakers: (shot.layers?.dialogue || [])
                        .map((entry) => entry?.speaker || "").filter(Boolean),
                    subjects: (shot.layers?.subjects_override || [])
                        .map((entry) => (typeof entry === "string"
                            ? entry : entry?.name || "")).filter(Boolean).join("、"),
                })),
            });
        const globals = node.__myangDirectorGlobals || [];
        const byOrdinal = new Map();
        for (const entry of directorMediaList(node, null)) {
            byOrdinal.set(`${KIND_TO_EN[entry.kind]}:${entry.ordinal}`, entry);
        }
        let attached = 0;
        for (const binding of (data.bindings || [])) {
            const entry = byOrdinal.get(`${binding.kind}:${binding.ordinal}`);
            const asset = globals.find((item) => item.id === entry?.assetId);
            if (!asset) continue;
            for (const number of binding.shots || []) {
                const shot = shots[number - 1];
                if (!shot) continue;
                if ((shot.assets || []).some((item) =>
                    item.file?.name === asset.file?.name
                    && item.kind === asset.kind)) continue;
                shot.assets = [...(shot.assets || []), {
                    ...asset, file: {...asset.file},
                    id: `${asset.kind}_${Date.now().toString(36)}_${number}`,
                }];
                attached += 1;
            }
        }
        const unbound = (data.unbound || []).length;
        storyboardNotice(node, attached
            ? `已按主体匹配挂上 ${attached} 处素材，剧本未改动`
              + (unbound ? `；${unbound} 个素材没有找到归属` : "")
            : `没有需要新挂的素材${unbound ? `；${unbound} 个素材没有找到归属` : ""}`,
            "success");
    } catch (error) {
        storyboardNotice(node, `素材匹配失败：${error.message || error}`, "error");
    } finally {
        if (trigger) { trigger.disabled = false; trigger.textContent = label; }
        saveTimeline(node); renderTimeline(node);
    }
}

/** Seed a layer set from a card that only ever had a flat prompt. */
function seedLayers(shot) {
    return {
        visual: String(shot.prompt || ""),
        subjects_override: [],
        timeline: [],
        sound: {ambient: "", bgm: "", sfx: []},
        dialogue: [],
    };
}

/**
 * Per-card layer editor: 主体 / 时间轴 / 声音 / 台词, each reviewable on its own.
 *
 * The point of showing dialogue as rows rather than as text inside the prompt is
 * the budget readout: `dialogue_audit` bills every line against the shot's
 * seconds at run time, and a line that cannot be spoken in the window either
 * gets deferred to the next segment or forces the shot to talk over its own cut.
 * Seeing 「48/32 字」 while typing is the difference between noticing that here
 * and noticing it in the finished video.
 */
function renderShotLayers(node, shot, promptEditor) {
    const wrap = document.createElement("details");
    wrap.dataset.myangDirectorSection = `shot-layers:${shot.id}`;
    wrap.open = Boolean(shot.layers_open);
    wrap.style.cssText = "margin-top:5px;border-top:1px solid #1f2937;padding-top:4px;";
    wrap.addEventListener("toggle", () => {
        shot.layers_open = wrap.open;
        saveTimeline(node);
    });

    const layers = shot.layers && typeof shot.layers === "object" ? shot.layers : null;
    const spent = (layers?.dialogue || []).reduce(
        (total, entry) => total + dialogueSeconds(entry), 0);
    const window_ = Math.max(0.1, Number(shot.duration_seconds) || 0);

    const summary = document.createElement("summary");
    summary.style.cssText = "cursor:pointer;color:#7dd3fc;font-size:9px;"
        + "font-weight:700;outline:none;";
    summary.textContent = layers
        ? `分层编辑 · 台词 ${(layers.dialogue || []).length} 句 ${spent.toFixed(1)}/${window_.toFixed(1)}s`
        : "分层编辑（未启用）";
    wrap.appendChild(summary);

    if (!layers) {
        const hint = document.createElement("div");
        hint.style.cssText = "font-size:9px;color:#778493;margin:4px 0;";
        hint.textContent = "把这张卡拆成主体 / 时间轴 / 声音 / 台词四层后，"
            + "台词会按本段时长显示字数预算，超出的会在运行时顺延到下一段。";
        const row = document.createElement("div");
        row.style.cssText = "display:flex;gap:4px;align-items:center;";
        const ai = button("AI 分层", "让 LLM 把现有提示词拆成四层；正文原样保留为画面层");
        ai.style.background = "#3f6212";
        ai.style.color = "#ecfccb";
        ai.onclick = () => runShotLayering(node, shot, ai);
        const manual = button("手动分层", "把当前提示词作为画面层，其余层自己填");
        manual.onclick = () => {
            shot.layers = seedLayers(shot);
            shot.layers_open = true;
            saveTimeline(node); renderTimeline(node);
        };
        row.append(ai, manual);
        wrap.append(hint, row);
        return wrap;
    }

    const recompose = () => {
        shot.prompt = composeSegmentPrompt(directorGlobalLayers(node), layers);
        saveTimeline(node);
    };
    const title = (text) => {
        const element = document.createElement("div");
        element.textContent = text;
        element.style.cssText = LAYER_TITLE_CSS;
        return element;
    };
    const area = (value, placeholder, commit) => {
        const field = document.createElement("textarea");
        field.value = value;
        field.placeholder = placeholder;
        field.rows = 2;
        field.style.cssText = LAYER_FIELD_CSS + "resize:vertical;line-height:1.5;";
        field.oninput = () => { commit(field.value); recompose(); };
        return field;
    };

    wrap.append(title("主体（一行一个：名字：外观 @图片1）"), area(
        (layers.subjects_override || []).map((entry) => typeof entry === "string"
            ? entry
            : [entry?.name, entry?.appearance, entry?.wardrobe].filter(Boolean).join("：")
        ).join("\n"),
        "留空则沿用全局主体定义",
        (value) => { layers.subjects_override = layerLines(value); }));

    wrap.append(title("时间轴（一行一个节拍：0-3s 缓慢推近，她抬头）"), area(
        (layers.timeline || []).map((entry) => typeof entry === "string"
            ? entry
            : [entry?.beat || entry?.time_range,
               entry?.camera || entry?.camera_movement,
               entry?.action || entry?.content].filter(Boolean).join(" ")
        ).join("\n"),
        "留空则完全依赖上方画面正文",
        (value) => { layers.timeline = layerLines(value); }));

    const sound = layers.sound && typeof layers.sound === "object"
        ? layers.sound : {};
    wrap.appendChild(title("声音"));
    const soundGrid = document.createElement("div");
    soundGrid.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:4px;";
    for (const [key, label] of [["ambient", "环境音"], ["bgm", "背景音乐"]]) {
        const field = document.createElement("input");
        field.type = "text";
        field.value = String(sound[key] || "");
        field.placeholder = label;
        field.style.cssText = LAYER_FIELD_CSS;
        field.oninput = () => {
            layers.sound = {...sound, [key]: field.value};
            Object.assign(sound, layers.sound);
            recompose();
        };
        soundGrid.appendChild(field);
    }
    wrap.appendChild(soundGrid);
    const sfx = document.createElement("input");
    sfx.type = "text";
    sfx.value = (Array.isArray(sound.sfx) ? sound.sfx : []).join("、");
    sfx.placeholder = "音效，用、或，分隔";
    sfx.style.cssText = LAYER_FIELD_CSS + "margin-top:4px;";
    sfx.oninput = () => {
        layers.sound = {...sound, sfx: sfx.value.split(/[、,，]/)
            .map((item) => item.trim()).filter(Boolean)};
        Object.assign(sound, layers.sound);
        recompose();
    };
    wrap.appendChild(sfx);

    wrap.appendChild(renderShotDialogueLayer(node, shot, layers, recompose));
    return wrap;
}

/** Global layers travel with the LLM plan; a manual card has none of its own. */
function directorGlobalLayers(node) {
    const plan = node.__myangDirectorPlan;
    const layers = plan?.layers;
    return layers && typeof layers === "object" ? layers : {};
}

/**
 * Dialogue rows with a live 字数 budget, because that is the reviewable part.
 *
 * The ceiling shown per row is `budgetUnits(shot seconds, tone)` -- the same
 * number `dialogue_audit` compresses against and `enforce_dialogue_budget`
 * defers against, so the card cannot promise a line will fit and then have the
 * backend move it.
 */
function renderShotDialogueLayer(node, shot, layers, recompose) {
    const box = document.createElement("div");
    const window_ = Math.max(0.1, Number(shot.duration_seconds) || 0);
    const rows = Array.isArray(layers.dialogue) ? layers.dialogue : [];

    const head = document.createElement("div");
    head.style.cssText = LAYER_TITLE_CSS + "display:flex;justify-content:space-between;";
    const spent = rows.reduce((total, entry) => total + dialogueSeconds(entry), 0);
    const over = spent > window_ + 0.05;
    const label = document.createElement("span");
    label.textContent = "台词";
    const total = document.createElement("span");
    total.textContent = `${spent.toFixed(2)} / ${window_.toFixed(2)}s`;
    total.style.color = over ? "#fb7185" : "#86efac";
    total.title = over
        ? "超出本段时长，运行时会把放不下的台词顺延到下一段"
        : "全部台词都能在本段说完";
    head.append(label, total);
    box.appendChild(head);

    rows.forEach((entry, index) => {
        const row = document.createElement("div");
        row.style.cssText = "display:grid;grid-template-columns:64px 66px 1fr 58px 20px;"
            + "gap:4px;align-items:center;margin-bottom:3px;";
        const speaker = document.createElement("input");
        speaker.type = "text";
        speaker.value = String(entry?.speaker || "");
        speaker.placeholder = "说话人";
        speaker.style.cssText = LAYER_FIELD_CSS;
        speaker.oninput = () => { rows[index] = {...rows[index], speaker: speaker.value}; layers.dialogue = rows; recompose(); };

        const tone = document.createElement("select");
        tone.style.cssText = LAYER_FIELD_CSS;
        for (const name of Object.keys(SPEECH_RATES)) {
            const option = document.createElement("option");
            option.value = name;
            const [low, high] = SPEECH_RATES[name];
            option.textContent = `${name} ${low}-${high}字/s`;
            tone.appendChild(option);
        }
        tone.value = SPEECH_RATES[entry?.tone] ? entry.tone : "calm";

        const text = document.createElement("input");
        text.type = "text";
        text.value = String(entry?.text || "");
        text.placeholder = "这句台词说什么";
        text.style.cssText = LAYER_FIELD_CSS;

        const count = document.createElement("span");
        count.style.cssText = "font-size:9px;text-align:right;";
        const refresh = () => {
            const used = speechUnits(text.value);
            const cap = budgetUnits(window_, tone.value);
            count.textContent = `${used}/${cap} 字`;
            count.style.color = used > cap ? "#fb7185" : "#778493";
            count.title = used > cap
                ? `这句最快也要 ${(used / SPEECH_RATES[tone.value][1]).toFixed(2)} 秒，`
                  + `本段只有 ${window_.toFixed(2)} 秒；请压到 ${cap} 字以内`
                : `本段 ${window_.toFixed(2)} 秒、${tone.value} 语速下最多 ${cap} 字`;
        };
        refresh();
        const commit = () => {
            rows[index] = {...rows[index], tone: tone.value, text: text.value};
            layers.dialogue = rows;
            refresh();
            recompose();
        };
        tone.onchange = commit;
        text.oninput = commit;

        const drop = button("×", "删除这句台词");
        drop.style.color = "#ff6188";
        drop.onclick = () => {
            rows.splice(index, 1);
            layers.dialogue = rows;
            saveTimeline(node); renderTimeline(node);
        };
        row.append(speaker, tone, text, count, drop);
        box.appendChild(row);
    });

    const add = button("+ 添加台词", "在本段末尾追加一句台词");
    add.onclick = () => {
        rows.push({speaker: "", tone: "calm", text: ""});
        layers.dialogue = rows;
        shot.layers_open = true;
        saveTimeline(node); renderTimeline(node);
    };
    box.appendChild(add);
    return box;
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
        refining: false, audioRefine: false, correcting: false,
        activity: "", error: "",
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
    if (elements.stop) elements.stop.style.display = state.status === "running" ? "inline-block" : "none";

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
        let text = state.activity
            ? state.activity
            : `第 ${state.seg}/${state.total} 段 · ${phase.label}`;
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
        const params = new URLSearchParams({
            filename: String(state.previewFile),
            subfolder: "",
            type: "temp",
            _t: String(state.previewTs || Date.now()),
        });
        const url = `/api/view?${params.toString()}`;
        // Assigning src directly blanks the element until the new frame decodes.
        // Warm it in a detached Image first and swap only once it is ready, so
        // the visible frame goes straight from the old one to the new one. The
        // token drops responses that arrive out of order.
        const token = Number(node.__myangPreviewToken || 0) + 1;
        node.__myangPreviewToken = token;
        const warm = new Image();
        const swap = () => {
            if (node.__myangPreviewToken !== token) return;
            const target = node.__myangDirectorProgressEls?.preview;
            if (!target?.isConnected) return;
            target.hidden = false;
            target.src = url;
        };
        warm.onload = swap;
        warm.onerror = swap;
        warm.src = url;
        if (warm.complete) swap();
    } else {
        node.__myangPreviewToken = Number(node.__myangPreviewToken || 0) + 1;
        elements.preview.hidden = true;
        elements.preview.removeAttribute("src");
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
    // The Director root owns scrolling; keep live status pinned within it.
    panel.style.cssText += ";margin-bottom:8px;background:#101923;border-color:#33465d;position:sticky;top:0;z-index:40;box-shadow:0 3px 10px rgba(4,10,18,.5);backdrop-filter:blur(4px);";
    const heading = document.createElement("div");
    heading.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:8px;color:#bfdbfe;font-size:10px;font-weight:700;";
    const title = document.createElement("span");
    title.textContent = "生成进度";
    const live = document.createElement("span");
    live.textContent = "实时";
    live.style.cssText = "color:#60a5fa;font-size:9px;font-weight:500;";
    const actions = document.createElement("span");
    actions.style.cssText = "display:flex;align-items:center;gap:7px;flex:0 0 auto;";
    const stop = document.createElement("button");
    stop.type = "button";
    stop.textContent = "停止并释放";
    stop.title = "中断当前生成，卸载模型并清空执行缓存";
    stop.style.cssText = "display:none;padding:3px 8px;border:1px solid #7f1d1d;border-radius:4px;background:#3f1518;color:#fecaca;font-size:9px;line-height:1.2;cursor:pointer;white-space:nowrap;";
    stop.onclick = async (event) => {
        event.preventDefault();
        event.stopPropagation();
        if (stop.disabled) return;
        stop.disabled = true;
        stop.textContent = "正在停止…";
        try {
            await api.interrupt();
            await requestDirectorMemoryRelease();
        } catch (error) {
            console.warn("[Myang Director] 停止并释放失败", error);
        } finally {
            stop.disabled = false;
            stop.textContent = "停止并释放";
        }
    };
    actions.append(live, stop);
    heading.append(title, actions);
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
    node.__myangDirectorProgressEls = {panel, fill, text, prompt, preview, stop};
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
    node.__myangDirectorPendingVideoOutput = null;
    const video = node.__myangDirectorVideoElement;
    try { video?.pause?.(); } catch (_error) { /* detached media can already be closed */ }
    // Preserve the native element while the next run is executing.  Newer
    // ComfyUI builds often reuse this exact <video> instead of creating a new
    // one; deleting it here left the Director with only its sampling still.
    if (video && node.__myangDirectorVideoPanel) {
        node.__myangDirectorVideoPanel.hidden = false;
    }
}

function videoResultFromOutput(output) {
    if (!output || typeof output !== "object") return null;
    for (const key of ["videos", "images", "gifs", "files"]) {
        const values = Array.isArray(output[key]) ? output[key] : [];
        for (const value of values) {
            const filename = String(value?.filename || value?.name || "");
            if (!/\.(mp4|webm|mov|mkv|avi|m4v)$/i.test(filename)) continue;
            return {
                filename,
                subfolder: String(value?.subfolder || ""),
                type: String(value?.type || "output"),
            };
        }
    }
    for (const envelope of ["output", "ui"]) {
        const nested = videoResultFromOutput(output[envelope]);
        if (nested) return nested;
    }
    return null;
}

function directorForLlmDiagnostic(detail) {
    const owner = detail?.owner_id ?? detail?.node_id;
    const direct = directorByOwner(owner);
    if (direct) return direct;
    const sourceId = String(detail?.node_id || "");
    if (sourceId) {
        const nested = (app.graph?._nodes || []).find((node) => {
            if (node.type !== NODE) return false;
            const id = String(node.id);
            return sourceId.startsWith(`${id}:`)
                || sourceId.startsWith(`${id}.`)
                || sourceId.startsWith(`${id}/`);
        });
        if (nested) return nested;
    }
    // Skill learning and standalone Agent calls share the same websocket.
    // Only use this fallback when exactly one Director is already known to be
    // preparing, so unrelated traffic can never overwrite another card.
    const preparing = (app.graph?._nodes || []).filter((node) => {
        if (node.type !== NODE) return false;
        const state = progressState(node);
        return state.status === "running" && state.phase === "prepare";
    });
    return preparing.length === 1 ? preparing[0] : null;
}

function llmDiagnosticActivity(detail) {
    const service = [detail?.service, detail?.model].filter(Boolean).join("/");
    const route = detail?.route ? ` · ${detail.route}` : "";
    const elapsed = detail?.elapsed != null ? ` · ${Number(detail.elapsed).toFixed(1)}s` : "";
    const counts = detail?.content_chars != null || detail?.reasoning_chars != null
        ? ` · 正文 ${Number(detail.content_chars || 0)}字 / 思考 ${Number(detail.reasoning_chars || 0)}字`
        : "";
    const phase = String(detail?.phase || "");
    if (phase === "queued") return `LLM 已排队 · ${service || "等待选择服务"}`;
    if (phase === "route_attempt") {
        const round = Number(detail?.route_round || 1);
        const rounds = Number(detail?.route_rounds || 1);
        const routeIndex = Number(detail?.route_index || 1);
        const routeCount = Number(detail?.route_count || 1);
        const timeout = Number(detail?.first_output_timeout || detail?.timeout || 0);
        const kind = rounds > 1 ? (round === 1 ? "15秒快探" : "30秒复查") : "单线路等待";
        return `LLM ${kind} · 第${round}/${rounds}轮 · 线路${routeIndex}/${routeCount}${route}${timeout ? ` · 首内容≤${timeout}秒` : ""}${elapsed}`;
    }
    if (phase === "route_retry_round") return String(detail?.message || "所有线路快探未命中，进入30秒复查轮");
    if (phase === "connecting") return `LLM 正在连接 ${service}${route}${elapsed}`;
    if (["connected", "waiting_first_byte"].includes(phase)) {
        return `LLM 已连接，等待首个正文或思考内容${route}${elapsed}`;
    }
    if (phase === "waiting_generation") return `LLM 已连接，等待正文或思考内容${route}${elapsed}`;
    if (phase === "reasoning") return `LLM 正在思考${route}${elapsed}${counts} · 已锁定本线路，等待完整结束`;
    if (phase === "streaming") return `LLM 流式接收中${route}${elapsed}${counts} · 已锁定本线路，等待完整结束`;
    if (phase === "stream_complete") return `LLM 数据流已结束 · 正在整理完整回复${counts}`;
    if (phase === "stream_ignored") return `服务端返回普通整包 · 正在解析${elapsed}`;
    if (phase === "stream_fallback") return `该线路不支持流式，已改用普通整包；无法观察思考进度${route}`;
    if (phase === "route_switch") return String(detail?.message || `正在切换探测线路${route}`);
    if (phase === "cooldown_wait") return String(detail?.message || "线路限流，等待冷却后重试");
    if (phase === "routes_unavailable") return String(detail?.message || "所有 LLM 线路当前都不可用");
    if (phase === "route_error") {
        if (detail?.will_retry_round) return String(detail?.message || "本线路15秒内未开始生成，稍后30秒复查");
        const next = detail?.will_failover ? "，正在切换下一线路" : "";
        return `LLM 线路失败（${detail?.error_reason || "未知原因"}）${next}`;
    }
    if (phase === "cancelled") return "LLM 请求已停止 · 正在释放连接";
    if (phase === "failed") {
        return `LLM 请求失败（${detail?.error_reason || "未知原因"}） · ${detail?.message || "未取得可用正文"}`;
    }
    if (phase === "done") {
        const finish = detail?.finish_reason ? ` · finish=${detail.finish_reason}` : "";
        return `LLM 回复完成${finish}${counts} · 正在解析分镜`;
    }
    return "";
}

function nativeVideoElement(node) {
    const nativeWidget = node.widgets?.find((item) =>
        item.name === NATIVE_VIDEO_WIDGET
        || String(item.type || "").toLowerCase().includes("video"));
    for (const candidate of [
        node.videoContainer, nativeWidget?.element, nativeWidget?.el,
        nativeWidget?.inputEl, nativeWidget?.container,
        nativeWidget?.widget?.element,
    ]) {
        if (!candidate) continue;
        if (String(candidate.tagName || "").toLowerCase() === "video") return candidate;
        const video = candidate.querySelector?.("video");
        if (video) return video;
    }
    return node.__myangDirectorVideoElement || null;
}

function outputVideoUrl(result) {
    if (!result) return "";
    const params = new URLSearchParams({
        filename: result.filename,
        subfolder: result.subfolder || "",
        type: result.type || "output",
    });
    return `/api/view?${params.toString()}`;
}

function mountNativeVideoPreview(node, output = null) {
    const host = node.__myangDirectorVideoHost;
    const panel = node.__myangDirectorVideoPanel;
    if (!host || !panel) return false;

    const payload = videoResultFromOutput(
        output || node.__myangDirectorPendingVideoOutput);
    let incoming = nativeVideoElement(node);
    // Fallback for frontend builds that expose PreviewVideo metadata without
    // attaching the native player to node.videoContainer.
    if (!incoming && payload) incoming = document.createElement("video");
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

    if (payload) {
        const source = outputVideoUrl(payload);
        if (source && video.dataset.myangResultSource !== source) {
            video.dataset.myangResultSource = source;
            video.src = source;
            video.load?.();
        }
        node.__myangDirectorPendingVideoOutput = output;
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

    return true;
}

function outputWithoutNativeMediaPreview(output) {
    if (!output || typeof output !== "object") return output;
    // The original result is mounted in the Director's native <video>. Passing
    // media to the base handler first creates a full-node still or a separate
    // widget, causing both the flash and the unexpected node resize.
    const filtered = {...output};
    for (const key of ["images", "gifs", "videos", "files"]) {
        if (Array.isArray(output[key])) filtered[key] = [];
    }
    for (const envelope of ["output", "ui"]) {
        if (output[envelope] && typeof output[envelope] === "object") {
            filtered[envelope] = outputWithoutNativeMediaPreview(output[envelope]);
        }
    }
    return filtered;
}

function installCanvasPreviewGuard(node) {
    if (node.__myangDirectorCanvasPreviewGuard) return;
    try {
        const descriptor = Object.getOwnPropertyDescriptor(node, "imgs");
        Object.defineProperty(node, "imgs", {
            configurable: true,
            enumerable: descriptor?.enumerable ?? true,
            get: () => null,
            set: () => {},
        });
        node.__myangDirectorCanvasPreviewGuard = true;
    } catch (error) {
        console.warn("[Myang Director] 无法安装画布预览隔离", error);
    }
}

function hideNativeVideoWidget(node) {
    const nativeWidget = node.widgets?.find((item) => item.name === NATIVE_VIDEO_WIDGET);
    if (nativeWidget && !nativeWidget.__myangDirectorHidden) hideWidget(nativeWidget);
}

function clearNativeStillPreview(node) {
    clearNativeStillPreviewState(node);
    app.graph?.setDirtyCanvas?.(true, true);
    app.canvas?.setDirty?.(true, true);
}

function clearNativeStillPreviewState(node) {
    node.imgs = null;
    node.imageIndex = null;
    node.imageOffset = null;
}

function clearNativeStillPreviewSoon(node) {
    clearNativeStillPreview(node);
    // The base ComfyUI node may apply its UI result on the next tick. Clear
    // again after those result envelopes have been consumed, but leave the
    // native video element mounted in the Director card.
    for (const delay of [0, 80, 240, 600]) {
        setTimeout(() => clearNativeStillPreview(node), delay);
    }
}

function scheduleNativeVideoMount(node, output = null) {
    if (output) node.__myangDirectorPendingVideoOutput = output;
    const token = Number(node.__myangDirectorVideoMountToken || 0) + 1;
    node.__myangDirectorVideoMountToken = token;
    let attempts = 24;
    const check = () => {
        if (node.__myangDirectorVideoMountToken !== token) return;
        if (mountNativeVideoPreview(node, output) || --attempts <= 0) return;
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
    if (stage === "preparing") {
        state.phase = "prepare";
        state.step = 0;
        state.stepMax = 0;
        state.activity = String(detail?.activity || "导演台准备中");
    } else if (stage === "sampling") {
        const passLabel = String(detail?.pass_label || "sample1");
        state.phase = passLabel === "audio_refine" ? "audio_refine"
            : passLabel.startsWith("sample2") ? "sample2" : "sample1";
        state.step = Number(detail?.step || 0);
        state.stepMax = Number(detail?.step_total || 0);
        state.activity = "";
    } else {
        state.step = 0;
        state.stepMax = 0;
        if (stage === "sampled") state.phase = state.correcting
            ? "drift" : state.refining ? "refine_prep" : "finalizing";
        else if (stage === "drifted") state.phase = state.refining ? "refine_prep" : "finalizing";
        else if (stage === "refine_start") state.phase = "refine_prep";
        else if (stage === "refined") state.phase = "finalizing";
        else if (stage === "assembling") {
            state.phase = "assembling";
            state.activity = `全部 ${state.total} 段已完成 · 正在合并画面与音频`;
        } else if (stage === "assembled") {
            state.phase = "exporting";
            state.activity = "分段合并完成 · 正在编码并写出最终视频";
        }
        else if (stage === "done") {
            if (state.seg < state.total) {
                state.seg += 1;
                state.phase = "sample1";
                state.activity = "";
            } else {
                // The final segment is ready, but H3SegmentCollector and the
                // downstream video encoder still have real work to do. Keep
                // the last compact preview visible until their events arrive.
                state.phase = "assembling";
                state.activity = `全部 ${state.total} 段已完成 · 等待合并分段`;
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

function markDirectorPreparing(node, activity) {
    if (!node || node.type !== NODE) return;
    const state = progressState(node);
    // Expanded/subgraph execution can announce the parent Director more than
    // once. Only the first announcement may initialise a run; later ones must
    // never rewind sample/refine/merge progress to the preparation caption.
    if (state.status === "running") return;
    clearOutputVideo(node);
    state.status = "running";
    state.runId = "";
    state.phase = "prepare";
    state.seg = 1;
    state.total = Math.max(1, Number(
        node.__myangDirectorPlan?.segment_count
        || node.__myangDirectorShots?.filter?.((shot) => shot.enabled !== false)?.length
        || 1));
    state.step = 0;
    state.stepMax = 0;
    state.previewFile = "";
    state.previewTs = 0;
    state.activity = String(activity || "已进入导演台，正在准备执行");
    state.error = "";
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

function droppedFileKind(file) {
    const mime = String(file?.type || "").toLowerCase();
    if (mime.startsWith("image/")) return "image";
    if (mime.startsWith("video/")) return "video";
    if (mime.startsWith("audio/")) return "audio";
    const extension = String(file?.name || "").split(".").pop()?.toLowerCase();
    if (["png", "jpg", "jpeg", "webp", "bmp", "gif", "tif", "tiff"].includes(extension)) return "image";
    if (["mp4", "mov", "mkv", "webm", "avi", "m4v", "wmv"].includes(extension)) return "video";
    if (["wav", "mp3", "flac", "m4a", "aac", "ogg", "opus"].includes(extension)) return "audio";
    return "";
}

function renderShotAssets(node, shot, prompt, options = {}) {
    const allowedKinds = new Set(options.allowedKinds || ["image", "video", "audio"]);
    const limits = options.limits || {};
    const showAssetMode = options.showAssetMode !== false;
    const section = document.createElement("div");
    section.style.cssText = "margin-top:7px;border:1px solid #2f3e50;background:#111923;border-radius:6px;padding:7px;";
    const head = document.createElement("div");
    head.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:8px;margin-bottom:6px;";
    const resolvedForQuota = options.resolve
        ? options.resolve() : directorMediaList(node, shot);
    const quota = materialQuota(resolvedForQuota, allowedKinds, limits);
    const title = document.createElement("div");
    title.textContent = options.title || "镜头素材";
    title.style.cssText = "min-width:0;font-size:11px;font-weight:700;color:#7dd3fc;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;";
    const quotaBadge = document.createElement("span");
    quotaBadge.textContent = quota.text;
    quotaBadge.title = "当前模式实际参与编号和生成的素材额度";
    quotaBadge.style.cssText = `flex:0 0 auto;font-size:9px;color:${quota.total > MEDIA_TOTAL_LIMIT ? "#fb7185" : "#9fb4c9"};white-space:nowrap;`;
    const heading = document.createElement("div");
    heading.style.cssText = "display:flex;align-items:center;gap:7px;min-width:0;overflow:hidden;";
    heading.append(title, quotaBadge);
    head.appendChild(heading);
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
    const addFiles = async (files) => {
        const accepted = [];
        let total = materialQuota(
            options.resolve ? options.resolve() : directorMediaList(node, shot),
            allowedKinds, limits).total;
        const counts = Object.fromEntries([...allowedKinds].map((kind) => [
            kind, shot.assets.filter((asset) => asset.kind === kind).length,
        ]));
        for (const file of Array.from(files || [])) {
            const kind = droppedFileKind(file);
            if (!allowedKinds.has(kind)) continue;
            if (total >= MEDIA_TOTAL_LIMIT) continue;
            const limit = Number(limits[kind] ?? MEDIA_META[kind].limit);
            if ((counts[kind] || 0) >= limit) continue;
            counts[kind] = (counts[kind] || 0) + 1;
            total += 1;
            accepted.push({file, kind});
        }
        if (!accepted.length) {
            status.textContent = "没有可添加的素材（格式不支持或已达上限）";
            return;
        }
        status.textContent = `正在上传 ${accepted.length} 个素材…`;
        try {
            for (const {file, kind} of accepted) {
                const saved = await uploadShotFile(shot, file);
                const asset = normalizeAsset({kind, label: file.name, file: saved}, shot.assets.length);
                if (kind === "video" && options.videoRole) asset.role = options.videoRole;
                shot.assets.push(asset);
            }
            saveTimeline(node);
            renderTimeline(node);
        } catch (error) {
            console.error("[Myang Director] 拖入素材上传失败", error);
            status.textContent = error?.message || "上传失败";
            status.style.color = "#fb7185";
        }
    };
    const addLibraryAsset = async (asset) => {
        if (!asset || !allowedKinds.has(asset.kind)) return;
        const effective = options.resolve
            ? options.resolve() : directorMediaList(node, shot);
        if (materialQuota(effective, allowedKinds, limits).total >= MEDIA_TOTAL_LIMIT) {
            status.textContent = `当前模式素材总计已达到 ${MEDIA_TOTAL_LIMIT} 个上限`;
            return;
        }
        const limit = Number(limits[asset.kind] ?? MEDIA_META[asset.kind].limit);
        if (shot.assets.filter((entry) => entry.kind === asset.kind).length >= limit) {
            status.textContent = `${MEDIA_META[asset.kind].label}已达到上限`;
            return;
        }
        const normalized = normalizeAsset(asset, shot.assets.length);
        if (normalized.kind === "video" && options.videoRole) normalized.role = options.videoRole;
        shot.assets.push(normalized);
        saveTimeline(node);
        renderTimeline(node);
    };
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
        add.onclick = () => openMaterialSourceDialog(node, {
            kind,
            local: () => picker.click(),
            library: () => openMaterialLibraryChooser(node, {
                kind, onSelect: addLibraryAsset,
            }),
        });
        picker.onchange = async () => {
            const current = shot.assets.filter((asset) => asset.kind === kind).length;
            const files = Array.from(picker.files || []).slice(0, Math.max(0, limit - current));
            if (!files.length) {
                status.textContent = current >= limit ? `${meta.label}已达到上限` : "未选择文件";
                return;
            }
            add.disabled = true;
            try { await addFiles(files); }
            finally { add.disabled = false; picker.value = ""; }
        };
        tools.append(add, picker);
    }
    tools.appendChild(status);
    section.appendChild(tools);

    const dropZone = document.createElement("div");
    dropZone.tabIndex = 0;
    dropZone.textContent = "拖入图片 / 视频 / 音频到这里 · 下方卡片可拖动换序；提示词序号保持不变，由新顺序决定对应素材";
    dropZone.setAttribute("aria-label", "拖入素材上传区域");
    dropZone.style.cssText = "display:flex;align-items:center;justify-content:center;min-height:44px;margin-bottom:6px;padding:7px;border:1px dashed #45627f;border-radius:5px;background:#0b141e;color:#8fb3d8;font-size:9px;text-align:center;transition:border-color .18s ease,background .18s ease;";
    const setDropping = (active) => {
        dropZone.style.borderColor = active ? "#60a5fa" : "#45627f";
        dropZone.style.background = active ? "#10243a" : "#0b141e";
    };
    dropZone.ondragenter = (event) => { event.preventDefault(); setDropping(true); };
    dropZone.ondragover = (event) => { event.preventDefault(); event.dataTransfer.dropEffect = "copy"; setDropping(true); };
    dropZone.ondragleave = () => setDropping(false);
    dropZone.ondrop = (event) => {
        event.preventDefault();
        setDropping(false);
        addFiles(event.dataTransfer?.files || []);
    };
    section.appendChild(dropZone);

    if (!shot.assets.length) {
        const empty = document.createElement("div");
        empty.textContent = options.emptyText || "直接添加本镜头专用素材；文件保存在 ComfyUI/input，工作流只记录引用。";
        empty.style.cssText = "font-size:9px;color:#64748b;padding:4px 1px;";
        section.appendChild(empty);
        return section;
    }

    const list = document.createElement("div");
    list.style.cssText = `display:grid;grid-template-columns:repeat(auto-fill,minmax(${options.referenceWeights ? 300 : 180}px,1fr));gap:6px;`;
    const resolvedMaterials = resolvedForQuota;
    const typeOrdinals = {image: 0, video: 0, audio: 0};
    shot.assets.forEach((asset, assetIndex) => {
        typeOrdinals[asset.kind] += 1;
        const ordinal = typeOrdinals[asset.kind];
        const meta = MEDIA_META[asset.kind];
        const resolved = resolvedMaterials.find((entry) => entry.assetId === asset.id);
        const mention = resolved?.token || `@${meta.token}${ordinal}`;
        const item = document.createElement("div");
        item.style.cssText = "min-width:0;border:1px solid #334155;background:#0d141d;border-radius:5px;padding:5px;";
        item.draggable = true;
        item.dataset.assetId = asset.id;
        item.setAttribute("aria-label", `${asset.label || asset.file.name}，可拖动调整素材顺序`);
        item.ondragstart = (event) => {
            node.__myangDraggedAssetId = asset.id;
            event.dataTransfer.effectAllowed = "move";
            event.dataTransfer.setData("text/x-myang-asset-id", asset.id);
            item.style.opacity = ".5";
        };
        item.ondragend = () => { item.style.opacity = "1"; node.__myangDraggedAssetId = ""; };
        item.ondragover = (event) => {
            const moving = node.__myangDraggedAssetId || event.dataTransfer?.getData("text/x-myang-asset-id");
            const movingAsset = shot.assets.find((candidate) => candidate.id === moving);
            if (!moving || moving === asset.id || !movingAsset || movingAsset.kind !== asset.kind) return;
            event.preventDefault();
            event.dataTransfer.dropEffect = "move";
            item.style.borderColor = "#60a5fa";
        };
        item.ondragleave = () => { item.style.borderColor = "#334155"; };
        item.ondrop = (event) => {
            const moving = node.__myangDraggedAssetId || event.dataTransfer?.getData("text/x-myang-asset-id");
            const movingAsset = shot.assets.find((candidate) => candidate.id === moving);
            if (!moving || moving === asset.id || !movingAsset || movingAsset.kind !== asset.kind) return;
            event.preventDefault();
            const from = shot.assets.findIndex((candidate) => candidate.id === moving);
            const to = shot.assets.findIndex((candidate) => candidate.id === asset.id);
            if (from < 0 || to < 0) return;
            const [moved] = shot.assets.splice(from, 1);
            shot.assets.splice(to, 0, moved);
            saveTimeline(node);
            renderTimeline(node);
        };
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
        } else {
            preview.controls = true;
            preview.preload = "metadata";
            if (asset.kind === "video") preview.muted = true;
        }
        if (asset.kind === "image" || asset.kind === "video") {
            // The card is a thumbnail viewport, not an aspect-ratio override.
            // `contain` preserves portrait, landscape and square material; the
            // click opens a separate viewer at the media's intrinsic ratio.
            preview.controls = false;
            preview.style.cssText = "display:block;width:100%;height:100%;object-fit:contain;background:#05090e;pointer-events:none;";
            const previewButton = document.createElement("button");
            previewButton.type = "button";
            previewButton.setAttribute("aria-label", `预览素材：${asset.label || asset.file.name}`);
            previewButton.title = "点击按原始比例预览";
            previewButton.style.cssText = "position:relative;display:block;width:100%;height:104px;padding:0;overflow:hidden;border:1px solid #263548;border-radius:4px;background:#05090e;cursor:pointer;transition:border-color .18s ease,background .18s ease;";
            previewButton.onmouseenter = () => {
                previewButton.style.borderColor = "#60a5fa";
                previewButton.style.background = "#0b1623";
            };
            previewButton.onmouseleave = () => {
                previewButton.style.borderColor = "#263548";
                previewButton.style.background = "#05090e";
            };
            previewButton.onclick = () => openAssetPreview(asset, previewButton);
            const badge = document.createElement("span");
            badge.textContent = "预览";
            badge.style.cssText = "position:absolute;right:5px;bottom:5px;padding:2px 6px;border-radius:4px;background:rgba(6,12,20,.82);border:1px solid #52647b;color:#dbeafe;font-size:9px;line-height:1.4;pointer-events:none;";
            previewButton.append(preview, badge);
            item.appendChild(previewButton);
        } else {
            preview.style.cssText = "display:block;width:100%;height:30px;margin:21px 0;";
            item.appendChild(preview);
        }
        const row = document.createElement("div");
        row.style.cssText = "display:flex;gap:4px;align-items:center;margin-top:4px;";
        const ref = button(mention, prompt
            ? `插入 ${mention} 到提示词光标处`
            : `${mention}（LLM 会自动分配，也可手动写进剧本）`);
        ref.style.color = "#7dd3fc";
        if (prompt) ref.onclick = () => insertAtCursor(prompt, mention);
        else ref.disabled = true;
        const label = document.createElement("input");
        label.dataset.myangControl = `shot:${shot.id}:asset:${asset.id}:label`;
        label.value = asset.label && asset.label !== asset.file.name ? asset.label : "";
        label.placeholder = "主体名（可选）";
        label.title = `绑定主体名；留空时标签只显示 ${mention}\n文件：${asset.file.name}`;
        label.style.cssText = "min-width:0;width:100%;box-sizing:border-box;background:#0b1118;color:#cbd5e1;border:1px solid #293548;border-radius:3px;padding:4px 5px;font-size:9px;";
        label.oninput = () => {
            asset.label = label.value.trim() || asset.file.name;
            saveTimeline(node);
        };
        const remove = button("×", "移除素材引用（不会删除 input 中的文件）");
        remove.style.flex = "0 0 auto";
        remove.style.color = "#fb7185";
        remove.onclick = () => {
            shot.assets.splice(assetIndex, 1);
            saveTimeline(node);
            renderTimeline(node);
        };
        const archive = button("入库", "加入角色、场景、音色、音乐或视频素材库；也可在卡片上点右键");
        archive.style.flex = "0 0 auto";
        archive.style.color = "#86efac";
        archive.onclick = () => openAssetLibraryDialog(node, asset, {
            name: label.value.trim(), identity: "",
        });
        let identityControl = label;
        if (options.referenceWeights) {
            ref.style.flex = "0 0 auto";
            const identity = document.createElement("div");
            identity.style.cssText = "min-width:160px;flex:1 1 180px;display:grid;grid-template-columns:minmax(72px,1fr) 112px;gap:4px;align-items:stretch;";
            const weight = document.createElement("div");
            weight.setAttribute("role", "group");
            weight.setAttribute("aria-label", `${mention} 参考权重`);
            weight.title = "自动：按运行时实际 DiT token 数温和补偿，动作视频为 1.00 基准；手动范围 0.25～3.00。不会改变素材分辨率或视频帧。";
            weight.style.cssText = "display:grid;grid-template-columns:54px 54px;gap:3px;min-width:0;";
            const mode = document.createElement("select");
            mode.dataset.myangControl = `shot:${shot.id}:asset:${asset.id}:weight-mode`;
            for (const [value, text] of [["auto", "自动"], ["manual", "手动"]]) {
                const option = document.createElement("option");
                option.value = value;
                option.textContent = text;
                mode.appendChild(option);
            }
            mode.value = asset.reference_weight_mode === "manual" ? "manual" : "auto";
            mode.style.cssText = "min-width:0;width:54px;background:#101923;color:#bfdbfe;border:1px solid #36506b;border-radius:3px;padding:3px 2px;font-size:9px;cursor:pointer;";
            const amount = document.createElement("input");
            amount.dataset.myangControl = `shot:${shot.id}:asset:${asset.id}:weight`;
            amount.type = "number";
            amount.min = ".25";
            amount.max = "3";
            amount.step = ".05";
            amount.value = mode.value === "manual"
                ? Number(asset.reference_weight || 1).toFixed(2) : "";
            amount.placeholder = "运行时";
            amount.style.cssText = "min-width:0;width:54px;box-sizing:border-box;background:#0b1118;color:#fbbf24;border:1px solid #36506b;border-radius:3px;padding:3px 2px;font-size:9px;text-align:center;";
            const syncWeightControl = () => {
                const manual = mode.value === "manual";
                amount.disabled = !manual;
                amount.style.opacity = manual ? "1" : ".62";
                if (manual && !amount.value) amount.value = Number(asset.reference_weight || 1).toFixed(2);
                if (!manual) amount.value = "";
            };
            mode.onchange = () => {
                asset.reference_weight_mode = mode.value;
                asset.reference_weight = Math.max(.25, Math.min(3,
                    Number(asset.reference_weight) || 1));
                syncWeightControl();
                saveTimeline(node);
            };
            amount.onchange = () => {
                const value = Math.max(.25, Math.min(3, Number(amount.value) || 1));
                asset.reference_weight = value;
                amount.value = value.toFixed(2);
                saveTimeline(node);
            };
            syncWeightControl();
            weight.append(mode, amount);
            identity.append(label, weight);
            identityControl = identity;
        }
        row.append(ref, identityControl, archive, remove);
        row.style.flexWrap = "wrap";
        item.appendChild(row);
        item.title = "右键加入导演台素材库";
        item.oncontextmenu = (event) => {
            event.preventDefault();
            openAssetLibraryDialog(node, asset, {
                name: label.value.trim(), identity: "",
            });
        };
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

function agentConnectionKey(connection) {
    const sourceId = Number(connection?.source_id);
    const sourceSlot = Number(connection?.source_slot || 0);
    return Number.isFinite(sourceId) && sourceId >= 0
        ? `${sourceId}:${sourceSlot}` : "";
}

function reorderAgentMedia(node, movingKey, targetKey) {
    const agent = upstream(node, "media");
    if (!agent?.properties) return false;
    const property = Array.isArray(agent.properties[AGENT_MEDIA_PROP])
        ? AGENT_MEDIA_PROP : AGENT_LINKS;
    const current = agent.properties[property];
    if (!Array.isArray(current)) return false;
    const list = current.slice();
    const from = list.findIndex((entry) => agentConnectionKey(entry) === movingKey);
    const to = list.findIndex((entry) => agentConnectionKey(entry) === targetKey);
    if (from < 0 || to < 0 || from === to) return false;
    const moving = list[from];
    const target = list[to];
    const movingKind = KIND_OF_TYPE[String(moving?.media_type || "image").toLowerCase()] || "图片";
    const targetKind = KIND_OF_TYPE[String(target?.media_type || "image").toLowerCase()] || "图片";
    if (movingKind !== targetKind) return false;
    list.splice(from, 1);
    list.splice(to, 0, moving);
    list.forEach((entry, index) => {
        if (entry && typeof entry === "object") entry.order = index + 1;
    });
    agent.properties[property] = list;
    node.graph?.setDirtyCanvas?.(true, true);
    return true;
}

function renderAgentMediaOrderPanel(node) {
    const entries = globalMediaList(node, {allowAgent: true})
        .filter((entry) => entry.source === "global" && entry.assetId && entry.source_id != null);
    if (!entries.length) return null;

    const panel = document.createElement("div");
    panel.style.cssText = "border:1px solid #31465d;background:#0d1722;border-radius:5px;padding:6px;margin-bottom:7px;";
    const title = document.createElement("div");
    title.textContent = "Media Agent 素材顺序";
    title.style.cssText = "color:#bae6fd;font-size:10px;font-weight:700;margin-bottom:2px;";
    const help = document.createElement("div");
    help.textContent = "拖动同类型素材调整 @图片N / @视频N / @音频N 槽位；提示词中的序号文字保持不变。";
    help.style.cssText = "color:#7891a8;font-size:9px;line-height:1.45;margin-bottom:5px;";
    panel.append(title, help);

    const grouped = new Map([["图片", []], ["视频", []], ["音频", []]]);
    for (const entry of entries) grouped.get(entry.kind)?.push(entry);
    for (const [kind, items] of grouped) {
        if (!items.length) continue;
        const group = document.createElement("div");
        group.style.cssText = "display:flex;align-items:center;gap:5px;flex-wrap:wrap;margin-top:4px;";
        const label = document.createElement("span");
        label.textContent = `${kind}槽位`;
        label.style.cssText = "flex:0 0 46px;color:#94a3b8;font-size:9px;font-weight:700;";
        group.appendChild(label);
        items.forEach((entry, index) => {
            const card = document.createElement("div");
            const key = agentConnectionKey(entry);
            const token = `@${kind}${index + 1}`;
            const display = entry.subject || entry.name || entry.file || "未命名素材";
            card.textContent = `${token} · ${display}`;
            card.title = `${display}\n拖动调整 ${token} 对应的素材`;
            card.draggable = true;
            card.dataset.agentMediaKey = key;
            card.style.cssText = "flex:0 1 auto;min-width:120px;max-width:260px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;border:1px solid #34495e;background:#111d2a;color:#cfe8ff;border-radius:4px;padding:4px 6px;font-size:9px;cursor:grab;transition:border-color .18s ease,background .18s ease;";
            card.ondragstart = (event) => {
                node.__myangDraggedAgentKey = key;
                event.dataTransfer.effectAllowed = "move";
                event.dataTransfer.setData("text/x-myang-agent-media-key", key);
                card.style.opacity = ".55";
            };
            card.ondragend = () => {
                card.style.opacity = "1";
                node.__myangDraggedAgentKey = "";
            };
            card.ondragover = (event) => {
                const moving = node.__myangDraggedAgentKey
                    || event.dataTransfer?.getData("text/x-myang-agent-media-key");
                const movingEntry = entries.find((candidate) => agentConnectionKey(candidate) === moving);
                if (!moving || moving === key || movingEntry?.kind !== entry.kind) return;
                event.preventDefault();
                event.dataTransfer.dropEffect = "move";
                card.style.borderColor = "#60a5fa";
                card.style.background = "#152b42";
            };
            card.ondragleave = () => {
                card.style.borderColor = "#34495e";
                card.style.background = "#111d2a";
            };
            card.ondrop = (event) => {
                const moving = node.__myangDraggedAgentKey
                    || event.dataTransfer?.getData("text/x-myang-agent-media-key");
                const movingEntry = entries.find((candidate) => agentConnectionKey(candidate) === moving);
                if (!moving || moving === key || movingEntry?.kind !== entry.kind) return;
                event.preventDefault();
                if (reorderAgentMedia(node, moving, key)) renderTimeline(node);
            };
            group.appendChild(card);
        });
        panel.appendChild(group);
    }
    return panel;
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
    let addedReferenceWeights = false;
    for (const asset of shot.assets) {
        if (!["auto", "manual"].includes(asset.reference_weight_mode)) {
            asset.reference_weight_mode = "auto";
            asset.reference_weight = 1;
            addedReferenceWeights = true;
        }
    }
    if (addedReferenceWeights) saveTimeline(node);
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
    if (inputLinked(node, "media")) {
        const mediaNote = modeNotice(
            "已隔离外接 Media Agent 素材",
            "动作迁移只读取当前动作迁移模式的导演台素材；外接 Media Agent 的旧素材不会参与编号、提示词或生成。",
            true);
        root.appendChild(mediaNote);
    }

    const card = document.createElement("div");
    card.style.cssText = "border:1px solid #42546a;background:#1a222c;border-radius:7px;padding:9px;";
    const row = document.createElement("div");
    row.style.cssText = "display:grid;grid-template-columns:minmax(220px,1fr) 160px;gap:7px;align-items:end;margin-bottom:7px;";
    const durationLabel = document.createElement("label");
    durationLabel.textContent = "自动分段时长（秒）";
    durationLabel.style.cssText = "color:#fbbf24;font-size:10px;font-weight:700;";
    const duration = document.createElement("input");
    duration.dataset.myangControl = "transfer:segment_seconds";
    duration.type = "number";
    duration.min = "1";
    duration.max = "30";
    duration.step = "0.5";
    duration.value = String(widget(node, "segment_seconds")?.value || 10);
    const autoSegmentEnabled = widget(node, "动作迁移自动分段")?.value !== false;
    duration.disabled = !autoSegmentEnabled;
    duration.title = autoSegmentEnabled
        ? "动作迁移按此时长自动连续切段"
        : "已关闭自动分段，将按动作视频完整长度生成一段";
    duration.style.cssText = "width:100%;box-sizing:border-box;background:#111820;color:#ffd866;border:1px solid #44546a;border-radius:4px;padding:6px;";
    duration.style.opacity = autoSegmentEnabled ? "1" : ".45";
    duration.onchange = () => setNativeWidget(node, "segment_seconds", Number(duration.value) || 10);
    const durationWrap = document.createElement("div");
    durationWrap.append(durationLabel, duration);
    const modeWrap = document.createElement("label");
    modeWrap.style.cssText = "display:flex;flex-direction:column;gap:3px;min-width:0;color:#aebdcd;font-size:9px;";
    const modeCaption = document.createElement("span");
    modeCaption.textContent = "生成长度模式";
    const modeSelect = document.createElement("select");
    modeSelect.dataset.myangControl = "动作迁移自动分段";
    for (const value of ["按单段时长自动分段", "整段生成（自动匹配视频长度）"]) {
        const option = document.createElement("option");
        option.value = option.textContent = value;
        modeSelect.appendChild(option);
    }
    modeSelect.value = autoSegmentEnabled
        ? "按单段时长自动分段"
        : "整段生成（自动匹配视频长度）";
    modeSelect.style.cssText = "min-width:0;width:100%;box-sizing:border-box;background:#0c131c;color:#dbe7f3;border:1px solid #37475b;border-radius:4px;padding:6px;font-size:10px;";
    modeSelect.onchange = () => setNativeWidget(
        node, "动作迁移自动分段", modeSelect.value === "按单段时长自动分段");
    modeWrap.append(modeCaption, modeSelect);
    row.append(modeWrap, durationWrap);
    card.appendChild(row);
    card.appendChild(renderReferenceVideoPanel(node));
    card.appendChild(renderResumePanel(node));

    const promptLabel = document.createElement("div");
    promptLabel.textContent = "全片统一提示词";
    promptLabel.style.cssText = "color:#7dd3fc;font-size:10px;font-weight:700;margin:8px 0 4px;";
    const prompt = createPromptEditor(node, shot, {
        placeholder: "描述要迁移的动作；键入 @ 或直接输入 @图片1 / @视频1，所有自动分段共用此提示词",
    });
    card.append(promptLabel, prompt.__myangDialogueToolbar, prompt);
    card.appendChild(renderShotAssets(node, shot, prompt, {
        title: "动作源与全片辅助素材",
        allowedKinds: ["image", "video", "audio"],
        limits: {video: 1},
        videoRole: "action",
        showAssetMode: false,
        referenceWeights: true,
        emptyText: "上传一个完整动作视频；还可添加目标人物图片和辅助音频。",
    }));
    root.appendChild(card);
}

const DETAIL_FIELDS = [
    "二采开启", "二采模式", "二采分辨率", "二采自定义宽", "二采自定义高",
    "二采步数", "二采重绘幅度", "二采调度器", "二采采样器", "二采放大方式",
    "二采分块帧数", "二采Latent模型", "二采精度", "二采时间分块", "二采轮数",
    "二采种子策略", "二采复用一采条件", "save_raw_segments",
    "二采显存策略", "二采自定义显存预留", "二采自定义预览间隔",
    "二采连续Sigma",
    "二采后VSR增强",
    "一采断点模式",
];

const ENHANCEMENT_FIELDS = [
    "脸部精修开启", "脸部检测器", "脸部精修步数", "脸部精修重绘",
    "脸部裁剪倍率", "脸部身份图序号", "动作修复开启", "动作修复档位",
    "动作修复步数", "动作修复注入", "多视角分镜开启",
    "多视角角色图片序号", "多视角尺寸", "多视角步数", "多视角LoRA",
    "多视角LoRA强度",
];

const AUDIO_FIELDS = [
    "音频精修开启", "音频精修步数", "音频去噪强度", "音频精修采样器",
    "音频精修调度器", "音频接缝平滑", "音频接缝时长",
];

const REFERENCE_FIELDS = [
    "参考视频分辨率", "参考视频自定义宽", "参考视频自定义高",
];

const SKILL_FIELDS = ["skill_preset", "skill_text", "vlm_service",
    "分层提示词"];
const FIRST_MEMORY_FIELDS = ["一采显存策略"];

// Most Director controls mirror hidden ComfyUI widgets.  Updating a mirrored
// value must not rebuild the whole DOM form: replacing the active input drops
// its focus/caret and makes continuous typing impossible.  Only controls that
// genuinely change visible structure are allowed to request a render, and
// those renders are kept to the smallest affected card whenever possible.
const DIRECTOR_SECTION_REFRESH = new Map([
    ["resolution", "generation-settings"],
    ["aspect_ratio", "generation-settings"],
    ["width", "generation-settings"],
    ["height", "generation-settings"],
    ["denoise", "generation-settings"],
    ["scheduler", "generation-settings"],
    ["context_length", "generation-settings"],
    ["ref_image_size", "generation-settings"],
    ["save_segments", "generation-settings"],
    ["segment_prefix", "generation-settings"],
    ["一采显存策略", "firstMemory"],
    ["二采开启", "detail"],
    ["二采模式", "detail"],
    ["二采分辨率", "detail"],
    ["二采放大方式", "detail"],
    ["二采显存策略", "detail"],
    ["二采连续Sigma", "detail"],
    ["二采后VSR增强", "detail"],
    ["一采断点模式", "firstMemory"],
    ["音频精修开启", "audio"],
    ["音频接缝平滑", "audio"],
    ["脸部精修开启", "enhancement"],
    ["动作修复开启", "enhancement"],
    ["多视角分镜开启", "enhancement"],
    ["参考视频分辨率", "reference"],
    ["动作迁移自动分段", "full"],
    ["从指定段开始", "resume"],
    ["起始段", "resume"],
    ["skill_preset", "skill"],
    ["vlm_service", "skill"],
    ["分层提示词", "skill"],
]);

const DIRECTOR_FULL_REFRESH = new Set(["task_mode", "source_mode"]);

function captureDirectorView(node) {
    const root = node.__myangDirectorRoot;
    if (!root) return null;
    const state = {scrollTop: root.scrollTop, control: "", start: null, end: null, caret: null};
    const active = document.activeElement;
    if (!active || !root.contains(active)) return state;
    state.control = String(active.dataset?.myangControl || "");
    if (!state.control) return state;
    if (active.isContentEditable) state.caret = promptCaretOffset(active);
    else if (typeof active.selectionStart === "number") {
        state.start = active.selectionStart;
        state.end = active.selectionEnd;
    }
    return state;
}

function restoreDirectorView(node, state) {
    const root = node.__myangDirectorRoot;
    if (!root || !state) return;
    root.scrollTop = Number(state.scrollTop || 0);
    requestAnimationFrame(() => {
        if (!root.isConnected) return;
        root.scrollTop = Number(state.scrollTop || 0);
        if (!state.control) return;
        const target = Array.from(root.querySelectorAll("[data-myang-control]")).find(
            (element) => element.dataset.myangControl === state.control);
        if (!target) return;
        target.focus?.({preventScroll: true});
        if (target.isContentEditable && state.caret != null) {
            restorePromptCaret(target, state.caret);
        } else if (state.start != null && typeof target.setSelectionRange === "function") {
            target.setSelectionRange(state.start, state.end ?? state.start);
        }
        root.scrollTop = Number(state.scrollTop || 0);
    });
}

function replaceDirectorSection(node, section) {
    const root = node.__myangDirectorRoot;
    if (!root?.isConnected) return;
    const renderers = {
        firstMemory: renderFirstPassMemoryPanel,
        "generation-settings": renderGenerationSettingsPanel,
        detail: renderDetailPanel,
        audio: renderAudioPanel,
        enhancement: renderEnhancementPanel,
        reference: renderReferenceVideoPanel,
        resume: renderResumePanel,
        skill: renderSkillPanel,
    };
    const renderer = renderers[section];
    if (!renderer) return;
    const current = Array.from(root.querySelectorAll(
        `[data-myang-director-section="${section}"], [data-myang-collapsible="${section}"]`));
    if (!current.length) return;
    const view = captureDirectorView(node);
    for (const element of current) element.replaceWith(renderer(node));
    restoreDirectorView(node, view);
}

async function assetLibraryRequest(path = "", options = {}) {
    const response = await fetch(`/minimax-h3-myang/assets${path}`, options);
    if (!response.ok) throw new Error((await response.text()) || `HTTP ${response.status}`);
    return response.json();
}

async function mediaLibraryRequest(path = "", options = {}) {
    const response = await fetch(`${MEDIA_LIBRARY_ROUTE}${path}`, options);
    if (!response.ok) throw new Error((await response.text()) || `HTTP ${response.status}`);
    return response.json();
}

function isTextEditingTarget(target) {
    const tag = String(target?.tagName || "").toLowerCase();
    return ["input", "textarea", "select"].includes(tag)
        || target?.isContentEditable
        || target?.getAttribute?.("role") === "textbox";
}

// Dialogs live above the LiteGraph canvas. Keep destructive keyboard shortcuts
// inside the dialog so Backspace/Delete can never remove the underlying node,
// while preserving normal text editing in inputs and textareas.
function guardDialogKeys(root) {
    if (!root || root.__myangDialogKeyGuard) return;
    root.__myangDialogKeyGuard = true;
    root.addEventListener("keydown", (event) => {
        const key = String(event.key || "").toLowerCase();
        if ((key === "backspace" || key === "delete")
            && !isTextEditingTarget(event.target)) {
            event.preventDefault();
            event.stopPropagation();
            event.stopImmediatePropagation?.();
            return;
        }
        // Bubble-phase isolation lets the focused control process the key
        // first, then prevents ComfyUI canvas shortcuts from seeing it.
        event.stopPropagation();
    });
    for (const type of ["keyup", "keypress"]) {
        root.addEventListener(type, (event) => event.stopPropagation());
    }
}

function materialPickerShell(id, titleText, {closeOnBackdrop = true} = {}) {
    document.getElementById(id)?.remove();
    const overlay = document.createElement("div");
    overlay.id = id;
    overlay.setAttribute("role", "dialog");
    overlay.setAttribute("aria-modal", "true");
    overlay.tabIndex = -1;
    guardDialogKeys(overlay);
    overlay.style.cssText = "position:fixed;inset:0;z-index:100006;display:flex;align-items:center;justify-content:center;padding:20px;background:rgba(2,6,12,.78);";
    const card = document.createElement("div");
    card.style.cssText = "width:min(94vw,920px);max-height:min(86vh,760px);display:flex;flex-direction:column;overflow:hidden;border:1px solid #40536a;border-radius:10px;background:#101822;color:#dbeafe;box-shadow:0 22px 70px rgba(0,0,0,.58);";
    const head = document.createElement("div");
    head.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:10px;padding:11px 13px;border-bottom:1px solid #29394a;";
    const title = document.createElement("b");
    title.textContent = titleText;
    const close = button("关闭", "关闭素材选择器");
    close.style.cssText += ";flex:0 0 66px;min-height:32px;";
    const dismiss = () => {
        const callback = overlay.__myangOnClose;
        overlay.__myangOnClose = null;
        overlay.remove();
        callback?.();
    };
    close.onclick = dismiss;
    head.append(title, close);
    const body = document.createElement("div");
    body.style.cssText = "min-height:0;overflow:auto;padding:12px;";
    card.append(head, body);
    overlay.appendChild(card);
    overlay.onclick = (event) => {
        if (closeOnBackdrop && event.target === overlay) dismiss();
    };
    overlay.__myangDismiss = dismiss;
    document.body.appendChild(overlay);
    requestAnimationFrame(() => {
        if (overlay.isConnected && !overlay.contains(document.activeElement)) {
            overlay.focus({preventScroll: true});
        }
    });
    return {overlay, card, body};
}

function openLibraryMediaPreview(entry, url) {
    const {body} = materialPickerShell(
        "myang-library-preview-dialog", `素材预览 · ${entry.name || "未命名"}`);
    const media = document.createElement(entry.kind === "image" ? "img" : entry.kind);
    media.src = url;
    if (entry.kind !== "image") media.controls = true;
    media.preload = "metadata";
    media.style.cssText = entry.kind === "audio"
        ? "display:block;width:100%;min-height:54px;"
        : "display:block;width:100%;max-height:68vh;object-fit:contain;background:#05090e;border-radius:6px;";
    body.appendChild(media);
}

function openMaterialSourceDialog(node, {kind, local, library}) {
    const meta = MEDIA_META[kind];
    const {overlay, body} = materialPickerShell("myang-material-source-dialog", `添加${meta?.label || "素材"}`);
    body.style.cssText += ";display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:10px;";
    const choice = (title, description, color, action) => {
        const control = document.createElement("button");
        control.type = "button";
        control.style.cssText = `min-height:112px;padding:14px;text-align:left;border:1px solid ${color};border-radius:8px;background:#0b131d;color:#e5edf5;cursor:pointer;`;
        const heading = document.createElement("b");
        heading.textContent = title;
        heading.style.cssText = "display:block;font-size:13px;margin-bottom:7px;";
        const help = document.createElement("span");
        help.textContent = description;
        help.style.cssText = "display:block;color:#8ea1b6;font-size:10px;line-height:1.5;";
        control.append(heading, help);
        control.onclick = () => { overlay.remove(); action(); };
        return control;
    };
    body.append(
        choice("从素材库选择", "浏览已收藏素材和已经登记的素材文件夹。", "#3973a8", library),
        choice("从本地上传", "从电脑选择文件并复制到 ComfyUI/input。", "#4d7653", local),
    );
}

async function openMaterialLibraryChooser(node, {kind = "", onSelect = null} = {}) {
    const {overlay, body} = materialPickerShell(
        "myang-material-library-chooser", kind ? `素材库 · 选择${MEDIA_META[kind]?.label || "素材"}` : "浏览素材库");
    body.style.cssText += ";display:grid;grid-template-columns:220px minmax(0,1fr);gap:10px;";
    const sidebar = document.createElement("div");
    sidebar.style.cssText = "display:flex;flex-direction:column;gap:6px;min-height:260px;border-right:1px solid #29394a;padding-right:10px;";
    const main = document.createElement("div");
    main.style.cssText = "min-width:0;";
    const top = document.createElement("div");
    top.style.cssText = "display:flex;gap:6px;align-items:center;margin-bottom:9px;";
    const search = document.createElement("input");
    search.placeholder = "搜索素材名称";
    search.style.cssText = "flex:1;min-width:0;min-height:34px;border:1px solid #34465b;border-radius:5px;background:#0b1118;color:#e5edf5;padding:6px 8px;";
    const addFolder = button("＋ 添加素材文件夹", "使用系统文件夹选择器添加素材库");
    addFolder.style.cssText += ";flex:0 0 126px;min-height:34px;";
    top.append(search, addFolder);
    const grid = document.createElement("div");
    grid.style.cssText = "display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:7px;";
    main.append(top, grid);
    body.append(sidebar, main);

    let selected = {type: "catalogue", id: "catalogue", name: "已收藏素材"};
    let libraries = [];
    let entries = [];
    const entryUrl = (entry) => entry.source === "folder"
        ? `${MEDIA_LIBRARY_ROUTE}/media?${new URLSearchParams({library_id: entry.library_id, path: entry.relative_path})}`
        : assetViewUrl({kind: entry.kind, file: entry.file});
    const redrawGrid = () => {
        grid.replaceChildren();
        const query = search.value.trim().toLowerCase();
        const filtered = entries.filter((entry) => (!kind || entry.kind === kind)
            && (!query || `${entry.name || ""} ${entry.relative_path || ""}`.toLowerCase().includes(query)));
        if (!filtered.length) {
            const empty = document.createElement("div");
            empty.textContent = selected.type === "catalogue"
                ? "这里没有匹配的已收藏素材，可先从镜头素材卡点击“入库”。"
                : "这个文件夹里没有匹配的素材。";
            empty.style.cssText = "grid-column:1/-1;padding:28px;color:#718096;text-align:center;font-size:10px;";
            grid.appendChild(empty);
            return;
        }
        for (const entry of filtered) {
            const item = document.createElement("button");
            item.type = "button";
            item.title = entry.relative_path || entry.file?.name || entry.name;
            item.style.cssText = "min-width:0;padding:5px;border:1px solid #304155;border-radius:6px;background:#0b121b;color:#dbeafe;text-align:left;cursor:pointer;";
            const preview = document.createElement(entry.kind === "image" ? "img" : entry.kind);
            preview.src = entryUrl(entry);
            preview.preload = "metadata";
            if (entry.kind === "video") preview.muted = true;
            if (entry.kind === "audio") preview.controls = true;
            preview.style.cssText = entry.kind === "audio"
                ? "display:block;width:100%;height:34px;margin:23px 0;"
                : "display:block;width:100%;height:80px;object-fit:contain;background:#05090e;border-radius:4px;";
            const name = document.createElement("div");
            name.textContent = entry.name || entry.file?.name || "未命名素材";
            name.style.cssText = "margin-top:5px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:9px;";
            item.append(preview, name);
            item.onclick = async () => {
                if (!onSelect) {
                    openLibraryMediaPreview(entry, entryUrl(entry));
                    return;
                }
                item.disabled = true;
                try {
                    let file = entry.file;
                    if (entry.source === "folder") {
                        const result = await mediaLibraryRequest("/import", {
                            method: "POST", headers: {"Content-Type": "application/json"},
                            body: JSON.stringify({library_id: entry.library_id, path: entry.relative_path}),
                        });
                        file = result.file;
                    }
                    await onSelect({kind: entry.kind, label: entry.name, file});
                    overlay.remove();
                } catch (error) {
                    alert(`素材添加失败：${error.message}`);
                    item.disabled = false;
                }
            };
            grid.appendChild(item);
        }
    };
    const loadSelected = async () => {
        grid.replaceChildren();
        const loading = document.createElement("div");
        loading.textContent = "正在读取素材…";
        loading.style.cssText = "grid-column:1/-1;padding:28px;color:#94a3b8;text-align:center;";
        grid.appendChild(loading);
        try {
            if (selected.type === "catalogue") {
                const result = await assetLibraryRequest();
                entries = (result.assets || []).map((entry) => ({
                    ...entry, source: "catalogue", name: entry.name || entry.file?.name,
                }));
            } else {
                const result = await mediaLibraryRequest(`/assets?library_id=${encodeURIComponent(selected.id)}`);
                entries = (result.assets || []).map((entry) => ({...entry, source: "folder"}));
            }
            redrawGrid();
        } catch (error) {
            entries = [];
            loading.textContent = `读取失败：${error.message}`;
        }
    };
    const redrawSidebar = () => {
        sidebar.replaceChildren();
        const sourceButton = (source) => {
            const control = document.createElement("button");
            control.type = "button";
            control.textContent = source.name;
            control.title = source.path || source.name;
            const active = selected.type === source.type && selected.id === source.id;
            control.style.cssText = `min-height:34px;padding:6px 8px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;text-align:left;border:1px solid ${active ? "#4f91cc" : "#304155"};border-radius:5px;background:${active ? "#173451" : "#0b121b"};color:${active ? "#dbeafe" : "#94a3b8"};cursor:pointer;`;
            control.onclick = () => { selected = source; redrawSidebar(); loadSelected(); };
            return control;
        };
        sidebar.appendChild(sourceButton({type: "catalogue", id: "catalogue", name: "★ 已收藏素材"}));
        for (const library of libraries) sidebar.appendChild(sourceButton({
            type: "folder", id: library.id, name: library.name, path: library.path,
        }));
    };
    const loadLibraries = async () => {
        const result = await mediaLibraryRequest("/libraries");
        libraries = result.libraries || [];
        redrawSidebar();
    };
    addFolder.onclick = async () => {
        addFolder.disabled = true;
        try {
            const result = await mediaLibraryRequest("/browse-folder", {
                method: "POST", headers: {"Content-Type": "application/json"}, body: "{}",
            });
            if (!result.cancelled) {
                await loadLibraries();
                selected = {type: "folder", id: result.library.id, name: result.library.name, path: result.library.path};
                redrawSidebar();
                await loadSelected();
            }
        } catch (error) { alert(`添加素材文件夹失败：${error.message}`); }
        finally { addFolder.disabled = false; }
    };
    search.oninput = redrawGrid;
    await loadLibraries();
    await loadSelected();
}

function defaultAssetCategory(asset, subject = null) {
    if (subject?.id || subject?.name) return "character";
    if (asset.kind === "video") return "video";
    if (asset.kind === "audio") return "music";
    return "image";
}

function openAssetLibraryDialog(node, asset, subject = null) {
    document.getElementById("myang-asset-library-dialog")?.remove();
    const overlay = document.createElement("div");
    overlay.id = "myang-asset-library-dialog";
    overlay.setAttribute("role", "dialog");
    overlay.setAttribute("aria-modal", "true");
    guardDialogKeys(overlay);
    overlay.style.cssText = "position:fixed;inset:0;z-index:100002;display:flex;align-items:center;justify-content:center;padding:20px;background:rgba(2,6,12,.76);";
    const card = document.createElement("div");
    card.style.cssText = "width:min(92vw,460px);border:1px solid #40536a;border-radius:9px;background:#111923;color:#dbeafe;padding:14px;box-shadow:0 22px 70px rgba(0,0,0,.55);";
    const title = document.createElement("div");
    title.textContent = "加入导演台素材库";
    title.style.cssText = "font-size:14px;font-weight:700;margin-bottom:10px;";
    const field = (labelText, input) => {
        const label = document.createElement("label");
        label.textContent = labelText;
        label.style.cssText = "display:flex;flex-direction:column;gap:5px;margin-top:9px;color:#93a4b8;font-size:10px;";
        input.style.cssText = "min-height:36px;box-sizing:border-box;border:1px solid #34465b;border-radius:5px;background:#0b1118;color:#e5edf5;padding:7px 8px;outline:none;";
        label.appendChild(input);
        return label;
    };
    const name = document.createElement("input");
    name.value = String(subject?.name || asset.label || asset.file?.name || "");
    name.maxLength = 120;
    const category = document.createElement("select");
    for (const [value, label] of Object.entries(ASSET_CATEGORY_LABELS)) {
        const option = document.createElement("option");
        option.value = value;
        option.textContent = label;
        category.appendChild(option);
    }
    category.value = defaultAssetCategory(asset, subject);
    const identity = document.createElement("textarea");
    identity.rows = 3;
    identity.value = String(subject?.identity || subject?.declaration || "");
    identity.placeholder = "可选：记录不可变身份、场景基准或声音特征";
    const status = document.createElement("div");
    status.setAttribute("role", "status");
    status.style.cssText = "min-height:16px;margin-top:8px;color:#fbbf24;font-size:9px;";
    const actions = document.createElement("div");
    actions.style.cssText = "display:flex;justify-content:flex-end;gap:8px;margin-top:10px;";
    const cancel = button("取消", "不加入素材库");
    const save = button("保存到素材库", "保存引用和主体信息，不复制原文件");
    for (const control of [cancel, save]) control.style.cssText += ";min-height:36px;min-width:88px;";
    save.style.cssText += ";background:#2563a8;border-color:#3b82c4;color:#fff;font-weight:700;";
    cancel.onclick = () => overlay.remove();
    save.onclick = async () => {
        if (!name.value.trim()) {
            status.textContent = "请填写素材名称";
            name.focus();
            return;
        }
        save.disabled = true;
        status.textContent = "正在保存…";
        try {
            const result = await assetLibraryRequest("", {
                method: "POST", headers: {"Content-Type": "application/json"},
                body: JSON.stringify({
                    name: name.value.trim(), category: category.value,
                    kind: asset.kind, subject_id: subject?.id || "",
                    subject_name: subject?.name || name.value.trim(),
                    identity: identity.value.trim(),
                    file: {...asset.file, type: asset.file?.type || "input"},
                }),
            });
            node.__myangAssetCatalogue = [
                ...(node.__myangAssetCatalogue || []).filter((item) => item.id !== result.asset.id),
                result.asset,
            ];
            node.__myangAssetCatalogueLoaded = true;
            node.__myangAssetLibraryStatus = `已保存「${result.asset.name}」`;
            overlay.remove();
            renderTimeline(node);
        } catch (error) {
            status.textContent = `保存失败：${error.message}`;
            save.disabled = false;
        }
    };
    actions.append(cancel, save);
    card.append(title, field("素材名称", name), field("分类", category),
        field("主体 / 素材说明", identity), status, actions);
    overlay.appendChild(card);
    overlay.onclick = (event) => { if (event.target === overlay) overlay.remove(); };
    document.body.appendChild(overlay);
    name.focus();
    name.select();
}

function planAssetCandidates(node) {
    const materials = globalMediaList(node, {allowAgent: currentTask(node) === FRESH});
    const byToken = new Map(materials.map((entry) => [entry.token, entry]));
    const candidates = [];
    const seen = new Set();
    for (const segment of node.__myangDirectorPlan?.segments || []) {
        for (const subject of segment.subjects || []) {
            for (const token of subject.media || []) {
                const entry = byToken.get(String(token));
                if (!entry?.file || seen.has(`${subject.id}|${entry.file}`)) continue;
                seen.add(`${subject.id}|${entry.file}`);
                candidates.push({
                    subject,
                    asset: {
                        kind: TYPE_OF_KIND[entry.kind] || "image",
                        label: entry.subject || entry.name || entry.file,
                        file: {name: entry.file, subfolder: entry.subfolder || "", type: "input"},
                    },
                });
            }
        }
    }
    return candidates;
}

async function directorTemplateRequest(path = "", options = {}) {
    const response = await fetch(`/minimax-h3-myang/director-templates${path}`, options);
    if (!response.ok) throw new Error((await response.text()) || `HTTP ${response.status}`);
    return response.json();
}

function directorTemplateSource(node, taskMode = currentTask(node)) {
    // Resolve the active private mode bucket at the moment the dialog opens.
    // Action transfer is a single-source workflow: only its one action card is
    // meaningful, and old global cards must never leak into the template.
    syncModeBucket(node);
    const shots = Array.isArray(node.__myangDirectorShots)
        ? node.__myangDirectorShots : [];
    if (taskMode === TRANSFER) {
        const source = shots[0] || freshShot(1);
        return {
            shots: [{
                ...source,
                asset_mode: "仅本镜头",
                assets: (source.assets || []).map((asset) => ({
                    ...asset,
                    file: {...asset.file},
                    role: asset.kind === "video" ? "action" : asset.role,
                })),
            }],
            globalAssets: [],
        };
    }
    return {
        shots,
        globalAssets: Array.isArray(node.__myangDirectorGlobals)
            ? node.__myangDirectorGlobals : [],
    };
}

function templateSettings(node) {
    const allowed = new Set([
        "resolution", "aspect_ratio", "width", "height", "fps", "noise_seed",
        "steps", "denoise", "scheduler", "context_length", "ref_image_size",
        "seed", "cfg", "sampler_name", "turbo_lora", "turbo_profile",
        "total_seconds", "segment_seconds", "动作迁移自动分段",
        ...REFERENCE_FIELDS,
        ...DETAIL_FIELDS, ...ENHANCEMENT_FIELDS, ...AUDIO_FIELDS, ...FIRST_MEMORY_FIELDS,
    ]);
    const result = {};
    for (const entry of node.widgets || []) {
        if (!allowed.has(entry.name)) continue;
        if (["string", "number", "boolean"].includes(typeof entry.value)) result[entry.name] = entry.value;
    }
    return result;
}

function templateInputPurpose(asset) {
    const searchable = `${asset?.label || ""} ${asset?.file?.name || ""}`;
    if (asset?.kind === "video" && asset?.role === "action") return "所需替换的动作参考视频";
    if (asset?.kind === "image" && /(背景|场景|环境|地点|室内|室外)/i.test(searchable)) {
        return "所需替换的背景或场景";
    }
    if (asset?.kind === "image") return "所需替换的角色或参考图";
    if (asset?.kind === "video") return "所需替换的参考视频";
    if (asset?.kind === "audio" && /(音乐|配乐|bgm)/i.test(searchable)) return "所需替换的音乐";
    if (asset?.kind === "audio") return "所需替换的音色或音频";
    return "所需替换的素材";
}

function templateMaterialChoice(asset, checked, onChange, options = {}) {
    const row = document.createElement("div");
    row.style.cssText = "display:grid;grid-template-columns:72px minmax(0,1fr) auto;gap:8px;align-items:center;margin-top:6px;border:1px solid #2f4054;border-radius:6px;background:#0b121b;padding:6px;";
    const preview = document.createElement("button");
    preview.type = "button";
    preview.title = "打开原素材预览";
    preview.style.cssText = "width:72px;height:54px;padding:0;overflow:hidden;border:1px solid #3a4c61;border-radius:5px;background:#060a0f;color:#93c5fd;cursor:pointer;";
    const url = asset?.file?.name ? assetViewUrl(asset) : "";
    if (asset.kind === "image") {
        const image = document.createElement("img");
        image.src = url;
        image.alt = asset.label || asset.file?.name || "图片素材";
        image.loading = "lazy";
        image.style.cssText = "display:block;width:100%;height:100%;object-fit:contain;";
        preview.appendChild(image);
    } else if (asset.kind === "video") {
        const video = document.createElement("video");
        video.src = url;
        video.muted = true;
        video.preload = "metadata";
        video.playsInline = true;
        video.style.cssText = "display:block;width:100%;height:100%;object-fit:contain;";
        preview.appendChild(video);
    } else {
        preview.textContent = "音频预览";
        preview.style.cssText += ";font-size:10px;font-weight:700;";
    }
    preview.onclick = () => openAssetPreview(asset, preview);
    const info = document.createElement("div");
    info.style.cssText = "min-width:0;display:flex;flex-direction:column;gap:3px;";
    const title = document.createElement("b");
    title.textContent = asset.label || asset.file?.name || "未命名素材";
    title.title = title.textContent;
    title.style.cssText = "font-size:10px;color:#dbeafe;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;";
    const file = document.createElement("span");
    file.textContent = `${MEDIA_META[asset.kind]?.label || asset.kind} · ${asset.file?.subfolder ? `${asset.file.subfolder}/` : ""}${asset.file?.name || ""}`;
    file.title = file.textContent;
    file.style.cssText = "font-size:9px;color:#7f91a5;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;";
    const purposeRow = document.createElement("label");
    purposeRow.style.cssText = "display:grid;grid-template-columns:auto minmax(0,1fr);gap:5px;align-items:center;color:#8fa5ba;font-size:9px;";
    const purposeCaption = document.createElement("span");
    purposeCaption.textContent = "输入用途";
    const purpose = document.createElement("input");
    purpose.type = "text";
    purpose.maxLength = 120;
    purpose.value = String(options.inputLabel || templateInputPurpose(asset));
    purpose.placeholder = "例如：所需替换的角色";
    purpose.style.cssText = "box-sizing:border-box;min-width:0;width:100%;border:1px solid #34465b;border-radius:4px;background:#080d13;color:#cfe4f7;padding:4px 6px;font-size:9px;";
    purpose.oninput = () => options.onInputLabel?.(purpose.value.trim());
    options.onInputLabel?.(purpose.value.trim());
    purposeRow.append(purposeCaption, purpose);
    info.append(title, file, purposeRow);
    if (options.showReferenceWeight) {
        if (!["auto", "manual"].includes(asset.reference_weight_mode)) {
            asset.reference_weight_mode = "auto";
            asset.reference_weight = 1;
        }
        const weightRow = document.createElement("label");
        weightRow.style.cssText = "display:grid;grid-template-columns:auto 66px 62px;gap:5px;align-items:center;color:#8fa5ba;font-size:9px;";
        const weightCaption = document.createElement("span");
        weightCaption.textContent = "参考权重";
        const weightMode = document.createElement("select");
        weightMode.append(new Option("自动", "auto"), new Option("手动", "manual"));
        weightMode.value = asset.reference_weight_mode;
        weightMode.style.cssText = "min-width:0;border:1px solid #34465b;border-radius:4px;background:#080d13;color:#cfe4f7;padding:4px 3px;font-size:9px;";
        const weightValue = document.createElement("input");
        weightValue.type = "number";
        weightValue.min = ".25";
        weightValue.max = "3";
        weightValue.step = ".05";
        weightValue.value = Number(asset.reference_weight || 1).toFixed(2);
        weightValue.style.cssText = "box-sizing:border-box;min-width:0;width:100%;border:1px solid #34465b;border-radius:4px;background:#080d13;color:#fbbf24;padding:4px 3px;font-size:9px;text-align:center;";
        const refreshWeight = () => {
            const manual = weightMode.value === "manual";
            weightValue.disabled = !manual;
            weightValue.style.opacity = manual ? "1" : ".48";
        };
        weightMode.onchange = () => {
            asset.reference_weight_mode = weightMode.value;
            refreshWeight();
        };
        weightValue.onchange = () => {
            const value = Math.max(.25, Math.min(3, Number(weightValue.value) || 1));
            asset.reference_weight = value;
            weightValue.value = value.toFixed(2);
        };
        refreshWeight();
        weightRow.append(weightCaption, weightMode, weightValue);
        info.appendChild(weightRow);
    }
    const fixed = document.createElement("label");
    fixed.style.cssText = "display:flex;align-items:center;gap:5px;white-space:nowrap;color:#b8c7d8;font-size:9px;";
    const input = document.createElement("input");
    input.type = "checkbox";
    input.checked = checked;
    const modeCaption = document.createElement("span");
    const refreshMode = () => {
        const isFixed = input.checked;
        modeCaption.textContent = isFixed ? "固定素材" : "模板输入";
        purpose.disabled = isFixed;
        purposeRow.style.opacity = isFixed ? ".42" : "1";
        options.onModeChange?.(!isFixed);
    };
    input.onchange = () => {
        onChange(input.checked);
        refreshMode();
    };
    onChange(checked);
    fixed.append(input, modeCaption);
    refreshMode();
    row.append(preview, info, fixed);
    if (asset.kind === "audio" && url) {
        const audio = document.createElement("audio");
        audio.controls = true;
        audio.preload = "metadata";
        audio.src = url;
        audio.style.cssText = "grid-column:1/-1;width:100%;height:30px;";
        row.appendChild(audio);
    }
    row.__myangTemplateInput = {
        isInput: () => !input.checked,
        setIndex: (index) => {
            purposeCaption.textContent = index > 0 ? `输入${index}用途` : "固定素材";
        },
    };
    return row;
}

const TEMPLATE_SETTING_GROUPS = [
    {title: "画面规格（分别固定）", fields: [
        "resolution", "二采分辨率", "aspect_ratio", "width", "height",
        "二采自定义宽", "二采自定义高", "fps",
    ]},
    {title: "一采与采样", fields: [
        "noise_seed", "seed", "steps", "denoise", "cfg", "sampler_name", "scheduler",
        "context_length", "ref_image_size", "turbo_lora", "turbo_profile", ...FIRST_MEMORY_FIELDS,
    ]},
    {title: "二采", fields: DETAIL_FIELDS.filter((name) => ![
        "二采分辨率", "二采自定义宽", "二采自定义高",
    ].includes(name))},
    {title: "画质与分镜增强", fields: ENHANCEMENT_FIELDS},
    {title: "音频", fields: AUDIO_FIELDS},
    {title: "参考素材处理", fields: REFERENCE_FIELDS},
];

function templateSettingEditor(node, name, initial, onChange) {
    const source = widget(node, name);
    let values = source?.options?.values;
    if (typeof values === "function") {
        try { values = values(); } catch (_error) { values = null; }
    }
    let control;
    if (Array.isArray(values) && values.length) {
        control = document.createElement("select");
        for (const value of values) control.append(new Option(String(value), String(value)));
        control.value = String(initial ?? "");
        control.onchange = () => onChange(control.value);
    } else if (typeof initial === "boolean") {
        control = document.createElement("select");
        control.append(new Option("开启", "true"), new Option("关闭", "false"));
        control.value = initial ? "true" : "false";
        control.onchange = () => onChange(control.value === "true");
    } else {
        control = document.createElement("input");
        control.type = typeof initial === "number" ? "number" : "text";
        if (control.type === "number") {
            if (Number.isFinite(Number(source?.options?.min))) control.min = String(source.options.min);
            if (Number.isFinite(Number(source?.options?.max))) control.max = String(source.options.max);
            if (Number.isFinite(Number(source?.options?.step))) control.step = String(source.options.step);
        }
        control.value = String(initial ?? "");
        control.oninput = () => onChange(control.type === "number" ? Number(control.value) : control.value);
    }
    control.style.cssText = "box-sizing:border-box;min-width:0;width:100%;height:30px;border:1px solid #34465b;border-radius:4px;background:#080d13;color:#dbeafe;padding:4px 7px;font-size:9px;";
    return control;
}

function templateSettingsPanel(node, settings, settingModes) {
    const panel = document.createElement("div");
    panel.style.cssText = "border:1px solid #31506c;border-radius:6px;background:#0b141e;padding:8px;";
    const title = document.createElement("b");
    title.textContent = "固定生成参数";
    title.style.cssText = "display:block;color:#bae6fd;font-size:10px;";
    const help = document.createElement("div");
    help.textContent = "勾选才会写入模板；未勾选的项目在使用模板时沿用本地节点设置。当前值已从本地导演台带入，可在固定前后修改。";
    help.style.cssText = "margin:4px 0 7px;color:#8297ab;font-size:9px;line-height:1.45;";
    const groups = document.createElement("div");
    groups.style.cssText = "display:flex;flex-direction:column;gap:5px;max-height:285px;overflow:auto;padding-right:3px;";
    panel.append(title, help, groups);
    const used = new Set();
    for (const [groupIndex, definition] of TEMPLATE_SETTING_GROUPS.entries()) {
        const names = definition.fields.filter((name) => Object.hasOwn(settings, name) && !used.has(name));
        if (!names.length) continue;
        names.forEach((name) => used.add(name));
        const details = document.createElement("details");
        details.open = groupIndex === 0;
        details.style.cssText = "border:1px solid #273a4d;border-radius:5px;background:#0d1721;";
        const summary = document.createElement("summary");
        summary.style.cssText = "cursor:pointer;padding:7px;color:#c5d7e8;font-size:9px;font-weight:700;";
        const summaryRow = document.createElement("span");
        summaryRow.style.cssText = "display:inline-flex;width:calc(100% - 18px);align-items:center;justify-content:space-between;gap:10px;vertical-align:middle;";
        const summaryTitle = document.createElement("span");
        summaryTitle.textContent = `${definition.title} · ${names.length} 项`;
        const selectAllLabel = document.createElement("label");
        selectAllLabel.style.cssText = "display:inline-flex;align-items:center;gap:5px;min-height:24px;color:#9fc8e8;font-weight:600;cursor:pointer;white-space:nowrap;";
        selectAllLabel.title = "一次固定或取消本大项内的全部参数";
        const selectAll = document.createElement("input");
        selectAll.type = "checkbox";
        const selectAllText = document.createElement("span");
        selectAllText.textContent = "全选";
        selectAllLabel.append(selectAll, selectAllText);
        summaryRow.append(summaryTitle, selectAllLabel);
        summary.appendChild(summaryRow);
        const rows = document.createElement("div");
        rows.style.cssText = "display:grid;grid-template-columns:minmax(128px,.8fr) minmax(130px,1.2fr);gap:5px 8px;padding:0 7px 7px;";
        const groupControls = [];
        let syncGroupCheckbox = () => {};
        for (const name of names) {
            const label = document.createElement("label");
            label.style.cssText = "display:flex;align-items:center;gap:6px;min-width:0;color:#b8c9d9;font-size:9px;";
            const fixed = document.createElement("input");
            fixed.type = "checkbox";
            const caption = document.createElement("span");
            caption.textContent = name === "resolution" ? "固定一采分辨率"
                : name === "二采分辨率" ? "固定二采分辨率"
                    : name === "aspect_ratio" ? "固定画面比例" : (LABELS[name] || name);
            caption.title = caption.textContent;
            caption.style.cssText = "overflow:hidden;text-overflow:ellipsis;white-space:nowrap;";
            label.append(fixed, caption);
            const editor = templateSettingEditor(node, name, settings[name],
                (value) => { settings[name] = value; });
            const refresh = () => {
                settingModes[name] = fixed.checked ? "fixed" : "local";
                editor.disabled = !fixed.checked;
                editor.style.opacity = fixed.checked ? "1" : ".48";
                syncGroupCheckbox();
            };
            fixed.onchange = refresh;
            refresh();
            groupControls.push({fixed, refresh});
            rows.append(label, editor);
        }
        syncGroupCheckbox = () => {
            const selected = groupControls.filter(({fixed}) => fixed.checked).length;
            selectAll.checked = selected === groupControls.length && groupControls.length > 0;
            selectAll.indeterminate = selected > 0 && selected < groupControls.length;
            selectAllText.textContent = selectAll.checked ? "全不选" : "全选";
        };
        selectAllLabel.onclick = (event) => event.stopPropagation();
        selectAll.onchange = () => {
            const checked = selectAll.checked;
            for (const control of groupControls) {
                control.fixed.checked = checked;
                control.refresh();
            }
            syncGroupCheckbox();
        };
        syncGroupCheckbox();
        details.append(summary, rows);
        groups.appendChild(details);
    }
    return panel;
}

function openSaveDirectorTemplateDialog(node) {
    const taskMode = currentTask(node);
    const transferring = taskMode === TRANSFER;
    const source = directorTemplateSource(node, taskMode);
    const editableShots = source.shots.map((shot) => ({
        ...shot,
        assets: (shot.assets || []).map((asset) => ({...asset, file: {...asset.file}})),
    }));
    const {overlay, body} = materialPickerShell(
        "myang-save-template-dialog",
        transferring ? "创建动作迁移模板" : "创建导演台视频模板",
        {closeOnBackdrop: false});
    body.style.cssText += ";display:flex;flex-direction:column;gap:9px;";
    const inputStyle = "min-height:34px;border:1px solid #34465b;border-radius:5px;background:#0b1118;color:#e5edf5;padding:6px 8px;";
    const name = document.createElement("input");
    name.placeholder = "模板名称";
    name.value = node.__myangStoryboardMetadata?.title
        || (transferring ? "我的动作迁移模板" : "我的导演台模板");
    name.style.cssText = inputStyle;
    const description = document.createElement("textarea");
    description.rows = 2;
    description.placeholder = "模板用途说明（可选）";
    description.style.cssText = inputStyle;
    const settingsValues = templateSettings(node);
    const settingModes = {};
    const settingsBox = templateSettingsPanel(node, settingsValues, settingModes);
    const durationBox = document.createElement("div");
    durationBox.style.cssText = "display:grid;grid-template-columns:minmax(150px,1fr) 110px;gap:6px 10px;border:1px solid #31506c;border-radius:6px;background:#0c1722;padding:8px;";
    const durationControl = (caption, current, minimum, maximum, step) => {
        const label = document.createElement("label");
        label.style.cssText = "display:flex;align-items:center;gap:7px;color:#c4d5e8;font-size:10px;";
        const fixed = document.createElement("input");
        fixed.type = "checkbox";
        const text = document.createElement("span");
        text.textContent = caption;
        label.append(fixed, text);
        const value = document.createElement("input");
        value.type = "number";
        value.min = String(minimum);
        value.max = String(maximum);
        value.step = String(step);
        value.value = String(current);
        value.style.cssText = "min-width:0;border:1px solid #34465b;border-radius:5px;background:#0b1118;color:#ffd866;padding:6px 7px;";
        durationBox.append(label, value);
        return {fixed, value};
    };
    const totalDuration = transferring ? null : durationControl(
        "固定目标总时长", Math.max(1, Number(widget(node, "total_seconds")?.value)
            || timelineStats(node).seconds || 5), 1, 3600, 1);
    if (transferring) {
        const automaticDuration = document.createElement("div");
        automaticDuration.textContent = "目标总时长：由动作参考视频自动解析（参考视频为必选项）";
        automaticDuration.style.cssText = "grid-column:1/-1;color:#86efac;font-size:10px;line-height:1.5;";
        durationBox.appendChild(automaticDuration);
    }
    const segmentDuration = durationControl(
        "固定智能分段上限", Math.max(.2, Number(widget(node, "segment_seconds")?.value) || 5),
        .2, 30, .1);
    const lockLabel = document.createElement("label");
    lockLabel.style.cssText = "display:flex;align-items:flex-start;gap:7px;color:#c4b5fd;font-size:10px;line-height:1.45;";
    const interfaceLocked = document.createElement("input");
    interfaceLocked.type = "checkbox";
    interfaceLocked.checked = true;
    lockLabel.append(interfaceLocked,
        "锁定为接口模板：使用者只填写下方未固定的提示词/素材/时长，应用后不能改固定内容");
    const help = document.createElement("div");
    help.textContent = transferring
        ? "动作迁移不使用独立公共素材桶；下方展示当前动作视频、人物图与音频。参考权重会带入当前值，可在保存前调整。素材不固定时就是模板输入，可把用途写成“所需替换的角色/背景”等，系统会按界面顺序自动编号。"
        : "每个提示词和素材都能独立决定固定或输入。素材不固定时可编辑输入用途，应用模板时会连续显示为“输入1、输入2…”；素材行保留原内容预览。";
    help.style.cssText = "border:1px solid #31506c;border-radius:5px;background:#0d1c29;padding:7px;color:#9bc5ea;font-size:9px;line-height:1.5;";
    const choices = document.createElement("div");
    choices.style.cssText = transferring
        ? "min-height:340px;max-height:min(54vh,560px);overflow:auto;display:flex;flex-direction:column;gap:7px;padding-right:3px;"
        : "max-height:390px;overflow:auto;display:flex;flex-direction:column;gap:7px;padding-right:3px;";
    const promptModes = {};
    const materialModes = {};
    const promptInputLabels = {};
    const materialInputLabels = {};
    const interfaceRows = [];
    const refreshInterfaceNumbers = () => {
        let index = 0;
        for (const entry of interfaceRows) {
            const active = entry?.isInput?.() === true;
            entry?.setIndex?.(active ? ++index : 0);
        }
    };
    const toggle = (textValue, checked, change) => {
        const label = document.createElement("label");
        label.style.cssText = "display:flex;align-items:center;gap:6px;min-width:0;color:#aebdcd;font-size:9px;";
        const input = document.createElement("input"); input.type = "checkbox"; input.checked = checked;
        const caption = document.createElement("span"); caption.textContent = textValue;
        caption.style.cssText = "overflow:hidden;text-overflow:ellipsis;white-space:nowrap;";
        input.onchange = () => change(input.checked);
        change(checked);
        label.append(input, caption);
        return label;
    };
    const globalGroup = document.createElement("div");
    globalGroup.style.cssText = "border:1px solid #2d3d50;border-radius:6px;padding:7px;";
    const globalTitle = document.createElement("b");
    globalTitle.textContent = "公共素材";
    globalTitle.style.cssText = "display:block;margin-bottom:5px;font-size:10px;color:#93c5fd;";
    globalGroup.appendChild(globalTitle);
    if (!(source.globalAssets || []).length) globalGroup.append("无公共素材");
    // Action transfer keeps every asset on its single action-source card. An
    // empty global group used to imply that the uploaded action media vanished.
    if (!transferring) {
        for (const asset of source.globalAssets || []) {
            const key = `global:${asset.id}`;
            const row = templateMaterialChoice(asset, false,
                (fixed) => { materialModes[key] = fixed ? "fixed" : "input"; }, {
                    onInputLabel: (value) => { materialInputLabels[key] = value; },
                    onModeChange: refreshInterfaceNumbers,
                });
            interfaceRows.push(row.__myangTemplateInput);
            globalGroup.appendChild(row);
        }
        choices.appendChild(globalGroup);
    }
    for (const [index, shot] of editableShots.entries()) {
        const group = document.createElement("div");
        group.style.cssText = `border:1px solid #2d3d50;border-radius:6px;padding:7px;${transferring ? "min-height:310px;" : ""}`;
        const title = document.createElement("b");
        title.textContent = transferring
            ? "动作源与辅助素材（当前模式）"
            : `${index + 1}. ${shot.brief || "未命名分镜"}`;
        title.style.cssText = "display:block;margin-bottom:5px;font-size:10px;color:#dbeafe;";
        const promptEditor = document.createElement("textarea");
        promptEditor.rows = 5;
        promptEditor.value = String(shot.prompt || "");
        promptEditor.placeholder = "模板固定的提示词内容";
        promptEditor.style.cssText = "box-sizing:border-box;width:100%;min-height:84px;max-height:190px;margin-top:6px;resize:vertical;border:1px solid #3c4e63;border-radius:5px;background:#080d13;color:#dbeafe;padding:7px;line-height:1.45;";
        const promptPurposeRow = document.createElement("label");
        promptPurposeRow.style.cssText = "display:grid;grid-template-columns:auto minmax(0,1fr);gap:6px;align-items:center;margin-top:5px;color:#8fa5ba;font-size:9px;";
        const promptPurposeCaption = document.createElement("span");
        promptPurposeCaption.textContent = "固定提示词";
        const promptPurpose = document.createElement("input");
        promptPurpose.type = "text";
        promptPurpose.maxLength = 120;
        promptPurpose.value = transferring
            ? "动作迁移提示词"
            : `所需替换的${shot.brief || `分镜${index + 1}`}提示词`;
        promptPurpose.style.cssText = "box-sizing:border-box;min-width:0;width:100%;border:1px solid #34465b;border-radius:4px;background:#080d13;color:#cfe4f7;padding:4px 6px;font-size:9px;";
        promptInputLabels[shot.id] = promptPurpose.value;
        promptPurpose.oninput = () => { promptInputLabels[shot.id] = promptPurpose.value.trim(); };
        promptPurposeRow.append(promptPurposeCaption, promptPurpose);
        const promptInterface = {
            isInput: () => promptModes[shot.id] === "input",
            setIndex: (inputIndex) => {
                promptPurposeCaption.textContent = inputIndex > 0 ? `输入${inputIndex}用途` : "固定提示词";
                promptPurpose.disabled = inputIndex <= 0;
                promptPurposeRow.style.opacity = inputIndex > 0 ? "1" : ".42";
            },
        };
        interfaceRows.push(promptInterface);
        const promptToggle = toggle("固定当前提示词（下方可查看和修改）", true,
            (fixed) => {
                promptModes[shot.id] = fixed ? "fixed" : "input";
                promptEditor.disabled = !fixed;
                promptEditor.style.opacity = fixed ? "1" : ".45";
                refreshInterfaceNumbers();
            });
        promptEditor.oninput = () => { shot.prompt = promptEditor.value; };
        group.append(title, promptToggle, promptPurposeRow, promptEditor);
        for (const asset of shot.assets || []) {
            const key = `shot:${shot.id}:${asset.id}`;
            const row = templateMaterialChoice(asset, false,
                (fixed) => { materialModes[key] = fixed ? "fixed" : "input"; }, {
                    onInputLabel: (value) => { materialInputLabels[key] = value; },
                    onModeChange: refreshInterfaceNumbers,
                    showReferenceWeight: transferring,
                });
            interfaceRows.push(row.__myangTemplateInput);
            group.appendChild(row);
        }
        choices.appendChild(group);
    }
    refreshInterfaceNumbers();
    const status = document.createElement("div");
    status.style.cssText = "min-height:16px;color:#fbbf24;font-size:9px;";
    const actions = document.createElement("div");
    actions.style.cssText = "display:flex;justify-content:flex-end;gap:7px;";
    const cancel = button("取消"); cancel.onclick = () => overlay.remove();
    const save = button("保存模板");
    save.style.cssText += ";min-width:96px;min-height:34px;background:#2563a8;color:#fff;";
    save.onclick = async () => {
        save.disabled = true;
        try {
            const documentData = createDirectorTemplateDocument({
                name: name.value, description: description.value,
                shots: editableShots, globalAssets: source.globalAssets,
                settings: settingsValues, settingModes,
                promptModes, promptInputLabels, materialModes, materialInputLabels, taskMode,
                durationPolicy: {
                    total: transferring ? {mode: "reference", value: 0} : {
                        mode: totalDuration.fixed.checked ? "fixed" : "input",
                        value: Number(totalDuration.value.value),
                    },
                    segment: {mode: segmentDuration.fixed.checked ? "fixed" : "input",
                        value: Number(segmentDuration.value.value)},
                },
                interfaceLocked: interfaceLocked.checked,
            });
            const result = await directorTemplateRequest("", {
                method: "POST", headers: {"Content-Type": "application/json"},
                body: JSON.stringify(documentData),
            });
            node.__myangDirectorTemplates = [
                ...(node.__myangDirectorTemplates || []).filter((item) => item.id !== result.template.id),
                result.template,
            ];
            node.__myangSelectedTemplateByTask ||= {};
            node.__myangSelectedTemplateByTask[taskMode] = result.template.id;
            node.__myangDirectorTemplatesLoaded = true;
            node.__myangDirectorTemplatesLoadedAt = Date.now();
            node.__myangTemplateNotice = `模板「${result.template.name}」已保存`;
            overlay.remove();
            renderTimeline(node);
        } catch (error) {
            status.textContent = `保存失败：${error.message}`;
            save.disabled = false;
        }
    };
    actions.append(cancel, save);
    body.append(name, description, durationBox, settingsBox, lockLabel,
        help, choices, status, actions);
    name.focus(); name.select();
}

function chooseTemplateLocalMaterial(slot, onReady) {
    const picker = document.createElement("input");
    picker.type = "file";
    picker.accept = MEDIA_META[slot.kind]?.accept || "*/*";
    picker.style.display = "none";
    document.body.appendChild(picker);
    picker.onchange = async () => {
        try {
            const selected = picker.files?.[0];
            if (!selected) return;
            const file = await uploadShotFile({id: `template_${slot.slot_id}`}, selected);
            await onReady({kind: slot.kind, label: selected.name, file});
        } finally { picker.remove(); }
    };
    picker.click();
}

function openApplyDirectorTemplateDialog(node, template) {
    const {overlay, body} = materialPickerShell("myang-apply-template-dialog", `使用模板 · ${template.name}`);
    body.style.cssText += ";display:flex;flex-direction:column;gap:8px;";
    const slots = directorTemplateInputSlots(template);
    const prompts = {}, materials = {}, durations = {};
    const slotViews = new Map();
    const description = document.createElement("div");
    description.textContent = template.description || "填写模板要求的内容后，会生成可继续编辑的导演台分镜卡。";
    description.style.cssText = "color:#94a3b8;font-size:10px;line-height:1.5;";
    body.appendChild(description);
    const policy = template.settings_policy && typeof template.settings_policy === "object"
        ? template.settings_policy : {};
    const fixedSettings = Object.entries(template.settings || {}).filter(([setting]) =>
        Object.keys(policy).length ? policy[setting] === "fixed" : template.settings_fixed === true);
    if (fixedSettings.length) {
        const fixedBox = document.createElement("details");
        fixedBox.style.cssText = "border:1px solid #31506c;border-radius:6px;background:#0c1722;padding:7px;";
        const summary = document.createElement("summary");
        summary.textContent = `本模板固定 ${fixedSettings.length} 项生成参数（点击查看）`;
        summary.style.cssText = "cursor:pointer;color:#bae6fd;font-size:10px;font-weight:700;";
        const values = document.createElement("div");
        values.style.cssText = "display:grid;grid-template-columns:minmax(120px,.8fr) minmax(100px,1.2fr);gap:4px 8px;margin-top:7px;color:#9fb1c3;font-size:9px;";
        for (const [setting, value] of fixedSettings) {
            const label = document.createElement("span");
            label.textContent = setting === "resolution" ? "一采分辨率"
                : setting === "二采分辨率" ? "二采分辨率" : (LABELS[setting] || setting);
            const shown = document.createElement("code");
            shown.textContent = String(value);
            shown.style.color = "#fde68a";
            values.append(label, shown);
        }
        fixedBox.append(summary, values);
        body.appendChild(fixedBox);
    }
    for (const slot of slots) {
        const row = document.createElement("div");
        row.style.cssText = "border:1px solid #2d3d50;border-radius:6px;background:#0b121b;padding:8px;";
        const title = document.createElement("b");
        title.textContent = slot.label || slot.slot_id;
        title.style.cssText = "display:block;margin-bottom:6px;color:#cfe8ff;font-size:10px;";
        row.appendChild(title);
        if (slot.type === "duration") {
            const input = document.createElement("input");
            input.type = "number";
            input.min = String(slot.min || .2);
            input.max = String(slot.max || 3600);
            input.step = String(slot.step || .1);
            input.value = String(Math.max(Number(slot.min || .2),
                Number(widget(node, slot.setting)?.value)
                    || (slot.setting === "total_seconds" ? timelineStats(node).seconds : 5)));
            input.style.cssText = "width:100%;box-sizing:border-box;border:1px solid #34465b;border-radius:5px;background:#101923;color:#ffd866;padding:7px;";
            const sync = () => { durations[slot.slot_id] = Number(input.value); };
            input.oninput = sync;
            sync();
            row.appendChild(input);
        } else if (slot.type === "prompt") {
            const input = document.createElement("textarea");
            input.rows = 4;
            input.placeholder = "填写本分镜提示词";
            input.style.cssText = "width:100%;box-sizing:border-box;border:1px solid #34465b;border-radius:5px;background:#101923;color:#e5edf5;padding:7px;resize:vertical;";
            input.oninput = () => { prompts[slot.slot_id] = input.value; };
            row.appendChild(input);
        } else {
            const state = document.createElement("span");
            state.textContent = `需要${MEDIA_META[slot.kind]?.label || "素材"}`;
            state.style.cssText = "color:#fbbf24;font-size:9px;";
            const select = button("选择素材", "从素材库或本地选择");
            select.style.marginLeft = "8px";
            const ready = async (asset) => {
                materials[slot.slot_id] = asset;
                state.textContent = `已选择：${asset.label || asset.file?.name}`;
                state.style.color = "#86efac";
            };
            select.onclick = () => openMaterialSourceDialog(node, {
                kind: slot.kind,
                local: () => chooseTemplateLocalMaterial(slot, ready),
                library: () => openMaterialLibraryChooser(node, {kind: slot.kind, onSelect: ready}),
            });
            row.append(state, select);
            slotViews.set(slot.slot_id, state);
        }
        body.appendChild(row);
    }
    if (!slots.length) {
        const ready = document.createElement("div");
        ready.textContent = "这个模板没有运行时输入，应用后可直接生成。";
        ready.style.cssText = "color:#86efac;font-size:10px;";
        body.appendChild(ready);
    }
    const status = document.createElement("div");
    status.style.cssText = "min-height:16px;color:#fbbf24;font-size:9px;";
    const actions = document.createElement("div");
    actions.style.cssText = "display:flex;justify-content:flex-end;gap:7px;";
    const cancel = button("取消"); cancel.onclick = () => overlay.remove();
    const apply = button("生成分镜卡");
    apply.style.cssText += ";min-width:110px;min-height:34px;background:#347a4a;color:#fff;";
    apply.onclick = () => {
        try {
            const result = instantiateDirectorTemplate(template, {prompts, materials, durations});
            const targetTask = DIRECTOR_TASK_MODES.includes(result.taskMode)
                ? result.taskMode : FRESH;
            // Switch the mode bucket before replacing its content. Otherwise
            // applying an action template while viewing another mode writes
            // the new action source into the previous mode's private bucket.
            setNativeWidget(node, "task_mode", targetTask);
            syncModeBucket(node);
            if (targetTask !== TRANSFER) setNativeWidget(node, "source_mode", MANUAL);
            node.__myangDirectorShots = result.shots.map((shot, index) => ({
                ...shot,
                assets: shot.assets.map((asset, assetIndex) => normalizeAsset(asset, assetIndex)),
            }));
            node.__myangDirectorGlobals = result.globalAssets.map((asset, index) => normalizeAsset(asset, index));
            for (const [name, value] of Object.entries(result.settings || {})) setNativeWidget(node, name, value);
            node.__myangTemplateLock = result.interfaceLocked ? {
                enabled: true, template_id: result.templateId,
                template_name: result.templateName || template.name,
            } : null;
            node.__myangTemplateContract = result.durationContract;
            node.__myangTemplateNotice = result.interfaceLocked
                ? `已应用接口模板「${template.name}」；固定内容已锁定，重新点“使用模板”可更换输入`
                : `已应用模板「${template.name}」；标题与提示词保持分离`;
            saveTimeline(node);
            overlay.remove();
            renderTimeline(node);
        } catch (error) { status.textContent = error.message; }
    };
    actions.append(cancel, apply);
    body.append(status, actions);
}

function safeTemplateFileName(name, encrypted = false) {
    const clean = String(name || "Myang_导演台模板")
        .replace(/[\\/:*?"<>|]+/g, "_").slice(0, 80);
    return `${clean}.${encrypted ? "encrypted." : ""}myang-template.json`;
}

function cloneTemplateDocument(value) {
    return JSON.parse(JSON.stringify(value));
}

function templateFixedMaterials(documentData) {
    const result = [];
    const append = (materials) => {
        for (const material of Array.isArray(materials) ? materials : []) {
            if (material?.mode === "fixed" && material.file?.name && material.slot_id) {
                result.push(material);
            }
        }
    };
    append(documentData?.global_materials);
    for (const card of documentData?.cards || []) append(card?.materials);
    return result;
}

function templateBytesToBase64(bytes) {
    let binary = "";
    for (let index = 0; index < bytes.length; index += 0x8000) {
        binary += String.fromCharCode(...bytes.subarray(index, index + 0x8000));
    }
    return btoa(binary);
}

function templateBase64ToBytes(value) {
    const binary = atob(String(value || ""));
    return Uint8Array.from(binary, (character) => character.charCodeAt(0));
}

async function buildEmbeddedTemplateDocument(documentData, status = null) {
    const source = cloneTemplateDocument(documentData);
    const assets = {};
    let totalBytes = 0;
    const materials = templateFixedMaterials(source);
    if (!materials.length) return source;
    for (const [index, material] of materials.entries()) {
        status && (status.textContent = `正在打包固定素材 ${index + 1}/${materials.length}：${material.label || material.file.name}`);
        const response = await fetch(assetViewUrl(material));
        if (!response.ok) throw new Error(`固定素材「${material.label || material.file.name}」无法读取（HTTP ${response.status}）`);
        const blob = await response.blob();
        if (blob.size > TEMPLATE_BUNDLE_MAX_ASSET_BYTES) {
            throw new Error(`固定素材「${material.label || material.file.name}」超过 ${Math.round(TEMPLATE_BUNDLE_MAX_ASSET_BYTES / 1048576)} MB，无法嵌入`);
        }
        totalBytes += blob.size;
        if (totalBytes > TEMPLATE_BUNDLE_MAX_BYTES) {
            throw new Error(`固定素材总大小超过 ${Math.round(TEMPLATE_BUNDLE_MAX_BYTES / 1048576)} MB，请减少素材或取消便携导出`);
        }
        const bytes = new Uint8Array(await blob.arrayBuffer());
        assets[String(material.slot_id)] = {
            encoding: "base64",
            mime: blob.type || (material.kind === "image" ? "image/png" : material.kind === "audio" ? "audio/mpeg" : "video/mp4"),
            name: material.file.name,
            size: bytes.byteLength,
            data: templateBytesToBase64(bytes),
        };
    }
    source.bundle = {version: TEMPLATE_BUNDLE_VERSION, assets};
    return source;
}

function replaceTemplateMaterialFile(documentData, slotId, fileRef) {
    let replaced = false;
    const replace = (materials) => {
        for (const material of Array.isArray(materials) ? materials : []) {
            if (String(material?.slot_id) !== String(slotId)) continue;
            material.file = {...fileRef, type: "input"};
            material.mode = "fixed";
            replaced = true;
        }
    };
    replace(documentData?.global_materials);
    for (const card of documentData?.cards || []) replace(card?.materials);
    return replaced;
}

async function importEmbeddedTemplateAssets(documentData, status = null) {
    const bundle = documentData?.bundle;
    if (!bundle) return documentData;
    if (Number(bundle.version) !== TEMPLATE_BUNDLE_VERSION
        || !bundle.assets || typeof bundle.assets !== "object") {
        throw new Error("模板内的素材包版本不受支持");
    }
    const source = cloneTemplateDocument(documentData);
    delete source.bundle;
    const entries = Object.entries(bundle.assets);
    let totalBytes = 0;
    for (const [index, [slotId, record]] of entries.entries()) {
        if (!record || record.encoding !== "base64" || typeof record.data !== "string") {
            throw new Error(`模板素材包中的第 ${index + 1} 项数据损坏`);
        }
        const bytes = templateBase64ToBytes(record.data);
        totalBytes += bytes.byteLength;
        if (bytes.byteLength > TEMPLATE_BUNDLE_MAX_ASSET_BYTES
            || totalBytes > TEMPLATE_BUNDLE_MAX_BYTES) {
            throw new Error("模板内嵌素材超过允许大小，已停止导入");
        }
        const name = String(record.name || `template-${index + 1}`)
            .replace(/[\\/:*?"<>|]+/g, "_").slice(0, 180);
        status && (status.textContent = `正在恢复固定素材 ${index + 1}/${entries.length}：${name}`);
        const file = new File([bytes], name, {type: String(record.mime || "application/octet-stream")});
        const uploaded = await uploadShotFile({id: `template_bundle_${slotId}`}, file);
        if (!replaceTemplateMaterialFile(source, slotId, uploaded)) {
            throw new Error(`模板素材槽「${slotId}」不存在，无法恢复`);
        }
    }
    return source;
}

function downloadTemplateDocument(documentData, name, encrypted = false) {
    const blob = new Blob([JSON.stringify(documentData, null, 2)], {
        type: "application/json;charset=utf-8",
    });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = safeTemplateFileName(name, encrypted);
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    setTimeout(() => URL.revokeObjectURL(url), 0);
}

function requestTemplatePassword(titleText, confirmation = false) {
    return new Promise((resolve) => {
        const {overlay, body} = materialPickerShell(
            "myang-template-password-dialog", titleText);
        body.style.cssText += ";display:flex;flex-direction:column;gap:8px;";
        const password = document.createElement("input");
        password.type = "password";
        password.placeholder = "模板密码（至少 8 个字符）";
        password.autocomplete = "new-password";
        password.style.cssText = "min-height:36px;border:1px solid #34465b;border-radius:5px;background:#0b1118;color:#e5edf5;padding:6px 8px;";
        const repeated = document.createElement("input");
        repeated.type = "password";
        repeated.placeholder = "再次输入密码";
        repeated.style.cssText = password.style.cssText;
        const note = document.createElement("div");
        note.textContent = confirmation
            ? "AES-GCM 会保护导出文件，但无法阻止本机管理员检查正在运行的本地工作流。"
            : "密码只用于本次本地解密，不会上传或保存。";
        note.style.cssText = "color:#94a3b8;font-size:9px;line-height:1.5;";
        const status = document.createElement("div");
        status.style.cssText = "min-height:15px;color:#fbbf24;font-size:9px;";
        const actions = document.createElement("div");
        actions.style.cssText = "display:flex;justify-content:flex-end;gap:7px;";
        const cancel = button("取消");
        const accept = button(confirmation ? "加密导出" : "解密导入");
        accept.style.cssText += ";min-width:96px;min-height:34px;background:#2563a8;color:#fff;";
        let settled = false;
        const finish = (value) => {
            if (settled) return;
            settled = true;
            overlay.__myangOnClose = null;
            overlay.remove();
            resolve(value);
        };
        overlay.__myangOnClose = () => finish("");
        cancel.onclick = () => finish("");
        accept.onclick = () => {
            if (password.value.length < 8) {
                status.textContent = "密码至少需要 8 个字符";
                return;
            }
            if (confirmation && password.value !== repeated.value) {
                status.textContent = "两次密码不一致";
                return;
            }
            finish(password.value);
        };
        actions.append(cancel, accept);
        body.append(password);
        if (confirmation) body.append(repeated);
        body.append(note, status, actions);
        password.focus();
    });
}

function openExportDirectorTemplateDialog(node, template) {
    const {overlay, body} = materialPickerShell(
        "myang-export-template-dialog", `导出模板 · ${template.name}`);
    body.style.cssText += ";display:flex;flex-direction:column;gap:9px;";
    const encryptionLabel = document.createElement("label");
    encryptionLabel.style.cssText = "display:flex;align-items:flex-start;gap:7px;color:#dbeafe;font-size:10px;line-height:1.5;";
    const encrypted = document.createElement("input");
    encrypted.type = "checkbox";
    encryptionLabel.append(encrypted,
        "密码加密导出（AES-GCM）：文件中不出现明文提示词与固定素材引用");
    const bundleLabel = document.createElement("label");
    bundleLabel.style.cssText = "display:flex;align-items:flex-start;gap:7px;color:#dbeafe;font-size:10px;line-height:1.5;";
    const bundle = document.createElement("input");
    bundle.type = "checkbox";
    bundleLabel.append(bundle, "嵌入固定素材（便携模板，最大 128 MB）");
    const note = document.createElement("div");
    note.textContent = "普通导出只保存素材引用；勾选便携模板后会把固定图片、视频和音频一起写入文件，导入时自动恢复。加密导出保护传输/备份文件。";
    note.style.cssText = "border:1px solid #31506c;border-radius:5px;background:#0d1c29;padding:7px;color:#9bc5ea;font-size:9px;line-height:1.5;";
    const actions = document.createElement("div");
    actions.style.cssText = "display:flex;justify-content:flex-end;gap:7px;";
    const cancel = button("取消");
    cancel.onclick = () => overlay.remove();
    const save = button("导出文件");
    save.style.cssText += ";min-width:96px;min-height:34px;background:#2563a8;color:#fff;";
    save.onclick = async () => {
        save.disabled = true;
        try {
            const source = parseDirectorTemplateDocument(template);
            const exportSource = bundle.checked
                ? await buildEmbeddedTemplateDocument(source, note) : source;
            if (!encrypted.checked) {
                downloadTemplateDocument(exportSource, exportSource.name, false);
            } else {
                const password = await requestTemplatePassword("设置模板导出密码", true);
                if (!password) {
                    save.disabled = false;
                    return;
                }
                downloadTemplateDocument(
                    await encryptDirectorTemplateDocument(exportSource, password),
                    exportSource.name, true);
            }
            node.__myangTemplateNotice = `模板「${source.name}」已导出`;
            overlay.remove();
            renderTimeline(node);
        } catch (error) {
            note.textContent = `导出失败：${error.message}`;
            save.disabled = false;
        }
    };
    actions.append(cancel, save);
    body.append(encryptionLabel, bundleLabel, note, actions);
}

function chooseDirectorTemplateFile(node, onImported) {
    const picker = document.createElement("input");
    picker.type = "file";
    picker.accept = ".json,.myang-template.json,application/json";
    picker.onchange = async () => {
        const file = picker.files?.[0];
        picker.remove();
        if (!file) return;
        try {
            if (file.size > TEMPLATE_BUNDLE_MAX_FILE_BYTES) {
                throw new Error(`模板文件超过 ${Math.round(TEMPLATE_BUNDLE_MAX_FILE_BYTES / 1048576)} MB`);
            }
            const raw = JSON.parse(await file.text());
            let documentData;
            if (raw?.format === DIRECTOR_TEMPLATE_ENCRYPTED_FORMAT) {
                const password = await requestTemplatePassword("输入模板解密密码");
                if (!password) return;
                documentData = await decryptDirectorTemplateDocument(raw, password);
            } else {
                documentData = parseDirectorTemplateDocument(raw);
            }
            documentData = await importEmbeddedTemplateAssets(documentData);
            const result = await directorTemplateRequest("", {
                method: "POST", headers: {"Content-Type": "application/json"},
                body: JSON.stringify(documentData),
            });
            node.__myangDirectorTemplates = [
                ...(node.__myangDirectorTemplates || []).filter(
                    (item) => item.id !== result.template.id),
                result.template,
            ];
            node.__myangSelectedTemplateByTask ||= {};
            node.__myangSelectedTemplateByTask[result.template.task_mode] = result.template.id;
            node.__myangDirectorTemplatesLoaded = true;
            node.__myangDirectorTemplatesLoadedAt = Date.now();
            node.__myangTemplateNotice = `已导入模板「${result.template.name}」`;
            onImported?.();
            renderTimeline(node);
        } catch (error) {
            node.__myangTemplateNotice = `模板导入失败：${error.message}`;
            renderTimeline(node);
        }
    };
    document.body.appendChild(picker);
    picker.click();
}

function templateRepositoryFixedSettings(template) {
    const policy = template?.settings_policy && typeof template.settings_policy === "object"
        ? template.settings_policy : {};
    return Object.entries(template?.settings || {}).filter(([name]) =>
        Object.keys(policy).length ? policy[name] === "fixed" : template.settings_fixed === true);
}

function templateRepositoryMaterialRow(material) {
    const row = document.createElement("div");
    row.style.cssText = "display:grid;grid-template-columns:58px minmax(0,1fr) auto;gap:8px;align-items:center;border:1px solid #293b4e;border-radius:5px;background:#09111a;padding:6px;";
    const icon = document.createElement("div");
    icon.style.cssText = "display:grid;place-items:center;width:58px;height:42px;overflow:hidden;border:1px solid #344b61;border-radius:4px;background:#05090e;color:#93c5fd;font-size:9px;";
    const fixed = material?.mode === "fixed" && material?.file?.name;
    if (fixed && material.kind === "image") {
        const image = document.createElement("img");
        image.src = assetViewUrl(material);
        image.alt = "";
        image.loading = "lazy";
        image.style.cssText = "width:100%;height:100%;object-fit:contain;";
        icon.appendChild(image);
    } else {
        icon.textContent = MEDIA_META[material?.kind]?.label || material?.kind || "素材";
    }
    const info = document.createElement("div");
    info.style.cssText = "min-width:0;display:flex;flex-direction:column;gap:3px;";
    const title = document.createElement("b");
    title.textContent = material?.label || material?.input_label || "未命名素材";
    title.style.cssText = "overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:#dbeafe;font-size:10px;";
    const detail = document.createElement("span");
    detail.textContent = fixed
        ? `${material.file.subfolder ? `${material.file.subfolder}/` : ""}${material.file.name}`
        : `运行时输入 · ${material?.input_label || material?.label || "素材"}`;
    detail.title = detail.textContent;
    detail.style.cssText = "overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:#7f93a8;font-size:9px;";
    info.append(title, detail);
    const action = fixed ? button("预览", "打开模板固定素材") : document.createElement("span");
    if (fixed) action.onclick = () => openAssetPreview(material, action);
    else {
        action.textContent = material?.required === false ? "可选输入" : "必填输入";
        action.style.cssText = "color:#c4b5fd;font-size:9px;white-space:nowrap;";
    }
    row.append(icon, info, action);
    return row;
}

function openDirectorTemplateRepository(node) {
    const {overlay, card, body} = materialPickerShell(
        "myang-template-repository-dialog", "本地模板仓库", {closeOnBackdrop: false});
    card.style.width = "min(96vw,1160px)";
    card.style.maxHeight = "min(92vh,860px)";
    body.style.cssText = "min-height:0;overflow:hidden;padding:10px;display:grid;grid-template-columns:270px minmax(0,1fr);grid-template-rows:auto minmax(0,1fr);gap:9px;";

    const toolbar = document.createElement("div");
    toolbar.style.cssText = "grid-column:1/-1;display:flex;align-items:center;gap:7px;flex-wrap:wrap;";
    const search = document.createElement("input");
    search.type = "search";
    search.placeholder = "搜索模板名称、说明或分镜";
    search.style.cssText = "flex:1 1 260px;min-height:34px;border:1px solid #344b61;border-radius:5px;background:#09111a;color:#e5edf5;padding:6px 8px;";
    const mode = document.createElement("select");
    mode.style.cssText = "min-height:34px;min-width:170px;border:1px solid #344b61;border-radius:5px;background:#09111a;color:#dbeafe;padding:5px;";
    mode.append(new Option("全部模式", ""));
    for (const value of [FRESH, TRANSFER, CONTINUE]) mode.append(new Option(value, value));
    const create = button("创建当前模板");
    const importTemplate = button("导入模板");
    const reload = button("刷新");
    for (const control of [create, importTemplate, reload]) {
        control.style.cssText += ";min-height:34px;white-space:nowrap;";
    }
    toolbar.append(search, mode, create, importTemplate, reload);

    const list = document.createElement("div");
    list.style.cssText = "min-height:0;overflow:auto;border:1px solid #2e4155;border-radius:7px;background:#0a1119;padding:6px;";
    const details = document.createElement("div");
    details.style.cssText = "min-width:0;min-height:0;overflow:auto;border:1px solid #2e4155;border-radius:7px;background:#0a1119;padding:10px;";
    body.append(toolbar, list, details);

    let templates = [];
    let selectedId = node.__myangSelectedTemplateByTask?.[currentTask(node)] || "";
    const fixedMaterials = (template) => [
        ...(template?.global_materials || []),
        ...(template?.cards || []).flatMap((item) => item?.materials || []),
    ].filter((item) => item?.mode === "fixed");
    const inputCount = (template) => {
        try { return directorTemplateInputSlots(template).length; }
        catch (_error) { return 0; }
    };
    const durationText = (rule, label) => {
        if (!rule) return `${label}：沿用本地`;
        if (rule.mode === "reference") return `${label}：参考视频自动解析`;
        if (rule.mode === "input") return `${label}：使用时输入`;
        return `${label}：固定 ${Number(rule.value || 0)} 秒`;
    };
    const renderDetails = (template) => {
        details.replaceChildren();
        if (!template) {
            const empty = document.createElement("div");
            empty.textContent = templates.length ? "选择左侧模板查看详细内容" : "本地模板仓库还没有模板";
            empty.style.cssText = "display:grid;place-items:center;min-height:240px;color:#708398;";
            details.appendChild(empty);
            return;
        }
        const heading = document.createElement("div");
        heading.style.cssText = "display:flex;align-items:flex-start;gap:8px;";
        const titleBox = document.createElement("div");
        titleBox.style.cssText = "min-width:0;flex:1;";
        const title = document.createElement("h3");
        title.textContent = template.name;
        title.style.cssText = "margin:0;color:#e8f3ff;font-size:15px;";
        const meta = document.createElement("div");
        const updated = Number(template.updated || template.created || 0);
        meta.textContent = `${template.task_mode || FRESH}${updated ? ` · 更新于 ${new Date(updated * 1000).toLocaleString()}` : ""}`;
        meta.style.cssText = "margin-top:4px;color:#8195aa;font-size:9px;";
        titleBox.append(title, meta);
        const use = button("使用模板");
        const exportTemplate = button("导出");
        const remove = button("删除");
        remove.style.color = "#fb7185";
        heading.append(titleBox, use, exportTemplate, remove);
        details.appendChild(heading);

        const description = document.createElement("div");
        description.textContent = template.description || "暂无模板用途说明";
        description.style.cssText = "margin-top:9px;border:1px solid #2d4155;border-radius:5px;background:#0d1722;padding:8px;color:#9fb1c3;font-size:10px;line-height:1.5;white-space:pre-wrap;";
        details.appendChild(description);

        const settings = templateRepositoryFixedSettings(template);
        const stats = document.createElement("div");
        stats.style.cssText = "display:grid;grid-template-columns:repeat(4,minmax(90px,1fr));gap:6px;margin-top:8px;";
        const transferring = template.task_mode === TRANSFER;
        for (const [label, value] of [
            [transferring ? "模板内容" : "分镜",
                transferring ? "动作迁移" : (template.cards || []).length],
            ["开放输入", inputCount(template)],
            ["固定参数", settings.length], ["固定素材", fixedMaterials(template).length],
        ]) {
            const item = document.createElement("div");
            item.style.cssText = "border:1px solid #2e4358;border-radius:5px;background:#0c1823;padding:7px;color:#91a5ba;font-size:9px;";
            item.innerHTML = `<b style="display:block;color:#dbeafe;font-size:13px">${value}</b>${label}`;
            stats.appendChild(item);
        }
        details.appendChild(stats);

        const duration = document.createElement("div");
        duration.textContent = `${durationText(template.duration_policy?.total, "总时长")} · ${durationText(template.duration_policy?.segment, "分段上限")} · ${template.interface_locked ? "接口已锁定" : "应用后可编辑"}`;
        duration.style.cssText = "margin-top:8px;color:#93c5fd;font-size:9px;line-height:1.5;";
        details.appendChild(duration);

        let slots = [];
        try { slots = directorTemplateInputSlots(template); } catch (_error) { /* shown as zero */ }
        if (slots.length) {
            const inputsBox = document.createElement("details");
            inputsBox.open = true;
            const summary = document.createElement("summary");
            summary.textContent = `使用时需要填写 · ${slots.length} 项`;
            summary.style.cssText = "cursor:pointer;margin-top:9px;color:#c4b5fd;font-weight:700;font-size:10px;";
            const rows = document.createElement("div");
            rows.style.cssText = "display:flex;flex-direction:column;gap:4px;margin-top:6px;";
            for (const slot of slots) {
                const row = document.createElement("div");
                row.textContent = `${slot.label || slot.slot_id} · ${slot.type === "prompt" ? "提示词" : slot.type === "duration" ? "时长" : MEDIA_META[slot.kind]?.label || "素材"}`;
                row.style.cssText = "border-left:2px solid #8b5cf6;background:#151329;padding:5px 7px;color:#d8ccff;font-size:9px;";
                rows.appendChild(row);
            }
            inputsBox.append(summary, rows);
            details.appendChild(inputsBox);
        }
        if (settings.length) {
            const box = document.createElement("details");
            const summary = document.createElement("summary");
            summary.textContent = `固定生成参数 · ${settings.length} 项`;
            summary.style.cssText = "cursor:pointer;margin-top:9px;color:#bae6fd;font-weight:700;font-size:10px;";
            const grid = document.createElement("div");
            grid.style.cssText = "display:grid;grid-template-columns:minmax(130px,.8fr) minmax(100px,1.2fr);gap:4px 8px;margin-top:6px;font-size:9px;";
            for (const [name, value] of settings) {
                const label = document.createElement("span");
                label.textContent = LABELS[name] || name;
                const shown = document.createElement("code");
                shown.textContent = String(value);
                shown.style.color = "#fde68a";
                grid.append(label, shown);
            }
            box.append(summary, grid);
            details.appendChild(box);
        }
        if ((template.global_materials || []).length) {
            const group = document.createElement("details");
            group.open = true;
            const summary = document.createElement("summary");
            summary.textContent = `公共素材 · ${template.global_materials.length} 个`;
            summary.style.cssText = "cursor:pointer;margin-top:9px;color:#86efac;font-weight:700;font-size:10px;";
            const rows = document.createElement("div");
            rows.style.cssText = "display:flex;flex-direction:column;gap:5px;margin-top:6px;";
            for (const material of template.global_materials) rows.appendChild(templateRepositoryMaterialRow(material));
            group.append(summary, rows);
            details.appendChild(group);
        }
        for (const [index, shot] of (template.cards || []).entries()) {
            const group = document.createElement("details");
            const summary = document.createElement("summary");
            summary.textContent = transferring
                ? "动作迁移内容 · 统一提示词与素材"
                : `分镜${index + 1}：${shot.title || "未命名"} · ${Number(shot.duration_seconds || 0)}秒 · ${shot.transition || "承接"}`;
            summary.style.cssText = "cursor:pointer;margin-top:9px;color:#dbeafe;font-weight:700;font-size:10px;";
            const content = document.createElement("div");
            content.style.cssText = "display:flex;flex-direction:column;gap:5px;margin-top:6px;";
            const prompt = document.createElement("pre");
            prompt.textContent = shot.prompt_mode === "input"
                ? `运行时输入：${shot.prompt_label || "分镜提示词"}` : (shot.prompt || "无固定提示词");
            prompt.style.cssText = "max-height:150px;overflow:auto;margin:0;border:1px solid #293b4e;border-radius:5px;background:#070c12;padding:7px;color:#9fb5c9;font:9px/1.5 ui-monospace,Consolas,monospace;white-space:pre-wrap;word-break:break-word;";
            content.appendChild(prompt);
            for (const material of shot.materials || []) content.appendChild(templateRepositoryMaterialRow(material));
            group.append(summary, content);
            details.appendChild(group);
        }
        use.onclick = () => {
            node.__myangSelectedTemplateByTask ||= {};
            node.__myangSelectedTemplateByTask[template.task_mode] = template.id;
            overlay.remove();
            openApplyDirectorTemplateDialog(node, template);
        };
        exportTemplate.onclick = () => openExportDirectorTemplateDialog(node, template);
        remove.onclick = async () => {
            if (!confirm(`删除模板「${template.name}」？素材文件不会删除。`)) return;
            try {
                await directorTemplateRequest(`/${encodeURIComponent(template.id)}`, {method: "DELETE"});
                templates = templates.filter((item) => item.id !== template.id);
                node.__myangDirectorTemplates = templates;
                node.__myangDirectorTemplatesLoadedAt = Date.now();
                for (const [task, templateId] of Object.entries(node.__myangSelectedTemplateByTask || {})) {
                    if (templateId === template.id) node.__myangSelectedTemplateByTask[task] = "";
                }
                selectedId = templates[0]?.id || "";
                drawList();
                renderTimeline(node);
            } catch (error) {
                alert(`删除失败：${error.message}`);
            }
        };
    };
    const drawList = () => {
        const query = search.value.trim().toLowerCase();
        const filtered = templates.filter((template) => {
            if (mode.value && template.task_mode !== mode.value) return false;
            if (!query) return true;
            const text = `${template.name || ""} ${template.description || ""} ${(template.cards || []).map((item) => item.title || "").join(" ")}`.toLowerCase();
            return text.includes(query);
        });
        list.replaceChildren();
        const counter = document.createElement("div");
        counter.textContent = `本地永久模板 ${filtered.length}/${templates.length}`;
        counter.style.cssText = "padding:3px 5px 7px;color:#71869b;font-size:9px;";
        list.appendChild(counter);
        for (const template of filtered) {
            const item = document.createElement("button");
            item.type = "button";
            item.style.cssText = `display:block;width:100%;margin:0 0 5px;padding:8px;text-align:left;border:1px solid ${template.id === selectedId ? "#60a5fa" : "#293d50"};border-radius:6px;background:${template.id === selectedId ? "#122b40" : "#0d1721"};color:#dbeafe;cursor:pointer;`;
            const name = document.createElement("b");
            name.textContent = template.name;
            name.style.cssText = "display:block;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:10px;";
            const meta = document.createElement("span");
            meta.textContent = template.task_mode === TRANSFER
                ? `${TRANSFER} · 统一提示词与素材 · ${inputCount(template)} 输入`
                : `${template.task_mode || FRESH} · ${(template.cards || []).length} 分镜 · ${inputCount(template)} 输入`;
            meta.style.cssText = "display:block;margin-top:4px;color:#7f93a8;font-size:8px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;";
            item.append(name, meta);
            item.onclick = () => { selectedId = template.id; drawList(); };
            list.appendChild(item);
        }
        const selected = filtered.find((item) => item.id === selectedId) || filtered[0] || null;
        if (selected && selected.id !== selectedId) selectedId = selected.id;
        renderDetails(selected);
    };
    const load = async () => {
        reload.disabled = true;
        try {
            const result = await directorTemplateRequest();
            templates = result.templates || [];
            details.style.color = "";
            node.__myangDirectorTemplates = templates;
            node.__myangDirectorTemplatesLoaded = true;
            node.__myangDirectorTemplatesLoadedAt = Date.now();
            if (!templates.some((item) => item.id === selectedId)) selectedId = templates[0]?.id || "";
            drawList();
        } catch (error) {
            details.textContent = `模板仓库读取失败：${error.message}`;
            details.style.color = "#fda4af";
        } finally { reload.disabled = false; }
    };
    search.oninput = drawList;
    mode.onchange = drawList;
    reload.onclick = load;
    create.onclick = () => { overlay.remove(); openSaveDirectorTemplateDialog(node); };
    importTemplate.onclick = () => chooseDirectorTemplateFile(node, load);
    renderDetails(null);
    load();
    search.focus();
}

function renderDirectorTemplatePanel(node) {
    const panel = document.createElement("section");
    panel.dataset.myangTemplatePanel = "1";
    panel.style.cssText = "border:1px solid #3b4c64;background:#101923;border-radius:7px;padding:8px;margin-bottom:8px;";
    const head = document.createElement("div");
    head.style.cssText = "display:flex;align-items:center;gap:6px;flex-wrap:wrap;";
    const title = document.createElement("b");
    title.textContent = currentTask(node) === TRANSFER
        ? "动作迁移模板" : "导演台视频模板";
    title.style.cssText = "flex:1;color:#c4b5fd;font-size:10px;";
    const select = document.createElement("select");
    select.style.cssText = "flex:1 1 220px;min-width:180px;max-width:340px;background:#0b1118;color:#dbeafe;border:1px solid #34465b;border-radius:4px;padding:5px;";
    const use = button("使用模板");
    const repository = button("模板仓库", "浏览本地永久保存的模板详情");
    const create = button("创建模板");
    const importTemplate = button("导入");
    const exportTemplate = button("导出");
    const remove = button("删除", "只删除模板定义，不删除素材文件");
    remove.style.color = "#fb7185";
    for (const control of [use, repository, create, importTemplate, exportTemplate, remove]) {
        control.style.cssText += ";flex:0 0 auto;min-height:30px;white-space:nowrap;";
    }
    head.append(title, select, use, repository, create, importTemplate, exportTemplate, remove);
    const notice = document.createElement("div");
    notice.textContent = node.__myangTemplateNotice || (currentTask(node) === TRANSFER
        ? "保存统一提示词、动作视频槽、目标人物素材以及可选生成参数；使用时会自动切回动作迁移模式。"
        : "把当前分镜、提示词和素材槽保存成可复用模板；固定内容与每次输入可自行决定。");
    notice.style.cssText = "margin-top:6px;color:#8498ad;font-size:9px;line-height:1.45;";
    panel.append(head, notice);
    const availableTemplates = () => (node.__myangDirectorTemplates || [])
        .filter((item) => item.task_mode === currentTask(node));
    node.__myangSelectedTemplateByTask ||= {};
    const draw = () => {
        const task = currentTask(node);
        const previous = select.value || node.__myangSelectedTemplateByTask[task] || "";
        select.replaceChildren();
        const templates = availableTemplates();
        select.append(new Option("不使用模板", ""));
        for (const item of templates) select.append(new Option(item.name, item.id));
        if (templates.some((item) => item.id === previous)) select.value = previous;
        node.__myangSelectedTemplateByTask[task] = select.value;
        use.disabled = exportTemplate.disabled = remove.disabled = !select.value;
    };
    const load = async () => {
        try {
            const result = await directorTemplateRequest();
            node.__myangDirectorTemplates = result.templates || [];
            node.__myangDirectorTemplatesLoadedAt = Date.now();
            draw();
            return true;
        } catch (error) {
            notice.textContent = `模板读取失败：${error.message}`;
            return false;
        }
    };
    use.onclick = () => {
        const selected = availableTemplates().find((item) => item.id === select.value);
        if (selected) openApplyDirectorTemplateDialog(node, selected);
    };
    select.onchange = () => {
        node.__myangSelectedTemplateByTask[currentTask(node)] = select.value;
        draw();
    };
    create.onclick = () => openSaveDirectorTemplateDialog(node);
    repository.onclick = () => openDirectorTemplateRepository(node);
    importTemplate.onclick = () => chooseDirectorTemplateFile(node, draw);
    exportTemplate.onclick = () => {
        const selected = availableTemplates().find((item) => item.id === select.value);
        if (selected) openExportDirectorTemplateDialog(node, selected);
    };
    remove.onclick = async () => {
        const selected = availableTemplates().find((item) => item.id === select.value);
        if (!selected || !confirm(`删除模板「${selected.name}」？素材文件不会删除。`)) return;
        await directorTemplateRequest(`/${encodeURIComponent(selected.id)}`, {method: "DELETE"});
        node.__myangDirectorTemplates = node.__myangDirectorTemplates.filter((item) => item.id !== selected.id);
        node.__myangDirectorTemplatesLoadedAt = Date.now();
        node.__myangSelectedTemplateByTask[currentTask(node)] = "";
        draw();
    };
    draw();
    const storageStale = !node.__myangDirectorTemplatesLoaded
        || Date.now() - Number(node.__myangDirectorTemplatesLoadedAt || 0) > 1500;
    if (storageStale && !node.__myangDirectorTemplatesLoading) {
        node.__myangDirectorTemplatesLoading = true;
        load().then((ok) => {
            if (ok) node.__myangDirectorTemplatesLoaded = true;
        }).finally(() => { node.__myangDirectorTemplatesLoading = false; });
    }
    return panel;
}

function renderAssetLibraryPanel(node) {
    const panel = document.createElement("section");
    panel.style.cssText = "border:1px solid #334155;background:#101923;border-radius:7px;padding:8px;margin-bottom:8px;";
    const head = document.createElement("div");
    head.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:8px;";
    const title = document.createElement("div");
    title.textContent = "导演台素材库";
    title.style.cssText = "color:#93c5fd;font-size:10px;font-weight:700;";
    const headActions = document.createElement("div");
    headActions.style.cssText = "display:flex;gap:5px;align-items:center;";
    const openButton = button("打开素材库", "浏览已收藏素材和素材文件夹");
    openButton.onclick = () => openMaterialLibraryChooser(node);
    const refreshButton = button("刷新", "重新读取角色、场景、音色、音乐和视频素材库");
    headActions.append(openButton, refreshButton);
    head.append(title, headActions);
    panel.appendChild(head);
    const body = document.createElement("div");
    body.style.cssText = "display:flex;flex-direction:column;gap:6px;margin-top:7px;";
    panel.appendChild(body);

    const draw = () => {
        if (!body.isConnected) return;
        body.replaceChildren();
        const candidates = planAssetCandidates(node);
        if (candidates.length) {
            const notice = document.createElement("div");
            notice.style.cssText = "border:1px solid #35634f;border-radius:5px;background:#11231d;padding:7px;color:#b7f7d8;font-size:9px;line-height:1.45;";
            notice.textContent = `AI 已识别 ${candidates.length} 个本次实际出镜主体，可按原主体名加入素材库：`;
            const tools = document.createElement("div");
            tools.style.cssText = "display:flex;gap:5px;flex-wrap:wrap;margin-top:5px;";
            for (const candidate of candidates) {
                const add = button(`＋ ${candidate.subject.name || candidate.asset.label}`, "确认名称和分类后加入素材库");
                add.onclick = () => openAssetLibraryDialog(node, candidate.asset, candidate.subject);
                tools.appendChild(add);
            }
            notice.appendChild(tools);
            body.appendChild(notice);
        }
        if (node.__myangAssetLibraryStatus) {
            const status = document.createElement("div");
            status.textContent = node.__myangAssetLibraryStatus;
            status.style.cssText = "color:#86efac;font-size:9px;";
            body.appendChild(status);
        }
        const assets = node.__myangAssetCatalogue || [];
        if (!node.__myangAssetCatalogueLoaded) {
            const loading = document.createElement("div");
            loading.textContent = "正在读取素材库…";
            loading.style.cssText = "color:#718096;font-size:9px;";
            body.appendChild(loading);
            return;
        }
        if (!assets.length) {
            const empty = document.createElement("div");
            empty.textContent = "素材库为空。可点击素材卡的“入库”，或在素材卡上点右键添加；只保存引用，不复制文件。";
            empty.style.cssText = "color:#718096;font-size:9px;line-height:1.45;";
            body.appendChild(empty);
            return;
        }
        const list = document.createElement("div");
        list.style.cssText = "display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:5px;";
        for (const asset of assets) {
            const row = document.createElement("div");
            row.style.cssText = "display:grid;grid-template-columns:minmax(0,1fr) 74px 28px;gap:5px;align-items:center;border:1px solid #2f3e50;border-radius:5px;background:#0b1118;padding:5px;";
            const name = document.createElement("input");
            name.value = asset.name;
            name.title = `${asset.file?.name || ""}\n可直接改名`;
            name.style.cssText = "min-width:0;background:#101923;color:#dbe7f3;border:1px solid #34465b;border-radius:4px;padding:5px;";
            const category = document.createElement("select");
            for (const [value, label] of Object.entries(ASSET_CATEGORY_LABELS)) {
                const option = document.createElement("option");
                option.value = value; option.textContent = label; category.appendChild(option);
            }
            category.value = asset.category;
            category.style.cssText = "min-width:0;background:#101923;color:#cbd5e1;border:1px solid #34465b;border-radius:4px;padding:5px;";
            const update = async () => {
                try {
                    const result = await assetLibraryRequest(`/${encodeURIComponent(asset.id)}`, {
                        method: "PATCH", headers: {"Content-Type": "application/json"},
                        body: JSON.stringify({name: name.value.trim() || asset.name, category: category.value}),
                    });
                    Object.assign(asset, result.asset);
                    node.__myangAssetLibraryStatus = `已更新「${asset.name}」`;
                } catch (error) { node.__myangAssetLibraryStatus = `更新失败：${error.message}`; }
            };
            name.onchange = update;
            category.onchange = update;
            const remove = button("×", "仅从素材库移除，不删除原文件");
            remove.style.color = "#fb7185";
            remove.onclick = async () => {
                await assetLibraryRequest(`/${encodeURIComponent(asset.id)}`, {method: "DELETE"});
                node.__myangAssetCatalogue = assets.filter((item) => item.id !== asset.id);
                draw();
            };
            row.append(name, category, remove);
            list.appendChild(row);
        }
        body.appendChild(list);
    };
    const load = async () => {
        node.__myangAssetCatalogueLoaded = false;
        draw();
        try {
            const result = await assetLibraryRequest();
            node.__myangAssetCatalogue = result.assets || [];
            node.__myangAssetCatalogueLoaded = true;
            node.__myangAssetLibraryStatus = "";
        } catch (error) {
            node.__myangAssetCatalogueLoaded = true;
            node.__myangAssetLibraryStatus = `读取失败：${error.message}`;
        }
        draw();
    };
    refreshButton.onclick = load;
    if (!node.__myangAssetCatalogueLoaded && !node.__myangAssetCatalogueLoading) {
        node.__myangAssetCatalogueLoading = true;
        load().finally(() => { node.__myangAssetCatalogueLoading = false; });
    } else draw();
    return panel;
}

function bindDirectorCollapsible(node, summary, body, indicator,
        stateProperty, expandedDisplay = "block") {
    summary.onclick = () => {
        const next = summary.getAttribute("aria-expanded") !== "true";
        node[stateProperty] = next;
        summary.setAttribute("aria-expanded", String(next));
        indicator.textContent = next ? "−" : "+";
        body.hidden = !next;
        body.style.display = next ? expandedDisplay : "none";
        // Folding is a presentation-only action. Keep every other card,
        // especially the live progress image and sampling state, mounted.
        syncPanelGeometry(node);
    };
}

function updateDirectorStats(node) {
    const stats = node.__myangDirectorRoot?.querySelector?.("[data-myang-director-stats]");
    if (stats) stats.textContent = directorStatsText(node);
}

function scheduleDirectorRender(node, scope) {
    if (!scope) return;
    node.__myangDirectorRenderScopes ||= new Set();
    if (scope === "full") {
        node.__myangDirectorRenderScopes.clear();
        node.__myangDirectorRenderScopes.add("full");
    } else if (!node.__myangDirectorRenderScopes.has("full")) {
        node.__myangDirectorRenderScopes.add(scope);
    }
    if (node.__myangDirectorRenderFrame) return;
    node.__myangDirectorRenderFrame = requestAnimationFrame(() => {
        node.__myangDirectorRenderFrame = 0;
        const scopes = node.__myangDirectorRenderScopes || new Set();
        node.__myangDirectorRenderScopes = new Set();
        if (scopes.has("full")) {
            refresh(node);
            return;
        }
        for (const section of scopes) replaceDirectorSection(node, section);
    });
}

function syncDirectorWidgetChange(node, name) {
    if (!name || name === TIMELINE_WIDGET || node.__myangDirectorSaving) return;
    if (node.__myangTemplateContract?.active
        && ["total_seconds", "segment_seconds"].includes(name)) {
        node.__myangTemplateContract[name] = Math.max(
            0, Number(widget(node, name)?.value) || 0);
        saveTimeline(node);
    }
    if (name === "resolution") {
        syncPanelGeometry(node);
        scheduleDirectorRender(node, "generation-settings");
        return;
    }
    if (name === "save_segments") {
        scheduleDirectorRender(node, "generation-settings");
        return;
    }
    if (name === "segment_seconds") {
        updateDirectorStats(node);
        return;
    }
    if (DIRECTOR_FULL_REFRESH.has(name)) {
        scheduleDirectorRender(node, "full");
        return;
    }
    scheduleDirectorRender(node, DIRECTOR_SECTION_REFRESH.get(name));
}

function setNativeWidget(node, name, value) {
    const target = widget(node, name);
    if (!target) return;
    target.value = value;
    target.callback?.(value);
    node.graph?.setDirtyCanvas?.(true, true);
    // H3Director's native callbacks are wrapped in onNodeCreated and already
    // schedule the smallest required card refresh. Calling the synchronizer a
    // second time here caused duplicate section rebuilds and visible flicker.
    if (!target.callback?.__myangDirectorSyncs) {
        syncDirectorWidgetChange(node, name);
    }
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
        input.dataset.myangControl = name;
        input.onchange = () => setNativeWidget(node, name, input.value);
    } else if (options.type === "checkbox") {
        input = document.createElement("input");
        input.type = "checkbox";
        input.checked = target.value !== false;
        input.dataset.myangControl = name;
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
        input.dataset.myangControl = name;
        input.onchange = () => setNativeWidget(node, name, Number(input.value));
    }
    input.setAttribute("aria-label", label);
    input.style.cssText = "min-width:0;width:100%;box-sizing:border-box;background:#0c131c;color:#dbe7f3;border:1px solid #37475b;border-radius:4px;padding:5px 6px;font-size:10px;outline:none;";
    input.onfocus = () => { input.style.borderColor = "#60a5fa"; };
    input.onblur = () => { input.style.borderColor = "#37475b"; };
    wrap.appendChild(input);
    return wrap;
}

function textDetailControl(node, name, label) {
    const target = widget(node, name);
    if (!target) return null;
    const wrap = document.createElement("label");
    wrap.style.cssText = "display:flex;flex-direction:column;gap:3px;min-width:0;color:#aebdcd;font-size:9px;line-height:1.35;";
    const caption = document.createElement("span");
    caption.textContent = label;
    const input = document.createElement("input");
    input.type = "text";
    input.value = String(target.value ?? "");
    input.dataset.myangControl = name;
    input.setAttribute("aria-label", label);
    input.style.cssText = "min-width:0;width:100%;box-sizing:border-box;background:#0c131c;color:#dbe7f3;border:1px solid #37475b;border-radius:4px;padding:5px 6px;font-size:10px;outline:none;";
    input.onchange = () => setNativeWidget(node, name, input.value);
    input.onfocus = () => { input.style.borderColor = "#60a5fa"; };
    input.onblur = () => { input.style.borderColor = "#37475b"; };
    wrap.append(caption, input);
    return wrap;
}

function renderGenerationSettingsPanel(node) {
    const panel = document.createElement("section");
    panel.dataset.myangDirectorSection = "generation-settings";
    panel.style.cssText = "border:1px solid #334155;background:#101923;border-radius:7px;margin-bottom:8px;overflow:hidden;";
    const expanded = node.__myangGenerationSettingsExpanded !== false;
    const header = document.createElement("button");
    header.type = "button";
    header.setAttribute("aria-expanded", String(expanded));
    header.style.cssText = "display:flex;align-items:center;gap:8px;width:100%;min-height:38px;padding:0 9px;border:0;background:transparent;color:#dbeafe;text-align:left;cursor:pointer;";
    const indicator = document.createElement("span");
    indicator.textContent = expanded ? "−" : "+";
    indicator.style.cssText = "flex:0 0 18px;color:#93c5fd;font-size:16px;text-align:center;";
    const title = document.createElement("span");
    title.textContent = "生成参数 · 导演台内设置";
    title.style.cssText = "font-size:11px;font-weight:700;";
    const hint = document.createElement("span");
    hint.textContent = "种子与一采步数仍保留在节点外";
    hint.style.cssText = "margin-left:auto;color:#718096;font-size:9px;white-space:nowrap;";
    header.append(indicator, title, hint);
    const body = document.createElement("div");
    body.style.cssText = "border-top:1px solid #2b3c52;padding:8px;";
    body.hidden = !expanded;
    if (!expanded) body.style.display = "none";
    const grid = document.createElement("div");
    grid.style.cssText = "display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:6px;";
    const add = (control) => { if (control) grid.appendChild(control); };
    add(detailControl(node, "task_mode", "生成任务", {values: true}));
    add(detailControl(node, "resolution", "一采分辨率", {values: true}));
    add(detailControl(node, "aspect_ratio", "画面比例", {values: true}));
    if (String(widget(node, "resolution")?.value || "") === "自定义") {
        add(detailControl(node, "width", "自定义宽", {min: 32, max: 16384, step: 32}));
        add(detailControl(node, "height", "自定义高", {min: 32, max: 16384, step: 32}));
    }
    add(detailControl(node, "denoise", "一采重绘幅度", {min: 0.01, max: 1, step: 0.01}));
    add(detailControl(node, "scheduler", "一采调度器", {values: true}));
    add(detailControl(node, "context_length", "段间锚点帧", {values: true}));
    add(detailControl(node, "ref_image_size", "参考图尺寸", {values: true}));
    add(detailControl(node, "save_segments", "保存每段", {type: "checkbox"}));
    add(textDetailControl(node, "segment_prefix", "分段文件名前缀"));
    body.appendChild(grid);
    const note = document.createElement("div");
    note.textContent = "这些参数随导演台保存；修改后会立即同步到节点输入。种子和一采步数仍可从外部连线或节点控件控制。";
    note.style.cssText = "margin-top:7px;color:#718096;font-size:9px;line-height:1.45;";
    body.appendChild(note);
    header.onclick = () => {
        const next = body.hidden;
        node.__myangGenerationSettingsExpanded = next;
        header.setAttribute("aria-expanded", String(next));
        indicator.textContent = next ? "−" : "+";
        body.hidden = !next;
        body.style.display = next ? "block" : "none";
        syncPanelGeometry(node);
    };
    panel.append(header, body);
    return panel;
}

function renderResumePanel(node) {
    const enabled = widget(node, "从指定段开始")?.value === true;
    const start = Math.max(1, Number(widget(node, "起始段")?.value || 1));
    const linked = inputLinked(node, "前段视频");
    const transferring = currentTask(node) === TRANSFER;
    const enabledShots = (node.__myangDirectorShots || []).filter(
        (shot) => shot.enabled !== false);
    const startShot = !transferring ? enabledShots[start - 1] : null;
    const startsWithCut = !transferring
        && String(startShot?.transition || "").trim() === "切镜";
    const contextRequired = enabled && start > 1 && !startsWithCut;
    const panel = document.createElement("div");
    panel.dataset.myangDirectorSection = "resume";
    const ok = !contextRequired || linked;
    panel.style.cssText = `border:1px solid ${ok ? "#334155" : "#7c5c2c"};background:#101923;border-radius:6px;padding:7px;margin-bottom:7px;`;
    const title = document.createElement("div");
    title.textContent = "从指定段开始生成";
    title.style.cssText = "color:#93c5fd;font-size:10px;font-weight:700;margin-bottom:5px;";
    const grid = document.createElement("div");
    grid.style.cssText = "display:grid;grid-template-columns:minmax(210px,1.35fr) minmax(120px,1fr);gap:8px;align-items:stretch;";

    const toggleLabel = document.createElement("label");
    toggleLabel.style.cssText = "display:flex;align-items:center;gap:8px;min-height:44px;box-sizing:border-box;padding:7px 9px;border:1px solid #37475b;border-radius:5px;color:#dbe7f3;font-size:10px;cursor:pointer;background:#0c131c;";
    const toggle = document.createElement("input");
    toggle.type = "checkbox";
    toggle.checked = enabled;
    toggle.dataset.myangControl = "从指定段开始";
    toggle.setAttribute("aria-label", "启用从指定段开始生成");
    toggle.style.cssText = "width:17px;height:17px;flex:0 0 17px;accent-color:#60a5fa;";
    const toggleText = document.createElement("span");
    toggleText.textContent = enabled ? "已启用：只生成后续分段" : "未启用：默认从第 1 段生成";
    toggle.onchange = () => {
        if (toggle.checked && Number(widget(node, "起始段")?.value || 1) < 2) {
            const startWidget = widget(node, "起始段");
            if (startWidget) startWidget.value = 2;
        }
        setNativeWidget(node, "从指定段开始", toggle.checked);
    };
    toggleLabel.append(toggle, toggleText);
    grid.appendChild(toggleLabel);

    const startControl = detailControl(node, "起始段", "从第几段开始", {
        min: 1, max: 64, step: 1,
    });
    const startInput = startControl?.querySelector("input");
    if (startInput) {
        startInput.disabled = !enabled;
        startInput.setAttribute("aria-disabled", String(!enabled));
        startInput.style.opacity = enabled ? "1" : "0.42";
        startInput.style.cursor = enabled ? "text" : "not-allowed";
    }
    if (startControl) grid.appendChild(startControl);
    const help = document.createElement("div");
    help.textContent = !enabled
        ? "未勾选时段号不会生效；无论旧工作流保存了什么数值，本次都从第 1 段开始。"
        : start === 1
            ? "已启用但选择第 1 段，与完整生成相同。"
            : startsWithCut
                ? `从第 ${start} 段独立切镜开始，不参考上一段末尾，也不需要连接『前段视频』。`
                : linked
                    ? `从第 ${start} 段开始；『前段视频』只取第 ${start - 1} 段末尾锚点做承接，不会当成 @视频1。文件名保留绝对段号。`
                    : `第 ${start} 段设置为承接，请把第 ${start - 1} 段成片接到左侧『前段视频』。`;
    help.style.cssText = `color:${ok ? "#718096" : "#fbbf24"};font-size:9px;line-height:1.45;margin-top:5px;`;
    panel.append(title, grid, help);
    return panel;
}

function renderReferenceVideoPanel(node) {
    const resolution = String(widget(node, "参考视频分辨率")?.value || REFERENCE_ORIGINAL);
    const panel = document.createElement("div");
    panel.dataset.myangDirectorSection = "reference";
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

function renderFirstPassMemoryPanel(node) {
    const profile = String(widget(node, "一采显存策略")?.value || "自动平衡（16GB推荐）");
    const transferring = currentTask(node) === TRANSFER;
    const originalReferences = String(widget(node, "ref_image_size")?.value || "") === "匹配素材（原尺寸）";
    const panel = document.createElement("div");
    panel.dataset.myangDirectorSection = "firstMemory";
    panel.style.cssText = "border:1px solid #334155;background:#101923;border-radius:6px;padding:7px;margin-bottom:8px;";
    const title = document.createElement("div");
    title.textContent = "一采显存与预览";
    title.style.cssText = "color:#93c5fd;font-size:10px;font-weight:700;margin-bottom:5px;";
    const grid = document.createElement("div");
    grid.style.cssText = "display:grid;grid-template-columns:minmax(220px,.8fr) minmax(260px,1.4fr);gap:7px;align-items:stretch;";
    const profileControl = detailControl(node, "一采显存策略", "运行策略", {values: [
        "自动平衡（16GB推荐）",
        "兼容模式（每步清晰预览）",
        "显存优先（更大预留）",
        "关闭",
    ]});
    if (profileControl) grid.appendChild(profileControl);
    const help = document.createElement("div");
    help.style.cssText = "display:flex;align-items:center;border:1px solid #29394c;border-radius:5px;padding:6px 8px;color:#8fa5bb;font-size:9px;line-height:1.45;background:#0c1624;";
    const profileHelp = transferring && profile !== "兼容模式（每步清晰预览）"
        ? "动作迁移在自动平衡/显存优先档关闭一采中途图像预览，只显示采样步数并在分段完成后显示正式清晰帧；仍可切换兼容模式恢复每步清晰预览。"
        : profile === "关闭"
        ? "兼容旧工作流：不额外清理、不预留显存、关闭一采中途预览；建议新任务使用自动平衡或显存优先。"
        : profile === "兼容模式（每步清晰预览）"
        ? "保留旧行为：一采每一步都调用视频 VAE 解码清晰帧。16GB 显卡可能反复换页，640P 也会显著变慢。"
        : profile === "显存优先（更大预留）"
            ? "每段条件编码后卸载 CLIP/VAE，预留 1.5GB 显存；每步用 CPU 潜空间轻量预览，段落完成后替换为正式清晰帧。"
            : "16GB 推荐：条件编码后卸载 CLIP/VAE，预留 1.25GB 显存；每步用 CPU 潜空间轻量预览，不加载视频 VAE，段落完成后自动替换为正式清晰帧。";
    help.textContent = profileHelp + (originalReferences
        ? " 当前参考图尺寸为『匹配素材（原尺寸）』，2K/4K素材会额外放大条件显存；建议改为『匹配生成分辨率』。"
        : "");
    grid.appendChild(help);
    // 一采断点 / 直接二采 belongs to the first pass, not to the 二采 card. It used
    // to render inside the detail panel, which collapses whenever 二采 is off, so
    // the switch was unreachable exactly when a user wanted to skip pass 1.
    const pass1Mode = String(widget(node, "一采断点模式")?.value || "关闭");
    const pass1VideoConnected = (node.inputs || []).some(
        (input) => input.name === "一采成片" && input.link != null);
    const pass1Control = detailControl(node, "一采断点模式", "一采断点 / 直接二采", {
        values: [
            "关闭", "保存一采检查点", "恢复一采进度（已有跳过，缺失继续）",
            "读取检查点，直接二采",
            "使用接入的一采成片，直接二采（单段）",
        ],
    });
    if (pass1Control) grid.appendChild(pass1Control);
    const pass1Help = document.createElement("div");
    const pass1Reuse = pass1Mode === "使用接入的一采成片，直接二采（单段）";
    const pass1Bad = (pass1Reuse && !pass1VideoConnected)
        || (!pass1Reuse && pass1VideoConnected);
    pass1Help.style.cssText = `display:flex;align-items:center;border:1px solid ${pass1Bad ? "#92400e" : "#29394c"};border-radius:5px;padding:6px 8px;color:${pass1Bad ? "#fbbf24" : "#8fa5bb"};font-size:9px;line-height:1.45;background:${pass1Bad ? "#2b1d0d" : "#0c1624"};`;
    pass1Help.textContent = pass1Reuse
        ? (pass1VideoConnected
            ? "将跳过一采：接入的成片先 VAE 重编码,再直接进二采。仅支持单段,且帧数必须与分镜完全一致；二采必须开启。"
            : "已选『直接二采』但左侧『一采成片』没连接,运行会直接报错。")
        : (pass1VideoConnected
            ? "已连接『一采成片』,但模式不是『直接二采』——这个输入会被忽略,仍会正常跑一采。要用它请把本项改成『使用接入的一采成片,直接二采（单段）』。"
            : "关闭=正常跑一采。保存/恢复用于分段断点；『读取检查点』和『使用接入的一采成片』都会跳过一采直接进二采（需二采开启）。");
    grid.appendChild(pass1Help);
    panel.append(title, grid);
    return panel;
}

function renderDetailPanel(node) {
    const enabled = widget(node, "二采开启")?.value === true;
    const mode = String(widget(node, "二采模式")?.value || "放大 + 二采（推荐）");
    const resolution = String(widget(node, "二采分辨率")?.value || "832P");
    const method = String(widget(node, "二采放大方式")?.value || "");
    const memoryProfile = String(widget(node, "二采显存策略")?.value || "自动平衡（16GB推荐）");
    const pass1Mode = String(widget(node, "一采断点模式")?.value || "关闭");
    const sampling = mode !== "仅放大（不二采·最快）";
    const continuousSigma = enabled && sampling
        && widget(node, "二采连续Sigma")?.value === true;
    const upscaling = mode !== "同分辨率二采（不放大）";
    const sameResolution = sampling && !upscaling;
    const neural = upscaling && method.includes("neural_3d");
    const pixelLike = upscaling && (method.includes("pixel") || method.includes("vsr"));
    const externalConnected = (node.inputs || []).some(
        (input) => input.name === "二采设置" && input.link != null);
    const modelConnected = (node.inputs || []).some(
        (input) => input.name === "二采模型" && input.link != null);
    // Read by the 一采成片 branch of the checkpoint hint below. Without this
    // declaration that branch threw ReferenceError, and because it is only
    // reachable when 一采断点模式 is the 一采成片 mode, picking that mode aborted
    // the card render mid-way and every panel after this one vanished.
    const pass1VideoConnected = (node.inputs || []).some(
        (input) => input.name === "一采成片" && input.link != null);

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
        ? `二采已开启 · ${mode}${continuousSigma ? " · 连续Sigma实验" : ""}${upscaling ? ` · ${resolution}` : ""}${sampling ? ` · ${memoryProfile}` : ""}`
        : "二采已关闭 · 点击展开设置";
    summaryText.style.cssText = "min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;";
    const indicator = document.createElement("span");
    indicator.textContent = expanded ? "−" : "+";
    indicator.setAttribute("aria-hidden", "true");
    indicator.style.cssText = "flex:0 0 20px;text-align:center;color:#93c5fd;font-size:16px;line-height:1;";
    summary.append(indicator, summaryText);
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
    bindDirectorCollapsible(node, summary, body, indicator,
        "__myangDetailPanelExpanded", "block");
    if (externalConnected) {
        const warning = document.createElement("div");
        warning.textContent = "已连接旧版『二采设置』：为兼容旧工作流，外部设置优先于本面板。断开后由导演台面板接管。";
        warning.style.cssText = "border:1px solid #92400e;background:#2b1d0d;color:#fbbf24;border-radius:5px;padding:6px;margin-bottom:7px;font-size:9px;line-height:1.45;";
        body.appendChild(warning);
    }
    if (enabled && sampling) {
        const modelState = document.createElement("div");
        modelState.textContent = continuousSigma
            ? "连续 Sigma：前后两段复用一采模型、采样器、调度器与同一噪声轨迹；不加载独立二采模型。"
            : modelConnected
                ? "二采模型已连接：将使用未挂 Turbo LoRA 的 Ref2VA 基模。"
                : "需要连接左侧『二采 Ref2VA 基模』；仅放大模式不需要模型。";
        modelState.style.cssText = `margin-bottom:7px;color:${continuousSigma || modelConnected ? "#86efac" : "#fb7185"};font-size:9px;`;
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
            "pixel (像素放大·自用版工作流方式)",
            "nvidia_rtx_vsr (NVIDIA RTX 视频超分·实验)",
        ]}));
        if (resolution === "自定义") {
            add(detailControl(node, "二采自定义宽", "自定义宽", {min: 32, max: 8192, step: 32}));
            add(detailControl(node, "二采自定义高", "自定义高", {min: 32, max: 8192, step: 32}));
        }
    }
    if (sampling) {
        add(detailControl(node, "二采连续Sigma", "连续 Sigma 二采（实验）", {type: "checkbox"}));
    }
    if (sameResolution) {
        // 只有同分辨率档没有缩放步骤，VSR 才需要单独作为解码后的 1:1 增强出现。
        // 放大档请用「放大算法」里的 VSR，仅放大档同理，否则同一批帧过两遍。
        add(detailControl(node, "二采后VSR增强", "二采后 VSR 增强（原尺寸）", {type: "checkbox"}));
    }
    if (sampling) {
        add(detailControl(node, "二采显存策略", "显存 / 速度平衡", {values: [
            "自动平衡（16GB推荐）",
            "速度优先（关闭二采逐步预览）",
            "完整逐步预览（占显存）",
            "显存优先（832P保底）",
            "自定义",
        ]}));
        if (memoryProfile === "自定义") {
            add(detailControl(node, "二采自定义显存预留", "二采激活空间（GB）", {
                min: 0, max: 8, step: 0.05,
            }));
            add(detailControl(node, "二采自定义预览间隔", "清晰预览间隔（步）", {
                min: 0, max: 100, step: 1,
            }));
        }
        add(detailControl(node, "二采步数", continuousSigma ? "Sigma 尾段步数" : "采样步数", {min: 1, max: 100, step: 1}));
        if (!continuousSigma) {
            add(detailControl(node, "二采重绘幅度", "重绘幅度", {min: 0.01, max: 1, step: 0.01}));
            add(detailControl(node, "二采调度器", "调度器", {values: ["beta", "simple", "normal"]}));
            add(detailControl(node, "二采采样器", "采样器", {values: ["res_multistep", "euler"]}));
            add(detailControl(node, "二采轮数", "二采轮数", {min: 1, max: 8, step: 1}));
            add(detailControl(node, "二采种子策略", "种子策略", {values: [
                "每轮沿用同一种子", "每轮种子 +1",
            ]}));
            add(detailControl(node, "二采复用一采条件", "复用文本/素材条件（不含成片）", {type: "checkbox"}));
        }
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
        add(detailControl(node, "二采时间分块", "神经 3D 时间分块", {min: 0, max: 256, step: 1}));
    }
    if (enabled && widget(node, "save_segments")?.value !== false) {
        add(detailControl(node, "save_raw_segments", "同时保存二采前分段", {type: "checkbox"}));
    }
    body.appendChild(grid);
    if (enabled && sampling && !continuousSigma) {
        const reuseHelp = document.createElement("div");
        const reuseOn = widget(node, "二采复用一采条件")?.value !== false;
        reuseHelp.style.cssText = "margin-top:7px;padding:6px 8px;border-left:2px solid #22c55e;background:#0c1624;color:#8fa5bb;font-size:9px;line-height:1.5;";
        reuseHelp.textContent = reuseOn
            ? "只沿用一采已编码的提示词 token 与参考素材 latent，不会把640P一采成片塞进条件，也不会复制参考视频。二采高分辨率目标 latent 仍单独存在；通常更省显存和时间。"
            : "将重新加载文本/素材编码链并按二采设置重建条件；可能增加瞬时显存与耗时，参考图 token 变化时也更容易产生轻微漂移。";
        body.appendChild(reuseHelp);
    }
    if (enabled && continuousSigma) {
        const experimentalHelp = document.createElement("div");
        const firstSteps = Number(widget(node, "steps")?.value || 0);
        const tailSteps = Number(widget(node, "二采步数")?.value || 0);
        const firstDenoise = Number(widget(node, "denoise")?.value ?? 1);
        const incompatibleMethod = upscaling && !neural && !method.includes("latent");
        const incompatibleCheckpoint = pass1Mode !== "关闭";
        const incompatibleEnhancement = widget(node, "音频精修开启")?.value === true
            || widget(node, "脸部精修开启")?.value === true
            || widget(node, "动作修复开启")?.value === true;
        const invalid = firstDenoise !== 1 || incompatibleMethod
            || incompatibleCheckpoint || incompatibleEnhancement;
        experimentalHelp.style.cssText = `margin-top:7px;padding:7px 8px;border:1px solid ${invalid ? "#92400e" : "#4c3e75"};border-left:3px solid ${invalid ? "#f59e0b" : "#a78bfa"};border-radius:5px;background:${invalid ? "#2b1d0d" : "#161329"};color:${invalid ? "#fbbf24" : "#c4b5fd"};font-size:9px;line-height:1.55;`;
        experimentalHelp.textContent = invalid
            ? "连续 Sigma 当前配置不兼容：一采重绘必须为 1.0，只能使用 neural_3d 或同分辨率模式，并关闭一采断点、音频精修、小脸精修和动作修复。运行前会明确停止，不会静默退回独立二采。"
            : `实验轨迹共 ${firstSteps + tailSteps} 步：一采执行前 ${firstSteps} 步，放大仍带噪的 latent 后继续 ${tailSteps} 步；二采重绘幅度、独立调度器、独立采样器、轮数和种子策略不参与。`;
        body.appendChild(experimentalHelp);
    }
    if (pass1Mode !== "关闭") {
        const checkpointHelp = document.createElement("div");
        const prefix = String(widget(node, "segment_prefix")?.value || "video/H3_导演台");
        let text = "";
        let color = "#93c5fd";
        if (pass1Mode === "保存一采检查点") {
            text = `每段一采完成后保存完整音画 latent；前缀：${prefix}。中途停止时，已完成段仍可复用。`;
        } else if (pass1Mode === "恢复一采进度（已有跳过，缺失继续）") {
            text = `按前缀 ${prefix} 恢复：已有检查点的段跳过一采，缺失段继续一采并立即补存；可继续接当前二采。`;
        } else if (pass1Mode === "读取检查点，直接二采") {
            text = `按前缀 ${prefix} 和原始段号读取检查点，完全跳过一采；分镜时长和锚点必须与保存时一致。`;
            if (!enabled) {
                text += " 当前二采未开启，运行会在采样前停止。";
                color = "#fb7185";
            }
        } else {
            text = pass1VideoConnected
                ? "已连接一采成片：仅支持单段；会先 VAE 重编码，再直接进入二采。"
                : "请连接左侧『已保存的一采成片』；可选连接原音轨。此兼容入口仅支持单段。";
            if (!pass1VideoConnected || !enabled) color = "#fb7185";
        }
        checkpointHelp.textContent = text;
        checkpointHelp.style.cssText = `margin-top:7px;padding:6px 8px;border-left:2px solid ${color};background:#0c1624;color:${color};font-size:9px;line-height:1.5;`;
        body.appendChild(checkpointHelp);
    }
    if (enabled && sampling) {
        const memoryHelp = document.createElement("div");
        memoryHelp.style.cssText = "margin-top:7px;padding:6px 8px;border-left:2px solid #3b82f6;background:#0c1624;color:#8fa5bb;font-size:9px;line-height:1.5;";
        memoryHelp.textContent = memoryProfile === "速度优先（关闭二采逐步预览）"
            ? "不预留显存、不限制模型驻留，并关闭中途 VAE 预览。由AIMDO按实时压力换页；若真的显存不足，再自动换出部分动态权重页重试。"
            : memoryProfile === "完整逐步预览（占显存）"
                ? "每一步都解码清晰预览；832P 时 VAE 会与 H3 模型争抢显存，16GB 显卡可能明显变慢。"
                : memoryProfile === "显存优先（832P保底）"
                    ? "给832P激活留约 4.5GB，并关闭中途 VAE 预览。DynamicVRAM 只限制权重水位，这部分空间仍由激活使用，不会被空置；其余权重驻留64GB内存。"
                    : memoryProfile === "自定义"
                        ? "这里设置的是激活空间，不是永久空置显存。数值越大越稳，但权重换入越多；间隔0关闭中途预览，最终预览仍保留。"
                        : "16GB 推荐：不做固定显存预留。二采前主动释放一采、VAE、放大器、预取队列与CUDA复用缓冲，首次由AIMDO按实时压力调度模型页；若真的OOM，再只换出部分GPU权重页并从64GB内存快速重试。";
        body.appendChild(memoryHelp);
    }
    panel.appendChild(body);
    return panel;
}

function renderAudioPanel(node) {
    const refineOn = widget(node, "音频精修开启")?.value === true;
    const seamOn = widget(node, "音频接缝平滑")?.value !== false;
    const modelConnected = (node.inputs || []).some(
        (input) => input.name === "二采模型" && input.link != null);
    const pass1VideoConnected = (node.inputs || []).some(
        (input) => input.name === "一采成片" && input.link != null);
    const turboConnected = (node.inputs || []).some(
        (input) => input.name === "Turbo联合模型" && input.link != null);
    const expanded = node.__myangAudioPanelExpanded ?? refineOn;
    const active = Number(refineOn) + Number(seamOn);
    const panel = document.createElement("section");
    panel.dataset.myangCollapsible = "audio";
    panel.style.cssText = `flex:0 0 auto;min-height:38px;border:1px solid ${active ? "#7c5cbf" : "#334155"};background:${active ? "#181526" : "#111820"};border-radius:7px;margin-bottom:8px;overflow:hidden;`;

    const summary = document.createElement("button");
    summary.type = "button";
    summary.setAttribute("aria-expanded", String(expanded));
    summary.setAttribute("aria-controls", `myang-audio-body-${node.id}`);
    summary.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:8px;width:100%;min-height:38px;padding:8px;border:0;background:transparent;color:#e9d5ff;font-size:11px;font-weight:700;text-align:left;cursor:pointer;outline:none;";
    const summaryText = document.createElement("span");
    const modes = [];
    if (refineOn) modes.push("单段精修");
    if (seamOn) modes.push("接缝平滑");
    summaryText.textContent = modes.length
        ? `音频处理 · ${modes.join(" + ")}`
        : "音频处理 · 全部关闭";
    summaryText.style.cssText = "min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;";
    const indicator = document.createElement("span");
    indicator.textContent = expanded ? "−" : "+";
    indicator.setAttribute("aria-hidden", "true");
    indicator.style.cssText = "flex:0 0 20px;text-align:center;color:#c4b5fd;font-size:16px;line-height:1;";
    summary.append(indicator, summaryText);
    summary.onfocus = () => { summary.style.boxShadow = "inset 0 0 0 2px #7c5cbf"; };
    summary.onblur = () => { summary.style.boxShadow = "none"; };
    panel.appendChild(summary);

    const body = document.createElement("div");
    body.id = `myang-audio-body-${node.id}`;
    body.hidden = !expanded;
    body.style.cssText = "display:flex;flex-direction:column;gap:7px;border-top:1px solid #443568;padding:8px;";
    if (!expanded) body.style.display = "none";
    bindDirectorCollapsible(node, summary, body, indicator,
        "__myangAudioPanelExpanded", "flex");

    const card = (title, hint, toggleName, enabled) => {
        const section = document.createElement("section");
        section.style.cssText = `border:1px solid ${enabled ? "#66509a" : "#293847"};background:#0c121a;border-radius:6px;padding:7px;`;
        const head = document.createElement("div");
        head.style.cssText = "display:grid;grid-template-columns:minmax(0,1fr) 118px;gap:8px;align-items:center;";
        const copy = document.createElement("div");
        const heading = document.createElement("div");
        heading.textContent = title;
        heading.style.cssText = "color:#ede9fe;font-size:10px;font-weight:700;";
        const description = document.createElement("div");
        description.textContent = hint;
        description.style.cssText = "color:#8290a3;font-size:9px;line-height:1.45;margin-top:2px;";
        copy.append(heading, description);
        const toggle = detailControl(node, toggleName, enabled ? "已开启" : "开启", {type: "checkbox"});
        toggle.style.cssText += ";box-sizing:border-box;width:118px;min-width:118px;justify-content:flex-end;white-space:nowrap;";
        head.append(copy, toggle);
        section.appendChild(head);
        body.appendChild(section);
        return section;
    };

    const refine = card(
        "H3 单段音频精修",
        "冻结视频，只让基模重新去噪音频；改善 Turbo 低步数的噪声、毛刺与清晰度。",
        "音频精修开启", refineOn);
    if (refineOn) {
        const state = document.createElement("div");
        state.textContent = turboConnected && !modelConnected
            ? "当前一采使用 Turbo：请连接左侧『二采 Ref2VA 基模』。"
            : modelConnected
                ? "将复用未挂 Turbo LoRA 的二采基模，视频 latent 保持冻结。"
                : "当前一采是基模：将直接复用一采模型。";
        state.style.cssText = `margin-top:7px;color:${turboConnected && !modelConnected ? "#fb7185" : "#86efac"};font-size:9px;line-height:1.4;`;
        const grid = document.createElement("div");
        grid.style.cssText = "display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:6px;margin-top:7px;padding-top:7px;border-top:1px solid #302844;";
        grid.append(
            detailControl(node, "音频精修步数", "附加步数", {min: 1, max: 100, step: 1}),
            detailControl(node, "音频去噪强度", "去噪强度", {min: 0.01, max: 1, step: 0.01}),
            detailControl(node, "音频精修采样器", "采样器", {values: ["euler", "res_multistep"]}),
            detailControl(node, "音频精修调度器", "调度器", {values: ["simple", "beta", "normal"]}),
        );
        refine.append(state, grid);
    }

    const seam = card(
        "段间音频接缝平滑",
        "使用下一段被裁掉的重叠锚点声音改写上一段尾部；不重复声音、不改时长和口型同步。",
        "音频接缝平滑", seamOn);
    if (seamOn) {
        const grid = document.createElement("div");
        grid.style.cssText = "display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:6px;margin-top:7px;padding-top:7px;border-top:1px solid #302844;";
        grid.appendChild(detailControl(
            node, "音频接缝时长", "融合时长（毫秒）", {min: 0, max: 500, step: 5}));
        seam.appendChild(grid);
    }

    const foot = document.createElement("div");
    foot.textContent = "推荐先只开接缝平滑（几乎不增加显存与耗时）；单段音频本身较差时再开精修。精修未采用大内存缓存。";
    foot.style.cssText = "color:#8c80a8;font-size:9px;line-height:1.45;";
    body.appendChild(foot);
    panel.appendChild(body);
    return panel;
}

function renderEnhancementPanel(node) {
    const faceOn = widget(node, "脸部精修开启")?.value === true;
    const motionOn = widget(node, "动作修复开启")?.value === true;
    const viewsOn = widget(node, "多视角分镜开启")?.value === true;
    const active = [faceOn, motionOn, viewsOn].filter(Boolean).length;
    const expanded = node.__myangEnhancementPanelExpanded ?? active > 0;
    const panel = document.createElement("section");
    panel.dataset.myangCollapsible = "enhancement";
    panel.style.cssText = `flex:0 0 auto;min-height:38px;border:1px solid ${active ? "#2f8060" : "#334155"};background:${active ? "#0f211d" : "#111820"};border-radius:7px;margin-bottom:8px;overflow:hidden;`;
    const summary = document.createElement("button");
    summary.type = "button";
    summary.setAttribute("aria-expanded", String(expanded));
    summary.setAttribute("aria-controls", `myang-enhancement-body-${node.id}`);
    summary.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:8px;width:100%;min-height:38px;padding:8px;border:0;background:transparent;color:#b7f7d8;font-size:11px;font-weight:700;text-align:left;cursor:pointer;outline:none;";
    const summaryText = document.createElement("span");
    summaryText.textContent = active
        ? `画质与分镜增强 · 已开启 ${active} 项`
        : "画质与分镜增强 · 全部关闭";
    summaryText.style.cssText = "min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;";
    const indicator = document.createElement("span");
    indicator.textContent = expanded ? "−" : "+";
    indicator.setAttribute("aria-hidden", "true");
    indicator.style.cssText = "flex:0 0 20px;text-align:center;color:#86efac;font-size:16px;line-height:1;";
    summary.append(indicator, summaryText);
    summary.onfocus = () => { summary.style.boxShadow = "inset 0 0 0 2px #2f8060"; };
    summary.onblur = () => { summary.style.boxShadow = "none"; };
    panel.appendChild(summary);

    const body = document.createElement("div");
    body.id = `myang-enhancement-body-${node.id}`;
    body.hidden = !expanded;
    body.style.cssText = "display:flex;flex-direction:column;gap:7px;border-top:1px solid #28483e;padding:8px;";
    if (!expanded) body.style.display = "none";
    bindDirectorCollapsible(node, summary, body, indicator,
        "__myangEnhancementPanelExpanded", "flex");
    const section = (title, description, toggleName, enabled, controls) => {
        const card = document.createElement("section");
        card.style.cssText = `border:1px solid ${enabled ? "#34785f" : "#293847"};background:#0b141b;border-radius:6px;padding:7px;`;
        const head = document.createElement("div");
        head.style.cssText = "display:grid;grid-template-columns:minmax(0,1fr) 118px;gap:8px;align-items:center;";
        const text = document.createElement("div");
        const heading = document.createElement("div");
        heading.textContent = title;
        heading.style.cssText = "color:#d9f7e9;font-size:10px;font-weight:700;";
        const hint = document.createElement("div");
        hint.textContent = description;
        hint.style.cssText = "color:#718096;font-size:9px;line-height:1.4;margin-top:2px;";
        text.append(heading, hint);
        const toggle = detailControl(node, toggleName, enabled ? "已开启" : "开启", {type: "checkbox"});
        toggle.style.cssText += ";box-sizing:border-box;width:118px;min-width:118px;justify-content:flex-end;white-space:nowrap;";
        head.append(text, toggle);
        card.appendChild(head);
        if (enabled) {
            const grid = document.createElement("div");
            grid.style.cssText = "display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:6px;margin-top:7px;padding-top:7px;border-top:1px solid #263641;";
            for (const control of controls()) if (control) grid.appendChild(control);
            card.appendChild(grid);
        }
        body.appendChild(card);
        return card;
    };

    section(
        "H3 小脸精修",
        "逐段跟踪并重绘小脸；放在二采之后，保留原音频。",
        "脸部精修开启", faceOn, () => [
            detailControl(node, "脸部检测器", "检测模型", {values: true}),
            detailControl(node, "脸部精修步数", "精修步数", {min: 1, max: 50, step: 1}),
            detailControl(node, "脸部精修重绘", "重绘幅度", {min: 0.01, max: 1, step: 0.01}),
            detailControl(node, "脸部裁剪倍率", "裁剪倍率", {min: 1.2, max: 8, step: 0.1}),
            detailControl(node, "脸部身份图序号", "身份图序号（0=自动）", {min: 0, max: 9, step: 1}),
        ]);

    section(
        "MAINodes 高速动作修复",
        "识别动作突变并对局部时间加密；逐段运行，耗时会明显增加。",
        "动作修复开启", motionOn, () => [
            detailControl(node, "动作修复档位", "修复档位", {values: true}),
            detailControl(node, "动作修复步数", "附加步数", {min: 4, max: 50, step: 1}),
            detailControl(node, "动作修复注入", "注入强度", {min: 0.05, max: 1, step: 0.05}),
        ]);

    const viewCard = section(
        "角色五视图分镜",
        "运行前生成一次角色多视角表，作为全片公共素材交给每个分镜。",
        "多视角分镜开启", viewsOn, () => [
            detailControl(node, "多视角角色图片序号", "角色图片序号", {min: 1, max: 9, step: 1}),
            detailControl(node, "多视角尺寸", "生成尺寸", {values: true}),
            detailControl(node, "多视角步数", "生成步数", {min: 4, max: 50, step: 1}),
            detailControl(node, "多视角LoRA", "Turnaround LoRA", {values: true}),
            detailControl(node, "多视角LoRA强度", "LoRA 强度", {min: 0, max: 2, step: 0.05}),
        ]);
    if (viewsOn && node.__myangTurnaroundPreview?.file) {
        const preview = document.createElement("img");
        preview.alt = "角色五视图分镜预览";
        preview.src = `/api/view?filename=${encodeURIComponent(node.__myangTurnaroundPreview.file)}&type=temp&subfolder=&_t=${node.__myangTurnaroundPreview.ts || Date.now()}`;
        preview.style.cssText = "display:block;width:100%;max-height:210px;object-fit:contain;background:#05070a;border-radius:5px;margin-top:7px;";
        viewCard.appendChild(preview);
    }

    const foot = document.createElement("div");
    foot.textContent = "默认全部关闭；关闭时不创建插件子图，不增加显存和采样时间。建议先单独开启一项做 A/B 对比。";
    foot.style.cssText = "color:#739187;font-size:9px;line-height:1.45;";
    body.appendChild(foot);
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
    const agentLinked = inputLinked(node, "media") && currentTask(node) === FRESH;
    const panel = document.createElement("div");
    panel.style.cssText = "border:1px solid #334155;background:#101923;border-radius:6px;padding:7px;margin-bottom:8px;";
    const title = document.createElement("div");
    const modeLabel = modeTaskValue(node).replace(/（.*$/, "");
    title.textContent = `公共素材（${modeLabel}模式内共享）`;
    title.style.cssText = "color:#93c5fd;font-size:10px;font-weight:700;margin-bottom:5px;";
    panel.appendChild(title);
    const help = document.createElement("div");
    help.textContent = options.llmPicks
        ? "在这里上传当前模式共用的角色图、场景图、参考视频和配乐，并给每个素材起一个主体名。切换动作迁移、纯生成或视频续写后，各模式使用各自独立的素材清单。"
          + "智能切分时素材清单会连同剧本一起交给 LLM，由它判断每一段该引用哪些素材并写上 @图片N 标签。"
        : "当前模式内共用素材。镜头卡选择「叠加全局素材」时，这里的素材排在镜头专属素材之前；"
          + "选择「仅本镜头」的镜头不会拿到它们。";
    help.style.cssText = "color:#718096;font-size:9px;line-height:1.5;margin-bottom:6px;";
    panel.appendChild(help);
    if (agentLinked) {
        const note = document.createElement("div");
        note.textContent = "已接 Media Agent：这里上传的素材排在 Agent 素材之后编号。";
        note.style.cssText = "color:#fbbf24;font-size:9px;line-height:1.5;margin-bottom:6px;";
        panel.appendChild(note);
    }
    if (inputLinked(node, "media") && !agentLinked) {
        const note = document.createElement("div");
        note.textContent = "当前模式不读取外接 Media Agent 素材；这里只使用本模式公共素材，避免跨模式残留。";
        note.style.cssText = "color:#fbbf24;font-size:9px;line-height:1.5;margin-bottom:6px;";
        panel.appendChild(note);
    }
    if (agentLinked) {
        const agentOrderPanel = renderAgentMediaOrderPanel(node);
        if (agentOrderPanel) panel.appendChild(agentOrderPanel);
    }
    panel.appendChild(renderShotAssets(node, bucket, options.prompt || null, {
        title: "公共素材",
        allowedKinds: options.allowedKinds || ["image", "video", "audio"],
        showAssetMode: false,
        resolve: () => globalMediaList(node, {allowAgent: agentLinked}),
        emptyText: "还没有公共素材。文件保存在 ComfyUI/input，工作流只记录引用。",
    }));
    return panel;
}

function renderSkillPanel(node) {
    const preset = String(widget(node, "skill_preset")?.value || "auto");
    const vlm = String(widget(node, "vlm_service")?.value || "off");
    const custom = String(widget(node, "skill_text")?.value || "");
    const panel = document.createElement("div");
    panel.dataset.myangDirectorSection = "skill";
    panel.style.cssText = "border:1px solid #334155;background:#101923;border-radius:6px;padding:7px;margin-bottom:8px;";
    const title = document.createElement("div");
    title.textContent = "写作技能与素材识图";
    title.style.cssText = "color:#93c5fd;font-size:10px;font-weight:700;margin-bottom:5px;";
    const grid = document.createElement("div");
    grid.style.cssText = "display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:6px;";
    grid.appendChild(detailControl(node, "skill_preset", "技能", {values: true}));
    grid.appendChild(detailControl(node, "vlm_service", "素材识图 VLM", {values: true}));
    // 分层只对 Agent 智能切分这条路生效：手动分镜卡的正文由操作者自己写。
    grid.appendChild(detailControl(node, "分层提示词", "分层生成提示词", {type: "checkbox"}));
    panel.append(title, grid);

    const details = document.createElement("details");
    details.open = Boolean(custom.trim());
    details.style.cssText = "margin-top:6px;";
    const summary = document.createElement("summary");
    summary.textContent = custom.trim()
        ? `自定义写作规则（已填 ${custom.trim().length} 字）` : "自定义写作规则（可选）";
    summary.style.cssText = "cursor:pointer;color:#7dd3fc;font-size:9px;font-weight:700;outline:none;";
    const area = document.createElement("textarea");
    area.dataset.myangControl = "skill_text";
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
        : preset === "none" ? "不使用技能，按默认 Easy Prompt 写法拆分。"
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
        add(detailControl(node, "segment_seconds", "智能切分单段上限", {
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
    area.dataset.myangControl = "script_fallback";
    area.id = `myang-script-${node.id}`;
    area.value = String(target?.value || "");
    area.readOnly = linked;
    area.placeholder = linked
        ? "运行时使用左侧接入的 Agent easy_prompt；断开连接后可在这里编辑。"
        : continuing
            ? "填写需要续写的后续剧情；前文视频只走 Motion Context 输入。"
            : "填写长剧本，或在左侧把 Agent easy_prompt 转换并连接到此输入。";
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
    llmGrid.style.cssText = "display:grid;grid-template-columns:minmax(140px,.55fr) minmax(220px,1.25fr) 132px;gap:6px;align-items:end;margin-top:7px;min-width:0;";
    const enabledControl = detailControl(node, "llm_enabled", "智能切片", {type: "checkbox"});
    const serviceControl = detailControl(node, "llm_service", "LLM 服务", {values: true});
    if (enabledControl) llmGrid.appendChild(enabledControl);
    if (serviceControl) llmGrid.appendChild(serviceControl);
    const stopBox = document.createElement("div");
    stopBox.style.cssText = "display:flex;flex-direction:column;gap:2px;min-width:0;";
    const stopLabel = document.createElement("label");
    stopLabel.textContent = "LLM 控制";
    stopLabel.style.cssText = "font-size:9px;color:#94a3b8;line-height:1.2;";
    const stopButton = button(
        "停止 LLM",
        "立即中断当前沐阳 LLM/VLM 网络等待；不会关闭 ComfyUI，也不会误停已经进入的一采/二采");
    stopButton.setAttribute("aria-label", "停止当前导演台 LLM 提示词生成");
    stopButton.style.cssText += ";width:132px;min-height:44px;padding:6px 10px;background:#3a1820;border-color:#9f3448;color:#fecdd3;font-weight:700;transition:background .2s ease,border-color .2s ease,color .2s ease;";
    stopButton.disabled = widget(node, "llm_enabled")?.value === false;
    stopButton.style.opacity = stopButton.disabled ? ".45" : "1";
    stopButton.style.cursor = stopButton.disabled ? "not-allowed" : "pointer";
    stopButton.onfocus = () => { stopButton.style.borderColor = "#fb7185"; };
    stopButton.onblur = () => { stopButton.style.borderColor = "#9f3448"; };
    const stopStatus = document.createElement("div");
    stopStatus.setAttribute("role", "status");
    stopStatus.setAttribute("aria-live", "polite");
    stopStatus.textContent = "卡住时可直接切断";
    stopStatus.style.cssText = "min-height:14px;font-size:8px;line-height:1.35;color:#718096;white-space:normal;";
    stopButton.onclick = () => stopDirectorLlm(node, stopButton, stopStatus);
    stopBox.append(stopLabel, stopButton, stopStatus);
    llmGrid.appendChild(stopBox);
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
            brief: storyboardShortTitle(segment.title, segment.brief || segment.prompt),
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
    const materials = globalMediaList(node, {allowAgent: currentTask(node) === FRESH});
    const state = progressState(node);
    const active = state.status === "running" ? Number(state.seg || 0) : 0;
    segments.forEach((segment, offset) => {
        const running = active > 0 && offset + 1 === active;
        const card = document.createElement("div");
        card.style.cssText = `border:1px solid ${running ? "#60a5fa" : "#334155"};background:${running ? "#132133" : "#0d141d"};border-radius:5px;padding:6px 7px;`;
        const head = document.createElement("div");
        head.style.cssText = "display:flex;align-items:center;gap:6px;margin-bottom:4px;";
        const tag = document.createElement("span");
        const seconds = Number(segment.duration_seconds || 0);
        const shortTitle = storyboardShortTitle(segment.title, segment.brief || segment.prompt);
        tag.textContent = `分镜${segment.index}：${shortTitle}（时长 ${seconds ? seconds.toFixed(2) : "--"}s）`;
        tag.title = tag.textContent;
        tag.style.cssText = `font-size:9px;font-weight:700;color:${running ? "#93c5fd" : "#7dd3fc"};min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;`;
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
        if (Array.isArray(segment.skills) && segment.skills.length) {
            const skillBadge = document.createElement("span");
            skillBadge.textContent = segment.skills.length > 1
                ? `技能 ${segment.skills.length}` : "技能 1";
            skillBadge.title = `本段技能：${segment.skills.join(" + ")}`;
            skillBadge.style.cssText = "flex:none;font-size:8px;padding:1px 4px;border-radius:3px;border:1px solid #4c3f7a;color:#c4b5fd;background:#1c1730;";
            head.appendChild(skillBadge);
        }
        const meta = document.createElement("span");
        meta.textContent = `${segment.frames ? `${segment.frames} 帧` : ""}`;
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

function directorStatsText(node) {
    const task = currentTask(node);
    const transferring = task === TRANSFER;
    const manual = !transferring && String(widget(node, "source_mode")?.value || MANUAL) === MANUAL;
    const importedStoryboard = manual && (node.__myangDirectorShots || []).some(
        (shot) => shot.imported_storyboard === true);
    const fixedStoryboard = manual && (node.__myangDirectorShots || []).some(
        (shot) => shot.fixed_from_plan === true);
    const info = timelineStats(node);
    const turboConnected = (node.inputs || []).some(
        (input) => input.name === "Turbo联合模型" && input.link != null);
    const turboSpec = turboConnected ? turboStepSpec(node) : null;
    const baseStats = transferring
        ? (widget(node, "动作迁移自动分段")?.value === false
            ? "单一动作源 · 整段生成（按视频长度）"
            : `单一动作源 · ${Number(widget(node, "segment_seconds")?.value || 10).toFixed(1)} 秒自动分段`)
        : manual
            ? `${info.count} 个启用镜头 · 成片约 ${info.seconds.toFixed(2)} 秒${importedStoryboard || fixedStoryboard ? " · 不调用 LLM" : ""}`
            : "执行时由分段计划节点一次拆镜头";
    return `${baseStats}${turboConnected ? ` · Turbo ${turboSpec?.label || "官方档位"}` : ""}`;
}

function roughCutDirectorDurationSeconds(node) {
    const task = currentTask(node);
    if (task === TRANSFER) {
        // The full reference-video duration is only known by Python at queue
        // time. For an interactive one-point range, use the Director's own
        // authored segment duration as the predictable editing unit.
        return Math.max(1 / 24, Number(widget(node, "segment_seconds")?.value || 10));
    }
    const manual = String(widget(node, "source_mode")?.value || MANUAL) === MANUAL;
    if (manual) {
        const seconds = Number(timelineStats(node).seconds || 0);
        if (seconds > 0) return seconds;
    }
    return Math.max(1 / 24, Number(widget(node, "total_seconds")?.value || 5));
}

function applyRoughCutDurationSeconds(node, seconds) {
    const value = Math.max(1 / 24, Number(seconds) || 0);
    // Action transfer authors a single transfer chunk; every other Director
    // mode uses the total-duration contract. Python still performs the exact
    // frame-grid fit at execution time, so this UI sync is informative and
    // keeps the visible Director control consistent with the timeline range.
    setNativeWidget(node, currentTask(node) === TRANSFER ? "segment_seconds" : "total_seconds", value);
}

function applyTemplateInterfaceLock(node, root) {
    const lock = node.__myangTemplateLock;
    if (!lock?.enabled) return;
    root.dataset.myangTemplateLocked = "true";
    for (const control of root.querySelectorAll("input,textarea,select,button")) {
        if (control.closest('[data-myang-template-panel="1"]')
            || control.closest('[data-myang-progress-panel="1"]')
            || String(control.getAttribute("aria-label") || "").startsWith("预览素材")) {
            continue;
        }
        control.disabled = true;
        control.setAttribute("aria-disabled", "true");
        control.title = "接口模板已锁定；请在模板面板重新使用模板并填写开放输入";
    }
    for (const editor of root.querySelectorAll('[contenteditable="true"]')) {
        editor.contentEditable = "false";
        editor.setAttribute("aria-readonly", "true");
    }
}

/**
 * The one line a folded card still has to answer: how long, what happens, what
 * is attached, and whether its dialogue fits.
 *
 * Folding is for reviewing a long storyboard, so the summary carries the facts
 * an operator scans for. The dialogue figure turns red on overflow because that
 * is the one problem which is invisible in the finished prompt text and only
 * shows up as a shot talking over its own cut.
 */
function renderCollapsedShotSummary(shot) {
    const box = document.createElement("div");
    box.style.cssText = "display:flex;gap:8px;align-items:baseline;font-size:9px;"
        + "color:#778493;line-height:1.5;min-width:0;";
    const facts = document.createElement("span");
    facts.style.cssText = "flex:0 0 auto;color:#8e9aaa;";
    facts.textContent = `${Number(shot.duration_seconds).toFixed(1)}s`
        + ` · ${alignedFrames(shot.duration_seconds)} 帧`
        + (shot.assets?.length ? ` · 素材 ${shot.assets.length}` : "");
    box.appendChild(facts);

    const layers = shot.layers && typeof shot.layers === "object" ? shot.layers : null;
    if (layers?.dialogue?.length) {
        const window_ = Math.max(0.1, Number(shot.duration_seconds) || 0);
        const spent = layers.dialogue.reduce(
            (total, entry) => total + dialogueSeconds(entry), 0);
        const over = spent > window_ + 0.05;
        const talk = document.createElement("span");
        talk.style.cssText = `flex:0 0 auto;color:${over ? "#fb7185" : "#86efac"};`;
        talk.textContent = `台词 ${layers.dialogue.length} 句 ${spent.toFixed(1)}/${window_.toFixed(1)}s`;
        talk.title = over ? "超出本段时长，运行时会顺延到下一段" : "台词能在本段说完";
        box.appendChild(talk);
    }

    const preview = document.createElement("span");
    preview.style.cssText = "flex:1;min-width:0;overflow:hidden;"
        + "text-overflow:ellipsis;white-space:nowrap;";
    const body = String((layers ? layers.visual : shot.prompt) || "")
        .replace(/\s+/g, " ").trim();
    preview.textContent = body || "（还没有提示词）";
    preview.title = body;
    box.appendChild(preview);
    return box;
}

function renderTimeline(node) {
    const root = node.__myangDirectorRoot;
    if (!root?.isConnected) return;
    const view = captureDirectorView(node);
    // Keep the same progress DOM across legitimate full form rebuilds (mode
    // changes, storyboard import, material edits). Recreating it used to blank
    // the current step and preview even though the sampling run was untouched.
    const progressPanel = node.__myangDirectorProgressEls?.panel || null;
    root.replaceChildren();
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
    stats.dataset.myangDirectorStats = "1";
    stats.style.cssText = "font-size:10px;color:#8e9aaa;";
    stats.textContent = directorStatsText(node);
    header.appendChild(stats);
    root.appendChild(header);
    const liveProgressPanel = progressPanel || renderDirectorProgressPanel(node);
    liveProgressPanel.dataset.myangProgressPanel = "1";
    root.appendChild(liveProgressPanel);
    // renderDirectorProgressPanel builds its nodes before they are connected;
    // update once after mounting for both first render and preserved renders.
    updateDirectorProgress(node);
    if (node.__myangTemplateLock?.enabled) {
        const locked = document.createElement("div");
        locked.setAttribute("role", "status");
        locked.style.cssText = "border:1px solid #6d55a5;border-radius:6px;background:#19142a;color:#d8ccff;padding:7px 9px;margin-bottom:8px;font-size:9px;line-height:1.5;";
        locked.textContent = `接口模板「${node.__myangTemplateLock.template_name || "未命名"}」已锁定：固定内容只读；请从模板面板重新使用模板来更换开放输入。`;
        root.appendChild(locked);
    }
    root.appendChild(renderRoughCutCard(node, {
        directorDurationSeconds: roughCutDirectorDurationSeconds(node),
        onDurationSeconds: (seconds) => applyRoughCutDurationSeconds(node, seconds),
    }));
    if (!transferring) root.appendChild(renderSourcePanel(node, {manual, continuing}));
    root.appendChild(renderGenerationSettingsPanel(node));
    // Agent/长剧本每次都会重新规划分段，段号不具备稳定的断点语义；
    // 只有手动分镜卡和动作迁移显示这组控件。
    if (!transferring && manual) root.appendChild(renderResumePanel(node));
    // These are generation-wide post-processing controls.  Keep them outside
    // every source-mode branch so manual cards, Agent splitting, continuation
    // and action transfer all expose exactly the same second-pass toolchain.
    root.appendChild(renderFirstPassMemoryPanel(node));
    root.appendChild(renderDetailPanel(node));
    root.appendChild(renderAudioPanel(node));
    root.appendChild(renderEnhancementPanel(node));

    if (transferring) {
        renderTransferPanel(node, root);
        root.appendChild(renderDirectorTemplatePanel(node));
        root.appendChild(renderAssetLibraryPanel(node));
        root.appendChild(renderOutputVideoPanel(node));
        applyTemplateInterfaceLock(node, root);
        restoreDirectorView(node, view);
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
    root.appendChild(renderAssetLibraryPanel(node));
    if (!manual) root.appendChild(renderSkillPanel(node));
    if (!manual) root.appendChild(renderSegmentPlanPanel(node));

    if (!manual) {
        const help = document.createElement("div");
        help.style.cssText = "border:1px solid #334155;background:#17202b;border-radius:7px;padding:12px;color:#aab7c6;font-size:11px;line-height:1.6;";
        help.textContent = continuing
            ? "把要续写的剧情接到 script 或填写长剧本。前文视频仍只走左侧 Motion Context 输入；LLM 只负责拆分未来剧情，并按上方公共素材清单逐段分配 @图片N 标签。"
            : "把 Media Agent 的 easy_prompt 接到 script，或在下方填写长剧本。LLM 开启时逐镜头拆分，并按上方公共素材清单自行判断每段引用哪些素材；关闭时各段共用原提示词和全部公共素材，不消耗 Token。";
        root.appendChild(help);
        root.appendChild(renderOutputVideoPanel(node));
        applyTemplateInterfaceLock(node, root);
        restoreDirectorView(node, view);
        return;
    }

    root.appendChild(renderDirectorTemplatePanel(node));

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
    const shots = node.__myangDirectorShots || [];
    const anyOpen = shots.some((shot) => !shot.collapsed);    const foldAll = button(anyOpen ? "全部折叠" : "全部展开",
        anyOpen ? "折叠所有镜头卡，只保留标题行" : "展开所有镜头卡");
    foldAll.setAttribute("aria-label", anyOpen ? "折叠所有镜头卡" : "展开所有镜头卡");
    foldAll.dataset.myangControl = "storyboard:fold-all";
    foldAll.style.cssText += ";flex:0 0 68px;width:68px;min-height:30px;";
    foldAll.onclick = () => {
        // A new card should appear expanded, so the button state follows "is
        // anything still open" rather than a separate remembered mode.
        for (const shot of shots) shot.collapsed = anyOpen;
        saveTimeline(node); renderTimeline(node);
    };
    const rebind = button("素材匹配", "让 LLM 把新素材按主体/说话人挂到对应分镜，不改剧本一个字");
    rebind.setAttribute("aria-label", "按主体匹配素材，不修改剧情");
    rebind.style.cssText += ";flex:0 0 68px;width:68px;min-height:30px;";
    rebind.onclick = () => runMaterialRebind(node, rebind);
    toolbarActions.append(importCards, exportCards, rebind, foldAll, add);
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
        const transitionControl = renderShotTransition(node, shot, index);
        if (transitionControl) list.appendChild(transitionControl);
        const card = document.createElement("div");
        card.style.cssText = `border:1px solid ${shot.enabled ? "#42546a" : "#30343b"};background:${shot.enabled ? "#1a222c" : "#191b20"};border-radius:7px;padding:8px;opacity:${shot.enabled ? "1" : ".58"};`;
        const top = document.createElement("div");
        top.style.cssText = "display:grid;grid-template-columns:18px 24px 1fr 82px auto;gap:6px;align-items:center;margin-bottom:6px;";
        const fold = button(shot.collapsed ? "▸" : "▾",
            shot.collapsed ? "展开这个镜头" : "折叠这个镜头（只留标题行）");
        fold.setAttribute("aria-expanded", shot.collapsed ? "false" : "true");
        // Folding rebuilds the whole form, so carry a control key and let
        // restoreDirectorView hand focus back to this same button.
        fold.dataset.myangControl = `shot:${shot.id}:fold`;
        fold.style.cssText += ";padding:0;min-width:18px;color:#7dd3fc;";
        fold.onclick = () => {
            shot.collapsed = !shot.collapsed;
            saveTimeline(node); renderTimeline(node);
        };
        top.appendChild(fold);
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
        brief.dataset.myangControl = `shot:${shot.id}:brief`;
        brief.value = shot.brief;
        brief.placeholder = `镜头 ${index + 1} 标题`;
        brief.style.cssText = "background:#111820;color:#e7edf4;border:1px solid #344254;border-radius:4px;padding:5px 7px;min-width:0;";
        brief.oninput = () => { shot.brief = brief.value; saveTimeline(node); };
        top.appendChild(brief);
        const duration = document.createElement("input");
        duration.dataset.myangControl = `shot:${shot.id}:duration`;
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
        const revise = button("✎AI", "让 LLM 只改写这一个镜头（改剧情）");
        revise.style.color = "#a3e635";
        revise.onclick = () => runShotRewrite(node, shot, index, revise);
        actions.append(up, down, duplicate, revise, remove);
        top.appendChild(actions);
        card.appendChild(top);

        if (shot.collapsed) {
            // Deliberately not building the editor at all rather than hiding it
            // with CSS: a long storyboard's cost is the contenteditable bodies
            // and their material menus, so a folded card should not create them.
            card.appendChild(renderCollapsedShotSummary(shot));
            list.appendChild(card);
            return;
        }

        const prompt = createPromptEditor(node, shot, {
            placeholder: "写完整 H3 提示词；键入 @ 选择素材，@图片1 / @视频1 会自动显示为匹配标签",
            // A layered card keeps its prose in the visual layer and recomposes
            // the flat prompt, so the two can never disagree.
            read: (target) => (target.layers
                ? String(target.layers.visual || "") : target.prompt),
            write: (target, text) => {
                if (!target.layers) { target.prompt = text; return; }
                target.layers.visual = text;
                target.prompt = composeSegmentPrompt(
                    directorGlobalLayers(node), target.layers);
            },
        });
        card.append(prompt.__myangDialogueToolbar, prompt);
        card.appendChild(renderShotAssets(node, shot, prompt, continuing ? {
            title: "续写辅助图片 / 音频",
            allowedKinds: ["image", "audio"],
            emptyText: "前文视频不要放在镜头素材里；它只接左侧 Motion Context 视频输入。",
        } : {}));
        card.appendChild(renderShotLayers(node, shot, prompt));

        const grid = document.createElement("div");
        grid.style.cssText = "font-size:9px;color:#778493;margin-top:4px;";
        grid.textContent = `${Number(shot.duration_seconds).toFixed(1)}s → ${alignedFrames(shot.duration_seconds)} 帧 @24fps`;
        card.appendChild(grid);
        list.appendChild(card);
    });
    root.appendChild(list);
    root.appendChild(renderOutputVideoPanel(node));
    applyTemplateInterfaceLock(node, root);
    restoreDirectorView(node, view);
}

function refresh(node) {
    if (syncModeBucket(node)) saveTimeline(node);
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
    hideWidget(by["粗剪时间轴开启"]);
    hideWidget(by[ROUGHCUT_WIDGET]);
    for (const name of DETAIL_FIELDS) hideWidget(by[name]);
    for (const name of AUDIO_FIELDS) hideWidget(by[name]);
    for (const name of ENHANCEMENT_FIELDS) hideWidget(by[name]);
    for (const name of REFERENCE_FIELDS) hideWidget(by[name]);
    for (const name of SKILL_FIELDS) hideWidget(by[name]);
    for (const name of FIRST_MEMORY_FIELDS) hideWidget(by[name]);
    hideWidget(by["起始段"]);
    hideWidget(by["从指定段开始"]);
    // Source controls are rendered inside the full-width Director source card.
    // Keeping the native multiline widget visible made it retain ComfyUI's
    // narrow pre-resize width and pushed the universal second-pass cards away.
    for (const name of [
        "source_mode", "script_fallback", "total_seconds", "segment_seconds",
        "llm_enabled", "llm_service", "动作迁移自动分段",
    ]) hideWidget(by[name]);
    // Keep the generation controls together in the Director card. Seed and
    // first-pass steps intentionally remain native so they can still be wired
    // from external nodes.
    for (const name of [
        "task_mode", "resolution", "aspect_ratio", "width", "height",
        "denoise", "scheduler", "context_length", "ref_image_size",
        "save_segments", "segment_prefix",
    ]) hideWidget(by[name]);
    // These controls are mirrored in the in-card generation settings even
    // when a custom resolution or segment prefix is selected.
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
    // denoise/scheduler are rendered in the in-card generation settings. Turbo
    // profiles may constrain them at execution time, but must not reinsert the
    // native widgets and split the Director layout.
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
            input.label = task === FRESH ? "Media Agent 素材包（纯生成模式）"
                : "Media Agent 素材包（当前模式忽略）";
        } else if (input.name === "前段视频") {
            input.label = transferring || manual
                ? "断点前一段成片（仅承接上下文，不当参考视频）"
                : "Agent 模式不使用断点前段视频";
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
    const panelWidget = node.__myangDirectorPanelWidget;
    const panelTop = Number(panelWidget?.y);
    const nodeHeight = Math.max(520, Number(node.size?.[1] || 920));
    // Fill the live node body below its sockets/widgets. Do not derive this
    // from the node's creation-time height: that leaves a large blue canvas
    // strip when the user later makes the node taller.
    const height = Math.max(260, Math.floor(
        nodeHeight - (Number.isFinite(panelTop) && panelTop > 0 ? panelTop : 28) - 8));
    const cssHeight = `${height}px`;
    if (panelWidget) panelWidget.computedHeight = height;
    for (const element of [root, root.parentElement]) {
        if (!element) continue;
        element.style.boxSizing = "border-box";
        element.style.width = cssWidth;
        element.style.minWidth = cssWidth;
        element.style.maxWidth = cssWidth;
        element.style.height = cssHeight;
        element.style.minHeight = cssHeight;
        element.style.maxHeight = cssHeight;
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
                        syncPanelGeometry(node);
                        mountNativeVideoPreview(node);
                    }
                } catch (error) {
                    console.warn("[Myang Director] 素材 / Turbo 状态同步失败", error);
                }
            }, 1000);
        }
        // ComfyUI publishes this before the Director's Python expansion runs.
        // Starting the card here removes the long "等待执行" gap while graph
        // planning, VLM/LLM work and material binding are still preparing.
        api.addEventListener("executing", (event) => {
            try {
                const detail = event?.detail;
                const nodeId = detail && typeof detail === "object"
                    ? (detail.node ?? detail.display_node) : detail;
                if (nodeId == null) return;
                const node = (app.graph?._nodes || []).find(
                    (candidate) => String(candidate.id) === String(nodeId));
                if (node?.type === NODE) {
                    markDirectorPreparing(
                        node, "已进入导演台 · 读取模式、分镜卡与运行策略");
                }
            } catch (error) {
                console.warn("[Myang Director] 进入准备状态失败", error);
            }
        });
        api.addEventListener("myh3_director_plan", (event) => {
            try {
                const detail = event?.detail || {};
                const node = directorByOwner(detail.owner_id);
                if (!node) return;
                node.__myangDirectorPlan = normalizePlanSnapshot({
                    ...detail,
                    saved_at: new Date().toISOString(),
                });
                const state = progressState(node);
                if (state.status !== "done") {
                    state.status = "running";
                    state.phase = "prepare";
                    state.total = Math.max(1, Number(detail.segment_count || 1));
                    state.activity = `分镜计划已就绪（${state.total} 段） · 正在绑定素材并构建采样链`;
                    updateDirectorProgress(node);
                }
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
        api.addEventListener("myh3_roughcut_commit", (event) => {
            try {
                const detail = event?.detail || {};
                const node = directorByOwner(detail.owner_id);
                if (!node) return;
                if (!applyRoughCutCommit(node, detail)) return;
                node.__myangDirectorPendingVideoOutput = {videos: [detail.video]};
                scheduleNativeVideoMount(node, node.__myangDirectorPendingVideoOutput);
            } catch (error) {
                console.warn("[Myang Director] 粗剪时间轴写回失败", error);
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
                    phase: "prepare", step: 0, stepMax: 0,
                    previewFile: "", previewTs: 0, prompt: "", brief: "",
                    refining: !!detail.refining,
                    audioRefine: !!detail.audio_refine,
                    correcting: !!detail.correcting,
                    activity: "采样链已展开 · 正在整理第 1 段提示词、素材与条件编码",
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
                if (node) {
                    // A transient preview message can populate node.imgs just
                    // before this progress event. Remove it immediately; the
                    // same still belongs only in the compact progress card.
                    clearNativeStillPreview(node);
                    applyDirectorProgress(node, detail);
                }
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
                scheduleNativeVideoMount(
                    node, node.__myangDirectorPendingVideoOutput);
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
            // Covers interrupts from ComfyUI's toolbar, keyboard shortcuts and
            // other extensions in addition to the Director's own stop button.
            void requestDirectorMemoryRelease().catch(
                (error) => console.warn("[Myang Director] 中断后释放内存失败", error));
        });
        api.addEventListener("myh3_llm_stream", (event) => {
            try {
                const detail = event?.detail || {};
                const node = directorForLlmDiagnostic(detail);
                if (!node) return;
                const state = progressState(node);
                // LLM generation belongs to preparation. Once sampling has
                // started, a late websocket packet must not rewind progress.
                if (state.status !== "running" || state.phase !== "prepare") return;
                const activity = llmDiagnosticActivity(detail);
                if (!activity) return;
                state.llmCallId = String(detail.call_id || state.llmCallId || "");
                state.activity = activity;
                updateDirectorProgress(node);
            } catch (error) {
                console.warn("[Myang Director] 更新 LLM 流式诊断失败", error);
            }
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
            installCanvasPreviewGuard(this);
            hideNativeVideoWidget(this);
            this.__myangDirectorModeBuckets = parseModeBuckets(this);
            loadModeBucket(this, modeTaskValue(this));
            const panel = this.addDOMWidget(
                "myang_director_panel", "director", makePanel(this), {
                    serialize: false,
                    hideOnZoom: false,
                    getMinHeight: () => 260,
                    // Let LiteGraph allocate every remaining pixel to the one
                    // Director card. `syncPanelGeometry` then mirrors that
                    // allocation to the DOM wrapper on every resize.
                    getMaxHeight: () => Number.MAX_SAFE_INTEGER,
                });
            panel.serialize = false;
            this.__myangDirectorPanelWidget = panel;
            for (const item of this.widgets || []) {
                if (item === panel) continue;
                const previous = item.callback;
                const syncedCallback = (...args) => {
                    const value = previous?.apply(item, args);
                    if (item.name !== TIMELINE_WIDGET
                        && item.name !== ROUGHCUT_WIDGET
                        && !this.__myangDirectorSaving) {
                        syncDirectorWidgetChange(this, item.name);
                    }
                    return value;
                };
                syncedCallback.__myangDirectorSyncs = true;
                item.callback = syncedCallback;
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
                    this.__myangDirectorModeBuckets = parseModeBuckets(this);
                    loadModeBucket(this, modeTaskValue(this));
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
            // New frontend builds arrange DOM widgets after onResize; run once
            // in each of the next two frames so our card follows the final y.
            requestAnimationFrame(() => {
                syncPanelGeometry(this);
                requestAnimationFrame(() => syncPanelGeometry(this));
            });
            return result;
        };

        // Clearing node.imgs only from onExecuted is one paint too late: the
        // canvas can draw a full-node sampling still for a single frame and
        // then remove it, which looks like a large flash. Strip only that
        // canvas still state immediately before every paint. The progress-card
        // <img> and the native HTMLVideoElement use separate DOM state and are
        // therefore preserved.
        const onDrawBackground = nodeType.prototype.onDrawBackground;
        nodeType.prototype.onDrawBackground = function () {
            clearNativeStillPreviewState(this);
            return onDrawBackground?.apply(this, arguments);
        };

        // Keep the native video output, but never pass temporary sampling
        // stills to ComfyUI's node-canvas image renderer. Those stills are
        // already shown in the compact "生成进度" card.
        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (output) {
            const stableSize = [Number(this.size?.[0] || 0), Number(this.size?.[1] || 0)];
            const result = onExecuted?.call(this, outputWithoutNativeMediaPreview(output));
            clearNativeStillPreviewSoon(this);
            scheduleNativeVideoMount(this, output);
            if (stableSize[0] > 0 && stableSize[1] > 0
                && (Number(this.size?.[0]) !== stableSize[0]
                    || Number(this.size?.[1]) !== stableSize[1])) {
                this.setSize?.(stableSize);
            }
            return result;
        };
    },
});
