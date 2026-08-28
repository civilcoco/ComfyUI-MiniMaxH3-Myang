// 沐阳 H3 · 长视频节点的提示词编辑器
//
// The prompt is plain text the whole way down -- "@图片1" mentions and "<d>台词</d>" blocks
// are exactly what the H3 sampler already parses -- so this is purely a nicer
// way to look at and write that text.
//
// Two rules this file follows, both learned the hard way:
//
//   1. Never let an exception escape. Extensions share hooks with every other
//      pack; one throw inside a shared hook takes unrelated UI down with it.
//      Every entry point below is wrapped, and nothing patches app.graph.
//   2. Never change how many widgets the node has, or their order. The frontend
//      maps a saved workflow's widgets_values onto widgets *positionally*
//      (migrateWidgetsValues bails out and returns the array untouched when the
//      lengths disagree), so an extra widget shifts every value after it into
//      the wrong field. Hiding is fine -- it keeps the index -- adding is not.

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE = "H3LongVideo";
const DETAIL_NODE = "H3DetailSettings";
const AGENT_LINKS = "myang_h3_asset_sources_v2";
const PREVIEW_PROP = "myh3_upstream_prompt";
const TEXT_PROP = "myh3_prompt";

const KINDS = ["图片", "视频", "音频"];
const KIND_OF_TYPE = { image: "图片", video: "视频", audio: "音频" };
const GLYPH = { 图片: "▣", 视频: "▶", 音频: "♪" };
const IMAGE_EXT = /\.(png|jpe?g|webp|gif|bmp|avif)$/i;
const VIDEO_EXT = /\.(mp4|webm|mov|mkv|avi|m4v)$/i;

// Same grammar the sampler accepts: editor mentions (@图片1) and official tags (<Picture 1>).
const TAG_MAP = { picture: "图片", video: "视频", audio: "音频" };
const MENTION_RE = /(@(图片|视频|音频)[ \t_]*(\d+)|<(Picture|Video|Audio)[ \t_]*(\d+)>)/gi;
const DIALOGUE_RE = /<d>([\s\S]*?)<\/d>/g;

const LABELS = {
    task_mode: "任务模式",
    prompt: "提示词（全片共用）",
    reference_mention_mode: "@ 引用解析方式",
    resolution: "分辨率档位",
    aspect_ratio: "画面比例",
    width: "自定义宽",
    height: "自定义高",
    steps: "采样步数",
    denoise: "重绘幅度",
    scheduler: "调度器",
    noise_seed: "种子",
    context_length: "段间重叠帧数",
    prompt_mode: "分段提示词来源",
    media_prefix: "分段提示词前缀",
    ref_image_size: "参考图尺寸预算",
    save_segments: "每段单独保存",
    segment_prefix: "分段文件名前缀",
};

const SLOT_LABELS = {
    h3_bundle: "H3 模型包（VAE / 文本编码）",
    h3: "H3 模型包",
    model: "模型（已挂补丁）",
    sampler: "采样器",
    plan_json: "分段计划（含分段提示词与时间线）",
    ref_video: "参考视频（完整）",
    ref_audio: "参考音频（续写用）",
    prompt: "全片共用提示词（可选覆盖·留空则自动使用 plan_json 的分段提示词）",
    media: "素材（来自 Agent）",
};

const DETAIL_LABELS = {
    enabled: "开启二采",
    mode: "处理模式",
    resolution: "二采输出短边",
    width: "自定义宽",
    height: "自定义高",
    steps: "二采步数",
    denoise: "二采重绘幅度",
    scheduler: "调度器",
    sampler_name: "采样器",
    upscale_method: "放大方式",
    chunk_frames: "像素分组帧数",
    latent_upscale_model: "神经 3D 权重",
    latent_precision: "神经放大精度",
    latent_chunk_steps: "神经时间分块",
    passes: "二采轮数",
    seed_mode: "多轮种子",
};

let warned = false;

// ---- 执行进度 ----
// 一段长视频内部细分为几个阶段，每个阶段在进度条上占一段权重。带 steps 的阶段
// （一采/二采）还会用采样步数在该段内做精细插值，所以进度条能跟着步数走。
const PHASE = {
    sample1:    { start: 0.00, span: 0.40, label: "一采采样", steps: true  },
    drift:      { start: 0.40, span: 0.10, label: "漂移校正", steps: false },
    refine_prep:{ start: 0.50, span: 0.10, label: "二采准备", steps: false },
    sample2:    { start: 0.60, span: 0.25, label: "二采采样", steps: true  },
    finalizing: { start: 0.85, span: 0.15, label: "解码保存", steps: false },
};
// 一次队列通常只跑一个长视频任务，用全局变量记这次 run 的实时状态。
let activeRun = null;

/** 把当前 activeRun 状态渲染到某个 H3LongVideo 节点的进度面板上。 */
function renderProgress(node) {
    if (!activeRun) return;
    const seg = activeRun.seg || 1;
    const total = activeRun.total || 1;
    const phase = activeRun.phase || "sample1";
    const step = activeRun.step || 0;
    const stepMax = activeRun.stepMax || 0;
    const wrap = node.__myh3Progress;
    if (!wrap) return;
    wrap.hidden = false;
    const fill = node.__myh3ProgressFill;
    const text = node.__myh3ProgressText;
    const prev = node.__myh3ProgressPreview;
    // 进度条 = 段 + 阶段 + 步数精细插值
    const ph = PHASE[phase] || PHASE.sample1;
    let segInner = ph.start;
    if (phase === "done") {
        segInner = 1.0;
    } else if (ph.steps && stepMax > 0) {
        segInner += (step / stepMax) * ph.span;
    } else {
        segInner += ph.span * 0.5;  // 无步数阶段，取段内中点
    }
    const pct = total > 0 ? ((Math.max(1, seg) - 1 + Math.min(1, segInner)) / total) * 100 : 0;
    if (fill) fill.style.width = Math.min(100, pct).toFixed(1) + "%";
    // 阶段文本（带步数）
    if (text) {
        text.classList.remove("is-done", "is-error");
        if (phase === "done") {
            text.classList.add("is-done");
            text.textContent = `✔ 全部 ${total} 段完成`;
        } else {
            text.classList.add("is-running");
            let s = `第 ${seg}/${total} 段 · ${ph.label}`;
            if (ph.steps && stepMax > 0) s += ` · 第 ${step}/${stepMax} 步`;
            else s += "…";
            text.textContent = s;
        }
    }
    // 预览帧：URL 自带时间戳，绕开浏览器缓存。
    if (activeRun.previewFile && prev) {
        prev.hidden = false;
        prev.src = `/api/view?filename=${encodeURIComponent(activeRun.previewFile)}&type=temp&subfolder=&_t=${activeRun.previewTs || Date.now()}`;
    }
    // hint 简短状态
    if (node.__myh3Hint) {
        if (phase === "done") {
            node.__myh3Hint.textContent = "✔ 完成";
        } else {
            let h = `▶ 第 ${seg}/${total} 段 · ${ph.label}`;
            if (ph.steps && stepMax > 0) h += ` ${step}/${stepMax}`;
            node.__myh3Hint.textContent = h;
            node.__myh3Hint.classList.add("is-linked");
        }
    }
    node.graph?.setDirtyCanvas?.(true, true);
}

/** 处理一次 myh3_progress 事件（阶段级，来自子图里的 H3ProgressSignal）。 */
function applyProgress(node, d) {
    const seg = Number(d?.segment_index || 1);
    const total = Number(d?.total_segments || (activeRun?.total ?? 1)) || 1;
    const stage = String(d?.stage || "");
    const refining = !!(activeRun?.refining);
    const correcting = !!(activeRun?.correcting);

    // 步级预览：来自 H3SamplerAdvanced 的 sampling 事件
    // 每步采样更新步数和清晰预览帧，不走 stage → phase 状态机
    if (stage === "sampling" && activeRun) {
        activeRun.seg = seg;
        activeRun.total = total;
        activeRun.step = Number(d?.step || 0);
        activeRun.stepMax = Number(d?.step_total || 0);
        // 根据 pass_label 确定当前是一采还是二采
        const passLabel = String(d?.pass_label || "sample1");
        if (passLabel === "sample2") {
            activeRun.phase = "sample2";
        } else {
            activeRun.phase = "sample1";
        }
        if (d?.preview_file) {
            activeRun.previewFile = d.preview_file;
            activeRun.previewTs = d.preview_ts || Date.now();
        }
        renderProgress(node);
        return;
    }

    if (activeRun) {
        activeRun.seg = seg;
        activeRun.total = total;
        // 进入新阶段，清掉上一阶段的步数，等 progress 事件重新填
        activeRun.step = 0;
        activeRun.stepMax = 0;
        // stage → phase 状态机：推断当前在一采/漂移/二采准备/二采采样/收尾
        if (stage === "sampled") {
            activeRun.phase = correcting ? "drift" : (refining ? "refine_prep" : "finalizing");
        } else if (stage === "drifted") {
            activeRun.phase = refining ? "refine_prep" : "finalizing";
        } else if (stage === "refine_start") {
            activeRun.phase = "refine_prep";
        } else if (stage === "refined") {
            activeRun.phase = "finalizing";
        } else if (stage === "done") {
            if (seg < total) {
                activeRun.seg = seg + 1;
                activeRun.phase = "sample1";
                activeRun.step = 0;
                activeRun.stepMax = 0;
            } else {
                activeRun.phase = "done";
            }
        }
        if (d?.preview_file) {
            activeRun.previewFile = d.preview_file;
            activeRun.previewTs = d.preview_ts || Date.now();
        }
    }
    // plan_json 模式下，把当前段提示词实时渲染进编辑器（signal 在执行到该段才发，
    // 所以这是真正"当前段"的提示词，而不是构图阶段一闪而过的最后一段）。
    const promptText = String(d?.prompt || "").trim();
    if (promptText && node.__myh3Editor && linkedInput(node, "plan_json")) {
        node.__myh3CurrentSegmentPrompt = promptText;
        renderInto(node.__myh3Editor, promptText, mediaList(node));
        node.__myh3Editor.classList.add("is-preview");
    }
    renderProgress(node);
}

/** Run f, and never let it escape into ComfyUI. */
function guard(f) {
    return function guarded(...args) {
        try {
            return f.apply(this, args);
        } catch (err) {
            if (!warned) {
                warned = true;
                console.error("[沐阳 H3] 提示词编辑器出错，已降级为普通输入框：", err);
            }
            return undefined;
        }
    };
}

function styleOnce() {
    if (document.getElementById("myh3-style")) return;
    const link = document.createElement("link");
    link.id = "myh3-style";
    link.rel = "stylesheet";
    link.href = new URL("./h3_prompt_editor.css", import.meta.url).href;
    document.head.appendChild(link);
}

// ---------------------------------------------------------------- media ----

function viewUrl(name) {
    return `/api/view?filename=${encodeURIComponent(name)}&type=input&subfolder=`;
}

function linkedInput(node, name) {
    const input = node?.inputs?.find((i) => i?.name === name);
    return input && input.link != null ? input : null;
}

function upstream(node, name) {
    const input = linkedInput(node, name);
    if (!input || !node.graph) return null;
    const link = node.graph.links?.[input.link];
    return link ? node.graph.getNodeById?.(link.origin_id) : null;
}

/** Media available here, numbered exactly as the sampler will number it.
 *
 * Read off the Agent's registry rather than re-derived: the Agent decides what
 * @图片1 means, so a second numbering computed here could disagree with the one
 * the prompt is actually resolved against.
 */
function mediaList(node) {
    const agent = upstream(node, "media");
    const raw = agent?.properties?.[AGENT_LINKS];
    if (!Array.isArray(raw)) return [];
    const counts = {};
    return raw.map((link) => {
        const kind = KIND_OF_TYPE[String(link?.media_type || "image").toLowerCase()] || "图片";
        counts[kind] = (counts[kind] || 0) + 1;
        const file = String(link?.filename || "");
        return {
            kind,
            ordinal: counts[kind],
            token: `@${kind}${counts[kind]}`,
            name: String(link?.subject || "") || file || String(link?.label || ""),
            subject: String(link?.subject || "").trim(),
            file,
        };
    });
}

function mediaSignature(node) {
    return mediaList(node).map((m) => `${m.token}|${m.file}`).join(",");
}

function glyph(entry) {
    const el = document.createElement("span");
    el.className = "myh3-chip-thumb";
    el.textContent = GLYPH[entry?.kind] || "▣";
    return el;
}

function makeThumb(entry) {
    const file = entry?.file || "";
    if (entry?.kind === "图片" && IMAGE_EXT.test(file)) {
        const img = document.createElement("img");
        img.className = "myh3-chip-thumb";
        img.src = viewUrl(file);
        img.onerror = () => img.replaceWith(glyph(entry));
        return img;
    }
    if (entry?.kind === "视频" && VIDEO_EXT.test(file)) {
        const vid = document.createElement("video");
        vid.className = "myh3-chip-thumb";
        vid.src = viewUrl(file);
        vid.muted = true;
        vid.preload = "metadata";
        return vid;
    }
    return glyph(entry);
}

// --------------------------------------------------------------- render ----

function makeChip(kind, ordinal, entry) {
    const chip = document.createElement("span");
    chip.className = "myh3-chip";
    chip.dataset.kind = kind;
    chip.dataset.token = `@${kind}${ordinal}`;
    chip.contentEditable = "false";
    if (!entry) chip.classList.add("is-missing");
    chip.appendChild(makeThumb(entry || { kind }));
    const text = document.createElement("span");
    text.textContent = entry?.subject
        ? `${kind}${ordinal} · ${entry.subject}` : `${kind}${ordinal}`;
    chip.appendChild(text);
    chip.title = entry
        ? `${chip.dataset.token}\n${entry.file || entry.name}`
        : `${chip.dataset.token}\n没有对应素材 —— 检查 Agent 上挂了几个${kind}`;
    return chip;
}

function makeDialogue(text) {
    const block = document.createElement("span");
    block.className = "myh3-line";
    block.dataset.dialogue = "1";
    block.textContent = text;
    block.title = "台词块，会以 <d>…</d> 送给模型";
    return block;
}

function appendText(container, chunk) {
    for (const [i, piece] of String(chunk).split("\n").entries()) {
        if (i) container.appendChild(document.createElement("br"));
        if (piece) container.appendChild(document.createTextNode(piece));
    }
}

function renderMentions(container, text, byToken) {
    MENTION_RE.lastIndex = 0;
    let cursor = 0;
    for (let m = MENTION_RE.exec(text); m; m = MENTION_RE.exec(text)) {
        if (m.index > cursor) appendText(container, text.slice(cursor, m.index));
        let kind = m[2] || TAG_MAP[(m[4] || "").toLowerCase()] || "图片";
        let ordinal = Number(m[3] || m[5] || 1);
        const token = `@${kind}${ordinal}`;
        container.appendChild(makeChip(kind, ordinal, byToken.get(token)));
        cursor = m.index + m[0].length;
    }
    if (cursor < text.length) appendText(container, text.slice(cursor));
}

/** Plain text -> rich nodes. */
function renderInto(container, text, list) {
    container.replaceChildren();
    const byToken = new Map(list.map((m) => [m.token, m]));
    const src = String(text || "");
    let cursor = 0;
    DIALOGUE_RE.lastIndex = 0;
    for (let m = DIALOGUE_RE.exec(src); m; m = DIALOGUE_RE.exec(src)) {
        if (m.index > cursor) renderMentions(container, src.slice(cursor, m.index), byToken);
        container.appendChild(makeDialogue(m[1]));
        cursor = m.index + m[0].length;
    }
    if (cursor < src.length) renderMentions(container, src.slice(cursor), byToken);
}

/** Rich nodes -> plain text. Exactly inverts renderInto. */
function readText(container) {
    let out = "";
    const walk = (node) => {
        for (const child of node.childNodes) {
            if (child.nodeType === Node.TEXT_NODE) { out += child.nodeValue; continue; }
            if (child.nodeName === "BR") { out += "\n"; continue; }
            if (child.dataset?.token) { out += child.dataset.token; continue; }
            if (child.dataset?.dialogue) { out += `<d>${child.textContent}</d>`; continue; }
            if (child.nodeName === "DIV" || child.nodeName === "P") { out += "\n"; walk(child); continue; }
            walk(child);
        }
    };
    walk(container);
    return out.replace(/ /g, " ");
}

// ----------------------------------------------------------------- menu ----

let openMenu = null;

function closeMenu() {
    openMenu?.remove();
    openMenu = null;
}

function caretRect(editor) {
    const selection = window.getSelection();
    if (!selection?.rangeCount) return editor.getBoundingClientRect();
    const range = selection.getRangeAt(0).cloneRange();
    range.collapse(true);
    const rect = range.getBoundingClientRect();
    return rect && (rect.width || rect.height || rect.top) ? rect : editor.getBoundingClientRect();
}

function showMenu(editor, list, onPick) {
    closeMenu();
    const menu = document.createElement("div");
    menu.className = "myh3-menu";
    if (!list.length) {
        const empty = document.createElement("div");
        empty.className = "myh3-menu-empty";
        empty.textContent = "没有可引用的素材 —— 先把 Agent 的 media 接进来";
        menu.appendChild(empty);
    }
    for (const entry of list) {
        const item = document.createElement("div");
        item.className = "myh3-menu-item";
        item.appendChild(makeThumb(entry));
        const label = document.createElement("span");
        label.textContent = entry.token;
        item.appendChild(label);
        if (entry.name || entry.file) {
            const file = document.createElement("span");
            file.className = "myh3-menu-file";
            file.textContent = entry.name || entry.file;
            item.appendChild(file);
        }
        item.onmousedown = guard((event) => { event.preventDefault(); onPick(entry); closeMenu(); });
        menu.appendChild(item);
    }
    const rect = caretRect(editor);
    menu.style.left = `${Math.round(rect.left)}px`;
    menu.style.top = `${Math.round(rect.bottom + 4)}px`;
    document.body.appendChild(menu);
    openMenu = menu;
}

function insertAtCaret(editor, fragment) {
    const selection = window.getSelection();
    if (!selection?.rangeCount || !editor.contains(selection.anchorNode)) {
        editor.appendChild(fragment);
        return;
    }
    const range = selection.getRangeAt(0);
    range.deleteContents();
    range.insertNode(fragment);
    range.setStartAfter(fragment);
    range.collapse(true);
    selection.removeAllRanges();
    selection.addRange(range);
}

// --------------------------------------------------------------- widgets ----

function widgetOf(node, name) {
    return node?.widgets?.find((w) => w.name === name);
}

// Collapsing the layout row is not enough for a multiline STRING: it is a DOM
// widget whose textarea is positioned independently of the row, so hiding by
// type alone leaves a thin visible sliver. Hiding never adds or removes a
// widget, so the positional widgets_values mapping stays intact.
function hide(w) {
    if (!w || w.__myh3Hidden) return;
    w.__myh3Hidden = true;
    w.__myh3 = {
        type: w.type,
        hidden: w.hidden,
        computeSize: w.computeSize,
        hadComputeSize: Object.prototype.hasOwnProperty.call(w, "computeSize"),
        computedHeight: w.computedHeight,
        hadComputedHeight: Object.prototype.hasOwnProperty.call(w, "computedHeight"),
    };
    w.hidden = true;
    w.type = "hidden";
    w.computeSize = () => [0, -4];
    w.computedHeight = 0;
    if (w.inputEl) w.inputEl.style.display = "none";
    if (w.element) w.element.style.display = "none";
    if (w.options) { w.options.hidden = true; w.options.canvasOnly = true; }
    if (w._state) { w._state.hidden = true; w._state.type = "hidden"; w._state.computedHeight = 0; }
}

function show(w) {
    if (!w || !w.__myh3Hidden) return;
    const saved = w.__myh3 || {};
    w.__myh3Hidden = false;
    w.hidden = saved.hidden ?? false;
    w.type = saved.type || "combo";
    if (saved.hadComputeSize) w.computeSize = saved.computeSize; else delete w.computeSize;
    if (saved.hadComputedHeight) w.computedHeight = saved.computedHeight; else delete w.computedHeight;
    if (w.inputEl) w.inputEl.style.display = "";
    if (w.element) w.element.style.display = "";
    if (w.options) { w.options.hidden = false; w.options.canvasOnly = false; }
    if (w._state) {
        w._state.hidden = w.hidden;
        w._state.type = w.type;
        if (saved.hadComputedHeight) w._state.computedHeight = w.computedHeight;
        else delete w._state.computedHeight;
    }
}

function setVisible(w, visible) { visible ? show(w) : hide(w); }

function refreshDetail(node) {
    const by = {};
    for (const w of node.widgets || []) {
        by[w.name] = w;
        if (DETAIL_LABELS[w.name]) w.label = DETAIL_LABELS[w.name];
    }
    const enabled = by.enabled?.value !== false;
    const mode = String(by.mode?.value || "放大 + 二采（推荐）");
    const upscale = enabled && !mode.includes("同分辨率");
    const sampling = enabled && !mode.includes("仅放大");
    const method = String(by.upscale_method?.value || "").toLowerCase();
    const neural = upscale && method.includes("neural_3d");
    const pixel = upscale && !neural && !method.includes("latent");
    const custom = upscale && String(by.resolution?.value || "") === "自定义";
    for (const [name, visible] of Object.entries({
        mode: enabled,
        resolution: upscale,
        width: custom,
        height: custom,
        steps: sampling,
        denoise: sampling,
        scheduler: sampling,
        sampler_name: sampling,
        upscale_method: upscale,
        chunk_frames: pixel,
        latent_upscale_model: neural,
        latent_precision: neural,
        latent_chunk_steps: neural,
        passes: sampling,
        seed_mode: sampling && Number(by.passes?.value || 1) > 1,
    })) setVisible(by[name], visible);
    if (node.__myh3DetailCustom === undefined) {
        node.__myh3DetailCustom = custom;
        if (!custom && Number(node.size?.[1] || 0) >= 380) {
            node.setSize?.([node.size[0], node.size[1] - 48]);
        }
    } else if (node.__myh3DetailCustom !== custom) {
        node.__myh3DetailCustom = custom;
        node.setSize?.([node.size[0], Math.max(240, node.size[1] + (custom ? 48 : -48))]);
    }
    for (const input of node.inputs || []) {
        if (DETAIL_LABELS[input.name]) input.label = DETAIL_LABELS[input.name];
        if (input.name === "二采模型") input.label = "二采模型（LoRA 前基模）";
    }
    node.graph?.setDirtyCanvas?.(true, true);
}

// ---------------------------------------------------------------- editor ----

function upstreamText(node) {
    const stored = node.properties?.[PREVIEW_PROP];
    if (stored) return stored;
    // Before the Agent has ever run, show what it is going to send.
    return String(widgetOf(upstream(node, "prompt"), "prompt")?.value || "");
}

function repaint(node) {
    const view = node.__myh3Editor;
    if (!view || !view.isConnected) return;
    if (document.activeElement === view) return;   // never yank the caret away
    const wrap = node.__myh3Wrap;
    const bar = node.__myh3Toolbar;
    const hasPlan = linkedInput(node, "plan_json") != null;
    let text = "";
    if (hasPlan) {
        text = node.__myh3CurrentSegmentPrompt || node.properties?.[PREVIEW_PROP] || "";
        renderInto(view, text, mediaList(node));
        view.contentEditable = "false";
        view.classList.add("is-preview");
        view.dataset.placeholder = "已连接分段计划 (plan_json)，生成时将在此实时预览当前分段提示词与素材";
        if (wrap) wrap.classList.add("is-linked-plan");
        if (node.__myh3Hint) {
            node.__myh3Hint.textContent = node.__myh3CurrentStatus || "已接入 plan_json · 分段计划驱动";
            node.__myh3Hint.classList.add("is-linked");
        }
        if (bar) {
            for (const btn of bar.querySelectorAll(".myh3-btn")) {
                btn.style.display = "none";
            }
        }
    } else {
        text = String(promptText(node) || "");
        renderInto(view, text, mediaList(node));
        view.contentEditable = "true";
        view.classList.remove("is-preview");
        if (wrap) wrap.classList.remove("is-linked-plan");
        view.dataset.placeholder = "手写提示词模式：写一句提示词。@ 引用素材，选中文字点「台词」变成台词块。";
        if (node.__myh3Hint) {
            node.__myh3Hint.textContent = "手写模式（未接分段计划）";
            node.__myh3Hint.classList.remove("is-linked");
        }
        if (bar) {
            for (const btn of bar.querySelectorAll(".myh3-btn")) {
                btn.style.display = "";
            }
        }
    }
    node.__myh3Sig = mediaSignature(node);
}

function promptText(node) {
    return String(node?.properties?.[TEXT_PROP] || "");
}

/** Store the prompt on the node, not in a widget.
 *
 * `properties` is serialized with the workflow just like widget values are, but
 * it carries no index, so nothing here can shift the positional
 * widgets_values mapping the way an added or removed widget does.
 */
function syncToWidget(node) {
    const view = node.__myh3Editor;
    if (!view) return;
    const text = readText(view);
    if (promptText(node) === text) return;
    node.properties ||= {};
    node.properties[TEXT_PROP] = text;
    node.graph?.setDirtyCanvas?.(true, true);
}

function refresh(node) {
    const by = {};
    for (const w of node.widgets || []) by[w.name] = w;

    const custom = String(by.resolution?.value || "") === "custom";
    setVisible(by.width, custom);
    setVisible(by.height, custom);

    // Prompt mode / media prefix are legacy fallbacks. legacy_plan_padding is
    // only a positional migration pad for old widgets_values arrays.
    // prompt slicing is managed by H3ScriptSplitter (plan_json) or the bottom editor.
    // Always hide them to keep the node card clean and free of noise.
    setVisible(by.prompt_mode, false);
    setVisible(by.media_prefix, false);
    setVisible(by.legacy_plan_padding, false);
    setVisible(by.segment_prefix, by.save_segments?.value !== false);

    // This frontend materialises every widget as an input slot as well, so a
    // widget-backed slot needs the same Chinese label the widget row carries --
    // otherwise the socket rows read as raw backend names.
    for (const input of node.inputs || []) {
        const label = SLOT_LABELS[input.name]
            || (input.widget ? LABELS[input.name] : null);
        if (label) input.label = label;
    }

    repaint(node);

    if (typeof node.computeSize === "function") {
        const minSize = node.computeSize();
        if (minSize) {
            node.size[0] = Math.max(520, node.size?.[0] || 520);
            if (!node.size[1] || node.size[1] < minSize[1]) {
                node.size[1] = minSize[1];
            }
            if (typeof node.setSize === "function") node.setSize(node.size);
        }
    }
    node.graph?.setDirtyCanvas?.(true, true);
}

function buildEditor(node) {
    const wrap = document.createElement("div");
    wrap.className = "myh3-editor-wrap";
    node.__myh3Wrap = wrap;

    const bar = document.createElement("div");
    bar.className = "myh3-toolbar";
    wrap.appendChild(bar);
    node.__myh3Toolbar = bar;

    const view = document.createElement("div");
    view.className = "myh3-editor";
    view.contentEditable = "true";
    view.spellcheck = false;
    wrap.appendChild(view);
    node.__myh3Editor = view;

    const insertMention = (entry) => {
        insertAtCaret(view, makeChip(entry.kind, entry.ordinal, entry));
        syncToWidget(node);
    };

    for (const kind of KINDS) {
        const btn = document.createElement("div");
        btn.className = "myh3-btn";
        btn.textContent = `@${kind}`;
        btn.title = `插入一个${kind}引用`;
        btn.onmousedown = guard((event) => {
            event.preventDefault();
            const list = mediaList(node).filter((m) => m.kind === kind);
            if (list.length === 1) insertMention(list[0]);
            else showMenu(view, list, insertMention);
        });
        bar.appendChild(btn);
    }

    const lineBtn = document.createElement("div");
    lineBtn.className = "myh3-btn";
    lineBtn.textContent = "台词";
    lineBtn.title = "把选中的文字变成台词块（<d>…</d>）";
    lineBtn.onmousedown = guard((event) => {
        event.preventDefault();
        const selection = window.getSelection();
        if (!selection?.rangeCount || !view.contains(selection.anchorNode)) return;
        const text = String(selection.toString());
        if (text) selection.getRangeAt(0).deleteContents();
        insertAtCaret(view, makeDialogue(text || "在这里写台词"));
        syncToWidget(node);
    });
    bar.appendChild(lineBtn);

    const hint = document.createElement("span");
    hint.className = "myh3-hint";
    node.__myh3Hint = hint;
    bar.appendChild(hint);

    view.addEventListener("input", guard(() => syncToWidget(node)));
    view.addEventListener("blur", guard(() => { syncToWidget(node); closeMenu(); repaint(node); }));
    view.addEventListener("keydown", guard((event) => {
        if (event.key === "Escape") { closeMenu(); return; }
        // Chips and dialogue blocks are atomic: contenteditable would happily
        // let a backspace eat half of one and leave unparseable text behind.
        if (event.key === "Backspace" || event.key === "Delete") {
            const anchor = window.getSelection()?.anchorNode;
            const atomic = (anchor?.nodeType === Node.ELEMENT_NODE ? anchor : anchor?.parentElement)
                ?.closest?.(".myh3-chip, .myh3-line");
            if (atomic && view.contains(atomic)) {
                event.preventDefault();
                atomic.remove();
                syncToWidget(node);
            }
        }
    }));
    view.addEventListener("keyup", guard((event) => {
        if (event.key !== "@") return;
        showMenu(view, mediaList(node), (entry) => {
            // Drop the "@" just typed, then insert the chip in its place.
            const selection = window.getSelection();
            const range = selection?.rangeCount ? selection.getRangeAt(0) : null;
            if (range?.startContainer?.nodeType === Node.TEXT_NODE && range.startOffset > 0
                && range.startContainer.nodeValue.slice(0, range.startOffset).endsWith("@")) {
                range.setStart(range.startContainer, range.startOffset - 1);
                range.deleteContents();
            }
            insertMention(entry);
        });
    }));
    view.addEventListener("paste", guard((event) => {
        // Paste as plain text; pasted markup would smuggle in nodes readText
        // cannot round-trip.
        event.preventDefault();
        insertAtCaret(view, document.createTextNode(event.clipboardData?.getData("text/plain") || ""));
        syncToWidget(node);
    }));

    // 执行进度面板：进度条 + 阶段文本 + 当前段预览帧。默认隐藏，收到
    // myh3_longvideo_start 时显示。它是编辑器 wrap 内部的 DOM 子元素，不新增
    // widget，不影响 widgets_values 的位置映射。
    const progress = document.createElement("div");
    progress.className = "myh3-progress";
    progress.hidden = true;
    const pBar = document.createElement("div");
    pBar.className = "myh3-progress-bar";
    const pFill = document.createElement("div");
    pFill.className = "myh3-progress-fill";
    pBar.appendChild(pFill);
    const pText = document.createElement("div");
    pText.className = "myh3-progress-text";
    const pPreview = document.createElement("img");
    pPreview.className = "myh3-progress-preview";
    pPreview.hidden = true;
    pPreview.alt = "当前段预览帧";
    progress.append(pBar, pText, pPreview);
    wrap.appendChild(progress);
    node.__myh3Progress = progress;
    node.__myh3ProgressFill = pFill;
    node.__myh3ProgressText = pText;
    node.__myh3ProgressPreview = pPreview;

    return wrap;
}

// The Agent's media can be rewired long after this node was built. Polling one
// short string beats patching app.graph.onAfterChange, which is shared with
// every other pack -- a throw in there takes unrelated panels down with it.
let watcher = null;
function startWatcher() {
    if (watcher) return;
    watcher = setInterval(guard(() => {
        for (const node of app.graph?._nodes || []) {
            if (node.type !== NODE || !node.__myh3Editor) continue;
            if (mediaSignature(node) !== node.__myh3Sig) repaint(node);
        }
    }), 1000);
}

// `prompt` is a socket with no widget behind it, so nothing submits the editor's
// text automatically. Fill it in here, and only when the socket is empty -- a
// connected Agent has already been serialized as a [nodeId, slot] reference and
// writing over it would silently discard the Agent's prompt.
let patched = false;
function patchGraphToPrompt() {
    if (patched || typeof app.graphToPrompt !== "function") return;
    patched = true;
    const original = app.graphToPrompt;
    app.graphToPrompt = async function graphToPromptWithMyangPrompt(...args) {
        const data = await original.apply(this, args);
        try {
            for (const node of app.graph?._nodes || []) {
                if (node.type !== NODE) continue;
                if (linkedInput(node, "plan_json")) continue;
                const text = promptText(node);
                if (!text) continue;
                const entry = data?.output?.[String(node.id)];
                if (!entry) continue;
                entry.inputs ||= {};
                if (!entry.inputs.plan_json) {
                    entry.inputs.plan_json = JSON.stringify({
                        segment_count: 1,
                        frames_per_segment: 125,
                        segment_seconds_snapped: 5.0,
                        overlap_frames: 22,
                        fps: 24.0,
                        style_header: "",
                        segments: [{ index: 1, brief: text.slice(0, 50), prompt: text }],
                    });
                }
            }
        } catch (err) {
            if (!warned) {
                warned = true;
                console.error("[沐阳 H3] 提交手写提示词时出错：", err);
            }
        }
        return data;
    };
}

app.registerExtension({
    name: "myang.h3.longvideo",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== NODE) return;
        styleOnce();

        nodeType.prototype.computeSize = function (out) {
            const inputY = this.inputs ? this.inputs.length * 22 + 40 : 40;
            let widgetY = 40;
            const hasPlan = linkedInput(this, "plan_json") != null;
            const promptH = hasPlan ? 120 : 180;
            for (const w of this.widgets || []) {
                if (w.hidden) continue;
                if (w.name === "myh3_prompt") {
                    widgetY += promptH + 10;
                } else {
                    const widgetH = (window.LiteGraph?.NODE_WIDGET_HEIGHT || 24) + 4;
                    widgetY += widgetH;
                }
            }
            const totalH = Math.max(inputY, widgetY) + 12;
            return [520, totalH];
        };

        const onCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onCreated?.apply(this, arguments);
            guard(() => {
                const node = this;
                // Appended last and non-serializing: widgets before it keep
                // their indices, so widgets_values is unaffected.
                const widget = node.addDOMWidget("myh3_prompt", "myh3_prompt", buildEditor(node), {
                    serialize: false,
                    hideOnZoom: false,
                    // A floor, not a fixed size: the element is styled to fill
                    // whatever row it is given, so dragging the node taller
                    // grows the editor instead of leaving a blank strip.
                    getMinHeight: () => (linkedInput(node, "plan_json") ? 120 : 180),
                    getMaxHeight: () => 4096,
                });
                widget.serialize = false;
                for (const w of node.widgets || []) {
                    if (LABELS[w.name] && w !== widget) w.label = LABELS[w.name];
                    const prev = w.callback;
                    w.callback = function () {
                        const out = prev?.apply(this, arguments);
                        guard(refresh)(node);
                        return out;
                    };
                }
                requestAnimationFrame(guard(() => refresh(node)));
            }).call(this);
            return r;
        };

        const origDrawForeground = nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground = function (ctx, canvas) {
            const res = origDrawForeground ? origDrawForeground.apply(this, arguments) : undefined;
            guard(() => {
                const minSize = this.computeSize ? this.computeSize() : null;
                if (minSize && this.size[1] < minSize[1]) {
                    this.size[1] = minSize[1];
                    this.size[0] = Math.max(520, this.size[0] || 520);
                    if (typeof this.setSize === "function") this.setSize(this.size);
                }
            }).call(this);
            return res;
        };

        const origResize = nodeType.prototype.onResize;
        nodeType.prototype.onResize = function (size) {
            const minSize = this.computeSize ? this.computeSize() : null;
            if (minSize) {
                if (size[0] < minSize[0]) size[0] = minSize[0];
                if (size[1] < minSize[1]) size[1] = minSize[1];
            }
            return origResize ? origResize.apply(this, arguments) : undefined;
        };

        // 屏蔽 ComfyUI 自带的节点预览。子图里 SaveVideo/VAEDecode 等节点执行后，
        // executed 事件会通过 display_node 映射到本节点，把低分辨率视频/图像回显
        // 在节点底部（就是那个糊糊的、实时跳出来的预览）。进度面板已经有自己的
        // 高清新预览，这里把 output 里的 images/gifs/videos 清掉，避免出现两个预览。
        const origOnExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (info) {
            if (info && info.output
                && (info.output.images || info.output.gifs || info.output.videos)) {
                info = Object.assign({}, info, {
                    output: Object.assign({}, info.output, {
                        images: undefined, gifs: undefined, videos: undefined,
                    }),
                });
            }
            return origOnExecuted ? origOnExecuted.call(this, info) : undefined;
        };

        for (const hook of ["onConnectionsChange", "onConfigure", "onAdded", "onSelected", "onDeselected"]) {
            const original = nodeType.prototype[hook];
            nodeType.prototype[hook] = function () {
                const r = original?.apply(this, arguments);
                const node = this;
                requestAnimationFrame(guard(() => refresh(node)));
                return r;
            };
        }
    },

    setup() {
        guard(() => {
            startWatcher();
            patchGraphToPrompt();

            // 长视频任务开始：记录这次 run 的元信息（段数/是否二采/是否漂移），
            // 初始化所有 H3LongVideo 节点的进度面板。
            // 构图完即开始第 1 段采样，所以这里直接显示「第 1 段采样中」——
            // 第一个 sampled signal 要等采样完成才发，期间不能让面板停在「准备开始」。
            api.addEventListener("myh3_longvideo_start", guard((event) => {
                const d = event?.detail || {};
                // 带 owner_id 的任务来自导演台；它有自己的实例级进度面板，
                // 不能把状态广播到画布上的独立 H3LongVideo 节点。
                if (String(d.owner_id || "")) return;
                activeRun = {
                    run_id: d.run_id,
                    total: d.total_segments || 1,
                    refining: !!d.refining,
                    correcting: !!d.correcting,
                    seg: 1,
                    phase: "sample1",
                    step: 0,
                    stepMax: 0,
                    previewFile: null,
                    previewTs: 0,
                    progressSeen: false,
                };
                for (const node of app.graph?._nodes || []) {
                    if (node.type !== NODE) continue;
                    if (node.__myh3ProgressPreview) node.__myh3ProgressPreview.hidden = true;
                    renderProgress(node);
                }
            }));

            // 实时进度：由子图里的 H3ProgressSignal 节点在真正执行到该处时发出。
            // 它带当前段号、阶段、提示词和一帧预览图，是"此刻渲染到哪"的真实反映。
            api.addEventListener("myh3_progress", guard((event) => {
                const d = event?.detail;
                if (!d) return;
                if (String(d.owner_id || "")) return;
                // 首个 progress 到达 = signal 通路打通。打一条日志方便排查
                // （若一直见不到这条，说明 signal 节点没执行，多半是后端没重启加载新节点）。
                if (activeRun && !activeRun.progressSeen) {
                    activeRun.progressSeen = true;
                    console.info("[沐阳 H3] 阶段进度通路已打通：第", d.segment_index, "/", d.total_segments, "段", d.stage);
                }
                for (const node of app.graph?._nodes || []) {
                    if (node.type !== NODE) continue;
                    applyProgress(node, d);
                }
            }));

            // 步级进度：ComfyUI 采样器每步发的 progress 事件（value/max）。
            // 结合当前阶段（一采/二采）显示「第 X/Y 步」，进度条也跟着步数走。
            // 这个事件不依赖 H3ProgressSignal，是 ComfyUI 核心的 ProgressBar 发的。
            api.addEventListener("progress", guard((event) => {
                if (!activeRun) return;
                const d = event?.detail;
                if (!d || !d.max) return;
                activeRun.step = Number(d.value) || 0;
                activeRun.stepMax = Number(d.max) || 0;
                for (const node of app.graph?._nodes || []) {
                    if (node.type !== NODE) continue;
                    renderProgress(node);
                }
            }));

            // Plan ready event from H3ScriptSplitter
            api.addEventListener("myh3_plan_ready", guard((event) => {
                const detail = event?.detail;
                if (!detail) return;
                const { total_segments, first_prompt, first_brief } = detail;
                const text = String(first_prompt || "").trim();
                if (!text) return;
                for (const node of app.graph?._nodes || []) {
                    if (node.type !== NODE) continue;
                    node.properties ||= {};
                    node.properties[PREVIEW_PROP] = text;
                    if (!node.__myh3CurrentSegmentPrompt && node.__myh3Editor) {
                        renderInto(node.__myh3Editor, text, mediaList(node));
                    }
                    if (node.__myh3Hint && !node.__myh3CurrentSegmentPrompt) {
                        node.__myh3Hint.textContent = `分段计划已就绪（共 ${total_segments} 段）`;
                    }
                }
            }));

            // Execution finished event
            api.addEventListener("execution_success", guard(() => {
                if (activeRun) {
                    activeRun.phase = "done";
                    activeRun.step = activeRun.stepMax || 0;
                }
                for (const node of app.graph?._nodes || []) {
                    if (node.type !== NODE) continue;
                    renderProgress(node);
                }
                activeRun = null;
            }));

            // 失败或中断：标红进度文本，避免面板停留在"进行中"误导用户。
            const onRunFail = guard((msg) => {
                activeRun = null;
                for (const node of app.graph?._nodes || []) {
                    if (node.type !== NODE) continue;
                    if (node.__myh3ProgressText) {
                        node.__myh3ProgressText.classList.remove("is-running", "is-done");
                        node.__myh3ProgressText.classList.add("is-error");
                        node.__myh3ProgressText.textContent = "✘ " + (msg || "执行中断");
                    }
                    if (node.__myh3Hint) {
                        node.__myh3Hint.textContent = "✘ 执行中断";
                    }
                }
            });
            api.addEventListener("execution_error", guard((event) =>
                onRunFail(event?.detail?.exception_message || event?.detail?.error || "执行出错")));
            api.addEventListener("execution_interrupted", guard(() => onRunFail("已中断")));

            // The Agent publishes its finished prompt when it runs; mirror it
            // into whichever of our nodes it feeds so the preview is the real
            // text rather than a guess.
            api.addEventListener("executed", guard((event) => {
                const detail = event?.detail;
                const payload = detail?.output?.myang_prompt;
                if (!payload) return;
                const text = String(Array.isArray(payload) ? payload[0] : payload || "");
                for (const node of app.graph?._nodes || []) {
                    if (node.type !== NODE || !node.graph) continue;
                    const input = linkedInput(node, "prompt");
                    const link = input ? node.graph.links?.[input.link] : null;
                    if (!link || Number(link.origin_id) !== Number(detail.node)) continue;
                    node.properties ||= {};
                    node.properties[PREVIEW_PROP] = text;
                    repaint(node);
                }
            }));
            window.addEventListener("mousedown", guard((event) => {
                if (openMenu && !openMenu.contains(event.target)) closeMenu();
            }));
        })();
    },
});

app.registerExtension({
    name: "myang.h3.detail.settings",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== DETAIL_NODE) return;

        const onCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onCreated?.apply(this, arguments);
            const node = this;
            for (const widget of node.widgets || []) {
                const previous = widget.callback;
                widget.callback = function () {
                    const output = previous?.apply(this, arguments);
                    requestAnimationFrame(guard(() => refreshDetail(node)));
                    return output;
                };
            }
            requestAnimationFrame(guard(() => refreshDetail(node)));
            return result;
        };

        for (const hook of ["onConfigure", "onAdded"]) {
            const original = nodeType.prototype[hook];
            nodeType.prototype[hook] = function () {
                const result = original?.apply(this, arguments);
                const node = this;
                requestAnimationFrame(guard(() => refreshDetail(node)));
                return result;
            };
        }
    },
});
