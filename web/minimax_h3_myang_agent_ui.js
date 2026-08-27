import { app } from "../../scripts/app.js";

/*
 * Media Agent UI for ComfyUI-MiniMaxH3-Myang.
 * SPDX-License-Identifier: GPL-3.0-only
 */

function installSummaryViewerNode(nodeType, nodeData, comfyApp) {
    if (nodeData?.name !== "MiniMaxH3Viewer" && nodeData?.name !== "MiniMaxH3SummaryViewer") return;
    if (nodeType.prototype.__h3ViewerInstalled) return;
    nodeType.prototype.__h3ViewerInstalled = true;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
        onNodeCreated?.apply(this, arguments);
        // Default size for a freshly dropped node only. onNodeCreated also runs
        // when a saved workflow is loaded, before onConfigure restores the
        // stored size, so stamping this unconditionally is fine -- but flag it
        // so nothing downstream mistakes it for a size the user chose.
        this.size = [560, 440];
        this.__h3DefaultSized = true;

        requestAnimationFrame(() => {
            const w = this.widgets?.find((x) => x.name === "text");
            if (w && w.inputEl) {
                w.inputEl.readOnly = true;
                w.inputEl.style.fontSize = "12px";
                w.inputEl.style.fontFamily = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Consolas, monospace";
                w.inputEl.style.lineHeight = "1.5";
                w.inputEl.style.color = "#a9dc76";
                w.inputEl.style.background = "#14141c";
                w.inputEl.style.border = "1px solid #282836";
                w.inputEl.style.borderRadius = "6px";
                w.inputEl.style.padding = "10px";
            }
        });
    };

    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
        onExecuted?.apply(this, arguments);
        const text = message?.text?.[0] || message?.text || "";
        if (text && this.widgets) {
            const w = this.widgets.find((x) => x.name === "text");
            if (w) {
                w.value = text;
                if (w.inputEl) {
                    w.inputEl.value = text;
                }
            }
        }
    };
}

const AGENT_CLASS = "MiniMaxH3MediaAgent";
const AGENT_MEDIA_PROP = "myang_h3_asset_sources_v2";
const MAX_MEDIA = 15;
const MAX_SKILL_BYTES = 512 * 1024;
const MEDIA_KINDS = new Set(["IMAGE", "VIDEO", "AUDIO", "*"]);
const MEDIA_TAG_RE = /<(Picture|Video|Audio)\s+(\d+)>/gi;
const SETTINGS_NAMESPACE = "Myang_node.MiniMaxH3";
const LEGACY_SETTINGS_NAMESPACE = "ComfyUI.MiniMaxH3";

function settingId(name) {
    return `${SETTINGS_NAMESPACE}.${name}`;
}

function migratedSettingDefault(name, fallback, parse = (value) => value) {
    try {
        const currentId = settingId(name);
        const current = localStorage.getItem(currentId);
        if (current !== null) return parse(current);
        const legacy = localStorage.getItem(`${LEGACY_SETTINGS_NAMESPACE}.${name}`);
        if (legacy !== null) {
            localStorage.setItem(currentId, legacy);
            return parse(legacy);
        }
    } catch (error) {
        console.warn("[Myang H3] 设置迁移失败", error);
    }
    return fallback;
}

function isAgent(node) {
    return String(node?.comfyClass || node?.type || node?.constructor?.nodeData?.name || "") === AGENT_CLASS;
}

function getWidget(node, name) {
    return node?.widgets?.find((widget) => widget?.name === name) || null;
}

function slotIndex(node, inputName) {
    return node?.inputs?.findIndex((input) => String(input?.name || "") === inputName) ?? -1;
}

function slotType(output) {
    return String(output?.type || output?.datatype || output?.label || "*").toUpperCase();
}

const VIDEO_FILE_RE = /\.(mp4|webm|mov|mkv|avi|m4v|flv|wmv|mpg|mpeg)$/i;
const AUDIO_FILE_RE = /\.(mp3|wav|flac|ogg|m4a|aac|opus|wma)$/i;

function anyWidgetFilename(source) {
    for (const widget of source?.widgets || []) {
        const raw = typeof widget?.value === "object" ? widget.value?.filename || widget.value?.name : widget?.value;
        const filename = String(raw || "").trim();
        if (!filename || /^(data|blob|https?):/i.test(filename)) continue;
        if (/\.[a-z0-9]{2,5}$/i.test(filename)) return filename.split(/[\\/]/).pop();
    }
    return "";
}

function nodeMediaHint(source) {
    const identity = `${source?.comfyClass || source?.type || ""} ${source?.title || ""}`.toLowerCase();
    if (/audio|music|voice|speech|tts|vocal/.test(identity)) return "audio";
    if (/video|movie|clip|footage/.test(identity)) return "video";
    return "";
}

function slotMediaType(source, sourceSlot, sourceType) {
    // The media type is always derived, never typed in. A declared AUDIO/VIDEO
    // slot is authoritative; an IMAGE slot is not, because video loaders (VHS
    // and friends) hand back frame batches on an IMAGE output. For those, fall
    // back to the connected file's extension and then the node's identity.
    const declared = String(sourceType || slotType(source?.outputs?.[sourceSlot]) || "*").toUpperCase();
    if (declared.includes("AUDIO")) return "audio";
    if (declared.includes("VIDEO")) return "video";
    const filename = anyWidgetFilename(source);
    if (AUDIO_FILE_RE.test(filename)) return "audio";
    if (VIDEO_FILE_RE.test(filename)) return "video";
    return nodeMediaHint(source) || "image";
}

// LiteGraph node modes: 0 ALWAYS, 1 ON_EVENT, 2 NEVER (muted), 3 ON_TRIGGER,
// 4 BYPASS. Muting or bypassing a loader is how you say "not this one this
// time", and the backend duly sends nothing for it -- so collecting it anyway
// only produced a phantom <Picture N> in the whitelist that the prompt could
// then reference into thin air.
function isNodeDisabled(node) {
    const mode = Number(node?.mode ?? 0);
    return mode === 2 || mode === 4;
}

function isMediaSourceUsable(source) {
    if (!source || isNodeDisabled(source)) return false;
    // A muted node upstream of an enabled passthrough leaves the passthrough
    // with nothing to forward, so walk the chain before trusting it.
    let current = source;
    for (let depth = 0; depth < 8; depth += 1) {
        const inputs = current?.inputs || [];
        const feeding = inputs.filter((input) => input?.link != null);
        if (!feeding.length) return true;
        const link = app.graph?.links?.[feeding[0].link];
        const upstream = link ? app.graph?.getNodeById?.(Number(link.origin_id)) : null;
        if (!upstream) return true;
        if (isNodeDisabled(upstream)) return false;
        current = upstream;
    }
    return true;
}

function mediaConnections(node) {
    node.properties ||= {};
    if (!Array.isArray(node.properties[AGENT_MEDIA_PROP])) node.properties[AGENT_MEDIA_PROP] = [];
    return node.properties[AGENT_MEDIA_PROP];
}

function setMediaConnections(node, connections) {
    node.properties ||= {};
    node.properties[AGENT_MEDIA_PROP] = connections;
}

function sameConnection(left, right) {
    return Number(left?.source_id) === Number(right?.source_id)
        && Number(left?.source_slot || 0) === Number(right?.source_slot || 0);
}

function sourceLabel(source) {
    return String(source?.title || source?.type || "Media");
}

function sourceFilename(source, mediaType) {
    const preferred = new Set({
        image: ["image", "filename", "file"],
        video: ["video", "file", "filename", "video_file", "videofile"],
        audio: ["audio", "file", "filename", "audio_file", "audiofile"],
    }[mediaType] || ["file", "filename"]);
    for (const widget of source?.widgets || []) {
        const raw = typeof widget?.value === "object" ? widget.value?.filename || widget.value?.name : widget?.value;
        const filename = String(raw || "").trim();
        if (!filename || /^(data|blob|https?):/i.test(filename)) continue;
        const widgetName = String(widget?.name || "").toLowerCase();
        if (preferred.has(widgetName) || /\.(png|jpe?g|webp|gif|bmp|mp4|webm|mov|mkv|avi|m4v|mp3|wav|flac|ogg|m4a)$/i.test(filename)) {
            return filename.split(/[\\/]/).pop();
        }
    }
    return "";
}

const SUBJECT_PROP = "minimax_h3_agent_subjects";

function subjectMap(node) {
    node.properties ||= {};
    if (!node.properties[SUBJECT_PROP] || typeof node.properties[SUBJECT_PROP] !== "object") {
        node.properties[SUBJECT_PROP] = {};
    }
    return node.properties[SUBJECT_PROP];
}

function subjectKey(connection) {
    return `${Number(connection?.source_id)}:${Number(connection?.source_slot || 0)}`;
}

function subjectFor(node, connection) {
    // Keyed by source node rather than by tag index, so renaming survives
    // reordering: <Picture 2> becoming <Picture 1> must not hand its name to a
    // different image.
    return String(subjectMap(node)[subjectKey(connection)] || "");
}

function setSubjectFor(node, connection, name) {
    const map = subjectMap(node);
    const clean = String(name || "").trim().slice(0, 40);
    if (clean) map[subjectKey(connection)] = clean;
    else delete map[subjectKey(connection)];
}

function connectionMetadata(source, sourceSlot, sourceType) {
    const mediaType = slotMediaType(source, sourceSlot, sourceType);
    return {
        source_id: Number(source.id),
        source_slot: Number(sourceSlot || 0),
        source_type: String(sourceType || "*"),
        media_type: mediaType,
        label: sourceLabel(source),
        filename: sourceFilename(source, mediaType),
    };
}

function normalizeAgentConnections(node, removeMissing = true) {
    const result = [];
    const seen = new Set();
    for (const connection of mediaConnections(node)) {
        const sourceId = Number(connection?.source_id);
        const sourceSlot = Number(connection?.source_slot || 0);
        if (!Number.isFinite(sourceId) || sourceId < 0) continue;
        const source = app.graph?.getNodeById?.(sourceId);
        if (!source) {
            if (!removeMissing) result.push({ ...connection, source_id: sourceId, source_slot: sourceSlot });
            continue;
        }
        if (!isMediaSourceUsable(source)) continue;
        const output = source.outputs?.[sourceSlot];
        const sourceType = String(slotType(output) || connection?.source_type || "*");
        const key = `${sourceId}:${sourceSlot}`;
        if (seen.has(key)) continue;
        seen.add(key);
        // Always re-derive the media type from the live graph so a stale value
        // stored in a saved workflow can never override what is connected now.
        const metadata = connectionMetadata(source, sourceSlot, sourceType);
        result.push({
            ...metadata,
            subject: subjectFor(node, metadata),
            order: result.length + 1,
        });
    }
    setMediaConnections(node, result.slice(0, MAX_MEDIA));
    return mediaConnections(node);
}

function syncAgentConnectionsFromNativeLinks(node) {
    if (!node?.inputs) return normalizeAgentConnections(node);
    const next = [];
    const seenKeys = new Set();

    for (let index = 0; index < node.inputs.length; index += 1) {
        const input = node.inputs[index];
        const inputName = String(input?.name || "");
        if (!/^asset_\d+$/.test(inputName)) continue;

        const linkIds = [];
        if (Array.isArray(input.links) && input.links.length > 0) {
            linkIds.push(...input.links);
        } else if (input.link != null) {
            linkIds.push(input.link);
        }

        for (const linkId of linkIds) {
            const link = app.graph?.links?.[linkId];
            if (!link) continue;
            const source = app.graph?.getNodeById?.(Number(link.origin_id));
            if (!source) continue;
            if (!isMediaSourceUsable(source)) continue;
            const sourceSlot = Number(link.origin_slot || 0);
            const sourceType = slotType(source.outputs?.[sourceSlot]);
            const key = `${source.id}:${sourceSlot}`;
            if (seenKeys.has(key)) continue;
            seenKeys.add(key);

            const metadata = connectionMetadata(source, sourceSlot, sourceType);
            next.push({
                ...metadata,
                subject: subjectFor(node, metadata),
                order: next.length + 1,
            });
        }
    }

    if (next.length > 0) {
        setMediaConnections(node, next.slice(0, MAX_MEDIA));
        return mediaConnections(node);
    }
    return normalizeAgentConnections(node);
}

const MEDIA_TYPE_LABELS = { image: "图片", video: "视频", audio: "音频" };
const MEDIA_TAG_NAMES = { image: "Picture", video: "Video", audio: "Audio" };
const EMPTY_BADGE_TEXT = "未连接媒体素材";
const BADGE_WIDGET_NAME = "__h3_agent_media_badge";

function describeConnections(connections) {
    // Mirror the backend tag order (images, then videos, then audio) so the
    // badge shows the exact tag the prompt has to use.
    const counts = { image: 0, video: 0, audio: 0 };
    const lines = [];
    for (const type of ["image", "video", "audio"]) {
        for (const connection of connections) {
            if (String(connection?.media_type || "image") !== type) continue;
            counts[type] += 1;
            const title = String(connection?.filename || connection?.label || "Media");
            const subject = String(connection?.subject || "");
            const suffix = subject ? `  🎭${subject}` : "";
            lines.push(`<${MEDIA_TAG_NAMES[type]} ${counts[type]}> ${MEDIA_TYPE_LABELS[type]} · ${title}${suffix}`);
        }
    }
    return lines;
}

const AUDIO_ONLY_WIDGETS = ["音频识别", "音频主体"];

function hideAudioWidgetsWhenNoAudio(node) {
    // These two only mean anything once an audio clip is actually connected,
    // so they stay out of the way until then.
    const hasAudio = syncAgentConnectionsFromNativeLinks(node)
        .some((connection) => String(connection?.media_type || "") === "audio");
    let changed = false;
    for (const name of AUDIO_ONLY_WIDGETS) {
        const widget = getWidget(node, name);
        if (!widget) continue;
        const shouldHide = !hasAudio;
        if (Boolean(widget.hidden) === shouldHide) continue;
        widget.hidden = shouldHide;
        widget.computeSize = shouldHide ? () => [0, -4] : undefined;
        changed = true;
    }
    if (changed) {
        // Only nudge the height by the rows that actually appeared or
        // disappeared, and never during a restore. Resizing to computeSize()
        // here replaced whatever width and height the user had set, so a saved
        // workflow came back a different shape every time it was reopened.
        if (!node.__h3RestoringLayout && Array.isArray(node.size)) {
            const rowHeight = Number(globalThis.LiteGraph?.NODE_WIDGET_HEIGHT) || 20;
            const delta = (hasAudio ? 1 : -1) * rowHeight * AUDIO_ONLY_WIDGETS.length;
            node.size[1] = Math.max(80, Number(node.size[1] || 0) + delta);
        }
        node.setDirtyCanvas?.(true, true);
    }
    return hasAudio;
}

function refreshMediaBadge(node) {
    const widget = getWidget(node, BADGE_WIDGET_NAME);
    if (!widget) return;
    const connections = syncAgentConnectionsFromNativeLinks(node);
    const lines = describeConnections(connections);
    const next = lines.length ? lines.join("\n") : EMPTY_BADGE_TEXT;
    if (widget.value === next) return;
    widget.value = next;
    node.setDirtyCanvas?.(true, true);
}

function fitText(ctx, text, maxWidth) {
    const value = String(text || "");
    if (maxWidth <= 0 || ctx.measureText(value).width <= maxWidth) return value;
    let low = 0;
    let high = value.length;
    while (low < high) {
        const middle = Math.ceil((low + high) / 2);
        if (ctx.measureText(`${value.slice(0, middle)}…`).width <= maxWidth) low = middle;
        else high = middle - 1;
    }
    return `${value.slice(0, low)}…`;
}

function installMediaBadgeWidget(node) {
    if (getWidget(node, BADGE_WIDGET_NAME)) return;
    const widget = {
        type: "h3_media_badge",
        name: BADGE_WIDGET_NAME,
        value: EMPTY_BADGE_TEXT,
        // Keep this out of widgets_values: ComfyUI restores saved widget values
        // positionally, and a stray entry would shift every later widget.
        serialize: false,
        options: { serialize: false },
        serializeValue: () => undefined,
        computeSize(width) {
            const lines = String(this.value || EMPTY_BADGE_TEXT).split("\n").length;
            return [width, 10 + lines * 15];
        },
        draw(ctx, drawnNode, widgetWidth, widgetY) {
            const lines = String(this.value || EMPTY_BADGE_TEXT).split("\n");
            const connected = lines[0] !== EMPTY_BADGE_TEXT;
            const margin = 12;
            const width = Math.max(24, widgetWidth - margin * 2);
            const height = 10 + lines.length * 15 - 4;
            ctx.save();
            ctx.beginPath();
            ctx.fillStyle = connected ? "rgba(0,226,187,0.10)" : "rgba(255,97,136,0.10)";
            ctx.strokeStyle = connected ? "rgba(0,226,187,0.55)" : "rgba(255,97,136,0.55)";
            ctx.lineWidth = 1;
            if (typeof ctx.roundRect === "function") ctx.roundRect(margin, widgetY, width, height, 5);
            else ctx.rect(margin, widgetY, width, height);
            ctx.fill();
            ctx.stroke();
            ctx.fillStyle = connected ? "#8ff0dd" : "#ff9bb2";
            ctx.font = "11px Consolas, Menlo, monospace";
            ctx.textAlign = "left";
            ctx.textBaseline = "middle";
            lines.forEach((line, index) => {
                ctx.fillText(fitText(ctx, line, width - 14), margin + 7, widgetY + 12 + index * 15);
            });
            ctx.restore();
        },
    };
    if (typeof node.addCustomWidget === "function") node.addCustomWidget(widget);
    else (node.widgets ||= []).push(widget);
}

function patchGraphToPrompt() {
    if (app.__minimaxH3AgentPromptPatched || typeof app.graphToPrompt !== "function") return;
    app.__minimaxH3AgentPromptPatched = true;
    const original = app.graphToPrompt;
    app.graphToPrompt = async function graphToPromptWithMediaAgent() {
        // Refresh native media links before ComfyUI serializes the graph so
        // muted, bypassed or rewired sources cannot leave stale tags.
        for (const node of app.graph?._nodes || []) {
            if (!isAgent(node)) continue;
            syncAgentConnectionsFromNativeLinks(node);
            refreshMediaBadge(node);
        }

        const promptData = await original.apply(this, arguments);
        const output = promptData?.output || {};
        for (const node of app.graph?._nodes || []) {
            if (!isAgent(node)) continue;
            const promptNode = output[String(node.id)];
            if (!promptNode) continue;
            promptNode.inputs ||= {};
            delete promptNode.inputs.catalog;
            for (let index = 1; index <= MAX_MEDIA; index += 1) {
                delete promptNode.inputs[`asset_${index}`];
            }
            // Only keep connections whose source actually survived into the
            // prompt (muted/bypassed loaders are pruned by ComfyUI). Filtering
            // before numbering keeps asset_N and asset_manifest_json in step --
            // otherwise a dropped slot shifts the rest and every filename and
            // media type after it is attributed to the wrong tag.
            const connections = normalizeAgentConnections(node)
                .filter((connection) => Boolean(output[String(connection.source_id)]));
            connections.forEach((connection, index) => {
                promptNode.inputs[`asset_${index + 1}`] = [String(connection.source_id), Number(connection.source_slot || 0)];
            });
            // Single metadata channel for the backend: it carries the derived
            // media type plus the filename/label, which cannot be recovered
            // from a decoded tensor.
            promptNode.inputs.asset_manifest_json = JSON.stringify(
                connections.map((connection, index) => ({...connection, slot: index + 1})),
            );
        }
        return promptData;
    };
}

async function refreshSkillOptions(node, selectName = null) {
    const [skillsRes, learnRes] = await Promise.all([
        fetch("/minimax-h3-agent/skills"),
        fetch("/minimax-h3-agent/learn"),
    ]);
    if (!skillsRes.ok) throw new Error(`Skill list failed: ${skillsRes.status}`);
    const payload = await skillsRes.json();
    const learnData = await learnRes.json().catch(() => ({}));
    const allSkills = Array.isArray(payload.skills) ? payload.skills : ["none"];
    const learnedNames = Object.keys(learnData.skills || {});

    // 预设下拉只显示 none + auto + 已学习的技能
    // 未学习的技能在技能管理面板里可见，但不出现在预设下拉
    let filtered = ["none", "auto"];
    for (const s of allSkills) {
        if (s !== "none" && s !== "auto" && learnedNames.includes(s)) {
            filtered.push(s);
        }
    }

    const widget = getWidget(node, "skill_preset");
    if (!widget) return filtered;
    widget.options ||= {};
    // 保留当前选中值（即使未学习），避免工作流恢复时丢失
    const currentVal = selectName || widget.value;
    if (currentVal && !filtered.includes(currentVal) && allSkills.includes(currentVal)) {
        filtered.push(currentVal);
    }
    widget.options.values = filtered;
    if (selectName && filtered.includes(selectName)) widget.value = selectName;
    else if (!filtered.includes(widget.value)) widget.value = filtered[0] || "none";
    node.setDirtyCanvas?.(true, true);
    return filtered;
}

async function uploadSkillFile(node, file) {
    if (!file) return;
    if (file.size <= 0 || file.size > MAX_SKILL_BYTES) {
        throw new Error("Skill 文件必须在 1-512KB 之间");
    }
    const buffer = await file.arrayBuffer();
    let binary = "";
    const bytes = new Uint8Array(buffer);
    const chunkSize = 0x8000;
    for (let index = 0; index < bytes.length; index += chunkSize) {
        binary += String.fromCharCode(...bytes.subarray(index, index + chunkSize));
    }
    const response = await fetch("/minimax-h3-agent/skills", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ filename: file.name, content_base64: btoa(binary) }),
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok || !payload.success) throw new Error(payload.error || `上传失败: ${response.status}`);
    await refreshSkillOptions(node, payload.filename);
}

async function openSkillManagerModal(node = null) {
    const dialog = document.createElement("div");
    dialog.style.cssText = `
        position: fixed;
        top: 0; left: 0; width: 100vw; height: 100vh;
        background: rgba(0,0,0,0.75);
        z-index: 10000;
        display: flex;
        align-items: center;
        justify-content: center;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    `;

    const box = document.createElement("div");
    box.style.cssText = `
        width: 720px;
        max-width: 92vw;
        height: 560px;
        max-height: 88vh;
        background: #181822;
        border: 1px solid #333348;
        border-radius: 12px;
        padding: 20px;
        display: flex;
        flex-direction: column;
        gap: 12px;
        color: #e0e0e0;
        box-shadow: 0 12px 40px rgba(0,0,0,0.6);
    `;

    box.innerHTML = `
        <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid #333; padding-bottom:10px;">
            <div style="font-size:16px; font-weight:bold; color:#00e2bb; display:flex; align-items:center; gap:8px;">
                <span>⚙️ Agent 技能管理面板</span>
            </div>
            <button class="h3-close-btn" style="background:transparent; border:none; color:#aaa; font-size:20px; cursor:pointer;">✖</button>
        </div>

        <div style="display:flex; gap:10px; align-items:center; background:#222230; padding:10px; border-radius:6px;">
            <label style="font-size:13px; color:#78dce8; font-weight:bold; white-space:nowrap;">选择 Skill 预设：</label>
            <select class="h3-skill-select" style="flex:1; background:#121218; color:#fff; border:1px solid #444; border-radius:4px; padding:6px 10px; font-size:13px;"></select>
            <button class="h3-new-btn" style="background:#00e2bb; color:#111; font-weight:bold; border:none; border-radius:4px; padding:6px 14px; cursor:pointer; font-size:12px;">➕ 新建 Skill</button>
            <button class="h3-del-btn" style="background:#ff6188; color:#fff; font-weight:bold; border:none; border-radius:4px; padding:6px 14px; cursor:pointer; font-size:12px;">🗑️ 删除此 Skill</button>
            <button class="h3-learn-btn" style="background:#ffd866; color:#111; font-weight:bold; border:none; border-radius:4px; padding:6px 14px; cursor:pointer; font-size:12px;">🧠 学习此 Skill</button>
            <button class="h3-learn-all-btn" style="background:#ab9df2; color:#111; font-weight:bold; border:none; border-radius:4px; padding:6px 14px; cursor:pointer; font-size:12px;">🧠 学习全部</button>
            <button class="h3-upload-dir-btn" style="background:#a9dc76; color:#111; font-weight:bold; border:none; border-radius:4px; padding:6px 14px; cursor:pointer; font-size:12px;">📁 导入技能目录</button>
        </div>

        <div style="font-size:12px; color:#aaa; display:flex; justify-content:space-between;">
            <span>Skill 内容编辑 (Markdown 格式)：</span>
            <span class="h3-file-hint" style="color:#ffd866;"></span>
        </div>
        <textarea class="h3-skill-editor" style="flex:1; background:#101016; color:#a9dc76; border:1px solid #333; border-radius:6px; padding:12px; font-family:Consolas, Monaco, monospace; font-size:13px; line-height:1.5; resize:none; outline:none;"></textarea>

        <div style="display:flex; justify-content:space-between; align-items:center; border-top:1px solid #333; padding-top:10px;">
            <span class="h3-status-msg" style="font-size:12px; color:#ffd866;"></span>
            <div style="display:flex; gap:10px;">
                <button class="h3-cancel-btn" style="background:#333; color:#fff; border:none; border-radius:4px; padding:8px 18px; cursor:pointer; font-size:13px;">关闭</button>
                <button class="h3-save-btn" style="background:#78dce8; color:#111; font-weight:bold; border:none; border-radius:4px; padding:8px 24px; cursor:pointer; font-size:13px;">💾 保存文件</button>
            </div>
        </div>
    `;

    dialog.appendChild(box);
    document.body.appendChild(dialog);

    const selectEl = box.querySelector(".h3-skill-select");
    const editorEl = box.querySelector(".h3-skill-editor");
    const hintEl = box.querySelector(".h3-file-hint");
    const statusEl = box.querySelector(".h3-status-msg");
    const closeBtn = box.querySelector(".h3-close-btn");
    const cancelBtn = box.querySelector(".h3-cancel-btn");
    const saveBtn = box.querySelector(".h3-save-btn");
    const newBtn = box.querySelector(".h3-new-btn");
    const delBtn = box.querySelector(".h3-del-btn");
    const learnBtn = box.querySelector(".h3-learn-btn");
    const learnAllBtn = box.querySelector(".h3-learn-all-btn");
    const uploadDirBtn = box.querySelector(".h3-upload-dir-btn");

    let currentSkills = [];

    async function loadSkills(selectName = null) {
        try {
            const res = await fetch("/minimax-h3-agent/skills");
            const data = await res.json();
            currentSkills = data.details || [];
            selectEl.innerHTML = "";
            currentSkills.forEach((item) => {
                const opt = document.createElement("option");
                opt.value = item.name;
                opt.textContent = item.name === "none" ? "none (无预设)" : item.name;
                selectEl.appendChild(opt);
            });
            if (selectName) selectEl.value = selectName;
            onSelectChange();
        } catch (err) {
            statusEl.textContent = `加载失败: ${err.message}`;
        }
    }

    function onSelectChange() {
        const name = selectEl.value;
        const found = currentSkills.find((item) => item.name === name);
        if (name === "none") {
            editorEl.value = "";
            editorEl.disabled = true;
            hintEl.textContent = "未选择任何 Skill 预设";
            delBtn.disabled = true;
            saveBtn.disabled = true;
        } else {
            editorEl.disabled = false;
            saveBtn.disabled = false;
            editorEl.value = found ? found.content || "" : "";
            hintEl.textContent = found?.deletable ? "自定义可编辑/删除预设" : "系统内置默认预设";
            delBtn.disabled = !found || !found.deletable;
        }
        delBtn.style.opacity = delBtn.disabled ? "0.4" : "1";
        saveBtn.style.opacity = saveBtn.disabled ? "0.4" : "1";
    }

    selectEl.onchange = onSelectChange;

    saveBtn.onclick = async () => {
        const name = selectEl.value;
        if (!name || name === "none") {
            statusEl.textContent = "请先选择一个有效的 Skill 文件";
            return;
        }
        try {
            const res = await fetch("/minimax-h3-agent/skills", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ filename: name, content: editorEl.value }),
            });
            const data = await res.json();
            if (data.success) {
                statusEl.textContent = `✅ 已成功保存 ${name}`;
                await loadSkills(name);
                if (node) refreshSkillOptions(node, name);
            } else {
                statusEl.textContent = `保存失败: ${data.error}`;
            }
        } catch (err) {
            statusEl.textContent = `保存错误: ${err.message}`;
        }
    };

    newBtn.onclick = async () => {
        const inputName = prompt("请输入新 Skill 文件名 (例如: my_style.md)：");
        if (!inputName || !inputName.trim()) return;
        let filename = inputName.trim();
        if (!/\.(md|txt)$/i.test(filename)) filename += ".md";

        try {
            const res = await fetch("/minimax-h3-agent/skills", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ filename, content: "# 自定义 MiniMax H3 提示词策略规则\n\n- 电影感高精质感描述\n" }),
            });
            const data = await res.json();
            if (data.success) {
                statusEl.textContent = `✅ 已创建 ${filename}`;
                await loadSkills(filename);
                if (node) refreshSkillOptions(node, filename);
            } else {
                statusEl.textContent = `创建失败: ${data.error}`;
            }
        } catch (err) {
            statusEl.textContent = `创建错误: ${err.message}`;
        }
    };

    delBtn.onclick = async () => {
        const name = selectEl.value;
        if (!name || name === "none" || name === "default.md") return;
        if (!confirm(`确定要删除 Skill 文件 ${name} 吗？`)) return;

        try {
            const res = await fetch(`/minimax-h3-agent/skills`, {
                method: "DELETE",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ filename: name }),
            });
            const data = await res.json();
            if (data.success) {
                statusEl.textContent = `🗑️ 已删除 ${name}`;
                await loadSkills();
                if (node) refreshSkillOptions(node);
            } else {
                statusEl.textContent = `删除失败: ${data.error}`;
            }
        } catch (err) {
            statusEl.textContent = `删除错误: ${err.message}`;
        }
    };

    // Learning sends each Skill through the LLM once so it can write a compact,
    // executable spec. That digest is what the writer receives at runtime --
    // an order of magnitude smaller than the raw package, and without the
    // pipeline chapters that dilute the rules that actually shape the output.
    async function runLearn(payload, label) {
        const buttons = [learnBtn, learnAllBtn];
        const llmService = node ? String(getWidget(node, "llm_service")?.value || "") : "";
        if (!llmService) {
            statusEl.textContent = "未找到 llm_service，请先在节点上选择 LLM 服务";
            return;
        }
        const unload = node ? Boolean(getWidget(node, "ollama_auto_unload")?.value) : false;
        buttons.forEach((btn) => { btn.disabled = true; btn.style.opacity = "0.5"; });
        statusEl.textContent = `🧠 正在让 ${llmService} 学习${label}，长技能会分段阅读，请稍候...`;
        try {
            const res = await fetch("/minimax-h3-agent/learn", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ ...payload, llm_service: llmService, ollama_auto_unload: unload }),
            });
            const data = await res.json();
            if (data.success) {
                const ok = (data.results || []).filter((item) => item.success);
                const viaLlm = ok.filter((item) => item.learned_by === "llm").length;
                const raw = ok.reduce((sum, item) => sum + Number(item.chars || 0), 0);
                const digest = ok.reduce((sum, item) => sum + Number(item.digest_chars || 0), 0);
                const failed = (data.results || []).filter((item) => !item.success);
                let text = `✅ 已学习 ${data.learned}/${data.total}（LLM 精炼 ${viaLlm} 个）`;
                if (digest) text += ` ｜ ${raw.toLocaleString()} → ${digest.toLocaleString()} 字符`;
                if (failed.length) text += ` ｜ 失败 ${failed.length}: ${failed[0].error || ""}`;
                const notes = ok.flatMap((item) => item.notes || []);
                if (notes.length) text += ` ｜ 注意: ${notes[0]}`;
                statusEl.textContent = text;
            } else {
                const first = (data.results || []).find((item) => !item.success);
                statusEl.textContent = `学习失败: ${data.error || first?.error || "未知错误"}`;
            }
        } catch (err) {
            statusEl.textContent = `学习错误: ${err.message}`;
        } finally {
            buttons.forEach((btn) => { btn.disabled = false; btn.style.opacity = "1"; });
        }
    }

    learnBtn.onclick = async () => {
        const name = selectEl.value;
        if (!name || name === "none") {
            statusEl.textContent = "请先选择一个有效的 Skill";
            return;
        }
        await runLearn({ name }, ` ${name}`);
    };

    learnAllBtn.onclick = async () => {
        await runLearn({ all: true }, "全部技能");
    };

    uploadDirBtn.onclick = async () => {
        const dirPath = window.prompt("请输入技能导入目录或其相对路径：\n（管理员必须先设置 MINIMAX_H3_SKILLS_IMPORT_DIR；只允许导入该目录内的 UTF-8 文本技能）");
        if (!dirPath || !dirPath.trim()) return;
        statusEl.textContent = "📁 正在导入技能目录...";
        uploadDirBtn.disabled = true;
        uploadDirBtn.style.opacity = "0.5";
        try {
            const res = await fetch("/minimax-h3-agent/skills/upload-dir", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ dir_path: dirPath.trim() }),
            });
            const data = await res.json().catch(() => ({}));
            if (!res.ok || !data.success) {
                statusEl.textContent = `导入失败: ${data.error || res.status}`;
                return;
            }
            const imported = data.imported || [];
            const skipped = data.skipped || [];
            let msg = `✅ 已导入 ${data.imported_count} 个技能`;
            if (imported.length) msg += `: ${imported.join(", ")}`;
            if (skipped.length) msg += ` ｜ 跳过 ${skipped.length} 个: ${skipped.slice(0, 3).join(", ")}${skipped.length > 3 ? "..." : ""}`;
            statusEl.textContent = msg;
            await loadSkills(selectEl.value);
            if (node) await refreshSkillOptions(node, selectEl.value);
        } catch (err) {
            statusEl.textContent = `导入错误: ${err.message}`;
        } finally {
            uploadDirBtn.disabled = false;
            uploadDirBtn.style.opacity = "1";
        }
    };

    const close = () => dialog.remove();
    closeBtn.onclick = close;
    cancelBtn.onclick = close;

    await loadSkills(node ? getWidget(node, "skill_preset")?.value : null);
}

async function openLLMConfigModal() {
    const dialog = document.createElement("div");
    dialog.style.cssText = "position:fixed;top:0;left:0;width:100vw;height:100vh;background:rgba(0,0,0,0.75);z-index:10000;display:flex;align-items:center;justify-content:center;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;";

    const box = document.createElement("div");
    box.style.cssText = "width:1120px;max-width:96vw;height:760px;max-height:92vh;background:#181822;border:1px solid #333348;border-radius:12px;padding:20px;display:flex;flex-direction:column;gap:12px;color:#e0e0e0;box-shadow:0 12px 40px rgba(0,0,0,0.6);";

    let services = [];
    let aliases = {};
    let selectedIdx = 0;

    box.innerHTML = `
        <div style="display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid #333;padding-bottom:10px;">
            <div>
                <div style="font-size:16px;font-weight:bold;color:#a9dc76;">🔧 Myang LLM 服务设置</div>
                <div class="llm-source" style="font-size:11px;color:#888;margin-top:3px;"></div>
            </div>
            <button class="llm-close" style="background:transparent;border:none;color:#aaa;font-size:20px;cursor:pointer;">✖</button>
        </div>
        <div style="display:flex;gap:12px;flex:1;min-height:0;">
            <div style="width:250px;display:flex;flex-direction:column;gap:6px;background:#222230;border-radius:6px;padding:10px;">
                <div style="font-size:12px;color:#78dce8;font-weight:bold;margin-bottom:4px;">服务列表</div>
                <div class="llm-svc-list" style="flex:1;overflow-y:auto;display:flex;flex-direction:column;gap:4px;"></div>
                <button class="llm-add-svc" style="background:#a9dc76;color:#111;font-weight:bold;border:none;border-radius:4px;padding:6px;cursor:pointer;font-size:12px;">➕ 添加服务</button>
            </div>
            <div class="llm-edit-panel" style="flex:1;overflow-y:auto;background:#222230;border-radius:6px;padding:14px;"></div>
        </div>
        <div style="display:flex;justify-content:space-between;align-items:center;border-top:1px solid #333;padding-top:10px;">
            <span class="llm-status" style="font-size:12px;color:#ffd866;"></span>
            <div style="display:flex;gap:10px;">
                <button class="llm-cancel" style="background:#333;color:#fff;border:none;border-radius:4px;padding:8px 18px;cursor:pointer;font-size:13px;">关闭</button>
                <button class="llm-save" style="background:#a9dc76;color:#111;font-weight:bold;border:none;border-radius:4px;padding:8px 24px;cursor:pointer;font-size:13px;">💾 保存配置</button>
            </div>
        </div>
    `;
    dialog.appendChild(box);
    document.body.appendChild(dialog);

    const svcListEl = box.querySelector(".llm-svc-list");
    const editPanelEl = box.querySelector(".llm-edit-panel");
    const statusEl = box.querySelector(".llm-status");
    const sourceEl = box.querySelector(".llm-source");
    const closeBtn = box.querySelector(".llm-close");
    const cancelBtn = box.querySelector(".llm-cancel");
    const saveBtn = box.querySelector(".llm-save");
    const addSvcBtn = box.querySelector(".llm-add-svc");

    const STY = "background:#121218;color:#fff;border:1px solid #444;border-radius:4px;padding:6px 10px;font-size:13px;width:100%;box-sizing:border-box;";
    const LBL = "font-size:12px;color:#78dce8;font-weight:bold;margin-bottom:3px;display:block;";
    const ROW = "margin-bottom:10px;";

    function newServiceId() {
        const used = new Set(services.map((service) => String(service.id || "").toLowerCase()));
        let suffix = Date.now().toString(36);
        let candidate = `myang_${suffix}`;
        let counter = 1;
        while (used.has(candidate.toLowerCase())) {
            candidate = `myang_${suffix}_${counter}`;
            counter += 1;
        }
        return candidate;
    }

    function newRouteId(service) {
        const used = new Set((service.routes || []).map((route) => String(route.id || "").toLowerCase()));
        let index = 1;
        let candidate = `route_${index}`;
        while (used.has(candidate.toLowerCase())) {
            index += 1;
            candidate = `route_${index}`;
        }
        return candidate;
    }

    function modelOption(service, model) {
        return `${service.name || service.id} :: ${model.name}`;
    }

    function migrateSelection(value, modelKey) {
        const current = String(value || "").trim();
        if (!current || current === "off") return current;
        let serviceRef = current;
        let modelName = "";
        if (current.includes(" :: ")) {
            [serviceRef, modelName] = current.split(" :: ", 2);
        } else if (current.includes("/")) {
            [serviceRef, modelName] = current.split("/", 2);
        }
        const aliasTarget = aliases[serviceRef];
        const service = services.find((item) =>
            item.id === serviceRef || item.name === serviceRef || item.id === aliasTarget);
        if (!service) return current;
        const models = service[modelKey] || [];
        const model = models.find((item) => item.name === modelName)
            || models.find((item) => item.is_default)
            || models[0];
        return model ? modelOption(service, model) : current;
    }

    function refreshAgentServiceWidgets(llmOptions, vlmOptions) {
        for (const node of app.graph?._nodes || []) {
            if (!isAgent(node)) continue;
            for (const [widgetName, values, modelKey] of [
                ["llm_service", llmOptions, "llm_models"],
                ["vlm_service", vlmOptions, "vlm_models"],
            ]) {
                const widget = getWidget(node, widgetName);
                if (!widget) continue;
                widget.options ||= {};
                widget.options.values = values;
                const migrated = migrateSelection(widget.value, modelKey);
                if (values.includes(migrated)) widget.value = migrated;
            }
            node.setDirtyCanvas?.(true, true);
        }
    }

    function renderSvcList() {
        svcListEl.innerHTML = "";
        services.forEach((svc, idx) => {
            const btn = document.createElement("button");
            btn.style.cssText = `background:${idx===selectedIdx?"#a9dc76":"#333"};color:${idx===selectedIdx?"#111":"#ccc"};border:none;border-radius:4px;padding:8px 10px;cursor:pointer;font-size:12px;text-align:left;`;
            const name = svc.name || svc.id || "?";
            const llmCount = (svc.llm_models || []).length;
            const vlmCount = (svc.vlm_models || []).length;
            const routeCount = (svc.routes || []).filter((route) => route.enabled !== false).length;
            const readyCount = (svc.routes || []).filter((route) =>
                route.enabled !== false && ["ready", "active"].includes(String(route.runtime?.status || "ready"))).length;
            btn.innerHTML = `<div style="font-weight:600;">${escapeHtml(name)}</div><div style="font-size:10px;opacity:.75;margin-top:2px;">LLM ${llmCount} · VLM ${vlmCount}</div>`;
            btn.innerHTML += `<div style="font-size:10px;opacity:.75;margin-top:2px;">API 可用 ${readyCount}/${routeCount} · 共 ${(svc.routes || []).length} 条</div>`;
            btn.onclick = () => { selectedIdx = idx; renderSvcList(); renderEditPanel(); };
            svcListEl.appendChild(btn);
        });
    }

    function renderEditPanel() {
        const svc = services[selectedIdx];
        if (!svc) {
            editPanelEl.innerHTML = '<div style="color:#888;text-align:center;padding:40px;">选择或添加一个服务</div>';
            return;
        }
        const llmRows = (svc.llm_models || []).map((m, i) => modelRow(m, i, "llm")).join("");
        const vlmRows = (svc.vlm_models || []).map((m, i) => modelRow(m, i, "vlm")).join("");
        svc.routes = Array.isArray(svc.routes) ? svc.routes : [];
        const routeRows = svc.routes.map((route, index) => routeRow(route, index)).join("");
        const canResetRuntime = Boolean(svc._original_id);

        editPanelEl.innerHTML = `
            <div style="${ROW}"><label style="${LBL}">显示名称（节点下拉中显示）</label><input class="fld-name" style="${STY}" value="${escapeHtml(svc.name || "")}"></div>
            <div style="${ROW}"><label style="${LBL}">内部 ID（工作流稳定标识）</label><input class="fld-id" style="${STY}color:#999;" value="${escapeHtml(svc.id || "")}" readonly title="为避免旧工作流失效，已有服务的内部 ID 不允许直接修改"><div style="font-size:10px;color:#777;margin-top:3px;">ID 只用于兼容旧工作流；日常选择使用上面的显示名称。</div></div>
            <div style="${ROW}"><label style="${LBL}">类型</label><select class="fld-type" style="${STY}">
                ${["openai_compatible","ollama"].map(t=>`<option value="${t}" ${svc.type===t?"selected":""}>${t}</option>`).join("")}
            </select></div>
            <div style="border-top:1px solid #444;padding-top:10px;margin-top:10px;">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;gap:10px;">
                    <div><span style="${LBL}margin:0;">API Key 组路由</span><div style="font-size:10px;color:#888;margin-top:3px;">每组线路独立保存 URL 和 Key；密钥不会回传到浏览器。</div></div>
                    <div style="display:flex;gap:6px;flex:0 0 auto;">
                        <button class="reset-route-state" ${canResetRuntime ? "" : "disabled"} title="${canResetRuntime ? "清除冷却、熔断和轮询位置，不修改配置" : "请先保存新服务"}" style="flex:0 0 104px;width:104px;background:#333348;color:#ddd;border:1px solid #4b4b62;border-radius:4px;padding:5px 8px;cursor:${canResetRuntime ? "pointer" : "not-allowed"};font-size:11px;white-space:nowrap;opacity:${canResetRuntime ? "1" : ".45"};">重置线路状态</button>
                        <button class="add-route" style="flex:0 0 88px;width:88px;background:#444;color:#fff;border:none;border-radius:4px;padding:5px 8px;cursor:pointer;font-size:11px;white-space:nowrap;">添加线路</button>
                    </div>
                </div>
                <div style="${ROW}"><label style="${LBL}">路由策略</label><select class="fld-route-strategy" style="${STY}">
                    <option value="round_robin" ${(svc.route_strategy || "round_robin") === "round_robin" ? "selected" : ""}>轮询 + 故障转移（推荐）</option>
                    <option value="failover" ${svc.route_strategy === "failover" ? "selected" : ""}>主线路优先 + 故障转移</option>
                </select><div style="font-size:10px;color:#777;margin-top:3px;">429/TPM、连接、超时、鉴权或服务端故障会自动尝试下一组；普通请求参数错误不会盲目换线。</div></div>
                <div class="route-groups" style="display:flex;flex-direction:column;gap:8px;">${routeRows}</div>
            </div>
            <div style="border-top:1px solid #444;padding-top:10px;margin-top:10px;">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
                    <span style="${LBL}margin:0;">LLM 模型</span>
                    <button class="add-llm" style="background:#444;color:#fff;border:none;border-radius:4px;padding:4px 10px;cursor:pointer;font-size:11px;">➕ 添加</button>
                </div>
                <div style="display:grid;grid-template-columns:22px 1fr 58px 58px 82px 32px;gap:6px;color:#777;font-size:9px;margin-bottom:3px;"><span>默认</span><span>模型名</span><span>温度</span><span>Top P</span><span>最大输出</span><span></span></div>
                <div class="llm-models">${llmRows}</div>
            </div>
            <div style="border-top:1px solid #444;padding-top:10px;margin-top:10px;">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
                    <span style="${LBL}margin:0;">VLM 模型</span>
                    <button class="add-vlm" style="background:#444;color:#fff;border:none;border-radius:4px;padding:4px 10px;cursor:pointer;font-size:11px;">➕ 添加</button>
                </div>
                <div style="display:grid;grid-template-columns:22px 1fr 58px 58px 82px 32px;gap:6px;color:#777;font-size:9px;margin-bottom:3px;"><span>默认</span><span>模型名</span><span>温度</span><span>Top P</span><span>最大输出</span><span></span></div>
                <div class="vlm-models">${vlmRows}</div>
            </div>
            <div style="margin-top:16px;">
                <button class="del-svc" style="background:#ff6188;color:#fff;border:none;border-radius:4px;padding:6px 14px;cursor:pointer;font-size:12px;">🗑️ 删除此服务</button>
            </div>
        `;

        const bindField = (cls, key) => {
            editPanelEl.querySelector(cls).oninput = (e) => {
                svc[key] = e.target.value;
                if (key === "name") renderSvcList();
            };
        };
        bindField(".fld-name", "name");
        editPanelEl.querySelector(".fld-type").onchange = (e) => { svc.type = e.target.value; };
        editPanelEl.querySelector(".fld-route-strategy").onchange = (e) => { svc.route_strategy = e.target.value; };
        editPanelEl.querySelector(".reset-route-state").onclick = async (event) => {
            if (!canResetRuntime) return;
            const button = event.currentTarget;
            button.disabled = true;
            button.style.opacity = "0.5";
            statusEl.textContent = `正在重置 ${svc.name || svc.id} 的线路状态…`;
            try {
                const res = await fetch("/minimax-h3-agent/llm-routes/reset", {
                    method: "POST",
                    headers: {"Content-Type": "application/json"},
                    body: JSON.stringify({service_id: svc.id || svc.name || ""}),
                });
                const data = await res.json().catch(() => ({}));
                if (!res.ok || !data.success) throw new Error(data.error || `HTTP ${res.status}`);
                // Preserve unsaved URL/model/key edits in this modal. Only
                // merge the freshly reset runtime fields from the server.
                const refreshed = (data.services || []).find((item) =>
                    String(item.id || "") === String(svc.id || ""));
                if (refreshed) {
                    svc.runtime = refreshed.runtime || {};
                    for (const route of svc.routes || []) {
                        const freshRoute = (refreshed.routes || []).find((item) =>
                            String(item.id || "") === String(route.id || ""));
                        if (freshRoute) route.runtime = freshRoute.runtime || {};
                    }
                }
                renderSvcList();
                renderEditPanel();
                statusEl.textContent = `已清除 ${svc.name || svc.id} 的冷却、熔断与轮询位置`;
            } catch (err) {
                statusEl.textContent = `线路状态重置失败: ${err.message}`;
                button.disabled = false;
                button.style.opacity = "1";
            }
        };
        editPanelEl.querySelector(".add-route").onclick = () => {
            const routeId = newRouteId(svc);
            const firstUrl = String(svc.routes?.[0]?.base_url || "");
            svc.routes.push({_original_id:"",id:routeId,name:`线路 ${svc.routes.length + 1}`,enabled:true,base_url:firstUrl,api_key_configured:false,api_key_action:"clear"});
            renderEditPanel(); renderSvcList();
        };
        editPanelEl.querySelectorAll(".route-card").forEach((card) => {
            const index = parseInt(card.dataset.idx, 10);
            const route = svc.routes[index];
            card.querySelector(".r-enabled").onchange = (e) => { route.enabled = e.target.checked; renderSvcList(); };
            card.querySelector(".r-name").oninput = (e) => { route.name = e.target.value; };
            card.querySelector(".r-url").oninput = (e) => { route.base_url = e.target.value; };
            card.querySelector(".r-key").oninput = (e) => {
                const value = e.target.value;
                if (value) {
                    route.api_key = value;
                    route.api_key_action = "set";
                } else {
                    delete route.api_key;
                    route.api_key_action = route.api_key_configured ? "keep" : "clear";
                }
            };
            card.querySelector(".r-clear-key").onclick = () => {
                if (!route.api_key_configured) return;
                route.api_key_action = route.api_key_action === "clear" ? "keep" : "clear";
                delete route.api_key;
                renderEditPanel();
            };
            card.querySelector(".r-del").onclick = () => {
                if (svc.routes.length <= 1) return;
                svc.routes.splice(index, 1);
                renderEditPanel(); renderSvcList();
            };
        });

        editPanelEl.querySelector(".add-llm").onclick = () => {
            svc.llm_models = svc.llm_models || [];
            svc.llm_models.push({name:`new-model-${svc.llm_models.length + 1}`,is_default:svc.llm_models.length===0,temperature:0.7,max_tokens:0,top_p:0.9});
            renderEditPanel(); renderSvcList();
        };
        editPanelEl.querySelector(".add-vlm").onclick = () => {
            svc.vlm_models = svc.vlm_models || [];
            svc.vlm_models.push({name:`new-vlm-model-${svc.vlm_models.length + 1}`,is_default:svc.vlm_models.length===0,temperature:0.7,max_tokens:4096,top_p:0.9});
            renderEditPanel(); renderSvcList();
        };
        editPanelEl.querySelector(".del-svc").onclick = () => {
            if (!confirm(`删除服务 ${svc.name || svc.id}？`)) return;
            services.splice(selectedIdx, 1);
            selectedIdx = Math.max(0, selectedIdx - 1);
            renderSvcList(); renderEditPanel();
        };

        editPanelEl.querySelectorAll(".model-row").forEach(row => {
            const type = row.dataset.type;
            const idx = parseInt(row.dataset.idx);
            const arr = type === "llm" ? (svc.llm_models||[]) : (svc.vlm_models||[]);
            row.querySelector(".m-name").oninput = (e) => { arr[idx].name = e.target.value; renderSvcList(); };
            row.querySelector(".m-temp").oninput = (e) => { arr[idx].temperature = parseFloat(e.target.value); };
            row.querySelector(".m-max").oninput = (e) => {
                const value = parseInt(e.target.value, 10);
                arr[idx].max_tokens = Number.isFinite(value) ? Math.max(0, value) : 0;
            };
            row.querySelector(".m-top-p").oninput = (e) => { arr[idx].top_p = parseFloat(e.target.value); };
            row.querySelector(".m-default").onchange = () => {
                arr.forEach((model, modelIndex) => { model.is_default = modelIndex === idx; });
                renderEditPanel();
            };
            row.querySelector(".m-del").onclick = () => {
                const wasDefault = Boolean(arr[idx]?.is_default);
                arr.splice(idx, 1);
                if (wasDefault && arr.length) arr[0].is_default = true;
                renderEditPanel(); renderSvcList();
            };
        });
    }

    function modelRow(m, idx, type) {
        const MS = "background:#121218;color:#fff;border:1px solid #444;border-radius:3px;padding:4px 8px;font-size:12px;";
        return `<div class="model-row" data-type="${type}" data-idx="${idx}" style="display:flex;gap:6px;align-items:center;margin-bottom:4px;">
            <input class="m-default" type="radio" name="${type}-default-${selectedIdx}" ${m.is_default ? "checked" : ""} title="设为默认模型">
            <input class="m-name" style="${MS}flex:1;min-width:180px;" value="${escapeHtml(m.name || "")}" placeholder="模型名">
            <input class="m-temp" style="${MS}width:52px;" value="${escapeHtml(String(m.temperature ?? 0.7))}" type="number" min="0" max="2" step="0.1" title="temperature">
            <input class="m-top-p" style="${MS}width:52px;" value="${escapeHtml(String(m.top_p ?? 0.9))}" type="number" min="0.01" max="1" step="0.05" title="top_p">
            <input class="m-max" style="${MS}width:76px;" value="${escapeHtml(String(m.max_tokens ?? 0))}" type="number" min="0" title="最大输出 token；0 = 不发送限制，由服务商/模型决定">
            <button class="m-del" style="background:#ff6188;color:#fff;border:none;border-radius:3px;padding:4px 8px;cursor:pointer;font-size:11px;">🗑️</button>
        </div>`;
    }

    function routeRow(route, index) {
        const keyState = route.api_key_action === "clear"
            ? "保存后将清除该线路的 API Key"
            : (route.api_key_configured ? "已安全保存；留空保持原值" : "尚未配置 Key（Ollama 可留空）");
        const canDelete = (services[selectedIdx]?.routes || []).length > 1;
        const routeView = routeStatusPresentation(route);
        return `<div class="route-card" data-idx="${index}" style="background:#191923;border:1px solid ${route.enabled === false ? "#3b3b46" : "#4f5968"};border-radius:6px;padding:9px;opacity:${route.enabled === false ? ".62" : "1"};">
            <div style="display:flex;align-items:center;gap:7px;margin-bottom:7px;">
                <input class="r-enabled" type="checkbox" ${route.enabled === false ? "" : "checked"} title="启用此线路">
                <input class="r-name" style="${STY}flex:1;min-width:120px;" value="${escapeHtml(route.name || `线路 ${index + 1}`)}" placeholder="线路名称">
                <span title="${escapeHtml(routeView.title)}" style="flex:0 0 auto;border:1px solid ${routeView.border};background:${routeView.background};color:${routeView.color};border-radius:999px;padding:3px 7px;font-size:10px;font-weight:bold;white-space:nowrap;">${escapeHtml(routeView.label)}</span>
                <span style="flex:0 0 auto;font-size:10px;color:#777;" title="稳定线路 ID">${escapeHtml(route.id || "")}</span>
                <button class="r-del" ${canDelete ? "" : "disabled"} style="flex:0 0 32px;background:#ff6188;color:#fff;border:none;border-radius:3px;padding:5px 6px;cursor:${canDelete ? "pointer" : "not-allowed"};font-size:11px;opacity:${canDelete ? "1" : ".35"};" title="删除线路">🗑️</button>
            </div>
            <input class="r-url" style="${STY}margin-bottom:6px;" value="${escapeHtml(route.base_url || "")}" placeholder="Base URL，例如 https://api.example.com/v1">
            <div style="display:flex;gap:6px;align-items:center;"><input class="r-key" style="${STY}flex:1;" value="" type="password" autocomplete="new-password" placeholder="输入新 Key；留空保持原值"><button class="r-clear-key" ${route.api_key_configured ? "" : "disabled"} style="flex:0 0 108px;white-space:nowrap;background:#444;color:#ddd;border:none;border-radius:4px;padding:6px 8px;cursor:${route.api_key_configured ? "pointer" : "not-allowed"};font-size:11px;opacity:${route.api_key_configured ? "1" : ".45"};">${route.api_key_action === "clear" && route.api_key_configured ? "↩ 保留原 Key" : "清除已存 Key"}</button></div>
            <div style="display:flex;justify-content:space-between;gap:10px;font-size:10px;margin-top:3px;">
                <span style="color:${route.api_key_action === "clear" ? "#ff6188" : "#888"};">${escapeHtml(keyState)}</span>
                <span style="color:#777;text-align:right;">成功 ${Number(route.runtime?.successes || 0)} · 失败 ${Number(route.runtime?.failures || 0)} · 转移 ${Number(route.runtime?.failovers || 0)}</span>
            </div>
        </div>`;
    }

    addSvcBtn.onclick = () => {
        const id = newServiceId();
        services.push({_original_id:"",id,type:"openai_compatible",name:`新服务 ${services.length + 1}`,route_strategy:"round_robin",routes:[{_original_id:"",id:"route_1",name:"线路 1",enabled:true,base_url:"",api_key_configured:false,api_key_action:"clear"}],llm_models:[],vlm_models:[]});
        selectedIdx = services.length - 1;
        renderSvcList(); renderEditPanel();
    };

    saveBtn.onclick = async () => {
        statusEl.textContent = "💾 正在保存...";
        saveBtn.disabled = true;
        try {
            const res = await fetch("/minimax-h3-agent/llm-config", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({services}),
            });
            const data = await res.json().catch(()=>({}));
            if (res.ok && data.success) {
                services = Array.isArray(data.services) ? data.services : services;
                aliases = data.aliases || {};
                selectedIdx = Math.min(selectedIdx, Math.max(0, services.length - 1));
                renderSvcList();
                renderEditPanel();
                refreshAgentServiceWidgets(data.llm_options || [], data.vlm_options || ["off"]);
                sourceEl.textContent = "配置位置：Myang_node（已与提示词小助手配置解耦）";
                statusEl.textContent = `✅ 已保存 ${data.service_count} 个服务，节点下拉已刷新`;
            } else {
                statusEl.textContent = `保存失败: ${data.error || res.status}`;
            }
        } catch (err) {
            statusEl.textContent = `保存错误: ${err.message}`;
        } finally {
            saveBtn.disabled = false;
        }
    };

    const close = () => dialog.remove();
    closeBtn.onclick = close;
    cancelBtn.onclick = close;

    // Load config
    statusEl.textContent = "加载配置中...";
    try {
        const res = await fetch("/minimax-h3-agent/llm-config");
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
        services = data.services || [];
        aliases = data.aliases || {};
        sourceEl.textContent = data.source === "legacy"
            ? "当前从旧 prompt-assistant 配置只读导入；首次保存后迁入 Myang_node"
            : "配置位置：Myang_node";
        statusEl.textContent = "";
        renderSvcList();
        renderEditPanel();
    } catch (err) {
        statusEl.textContent = `加载失败: ${err.message}`;
    }
}

function openSubjectBindingModal(node) {
    const connections = syncAgentConnectionsFromNativeLinks(node);
    if (!connections.length) {
        window.alert?.("请先连接媒体素材，再绑定主体名称。");
        return;
    }
    const counts = { image: 0, video: 0, audio: 0 };
    const rows = [];
    for (const type of ["image", "video", "audio"]) {
        for (const connection of connections) {
            if (String(connection?.media_type || "image") !== type) continue;
            counts[type] += 1;
            rows.push({
                connection,
                tag: `<${MEDIA_TAG_NAMES[type]} ${counts[type]}>`,
                title: String(connection?.filename || connection?.label || "Media"),
                typeLabel: MEDIA_TYPE_LABELS[type],
            });
        }
    }

    const dialog = document.createElement("div");
    dialog.style.cssText = "position:fixed;inset:0;z-index:10010;background:rgba(0,0,0,.55);display:flex;align-items:center;justify-content:center;";
    const box = document.createElement("div");
    box.style.cssText = "width:min(620px,92vw);max-height:82vh;overflow:auto;background:#1b1b24;border:1px solid #33333f;border-radius:10px;padding:18px;color:#eee;font-size:13px;";
    box.innerHTML = `
        <div style="font-weight:bold;font-size:15px;margin-bottom:4px;">🎭 主体绑定</div>
        <div style="color:#9a9aa8;margin-bottom:14px;line-height:1.5;">
            给素材起个名字（例如「女主角小林」「男主阿杰」）。Agent 写提示词时会用这个名字指代该素材，
            而不是只说「画面里的人」，多镜头之间的人物指代也更稳定。留空表示不绑定。
        </div>
        <div class="h3-subject-rows"></div>
        <div style="display:flex;justify-content:flex-end;gap:10px;margin-top:16px;">
            <button class="h3-sub-clear" style="background:#333;color:#fff;border:none;border-radius:4px;padding:8px 16px;cursor:pointer;">清空全部</button>
            <button class="h3-sub-cancel" style="background:#333;color:#fff;border:none;border-radius:4px;padding:8px 18px;cursor:pointer;">取消</button>
            <button class="h3-sub-save" style="background:#78dce8;color:#111;font-weight:bold;border:none;border-radius:4px;padding:8px 24px;cursor:pointer;">保存</button>
        </div>`;
    const rowsEl = box.querySelector(".h3-subject-rows");
    const inputs = [];
    for (const row of rows) {
        const line = document.createElement("div");
        line.style.cssText = "display:flex;align-items:center;gap:10px;margin-bottom:9px;";
        const label = document.createElement("div");
        label.style.cssText = "flex:1;min-width:0;color:#a9dc76;font-family:Consolas,monospace;font-size:12px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;";
        label.textContent = `${row.tag} ${row.typeLabel} · ${row.title}`;
        label.title = row.title;
        const input = document.createElement("input");
        input.type = "text";
        input.maxLength = 40;
        input.placeholder = "主体名称（可留空）";
        input.value = subjectFor(node, row.connection);
        input.style.cssText = "width:210px;background:#14141c;border:1px solid #33333f;border-radius:4px;color:#eee;padding:6px 8px;font-size:12px;";
        inputs.push({ connection: row.connection, input });
        line.append(label, input);
        rowsEl.appendChild(line);
    }

    dialog.appendChild(box);
    document.body.appendChild(dialog);
    inputs[0]?.input?.focus?.();

    const close = () => dialog.remove();
    box.querySelector(".h3-sub-cancel").onclick = close;
    box.querySelector(".h3-sub-clear").onclick = () => {
        for (const entry of inputs) entry.input.value = "";
    };
    box.querySelector(".h3-sub-save").onclick = () => {
        for (const entry of inputs) setSubjectFor(node, entry.connection, entry.input.value);
        refreshMediaBadge(node);
        node.setDirtyCanvas?.(true, true);
        close();
    };
    dialog.addEventListener("mousedown", (event) => {
        if (event.target === dialog) close();
    });
}

function installAgentWidgets(node) {
    if (node.__h3AgentWidgetsInstalled) return;
    node.__h3AgentWidgetsInstalled = true;
    const skillWidget = getWidget(node, "skill_preset");
    if (skillWidget) skillWidget.serializeValue = () => getWidget(node, "skill_preset")?.value || "none";
    const linksWidget = getWidget(node, "asset_manifest_json");
    if (linksWidget) {
        linksWidget.hidden = true;
        linksWidget.computeSize = () => [0, -4];
        linksWidget.serializeValue = () => JSON.stringify(syncAgentConnectionsFromNativeLinks(node));
    }
    // The skill manager dialog owns Skill editing now, so the inline textarea is
    // redundant. It stays declared and serialised rather than being deleted:
    // ComfyUI restores widgets_values by position, so removing it outright
    // would shift every later value (including the seed) in saved workflows.
    const skillTextWidget = getWidget(node, "skill_text");
    if (skillTextWidget) {
        skillTextWidget.hidden = true;
        skillTextWidget.computeSize = () => [0, -4];
        if (skillTextWidget.inputEl) skillTextWidget.inputEl.style.display = "none";
    }
    hideAudioWidgetsWhenNoAudio(node);

    node.addWidget("button", "⚙️ Agent 技能管理", null, () => {
        openSkillManagerModal(node);
    });
    node.addWidget("button", "🎭 主体绑定", null, () => {
        openSubjectBindingModal(node);
    });
    node.addWidget("button", "上传 Skill 文件", null, async () => {
        const input = document.createElement("input");
        input.type = "file";
        input.accept = ".md,.txt,.json,.yaml,.yml,text/plain,text/markdown,application/json";
        input.onchange = async () => {
            try {
                await uploadSkillFile(node, input.files?.[0]);
            } catch (error) {
                console.error("[MiniMax H3 Media Agent]", error);
                window.alert?.(String(error?.message || error));
            }
        };
        input.click();
    });
    node.addWidget("button", "刷新 Skill 列表", null, async () => {
        try {
            await refreshSkillOptions(node);
        } catch (error) {
            console.error("[MiniMax H3 Media Agent]", error);
        }
    });

    // 节点创建时自动刷新：过滤掉未学习的技能，只显示 none + auto + 已学习
    setTimeout(() => {
        refreshSkillOptions(node).catch(() => {});
    }, 100);
}

const AGENT_TRANSPORT_INPUT_RE = /^(catalog|asset_\d+|asset_manifest_json)$/;

function pruneAgentTransportInputs(nodeData) {
    // asset_* and asset_manifest_json are transport-only inputs that
    // graphToPrompt fills in. Keeping them in the node definition would show 15
    // dead sockets plus a media-type text box the user is expected to type into;
    // asset_1 is kept as the single starting dot and the rest grow on demand.
    const optional = nodeData?.input?.optional;
    if (!optional) return;
    for (const name of Object.keys(optional)) {
        if (name === "asset_1") continue;
        if (AGENT_TRANSPORT_INPUT_RE.test(name)) delete optional[name];
    }
}

function isMediaSlot(input) {
    return /^asset_\d+$/.test(String(input?.name || ""));
}

function isSlotLinked(input) {
    if (Array.isArray(input?.links) && input.links.length > 0) return true;
    return input?.link != null;
}

function mediaSlotPositions(node) {
    const positions = [];
    (node?.inputs || []).forEach((input, index) => {
        if (isMediaSlot(input)) positions.push(index);
    });
    return positions;
}

function ensureAgentMediaSlots(node) {
    if (!node || !Array.isArray(node.inputs)) return;

    // Only a trailing free slot may be removed. Removing one in the middle
    // renumbers the slots after it, and LiteGraph links address inputs by
    // index, so every later connection would silently retarget.
    for (let guard = 0; guard <= MAX_MEDIA + 1; guard += 1) {
        const positions = mediaSlotPositions(node);
        if (positions.length <= 1) break;
        const last = node.inputs[positions[positions.length - 1]];
        const previous = node.inputs[positions[positions.length - 2]];
        if (isSlotLinked(last) || isSlotLinked(previous)) break;
        node.removeInput(positions[positions.length - 1]);
    }

    const positions = mediaSlotPositions(node);
    if (positions.length === 0) {
        node.addInput("asset_1", "*");
    } else if (positions.length < MAX_MEDIA && isSlotLinked(node.inputs[positions[positions.length - 1]])) {
        node.addInput(`asset_${positions.length + 1}`, "*");
    }

    // Renaming in place is safe (links reference the index, not the name) and
    // keeps the socket labels readable after a slot is disconnected.
    let ordinal = 0;
    for (const input of node.inputs) {
        if (!isMediaSlot(input)) continue;
        ordinal += 1;
        input.name = `asset_${ordinal}`;
        input.label = ordinal === 1 ? "media" : `media ${ordinal}`;
        if (!input.type) input.type = "*";
    }
}

function installAgentNode(nodeType, nodeData) {
    if (nodeData?.name !== AGENT_CLASS) return;
    if (nodeType.prototype.__h3AgentNodeInstalled) return;
    nodeType.prototype.__h3AgentNodeInstalled = true;
    pruneAgentTransportInputs(nodeData);

    const originalCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function onNodeCreatedH3Agent() {
        const result = originalCreated?.apply(this, arguments);
        this.properties ||= {};
        // Appended after the definition widgets so saved widgets_values, which
        // ComfyUI restores by position, keep lining up.
        installMediaBadgeWidget(this);
        installAgentWidgets(this);
        ensureAgentMediaSlots(this);
        refreshMediaBadge(this);
        return result;
    };

    const originalConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function onConfigureH3Agent(info) {
        // Guard the restored size across the widget rebuild below. Installing
        // widgets and slots re-runs the audio-visibility pass, which would
        // otherwise resize a node the user had already sized by hand.
        this.__h3RestoringLayout = true;
        const savedSize = Array.isArray(info?.size)
            ? [Number(info.size[0]) || 0, Number(info.size[1]) || 0]
            : (Array.isArray(this.size) ? [this.size[0], this.size[1]] : null);
        const result = originalConfigure?.apply(this, arguments);
        installMediaBadgeWidget(this);
        ensureAgentMediaSlots(this);
        refreshMediaBadge(this);
        requestAnimationFrame(() => requestAnimationFrame(() => {
            if (savedSize && savedSize[0] > 0 && savedSize[1] > 0 && this.size) {
                this.size[0] = savedSize[0];
                this.size[1] = savedSize[1];
            }
            this.__h3RestoringLayout = false;
            this.setDirtyCanvas?.(true, true);
        }));
        return result;
    };

    const originalConnectionsChange = nodeType.prototype.onConnectionsChange;
    nodeType.prototype.onConnectionsChange = function onConnectionsChangeH3Agent() {
        const result = originalConnectionsChange?.apply(this, arguments);
        // LiteGraph finishes wiring the slot after this callback returns, so
        // read the graph back on the next frame instead of trusting it now.
        requestAnimationFrame(() => {
            if (!this.graph) return;
            ensureAgentMediaSlots(this);
            refreshMediaBadge(this);
            this.setDirtyCanvas?.(true, true);
        });
        return result;
    };

    const originalDraw = nodeType.prototype.onDrawForeground;
    nodeType.prototype.onDrawForeground = function onDrawForegroundH3Agent() {
        const result = originalDraw?.apply(this, arguments);
        // Picks up upstream changes that fire no connection event, such as
        // choosing a different file in the connected loader. Throttled because
        // this runs on every canvas frame.
        const now = performance.now();
        if (!this.__h3BadgeCheckedAt || now - this.__h3BadgeCheckedAt > 400) {
            this.__h3BadgeCheckedAt = now;
            refreshMediaBadge(this);
            hideAudioWidgetsWhenNoAudio(this);
        }
        return result;
    };

    const originalRemoved = nodeType.prototype.onRemoved;
    nodeType.prototype.onRemoved = function onRemovedH3Agent() {
        this.__h3BadgeCheckedAt = 0;
        return originalRemoved?.apply(this, arguments);
    };
}

function escapeHtml(str) {
    return String(str || "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

app.registerExtension({
    name: "Myang_node.MiniMaxH3.MediaAgent",
    async setup() {
        patchGraphToPrompt();
        if (app.ui?.settings?.addSetting) {
            let availableSkills = ["none"];
            try {
                const response = await fetch("/minimax-h3-agent/skills");
                if (response.ok) {
                    const payload = await response.json();
                    if (Array.isArray(payload.skills)) availableSkills = payload.skills;
                }
            } catch (e) {}

            app.ui.settings.addSetting({
                id: settingId("DefaultSkillPreset"),
                name: "沐阳 H3: 默认 Skill 预设 (Default Skill Preset)",
                type: "combo",
                defaultValue: migratedSettingDefault("DefaultSkillPreset", "none"),
                options: availableSkills.map((s) => ({ value: s, text: s, label: s })),
                onChange(newVal) {
                    if (newVal) {
                        try { localStorage.setItem(settingId("DefaultSkillPreset"), newVal); } catch (e) {}
                    }
                }
            });

            app.ui.settings.addSetting({
                id: settingId("GlobalSkillText"),
                name: "沐阳 H3: 全局 Skill 默认规则 (Global Skill Rules)",
                type: "text",
                defaultValue: migratedSettingDefault("GlobalSkillText", ""),
                multiline: true,
                onChange(newVal) {
                    try { localStorage.setItem(settingId("GlobalSkillText"), newVal || ""); } catch (e) {}
                }
            });

            app.ui.settings.addSetting({
                id: settingId("AutoUnloadOllama"),
                name: "沐阳 H3: 默认自动卸载 Ollama 模型 (Auto Unload Ollama)",
                type: "boolean",
                defaultValue: migratedSettingDefault(
                    "AutoUnloadOllama", true, (value) => value !== "false"),
                onChange(newVal) {
                    try { localStorage.setItem(settingId("AutoUnloadOllama"), String(newVal)); } catch (e) {}
                }
            });

            app.ui.settings.addSetting({
                id: settingId("OpenSkillManager"),
                name: "沐阳 H3: Agent技能管理 (预设编辑 / 新建 / 删除)",
                type: "hidden",
                defaultValue: "",
            });

            // Add a proper interactive button setting in ComfyUI Settings
            app.ui.settings.addSetting({
                id: settingId("OpenSkillManagerButton"),
                name: "沐阳 H3: Agent技能管理",
                type: (name, setter, value) => {
                    const btn = document.createElement("button");
                    btn.textContent = "⚙️ 打开 Agent技能管理 面板";
                    btn.className = "comfy-btn";
                    btn.style.cssText = "padding: 6px 14px; background: #78dce8; color: #111; font-weight: bold; border: none; border-radius: 4px; cursor: pointer;";
                    btn.onclick = () => openSkillManagerModal();
                    return btn;
                },
                defaultValue: "",
            });

            app.ui.settings.addSetting({
                id: settingId("OpenLLMConfigButton"),
                name: "沐阳 H3: LLM 服务设置",
                type: (name, setter, value) => {
                    const btn = document.createElement("button");
                    btn.textContent = "🔧 打开 LLM 服务设置 面板";
                    btn.className = "comfy-btn";
                    btn.style.cssText = "padding: 6px 14px; background: #a9dc76; color: #111; font-weight: bold; border: none; border-radius: 4px; cursor: pointer;";
                    btn.onclick = () => openLLMConfigModal();
                    return btn;
                },
                defaultValue: "",
            });
        }
    },
    beforeRegisterNodeDef(nodeType, nodeData, appInstance) {
        installAgentNode(nodeType, nodeData);
        installSummaryViewerNode(nodeType, nodeData, appInstance);
    },
});
