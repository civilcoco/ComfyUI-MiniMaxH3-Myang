import { api } from "../../scripts/api.js";

const ENABLED_WIDGET = "粗剪时间轴开启";
const PROJECT_WIDGET = "粗剪工程";
const PROJECT_FORMAT = "myang.roughcut";
const PROJECT_VERSION = 1;
const MEDIA_ROUTE = "/minimax-h3-myang/roughcut";
const DIRECTOR_LIBRARY_ID = "__director_catalogue__";
const TRACK_HEADER_WIDTH = 52;
const MODE_OPTIONS = [
    ["off", "关闭"],
    ["frame", "单帧关键图（轻量）"],
    ["motion", "MotionContext 窗口（推荐）"],
];
const DURATION_DIRECTOR = "director";
const DURATION_TIMELINE = "timeline";

let activeEditor = null;

function nodeWidget(node, name) {
    return node?.widgets?.find((item) => item.name === name) || null;
}

function uid(prefix) {
    if (globalThis.crypto?.randomUUID) return `${prefix}_${crypto.randomUUID().replaceAll("-", "")}`;
    return `${prefix}_${Date.now().toString(36)}_${Math.random().toString(36).slice(2)}`;
}

function emptyProject() {
    return {
        format: PROJECT_FORMAT,
        version: PROJECT_VERSION,
        id: uid("roughcut"),
        revision: 0,
        settings: {
            fps: 24, width: 1920, height: 1080,
            auto_align: true, snap_enabled: true, writeback_enabled: false,
        },
        selection: {
            in_frame: 0, out_frame: 120, target_track: "video_1",
            in_set: false, out_set: false, anchor_side: "",
            duration_mode: DURATION_DIRECTOR,
            start_mode: "off", end_mode: "off",
            use_first_frame: false, use_last_frame: false,
        },
        tracks: [
            {id: "video_1", kind: "video", name: "V1", clips: []},
            {id: "audio_1", kind: "audio", name: "A1", clips: []},
        ],
    };
}

function normalizeProject(value) {
    let raw = value;
    if (typeof raw === "string") {
        try { raw = JSON.parse(raw); } catch (_error) { raw = null; }
    }
    if (!raw || raw.format !== PROJECT_FORMAT || Number(raw.version) !== PROJECT_VERSION) {
        return emptyProject();
    }
    const project = structuredClone(raw);
    project.settings ||= {fps: 24, width: 1920, height: 1080};
    project.settings.fps = Math.max(1, Math.min(240, Number(project.settings.fps) || 24));
    const dimension = (value, fallback) => Math.max(32, Math.min(8192,
        Math.round((Number(value) || fallback) / 2) * 2));
    project.settings.width = dimension(project.settings.width, 1920);
    project.settings.height = dimension(project.settings.height, 1080);
    project.settings.auto_align = project.settings.auto_align !== false;
    project.settings.snap_enabled = project.settings.snap_enabled !== false;
    // Legacy projects did not have this field. They remain edit-only until
    // the user explicitly enables write-back in the current Director card.
    project.settings.writeback_enabled = project.settings.writeback_enabled === true;
    project.selection ||= {};
    const explicitPointState = ["in_set", "out_set", "anchor_side", "duration_mode"]
        .some((key) => Object.hasOwn(project.selection, key));
    project.selection.in_frame = Math.max(0, Math.round(Number(project.selection.in_frame) || 0));
    project.selection.out_frame = Math.max(
        project.selection.in_frame + 1,
        Math.round(Number(project.selection.out_frame) || project.selection.in_frame + project.settings.fps * 5));
    project.selection.start_mode ||= project.selection.use_first_frame ? "frame" : "off";
    project.selection.end_mode ||= project.selection.use_last_frame ? "frame" : "off";
    project.selection.use_first_frame = project.selection.start_mode === "frame";
    project.selection.use_last_frame = project.selection.end_mode === "frame";
    project.tracks = Array.isArray(project.tracks) ? project.tracks : [];
    if (!project.tracks.some((track) => track.kind === "video")) {
        project.tracks.unshift({id: "video_1", kind: "video", name: "V1", clips: []});
    }
    if (!project.tracks.some((track) => track.kind === "audio")) {
        project.tracks.push({id: "audio_1", kind: "audio", name: "A1", clips: []});
    }
    for (const track of project.tracks) track.clips = Array.isArray(track.clips) ? track.clips : [];
    const legacyUntouchedDefault = !explicitPointState
        && Number(project.revision || 0) === 0
        && !project.tracks.some((track) => track.clips.length)
        && project.selection.in_frame === 0
        && project.selection.out_frame === Math.max(1, Math.round(project.settings.fps * 5));
    project.selection.in_set = explicitPointState
        ? project.selection.in_set === true : !legacyUntouchedDefault;
    project.selection.out_set = explicitPointState
        ? project.selection.out_set === true : !legacyUntouchedDefault;
    project.selection.duration_mode = [DURATION_DIRECTOR, DURATION_TIMELINE]
        .includes(project.selection.duration_mode)
        ? project.selection.duration_mode : DURATION_DIRECTOR;
    project.selection.anchor_side = ["in", "out"].includes(project.selection.anchor_side)
        ? project.selection.anchor_side : "";
    if (project.selection.duration_mode === DURATION_DIRECTOR
        && !project.selection.anchor_side) {
        if (project.selection.in_set && !project.selection.out_set) project.selection.anchor_side = "in";
        else if (project.selection.out_set && !project.selection.in_set) project.selection.anchor_side = "out";
    }
    if (!project.tracks.some((track) => track.id === project.selection.target_track)) {
        project.selection.target_track = project.tracks.find((track) => track.kind === "video").id;
    }
    return project;
}

export function readRoughCutProject(node) {
    return normalizeProject(nodeWidget(node, PROJECT_WIDGET)?.value || "");
}

function saveProject(node, project) {
    const normalized = normalizeProject(project);
    const control = nodeWidget(node, PROJECT_WIDGET);
    if (control) control.value = JSON.stringify(normalized);
    node.graph?.setDirtyCanvas?.(true, true);
    updateCard(node, normalized);
    if (activeEditor?.node === node) {
        activeEditor.project = normalized;
        renderEditorTimeline(activeEditor);
    }
}

function setEnabled(node, enabled) {
    const active = enabled === true;
    const project = readRoughCutProject(node);
    project.settings.writeback_enabled = active;
    const control = nodeWidget(node, ENABLED_WIDGET);
    if (control) control.value = active;
    saveProject(node, project);
}

function selectionText(project) {
    const fps = Number(project.settings.fps) || 24;
    const start = Number(project.selection.in_frame) || 0;
    const end = Number(project.selection.out_frame) || start + 1;
    const mode = String(project.selection.duration_mode || DURATION_DIRECTOR);
    const inSet = project.selection.in_set === true;
    const outSet = project.selection.out_set === true;
    if (!inSet && !outSet) return "尚未设置 I/O · 将播放头移到目标位置后按 I 或 O";
    if (mode === DURATION_TIMELINE && !(inSet && outSet)) {
        return inSet
            ? `I ${timecode(start, fps)} 已设置 · 等待设置 O`
            : `O ${timecode(end, fps)} 已设置 · 等待设置 I`;
    }
    const automatic = mode === DURATION_DIRECTOR
        ? (project.selection.anchor_side === "in" ? " · O 自动"
            : project.selection.anchor_side === "out" ? " · I 自动" : "")
        : "";
    return `I ${timecode(start, fps)}  →  O ${timecode(end, fps)}  ·  ${((end - start) / fps).toFixed(2)} 秒${automatic}`;
}

function timecode(frame, fps) {
    const safeFps = Math.max(1, Number(fps) || 24);
    const total = Math.max(0, Math.round(frame));
    const frames = total % Math.round(safeFps);
    const secondsTotal = Math.floor(total / safeFps);
    const seconds = secondsTotal % 60;
    const minutesTotal = Math.floor(secondsTotal / 60);
    const minutes = minutesTotal % 60;
    const hours = Math.floor(minutesTotal / 60);
    return [hours, minutes, seconds, frames].map((value) => String(value).padStart(2, "0")).join(":");
}

function updateCard(node, supplied = null) {
    const els = node.__myangRoughCutCardEls;
    if (!els) return;
    const project = supplied || readRoughCutProject(node);
    const enabled = nodeWidget(node, ENABLED_WIDGET)?.value === true
        && project.settings.writeback_enabled === true;
    els.toggle.checked = enabled;
    els.summary.textContent = selectionText(project);
    els.state.textContent = enabled ? "运行后自动覆盖写入" : "仅编辑，不影响当前生成";
    els.state.dataset.enabled = enabled ? "true" : "false";
}

export function renderRoughCutCard(node, options = {}) {
    node.__myangRoughCutDirectorSeconds = Math.max(
        1 / 24, Number(options.directorDurationSeconds) || 5);
    node.__myangRoughCutApplyDuration = typeof options.onDurationSeconds === "function"
        ? options.onDurationSeconds : null;
    const card = document.createElement("section");
    card.className = "myh3-roughcut-card";
    const top = document.createElement("div");
    top.className = "myh3-roughcut-card-top";
    const identity = document.createElement("div");
    identity.className = "myh3-roughcut-identity";
    const toggle = document.createElement("input");
    toggle.type = "checkbox";
    toggle.className = "myh3-roughcut-toggle";
    toggle.setAttribute("aria-label", "开启粗剪时间轴写回");
    toggle.title = "开启后，本次导演台成片自动裁到 I/O 区间并覆盖写入时间轴";
    toggle.onchange = (event) => {
        event.stopPropagation();
        setEnabled(node, toggle.checked);
    };
    const titleWrap = document.createElement("div");
    titleWrap.className = "myh3-roughcut-title-wrap";
    const title = document.createElement("div");
    title.textContent = "粗剪时间轴";
    title.className = "myh3-roughcut-title";
    const summary = document.createElement("div");
    summary.className = "myh3-roughcut-summary";
    titleWrap.append(title, summary);
    identity.append(toggle, titleWrap);
    const actions = document.createElement("div");
    actions.className = "myh3-roughcut-actions";
    const state = document.createElement("span");
    state.className = "myh3-roughcut-state";
    const open = document.createElement("button");
    open.type = "button";
    open.textContent = "打开时间轴";
    open.className = "myh3-roughcut-open";
    open.onclick = (event) => {
        event.preventDefault();
        event.stopPropagation();
        openRoughCutEditor(node);
    };
    actions.append(state, open);
    top.append(identity, actions);
    const hint = document.createElement("div");
    hint.className = "myh3-roughcut-hint";
    hint.textContent = "素材与工程仅保存在本机 · I 点可接入 MotionContext · O 点可桥接后续镜头";
    card.append(top, hint);
    for (const eventName of ["pointerdown", "mousedown", "click", "dblclick", "keydown", "keyup"]) {
        card.addEventListener(eventName, (event) => event.stopPropagation());
    }
    node.__myangRoughCutCardEls = {toggle, summary, state};
    updateCard(node);
    return card;
}

async function apiJson(path, options = undefined) {
    const response = await api.fetchApi(path, options);
    const text = await response.text();
    if (!response.ok) throw new Error(text || `HTTP ${response.status}`);
    return text ? JSON.parse(text) : {};
}

function exportStamp() {
    const now = new Date();
    const pad = (value) => String(value).padStart(2, "0");
    return `${now.getFullYear()}${pad(now.getMonth() + 1)}${pad(now.getDate())}`
        + `_${pad(now.getHours())}${pad(now.getMinutes())}${pad(now.getSeconds())}`;
}

function downloadProjectJson(project) {
    const payload = JSON.stringify(normalizeProject(project), null, 2);
    const url = URL.createObjectURL(new Blob([payload], {type: "application/json;charset=utf-8"}));
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `Myang_粗剪工程_${exportStamp()}.json`;
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    setTimeout(() => URL.revokeObjectURL(url), 0);
}

function parseProjectJson(text) {
    let raw;
    try { raw = JSON.parse(String(text || "")); }
    catch (error) { throw new Error(`不是有效 JSON：${error.message}`); }
    if (!raw || raw.format !== PROJECT_FORMAT || Number(raw.version) !== PROJECT_VERSION) {
        throw new Error("不是受支持的沐阳粗剪工程文件");
    }
    return normalizeProject(raw);
}

function mediaUrl(asset) {
    const params = new URLSearchParams({library_id: asset.library_id, path: asset.relative_path});
    return `${MEDIA_ROUTE}/media?${params}`;
}

function outputUrl(source) {
    const params = new URLSearchParams({
        filename: source.filename, subfolder: source.subfolder || "", type: "output",
    });
    return `/view?${params}`;
}

function inputUrl(source) {
    const params = new URLSearchParams({
        filename: source.filename, subfolder: source.subfolder || "", type: "input",
    });
    return `/view?${params}`;
}

function assetUrl(asset) {
    return asset.source?.type === "input" ? inputUrl(asset.source) : mediaUrl(asset);
}

function clipUrl(clip) {
    if (clip.source?.type === "input") return inputUrl(clip.source);
    return clip.source?.type === "library" ? mediaUrl({
        library_id: clip.source.library_id, relative_path: clip.source.relative_path,
    }) : outputUrl(clip.source || {});
}

function applyOverwrite(track, incoming, projectRate = 24) {
    const start = incoming.timeline_start;
    const end = incoming.timeline_end;
    const kept = [];
    for (const clip of track.clips) {
        if (clip.id === incoming.id) continue;
        if (clip.timeline_end <= start || clip.timeline_start >= end) {
            kept.push(clip);
            continue;
        }
        const rate = Number(clip.source_fps) || 24;
        const safeProjectRate = Number(projectRate) || 24;
        if (clip.timeline_start < start) {
            kept.push({...clip, timeline_end: start,
                source_out: Math.min(clip.source_out,
                    clip.source_in + Math.round((start - clip.timeline_start) * rate / safeProjectRate))});
        }
        if (clip.timeline_end > end) {
            kept.push({...clip, id: uid("clip"), timeline_start: end,
                source_in: Math.max(clip.source_in,
                    clip.source_out - Math.round((clip.timeline_end - end) * rate / safeProjectRate))});
        }
    }
    kept.push(incoming);
    track.clips = kept.sort((a, b) => a.timeline_start - b.timeline_start);
}

function editorStyle() {
    if (document.getElementById("myh3-roughcut-style")) return;
    const style = document.createElement("style");
    style.id = "myh3-roughcut-style";
    style.textContent = `
      .myh3-rc-modal{position:fixed;inset:18px;z-index:10050;display:grid;grid-template-rows:auto 1fr;background:#0c1119;color:#dbe7f3;border:1px solid #3b5168;border-radius:10px;box-shadow:0 24px 80px #000b;font:12px system-ui;overflow:hidden}
      .myh3-rc-head{display:flex;align-items:center;gap:10px;padding:10px 12px;background:#111b28;border-bottom:1px solid #2e4052}.myh3-rc-head h2{font-size:15px;margin:0;flex:1}.myh3-rc-head button,.myh3-rc-btn{border:1px solid #3d5872;background:#172638;color:#dbeafe;border-radius:5px;padding:6px 10px;cursor:pointer}.myh3-rc-btn:hover{background:#203a55}
      .myh3-rc-grid{min-height:0;display:grid;grid-template-columns:240px minmax(320px,1fr) minmax(360px,.9fr);grid-template-rows:minmax(250px,44%) 1fr}.myh3-rc-library{grid-column:1;grid-row:1/3;border-right:1px solid #29394a;padding:10px;overflow:auto;background:#0e151f}.myh3-rc-assets{grid-column:2;grid-row:1;padding:10px;overflow:auto;border-bottom:1px solid #29394a}.myh3-rc-program{grid-column:3;grid-row:1;min-width:0;min-height:0;display:grid;grid-template-rows:auto minmax(0,1fr) auto;gap:7px;padding:10px;background:#080d13;border-left:1px solid #29394a;border-bottom:1px solid #29394a}.myh3-rc-program-head{display:flex;align-items:center;gap:8px;min-width:0}.myh3-rc-program-head b{white-space:nowrap}.myh3-rc-program-label{min-width:0;flex:1;color:#8fa4b8;font-size:10px;text-align:right;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.myh3-rc-timeline{grid-column:2/4;grid-row:2;min-height:0;padding:10px;display:flex;flex-direction:column;overflow:hidden}
      .myh3-rc-input,.myh3-rc-select{box-sizing:border-box;background:#080d13;color:#e5edf6;border:1px solid #34495e;border-radius:4px;padding:6px}.myh3-rc-row{display:flex;gap:6px;align-items:center}.myh3-rc-lib{padding:8px;margin:5px 0;border:1px solid #2b3d4e;border-radius:6px;cursor:pointer}.myh3-rc-lib:hover{background:#111f2d}.myh3-rc-lib.active{border-color:#60a5fa;background:#142a3d}.myh3-rc-lib.catalogue{border-color:#4a4d80;background:#16172c}.myh3-rc-lib.catalogue.active{border-color:#a78bfa;background:#242044}.myh3-rc-asset-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(130px,1fr));gap:7px}.myh3-rc-asset{border:1px solid #2f4356;background:#111a25;border-radius:6px;padding:7px;cursor:grab;min-width:0}.myh3-rc-asset:hover{border-color:#60a5fa;background:#152538}.myh3-rc-asset:focus-visible{outline:2px solid #60a5fa;outline-offset:2px}.myh3-rc-asset-thumb{display:block;width:100%;height:72px;margin-bottom:6px;border-radius:4px;object-fit:contain;background:#070b10}.myh3-rc-asset b{display:block;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.myh3-rc-kind{font-size:10px;color:#7dd3fc;margin-top:4px}.myh3-rc-preview{position:relative;display:grid;place-items:center;box-sizing:border-box;width:100%;height:100%;min-width:0;min-height:0;overflow:hidden;background:#020305;border:1px solid #26384a}.myh3-rc-preview-empty{padding:16px;color:#607286;text-align:center}.myh3-rc-preview-media{display:block;box-sizing:border-box;width:auto;height:auto;max-width:100%;max-height:100%;object-fit:contain;background:#020305}.myh3-rc-preview-audio{position:absolute;left:8px;right:8px;bottom:8px;padding:5px 7px;border:1px solid #55416c;background:#160f22e8;color:#c4b5fd;font-size:9px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.myh3-rc-preview-download{position:absolute;right:8px;top:8px;padding:5px 8px;border:1px solid #365b7d;background:#0e2235e8;color:#93c5fd;text-decoration:none}
      .myh3-rc-controls{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:8px}.myh3-rc-controls label{display:flex;gap:5px;align-items:center;color:#9fb0c2}.myh3-rc-controls input[type=number]{width:78px}.myh3-rc-transport{display:flex;align-items:center;justify-content:center;gap:4px;min-width:0}.myh3-rc-transport button{display:grid;place-items:center;width:38px;height:38px;border:1px solid #31485e;border-radius:4px;background:#172638;color:#e8f2fa;font-size:15px;cursor:pointer}.myh3-rc-transport button:hover{background:#26435e}.myh3-rc-transport button.active{background:#256b54;color:#fff}.myh3-rc-transport-time{min-width:86px;padding:0 7px;color:#fde68a;font:10px ui-monospace,SFMono-Regular,Consolas,monospace;white-space:nowrap}.myh3-rc-mode-group{display:inline-flex;gap:2px;padding:3px;background:#080d13;border:1px solid #30465a;border-radius:7px}.myh3-rc-mode{min-height:44px;border:0;background:transparent;color:#8fa4b8;border-radius:4px;padding:0 11px;white-space:nowrap;cursor:pointer}.myh3-rc-mode.active{background:#245079;color:#eff8ff}.myh3-rc-mode:focus-visible,.myh3-rc-point:focus-visible,.myh3-rc-switch:focus-within,.myh3-rc-transport button:focus-visible,.myh3-rc-delete:focus-visible{outline:2px solid #60a5fa;outline-offset:2px}.myh3-rc-point{min-height:44px;min-width:108px;border:1px solid #3f637e;background:#14283a;color:#e4f2ff;border-radius:6px;padding:0 12px;white-space:nowrap;cursor:pointer}.myh3-rc-point.i{border-color:#258b70}.myh3-rc-point.o{border-color:#a84e60}.myh3-rc-delete{min-height:44px;border:1px solid #8f4150;background:#351923;color:#fecdd3;border-radius:6px;padding:0 12px;white-space:nowrap;cursor:pointer}.myh3-rc-delete:hover:not(:disabled){background:#54202d}.myh3-rc-delete:disabled{opacity:.4;cursor:not-allowed}.myh3-rc-switch{min-height:44px;padding:0 8px;border:1px solid #2f4558;border-radius:5px;background:#101a25;white-space:nowrap}.myh3-rc-hintline{flex-basis:100%;color:#7f93a8;font-size:10px}.myh3-rc-scroll{position:relative;overflow:auto;min-height:170px;flex:1;border:1px solid #2b3e52;background:#080d13}.myh3-rc-canvas{position:relative;min-width:100%;min-height:100%;}.myh3-rc-ruler{position:relative;height:32px;border-bottom:1px solid #33475b;background:#101924;cursor:ew-resize;touch-action:none;user-select:none}.myh3-rc-ruler.dragging{cursor:grabbing}.myh3-rc-ruler-head{position:sticky;left:0;z-index:9;width:${TRACK_HEADER_WIDTH}px;height:100%;box-sizing:border-box;background:#13202d;border-right:1px solid #34495e}.myh3-rc-track{position:relative;height:70px;border-bottom:1px solid #253646}.myh3-rc-track-label{position:sticky;left:0;z-index:8;display:flex;align-items:center;justify-content:center;width:${TRACK_HEADER_WIDTH}px;height:100%;box-sizing:border-box;background:#13202d;color:#93c5fd;border-right:1px solid #34495e}.myh3-rc-clip{position:absolute;top:7px;height:54px;border:1px solid #4b7ba3;background:#173a56;border-radius:4px;padding:5px;box-sizing:border-box;overflow:hidden;cursor:grab}.myh3-rc-clip:hover{filter:brightness(1.12)}.myh3-rc-clip.selected{border-color:#facc15;box-shadow:inset 0 0 0 1px #facc15,0 0 0 1px #05070a}.myh3-rc-clip.audio{background:#3b2456;border-color:#8b5fb5}.myh3-rc-clip.audio.selected,.myh3-rc-clip.generated.selected{border-color:#facc15}.myh3-rc-clip.generated{background:#234d37;border-color:#58a878}.myh3-rc-clip b{display:block;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.myh3-rc-clip small{color:#b1c5d8}.myh3-rc-range{position:absolute;top:0;z-index:3;background:#38bdf81a;border-left:1px solid #38bdf855;border-right:1px solid #38bdf855;pointer-events:none}.myh3-rc-marker{position:absolute;top:0;bottom:0;width:2px;z-index:6;pointer-events:none}.myh3-rc-marker.i{background:#36d399;color:#36d399}.myh3-rc-marker.o{background:#fb7185;color:#fb7185}.myh3-rc-marker.auto{opacity:.55;background-image:repeating-linear-gradient(to bottom,currentColor 0 5px,transparent 5px 9px)}.myh3-rc-marker-label{position:absolute;top:1px;left:-7px;display:grid;place-items:center;width:16px;height:14px;border-radius:3px 3px 3px 0;background:currentColor;color:#071018;font-size:9px;font-weight:800;box-shadow:0 1px 4px #0008}.myh3-rc-playhead{position:absolute;left:${TRACK_HEADER_WIDTH}px;top:0;bottom:0;width:2px;background:#facc15;z-index:7;pointer-events:none;will-change:transform;filter:drop-shadow(0 0 2px #facc1566)}.myh3-rc-playhead-handle{position:absolute;top:0;left:-7px;width:16px;height:14px;border-radius:3px 3px 3px 0;background:#facc15;box-shadow:0 1px 5px #000a}.myh3-rc-playhead-handle::after{content:"";position:absolute;left:0;bottom:-5px;border-top:6px solid #facc15;border-right:6px solid transparent}.myh3-rc-playhead-time{position:absolute;top:1px;left:12px;padding:2px 5px;border:1px solid #8b7412;border-radius:4px;background:#171604ef;color:#fde68a;font:10px ui-monospace,SFMono-Regular,Consolas,monospace;white-space:nowrap}.myh3-rc-tick{position:absolute;top:0;height:100%;border-left:1px solid #334155;color:#7f91a4;padding:3px;font-size:9px;box-sizing:border-box}.myh3-rc-status{color:#fda4af;font-size:10px;min-height:15px;margin-top:7px}.myh3-rc-backdrop{position:fixed;inset:0;background:#0009;z-index:10049}
      @media(max-width:1100px){.myh3-rc-grid{grid-template-columns:220px 1fr;grid-template-rows:minmax(210px,31%) minmax(210px,31%) 1fr}.myh3-rc-library{grid-column:1;grid-row:1/4}.myh3-rc-assets{grid-column:2;grid-row:1}.myh3-rc-program{grid-column:2;grid-row:2;border-left:0}.myh3-rc-timeline{grid-column:2;grid-row:3}}
      .myh3-rc-clip{touch-action:none;user-select:none}.myh3-rc-clip.dragging{cursor:grabbing;opacity:.78;z-index:10}.myh3-rc-snap-guide{position:absolute;top:0;bottom:0;width:2px;z-index:8;background:#38bdf8;box-shadow:0 0 0 1px #07131c,0 0 10px #38bdf8;pointer-events:none}.myh3-rc-snap-guide-label{position:absolute;top:15px;left:5px;padding:3px 5px;border:1px solid #277696;border-radius:3px;background:#062638e8;color:#a5f3fc;font:9px ui-monospace,SFMono-Regular,Consolas,monospace;white-space:nowrap}
      @media(prefers-reduced-motion:reduce){.myh3-rc-btn,.myh3-rc-transport button,.myh3-rc-asset,.myh3-rc-clip{transition:none}}
      .myh3-rc-preview-media{width:100%;height:100%;max-width:100%;max-height:100%;object-fit:contain}.myh3-rc-preview audio.myh3-rc-preview-media{height:32px;max-height:none}
    `;
    document.head.appendChild(style);
}

async function loadLibraries(editor) {
    editor.libraryList.replaceChildren();
    const loading = document.createElement("div");
    loading.textContent = "正在读取导演台素材库与素材文件夹…";
    loading.style.cssText = "padding:9px;color:#8194a8;";
    editor.libraryList.appendChild(loading);
    const [folderResult, catalogueResult] = await Promise.allSettled([
        apiJson(`${MEDIA_ROUTE}/libraries`),
        apiJson("/minimax-h3-myang/assets"),
    ]);
    const folders = folderResult.status === "fulfilled"
        ? (folderResult.value.libraries || []) : [];
    editor.catalogueAssets = catalogueResult.status === "fulfilled"
        ? (catalogueResult.value.assets || []) : [];
    editor.libraries = [{
        id: DIRECTOR_LIBRARY_ID,
        name: "★ 导演台素材库",
        path: `已收藏 ${editor.catalogueAssets.length} 个素材`,
        catalogue: true,
    }, ...folders];
    if (!editor.libraryId || !editor.libraries.some((item) => item.id === editor.libraryId)) {
        editor.libraryId = DIRECTOR_LIBRARY_ID;
    }
    const failures = [folderResult, catalogueResult]
        .filter((result) => result.status === "rejected")
        .map((result) => result.reason?.message || String(result.reason));
    if (failures.length) setStatus(editor, `部分素材来源读取失败：${failures.join("；")}`);
    await loadAssets(editor);
}

async function loadAssets(editor) {
    editor.assets = [];
    if (editor.libraryId === DIRECTOR_LIBRARY_ID) {
        editor.assets = (editor.catalogueAssets || []).map((item) => ({
            ...item,
            name: item.name || item.file?.name || "未命名素材",
            kind: item.kind || "image",
            source: {
                type: "input",
                filename: item.file?.name || "",
                subfolder: item.file?.subfolder || "",
            },
        })).filter((item) => item.source.filename);
    } else if (editor.libraryId) {
        try {
            const payload = await apiJson(`${MEDIA_ROUTE}/assets?library_id=${encodeURIComponent(editor.libraryId)}`);
            editor.assets = payload.assets || [];
        } catch (error) { setStatus(editor, error.message); }
    }
    renderLibraries(editor);
    renderAssets(editor);
}

function setStatus(editor, message, good = false) {
    editor.status.textContent = String(message || "");
    editor.status.style.color = good ? "#86efac" : "#fda4af";
}

function renderLibraries(editor) {
    editor.libraryList.replaceChildren();
    for (const library of editor.libraries) {
        const row = document.createElement("div");
        row.className = `myh3-rc-lib${library.catalogue ? " catalogue" : ""}${library.id === editor.libraryId ? " active" : ""}`;
        const name = document.createElement("b");
        name.textContent = library.name;
        const path = document.createElement("div");
        path.textContent = library.path;
        path.title = library.path;
        path.style.cssText = "font-size:9px;color:#74879a;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;";
        const remove = document.createElement("button");
        if (!library.catalogue) {
            remove.type = "button";
            remove.textContent = "移除";
            remove.style.cssText = "float:right;border:0;background:transparent;color:#fb7185;cursor:pointer;";
            remove.onclick = async (event) => {
                event.stopPropagation();
                if (!confirm(`从百宝箱移除「${library.name}」？不会删除磁盘文件。`)) return;
                await apiJson(`${MEDIA_ROUTE}/libraries/${encodeURIComponent(library.id)}`, {method: "DELETE"});
                await loadLibraries(editor);
            };
        }
        row.onclick = async () => { editor.libraryId = library.id; await loadAssets(editor); };
        if (!library.catalogue) name.prepend(remove);
        row.append(name, path);
        editor.libraryList.appendChild(row);
    }
}

function renderAssets(editor) {
    editor.assetGrid.replaceChildren();
    const query = editor.search.value.trim().toLowerCase();
    const assets = editor.assets.filter((item) => !query || item.name.toLowerCase().includes(query));
    for (const asset of assets) {
        const item = document.createElement("div");
        item.className = "myh3-rc-asset";
        item.tabIndex = 0;
        item.setAttribute("role", "button");
        item.draggable = true;
        item.title = asset.relative_path || asset.source?.filename || asset.name;
        if (asset.kind === "image") {
            const thumb = document.createElement("img");
            thumb.className = "myh3-rc-asset-thumb";
            thumb.src = assetUrl(asset);
            thumb.alt = "";
            thumb.loading = "lazy";
            item.appendChild(thumb);
        }
        const name = document.createElement("b");
        name.textContent = asset.name;
        const kind = document.createElement("div");
        kind.className = "myh3-rc-kind";
        kind.textContent = asset.kind === "video" ? "视频" : asset.kind === "audio" ? "音频" : "图片";
        item.append(name, kind);
        item.ondragstart = (event) => {
            event.dataTransfer.effectAllowed = "copy";
            event.dataTransfer.dropEffect = "copy";
            event.dataTransfer.setData("application/x-myang-asset", JSON.stringify(asset));
        };
        item.ondragend = () => {
            if (activeEditor) hideSnapGuide(activeEditor);
        };
        item.ondblclick = () => previewAsset(editor, asset);
        item.onkeydown = (event) => {
            if (event.key !== "Enter") return;
            event.preventDefault();
            previewAsset(editor, asset);
        };
        editor.assetGrid.appendChild(item);
    }
    if (!assets.length) {
        const empty = document.createElement("div");
        empty.textContent = editor.libraryId === DIRECTOR_LIBRARY_ID
            ? "导演台素材库暂无收藏；可在导演台素材卡中加入素材库"
            : editor.libraryId ? "这个文件夹里没有支持的素材" : "先在左侧添加一个素材文件夹";
        empty.style.color = "#718096";
        editor.assetGrid.appendChild(empty);
    }
}

function previewAsset(editor, asset) {
    stopTimelinePlayback(editor);
    editor.playbackMedia = null;
    editor.preview.replaceChildren();
    let media;
    if (asset.kind === "image") media = document.createElement("img");
    else if (asset.kind === "audio") media = document.createElement("audio");
    else media = document.createElement("video");
    media.src = assetUrl(asset);
    media.controls = asset.kind !== "image";
    media.playsInline = true;
    media.className = "myh3-rc-preview-media";
    if (asset.kind === "audio") media.style.cssText = "width:min(88%,560px);height:auto;";
    if (editor.programLabel) editor.programLabel.textContent = `素材预览 · ${asset.name}`;
    editor.preview.appendChild(media);
}

function previewExport(editor, payload) {
    const video = payload?.video;
    if (!video?.filename) return;
    stopTimelinePlayback(editor);
    editor.playbackMedia = null;
    editor.preview.replaceChildren();
    const player = document.createElement("video");
    player.src = outputUrl(video);
    player.controls = true;
    player.playsInline = true;
    player.className = "myh3-rc-preview-media";
    const download = document.createElement("a");
    download.href = outputUrl(video);
    download.download = video.filename;
    download.textContent = "下载 MP4";
    download.className = "myh3-rc-preview-download";
    if (editor.programLabel) editor.programLabel.textContent = `导出成片 · ${video.filename}`;
    editor.preview.append(player, download);
}

function maxTimelineFrame(project) {
    const clipEnd = Math.max(0, ...project.tracks.flatMap((track) => track.clips.map((clip) => Number(clip.timeline_end) || 0)));
    const selectionEnd = selectionReady(project) ? Number(project.selection.out_frame) || 0 : 0;
    return Math.max(clipEnd, selectionEnd + project.settings.fps * 5,
        project.settings.fps * 30);
}

function xToFrame(editor, event, target) {
    const rect = target.getBoundingClientRect();
    // A row/ruler rect already moves left with the scrolled canvas. Adding
    // scrollLeft again double-counts the horizontal offset.
    const x = event.clientX - rect.left - TRACK_HEADER_WIDTH;
    return Math.max(0, Math.round(x / editor.pixelsPerFrame));
}

function durationMode(project) {
    return project.selection.duration_mode === DURATION_TIMELINE
        ? DURATION_TIMELINE : DURATION_DIRECTOR;
}

function directorDurationFrames(editor) {
    const fps = Math.max(1, Number(editor.project.settings.fps) || 24);
    return Math.max(1, Math.round(
        Math.max(1 / fps, Number(editor.node.__myangRoughCutDirectorSeconds) || 5) * fps));
}

function selectionReady(project) {
    const selection = project.selection;
    return durationMode(project) === DURATION_TIMELINE
        ? selection.in_set === true && selection.out_set === true
        : selection.in_set === true || selection.out_set === true;
}

function snapThresholdFrames(editor, pixels = 10) {
    return Math.max(1, Math.round(pixels / Math.max(0.0001, editor.pixelsPerFrame)));
}

function timelineEdges(editor, excludeClipId = "") {
    const points = [0];
    for (const track of editor.project.tracks) {
        for (const clip of track.clips) {
            if (clip.id === excludeClipId) continue;
            points.push(Number(clip.timeline_start) || 0, Number(clip.timeline_end) || 0);
        }
    }
    const selection = editor.project.selection;
    if (selection.in_set) points.push(Number(selection.in_frame) || 0);
    if (selection.out_set) points.push(Number(selection.out_frame) || 0);
    return [...new Set(points.map((value) => Math.max(0, Math.round(value))))];
}

function nearestFrame(rawFrame, candidates, threshold) {
    let best = Math.max(0, Math.round(rawFrame));
    let distance = threshold + 1;
    for (const candidate of candidates) {
        const delta = Math.abs(candidate - rawFrame);
        if (delta <= threshold && delta < distance) {
            best = candidate;
            distance = delta;
        }
    }
    return best;
}

function snapPlayheadFrame(editor, frame) {
    if (editor.project.settings.snap_enabled === false) return Math.max(0, Math.round(frame));
    return nearestFrame(frame, timelineEdges(editor), snapThresholdFrames(editor));
}

function alignClipStart(editor, frame, duration, excludeClipId = "") {
    const raw = Math.max(0, Math.round(frame));
    if (editor.project.settings.auto_align === false
        || editor.project.settings.snap_enabled === false) return raw;
    const threshold = snapThresholdFrames(editor, 12);
    const edges = timelineEdges(editor, excludeClipId);
    const startAligned = nearestFrame(raw, edges, threshold);
    const endAligned = nearestFrame(raw + duration, edges, threshold) - duration;
    return Math.max(0,
        Math.abs(startAligned - raw) <= Math.abs(endAligned - raw) ? startAligned : endAligned);
}

function syncTimelineDuration(editor) {
    if (durationMode(editor.project) !== DURATION_TIMELINE || !selectionReady(editor.project)) return;
    const fps = Math.max(1, Number(editor.project.settings.fps) || 24);
    const selection = editor.project.selection;
    const seconds = Math.max(1 / fps, (selection.out_frame - selection.in_frame) / fps);
    editor.node.__myangRoughCutDirectorSeconds = seconds;
    editor.node.__myangRoughCutApplyDuration?.(seconds);
}

function setSelectionPoint(editor, side) {
    const selection = editor.project.selection;
    const frame = snapPlayheadFrame(editor, editor.playhead);
    let shouldSyncDuration = false;
    editor.playhead = frame;
    if (durationMode(editor.project) === DURATION_DIRECTOR) {
        const duration = directorDurationFrames(editor);
        if (side === "in") {
            selection.in_frame = frame;
            selection.out_frame = frame + duration;
            selection.in_set = true;
            selection.out_set = false;
            selection.anchor_side = "in";
        } else {
            if (frame < duration) {
                setStatus(editor, `当前位置早于导演台时长（${(duration / editor.project.settings.fps).toFixed(2)} 秒），无法从 O 点向前匹配`);
                return;
            }
            selection.in_frame = frame - duration;
            selection.out_frame = frame;
            selection.in_set = false;
            selection.out_set = true;
            selection.anchor_side = "out";
        }
        setStatus(editor, `${side === "in" ? "入点 I" : "出点 O"} 已设置，另一端已按导演台时长自动匹配`, true);
    } else {
        selection.anchor_side = side;
        if (side === "in") {
            selection.in_frame = frame;
            selection.in_set = true;
            if (selection.out_set && selection.out_frame <= frame) {
                selection.out_set = false;
                setStatus(editor, "入点 I 已设置；原 O 点不在其后，请重新设置 O");
            }
        } else {
            if (selection.in_set && frame <= selection.in_frame) {
                setStatus(editor, "出点 O 必须位于入点 I 之后");
                return;
            }
            selection.out_frame = Math.max(selection.in_frame + 1, frame);
            selection.out_set = true;
        }
        if (selection.in_set && selection.out_set) {
            shouldSyncDuration = true;
            setStatus(editor, "I/O 范围已设置，导演台时长已匹配该范围", true);
        } else {
            setStatus(editor, `${side === "in" ? "I" : "O"} 已设置，请再设置另一端`);
        }
    }
    saveEditor(editor);
    if (shouldSyncDuration) syncTimelineDuration(editor);
}

function boundaryAvailability(editor, side) {
    const project = editor.project;
    const count = Math.max(1, Number(nodeWidget(editor.node, "context_length")?.value) || 22);
    const point = Number(side === "first" ? project.selection.in_frame : project.selection.out_frame);
    const track = project.tracks.find((item) =>
        item.id === project.selection.target_track && item.kind === "video");
    let available = 0;
    for (const clip of track?.clips || []) {
        if (!['video', 'generated'].includes(clip.kind)) continue;
        if (side === "first" && clip.timeline_start < point && clip.timeline_end >= point) {
            available = Math.max(available, point - clip.timeline_start);
        } else if (side === "last" && clip.timeline_start <= point && clip.timeline_end > point) {
            available = Math.max(available, clip.timeline_end - point);
        }
    }
    return {available, count};
}

function updateBoundaryStatus(editor) {
    const checks = [];
    if (editor.project.selection.start_mode === "motion") checks.push(["入点前", "first"]);
    if (editor.project.selection.end_mode === "motion") checks.push(["出点后", "last"]);
    if (!checks.length) return;
    const missing = [];
    for (const [label, side] of checks) {
        const {available, count} = boundaryAvailability(editor, side);
        if (available < count) missing.push(`${label}只有 ${available}/${count} 帧连续视频`);
    }
    setStatus(editor, missing.length
        ? `${missing.join("；")}，请移动 I/O、补素材或改用单帧约束`
        : "首尾 MotionContext 素材窗口可用", !missing.length);
}

async function assetClip(editor, asset, startFrame) {
    const isInput = asset.source?.type === "input";
    const params = isInput ? new URLSearchParams({
        source_type: "input",
        filename: asset.source.filename,
        subfolder: asset.source.subfolder || "",
    }) : new URLSearchParams({library_id: asset.library_id, path: asset.relative_path});
    const probe = await apiJson(`${MEDIA_ROUTE}/probe?${params}`);
    const fps = Number(editor.project.settings.fps) || 24;
    const duration = asset.kind === "image" ? 5 : Math.max(1 / fps, Number(probe.duration) || 5);
    const timelineFrames = Math.max(1, Math.round(duration * fps));
    const sourceFps = Number(probe.fps) || fps;
    const sourceFrames = asset.kind === "image" ? timelineFrames : Math.max(1, Math.round(duration * sourceFps));
    const hasVisualClip = editor.project.tracks.some((track) =>
        track.kind === "video" && track.clips.length > 0);
    if (asset.kind !== "audio" && !hasVisualClip
            && Number(probe.width) >= 32 && Number(probe.height) >= 32) {
        // The first visual asset establishes the rough-cut canvas.  This is
        // especially important for portrait projects; otherwise the default
        // 1920x1080 canvas would silently letterbox every vertical export.
        editor.project.settings.width = Math.round(Number(probe.width));
        editor.project.settings.height = Math.round(Number(probe.height));
    }
    return {
        id: uid("clip"), kind: asset.kind, name: asset.name,
        timeline_start: startFrame, timeline_end: startFrame + timelineFrames,
        source_in: 0, source_out: sourceFrames, source_fps: sourceFps,
        source: isInput ? structuredClone(asset.source) : {
            type: "library", library_id: asset.library_id, relative_path: asset.relative_path,
        },
    };
}

function timelineClipAt(editor, trackKind, frame) {
    const track = editor.project.tracks.find((entry) => entry.kind === trackKind);
    return [...(track?.clips || [])].reverse().find((clip) =>
        frame >= Number(clip.timeline_start) && frame < Number(clip.timeline_end)) || null;
}

function selectedTimelineClip(editor) {
    const selection = editor.selectedClip;
    if (!selection) return null;
    const track = editor.project.tracks.find((entry) => entry.id === selection.trackId);
    const clip = track?.clips.find((entry) => entry.id === selection.clipId);
    return track && clip ? {track, clip} : null;
}

function updateDeleteControl(editor) {
    if (!editor.deleteClip) return;
    const selected = selectedTimelineClip(editor);
    editor.deleteClip.disabled = !selected;
    editor.deleteClip.title = selected
        ? `从时间轴删除「${selected.clip.name}」（Delete / Backspace）`
        : "先在 V1 或 A1 轨道选择一个片段";
}

function selectTimelineClip(editor, track, clip, element = null) {
    editor.selectedClip = {trackId: track.id, clipId: clip.id};
    for (const item of editor.canvas?.querySelectorAll(".myh3-rc-clip") || []) {
        item.classList.toggle("selected",
            item.dataset.trackId === track.id && item.dataset.clipId === clip.id);
    }
    element?.classList.add("selected");
    updateDeleteControl(editor);
}

function clearTimelineSelection(editor) {
    editor.selectedClip = null;
    for (const item of editor.canvas?.querySelectorAll(".myh3-rc-clip.selected") || []) {
        item.classList.remove("selected");
    }
    updateDeleteControl(editor);
}

function showSnapGuide(editor, frame, label = "吸附落点") {
    if (!editor.canvas) return;
    if (!editor.snapGuide || !editor.snapGuide.isConnected) {
        const guide = document.createElement("div");
        guide.className = "myh3-rc-snap-guide";
        const guideLabel = document.createElement("span");
        guideLabel.className = "myh3-rc-snap-guide-label";
        guide.appendChild(guideLabel);
        editor.snapGuide = guide;
        editor.canvas.appendChild(guide);
    }
    editor.snapGuide.style.left = `${TRACK_HEADER_WIDTH + Math.max(0, frame) * editor.pixelsPerFrame}px`;
    editor.snapGuide.firstElementChild.textContent = label;
    editor.snapGuide.hidden = false;
}

function hideSnapGuide(editor) {
    if (editor.snapGuide) editor.snapGuide.hidden = true;
}

function beginClipPointerDrag(editor, track, clip, item, event) {
    if (event.button !== 0 || editor.clipDrag) return;
    event.preventDefault();
    event.stopPropagation();
    stopTimelinePlayback(editor);
    selectTimelineClip(editor, track, clip, item);
    const duration = Math.max(1, Number(clip.timeline_end) - Number(clip.timeline_start));
    const state = editor.clipDrag = {
        track, clip, item, pointerId: event.pointerId,
        originX: event.clientX, originalStart: Number(clip.timeline_start) || 0,
        duration, previewStart: Number(clip.timeline_start) || 0, moved: false,
    };
    item.classList.add("dragging");
    item.setPointerCapture?.(event.pointerId);
    const move = (moveEvent) => {
        if (editor.clipDrag !== state || moveEvent.pointerId !== state.pointerId) return;
        const delta = (moveEvent.clientX - state.originX) / Math.max(.0001, editor.pixelsPerFrame);
        const raw = state.originalStart + Math.round(delta);
        const aligned = alignClipStart(editor, raw, state.duration, clip.id);
        state.previewStart = aligned;
        state.moved = state.moved || Math.abs(moveEvent.clientX - state.originX) > 3;
        item.style.left = `${TRACK_HEADER_WIDTH + aligned * editor.pixelsPerFrame}px`;
        showSnapGuide(editor, aligned,
            `${(aligned / (Number(editor.project.settings.fps) || 24)).toFixed(2)}s`);
        setStatus(editor, `${clip.name} · 拖动到 ${(aligned / (Number(editor.project.settings.fps) || 24)).toFixed(2)} 秒${editor.project.settings.snap_enabled === false ? "" : " · 磁吸"}`, true);
    };
    const finish = (finishEvent, cancelled = false) => {
        if (editor.clipDrag !== state || finishEvent.pointerId !== state.pointerId) return;
        item.releasePointerCapture?.(state.pointerId);
        item.removeEventListener("pointermove", move);
        item.removeEventListener("pointerup", finish);
        item.removeEventListener("pointercancel", cancel);
        item.classList.remove("dragging");
        hideSnapGuide(editor);
        editor.clipDrag = null;
        if (cancelled || !state.moved || state.previewStart === state.originalStart) {
            item.style.left = `${TRACK_HEADER_WIDTH + state.originalStart * editor.pixelsPerFrame}px`;
            return;
        }
        clip.timeline_start = state.previewStart;
        clip.timeline_end = state.previewStart + state.duration;
        track.clips = track.clips.filter((candidate) => candidate.id !== clip.id);
        applyOverwrite(track, clip, editor.project.settings.fps);
        saveEditor(editor);
        setStatus(editor, `已将「${clip.name}」对齐到 ${(state.previewStart / (Number(editor.project.settings.fps) || 24)).toFixed(2)} 秒`, true);
    };
    const cancel = (cancelEvent) => finish(cancelEvent, true);
    item.addEventListener("pointermove", move);
    item.addEventListener("pointerup", finish);
    item.addEventListener("pointercancel", cancel);
}

function handleTimelineKeydown(editor, event) {
    if (!editor || editor !== activeEditor || !editor.modal?.isConnected
        || event.isComposing
        || event.ctrlKey || event.metaKey || event.altKey) return;
    const target = event.target;
    const tag = String(target?.tagName || "").toLowerCase();
    if (["input", "select", "textarea"].includes(tag) || target?.isContentEditable) return;
    const key = String(event.key || "").toLowerCase();
    let handled = true;
    if (key === "i" || key === "o") setSelectionPoint(editor, key === "i" ? "in" : "out");
    else if (key === " " || key === "spacebar" || event.code === "Space") toggleTimelinePlayback(editor);
    else if (key === "arrowleft") stepTimelineFrame(editor, -1);
    else if (key === "arrowright") stepTimelineFrame(editor, 1);
    else if (key === "delete" || key === "backspace") deleteSelectedTimelineClip(editor);
    else handled = false;
    if (handled) {
        event.preventDefault();
        event.stopPropagation();
        event.stopImmediatePropagation?.();
    }
}

function deleteSelectedTimelineClip(editor) {
    const selected = selectedTimelineClip(editor);
    if (!selected) {
        setStatus(editor, "先选择需要删除的时间轴片段");
        return false;
    }
    if (!confirm(`从时间轴删除「${selected.clip.name}」？不会删除磁盘文件。`)) return false;
    selected.track.clips = selected.track.clips.filter((clip) => clip.id !== selected.clip.id);
    editor.selectedClip = null;
    editor.playbackMedia = null;
    saveEditor(editor);
    syncTimelinePlaybackMedia(editor, editor.playhead);
    setStatus(editor, `已从时间轴删除「${selected.clip.name}」；磁盘素材保持不变`, true);
    return true;
}

function timelineContentEnd(project) {
    return Math.max(0, ...project.tracks.flatMap((track) =>
        track.clips.map((clip) => Number(clip.timeline_end) || 0)));
}

function clipSourceSecond(editor, clip, frame) {
    const projectRate = Math.max(1, Number(editor.project.settings.fps) || 24);
    const sourceRate = Math.max(1, Number(clip?.source_fps) || projectRate);
    const start = Math.max(0, Number(clip?.source_in) / sourceRate);
    return start + Math.max(0, frame - Number(clip?.timeline_start || 0)) / projectRate;
}

function pauseTimelineMedia(editor) {
    for (const media of [editor.playbackMedia?.visual, editor.playbackMedia?.audio]) {
        if (media?.pause) media.pause();
    }
}

function startTimelineMedia(editor, media, second) {
    if (!media?.play) return;
    const start = () => {
        if (!media.isConnected) return;
        const duration = Number(media.duration);
        const target = Number.isFinite(duration)
            ? Math.max(0, Math.min(Math.max(0, duration - .02), second)) : Math.max(0, second);
        if (Math.abs(Number(media.currentTime || 0) - target) > .18) {
            try { media.currentTime = target; } catch (_error) { /* metadata may still be settling */ }
        }
        if (!editor.playback?.playing) {
            media.pause();
        } else if (media.paused) {
            media.play().catch((error) => {
                if (error?.name !== "AbortError") {
                    setStatus(editor, `浏览器未能播放此素材：${error?.message || error}`);
                }
            });
        }
    };
    if (media.readyState >= 1) start();
    else if (!media.__myangMetadataPending) {
        media.__myangMetadataPending = true;
        media.addEventListener("loadedmetadata", () => {
            media.__myangMetadataPending = false;
            start();
        }, {once: true});
    }
}

function syncTimelinePlaybackMedia(editor, frame) {
    const visualClip = timelineClipAt(editor, "video", frame);
    const audioClip = timelineClipAt(editor, "audio", frame);
    const visualId = visualClip?.id || "";
    const audioId = audioClip?.id || "";
    let visual = editor.playbackMedia?.visual || null;
    let audio = editor.playbackMedia?.audio || null;
    if (editor.playbackMedia?.visualId !== visualId
            || editor.playbackMedia?.audioId !== audioId
            || (visualId && !editor.playbackMedia?.visual?.isConnected)
            || (audioId && !editor.playbackMedia?.audio?.isConnected)) {
        pauseTimelineMedia(editor);
        editor.preview.replaceChildren();
        if (editor.programLabel) {
            editor.programLabel.textContent = visualClip?.name || audioClip?.name || "时间轴空白区";
        }
        visual = null;
        audio = null;
        if (visualClip) {
            visual = document.createElement(visualClip.kind === "image" ? "img" : "video");
            visual.className = "myh3-rc-preview-media";
            visual.src = clipUrl(visualClip);
            visual.alt = visualClip.kind === "image" ? visualClip.name : "";
            if (visual instanceof HTMLVideoElement) {
                visual.preload = "auto";
                visual.playsInline = true;
                visual.controls = false;
                visual.muted = !!audioClip;
            }
            editor.preview.appendChild(visual);
        }
        if (!visualClip) {
            const empty = document.createElement("div");
            empty.className = "myh3-rc-preview-empty";
            empty.textContent = audioClip ? "当前只有音频片段" : "当前播放头位于时间轴空白区";
            editor.preview.appendChild(empty);
        }
        if (audioClip) {
            const badge = document.createElement("span");
            badge.className = "myh3-rc-preview-audio";
            badge.textContent = `A1 · ${audioClip.name}`;
            editor.preview.appendChild(badge);
            audio = document.createElement("audio");
            audio.src = clipUrl(audioClip);
            audio.preload = "auto";
            audio.style.cssText = "position:absolute;width:1px;height:1px;opacity:0;pointer-events:none;";
            editor.preview.appendChild(audio);
        }
        editor.playbackMedia = {visualId, audioId, visual, audio};
    }
    if (visual instanceof HTMLVideoElement && visualClip) {
        startTimelineMedia(editor, visual, clipSourceSecond(editor, visualClip, frame));
    }
    if (audio && audioClip) {
        startTimelineMedia(editor, audio, clipSourceSecond(editor, audioClip, frame));
    }
}

function setTransportState(editor, playing) {
    if (editor.playToggle) {
        editor.playToggle.textContent = playing ? "❚❚" : "▶";
        editor.playToggle.title = playing ? "暂停时间轴（空格）" : "播放时间轴（空格）";
        editor.playToggle.setAttribute("aria-label", editor.playToggle.title);
        editor.playToggle.classList.toggle("active", playing);
    }
}

function stopTimelinePlayback(editor, reset = false) {
    if (!editor) return;
    if (editor.timelinePlaybackAnimation) cancelAnimationFrame(editor.timelinePlaybackAnimation);
    editor.timelinePlaybackAnimation = 0;
    pauseTimelineMedia(editor);
    editor.playback = null;
    setTransportState(editor, false);
    if (reset) {
        updatePlayheadVisual(editor, 0);
        syncTimelinePlaybackMedia(editor, 0);
    }
}

function stepTimelineFrame(editor, delta) {
    stopTimelinePlayback(editor);
    updatePlayheadVisual(editor, Math.round(editor.playhead) + delta);
    syncTimelinePlaybackMedia(editor, editor.playhead);
    setStatus(editor, `已定位到 ${editor.transportTime?.textContent || "当前帧"}`, true);
}

function playbackEndFrame(editor, startFrame) {
    const contentEnd = timelineContentEnd(editor.project);
    if (selectionReady(editor.project)
            && startFrame >= editor.project.selection.in_frame
            && startFrame < editor.project.selection.out_frame) {
        return Math.max(startFrame + 1, Number(editor.project.selection.out_frame));
    }
    return contentEnd;
}

function playbackTick(editor, timestamp) {
    const playback = editor.playback;
    if (!playback?.playing) return;
    const fps = Math.max(1, Number(editor.project.settings.fps) || 24);
    const frame = playback.anchorFrame + (timestamp - playback.anchorTime) * fps / 1000;
    if (frame >= playback.endFrame) {
        updatePlayheadVisual(editor, playback.endFrame);
        syncTimelinePlaybackMedia(editor, Math.max(0, playback.endFrame - .001));
        stopTimelinePlayback(editor);
        setStatus(editor, "时间轴播放完成", true);
        return;
    }
    updatePlayheadVisual(editor, frame);
    syncTimelinePlaybackMedia(editor, frame);
    const x = TRACK_HEADER_WIDTH + frame * editor.pixelsPerFrame;
    const right = editor.scroll.scrollLeft + editor.scroll.clientWidth - 70;
    if (x > right) editor.scroll.scrollLeft = Math.max(0, x - editor.scroll.clientWidth * .35);
    editor.timelinePlaybackAnimation = requestAnimationFrame((next) => playbackTick(editor, next));
}

function startTimelinePlayback(editor) {
    const contentEnd = timelineContentEnd(editor.project);
    if (!(contentEnd > 0)) {
        setStatus(editor, "时间轴还没有可播放的素材");
        return;
    }
    let startFrame = Math.max(0, Number(editor.playhead) || 0);
    if (startFrame >= contentEnd) {
        startFrame = selectionReady(editor.project)
            ? Math.min(contentEnd - 1, Number(editor.project.selection.in_frame) || 0) : 0;
        updatePlayheadVisual(editor, startFrame);
    }
    const endFrame = playbackEndFrame(editor, startFrame);
    editor.playback = {
        playing: true, anchorFrame: startFrame,
        anchorTime: performance.now(), endFrame,
    };
    setTransportState(editor, true);
    syncTimelinePlaybackMedia(editor, startFrame);
    editor.timelinePlaybackAnimation = requestAnimationFrame((timestamp) => playbackTick(editor, timestamp));
    setStatus(editor, "正在播放 V1 / A1 时间轴；空格可暂停", true);
}

function toggleTimelinePlayback(editor) {
    if (editor.playback?.playing) stopTimelinePlayback(editor);
    else startTimelinePlayback(editor);
}

function updatePlayheadVisual(editor, frame = editor.playhead) {
    const fps = Math.max(1, Number(editor.project.settings.fps) || 24);
    const bounded = Math.max(0, Math.min(maxTimelineFrame(editor.project), Number(frame) || 0));
    editor.playhead = bounded;
    if (editor.playheadElement) {
        editor.playheadElement.style.transform =
            `translate3d(${bounded * editor.pixelsPerFrame}px,0,0)`;
    }
    if (editor.playheadTime) {
        editor.playheadTime.textContent = `${timecode(bounded, fps)} · ${Math.round(bounded)}f`;
    }
    if (editor.transportTime) editor.transportTime.textContent = timecode(bounded, fps);
}

function queuePlayheadVisual(editor, frame) {
    editor.pendingPlayhead = frame;
    if (editor.playheadAnimation) return;
    editor.playheadAnimation = requestAnimationFrame(() => {
        editor.playheadAnimation = 0;
        updatePlayheadVisual(editor, editor.pendingPlayhead);
    });
}

function bindPlayheadPointer(editor, ruler) {
    const move = (event) => {
        const frame = snapPlayheadFrame(editor, xToFrame(editor, event, ruler));
        queuePlayheadVisual(editor, frame);
    };
    ruler.onpointerdown = (event) => {
        if (event.button !== 0) return;
        event.preventDefault();
        stopTimelinePlayback(editor);
        editor.playheadPointer = event.pointerId;
        ruler.classList.add("dragging");
        ruler.setPointerCapture?.(event.pointerId);
        move(event);
    };
    ruler.onpointermove = (event) => {
        if (editor.playheadPointer !== event.pointerId) return;
        move(event);
    };
    const finish = (event) => {
        if (editor.playheadPointer !== event.pointerId) return;
        move(event);
        updatePlayheadVisual(editor, editor.pendingPlayhead);
        editor.playheadPointer = null;
        ruler.classList.remove("dragging");
        if (ruler.hasPointerCapture?.(event.pointerId)) {
            ruler.releasePointerCapture(event.pointerId);
        }
        syncTimelinePlaybackMedia(editor, editor.playhead);
    };
    ruler.onpointerup = finish;
    ruler.onpointercancel = finish;
}

function saveEditor(editor) {
    stopTimelinePlayback(editor);
    editor.project.revision = Number(editor.project.revision || 0) + 1;
    saveProject(editor.node, editor.project);
    updateBoundaryStatus(editor);
}

function renderEditorTimeline(editor) {
    if (!editor.canvas) return;
    const project = editor.project;
    const fps = Number(project.settings.fps) || 24;
    editor.pixelsPerFrame = editor.zoom / fps;
    const maxFrame = maxTimelineFrame(project);
    const width = Math.max(900, TRACK_HEADER_WIDTH + Math.round(maxFrame * editor.pixelsPerFrame) + 80);
    editor.canvas.style.width = `${width}px`;
    editor.canvas.replaceChildren();
    editor.snapGuide = null;

    const ruler = document.createElement("div");
    ruler.className = "myh3-rc-ruler";
    ruler.title = "按住并拖动黄色播放头；位置会实时显示";
    bindPlayheadPointer(editor, ruler);
    const rulerHead = document.createElement("div");
    rulerHead.className = "myh3-rc-ruler-head";
    rulerHead.title = "轨道头（时间轴 0 秒从右侧开始）";
    ruler.appendChild(rulerHead);
    const tickSeconds = editor.zoom < 40 ? 5 : editor.zoom < 90 ? 2 : 1;
    for (let second = 0; second <= maxFrame / fps; second += tickSeconds) {
        const tick = document.createElement("div");
        tick.className = "myh3-rc-tick";
        tick.style.left = `${TRACK_HEADER_WIDTH + second * editor.zoom}px`;
        tick.textContent = `${second}s`;
        ruler.appendChild(tick);
    }
    editor.canvas.appendChild(ruler);

    for (const track of project.tracks) {
        const row = document.createElement("div");
        row.className = "myh3-rc-track";
        row.dataset.trackId = track.id;
        row.onclick = (event) => {
            if (event.target === row || event.target?.classList?.contains("myh3-rc-track-label")) {
                clearTimelineSelection(editor);
            }
        };
        row.ondragenter = (event) => {
            if (!event.dataTransfer?.types?.length) return;
            showSnapGuide(editor, snapPlayheadFrame(editor, xToFrame(editor, event, row)));
        };
        row.ondragover = (event) => {
            event.preventDefault();
            event.dataTransfer.dropEffect = "copy";
            const raw = xToFrame(editor, event, row);
            const snapped = snapPlayheadFrame(editor, raw);
            showSnapGuide(editor, snapped, `${(snapped / fps).toFixed(2)}s · 磁吸`);
        };
        row.ondragleave = (event) => {
            if (!row.contains(event.relatedTarget)) hideSnapGuide(editor);
        };
        row.ondrop = async (event) => {
            event.preventDefault();
            hideSnapGuide(editor);
            const start = snapPlayheadFrame(editor, xToFrame(editor, event, row));
            const assetText = event.dataTransfer.getData("application/x-myang-asset");
            const clipText = event.dataTransfer.getData("application/x-myang-clip");
            try {
                if (assetText) {
                    const asset = JSON.parse(assetText);
                    const expectedKind = asset.kind === "audio" ? "audio" : "video";
                    if (track.kind !== expectedKind) {
                        throw new Error(expectedKind === "audio"
                            ? "音频素材只能拖到 A1" : "图片和视频素材只能拖到 V1");
                    }
                    const clip = await assetClip(editor, asset, start);
                    const duration = clip.timeline_end - clip.timeline_start;
                    const aligned = alignClipStart(editor, start, duration);
                    clip.timeline_start = aligned;
                    clip.timeline_end = aligned + duration;
                    applyOverwrite(track, clip, project.settings.fps);
                } else if (clipText) {
                    const info = JSON.parse(clipText);
                    const sourceTrack = project.tracks.find((item) => item.id === info.track_id);
                    const clip = sourceTrack?.clips.find((item) => item.id === info.clip_id);
                    if (!clip || sourceTrack.kind !== track.kind) return;
                    const duration = clip.timeline_end - clip.timeline_start;
                    const aligned = alignClipStart(editor, start, duration, clip.id);
                    sourceTrack.clips = sourceTrack.clips.filter((item) => item.id !== clip.id);
                    applyOverwrite(track, {...clip, timeline_start: aligned, timeline_end: aligned + duration},
                        project.settings.fps);
                }
                saveEditor(editor);
            } catch (error) { setStatus(editor, error.message); }
        };
        const label = document.createElement("div");
        label.className = "myh3-rc-track-label";
        label.textContent = track.name;
        row.appendChild(label);
        for (const clip of track.clips) {
            const item = document.createElement("div");
            const selected = editor.selectedClip?.trackId === track.id
                && editor.selectedClip?.clipId === clip.id;
            item.className = `myh3-rc-clip ${clip.kind}${selected ? " selected" : ""}`;
            item.dataset.trackId = track.id;
            item.dataset.clipId = clip.id;
            item.tabIndex = 0;
            item.setAttribute("role", "button");
            item.setAttribute("aria-label", `${clip.name}，${((clip.timeline_end - clip.timeline_start) / fps).toFixed(2)} 秒`);
            // Existing clips use pointer dragging so the drop position stays
            // visible and can be snapped frame-by-frame without browser drag
            // ghosts or losing the selected clip.
            item.draggable = false;
            item.style.left = `${TRACK_HEADER_WIDTH + clip.timeline_start * editor.pixelsPerFrame}px`;
            item.style.width = `${Math.max(24, (clip.timeline_end - clip.timeline_start) * editor.pixelsPerFrame)}px`;
            const name = document.createElement("b");
            name.textContent = clip.name;
            const duration = document.createElement("small");
            duration.textContent = `${((clip.timeline_end - clip.timeline_start) / fps).toFixed(2)}s`;
            item.append(name, duration);
            item.ondragstart = (event) => {
                selectTimelineClip(editor, track, clip, item);
                event.dataTransfer.setData(
                    "application/x-myang-clip", JSON.stringify({track_id: track.id, clip_id: clip.id}));
            };
            item.onpointerdown = (event) => beginClipPointerDrag(editor, track, clip, item, event);
            item.ondblclick = (event) => {
                event.preventDefault();
                event.stopPropagation();
                selectTimelineClip(editor, track, clip, item);
                updatePlayheadVisual(editor, clip.timeline_start);
                startTimelinePlayback(editor);
            };
            item.onclick = (event) => {
                event.stopPropagation();
                stopTimelinePlayback(editor);
                selectTimelineClip(editor, track, clip, item);
                const frame = editor.playhead >= clip.timeline_start && editor.playhead < clip.timeline_end
                    ? editor.playhead : clip.timeline_start;
                updatePlayheadVisual(editor, frame);
                syncTimelinePlaybackMedia(editor, frame);
            };
            item.oncontextmenu = (event) => {
                event.preventDefault();
                event.stopPropagation();
                selectTimelineClip(editor, track, clip, item);
                deleteSelectedTimelineClip(editor);
            };
            row.appendChild(item);
        }
        editor.canvas.appendChild(row);
    }
    const totalHeight = 32 + project.tracks.length * 70;
    const mode = durationMode(project);
    const markerSpecs = [];
    if (mode === DURATION_DIRECTOR && selectionReady(project)) {
        markerSpecs.push(["i", project.selection.in_frame, !project.selection.in_set]);
        markerSpecs.push(["o", project.selection.out_frame, !project.selection.out_set]);
    } else {
        if (project.selection.in_set) markerSpecs.push(["i", project.selection.in_frame, false]);
        if (project.selection.out_set) markerSpecs.push(["o", project.selection.out_frame, false]);
    }
    if (selectionReady(project)) {
        const range = document.createElement("div");
        range.className = "myh3-rc-range";
        range.style.left = `${TRACK_HEADER_WIDTH + project.selection.in_frame * editor.pixelsPerFrame}px`;
        range.style.width = `${Math.max(1, (project.selection.out_frame - project.selection.in_frame) * editor.pixelsPerFrame)}px`;
        range.style.height = `${totalHeight}px`;
        editor.canvas.appendChild(range);
    }
    for (const [kind, frame, automatic] of markerSpecs) {
        const marker = document.createElement("div");
        marker.className = `myh3-rc-marker ${kind}${automatic ? " auto" : ""}`;
        marker.style.left = `${TRACK_HEADER_WIDTH + frame * editor.pixelsPerFrame}px`;
        marker.style.height = `${totalHeight}px`;
        const label = document.createElement("span");
        label.className = "myh3-rc-marker-label";
        label.textContent = kind.toUpperCase();
        marker.appendChild(label);
        editor.canvas.appendChild(marker);
    }
    const playhead = document.createElement("div");
    playhead.className = "myh3-rc-playhead";
    playhead.style.height = `${totalHeight}px`;
    const playheadHandle = document.createElement("span");
    playheadHandle.className = "myh3-rc-playhead-handle";
    const playheadTime = document.createElement("span");
    playheadTime.className = "myh3-rc-playhead-time";
    playhead.append(playheadHandle, playheadTime);
    editor.canvas.appendChild(playhead);
    editor.playheadElement = playhead;
    editor.playheadTime = playheadTime;
    updatePlayheadVisual(editor);
    if (editor.inInput) editor.inInput.value = (project.selection.in_frame / fps).toFixed(3);
    if (editor.outInput) editor.outInput.value = (project.selection.out_frame / fps).toFixed(3);
    if (editor.inInput) editor.inInput.step = (1 / fps).toFixed(6);
    if (editor.outInput) editor.outInput.step = (1 / fps).toFixed(6);
    editor.startMode.value = project.selection.start_mode;
    editor.endMode.value = project.selection.end_mode;
    editor.startMode.disabled = !project.selection.in_set
        && !(mode === DURATION_DIRECTOR && selectionReady(project));
    editor.endMode.disabled = !project.selection.out_set
        && !(mode === DURATION_DIRECTOR && selectionReady(project));
    if (editor.directorMode) editor.directorMode.classList.toggle("active", mode === DURATION_DIRECTOR);
    if (editor.timelineMode) editor.timelineMode.classList.toggle("active", mode === DURATION_TIMELINE);
    if (editor.autoAlign) editor.autoAlign.checked = project.settings.auto_align !== false;
    if (editor.snapEnabled) editor.snapEnabled.checked = project.settings.snap_enabled !== false;
    if (editor.widthInput) editor.widthInput.value = project.settings.width;
    if (editor.heightInput) editor.heightInput.value = project.settings.height;
    if (editor.rateInput) editor.rateInput.value = fps;
    updateDeleteControl(editor);
    editor.rangeLabel.textContent = selectionText(project);
}

function buildEditor(node) {
    editorStyle();
    const backdrop = document.createElement("div");
    backdrop.className = "myh3-rc-backdrop";
    const modal = document.createElement("div");
    modal.className = "myh3-rc-modal";
    const head = document.createElement("div");
    head.className = "myh3-rc-head";
    const title = document.createElement("h2");
    title.textContent = "沐阳导演台 · 粗剪时间轴（实验）";
    const rangeLabel = document.createElement("span");
    rangeLabel.style.color = "#93c5fd";
    const exportProject = document.createElement("button");
    exportProject.textContent = "导出工程 JSON";
    const importProject = document.createElement("button");
    importProject.textContent = "导入工程 JSON";
    const projectFile = document.createElement("input");
    projectFile.type = "file";
    projectFile.accept = ".json,application/json";
    projectFile.style.display = "none";
    const exportVideo = document.createElement("button");
    exportVideo.textContent = "导出成片 MP4";
    const close = document.createElement("button");
    close.textContent = "完成并关闭";
    close.onclick = () => closeEditor();
    head.append(title, rangeLabel, importProject, exportProject, exportVideo, projectFile, close);
    const grid = document.createElement("div");
    grid.className = "myh3-rc-grid";
    const library = document.createElement("aside");
    library.className = "myh3-rc-library";
    const libTitle = document.createElement("b");
    libTitle.textContent = "素材百宝箱";
    const pathRow = document.createElement("div");
    pathRow.className = "myh3-rc-row";
    pathRow.style.marginTop = "8px";
    const add = document.createElement("button");
    add.className = "myh3-rc-btn";
    add.textContent = "＋ 浏览并添加文件夹";
    add.title = "打开系统文件夹选择器；不需要手工输入路径";
    add.style.width = "100%";
    pathRow.append(add);
    const libraryList = document.createElement("div");
    library.append(libTitle, pathRow, libraryList);
    const assets = document.createElement("section");
    assets.className = "myh3-rc-assets";
    const assetTop = document.createElement("div");
    assetTop.className = "myh3-rc-row";
    const assetTitle = document.createElement("b");
    assetTitle.textContent = "素材 · 拖到下方轨道（双击预览）";
    assetTitle.style.flex = "1";
    const search = document.createElement("input");
    search.className = "myh3-rc-input";
    search.placeholder = "搜索文件名";
    const assetGrid = document.createElement("div");
    assetGrid.className = "myh3-rc-asset-grid";
    assetGrid.style.marginTop = "8px";
    assetTop.append(assetTitle, search);
    assets.append(assetTop, assetGrid);
    const program = document.createElement("section");
    program.className = "myh3-rc-program";
    const programHead = document.createElement("div");
    programHead.className = "myh3-rc-program-head";
    const programTitle = document.createElement("b");
    programTitle.textContent = "节目监视器";
    const programLabel = document.createElement("span");
    programLabel.className = "myh3-rc-program-label";
    programLabel.textContent = "时间轴空白区";
    programHead.append(programTitle, programLabel);
    const preview = document.createElement("div");
    preview.className = "myh3-rc-preview";
    const previewEmpty = document.createElement("div");
    previewEmpty.className = "myh3-rc-preview-empty";
    previewEmpty.textContent = "将播放头移到片段上，或双击素材预览";
    preview.appendChild(previewEmpty);
    const timeline = document.createElement("section");
    timeline.className = "myh3-rc-timeline";
    const controls = document.createElement("div");
    controls.className = "myh3-rc-controls";
    const transport = document.createElement("div");
    transport.className = "myh3-rc-transport";
    const previousFrame = document.createElement("button");
    previousFrame.type = "button";
    previousFrame.textContent = "‹";
    previousFrame.title = "上一帧（左方向键）";
    previousFrame.setAttribute("aria-label", previousFrame.title);
    const playToggle = document.createElement("button");
    playToggle.type = "button";
    playToggle.textContent = "▶";
    playToggle.title = "播放时间轴（空格）";
    playToggle.setAttribute("aria-label", playToggle.title);
    const stopPlayback = document.createElement("button");
    stopPlayback.type = "button";
    stopPlayback.textContent = "■";
    stopPlayback.title = "停止并回到时间轴起点";
    stopPlayback.setAttribute("aria-label", stopPlayback.title);
    const nextFrame = document.createElement("button");
    nextFrame.type = "button";
    nextFrame.textContent = "›";
    nextFrame.title = "下一帧（右方向键）";
    nextFrame.setAttribute("aria-label", nextFrame.title);
    const transportTime = document.createElement("span");
    transportTime.className = "myh3-rc-transport-time";
    transportTime.textContent = "00:00:00:00";
    transport.append(previousFrame, playToggle, stopPlayback, nextFrame, transportTime);
    program.append(programHead, preview, transport);
    const modeGroup = document.createElement("div");
    modeGroup.className = "myh3-rc-mode-group";
    const directorMode = document.createElement("button");
    directorMode.type = "button"; directorMode.className = "myh3-rc-mode";
    directorMode.textContent = "沿用导演台时长";
    directorMode.title = "只设置 I 或 O，另一端按导演台当前时长自动匹配";
    const timelineMode = document.createElement("button");
    timelineMode.type = "button"; timelineMode.className = "myh3-rc-mode";
    timelineMode.textContent = "使用时间轴 I/O 时长";
    timelineMode.title = "分别设置 I 和 O，并让本次生成时长匹配所选范围";
    modeGroup.append(directorMode, timelineMode);
    const setI = document.createElement("button");
    setI.type = "button"; setI.className = "myh3-rc-point i"; setI.textContent = "设置入点 I";
    setI.title = "把当前播放头设为入点（快捷键 I）";
    const setO = document.createElement("button");
    setO.type = "button"; setO.className = "myh3-rc-point o"; setO.textContent = "设置出点 O";
    setO.title = "把当前播放头设为出点（快捷键 O）";
    const deleteClip = document.createElement("button");
    deleteClip.type = "button";
    deleteClip.className = "myh3-rc-delete";
    deleteClip.textContent = "删除片段";
    deleteClip.disabled = true;
    deleteClip.title = "先在 V1 或 A1 轨道选择一个片段";
    const makeSwitch = (text) => {
        const label = document.createElement("label");
        label.className = "myh3-rc-switch";
        const input = document.createElement("input");
        input.type = "checkbox";
        label.append(input, text);
        return [label, input];
    };
    const [autoAlignLabel, autoAlign] = makeSwitch("自动对齐");
    const [snapLabel, snapEnabled] = makeSwitch("自动吸附");
    const startMode = document.createElement("select"); startMode.className = "myh3-rc-select";
    const endMode = document.createElement("select"); endMode.className = "myh3-rc-select";
    for (const [value, label] of MODE_OPTIONS) {
        startMode.append(new Option(`入点：${label}`, value));
        endMode.append(new Option(`出点：${label}`, value));
    }
    const zoom = document.createElement("input"); zoom.type = "range"; zoom.min = "20"; zoom.max = "180"; zoom.value = "70";
    const widthInput = document.createElement("input");
    widthInput.type = "number"; widthInput.min = "32"; widthInput.max = "8192"; widthInput.step = "2"; widthInput.className = "myh3-rc-input";
    const heightInput = document.createElement("input");
    heightInput.type = "number"; heightInput.min = "32"; heightInput.max = "8192"; heightInput.step = "2"; heightInput.className = "myh3-rc-input";
    const canvasLabel = document.createElement("label");
    canvasLabel.append("导出画布", widthInput, "×", heightInput);
    const rateInput = document.createElement("input");
    rateInput.type = "number"; rateInput.min = "1"; rateInput.max = "240";
    rateInput.step = "0.001"; rateInput.className = "myh3-rc-input";
    const rateLabel = document.createElement("label");
    rateLabel.append("帧率", rateInput);
    const hintLine = document.createElement("div");
    hintLine.className = "myh3-rc-hintline";
    hintLine.textContent = "先在标尺定位播放头，再按 I/O 或点击设置按钮；刚打开时不会预设入点、出点。";
    controls.append(modeGroup, setI, setO, deleteClip, autoAlignLabel, snapLabel,
        startMode, endMode, canvasLabel, rateLabel, "缩放", zoom, hintLine);
    const scroll = document.createElement("div"); scroll.className = "myh3-rc-scroll";
    const canvas = document.createElement("div"); canvas.className = "myh3-rc-canvas";
    scroll.appendChild(canvas);
    const status = document.createElement("div"); status.className = "myh3-rc-status";
    timeline.append(controls, scroll, status);
    grid.append(library, assets, program, timeline);
    modal.append(head, grid);
    document.body.append(backdrop, modal);
    const editor = {
        node, project: readRoughCutProject(node), backdrop, modal,
        libraries: [], catalogueAssets: [], assets: [],
        libraryId: "", libraryList, assetGrid, search, preview, programLabel, scroll, canvas, status,
        rangeLabel, startMode, endMode, widthInput, heightInput, rateInput,
        directorMode, timelineMode, autoAlign, snapEnabled,
        playToggle, stopPlayback, previousFrame, nextFrame, transportTime, deleteClip,
        selectedClip: null,
        playhead: 0, pendingPlayhead: 0, playheadPointer: null,
        playheadAnimation: 0, playheadElement: null, playheadTime: null, zoom: 70,
        timelinePlaybackAnimation: 0, playback: null, playbackMedia: null,
        pixelsPerFrame: 70 / 24,
    };
    playToggle.onclick = () => toggleTimelinePlayback(editor);
    previousFrame.onclick = () => stepTimelineFrame(editor, -1);
    nextFrame.onclick = () => stepTimelineFrame(editor, 1);
    deleteClip.onclick = () => deleteSelectedTimelineClip(editor);
    stopPlayback.onclick = () => {
        stopTimelinePlayback(editor, true);
        setStatus(editor, "播放已停止并回到时间轴起点", true);
    };
    exportProject.onclick = () => {
        downloadProjectJson(editor.project);
        setStatus(editor, "粗剪工程 JSON 已导出，可用于备份或继续编辑", true);
    };
    importProject.onclick = () => projectFile.click();
    projectFile.onchange = async () => {
        const file = projectFile.files?.[0];
        if (!file) return;
        try {
            const imported = parseProjectJson(await file.text());
            const hasCurrentClips = editor.project.tracks.some((track) => track.clips.length > 0);
            if (hasCurrentClips && !confirm("导入会替换当前粗剪工程，是否继续？")) return;
            saveProject(editor.node, imported);
            setStatus(editor, `已导入：${file.name}`, true);
        } catch (error) {
            setStatus(editor, `导入失败：${error.message}`);
        } finally {
            projectFile.value = "";
        }
    };
    exportVideo.onclick = async () => {
        exportVideo.disabled = true;
        const previous = exportVideo.textContent;
        exportVideo.textContent = "正在导出…";
        setStatus(editor, "正在合成整条 V1/A1 时间轴；长素材需要一些时间", true);
        try {
            const payload = await apiJson(`${MEDIA_ROUTE}/export`, {
                method: "POST", headers: {"Content-Type": "application/json"},
                body: JSON.stringify({project_json: JSON.stringify(editor.project)}),
            });
            previewExport(editor, payload);
            setStatus(editor,
                `导出完成：${payload.frames || 0} 帧 · ${Number(payload.fps || 0).toFixed(3)} fps`, true);
        } catch (error) {
            setStatus(editor, `导出失败：${error.message}`);
        } finally {
            exportVideo.disabled = false;
            exportVideo.textContent = previous;
        }
    };
    add.onclick = async () => {
        add.disabled = true;
        const previous = add.textContent;
        add.textContent = "正在打开文件夹选择器…";
        try {
            const result = await apiJson(`${MEDIA_ROUTE}/browse-folder`, {
                method: "POST", headers: {"Content-Type": "application/json"},
                body: "{}",
            });
            if (result.cancelled) {
                setStatus(editor, "已取消选择文件夹", true);
                return;
            }
            setStatus(editor, `素材文件夹「${result.library?.name || "未命名"}」已加入百宝箱`, true);
            await loadLibraries(editor);
        } catch (error) { setStatus(editor, error.message); }
        finally {
            add.disabled = false;
            add.textContent = previous;
        }
    };
    search.oninput = () => renderAssets(editor);
    zoom.oninput = () => {
        const previousPixels = Math.max(0.0001, editor.pixelsPerFrame);
        const centerFrame = Math.max(0,
            (scroll.scrollLeft + scroll.clientWidth / 2 - TRACK_HEADER_WIDTH) / previousPixels);
        editor.zoom = Number(zoom.value);
        renderEditorTimeline(editor);
        scroll.scrollLeft = Math.max(0,
            TRACK_HEADER_WIDTH + centerFrame * editor.pixelsPerFrame - scroll.clientWidth / 2);
    };
    const switchDurationMode = (nextMode) => {
        const selection = editor.project.selection;
        const previousMode = durationMode(editor.project);
        if (previousMode === nextMode) return;
        selection.duration_mode = nextMode;
        if (nextMode === DURATION_TIMELINE) {
            // A Director-duration range has one authored point and one derived
            // point. Keep only the authored point so the user deliberately
            // confirms the other edge in the two-point mode.
            if (selection.anchor_side === "out") {
                selection.in_set = false;
                selection.out_set = true;
            } else if (selectionReady(editor.project)) {
                selection.in_set = true;
                selection.out_set = false;
                selection.anchor_side = "in";
            }
            setStatus(editor, "已切换为时间轴 I/O 时长；请分别确认 I 和 O");
        } else {
            const duration = directorDurationFrames(editor);
            if (selection.out_set && !selection.in_set) {
                selection.anchor_side = "out";
                selection.in_frame = Math.max(0, selection.out_frame - duration);
            } else if (selection.in_set || selection.out_set) {
                selection.anchor_side = "in";
                selection.in_set = true;
                selection.out_set = false;
                selection.out_frame = selection.in_frame + duration;
            }
            setStatus(editor, "已切换为导演台时长；设置一个边界即可", true);
        }
        saveEditor(editor);
    };
    directorMode.onclick = () => switchDurationMode(DURATION_DIRECTOR);
    timelineMode.onclick = () => switchDurationMode(DURATION_TIMELINE);
    autoAlign.onchange = () => {
        editor.project.settings.auto_align = autoAlign.checked;
        setStatus(editor, autoAlign.checked ? "素材自动对齐已开启" : "素材自动对齐已关闭", true);
        saveEditor(editor);
    };
    snapEnabled.onchange = () => {
        editor.project.settings.snap_enabled = snapEnabled.checked;
        setStatus(editor, snapEnabled.checked ? "播放头与 I/O 自动吸附已开启" : "播放头与 I/O 自动吸附已关闭", true);
        saveEditor(editor);
    };
    const applyCanvas = () => {
        const even = (value, fallback) => Math.max(32, Math.min(8192,
            Math.round((Number(value) || fallback) / 2) * 2));
        editor.project.settings.width = even(widthInput.value, editor.project.settings.width);
        editor.project.settings.height = even(heightInput.value, editor.project.settings.height);
        saveEditor(editor);
    };
    widthInput.onchange = applyCanvas; heightInput.onchange = applyCanvas;
    rateInput.onchange = () => {
        const previous = Math.max(1, Number(editor.project.settings.fps) || 24);
        const next = Math.max(1, Math.min(240, Number(rateInput.value) || previous));
        const ratio = next / previous;
        if (Math.abs(ratio - 1) < 1e-9) return;
        for (const track of editor.project.tracks) {
            for (const clip of track.clips) {
                clip.timeline_start = Math.max(0, Math.round(clip.timeline_start * ratio));
                clip.timeline_end = Math.max(
                    clip.timeline_start + 1, Math.round(clip.timeline_end * ratio));
            }
        }
        editor.project.selection.in_frame = Math.max(0,
            Math.round(editor.project.selection.in_frame * ratio));
        editor.project.selection.out_frame = Math.max(
            editor.project.selection.in_frame + 1,
            Math.round(editor.project.selection.out_frame * ratio));
        editor.playhead = Math.max(0, Math.round(editor.playhead * ratio));
        editor.project.settings.fps = next;
        saveEditor(editor);
    };
    setI.onclick = () => setSelectionPoint(editor, "in");
    setO.onclick = () => setSelectionPoint(editor, "out");
    startMode.onchange = () => {
        editor.project.selection.start_mode = startMode.value;
        editor.project.selection.use_first_frame = startMode.value === "frame";
        saveEditor(editor);
    };
    endMode.onchange = () => {
        editor.project.selection.end_mode = endMode.value;
        editor.project.selection.use_last_frame = endMode.value === "frame";
        saveEditor(editor);
    };
    modal.tabIndex = -1;
    editor.keyboardHandler = (event) => handleTimelineKeydown(editor, event);
    window.addEventListener("keydown", editor.keyboardHandler, true);
    backdrop.onclick = () => closeEditor();
    for (const eventName of ["pointerdown", "mousedown", "click", "dblclick", "wheel", "keydown", "keyup"]) {
        modal.addEventListener(eventName, (event) => event.stopPropagation());
    }
    renderEditorTimeline(editor);
    syncTimelinePlaybackMedia(editor, editor.playhead);
    modal.focus({preventScroll: true});
    void loadLibraries(editor);
    return editor;
}

export function openRoughCutEditor(node) {
    closeEditor();
    try {
        activeEditor = buildEditor(node);
    } catch (error) {
        document.querySelectorAll(".myh3-rc-modal,.myh3-rc-backdrop")
            .forEach((element) => element.remove());
        console.error("[H3-Myang] 打开粗剪时间轴失败", error);
        alert(`粗剪时间轴打开失败：${error?.message || error}`);
    }
}

function closeEditor() {
    if (activeEditor) {
        stopTimelinePlayback(activeEditor);
        if (activeEditor.keyboardHandler) {
            window.removeEventListener("keydown", activeEditor.keyboardHandler, true);
        }
    }
    if (activeEditor?.playheadAnimation) cancelAnimationFrame(activeEditor.playheadAnimation);
    activeEditor?.modal?.remove();
    activeEditor?.backdrop?.remove();
    document.querySelectorAll(".myh3-rc-modal,.myh3-rc-backdrop")
        .forEach((element) => element.remove());
    activeEditor = null;
}

export function applyRoughCutCommit(node, detail) {
    // A run queued while the editor was enabled may finish after the user has
    // turned write-back off.  Late/stale browser events must never mutate the
    // saved timeline merely because their owner node still exists.
    if (nodeWidget(node, ENABLED_WIDGET)?.value !== true
        || readRoughCutProject(node).settings.writeback_enabled !== true) return false;
    const serialized = String(detail?.project_json || "");
    if (!serialized) return false;
    const committed = normalizeProject(serialized);
    const current = readRoughCutProject(node);
    if (current.id !== committed.id) return false;
    let next = committed;
    let merged = false;
    // Rendering can take minutes. Preserve edits made after queueing by
    // merging only the generated clip from the execution snapshot back into
    // the current project instead of replacing the entire timeline.
    if (current.id === committed.id && Number(current.revision) >= Number(committed.revision)) {
        const selection = committed.selection;
        const committedTrack = committed.tracks.find((item) => item.id === selection.target_track);
        const generated = committedTrack?.clips.find((clip) =>
            clip.kind === "generated"
            && clip.timeline_start === selection.in_frame
            && clip.timeline_end === selection.out_frame
            && (!detail?.video?.filename || clip.source?.filename === detail.video.filename));
        if (generated) {
            next = structuredClone(current);
            const target = next.tracks.find((item) => item.id === selection.target_track)
                || next.tracks.find((item) => item.kind === "video");
            if (target) {
                applyOverwrite(target, structuredClone(generated), next.settings.fps);
                next.revision = Math.max(Number(current.revision), Number(committed.revision)) + 1;
                merged = true;
            }
        }
    }
    saveProject(node, next);
    const card = node.__myangRoughCutCardEls;
    if (card) {
        card.state.textContent = merged ? "已合并写入（保留期间编辑）" : "已覆盖写入时间轴";
        card.state.style.color = "#86efac";
    }
    return true;
}
