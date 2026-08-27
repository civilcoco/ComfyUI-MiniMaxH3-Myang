import { app } from "../../scripts/app.js";

const NODE = "H3TurboSchedule";
const MANUAL = "手动（高级）";
const AUTO = "自动匹配 LoRA 文件（推荐）";
const EXTERNAL = "不在本节点加载（兼容旧工作流）";
const PROFILE_8 = "LightX2V v1.0 · 8步（12/3·通用）";
const PROFILE_4_768 = "LightX2V v1.0 · 4步768P（6/3）";
const PROFILE_4 = "LightX2V v0.1 · 4步（12/3）";
const PROFILE_REF_4 = "LightX2V Ref2VA v0.1 · 4步（12/3）";

const LABELS = {
    profile: "LightX2V 官方档位",
    speed_cache: "加速缓存",
    shift_video: "自定义视频 Shift",
    shift_audio: "自定义音频 Shift",
    recommended_steps: "手动推荐步数",
    "LoRA文件": "Turbo LoRA 文件",
    "LoRA强度": "LoRA 模型强度",
    "手动覆盖Shift": "手动覆盖官方 Shift",
};

function inferredProfile(by) {
    let profile = String(by.profile?.value || "");
    if (profile !== AUTO) return profile;
    const filename = String(by["LoRA文件"]?.value || "").toLowerCase().replaceAll("-", "_");
    if ((filename.includes("ref2va") || filename.includes("ref2v"))
        && filename.includes("4step")) return PROFILE_REF_4;
    if (filename.includes("8step")) return PROFILE_8;
    if (filename.includes("4step") && filename.includes("768p")) return PROFILE_4_768;
    if (filename.includes("4step")) return PROFILE_4;
    return "";
}

function officialShift(by) {
    const profile = inferredProfile(by);
    if (profile === PROFILE_4_768) return {video: 6, audio: 3, profile};
    if ([PROFILE_8, PROFILE_4, PROFILE_REF_4].includes(profile)) {
        return {video: 12, audio: 3, profile};
    }
    return null;
}

function makeShiftStatus(node) {
    const root = document.createElement("div");
    root.style.cssText = "box-sizing:border-box;width:100%;padding:5px 8px;border:1px solid #35445a;border-radius:6px;background:#111923;color:#cbd5e1;font:11px/1.45 sans-serif;";
    root.setAttribute("role", "status");
    const row = document.createElement("div");
    row.style.cssText = "display:flex;align-items:center;gap:10px;flex-wrap:wrap;";
    const video = document.createElement("span");
    const audio = document.createElement("span");
    const mode = document.createElement("span");
    mode.style.cssText = "margin-left:auto;font-size:10px;";
    row.append(video, audio, mode);
    const hint = document.createElement("div");
    hint.style.cssText = "margin-top:3px;color:#8291a5;font-size:9px;white-space:normal;";
    root.append(row, hint);
    node.__myangTurboShiftStatus = {root, video, audio, mode, hint};
    return root;
}

function renderShiftStatus(node, by, manual, override, knownOfficial) {
    const status = node.__myangTurboShiftStatus;
    if (!status?.root?.isConnected) return;
    const video = Number(by.shift_video?.value || 0);
    const audio = Number(by.shift_audio?.value || 0);
    status.video.textContent = `视频 Shift  ${video.toFixed(2)}`;
    status.audio.textContent = `音频 Shift  ${audio.toFixed(2)}`;
    if (manual) {
        status.mode.textContent = "手动档";
        status.mode.style.color = "#fbbf24";
        status.hint.textContent = "当前数值会直接传入 MiniMaxH3SigmaShift。";
    } else if (override) {
        status.mode.textContent = "自定义覆盖";
        status.mode.style.color = "#fb923c";
        status.hint.textContent = "已偏离 LightX2V 官方 Shift；步数约束仍按所选档位执行。";
    } else if (knownOfficial) {
        status.mode.textContent = "官方档位";
        status.mode.style.color = "#86efac";
        status.hint.textContent = "当前显示的是实际传入值；打开“手动覆盖官方 Shift”后可修改。";
    } else {
        status.mode.textContent = "等待识别";
        status.mode.style.color = "#94a3b8";
        status.hint.textContent = "自动档尚未识别 LoRA 文件；请选择本节点中的 LoRA 或明确指定档位。";
    }
}

function hide(widget) {
    if (!widget || widget.__myangTurboHidden) return;
    widget.__myangTurboHidden = {
        type: widget.type,
        computeSize: widget.computeSize,
        ownCompute: Object.prototype.hasOwnProperty.call(widget, "computeSize"),
        computedHeight: widget.computedHeight,
        ownHeight: Object.prototype.hasOwnProperty.call(widget, "computedHeight"),
    };
    widget.hidden = true;
    widget.type = "hidden";
    widget.computeSize = () => [0, -4];
    widget.computedHeight = 0;
    if (widget.inputEl) widget.inputEl.style.display = "none";
    if (widget.element) widget.element.style.display = "none";
}

function show(widget) {
    if (!widget?.__myangTurboHidden) return;
    const state = widget.__myangTurboHidden;
    delete widget.__myangTurboHidden;
    widget.hidden = false;
    widget.type = state.type;
    if (state.ownCompute) widget.computeSize = state.computeSize;
    else delete widget.computeSize;
    if (state.ownHeight) widget.computedHeight = state.computedHeight;
    else delete widget.computedHeight;
    if (widget.inputEl) widget.inputEl.style.display = "";
    if (widget.element) widget.element.style.display = "";
}

function visible(widget, enabled) {
    if (enabled) show(widget);
    else hide(widget);
}

function refresh(node) {
    const by = {};
    for (const item of node.widgets || []) {
        by[item.name] = item;
        if (LABELS[item.name]) item.label = LABELS[item.name];
    }
    const manual = String(by.profile?.value || "") === MANUAL;
    const official = manual ? null : officialShift(by);
    const override = manual || by["手动覆盖Shift"]?.value === true;
    if (!override && official) {
        if (by.shift_video) by.shift_video.value = official.video;
        if (by.shift_audio) by.shift_audio.value = official.audio;
    }
    visible(by["手动覆盖Shift"], !manual);
    visible(by.shift_video, override);
    visible(by.shift_audio, override);
    visible(by.recommended_steps, manual);
    const loadsHere = String(by["LoRA文件"]?.value || EXTERNAL) !== EXTERNAL;
    visible(by["LoRA强度"], loadsHere);
    const modelInput = (node.inputs || []).find((input) => input.name === "model");
    if (modelInput) modelInput.label = "基础模型 / 上游已挂 LoRA 模型";
    const outputLabels = {
        model: "Turbo 联合模型",
        recommended_steps: "推荐步数",
        shift_video: "视频 Shift",
        shift_audio: "音频 Shift",
    };
    for (const output of node.outputs || []) {
        if (outputLabels[output.name]) output.label = outputLabels[output.name];
    }
    renderShiftStatus(node, by, manual, override, !!official);
    node.graph?.setDirtyCanvas?.(true, true);
}

app.registerExtension({
    name: "Myang_node.MiniMaxH3.TurboJointLoader",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== NODE) return;
        const onCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onCreated?.apply(this, arguments);
            for (const item of this.widgets || []) {
                const previous = item.callback;
                item.callback = (...args) => {
                    const value = previous?.apply(item, args);
                    requestAnimationFrame(() => refresh(this));
                    return value;
                };
            }
            const status = this.addDOMWidget(
                "myang_turbo_shift_status", "turbo_shift_status", makeShiftStatus(this), {
                    serialize: false,
                    hideOnZoom: false,
                    getMinHeight: () => 48,
                    getMaxHeight: () => 64,
                });
            status.serialize = false;
            requestAnimationFrame(() => refresh(this));
            return result;
        };
        for (const hook of ["onConfigure", "onAdded"]) {
            const original = nodeType.prototype[hook];
            nodeType.prototype[hook] = function () {
                const result = original?.apply(this, arguments);
                requestAnimationFrame(() => refresh(this));
                return result;
            };
        }
    },
});
