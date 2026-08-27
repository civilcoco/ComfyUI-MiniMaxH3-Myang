// Shared monotonic progress state for the Director and standalone long-video UI.
// ComfyUI may execute expanded graph branches out of segment order, so arrival
// order must never be treated as the global render order.

export const PROGRESS_PHASE = {
    sample1: {label: "一采采样", steps: true},
    drift: {label: "漂移校正", steps: false},
    refine_prep: {label: "二采准备", steps: false},
    sample2: {label: "二采采样", steps: true},
    finalizing: {label: "解码分段", steps: false},
    segments: {label: "完成剩余分段", steps: false, global: true},
    assembling: {label: "合并分段", steps: false, global: true},
    encoding: {label: "编码保存成片", steps: false, global: true},
};

const SEGMENT_PROGRESS_WEIGHT = 94;
const ASSEMBLING_PERCENT = 96;
const ENCODING_PERCENT = 99;
const EPSILON = 1e-6;

function positiveInt(value, fallback = 1) {
    const parsed = Math.trunc(Number(value));
    return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

function clamp01(value) {
    return Math.max(0, Math.min(1, Number(value) || 0));
}

export function createProgressState(options = {}) {
    const total = positiveInt(options.total, 1);
    return {
        status: options.status || "running",
        runId: String(options.runId || options.run_id || ""),
        total,
        seg: 1,
        phase: "sample1",
        step: 0,
        stepMax: 0,
        previewFile: "",
        previewTs: 0,
        prompt: "",
        brief: "",
        refining: !!options.refining,
        correcting: !!options.correcting,
        error: "",
        progressSeen: false,
        segmentFractions: {},
        completedSegments: {},
        maxSeenSegment: 0,
        outOfOrder: false,
        maxPercent: 0,
    };
}

export function ensureProgressState(state) {
    state.status ||= "running";
    state.runId = String(state.runId || state.run_id || "");
    state.total = positiveInt(state.total, 1);
    state.seg = positiveInt(state.seg, 1);
    state.phase ||= "sample1";
    state.segmentFractions ||= {};
    state.completedSegments ||= {};
    state.maxSeenSegment = Math.max(0, Number(state.maxSeenSegment) || 0);
    state.maxPercent = Math.max(0, Number(state.maxPercent) || 0);
    state.outOfOrder = !!state.outOfOrder;
    return state;
}

export function completedSegmentCount(state) {
    ensureProgressState(state);
    return Object.values(state.completedSegments).filter(Boolean).length;
}

export function isGlobalProgressPhase(phase) {
    return !!PROGRESS_PHASE[String(phase || "")]?.global;
}

function eventPhase(state, detail, stage) {
    if (stage === "sampling") {
        const pass = String(detail?.pass_label || "sample1");
        if (pass.startsWith("sample2")) return "sample2";
        return "sample1";
    }
    if (stage === "sampled") {
        return state.correcting ? "drift"
            : state.refining ? "refine_prep" : "finalizing";
    }
    if (stage === "drifted") return state.refining ? "refine_prep" : "finalizing";
    if (stage === "refine_start") return "refine_prep";
    if (stage === "refined") return "finalizing";
    return null;
}

function eventFraction(state, detail, stage) {
    if (stage === "sampling") {
        const ratio = clamp01(Number(detail?.step || 0) / positiveInt(detail?.step_total, 1));
        const pass = String(detail?.pass_label || "sample1");
        if (pass.startsWith("sample2")) return 0.60 + ratio * 0.25;
        return ratio * 0.40;
    }
    if (stage === "sampled") {
        if (state.correcting) return 0.40;
        return state.refining ? 0.50 : 0.85;
    }
    if (stage === "drifted" || stage === "refine_start") {
        return state.refining ? 0.50 : 0.85;
    }
    if (stage === "refined") return 0.85;
    if (stage === "done") return 1;
    return null;
}

function lowerSegmentsComplete(state, segment) {
    for (let index = 1; index < segment; index += 1) {
        if (!state.completedSegments[index]) return false;
    }
    return true;
}

export function applyProgressEvent(state, detail = {}) {
    ensureProgressState(state);
    const incomingRun = String(detail.run_id || "");
    if (state.runId && incomingRun && incomingRun !== state.runId) return false;
    if (["done", "error"].includes(state.status)) return false;

    const stage = String(detail.stage || "");
    const incomingTotal = positiveInt(detail.total_segments, state.total);
    state.total = Math.max(state.total, incomingTotal);

    if (stage === "assembling") {
        state.status = "running";
        state.phase = "assembling";
        state.seg = state.total;
        state.step = 0;
        state.stepMax = 0;
        return true;
    }
    if (stage === "assembled" || stage === "encoding") {
        state.status = "running";
        state.phase = "encoding";
        state.seg = state.total;
        state.step = 0;
        state.stepMax = 0;
        return true;
    }
    if (["assembling", "encoding"].includes(state.phase)) return false;

    const segment = Math.min(state.total, positiveInt(detail.segment_index, state.seg));
    if (state.completedSegments[segment] && stage !== "done") return false;
    const fraction = eventFraction(state, detail, stage);
    if (fraction === null) return false;
    const previous = Number(state.segmentFractions[segment] || 0);
    if (fraction + EPSILON < previous) return false;

    if (state.maxSeenSegment > 0 && segment < state.maxSeenSegment) state.outOfOrder = true;
    if (segment > 1 && !lowerSegmentsComplete(state, segment)) state.outOfOrder = true;
    state.maxSeenSegment = Math.max(state.maxSeenSegment, segment);
    state.segmentFractions[segment] = Math.max(previous, fraction);

    if (stage === "done") state.completedSegments[segment] = true;
    const completed = completedSegmentCount(state);
    if (completed >= state.total) {
        state.phase = "assembling";
        state.seg = state.total;
        state.step = 0;
        state.stepMax = 0;
        return true;
    }

    state.status = "running";
    if (state.outOfOrder) {
        state.phase = "segments";
        state.seg = Math.max(state.seg, segment);
        state.step = 0;
        state.stepMax = 0;
        return true;
    }

    if (stage === "done") {
        state.seg = Math.min(state.total, segment + 1);
        state.phase = "sample1";
        state.step = 0;
        state.stepMax = 0;
        return true;
    }

    state.seg = segment;
    state.phase = eventPhase(state, detail, stage) || state.phase;
    if (stage === "sampling") {
        state.step = Number(detail.step || 0);
        state.stepMax = Number(detail.step_total || 0);
    } else {
        state.step = 0;
        state.stepMax = 0;
    }
    return true;
}

export function progressPercent(state) {
    ensureProgressState(state);
    if (state.status === "done" || state.phase === "done") return 100;
    let fractionSum = 0;
    for (let index = 1; index <= state.total; index += 1) {
        fractionSum += clamp01(state.segmentFractions[index] || 0);
    }
    let percent = (fractionSum / state.total) * SEGMENT_PROGRESS_WEIGHT;
    if (state.phase === "assembling") percent = Math.max(percent, ASSEMBLING_PERCENT);
    if (state.phase === "encoding") percent = Math.max(percent, ENCODING_PERCENT);
    state.maxPercent = Math.max(state.maxPercent, percent);
    return Math.min(100, state.maxPercent);
}

export function finishProgress(state) {
    ensureProgressState(state);
    state.status = "done";
    state.phase = "done";
    state.seg = state.total;
    state.step = state.stepMax || 0;
    state.maxPercent = 100;
    return state;
}

export function failProgress(state, message) {
    ensureProgressState(state);
    state.status = "error";
    state.error = String(message || "执行中断");
    return state;
}
