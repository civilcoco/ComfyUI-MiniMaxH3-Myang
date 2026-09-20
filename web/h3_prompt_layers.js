// Layer composition and speech budgeting for the Director's storyboard cards.
//
// SPDX-License-Identifier: GPL-3.0-only
//
// This is a deliberate port of two Python units, and the port exists because
// the card editor has to answer "does this line fit?" and "what will the
// finished prompt look like?" while the operator types, with no round trip:
//
//   compose_segment_prompt   -> composeSegmentPrompt   (nodes.py)
//   speech_units/budget_units -> speechUnits/budgetUnits (dialogue_audit.py)
//
// Both sides are pinned to the same fixtures by tests/test_prompt_layers.mjs,
// so a change to the composition order or the speaking rates that lands on one
// side and not the other fails the suite rather than quietly making the card
// preview disagree with what actually gets rendered.

const HAN = /[一-鿿㐀-䶿]/g;
const KANA = /[぀-ゟ゠-ヿ]/g;
const HANGUL = /[가-힯ᄀ-ᇿ]/g;

// (min, max) 字/s, matching dialogue_audit.SPEECH_RATES.
export const SPEECH_RATES = {
    excited: [3.5, 6.0],
    calm: [2.5, 4.0],
    casual: [1.5, 3.0],
};
export const DEFAULT_TONE = "calm";

const EMPTY_SOUND = new Set(["", "无", "none", "null", "n/a", "无台词", "无音效"]);

function clean(value) {
    return String(value ?? "").replace(/\s+/g, " ").trim();
}

function stripDialogueTags(value) {
    return String(value ?? "").replace(/<\/?d>/g, "").trim();
}

function latinSyllables(text) {
    let total = 0;
    for (const word of String(text ?? "").match(/[A-Za-z]+/g) || []) {
        const lowered = word.toLowerCase();
        let groups = (lowered.match(/[aeiouy]+/g) || []).length;
        if (lowered.endsWith("e") && groups > 1
            && !/(le|ee|ye)$/.test(lowered)) groups -= 1;
        total += Math.max(1, groups);
    }
    return total + (String(text ?? "").match(/\d/g) || []).length;
}

/** Spoken beats in a line: CJK characters, or vowel groups for Latin script. */
export function speechUnits(value) {
    const text = String(value ?? "");
    const cjk = (text.match(HAN) || []).length
        + (text.match(KANA) || []).length
        + (text.match(HANGUL) || []).length;
    if (cjk > 0) return cjk + latinSyllables(text);
    const latin = latinSyllables(text);
    return latin > 0 ? latin : text.replace(/\s+/g, "").length;
}

/** Most beats that fit in a window; `fastest` false gives the comfortable fill. */
export function budgetUnits(seconds, tone = DEFAULT_TONE, fastest = true) {
    const [low, high] = SPEECH_RATES[tone] || SPEECH_RATES[DEFAULT_TONE];
    const rate = fastest ? high : low;
    return Math.max(1, Math.floor(Math.max(0, Number(seconds) || 0) * rate));
}

/** Fastest delivery time of one dialogue entry, in seconds. */
export function dialogueSeconds(entry) {
    const text = stripDialogueTags(clean(
        entry && typeof entry === "object" ? entry.text : entry));
    if (!text) return 0;
    const tone = entry && typeof entry === "object" ? clean(entry.tone) : "";
    const [, high] = SPEECH_RATES[tone] || SPEECH_RATES[DEFAULT_TONE];
    return speechUnits(text) / high;
}

function subjectLines(subjects) {
    const lines = [];
    for (const entry of subjects || []) {
        if (typeof entry === "string") {
            if (clean(entry)) lines.push(clean(entry));
            continue;
        }
        if (!entry || typeof entry !== "object") continue;
        const body = [clean(entry.appearance), clean(entry.wardrobe)]
            .filter(Boolean).join("，");
        let head = [clean(entry.name), body].filter(Boolean).join("：");
        const tag = clean(entry.ref_tag);
        if (tag) head = head ? `${head} ${tag}` : tag;
        if (head) lines.push(head);
    }
    return lines;
}

function timelineLines(timeline) {
    const lines = [];
    for (const entry of timeline || []) {
        if (typeof entry === "string") {
            if (clean(entry)) lines.push(clean(entry));
            continue;
        }
        if (!entry || typeof entry !== "object") continue;
        const beat = clean(entry.beat || entry.time_range);
        const body = [clean(entry.camera || entry.camera_movement),
                      clean(entry.action || entry.content)]
            .filter(Boolean).join("，");
        if (!body) continue;
        lines.push(beat ? `${beat} ${body}` : body);
    }
    return lines;
}

function soundLines(sound) {
    if (typeof sound === "string") {
        const text = clean(sound);
        return text && !EMPTY_SOUND.has(text.toLowerCase()) ? [text] : [];
    }
    if (!sound || typeof sound !== "object") return [];
    const lines = [];
    for (const [key, label] of [["ambient", "环境音"], ["bgm", "背景音乐"]]) {
        const text = clean(sound[key]);
        if (text && !EMPTY_SOUND.has(text.toLowerCase())) lines.push(`${label}：${text}`);
    }
    const raw = typeof sound.sfx === "string" ? [sound.sfx] : (sound.sfx || []);
    const named = raw.map(clean)
        .filter((item) => item && !EMPTY_SOUND.has(item.toLowerCase()));
    if (named.length) lines.push(`音效：${named.join("、")}`);
    return lines;
}

function dialogueLines(dialogue) {
    const lines = [];
    for (const entry of dialogue || []) {
        if (typeof entry === "string") {
            const text = stripDialogueTags(clean(entry));
            if (text) lines.push(`<d>${text}</d>`);
            continue;
        }
        if (!entry || typeof entry !== "object") continue;
        const text = stripDialogueTags(clean(entry.text));
        if (!text) continue;
        const tone = clean(entry.tone);
        const prefix = clean(entry.speaker) + (tone ? `（${tone}）` : "");
        lines.push(`${prefix}<d>${text}</d>`);
    }
    return lines;
}

function stripDialogueProse(value) {
    return String(value ?? "").replace(/<d>[\s\S]*?<\/d>/g, "")
        .split("\n")
        .map((line) => line.replace(/\s+/g, " ").trim())
        .filter((line) => line && /[\w一-鿿]/.test(line))
        .join("\n");
}

/**
 * Flatten authored layers into one Easy Prompt, mirroring the Python composer.
 *
 * Order is fixed because the tokenizer sees the concatenation: style and scene,
 * then who is in frame, then what happens, then what is heard, then what is
 * said. A `visual` layer is the writer's finished body and replaces the
 * structured head instead of being appended to it; its inline `<d>` lines drop
 * out because the dialogue layer is the budgeted rewrite of exactly those.
 */
export function composeSegmentPrompt(globalLayers, layers) {
    const shared = globalLayers && typeof globalLayers === "object" ? globalLayers : {};
    const own = layers && typeof layers === "object" ? layers : {};
    const spoken = dialogueLines(own.dialogue);
    const visual = String(own.visual ?? "").trim();
    let blocks;
    if (visual) {
        blocks = [spoken.length ? stripDialogueProse(visual) : visual];
    } else {
        blocks = [clean(shared.style), clean(shared.scene)];
        blocks.push(...subjectLines(own.subjects_override || shared.subjects));
        blocks.push(...timelineLines(own.timeline));
    }
    blocks.push(...soundLines(own.sound));
    blocks.push(...spoken);
    return blocks.filter(Boolean).join("\n");
}

/** Style, scene and the cast every shot in a plan inherits. */
export function normalizeGlobalLayers(raw) {
    if (!raw || typeof raw !== "object" || Array.isArray(raw)) return {};
    const subjects = (Array.isArray(raw.subjects) ? raw.subjects : []).map((entry) => {
        if (typeof entry === "string") return entry.trim();
        if (!entry || typeof entry !== "object") return "";
        return {
            name: String(entry.name || ""),
            appearance: String(entry.appearance || ""),
            wardrobe: String(entry.wardrobe || ""),
            ref_tag: String(entry.ref_tag || ""),
        };
    }).filter((entry) => (typeof entry === "string"
        ? entry : entry.name || entry.appearance || entry.ref_tag));
    const layers = {};
    if (clean(raw.style)) layers.style = String(raw.style);
    if (clean(raw.scene)) layers.scene = String(raw.scene);
    if (subjects.length) layers.subjects = subjects;
    return layers;
}

/**
 * The per-shot half of a layered prompt, or null when it carries nothing.
 *
 * Returning null for an empty set matters: the card editor uses "has layers" as
 * the switch between editing the flat prompt and editing the visual layer, so an
 * all-empty object would flip every legacy card into layered mode and blank the
 * prose it actually holds.
 */
export function normalizeSegmentLayers(raw) {
    if (!raw || typeof raw !== "object" || Array.isArray(raw)) return null;
    const lines = (value) => (Array.isArray(value) ? value : [])
        .map((entry) => (typeof entry === "string" ? entry.trim() : entry))
        .filter(Boolean);
    const rawSound = raw.sound && typeof raw.sound === "object"
        && !Array.isArray(raw.sound) ? raw.sound : {};
    const sound = {
        ambient: String(rawSound.ambient || ""),
        bgm: String(rawSound.bgm || ""),
        sfx: (Array.isArray(rawSound.sfx) ? rawSound.sfx : [])
            .map((item) => String(item).trim()).filter(Boolean),
    };
    const dialogue = (Array.isArray(raw.dialogue) ? raw.dialogue : [])
        .map((entry) => (typeof entry === "string"
            ? {speaker: "", tone: DEFAULT_TONE, text: entry}
            : {
                speaker: String(entry?.speaker || ""),
                tone: SPEECH_RATES[entry?.tone] ? String(entry.tone) : DEFAULT_TONE,
                text: stripDialogueTags(String(entry?.text || "")),
            }))
        .filter((entry) => entry.text.trim());
    const layers = {
        visual: String(raw.visual || ""),
        subjects_override: lines(raw.subjects_override),
        timeline: lines(raw.timeline),
        sound,
        dialogue,
    };
    const populated = layers.visual.trim() || layers.subjects_override.length
        || layers.timeline.length || dialogue.length
        || sound.ambient || sound.bgm || sound.sfx.length;
    return populated ? layers : null;
}
