export const DIRECTOR_TEMPLATE_FORMAT = "minimax-h3-myang-director-template";
export const DIRECTOR_TEMPLATE_VERSION = 2;
export const DIRECTOR_TEMPLATE_ENCRYPTED_FORMAT = "minimax-h3-myang-director-template-encrypted";
export const DIRECTOR_TEMPLATE_ENCRYPTED_VERSION = 1;

const SUPPORTED_VERSIONS = new Set([1, DIRECTOR_TEMPLATE_VERSION]);

const KINDS = new Set(["image", "video", "audio"]);
const TASK_MODES = new Set([
    "纯生成（不用参考视频）",
    "动作迁移（跟随参考视频）",
    "视频续写（接着往下演）",
]);
const DEFAULT_TASK_MODE = "纯生成（不用参考视频）";
const TRANSFER_TASK_MODE = "动作迁移（跟随参考视频）";

function text(value, maximum = 120000) {
    return String(value ?? "").trim().slice(0, maximum);
}

function material(value, fallback, defaultMode = "input") {
    const kind = text(value?.kind, 16).toLowerCase();
    if (!KINDS.has(kind)) return null;
    const mode = value?.mode === "fixed" ? "fixed" : defaultMode;
    const file = value?.file && typeof value.file === "object" ? value.file : {};
    if (mode === "fixed" && !text(file.name, 260)) return null;
    const result = {
        slot_id: text(value?.slot_id, 100) || fallback,
        mode,
        kind,
        role: kind === "video" && value?.role === "action" ? "action" : "reference",
        label: text(value?.label || file.name || "参考素材", 120),
        input_label: text(value?.input_label || value?.label || file.name || "参考素材", 120),
        required: value?.required !== false,
    };
    if (["auto", "manual"].includes(value?.reference_weight_mode)) {
        result.reference_weight_mode = value.reference_weight_mode;
        result.reference_weight = Math.max(.25, Math.min(3,
            Number(value?.reference_weight) || 1));
    }
    if (mode === "fixed") result.file = {
        name: text(file.name, 260), subfolder: text(file.subfolder, 260), type: "input",
    };
    return result;
}

function durationRule(value, fallbackValue = 5) {
    const mode = value?.mode === "reference"
        ? "reference" : value?.mode === "fixed" ? "fixed" : "input";
    if (mode === "reference") return {mode, value: 0};
    return {
        mode,
        value: Math.max(.2, Math.min(3600, Number(value?.value) || fallbackValue)),
    };
}

export function createDirectorTemplateDocument({
    name, description = "", shots = [], globalAssets = [], settings = {},
    settingsFixed = false, settingModes = {}, promptModes = {}, promptInputLabels = {}, materialModes = {},
    materialInputLabels = {},
    taskMode = DEFAULT_TASK_MODE, durationPolicy = null, interfaceLocked = false,
}) {
    if (!text(name, 120)) throw new Error("模板名称不能为空");
    if (!Array.isArray(shots) || !shots.length) throw new Error("模板至少需要一张分镜卡");
    const globalMaterials = globalAssets.map((asset, index) => material({
        ...asset,
        mode: materialModes[`global:${asset.id}`] || "input",
        slot_id: `global:${asset.id || index + 1}`,
        input_label: materialInputLabels[`global:${asset.id}`],
    }, `global:${index + 1}`)).filter(Boolean);
    const templateShots = taskMode === TRANSFER_TASK_MODE ? shots.slice(0, 1) : shots;
    const cards = templateShots.map((shot, index) => ({
        order: index + 1,
        enabled: shot.enabled !== false,
        // Action transfer owns one full-film prompt/material card, not a
        // storyboard shot. Do not persist a stale brief as a shot title.
        title: taskMode === TRANSFER_TASK_MODE
            ? "" : text(shot.brief || `分镜${index + 1}`, 120),
        duration_seconds: Math.max(.2, Math.min(30, Number(shot.duration_seconds) || 5)),
        transition: index === 0 ? "开场" : (shot.transition === "切镜" ? "切镜" : "承接"),
        prompt_mode: promptModes[shot.id] === "input" ? "input" : "fixed",
        prompt: promptModes[shot.id] === "input" ? "" : text(shot.prompt),
        prompt_label: text(promptInputLabels[shot.id]
            || (taskMode === TRANSFER_TASK_MODE
                ? "动作迁移提示词"
                : `所需替换的${shot.brief || `分镜${index + 1}`}提示词`), 120),
        material_policy: shot.asset_mode === "叠加全局素材" ? "叠加全局素材" : "仅本镜头",
        materials: (shot.assets || []).map((asset, materialIndex) => material({
            ...asset,
            mode: materialModes[`shot:${shot.id}:${asset.id}`] || "input",
            slot_id: `shot:${shot.id}:${asset.id || materialIndex + 1}`,
            input_label: materialInputLabels[`shot:${shot.id}:${asset.id}`],
        }, `card:${index + 1}:${materialIndex + 1}`)).filter(Boolean),
    }));
    if (taskMode === TRANSFER_TASK_MODE) {
        const actionVideos = cards.flatMap((card) => card.materials)
            .filter((entry) => entry.kind === "video" && entry.role === "action");
        if (actionVideos.length > 1) {
            throw new Error("动作迁移模板只能包含一个动作参考视频槽");
        }
        if (!actionVideos.length) {
            cards[0].materials.unshift({
                slot_id: "action:reference-video", mode: "input", kind: "video",
                role: "action", label: "动作参考视频", input_label: "所需替换的动作参考视频",
                required: true,
            });
        } else {
            actionVideos[0].required = true;
        }
    }
    const normalizedSettingPolicy = {};
    const fixedSettings = {};
    for (const [setting, value] of Object.entries(settings || {})) {
        const mode = settingsFixed || settingModes[setting] === "fixed" ? "fixed" : "local";
        normalizedSettingPolicy[setting] = mode;
        if (mode === "fixed") fixedSettings[setting] = value;
    }
    return {
        format: DIRECTOR_TEMPLATE_FORMAT,
        version: DIRECTOR_TEMPLATE_VERSION,
        name: text(name, 120),
        description: text(description, 1200),
        task_mode: TASK_MODES.has(taskMode) ? taskMode : DEFAULT_TASK_MODE,
        cards,
        global_materials: globalMaterials,
        settings: fixedSettings,
        settings_fixed: Object.keys(fixedSettings).length > 0,
        settings_policy: normalizedSettingPolicy,
        duration_policy: {
            // Action transfer is paced by the decoded action-reference video.
            // A saved total duration would silently truncate a longer source
            // and collapse automatic segmentation to a single shot.
            total: taskMode === TRANSFER_TASK_MODE
                ? {mode: "reference", value: 0}
                : durationRule(durationPolicy?.total || {mode: "fixed"},
                Number(settings.total_seconds) || cards.reduce(
                    (sum, card) => sum + card.duration_seconds, 0)),
            segment: durationRule(durationPolicy?.segment || {mode: "fixed"},
                Number(settings.segment_seconds) || 5),
        },
        interface_locked: interfaceLocked === true,
    };
}

export function parseDirectorTemplateDocument(value) {
    const source = typeof value === "string" ? JSON.parse(value) : value;
    if (!source || source.format !== DIRECTOR_TEMPLATE_FORMAT
        || !SUPPORTED_VERSIONS.has(Number(source.version))) {
        throw new Error("不是受支持的沐阳导演台模板");
    }
    if (!Array.isArray(source.cards) || !source.cards.length) throw new Error("模板没有分镜卡");
    return source;
}

export function directorTemplateInputSlots(template) {
    const source = parseDirectorTemplateDocument(template);
    const slots = [];
    let interfaceIndex = 0;
    const appendInterface = (slot) => {
        interfaceIndex += 1;
        const purpose = text(slot.input_label || slot.label || "模板输入", 120);
        slots.push({...slot, input_index: interfaceIndex,
            input_label: purpose, label: `输入${interfaceIndex}：${purpose}`});
    };
    const policy = source.duration_policy || {};
    if (policy.total?.mode === "input") slots.push({
        type: "duration", slot_id: "duration:total", label: "目标总时长",
        setting: "total_seconds", min: 1, max: 3600, step: 1,
    });
    if (policy.segment?.mode === "input") slots.push({
        type: "duration", slot_id: "duration:segment", label: "智能分段上限",
        setting: "segment_seconds", min: .2, max: 30, step: .1,
    });
    for (const entry of source.global_materials || []) if (entry.mode === "input") appendInterface({
        type: "material", card_index: -1, ...entry,
    });
    source.cards.forEach((card, index) => {
        if (card.prompt_mode === "input") appendInterface({
            type: "prompt", slot_id: `prompt:${index + 1}`,
            label: card.prompt_label || `分镜${index + 1}提示词`,
            input_label: card.prompt_label || `分镜${index + 1}提示词`, card_index: index,
        });
        for (const entry of card.materials || []) if (entry.mode === "input") appendInterface({
            type: "material", card_index: index, ...entry,
        });
    });
    return slots;
}

export function instantiateDirectorTemplate(template, inputs = {}) {
    const source = parseDirectorTemplateDocument(template);
    const prompts = inputs.prompts || {};
    const materials = inputs.materials || {};
    const durations = inputs.durations || {};
    const resolveMaterial = (slot) => {
        if (slot.mode === "fixed") return material(slot, slot.slot_id, "fixed");
        const supplied = materials[slot.slot_id];
        if (!supplied && slot.required !== false) throw new Error(`还没有填写素材槽：${slot.label}`);
        return supplied ? material({
            ...slot, ...supplied, mode: "fixed", slot_id: slot.slot_id,
            role: supplied.role || slot.role, label: supplied.label || slot.label,
        }, slot.slot_id, "fixed") : null;
    };
    const globalAssets = (source.global_materials || []).map(resolveMaterial).filter(Boolean);
    const templateCards = source.task_mode === TRANSFER_TASK_MODE
        ? source.cards.slice(0, 1) : source.cards;
    const shots = templateCards.map((card, index) => {
        const prompt = card.prompt_mode === "input" ? text(prompts[`prompt:${index + 1}`]) : text(card.prompt);
        if (card.prompt_mode === "input" && !prompt) throw new Error(`还没有填写${card.prompt_label || `分镜${index + 1}提示词`}`);
        return {
            id: `shot_${Date.now().toString(36)}_${index + 1}`,
            enabled: card.enabled !== false,
            brief: source.task_mode === TRANSFER_TASK_MODE
                ? "动作迁移" : text(card.title || `分镜${index + 1}`, 120),
            duration_seconds: Math.max(.2, Math.min(30, Number(card.duration_seconds) || 5)),
            transition: index === 0 ? "开场" : (card.transition === "切镜" ? "切镜" : "承接"),
            prompt,
            asset_mode: card.material_policy === "叠加全局素材" ? "叠加全局素材" : "仅本镜头",
            assets: (card.materials || []).map(resolveMaterial).filter(Boolean),
            imported_storyboard: false,
            fixed_from_plan: false,
        };
    });
    const durationPolicy = source.duration_policy || {};
    const durationValue = (key, fallback) => {
        if (!durationPolicy[key]) {
            const legacyName = key === "total" ? "total_seconds" : "segment_seconds";
            return Math.max(.2, Number(source.settings?.[legacyName]) || fallback);
        }
        const rule = durationRule(durationPolicy[key], fallback);
        if (rule.mode === "reference") return 0;
        const supplied = Number(durations[`duration:${key}`]);
        if (rule.mode === "input" && !(supplied > 0)) {
            throw new Error(`还没有填写${key === "total" ? "目标总时长" : "智能分段上限"}`);
        }
        return rule.mode === "fixed" ? rule.value : supplied;
    };
    const hasSettingPolicy = source.settings_policy
        && typeof source.settings_policy === "object"
        && Object.keys(source.settings_policy).length > 0;
    const runtimeSettings = hasSettingPolicy
        ? Object.fromEntries(Object.entries(source.settings || {})
            .filter(([name]) => source.settings_policy[name] === "fixed"))
        : source.settings_fixed ? {...source.settings} : {};
    if (durationPolicy.total && durationPolicy.total.mode !== "reference") {
        runtimeSettings.total_seconds = durationValue(
        "total", shots.reduce((sum, shot) => sum + shot.duration_seconds, 0) || 5);
    } else if (source.task_mode === TRANSFER_TASK_MODE) {
        delete runtimeSettings.total_seconds;
    }
    if (durationPolicy.segment) {
        runtimeSettings.segment_seconds = durationValue("segment", 5);
    }
    return {
        taskMode: TASK_MODES.has(source.task_mode) ? source.task_mode : DEFAULT_TASK_MODE,
        shots,
        globalAssets,
        settings: runtimeSettings,
        interfaceLocked: source.interface_locked === true,
        templateId: text(source.id, 100),
        templateName: text(source.name, 120),
        durationContract: (durationPolicy.total || durationPolicy.segment) ? {
            active: true,
            total_seconds: durationPolicy.total?.mode === "reference"
                ? 0 : Number(runtimeSettings.total_seconds || 0),
            segment_seconds: Number(runtimeSettings.segment_seconds || 0),
        } : null,
    };
}

function bytesToBase64(bytes) {
    let binary = "";
    for (let index = 0; index < bytes.length; index += 0x8000) {
        binary += String.fromCharCode(...bytes.subarray(index, index + 0x8000));
    }
    return btoa(binary);
}

function base64ToBytes(value) {
    const binary = atob(String(value || ""));
    return Uint8Array.from(binary, (character) => character.charCodeAt(0));
}

async function passwordKey(password, salt, usage) {
    if (!globalThis.crypto?.subtle) throw new Error("当前浏览器不支持模板加密");
    const source = await crypto.subtle.importKey(
        "raw", new TextEncoder().encode(String(password)), "PBKDF2", false,
        ["deriveKey"]);
    return crypto.subtle.deriveKey({
        name: "PBKDF2", salt, iterations: 240000, hash: "SHA-256",
    }, source, {name: "AES-GCM", length: 256}, false, [usage]);
}

export async function encryptDirectorTemplateDocument(template, password) {
    const source = parseDirectorTemplateDocument(template);
    if (String(password || "").length < 8) throw new Error("加密密码至少需要 8 个字符");
    const salt = crypto.getRandomValues(new Uint8Array(16));
    const iv = crypto.getRandomValues(new Uint8Array(12));
    const key = await passwordKey(password, salt, "encrypt");
    const payload = new TextEncoder().encode(JSON.stringify(source));
    const ciphertext = new Uint8Array(await crypto.subtle.encrypt(
        {name: "AES-GCM", iv}, key, payload));
    return {
        format: DIRECTOR_TEMPLATE_ENCRYPTED_FORMAT,
        version: DIRECTOR_TEMPLATE_ENCRYPTED_VERSION,
        display_name: text(source.name, 120),
        encryption: {
            algorithm: "AES-GCM-256", kdf: "PBKDF2-SHA256",
            iterations: 240000, salt: bytesToBase64(salt), iv: bytesToBase64(iv),
        },
        ciphertext: bytesToBase64(ciphertext),
    };
}

export async function decryptDirectorTemplateDocument(value, password) {
    const source = typeof value === "string" ? JSON.parse(value) : value;
    if (!source || source.format !== DIRECTOR_TEMPLATE_ENCRYPTED_FORMAT
        || Number(source.version) !== DIRECTOR_TEMPLATE_ENCRYPTED_VERSION) {
        throw new Error("不是受支持的沐阳加密模板");
    }
    try {
        const salt = base64ToBytes(source.encryption?.salt);
        const iv = base64ToBytes(source.encryption?.iv);
        const key = await passwordKey(password, salt, "decrypt");
        const plain = await crypto.subtle.decrypt(
            {name: "AES-GCM", iv}, key, base64ToBytes(source.ciphertext));
        return parseDirectorTemplateDocument(new TextDecoder().decode(plain));
    } catch (_error) {
        throw new Error("密码错误或模板文件已损坏");
    }
}
