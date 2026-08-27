export const STORYBOARD_CARD_FORMAT = "minimax-h3-myang-director-storyboard";
export const STORYBOARD_CARD_VERSION = 1;

const MAX_CARDS = 256;

function text(value) {
    return String(value ?? "");
}

function duration(value) {
    return Math.max(0.2, Math.min(30, Number(value) || 5));
}

function material(asset, index) {
    const file = asset?.file && typeof asset.file === "object" ? asset.file : {};
    const kind = ["image", "video", "audio"].includes(asset?.kind) ? asset.kind : "image";
    const name = text(file.name).trim();
    if (!name) return null;
    return {
        kind,
        role: kind === "video" && asset?.role === "action" ? "action" : "reference",
        label: text(asset?.label || name || `${kind} ${index + 1}`),
        file: {
            name,
            subfolder: text(file.subfolder),
            type: "input",
        },
    };
}

function materials(source) {
    return Array.isArray(source)
        ? source.map(material).filter(Boolean)
        : [];
}

function exportedCard(shot, index) {
    return {
        order: index + 1,
        enabled: shot?.enabled !== false,
        duration_seconds: duration(shot?.duration_seconds),
        title: text(shot?.brief || `镜头 ${index + 1}`),
        transition: text(shot?.transition || (index === 0 ? "开场" : "承接")),
        prompt: text(shot?.prompt),
        origin: shot?.imported_storyboard === true
            ? "imported_storyboard"
            : shot?.fixed_from_plan === true ? "llm_plan" : "manual",
        material_policy: shot?.asset_mode === "叠加全局素材" ? "叠加全局素材" : "仅本镜头",
        materials: materials(shot?.assets),
    };
}

export function createStoryboardCardDocument({shots, globalAssets, title, plan} = {}) {
    const cards = Array.isArray(shots) ? shots.slice(0, MAX_CARDS).map(exportedCard) : [];
    if (!cards.length) throw new Error("没有可导出的导演台分镜卡");
    return {
        format: STORYBOARD_CARD_FORMAT,
        version: STORYBOARD_CARD_VERSION,
        exported_at: new Date().toISOString(),
        storyboard: {
            title: text(title || "沐阳 H3 导演台分镜卡"),
            card_count: cards.length,
            cards,
            global_materials: materials(globalAssets),
            metadata: {
                source: text(plan?.source || "manual_storyboard"),
                style_header: text(plan?.style_header),
                skill_source: text(plan?.skill_source),
            },
        },
    };
}

function importedCard(card, index, stamp) {
    if (!card || typeof card !== "object" || Array.isArray(card)) {
        throw new Error(`第 ${index + 1} 张分镜卡不是有效对象`);
    }
    return {
        id: `shot_import_${stamp}_${index + 1}`,
        enabled: card.enabled !== false,
        duration_seconds: duration(card.duration_seconds),
        brief: text(card.title || `镜头 ${index + 1}`),
        prompt: text(card.prompt),
        transition: text(card.transition || (index === 0 ? "开场" : "承接")),
        fixed_from_plan: card.origin === "llm_plan",
        imported_storyboard: true,
        asset_mode: card.material_policy === "叠加全局素材" ? "叠加全局素材" : "仅本镜头",
        assets: materials(card.materials).map((asset, assetIndex) => ({
            ...asset,
            id: `${asset.kind}_${stamp}_${index + 1}_${assetIndex + 1}`,
        })),
    };
}

export function parseStoryboardCardDocument(source) {
    let value;
    try {
        value = typeof source === "string" ? JSON.parse(source) : source;
    } catch (error) {
        throw new Error(`文件不是有效 JSON：${error.message}`);
    }
    if (!value || typeof value !== "object" || Array.isArray(value)) {
        throw new Error("文件不是导演台分镜卡对象");
    }
    if (value.format !== STORYBOARD_CARD_FORMAT) {
        throw new Error("这不是沐阳 H3 导演台分镜卡文件");
    }
    if (Number(value.version) !== STORYBOARD_CARD_VERSION) {
        throw new Error(`不支持的分镜卡版本：${text(value.version || "未知")}`);
    }
    const storyboard = value.storyboard;
    const cards = storyboard?.cards;
    if (!Array.isArray(cards) || !cards.length) {
        throw new Error("分镜卡文件中没有镜头卡");
    }
    if (cards.length > MAX_CARDS) {
        throw new Error(`分镜卡数量超过上限 ${MAX_CARDS}`);
    }
    const stamp = Date.now().toString(36);
    return {
        shots: cards.map((card, index) => importedCard(card, index, stamp)),
        globalAssets: materials(storyboard.global_materials).map((asset, index) => ({
            ...asset,
            id: `${asset.kind}_${stamp}_global_${index + 1}`,
        })),
        metadata: {
            title: text(storyboard.title),
            source: text(storyboard.metadata?.source),
            style_header: text(storyboard.metadata?.style_header),
            skill_source: text(storyboard.metadata?.skill_source),
        },
    };
}

export function storyboardCardFileName(title = "") {
    const base = text(title || "H3导演台分镜卡")
        .replace(/[\\/:*?"<>|\u0000-\u001f]/g, "_")
        .replace(/\s+/g, " ")
        .trim()
        .slice(0, 60) || "H3导演台分镜卡";
    const date = new Date().toISOString().slice(0, 10);
    return `${base}_${date}.h3storyboard.json`;
}
