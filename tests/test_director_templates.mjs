import assert from "node:assert/strict";
import {webcrypto} from "node:crypto";
import {readFile} from "node:fs/promises";

globalThis.crypto ||= webcrypto;

// This repository is loaded as classic JavaScript by ComfyUI, while Node's
// standalone test runner otherwise interprets .js as CommonJS. Import the
// browser module through a data URL so the same source is parsed as ESM.
const source = await readFile(new URL("../web/h3_director_templates.js", import.meta.url), "utf8");
const {
    DIRECTOR_TEMPLATE_FORMAT,
    createDirectorTemplateDocument,
    decryptDirectorTemplateDocument,
    directorTemplateInputSlots,
    encryptDirectorTemplateDocument,
    instantiateDirectorTemplate,
} = await import(`data:text/javascript;base64,${Buffer.from(source).toString("base64")}`);

const template = createDirectorTemplateDocument({
    name: "草莓模板",
    shots: [{
        id: "shot_a", enabled: true, brief: "发现草莓", duration_seconds: 5,
        transition: "开场", prompt: "integrated_multimodal_description: 她看向草莓。",
        asset_mode: "仅本镜头",
        assets: [{id: "hero", kind: "image", label: "主角", file: {name: "hero.png"}}],
    }],
    globalAssets: [], promptModes: {shot_a: "input"},
    promptInputLabels: {shot_a: "所需替换的剧情提示词"},
    materialModes: {"shot:shot_a:hero": "input"},
    materialInputLabels: {"shot:shot_a:hero": "所需替换的角色"},
    settings: {
        resolution: "640P", "二采分辨率": "832P", aspect_ratio: "9:16",
        "二采开启": true,
    },
    settingModes: {
        resolution: "fixed", "二采分辨率": "fixed", aspect_ratio: "local",
        "二采开启": "local",
    },
});

assert.equal(template.format, DIRECTOR_TEMPLATE_FORMAT);
assert.equal(template.cards[0].title, "发现草莓");
assert.equal(template.cards[0].prompt, "");
assert.equal(directorTemplateInputSlots(template).length, 2);
assert.equal(directorTemplateInputSlots(template)[0].label, "输入1：所需替换的剧情提示词");
assert.equal(directorTemplateInputSlots(template)[1].label, "输入2：所需替换的角色");
assert.deepEqual(template.settings, {resolution: "640P", "二采分辨率": "832P"});
assert.equal(template.settings_policy.aspect_ratio, "local");

const applied = instantiateDirectorTemplate(template, {
    prompts: {"prompt:1": "integrated_multimodal_description: @图片1 举起草莓。"},
    materials: {"shot:shot_a:hero": {
        kind: "image", label: "新主角", file: {name: "new.png", subfolder: "refs"},
    }},
});
assert.equal(applied.shots[0].brief, "发现草莓");
assert.equal(applied.shots[0].prompt.includes("发现草莓"), false);
assert.equal(applied.shots[0].assets[0].file.name, "new.png");
assert.equal(applied.shots[0].assets[0].slot_id, "shot:shot_a:hero");
assert.equal(applied.settings.resolution, "640P");
assert.equal(applied.settings["二采分辨率"], "832P");
assert.equal(applied.settings.aspect_ratio, undefined);

const transferTask = "动作迁移（跟随参考视频）";
const transferTemplate = createDirectorTemplateDocument({
    name: "动作替换模板", taskMode: transferTask,
    shots: [{
        id: "transfer", brief: "全片动作", prompt: "@视频1 的动作由 @图片1 执行。",
        assets: [
            {id: "motion", kind: "video", role: "action", label: "动作源", file: {name: "motion.mp4"},
                reference_weight_mode: "manual", reference_weight: 1.65},
            {id: "hero", kind: "image", label: "目标人物", file: {name: "hero.png"}},
        ],
    }],
    promptModes: {transfer: "fixed"},
    materialModes: {
        "shot:transfer:motion": "input",
        "shot:transfer:hero": "input",
    },
});
const transferSlots = directorTemplateInputSlots(transferTemplate);
assert.equal(transferTemplate.task_mode, transferTask);
assert.equal(transferTemplate.cards.length, 1);
assert.equal(transferTemplate.cards[0].title, "");
assert.deepEqual(transferTemplate.duration_policy.total, {mode: "reference", value: 0});
assert.equal(transferSlots.length, 2);
assert.equal(transferSlots[0].role, "action");
assert.equal(transferSlots[0].reference_weight_mode, "manual");
assert.equal(transferSlots[0].reference_weight, 1.65);
const appliedTransfer = instantiateDirectorTemplate(transferTemplate, {materials: {
    "shot:transfer:motion": {
        kind: "video", role: "action", label: "新动作", file: {name: "new-motion.mp4"},
    },
    "shot:transfer:hero": {
        kind: "image", label: "新人物", file: {name: "new-hero.png"},
    },
}});
assert.equal(appliedTransfer.taskMode, transferTask);
assert.equal(appliedTransfer.shots[0].brief, "动作迁移");
assert.equal(appliedTransfer.shots[0].assets[0].file.name, "new-motion.mp4");
assert.equal(appliedTransfer.shots[0].assets[0].reference_weight_mode, "manual");
assert.equal(appliedTransfer.shots[0].assets[0].reference_weight, 1.65);
assert.equal(appliedTransfer.settings.total_seconds, undefined);
assert.equal(appliedTransfer.durationContract.total_seconds, 0);

const lockedTemplate = createDirectorTemplateDocument({
    name: "接口模板", shots: [{id: "locked", prompt: "固定提示词", assets: []}],
    settings: {total_seconds: 12, segment_seconds: 6},
    durationPolicy: {
        total: {mode: "fixed", value: 12},
        segment: {mode: "input", value: 6},
    },
    interfaceLocked: true,
});
assert.deepEqual(directorTemplateInputSlots(lockedTemplate).map((slot) => slot.slot_id),
    ["duration:segment"]);
const lockedApplied = instantiateDirectorTemplate(lockedTemplate, {
    durations: {"duration:segment": 4.5},
});
assert.equal(lockedApplied.settings.total_seconds, 12);
assert.equal(lockedApplied.settings.segment_seconds, 4.5);
assert.equal(lockedApplied.interfaceLocked, true);

const encrypted = await encryptDirectorTemplateDocument(lockedTemplate, "test-password-123");
assert.equal(encrypted.ciphertext.includes("固定提示词"), false);
const decrypted = await decryptDirectorTemplateDocument(encrypted, "test-password-123");
assert.equal(decrypted.cards[0].prompt, "固定提示词");

const portableTemplate = structuredClone(lockedTemplate);
portableTemplate.bundle = {version: 1, assets: {
    hero: {encoding: "base64", mime: "image/png", name: "hero.png", size: 3, data: "AQID"},
}};
const encryptedPortable = await encryptDirectorTemplateDocument(
    portableTemplate, "portable-password-123");
const decryptedPortable = await decryptDirectorTemplateDocument(
    encryptedPortable, "portable-password-123");
assert.equal(decryptedPortable.bundle.assets.hero.name, "hero.png");

console.log("PASS reusable Director and action-transfer template slots");
