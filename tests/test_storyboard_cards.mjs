import assert from "node:assert/strict";
import {
    STORYBOARD_CARD_FORMAT,
    STORYBOARD_CARD_VERSION,
    createStoryboardCardDocument,
    parseStoryboardCardDocument,
    storyboardCardFileName,
} from "../web/h3_storyboard_cards.js";

const source = {
    shots: [
        {
            enabled: true,
            duration_seconds: 5.4,
            brief: "开场近景",
            transition: "开场",
            prompt: "integrated_multimodal_description: [Shot 1] @图片1 抬头。",
            fixed_from_plan: true,
            asset_mode: "叠加全局素材",
            assets: [{
                kind: "image", role: "reference", label: "主角",
                file: {name: "hero.png", subfolder: "characters", type: "input"},
            }],
        },
        {
            enabled: false,
            duration_seconds: 7,
            brief: "结尾",
            transition: "承接",
            prompt: "integrated_multimodal_description: [Shot 1] 她走入森林。",
            asset_mode: "仅本镜头",
            assets: [],
        },
    ],
    globalAssets: [{
        kind: "audio", role: "reference", label: "配乐",
        file: {name: "music.wav", subfolder: "audio", type: "input"},
    }],
    title: "森林短片",
    plan: {source: "media_agent_writer", style_header: "电影感", skill_source: "h3-prompt-writing"},
};

const documentData = createStoryboardCardDocument(source);
assert.equal(documentData.format, STORYBOARD_CARD_FORMAT);
assert.equal(documentData.version, STORYBOARD_CARD_VERSION);
assert.equal(documentData.storyboard.card_count, 2);
assert.equal(documentData.storyboard.cards[0].materials[0].file.name, "hero.png");
assert.equal(documentData.storyboard.global_materials[0].file.name, "music.wav");
assert.equal(documentData.storyboard.metadata.skill_source, "h3-prompt-writing");

const imported = parseStoryboardCardDocument(JSON.stringify(documentData));
assert.equal(imported.shots.length, 2);
assert.equal(imported.shots[0].brief, "开场近景");
assert.equal(imported.shots[0].prompt, source.shots[0].prompt);
assert.equal(imported.shots[0].duration_seconds, 5.4);
assert.equal(imported.shots[0].asset_mode, "叠加全局素材");
assert.equal(imported.shots[0].assets[0].file.subfolder, "characters");
assert.equal(imported.shots[1].enabled, false);
assert.equal(imported.globalAssets[0].file.name, "music.wav");
assert.equal(imported.metadata.style_header, "电影感");
assert.equal(imported.shots.every((shot) => shot.imported_storyboard), true);
assert.notEqual(imported.shots[0].id, imported.shots[1].id);

assert.throws(
    () => parseStoryboardCardDocument('{"shots":[]}'),
    /不是沐阳 H3 导演台分镜卡文件/,
);
assert.throws(
    () => parseStoryboardCardDocument(JSON.stringify({
        format: STORYBOARD_CARD_FORMAT,
        version: 99,
        storyboard: {cards: [{}]},
    })),
    /不支持的分镜卡版本/,
);
assert.throws(() => parseStoryboardCardDocument("not json"), /不是有效 JSON/);

const fileName = storyboardCardFileName('森林:短片?');
assert.match(fileName, /^森林_短片__\d{4}-\d{2}-\d{2}\.h3storyboard\.json$/);

console.log("PASS structured storyboard card export/import round trip");
