import assert from "node:assert/strict";
import {readFile} from "node:fs/promises";

// Same trick as test_storyboard_cards.mjs: ComfyUI loads this as a browser ES
// module, but Node treats a plain .js in this package as CommonJS.
const sourceText = await readFile(
    new URL("../web/h3_prompt_layers.js", import.meta.url), "utf8");
const {
    SPEECH_RATES,
    budgetUnits,
    composeSegmentPrompt,
    dialogueSeconds,
    speechUnits,
} = await import(`data:text/javascript;base64,${Buffer.from(sourceText).toString("base64")}`);

// --- speaking rates must match dialogue_audit.SPEECH_RATES exactly ----------
assert.deepEqual(SPEECH_RATES.excited, [3.5, 6.0]);
assert.deepEqual(SPEECH_RATES.calm, [2.5, 4.0]);
assert.deepEqual(SPEECH_RATES.casual, [1.5, 3.0]);

// --- beats: Han characters are one beat each, Latin counts vowel groups -----
// Expected values are taken from dialogue_audit.speech_units, not hand-counted:
// punctuation carries no duration, and "PLAYER" is one vowel group plus a digit.
assert.equal(speechUnits("你终于来了，我等了整整三年。"), 12);
assert.equal(speechUnits("PLAYER 1 准备好了吗"), 7);
assert.equal(speechUnits("we are ready"), 4);

// --- the budget the card shows is the budget the backend enforces -----------
assert.equal(budgetUnits(8, "calm"), 32);
assert.equal(budgetUnits(8, "excited"), 48);
assert.equal(budgetUnits(8, "calm", false), 20);
assert.equal(budgetUnits(0), 1, "a zero window still reports one beat, never zero");

// A 60 character calm line cannot be said in 8 seconds, and the card must agree
// with the Python audit about by how much.
assert.ok(dialogueSeconds({tone: "calm", text: "字".repeat(60)}) > 8);
assert.equal(dialogueSeconds({tone: "calm", text: "字".repeat(60)}), 15);
assert.equal(dialogueSeconds(""), 0);

// --- composition order is style/scene, cast, action, sound, speech ----------
const composed = composeSegmentPrompt({style: "电影级写实", scene: "黄昏码头"}, {
    subjects_override: [
        {name: "阿岚", appearance: "黑色风衣", wardrobe: "战术手套", ref_tag: "@图片1"},
    ],
    timeline: [{beat: "0-3s", camera: "缓慢推近", action: "她抬头看向货轮"}],
    sound: {ambient: "浪声", bgm: "低沉弦乐", sfx: ["汽笛", "脚步声"]},
    dialogue: [{speaker: "阿岚", tone: "calm", text: "走吧。"}],
});
assert.equal(composed, [
    "电影级写实",
    "黄昏码头",
    "阿岚：黑色风衣，战术手套 @图片1",
    "0-3s 缓慢推近，她抬头看向货轮",
    "环境音：浪声",
    "背景音乐：低沉弦乐",
    "音效：汽笛、脚步声",
    "阿岚（calm）<d>走吧。</d>",
].join("\n"));

// --- a visual body replaces the structured head and loses its inline <d> ----
const withVisual = composeSegmentPrompt({style: "电影级写实"}, {
    visual: "码头黄昏，主角站在集装箱前。\n阿岚<d>这句会被预算层替换</d>",
    dialogue: [{speaker: "阿岚", tone: "calm", text: "走吧。"}],
});
assert.ok(!withVisual.includes("这句会被预算层替换"),
    "the writer's inline dialogue survived next to the budgeted layer");
assert.ok(!withVisual.includes("电影级写实"),
    "a finished visual body must not be prefixed with the structured head");
assert.equal((withVisual.match(/<d>/g) || []).length, 1);

// Speaker and tone stay outside the tag: dialogue_audit bills whatever is
// inside it as spoken time.
assert.ok(withVisual.includes("阿岚（calm）<d>走吧。</d>"));

// --- placeholders are dropped rather than rendered as content --------------
assert.equal(composeSegmentPrompt({}, {
    visual: "只有画面。",
    sound: {ambient: "无", bgm: "", sfx: ["none"]},
}), "只有画面。");
assert.equal(composeSegmentPrompt({}, {}), "");

console.log("PASS prompt layer composition and speech budget match the backend");
