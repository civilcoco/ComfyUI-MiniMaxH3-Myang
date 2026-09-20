import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const testDir = dirname(fileURLToPath(import.meta.url));
const source = readFileSync(join(testDir, "..", "web", "h3_roughcut_ui.js"), "utf8");

const required = [
    "节目监视器",
    "删除片段",
    "deleteSelectedTimelineClip(editor)",
    "item.oncontextmenu",
    'key === "delete" || key === "backspace"',
    "stepTimelineFrame(editor",
    "上一帧（左方向键）",
    "下一帧（右方向键）",
    ".myh3-rc-clip.selected",
    "syncTimelinePlaybackMedia(editor, editor.playhead)",
    'window.addEventListener("keydown", editor.keyboardHandler, true)',
    'window.removeEventListener("keydown", activeEditor.keyboardHandler, true)',
    ".myh3-rc-preview-media{width:100%;height:100%",
    "editor.snapGuide = null",
];

for (const contract of required) {
    if (!source.includes(contract)) throw new Error(`rough-cut frontend is missing: ${contract}`);
}

if (/item\.ondblclick\s*=\s*\(\)\s*=>\s*\{[\s\S]{0,220}confirm\(`从时间轴删除/.test(source)) {
    throw new Error("timeline deletion must not remain hidden behind double click");
}

console.log("PASS rough-cut program monitor, playback, selection and deletion contract");
