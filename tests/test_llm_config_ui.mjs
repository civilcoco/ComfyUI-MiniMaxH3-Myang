import {readFileSync} from "node:fs";

const source = readFileSync(
    new URL("../web/minimax_h3_myang_agent_ui.js", import.meta.url),
    "utf8",
);
const match = source.match(/function routeStatusPresentation\(route\) \{[\s\S]*?\n\}/);
if (!match) throw new Error("routeStatusPresentation definition is missing");
const routeStatusPresentation = (0, eval)(`(${match[0]})`);

function check(value, message) {
    if (!value) throw new Error(message);
}

check(routeStatusPresentation({runtime: {status: "ready"}}).label === "可用",
    "ready route was not presented as available");
check(routeStatusPresentation({runtime: {status: "active"}}).label === "可用",
    "active route was not presented as available");
check(routeStatusPresentation({runtime: {status: "cooling", cooldown_remaining: 2.2}}).label === "冷却中 3s",
    "cooling countdown was not rounded up");
check(routeStatusPresentation({runtime: {status: "blocked", reason: "quota_exhausted"}}).label === "配额阻断",
    "blocked route was not presented");
check(routeStatusPresentation({runtime: {status: "disabled"}}).label === "已停用",
    "disabled route was not presented");
check(routeStatusPresentation({runtime: {status: "future_state"}}).label === "未知",
    "unknown route state did not degrade safely");

console.log("PASS LLM route status presentation");
