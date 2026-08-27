import {readFileSync} from "node:fs";

const sourceUrl = new URL("../web/h3_progress_state.js", import.meta.url);
const source = readFileSync(sourceUrl, "utf8");
const progress = await import(
    `data:text/javascript;base64,${Buffer.from(source).toString("base64")}`
);

function check(value, message) {
    if (!value) throw new Error(message);
}

const state = progress.createProgressState({
    runId: "run-test", total: 5, refining: true,
});

function apply(detail, expected = true) {
    const before = progress.progressPercent(state);
    const accepted = progress.applyProgressEvent(state, {
        run_id: "run-test", total_segments: 5, ...detail,
    });
    const after = progress.progressPercent(state);
    check(accepted === expected, `unexpected acceptance for ${JSON.stringify(detail)}`);
    check(after >= before, `progress regressed from ${before} to ${after}`);
}

// Expanded graph branches may finish in this exact non-sequential order.
apply({segment_index: 5, stage: "sampled"});
apply({segment_index: 5, stage: "done"});
check(state.phase === "segments", "out-of-order completion was shown as sequential sampling");
apply({segment_index: 2, stage: "sampled"});
apply({segment_index: 5, stage: "refine_start"}, false);
apply({segment_index: 2, stage: "done"});
apply({segment_index: 3, stage: "done"});
apply({segment_index: 4, stage: "done"});
apply({segment_index: 1, stage: "done"});
check(state.phase === "assembling", "all segment completions did not enter assembly");
check(progress.completedSegmentCount(state) === 5, "completed segment set is incomplete");

// Once assembly starts, stale segment signals can never move the UI backwards.
apply({segment_index: 1, stage: "sampling", pass_label: "sample1", step: 1, step_total: 15}, false);
apply({segment_index: 5, stage: "assembling"});
apply({segment_index: 5, stage: "assembled"});
check(state.phase === "encoding", "assembled event did not enter encoding");
progress.finishProgress(state);
check(state.phase === "done" && progress.progressPercent(state) === 100,
    "execution success did not become the only whole-run completion");

const otherRun = progress.createProgressState({runId: "run-a", total: 1});
check(!progress.applyProgressEvent(otherRun, {
    run_id: "run-b", segment_index: 1, total_segments: 1, stage: "done",
}), "a stale run was allowed to update current progress");

console.log("PASS monotonic out-of-order progress and assembly stages");
