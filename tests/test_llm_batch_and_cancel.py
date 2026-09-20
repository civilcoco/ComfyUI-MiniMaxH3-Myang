"""Focused regressions for Director batch writing, fallback caching and stop."""

import importlib
import json
import os
import sys
import threading
import time
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parents[1]
CUSTOM_NODES = PACKAGE_DIR.parent
COMFY_ROOT = CUSTOM_NODES.parent
for path in (str(COMFY_ROOT), str(CUSTOM_NODES)):
    if path not in sys.path:
        sys.path.insert(0, path)

importlib.import_module("ComfyUI-MiniMaxH3-Myang")
nodes = importlib.import_module("ComfyUI-MiniMaxH3-Myang.nodes")
nodes = importlib.import_module("ComfyUI-MiniMaxH3-Myang.nodes")
llm = importlib.import_module("ComfyUI-MiniMaxH3-Myang.llm_service")
agent_nodes = importlib.import_module("ComfyUI-MiniMaxH3-Myang.agent_nodes")


def check(condition, message):
    if not condition:
        raise AssertionError(message)


class CacheProbe:
    def __init__(self):
        self.files = {}

    def __truediv__(self, name):
        return CacheProbeFile(self, str(name))


class CacheProbeFile:
    def __init__(self, owner, name):
        self.owner = owner
        self.name = name

    def is_file(self):
        return self.name in self.owner.files

    def read_text(self, encoding="utf-8"):
        return self.owner.files[self.name]

    def write_text(self, value, encoding="utf-8"):
        self.owner.files[self.name] = value


def _planner(count):
    blocks = []
    for index in range(1, count + 1):
        blocks.extend([
            f"[SEGMENT {index}]",
            "TRANSITION: " + ("开场" if index == 1 else "承接"),
            f"GOAL: 第{index}段剧情",
            f"SHOT: 0.00-7.00 || 构图{index} || 动作{index} || 平移 || 环境音 || 无",
        ])
    return "\n".join(blocks)


def _batch(indexes):
    return "\n".join(
        f"<<<H3_SEGMENT_{index}_BEGIN>>>\nLLM生成的第{index}段提示词\n"
        f"<<<H3_SEGMENT_{index}_END>>>"
        for index in indexes)


def _split(fake_llm, cache_dir, use_cache=True, llm_service="stub/model",
           script="第一段出场。第二段追逐。第三段停下。", total_seconds=22.0,
           segment_seconds=8.0):
    original_call = nodes.call_llm
    original_cache = nodes._cache_dir
    nodes.call_llm = fake_llm
    nodes._cache_dir = lambda: cache_dir
    try:
        return nodes.H3ScriptSplitter().split(
            script=script,
            total_seconds=total_seconds, length_source="手动设定总时长",
            segment_seconds=segment_seconds, overlap_frames=22, fps=24.0,
            llm_service=llm_service, max_segments=16,
            ollama_auto_unload=False, use_cache=use_cache, seed=7,
            llm_enabled=True, skill_preset="none", vlm_service="off")
    finally:
        nodes.call_llm = original_call
        nodes._cache_dir = original_cache


def test_batch_writer_uses_one_call_for_all_prompts():
    calls = []

    def fake(service, user, system, unload, seed, max_tokens=None):
        calls.append((user, system))
        return _planner(3) if len(calls) == 1 else _batch((1, 2, 3))

    plan = json.loads(_split(fake, CacheProbe(), use_cache=False)[0])
    check(len(calls) == 2, "expected one planner and one batch writer call")
    check(all(segment["prompt"].startswith("LLM生成的") for segment in plan["segments"]),
          "batch prompts were replaced by local fallback text")
    check(plan["writer_mode"] == "batch_plain_prompt_with_missing_segment_repair_v2",
          "plan does not identify the new batch writer")


def test_batch_writer_repairs_only_missing_segments():
    calls = []

    def fake(service, user, system, unload, seed, max_tokens=None):
        calls.append(user)
        if len(calls) == 1:
            return _planner(3)
        if len(calls) == 2:
            return _batch((1, 3))
        return "LLM补写的第2段提示词"

    plan = json.loads(_split(fake, CacheProbe(), use_cache=False)[0])
    check(len(calls) == 3, "partial batch should trigger exactly one repair call")
    check(plan["segments"][1]["prompt"] == "LLM补写的第2段提示词",
          "the missing segment was not repaired")
    check(not plan.get("writer_fallback_segments"),
          "a successfully repaired segment was marked as fallback")


def test_long_jobs_prewarm_segments_in_parallel_and_repair_sequentially():
    """Five sequential 60-90s writer calls are the wall-clock cost of a long
    script. In the happy path every brief is complete before the loop starts,
    so the writers fire concurrently; a segment that comes back empty is
    rewritten by the original sequential path."""
    import threading
    import time as time_module

    writer_threads = set()
    writer_calls = []
    seed_counts = {}
    planner_done = []
    lock = threading.Lock()

    def fake(service, user, system, unload, seed, max_tokens=None):
        # The storyboard planner strictly precedes the pre-warm pool.
        if not planner_done:
            planner_done.append(True)
            return _planner(6)
        with lock:
            writer_calls.append(seed)
            writer_threads.add(threading.get_ident())
            seen = seed_counts.get(seed, 0)
            seed_counts[seed] = seen + 1
        # Force the pool to overlap so distinct worker threads are observable.
        time_module.sleep(0.08)
        # Segment 3's pre-warm comes back empty; the sequential repair with the
        # same seed must then succeed.
        if seed == 7 + 3 * 131 and seen == 0:
            return ""
        return f"并行预写的第{(seed - 7) // 131}段提示词"

    plan = json.loads(_split(
        fake, CacheProbe(), use_cache=False,
        script="一。二。三。四。五。", total_seconds=40.0,
        segment_seconds=8.0)[0])

    segment_count = len(plan["segments"])
    check(segment_count == 6,
          "expected the 40s/8s overlap math to plan 6 segments, got %d"
          % segment_count)
    check(len(writer_calls) == segment_count + 1,
          "expected %d pre-warmed writers plus one sequential repair, got %d"
          % (segment_count, len(writer_calls)))
    check(len(writer_threads) >= 2,
          "writer calls ran sequentially instead of on a thread pool")
    prompts = {index: segment["prompt"] for index, segment
               in enumerate(plan["segments"], 1)}
    check(prompts[3] == "并行预写的第3段提示词",
          "the empty pre-warm segment was not repaired")
    check(all(prompts[index].startswith("并行预写")
              for index in range(1, segment_count + 1) if index != 3),
          "a successfully pre-warmed segment was rewritten or lost")


def test_writer_system_prompt_shares_a_stable_prefix_across_durations():
    """The Skill block is the expensive part of every writer call; providers
    with implicit prefix caching can only reuse it while the prompt prefix is
    byte-identical, so per-segment numbers must sit at the very end."""
    skill = "规则甲：保持人物连续。\n" * 40
    long_seg = agent_nodes._build_agent_system_prompt(skill, seconds=8.0, expand=False)
    short_seg = agent_nodes._build_agent_system_prompt(skill, seconds=5.875, expand=False)
    shared = os.path.commonprefix([long_seg, short_seg])
    check(skill.strip() in shared,
          "the Skill block is not inside the shared prompt prefix")
    check(long_seg.index("时长硬性约束") > long_seg.index(skill.strip()),
          "the varying duration block sits before the shared Skill block")
    check(long_seg.endswith("只写这一个视频，不要提前结束。\n")
          and short_seg.endswith("只写这一个视频，不要提前结束。\n"),
          "the duration block must close the prompt so nothing stable follows it")


def test_fallback_payload_is_never_cached():
    calls = []

    def silent(service, user, system, unload, seed, max_tokens=None):
        calls.append(user)
        return _planner(3) if len(calls) == 1 else ""

    cache = CacheProbe()
    try:
        _split(silent, cache, use_cache=True)
    except ValueError as error:
        check("不会拿整段剧本去凑" in str(error),
              "empty writer did not explain why generation stopped")
    else:
        raise AssertionError("empty LLM output was presented as fallback success")
    long_term_cache = [name for name in cache.files
                       if name.endswith(".json")
                       and not name.startswith(nodes.AGENT_CONTEXT_PREFIX)]
    check(not long_term_cache, "a local fallback payload poisoned the long-term cache")
    check(len(calls) == 4,
          "empty batch triggered more than one bounded probe retry")


def test_agent_context_resumes_missing_segments_across_models():
    first_calls = []

    def interrupted(service, user, system, unload, seed, max_tokens=None):
        first_calls.append((service, user))
        if len(first_calls) == 1:
            return _planner(3)
        if len(first_calls) == 2:
            return _batch((1, 2))
        raise llm.LLMQuotaError("API error 429: workspace quota exceeded")

    cache = CacheProbe()
    try:
        _split(interrupted, cache, use_cache=True)
    except RuntimeError as error:
        check("Agent上下文已保存" in str(error),
              "failed generation did not report its resumable context")
    else:
        raise AssertionError("interrupted generation unexpectedly succeeded")

    context_names = [name for name in cache.files
                     if name.startswith(nodes.AGENT_CONTEXT_PREFIX)]
    check(len(context_names) == 1, "expected one Agent context checkpoint")
    saved = json.loads(cache.files[context_names[0]])
    check([item["index"] for item in saved["segments"]] == [1, 2],
          "checkpoint did not retain completed segment prompts")
    check(len(saved.get("briefs") or []) == 3 and saved["briefs"][0].get("writer_input"),
          "checkpoint did not retain the shared writer context")

    resumed_calls = []

    def resumed(service, user, system, unload, seed, max_tokens=None):
        resumed_calls.append((service, user))
        return "跨模型续接的第3段提示词"

    result = _split(
        resumed, cache, use_cache=True, llm_service="another/model")
    plan = json.loads(result[0])
    check(len(resumed_calls) == 1,
          "resume regenerated planning or already completed segments")
    check(resumed_calls[0][0] == "another/model",
          "resume did not use the newly selected model")
    check("LLM生成的第2段提示词" in resumed_calls[0][1],
          "resume request did not carry forward the prior Agent context")
    check([segment["prompt"] for segment in plan["segments"]] == [
        "LLM生成的第1段提示词", "LLM生成的第2段提示词", "跨模型续接的第3段提示词"],
          "resumed plan did not merge saved and newly generated prompts")
    check(plan["agent_context"]["resumed"] is True
          and plan["agent_context"]["pending_segments"] == 0,
          "plan did not expose completed Agent context state")


def test_partial_batch_checkpoint_keeps_later_segments_on_first_repair_failure():
    calls = []

    def interrupted(service, user, system, unload, seed, max_tokens=None):
        calls.append(user)
        if len(calls) == 1:
            return _planner(3)
        if len(calls) == 2:
            return _batch((2, 3))
        raise llm.LLMQuotaError("API error 429: workspace quota exceeded")

    cache = CacheProbe()
    try:
        _split(interrupted, cache, use_cache=True)
    except RuntimeError as error:
        check("Agent上下文已保存" in str(error),
              "partial batch failure did not report its checkpoint")
    else:
        raise AssertionError("partial batch failure unexpectedly succeeded")
    context_names = [name for name in cache.files
                     if name.startswith(nodes.AGENT_CONTEXT_PREFIX)]
    check(len(context_names) == 1, "expected one partial-batch checkpoint")
    saved = json.loads(cache.files[context_names[0]])
    check([item["index"] for item in saved["segments"]] == [2, 3],
          "later batch results were lost when the first repair failed")


def test_agent_context_key_ignores_vlm_description_drift():
    first = ("- <Picture 1> 或 @图片1：静态图像（角色外观）"
             "，主体名：主角，文件：actor.png\n"
             "画面内容：人物站在明亮街道上")
    second = ("- <Picture 1> 或 @图片1：静态图像（角色外观）"
              "，主体名：主角，文件：actor.png\n"
              "画面内容：主角位于阴天街道，镜头略微靠近")
    check(nodes._agent_context_key("剧本", 2, first, "规则", "none", "",
                                   20, 8, 22, 24)
          == nodes._agent_context_key("剧本", 2, second, "规则", "none", "",
                                      20, 8, 22, 24),
          "VLM description drift unexpectedly invalidated the Agent context")
    changed_header = first.replace("actor.png", "other.png")
    check(nodes._agent_context_key("剧本", 2, first, "规则", "none", "",
                                   20, 8, 22, 24)
          != nodes._agent_context_key("剧本", 2, changed_header, "规则", "none", "",
                                      20, 8, 22, 24),
          "changing the referenced media did not invalidate the Agent context")


def test_http_wait_can_be_cancelled_without_restarting_comfyui():
    started = threading.Event()
    release = threading.Event()
    captured = []
    original = llm._http_post_json_blocking

    def blocking(url, headers, payload, timeout, state):
        started.set()
        release.wait(5)
        return {"choices": [{"message": {"content": "late"}}]}

    def invoke():
        try:
            llm._http_post_json("https://example.invalid", {}, {}, timeout=5)
        except BaseException as error:
            captured.append(error)

    llm._http_post_json_blocking = blocking
    caller = threading.Thread(target=invoke)
    try:
        caller.start()
        check(started.wait(1), "HTTP worker did not start")
        check(llm.active_http_request_count() == 1, "active request was not registered")
        check(llm.cancel_active_http_requests() == 1, "stop did not target the active request")
        caller.join(1)
        check(not caller.is_alive(), "cancel left the ComfyUI execution thread blocked")
        check(captured and type(captured[0]).__name__ == "InterruptProcessingException",
              "cancel did not propagate as a normal ComfyUI interruption")
    finally:
        release.set()
        caller.join(1)
        llm._http_post_json_blocking = original


def test_http_wait_has_a_real_wall_clock_deadline():
    release = threading.Event()
    original = llm._http_post_json_blocking

    def ignores_socket_timeout(url, headers, payload, timeout, state):
        release.wait(5)
        return {"choices": [{"message": {"content": "late"}}]}

    llm._http_post_json_blocking = ignores_socket_timeout
    started = time.monotonic()
    try:
        try:
            llm._http_post_json("https://example.invalid", {}, {}, timeout=0.2)
        except Exception as error:
            check(isinstance(error, llm.LLMRequestTimeoutError),
                  "wall-clock deadline raised the wrong error type")
        else:
            raise AssertionError("worker ignored the wall-clock deadline")
        check(time.monotonic() - started < 1.0,
              "outer wait still followed a stuck socket beyond its deadline")
    finally:
        release.set()
        llm._http_post_json_blocking = original


def test_service_unavailable_stops_before_segment_retry_storm():
    calls = []

    def unavailable(service, user, system, unload, seed, max_tokens=None):
        calls.append(user)
        raise llm.LLMRateLimitError("API error 429: TPM exhausted")

    try:
        _split(unavailable, CacheProbe(), use_cache=False)
    except RuntimeError as error:
        check("不会再逐段调用" in str(error),
              "unavailable service did not explain the stopped batch")
    else:
        raise AssertionError("transport failure silently produced fallback segments")
    check(len(calls) == 2,
          "one unavailable batch fanned out into per-segment calls: %s" % len(calls))


def test_director_exposes_accessible_stop_control():
    frontend = (PACKAGE_DIR / "web" / "h3_director_ui.js").read_text("utf-8")
    backend = (PACKAGE_DIR / "agent_nodes.py").read_text("utf-8")
    check('"/minimax-h3-agent/llm-stop"' in frontend
          and '@_prompt_server.routes.post("/minimax-h3-agent/llm-stop")' in backend,
          "Director stop button is not connected to the backend route")
    check('aria-label", "停止当前导演台 LLM 提示词生成"' in frontend,
          "stop button has no accessible name")
    check("min-height:44px" in frontend, "stop button touch target is too small")


def test_settings_restore_llm_and_skill_management_features():
    frontend = (PACKAGE_DIR / "web" / "minimax_h3_myang_agent_ui.js").read_text("utf-8")
    backend = (PACKAGE_DIR / "agent_nodes.py").read_text("utf-8")
    check('const SETTINGS_CATEGORY = ["Myang_node", "MiniMax H3"]' in frontend,
          "Myang settings are not registered in a stable settings category")
    check('function settingsCategory(itemName)' in frontend
          and frontend.count('category: settingsCategory("') == 5,
          "settings reuse one category leaf and overwrite each other")
    check("function routeStatusPresentation(route = {})" in frontend
          and "线路状态渲染降级" in frontend,
          "LLM route status can still blank the complete editor panel")
    check('class="h3-llm-select"' in frontend
          and 'fetch("/minimax-h3-agent/llm-config")' in frontend,
          "skill manager cannot select configured LLM services")
    check('class="h3-digest-tab"' in frontend
          and '/minimax-h3-agent/digest?name=' in frontend,
          "skill manager does not expose learned Chinese summaries")
    check('<textarea class="h3-digest-view"' in frontend
          and 'class="h3-save-digest-btn"' in frontend
          and 'fetch("/minimax-h3-agent/digest", {' in frontend,
          "learned Chinese summary is not editable and persistable")
    check('@_prompt_server.routes.post("/minimax-h3-agent/digest")' in backend,
          "manual learned-summary edits have no backend save route")
    check('learned_by == "llm_partial"' in backend
          and "待续学" in frontend,
          "partial Skill learning is still presented as complete")
    check(hasattr(llm, "LLMEmptyResponseError"),
          "empty LLM responses can still mask the original provider failure")
    check('class="h3-stop-learn-btn"' in frontend
          and 'fetch("/minimax-h3-agent/llm-stop"' in frontend,
          "skill learning cannot be interrupted")
    check(frontend.count("width:132px;height:44px") >= 8,
          "variable skill names can still compress neighboring action buttons")


def test_skill_is_injected_once_and_long_jobs_use_map_writer():
    skill = "规则甲：保持人物连续。\n" * 120
    system = agent_nodes._build_agent_system_prompt(skill, seconds=8, expand=False)
    check(system.count(skill) == 1,
          "long Skill text was duplicated in the Agent system prompt")
    source = __import__("inspect").getsource(nodes._write_segments_with_media_agent)
    check("use_batch_writer = int(count) <= 3" in source,
          "long jobs still force every segment into one batch request")


if __name__ == "__main__":
    for test in (
        test_batch_writer_uses_one_call_for_all_prompts,
        test_batch_writer_repairs_only_missing_segments,
        test_long_jobs_prewarm_segments_in_parallel_and_repair_sequentially,
        test_writer_system_prompt_shares_a_stable_prefix_across_durations,
        test_fallback_payload_is_never_cached,
        test_agent_context_resumes_missing_segments_across_models,
        test_partial_batch_checkpoint_keeps_later_segments_on_first_repair_failure,
        test_agent_context_key_ignores_vlm_description_drift,
        test_http_wait_can_be_cancelled_without_restarting_comfyui,
        test_http_wait_has_a_real_wall_clock_deadline,
        test_service_unavailable_stops_before_segment_retry_storm,
        test_director_exposes_accessible_stop_control,
        test_settings_restore_llm_and_skill_management_features,
        test_skill_is_injected_once_and_long_jobs_use_map_writer,
    ):
        test()
        print("PASS", test.__name__)
