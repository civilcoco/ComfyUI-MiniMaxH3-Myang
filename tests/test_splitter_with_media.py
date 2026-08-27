import importlib
import json
import sys
from pathlib import Path
import torch

TEST_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = TEST_DIR.parent
CUSTOM_NODES_DIR = PACKAGE_DIR.parent
COMFY_DIR = CUSTOM_NODES_DIR.parent

for p in (str(CUSTOM_NODES_DIR), str(COMFY_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

pkg = importlib.import_module("ComfyUI-MiniMaxH3-Myang")
core = importlib.import_module("ComfyUI-MiniMaxH3-Myang.core")
nodes = importlib.import_module("ComfyUI-MiniMaxH3-Myang.nodes")
media_catalog = importlib.import_module("ComfyUI-MiniMaxH3-Myang.media_catalog")


def test_splitter_media_integration():
    print("Testing H3ScriptSplitter media integration...")
    splitter = nodes.H3ScriptSplitter()
    schema = splitter.INPUT_TYPES()
    assert "media" in schema.get("optional", {})

    # Create dummy media bundle with 1 image and 1 video (12s video = 288 frames)
    img_tensor = torch.zeros(1, 480, 864, 3)
    vid_tensor = torch.zeros(288, 480, 864, 3) # 288 frames = 12.0s @ 24fps
    media = media_catalog.MyangMediaCatalog(
        assets=(
            media_catalog.MyangMediaAsset(slot=1, kind="image", payload=img_tensor),
            media_catalog.MyangMediaAsset(slot=2, kind="video", payload=vid_tensor),
        )
    )

    # Test auto-inferring duration from media when length_source is 匹配参考视频时长
    plan_json, count, sec, fps_f, preview, ref_needed = splitter.split(
        script="特写镜头中，白发少女（参考@图片1）模仿@视频1中的动作轻轻旋转起舞。",
        total_seconds=60.0, length_source="匹配参考视频时长",
        segment_seconds=6.0, overlap_frames=22, fps=24.0,
        llm_service="智谱", max_segments=16, ollama_auto_unload=True,
        use_cache=False, seed=42, llm_enabled=False,
        detail_boost=nodes.DETAIL_BOOST_NONE,
        media=media
    )
    data = json.loads(plan_json)
    assert count >= 1
    assert "media_manifest" in data
    manifest = data["media_manifest"]
    assert "@图片1" in manifest and "Picture 1" in manifest
    # The catalog carries the clip at slot 2, but it is the first video,
    # so the prompt tag is @视频1.  Advertising @视频2 would hand the LLM a tag
    # H3Condition rejects as a dangling reference.
    assert "@视频1" in manifest and "Video 1" in manifest
    assert "@视频2" not in manifest
    assert "【已绑定可用素材清单】" in preview

    # Every tag the manifest offers must survive H3Condition's own resolver.
    counts = {"Picture": 0, "Video": 0, "Audio": 0}
    tag_of = {"image": "Picture", "video": "Video", "audio": "Audio"}
    for kind, _payload, _name in core.iter_media(media):
        counts[tag_of[kind]] += 1
    for line in manifest.split("\n"):
        mention = line.split("或 ")[-1].split("：")[0].strip()
        assert "@" not in core.resolve_mentions(mention, counts), (
            "manifest offered a tag H3Condition cannot resolve: %s" % mention)
    print("PASS test_splitter_media_integration")


def test_skill_and_vision_reach_the_split_prompt():
    print("Testing splitter Skill + VLM material handling...")
    agent_nodes = importlib.import_module("ComfyUI-MiniMaxH3-Myang.agent_nodes")
    llm_service = importlib.import_module("ComfyUI-MiniMaxH3-Myang.llm_service")

    media = media_catalog.MyangMediaCatalog(
        assets=(
            media_catalog.MyangMediaAsset(
                slot=1, kind="image", payload=torch.zeros(1, 64, 64, 3),
                filename="hero.png", label="女主角正面照"),
            media_catalog.MyangMediaAsset(
                slot=2, kind="video", payload=torch.zeros(48, 64, 64, 3),
                filename="dance.mp4", label="旋转舞蹈"),
        ))

    captured = {}
    seen_prompts = []

    def fake_call_llm(service, user_text, system_prompt, unload, seed, max_tokens=None):
        captured["system"] = system_prompt
        captured["user"] = user_text
        return json.dumps({"style_header": "统一风格", "segments": [
            {"index": i, "brief": "第%d段" % i, "prompt": "第%d段提示词" % i}
            for i in range(1, 5)]})

    def fake_call_vlm(service, images, prompt, unload):
        seen_prompts.append(prompt)
        return "白发少女站在夜市street前" if len(images) == 1 else "少女连续旋转，镜头缓慢环绕"

    original_llm = nodes.call_llm
    original_vlm = llm_service.call_vlm
    original_b64 = llm_service.tensor_to_base64
    nodes.call_llm = fake_call_llm
    llm_service.call_vlm = fake_call_vlm
    llm_service.tensor_to_base64 = lambda tensor: "data:image/png;base64,stub"
    try:
        plan_json, count, *_rest = nodes.H3ScriptSplitter().split(
            script="白发少女在夜市里旋转起舞，然后走向摊位。",
            total_seconds=20.0, length_source="手动设定总时长",
            segment_seconds=5.0, overlap_frames=22, fps=24.0,
            llm_service="智谱", max_segments=16, ollama_auto_unload=False,
            use_cache=False, seed=7, llm_enabled=True, media=media,
            skill_preset="h3-prompt-writing", skill_text="每段必须以 [Shot N] 开头",
            vlm_service="stub-vlm")
    finally:
        nodes.call_llm = original_llm
        llm_service.call_vlm = original_vlm
        llm_service.tensor_to_base64 = original_b64

    system = captured["system"]
    request_text = system + "\n" + captured["user"]
    assert "【写作技能】" in system, "the Skill never reached the split system prompt"
    assert "每段必须以 [Shot N] 开头" in system, "pasted rules were dropped"
    assert "技能文档里的示例镜头数和示例秒数一律不作数" in system, (
        "the Skill was injected without pinning the segment count")

    # Vision: the VLM was asked about both the still and the clip, and what it
    # saw is in the whitelist the splitter hands to the LLM. The writer keeps
    # stable behavioral rules in the system prompt and request-specific media
    # content in the user prompt, so validate the complete request boundary.
    assert len(seen_prompts) == 2, "the VLM did not look at both materials"
    assert "白发少女站在夜市street前" in request_text, "the image description never reached the LLM"
    assert "少女连续旋转，镜头缓慢环绕" in request_text, "the clip description never reached the LLM"
    assert "subject_name: 女主角正面照" in request_text, "the user's subject name was dropped"
    assert "<Picture 1>" in request_text and "<Video 1>" in request_text, (
        "canonical material tags are missing from the LLM request")

    plan = json.loads(plan_json)
    assert plan.get("skill_source"), "the plan did not record which Skill was used"
    assert count == 4

    # Changing only the Skill must invalidate the split cache: the cached
    # segments were written to a different spec.
    manifest = nodes._format_media_manifest(media)
    base = nodes._cache_key("剧本", 4, "svc", 0, manifest, "技能A")
    assert base != nodes._cache_key("剧本", 4, "svc", 0, manifest, "技能B")
    print("PASS test_skill_and_vision_reach_the_split_prompt")


def test_skill_resolution_degrades_without_an_llm():
    print("Testing Skill fallbacks...")
    rules, source = nodes.resolve_skill("none", "")
    assert rules == "" and source == "", "none should cost nothing"

    rules, source = nodes.resolve_skill("none", "自定义规则一行")
    assert "自定义规则一行" in rules and "pasted" in source, (
        "pasted-only rules must still reach the writer")

    # auto without a configured service must not raise and must not guess.
    rules, source = nodes.resolve_skill("auto", "")
    assert "auto" in source, "auto routing did not report why it fell back"

    rules, source = nodes.resolve_skill("no-such-skill-xyz", "")
    assert rules == "" or source, "an unknown Skill must degrade, not raise"
    print("PASS test_skill_resolution_degrades_without_an_llm")


SCRIPT = "白发少女走进夜市，停在一个摊位前，然后旋转起舞。"


def _split(llm, **overrides):
    inputs = dict(
        script=SCRIPT, total_seconds=36.3, length_source="手动设定总时长",
        segment_seconds=8.0, overlap_frames=22, fps=24.0,
        llm_service="日日新/deepseek-v4-flash", max_segments=16,
        ollama_auto_unload=False, use_cache=False, seed=0, llm_enabled=True,
        skill_preset="none", vlm_service="off")
    inputs.update(overrides)
    original = nodes.call_llm
    nodes.call_llm = llm
    try:
        return nodes.H3ScriptSplitter().split(**inputs)
    finally:
        nodes.call_llm = original


def _payload(count, transitions=None):
    return json.dumps({"style_header": "夜市赛博风", "segments": [
        {"index": i, "brief": "第%d段简要" % i, "prompt": "@图片1 第%d段画面" % i,
         **({"transition": transitions[i - 1]} if transitions else {})}
        for i in range(1, count + 1)]}, ensure_ascii=False)


def test_split_refuses_to_pad_zero_segments_with_the_whole_script():
    """An empty provider response uses distinct deterministic local segments.

    The old code filled an empty response with the whole script repeated N
    times. The current fallback keeps chronology and records its source instead
    of pretending that the provider produced a valid storyboard.
    """
    print("Testing empty-split failure handling...")
    attempts = []

    def silent(service, user, system, unload, seed, max_tokens=None):
        attempts.append((len(system), max_tokens, seed))
        return ""  # a reasoning model that spent its budget thinking

    plan_json, count, *_rest = _split(silent, skill_preset="h3-prompt-writing")
    plan = json.loads(plan_json)
    assert count == 5
    assert plan.get("storyboard_source") == "local_fixed_count"
    prompts = [segment["prompt"] for segment in plan["segments"]]
    assert all(prompt.strip() for prompt in prompts), "local fallback emitted an empty prompt"
    assert len(set(prompts)) == count, "local fallback repeated one prompt"

    assert len(attempts) >= 1, "the empty response did not reach the fallback"
    seeds = [seed for _len, _cap, seed in attempts]
    assert len(set(seeds)) == len(seeds), (
        "provider calls reused a seed, which can reproduce the same empty answer")
    caps = [cap for _len, cap, _seed in attempts]
    assert all(cap is None for cap in caps), "provider output was unexpectedly capped"
    print("PASS test_split_refuses_to_pad_zero_segments_with_the_whole_script")


def test_split_recovers_on_a_later_rung():
    print("Testing split ladder recovery...")
    tries = [0]

    def flaky(service, user, system, unload, seed, max_tokens=None):
        tries[0] += 1
        return "" if tries[0] < 3 else _payload(5)

    plan_json, count, *_rest = _split(flaky, skill_preset="h3-prompt-writing")
    plan = json.loads(plan_json)
    assert count == 5 and len(plan["segments"]) == 5
    prompts = {segment["prompt"] for segment in plan["segments"]}
    assert len(prompts) == 5, "recovered segments collapsed to one prompt"
    print("PASS test_split_recovers_on_a_later_rung")


def test_lean_retry_keeps_media_and_repairs_character_binding():
    print("Testing compact media preservation and local character binding...")
    manifest = (
        "- <Picture 1> 或 @图片1：静态图像（角色外观/服装参考），主体名：拉毗\n"
        "  画面内容：红发少女，红色眼睛，黑色贝雷帽，红黑夹克。\n"
        "- <Picture 2> 或 @图片2：静态图像（场景构图参考），文件：forest.png\n"
        "  画面内容：森林溪流、苔藓岩石和晨雾。")
    compact = nodes._compact_media_manifest(manifest)
    assert "@图片1" in compact and "红发少女" in compact, (
        "the compact manifest discarded the character tag or description")
    assert "@图片2" in compact and "森林溪流" in compact, (
        "the compact manifest discarded the scene tag or description")
    required = nodes._persistent_character_picture_tags("拉毗进行五轮猜拳挑战。", manifest)
    assert required == ["@图片1"], "scene art was misclassified as a persistent character"
    plan = nodes._local_timeline_split(
        "拉毗依次进行五轮猜拳挑战，每轮动作和反应都不同。", 5,
        media_manifest=manifest, reason="provider unavailable")
    prompts = [segment["prompt"] for segment in plan["segments"]]
    assert all("@图片1" in prompt for prompt in prompts), (
        "the local fallback did not restore the persistent character per segment")
    assert all("@图片2" not in prompt for prompt in prompts), (
        "a scene image was incorrectly forced into every segment")
    assert len(set(prompts)) == 5, "local character fallback repeated one shot"
    assert nodes.SPLIT_PROMPT_VERSION >= 6, (
        "split caches created before media-binding repair were not invalidated")
    print("PASS test_lean_retry_keeps_media_and_repairs_character_binding")


def test_split_prompt_explains_the_overlap_arithmetic():
    """The planner gets fixed segment math while overlap stays in the graph."""
    print("Testing fixed-count planner arithmetic framing...")
    seen = {}

    def capture(service, user, system, unload, seed, max_tokens=None):
        seen.setdefault("user", user)
        seen.setdefault("system", system)
        return _payload(5)

    _split(capture)
    user = seen["user"]
    plan = nodes.plan_segments(36.3, 8.0, 22, 24.0, 16)
    assert plan["segment_count"] == 5
    assert "固定段落数：5" in user, "the planner lost the fixed segment count"
    assert "每段约 %.2f 秒" % plan["segment_seconds_snapped"] in user, (
        "the planner lost the frame-snapped segment duration")
    expected_frames = ((plan["segment_count"] - 1)
                       * (plan["frames_per_segment"] - plan["overlap_frames"])
                       + plan["frames_per_segment"])
    assert plan["ref_frames_needed"] == expected_frames, "overlap graph math drifted"
    print("PASS test_split_prompt_explains_the_overlap_arithmetic")


def test_transitions_are_chosen_per_boundary_and_normalised():
    print("Testing per-boundary transitions...")
    seen = {}

    def capture(service, user, system, unload, seed, max_tokens=None):
        seen.setdefault("system", system)
        # Segment 4 answers with a word outside the vocabulary, segment 5 omits
        # the field entirely; neither may take the plan down.
        return _payload(5, ["承接", "承接", "切镜", "胡说", None])

    plan = json.loads(_split(capture)[0])
    transitions = [segment["transition"] for segment in plan["segments"]]
    assert transitions == ["开场", "承接", "切镜", "承接", "承接"], transitions

    system = seen["system"]
    assert "TRANSITION: 开场" in system and "[SEGMENT 1]" in system, (
        "the planner lost its structured transition field")
    assert "固定数量" in system and "不能新增、删除、合并或移动段落" in system, (
        "the planner can change the fixed chronology")
    print("PASS test_transitions_are_chosen_per_boundary_and_normalised")


def test_vlm_retries_under_a_provider_token_cap():
    """glm-4v-flash caps max_tokens at 1024 and answers 400 instead of clamping."""
    print("Testing VLM max_tokens retry...")
    llm_service = importlib.import_module("ComfyUI-MiniMaxH3-Myang.llm_service")
    caps = []

    def fake_post(url, headers, payload, timeout=120):
        caps.append(int(payload["max_tokens"]))
        if int(payload["max_tokens"]) > llm_service.VLM_SAFE_MAX_TOKENS:
            raise RuntimeError(
                'API error 400: {"error":{"code":"1210",'
                '"message":"max_tokens参数非法：限制数值范围[1,1024]"}}')
        return {"choices": [{"message": {"content": "白发少女站在夜市前"}}]}

    original_post = llm_service._http_post_json
    original_find = llm_service._find_service
    original_model = llm_service._find_model
    llm_service._http_post_json = fake_post
    llm_service._find_service = lambda sid: {
        "base_url": "https://example.invalid/v1", "api_key": "k",
        "type": "openai_compatible"}
    llm_service._find_model = lambda svc, name, kind: {"name": "glm-4v-flash"}
    try:
        text = llm_service.call_vlm("智谱/glm-4v-flash", ["stub"], "描述这张图", max_tokens=4096)
    finally:
        llm_service._http_post_json = original_post
        llm_service._find_service = original_find
        llm_service._find_model = original_model

    assert text == "白发少女站在夜市前", "the retry did not return the description"
    assert caps == [4096, llm_service.VLM_SAFE_MAX_TOKENS], (
        "expected one retry under the cap, got %s" % caps)

    # An unrelated 400 must still propagate rather than be retried blindly.
    def always_bad(url, headers, payload, timeout=120):
        raise RuntimeError('API error 400: {"error":{"message":"invalid api key"}}')

    llm_service._http_post_json = always_bad
    llm_service._find_service = lambda sid: {
        "base_url": "https://example.invalid/v1", "api_key": "k",
        "type": "openai_compatible"}
    llm_service._find_model = lambda svc, name, kind: {"name": "glm-4v-flash"}
    try:
        llm_service.call_vlm("智谱/glm-4v-flash", ["stub"], "描述", max_tokens=4096)
    except RuntimeError as error:
        assert "invalid api key" in str(error)
    else:
        raise AssertionError("an unrelated 400 was swallowed by the retry")
    finally:
        llm_service._http_post_json = original_post
        llm_service._find_service = original_find
        llm_service._find_model = original_model
    print("PASS test_vlm_retries_under_a_provider_token_cap")


if __name__ == "__main__":
    test_splitter_media_integration()
    test_skill_and_vision_reach_the_split_prompt()
    test_skill_resolution_degrades_without_an_llm()
    test_split_refuses_to_pad_zero_segments_with_the_whole_script()
    test_split_recovers_on_a_later_rung()
    test_lean_retry_keeps_media_and_repairs_character_binding()
    test_split_prompt_explains_the_overlap_arithmetic()
    test_transitions_are_chosen_per_boundary_and_normalised()
    test_vlm_retries_under_a_provider_token_cap()
    print("ALL TESTS PASSED!")
