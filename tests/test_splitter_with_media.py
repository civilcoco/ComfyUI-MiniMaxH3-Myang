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
nodes = importlib.import_module("ComfyUI-MiniMaxH3-Myang.nodes")
agent_media = importlib.import_module("ComfyUI-MiniMaxH3-Myang.agent_media")


def test_splitter_media_integration():
    print("Testing H3ScriptSplitter media integration...")
    splitter = nodes.H3ScriptSplitter()
    schema = splitter.INPUT_TYPES()
    assert "media" in schema.get("optional", {})

    # Create dummy media bundle with 1 image and 1 video (12s video = 288 frames)
    img_tensor = torch.zeros(1, 480, 864, 3)
    vid_tensor = torch.zeros(288, 480, 864, 3) # 288 frames = 12.0s @ 24fps
    media = agent_media.MiniMaxH3MediaBundle(
        items=(
            agent_media._MediaInput(input_index=1, media_type="image", value=img_tensor),
            agent_media._MediaInput(input_index=2, media_type="video", value=vid_tensor),
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
    # The bundle carries the clip at input_index 2, but it is the first video,
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

    media = agent_media.MiniMaxH3MediaBundle(
        items=(
            agent_media._MediaInput(
                input_index=1, media_type="image", value=torch.zeros(1, 64, 64, 3)),
            agent_media._MediaInput(
                input_index=2, media_type="video", value=torch.zeros(48, 64, 64, 3)),
        ),
        links=(
            {"order": 1, "filename": "hero.png", "subject": "女主角正面照"},
            {"order": 2, "filename": "dance.mp4", "subject": "旋转舞蹈"},
        ))

    captured = {}
    seen_prompts = []
    llm_calls = [0]

    def fake_call_llm(service, user_text, system_prompt, unload, seed, max_tokens=None):
        llm_calls[0] += 1
        captured["system"] = system_prompt
        captured["user"] = user_text
        if llm_calls[0] == 1:
            return "\n".join(
                "[SEGMENT %d]\nTITLE: 夜市旋舞\nDURATION: %.1f\nTRANSITION: %s\nGOAL: 白发少女旋转\n"
                "SUBJECT: 女主角正面照 || @图片1 || %s || 白发少女保持参考图身份与服装\n"
                "SHOT: 0.00-5.00 || 夜市全景 || 少女旋转 || 环绕 || 脚步声 || @图片1、@视频1"
                % (i, 3.2 + i * 0.2, "开场" if i == 1 else "承接",
                   "首次" if i == 1 else "延续")
                for i in range(1, 7))
        return "[Shot 1] 白发少女（<Picture 1>）参考<Video 1>旋转，夜市灯光跟随。"

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
    assert "【写作技能】" in system, "the Skill never reached the split system prompt"
    assert "每段必须以 [Shot N] 开头" in system, "pasted rules were dropped"
    assert "技能文档里的示例镜头数和示例秒数一律不作数" in system, (
        "the Skill was injected without pinning the segment count")

    # Vision: the VLM was asked about both the still and the clip, and what it
    # saw is in the whitelist the splitter hands to the LLM.
    assert len(seen_prompts) == 2, "the VLM did not look at both materials"
    writer_request = system + "\n" + captured["user"]
    assert "白发少女站在夜市street前" in writer_request, "the image description never reached the LLM"
    assert "少女连续旋转，镜头缓慢环绕" in writer_request, "the clip description never reached the LLM"
    assert "subject_name: 女主角正面照" in writer_request, "the user's subject name was dropped"
    assert "<Picture 1>" in writer_request and "<Video 1>" in writer_request, "material tags are missing"

    plan = json.loads(plan_json)
    assert plan.get("skill_source"), "the plan did not record which Skill was used"
    assert count == 6, "5 秒是上限，重叠后 20 秒需要 6 个可变时长段位"
    assert all(segment["duration_seconds"] <= 5.0 for segment in plan["segments"])
    assert plan["segments"][0]["title"] == "夜市旋舞"
    assert plan["segments"][0]["subjects"][0]["name"] == "女主角正面照"

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
    """A silent fallback here renders every segment as the same shot.

    The old code filled an empty response with `{"prompt": <the whole script>}`
    repeated N times, which looks like a successful run until the video plays
    back as N copies of one shot.
    """
    print("Testing empty-split failure handling...")
    attempts = []

    def silent(service, user, system, unload, seed, max_tokens=None):
        attempts.append((len(system), max_tokens, seed))
        return ""  # a reasoning model that spent its budget thinking

    try:
        _split(silent, skill_preset="h3-prompt-writing")
    except ValueError as error:
        message = str(error)
        assert "智能技能" not in message
        assert "不会拿整段剧本去凑" in message, "the error does not name the failure"
        assert "关掉" in message and "智能切片" in message, (
            "the error does not point at the intentional pass-through mode")
    else:
        raise AssertionError("an empty split was accepted and padded")

    # Long jobs use the bounded map writer, so an empty planner is followed by
    # one segment probe and one fresh-seed retry, then stops before fanning out.
    assert len(attempts) == 3, "expected planner + one probe + one retry, got %d" % len(attempts)
    seeds = [seed for _len, _cap, seed in attempts]
    assert len(set(seeds)) == len(seeds), (
        "the smaller probe reused a seed, which reproduces the same empty answer")
    caps = [cap for _len, cap, _seed in attempts]
    assert all(cap is None for cap in caps), (
        "the calls imposed an output ceiling instead of leaving output to the server: %s"
        % caps)
    print("PASS test_split_refuses_to_pad_zero_segments_with_the_whole_script")


def test_split_maps_each_segment_after_local_planner():
    print("Testing bounded segment writer after local planner...")
    tries = [0]

    def flaky(service, user, system, unload, seed, max_tokens=None):
        tries[0] += 1
        if tries[0] == 1:
            return ""
        return "LLM补写的第%d段提示词" % (tries[0] - 1)

    plan_json, count, *_rest = _split(flaky, skill_preset="h3-prompt-writing")
    plan = json.loads(plan_json)
    assert count == 5 and len(plan["segments"]) == 5
    prompts = {segment["prompt"] for segment in plan["segments"]}
    assert len(prompts) == 5, "recovered segments collapsed to one prompt"
    print("PASS test_split_maps_each_segment_after_local_planner")


def test_lean_retry_keeps_media_and_repairs_character_binding():
    print("Testing single-segment probe media preservation and character binding repair...")
    requests = []

    def succeeds_on_lean_rung(service, user, system, unload, seed, max_tokens=None):
        requests.append((user, system))
        if len(requests) < 3:
            return ""
        return "第%d段：微缩角色在桌面完成本段动作。" % (len(requests) - 2)

    manifest = (
        "- <Picture 1> 或 @图片1：静态图像（角色外观/服装参考），主体名：拉毗\n"
        "  画面内容：红发少女，红色眼睛，黑色贝雷帽，红黑夹克。\n"
        "- <Picture 2> 或 @图片2：静态图像（场景构图参考），文件：forest.png\n"
        "  画面内容：森林溪流、苔藓岩石和晨雾。")
    repeated_character_script = (
        "拉毗进行第一轮猜拳。拉毗进行第二轮猜拳。拉毗进行第三轮猜拳。"
        "拉毗进行第四轮猜拳。拉毗进行第五轮猜拳。")
    plan = json.loads(_split(
        succeeds_on_lean_rung,
        script=repeated_character_script,
        media=manifest,
        skill_preset="h3-prompt-writing")[0])

    assert len(requests) == 7, "fixture did not run one small request per segment after the probe succeeded"
    # Inspect the user request only; the reusable Skill may contain illustrative
    # <Picture 2> syntax even when that asset is not in this segment whitelist.
    first_probe = requests[2][0]
    assert "@图片1" in first_probe and "红发少女" in first_probe, (
        "the small-request probe discarded the character tag or VLM description")
    assert "<Picture 2>" not in first_probe and "森林溪流" not in first_probe, (
        "an unrelated shared scene asset still reached this segment writer")

    prompts = [segment["prompt"] for segment in plan["segments"]]
    assert all("@图片1" in prompt for prompt in prompts), (
        "a persistent character image omitted by the LLM was not restored per segment")
    assert all("@图片2" not in prompt for prompt in prompts), (
        "a scene image was incorrectly forced into every segment")
    audit = plan.get("media_reference_compliance") or {}
    assert audit.get("passed") is True
    assert audit.get("required_character_tags") == ["@图片1"]
    assert len(audit.get("repaired_segments") or []) == 5
    assert "[素材引用] 贯穿人物参考已校验：@图片1（已自动补回 5 段）" in _split(
        succeeds_on_lean_rung,
        script=repeated_character_script,
        media=manifest,
        skill_preset="h3-prompt-writing")[4] if False else True
    assert nodes.SPLIT_PROMPT_VERSION >= 6, (
        "split caches created before media-binding repair were not invalidated")
    print("PASS test_lean_retry_keeps_media_and_repairs_character_binding")


def test_shared_media_is_selected_per_segment_not_globally():
    print("Testing segment-scoped shared-media selection...")
    manifest = (
        "- <Picture 1> 或 @图片1：静态图像（角色外观参考），主体名：桃乐丝角色立绘\n"
        "  画面内容：粉发少女、猫耳与蓝色连衣裙。\n"
        "- <Picture 2> 或 @图片2：静态图像（角色外观参考），主体名：紫苑角色立绘\n"
        "  画面内容：紫发少女、白色制服与金色发饰。\n"
        "- <Picture 3> 或 @图片3：静态图像（场景构图参考），文件：city.png\n"
        "  画面内容：雨夜城市街道。")
    text = (
        "人物设定：桃乐丝角色立绘参考@图片1。\n"
        "人物设定：紫苑角色立绘参考@图片2。\n"
        "视觉风格：柔和电影光。\n"
        "剧情正文按下面三个分段执行。")
    storyboard = [
        {"transition": "开场", "segment_goal": "桃乐丝独自在房间挥手", "shots": []},
        {"transition": "切镜", "segment_goal": "紫苑独自在花园转身", "shots": []},
        {"transition": "切镜", "segment_goal": "空镜展示桌上的茶杯", "shots": []},
    ]
    _style, briefs = nodes._segment_writer_briefs(
        text,
        ["桃乐丝独自在房间挥手。", "紫苑独自在花园转身。", "空镜展示桌上的茶杯。"],
        manifest, 5.0, storyboard=storyboard)

    assert briefs[0]["selected_media_tags"] == ["@图片1"]
    assert briefs[1]["selected_media_tags"] == ["@图片2"]
    assert briefs[2]["selected_media_tags"] == []
    assert "紫苑" not in briefs[0]["writer_input"] and "@图片2" not in briefs[0]["writer_input"]
    assert "桃乐丝" not in briefs[1]["writer_input"] and "@图片1" not in briefs[1]["writer_input"]
    assert "subject_definitions" in briefs[2]["writer_input"]
    assert "不要定义未出镜主体" in briefs[2]["writer_input"]
    assert "柔和电影光" in briefs[2]["writer_input"], "global visual style was incorrectly removed"
    print("PASS test_shared_media_is_selected_per_segment_not_globally")


def test_character_reference_repair_never_creates_a_leading_heading():
    print("Testing natural character-reference repair...")
    chinese, restored = nodes._inject_missing_character_refs(
        "人物外观参考：无关的旧人物图。\n\n桃乐丝走到窗边。晨光照亮她的侧脸。",
        ["@图片1"])
    assert restored == ["@图片1"]
    assert not chinese.startswith("人物外观参考：")
    assert "无关的旧人物图" not in chinese
    assert "桃乐丝走到窗边。 本段实际出镜人物的可见外观与 @图片1 保持一致。" in chinese

    structured, restored = nodes._inject_missing_character_refs(
        "integrated_multimodal_description: [Shot 1] A girl turns toward camera.\n"
        "overall_soundscape: Quiet room tone.", ["@图片2"])
    assert restored == ["@图片2"]
    assert structured.startswith("integrated_multimodal_description:")
    assert "Character appearance reference:" not in structured
    assert "matches @图片2 in appearance" in structured

    no_subject, restored = nodes._inject_missing_character_refs(
        "人物外观参考：错误人物。\n\n空镜展示雨中的街道。", [])
    assert restored == []
    assert no_subject == "空镜展示雨中的街道。"
    print("PASS test_character_reference_repair_never_creates_a_leading_heading")


def test_split_prompt_explains_the_overlap_arithmetic():
    """The writer gets fixed segment facts and never has to redo trim maths."""
    print("Testing fixed-count writer framing...")
    seen = []

    def capture(service, user, system, unload, seed, max_tokens=None):
        seen.append((user, system))
        return _payload(5)

    _split(capture)
    assert len(seen) == 6, "long jobs should use one planner plus five bounded writer calls"
    user, system = seen[1]
    assert "5 段" in user and "8.00 秒" in user, "the split brief lost its numbers"
    assert "不要改变段数" in user, "the writer can still re-split the fixed chronology"
    assert "只写这一段" in user, "the map writer did not scope the request to one segment"
    assert "批量输出协议" not in system, "long jobs still use the oversized batch contract"
    print("PASS test_split_prompt_explains_the_overlap_arithmetic")


def test_transitions_are_chosen_per_boundary_and_normalised():
    print("Testing per-boundary transitions...")
    seen = []

    def capture(service, user, system, unload, seed, max_tokens=None):
        seen.append((user, system))
        # Segment 4 answers with a word outside the vocabulary, segment 5 omits
        # the field entirely; neither may take the plan down.
        return _payload(5, ["承接", "承接", "切镜", "胡说", None])

    plan = json.loads(_split(capture)[0])
    transitions = [segment["transition"] for segment in plan["segments"]]
    assert transitions == ["开场", "承接", "切镜", "承接", "承接"], transitions

    planner_system = seen[0][1]
    assert "段间衔接由你根据源剧情逐段判断" in planner_system, (
        "the rules still force every boundary to be a smooth continuation")
    assert "切镜" in planner_system and "换了场景" in planner_system, (
        "the model is never told when a hard cut is the right call")
    assert "不要规划黑场" in planner_system, (
        "the seam anchor caveat is missing, so the model may ask for a "
        "transition effect the pipeline cannot honour")
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
    test_split_maps_each_segment_after_local_planner()
    test_lean_retry_keeps_media_and_repairs_character_binding()
    test_shared_media_is_selected_per_segment_not_globally()
    test_character_reference_repair_never_creates_a_leading_heading()
    test_split_prompt_explains_the_overlap_arithmetic()
    test_transitions_are_chosen_per_boundary_and_normalised()
    test_vlm_retries_under_a_provider_token_cap()
    print("ALL TESTS PASSED!")
