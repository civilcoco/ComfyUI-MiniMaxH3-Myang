import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


TEST_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = TEST_DIR.parent
CUSTOM_NODES_DIR = PACKAGE_DIR.parent
COMFY_DIR = CUSTOM_NODES_DIR.parent
for path in (str(COMFY_DIR), str(CUSTOM_NODES_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

package = importlib.import_module("ComfyUI-MiniMaxH3-Myang")
director = importlib.import_module("ComfyUI-MiniMaxH3-Myang.director")
neural = importlib.import_module("ComfyUI-MiniMaxH3-Myang.latent_upscale_3d")
legacy = importlib.import_module("ComfyUI-MiniMaxH3-Myang.nodes")
shot_media_module = importlib.import_module("ComfyUI-MiniMaxH3-Myang.media")
media_catalog = importlib.import_module("ComfyUI-MiniMaxH3-Myang.media_catalog")
turbo = importlib.import_module("ComfyUI-MiniMaxH3-Myang.turbo")


def check(value, message):
    if not value:
        raise AssertionError(message)


def test_director_registration_and_variable_timeline():
    check("H3Director" in package.NODE_CLASS_MAPPINGS,
          "Director is not registered")
    director_optional = director.H3Director.INPUT_TYPES()["optional"]
    redundant = {"script", "二采设置", "Turbo联合模型", "Turbo推荐一采步数"}
    check(redundant.isdisjoint(director_optional),
          "Director still exposes redundant compatibility inputs")
    plan = director._timeline_plan({"shots": [
        {"prompt": "镜头一", "duration_seconds": 5.0},
        {"prompt": "镜头二", "duration_seconds": 6.0},
        {"prompt": "禁用", "duration_seconds": 7.0, "enabled": False},
    ]}, 22)
    check(plan["segment_count"] == 2, "disabled shot was not filtered")
    check(plan["segments"][0]["frames"] != plan["segments"][1]["frames"],
          "variable shot durations collapsed to one value")
    expected = sum(item["frames"] for item in plan["segments"]) - 22
    check(plan["ref_frames_needed"] == expected,
          "variable timeline reference length is wrong")


def test_director_panel_tracks_node_selection_and_resize():
    source = (PACKAGE_DIR / "web" / "h3_director_ui.js").read_text("utf-8")
    check("function syncPanelGeometry(node)" in source,
          "Director DOM panel has no geometry synchronizer")
    selection_hook = 'for (const hook of ["onSelected", "onDeselected"])'
    check(selection_hook in source,
          "Director panel is not resynchronized after selection changes")
    check("root.parentElement" in source,
          "Director only resizes the child and leaves its narrow DOM holder")
    selection_start = source.index(selection_hook)
    selection_end = source.index("const onResize", selection_start)
    check("refresh(this)" not in source[selection_start:selection_end],
          "selection still rebuilds the Director form and destroys focus")
    check('event.stopPropagation()' in source,
          "Director form pointer events still leak into the canvas")
    check('item.name !== TIMELINE_WIDGET' in source,
          "saving timeline JSON still schedules a full form rebuild")
    check('["keydown", "keyup", "keypress"]' in source,
          "Director form keyboard events still leak into canvas shortcuts")
    check("createPromptEditor(node, shot" in source and "directorMediaList(node, shot)" in source,
          "Director prompt editor does not render ordinal material mentions")
    check("label.textContent = entry?.subject" in source,
          "Director material chips still expose filenames instead of optional subject names")
    long_source = (PACKAGE_DIR / "web" / "h3_longvideo_ui.js").read_text("utf-8")
    check("text.textContent = entry?.subject" in long_source,
          "Long-video material chips still expose filenames instead of optional subject names")
    check('api.addEventListener("myh3_progress"' in source
          and "renderDirectorProgressPanel(node)" in source,
          "Director has no live segment progress / preview panel")
    check('const NATIVE_VIDEO_WIDGET = "video-preview"' in source
          and "function mountNativeVideoPreview(node)" in source
          and 'video.style.cssText = "display:block;width:100%;height:100%' in source,
          "Director does not embed and contain ComfyUI's native video player")
    check("images: undefined" not in source
          and "gifs: undefined" not in source
          and "videos: undefined" not in source,
          "Director still discards the native player payload")
    check(source.count("root.appendChild(renderOutputVideoPanel(node))") == 3,
          "not every Director mode puts the output player inside its panel")
    check("overflow-y:auto;scrollbar-gutter:stable" in source
          and "max-height:340px;overflow-y:auto" not in source
          and "max-height:470px;overflow-y:auto" not in source,
          "Director still uses nested scrolling instead of one card-level scrollbar")
    check("function syncScriptInputHeight(node)" in source
          and "SCRIPT_INPUT_MIN_HEIGHT = 56" in source
          and "SCRIPT_INPUT_MAX_HEIGHT = 220" in source
          and "input.scrollHeight" in source
          and 'input.style.overflowY = contentHeight > SCRIPT_INPUT_MAX_HEIGHT ? "auto" : "hidden"' in source,
          "Director long-script input does not grow with content and cap into scrolling")
    check("function renderSourcePanel(node" in source
          and 'panel.dataset.myangDirectorSection = "source"' in source
          and 'display:block;width:100%;min-width:0;max-width:100%' in source
          and "fitScriptTextArea(area)" in source,
          "Agent script editor is not rendered as a full-width adaptive Director card")
    check('for (const name of [\n        "source_mode", "script_fallback", "total_seconds", "segment_seconds",'
          in source
          and "hideWidget(by[name]);" in source,
          "narrow native Agent widgets still duplicate and displace the Director cards")
    check('input.addEventListener("input"' in source
          and "syncScriptInputHeight(node);" in source,
          "Director long-script input is not resized after edits or node resizing")
    progress_css = (PACKAGE_DIR / "web" / "h3_prompt_editor.css").read_text("utf-8")
    check(".myh3-progress-preview" in progress_css and "max-height: 170px" in progress_css,
          "Director's replacement progress preview has no compact height cap")
    check('root.className = "myh3-director-root"' in source
          and ".myh3-director-root > *" in progress_css
          and "flex-shrink: 0" in progress_css
          and source.count("flex:0 0 auto;min-height:38px") >= 1,
          "Director flex children can still collapse the closed detail card into a border line")
    check("setVisible(by.steps, true)" in source,
          "Director still hides first-pass steps when Turbo is connected")
    check("function migrateLegacyInputs(node)" in source
          and 'upstream(node, "model")' in source,
          "Director does not migrate old Turbo wiring to its single model input")
    check("function ensureRecommendedStepsInput(node)" not in source,
          "Director still recreates the removed recommended-step socket")
    check("renderReferenceVideoPanel(node)" in source
          and "匹配参考视频原分辨率" in source,
          "Director has no reference-video resolution controls")
    check('button("转入导演台分镜卡"' in source
          and "function transferPlanToStoryboard(node)" in source,
          "Director cannot freeze the latest LLM plan into editable storyboard cards")
    check("plan_snapshot: normalizePlanSnapshot(node.__myangDirectorPlan)" in source
          and "this.__myangDirectorPlan = parsePlanSnapshot(this)" in source,
          "the latest LLM plan is not persisted/restored with the workflow")
    plan_event = source.index('api.addEventListener("myh3_director_plan"')
    start_event = source.index('api.addEventListener("myh3_longvideo_start"')
    check("saveTimeline(node);" in source[plan_event:start_event],
          "the LLM plan is not saved before video sampling can be interrupted")
    transfer_start = source.index("function transferPlanToStoryboard(node)")
    transfer_end = source.index("function updateSegmentPlan(node)", transfer_start)
    transfer_source = source[transfer_start:transfer_end]
    check("source.value = MANUAL" in transfer_source
          and "fixed_from_plan: true" in transfer_source,
          "transferring a plan does not lock future runs to the manual-card path")
    check('prompt: String(segment.prompt || "")' in transfer_source,
          "transferring a plan drops repaired @图片N bindings from its prompt")
    timeline_start = source.index("function renderTimeline(node)")
    timeline_end = source.index("function refresh(node)", timeline_start)
    timeline_source = source[timeline_start:timeline_end]
    transfer_branch = timeline_source.index("if (transferring) {")
    check(0 <= timeline_source.index("root.appendChild(renderDetailPanel(node));") < transfer_branch,
          "second-pass card is still conditional on the source mode")
    detail_start = source.index("function renderDetailPanel(node)")
    global_assets_start = source.index("function renderGlobalAssets(node", detail_start)
    detail_source = source[detail_start:global_assets_start]
    check('panel.dataset.myangCollapsible = "detail"' in detail_source
          and 'document.createElement("summary")' not in detail_source
          and 'min-height:38px' in detail_source
          and 'setAttribute("aria-expanded"' in detail_source,
          "collapsed second-pass card can still be flattened by global details/summary CSS")


def test_director_storyboard_card_import_export_ui():
    source = (PACKAGE_DIR / "web" / "h3_director_ui.js").read_text("utf-8")
    schema = (PACKAGE_DIR / "web" / "h3_storyboard_cards.js").read_text("utf-8")
    check('button("导入分镜卡"' in source and 'button("导出分镜卡"' in source,
          "Director manual-card toolbar has no structured import/export actions")
    check("createStoryboardCardDocument" in source
          and "parseStoryboardCardDocument" in source,
          "Director import/export buttons do not use the structured storyboard schema")
    check("node.__myangDirectorPlan = null" in source
          and "source.value = MANUAL" in source,
          "imported cards can still be overwritten by the previous LLM plan")
    check("storyboard_metadata: node.__myangStoryboardMetadata || null" in source
          and "parseStoryboardMetadata(this)" in source,
          "imported storyboard provenance is not persisted with the workflow")
    check('const STORYBOARD_CARD_FORMAT = "minimax-h3-myang-director-storyboard"' in schema
          and "STORYBOARD_CARD_VERSION = 1" in schema,
          "storyboard files have no stable format identity or schema version")
    check("global_materials" in schema and "material_policy" in schema
          and "duration_seconds" in schema and "transition" in schema,
          "storyboard export drops structured card or material fields")
    check("imported_storyboard: true" in schema,
          "imported cards cannot be distinguished from copy/duplicate cards")


def test_action_transfer_plan_uses_one_prompt_and_covers_reference():
    plan = director._single_prompt_transfer_plan({"shots": [{
        "prompt": "同一个动作迁移提示词",
        "duration_seconds": 5,
        "assets": [{"kind": "image", "file": {"name": "actor.png"}}],
    }]}, overlap=22, segment_seconds=5, ref_frames=500)
    frames = [item["frames"] for item in plan["segments"]]
    check(plan["source"] == "myang_director_action_transfer",
          "action transfer did not use its dedicated plan")
    check(len(set(item["prompt"] for item in plan["segments"])) == 1,
          "action transfer changed the prompt between segments")
    check(frames[:-1] == [124] * 4 and frames[-1] == 107,
          "action transfer did not use full windows plus a fitted tail")
    check(plan["ref_frames_needed"] >= 500,
          "action transfer cropped the reference tail")
    check(plan["ref_frames_needed"] - 500 < 17,
          "action transfer padded more than one H3 frame-grid step")
    check(all(segment["asset_mode"] == "叠加全局素材"
              for segment in plan["segments"]),
          "action transfer auxiliary images/audio are not global")


def test_external_action_transfer_rejects_embedded_video():
    try:
        director._single_prompt_transfer_plan({"shots": [{
            "prompt": "动作迁移",
            "assets": [{"kind": "video", "file": {"name": "extra.mp4"}}],
        }]}, overlap=22, segment_seconds=5, ref_frames=240)
    except ValueError as error:
        check("只允许一个" in str(error), "wrong extra-video validation message")
        return
    raise AssertionError("action transfer accepted a second Director video")


def test_director_action_source_loads_uploaded_video_and_soundtrack():
    timeline = {"shots": [{
        "prompt": "统一迁移动作",
        "assets": [{"kind": "video", "role": "action", "file": {
            "name": "motion.mp4", "subfolder": "Myang_node/director/shot_1"}}],
    }]}
    frames = torch.zeros(240, 1, 1, 3)
    soundtrack = {"waveform": torch.zeros(1, 1, 320000), "sample_rate": 32000}
    original = director._load_director_action_video
    director._load_director_action_video = lambda asset: (frames, soundtrack)
    try:
        plan_json, loaded_frames, loaded_audio = director.H3DirectorActionSource().load(
            json.dumps(timeline), "", 5, 22)
    finally:
        director._load_director_action_video = original
    plan = json.loads(plan_json)
    check(loaded_frames is frames and loaded_audio is soundtrack,
          "Director action source did not preserve the uploaded video/audio")
    check(plan["segment_count"] == 3 and plan["reference_tail_pad"] is True,
          "uploaded action video did not create an automatic plan")
    check(all(not any(asset["kind"] == "video" for asset in segment["assets"])
              for segment in plan["segments"]),
          "uploaded action source leaked into per-segment reference assets")


def test_reference_clip_and_audio_cover_unaligned_tail():
    frames = torch.arange(10, dtype=torch.float32).reshape(10, 1, 1, 1)
    clip = shot_media_module.H3ReferenceClip().slice(frames, 7, 6)[0]
    check(clip.shape[0] == 6, "reference tail was not padded to requested length")
    check(clip[:, 0, 0, 0].tolist() == [7, 8, 9, 9, 9, 9],
          "reference tail padding did not repeat the final frame")
    audio = {"waveform": torch.arange(20, dtype=torch.float32).reshape(1, 1, 20),
             "sample_rate": 10}
    sliced = shot_media_module.H3ReferenceAudioClip().slice(
        audio, start_frame=24, frame_count=48, fps=24.0)[0]
    check(sliced["waveform"].shape[-1] == 20,
          "reference audio slice has the wrong requested duration")
    check(sliced["waveform"][0, 0, :10].tolist() == list(range(10, 20)),
          "reference audio did not start at the matching frame time")
    check(torch.count_nonzero(sliced["waveform"][..., 10:]) == 0,
          "out-of-range reference audio tail was not silence padded")


def test_reference_video_resolution_preserves_aspect_and_caps_1080p():
    original = shot_media_module.reference_video_size(
        3840, 2160, shot_media_module.REFERENCE_VIDEO_ORIGINAL)
    check(original == (3840, 2160), "original reference resolution was not preserved")
    landscape = shot_media_module.reference_video_size(3840, 2160, "1080P")
    portrait = shot_media_module.reference_video_size(1080, 1920, "720P")
    custom = shot_media_module.reference_video_size(
        3000, 3000, "自定义", 1920, 1920)
    check(landscape == (1920, 1080), "landscape reference did not cap at 1080P")
    check(portrait == (720, 1280), "portrait preset lost its source aspect")
    check(custom == (1080, 1080), "custom square reference exceeded the 1080P cap")


def test_director_preserves_per_shot_materials():
    plan = director._timeline_plan({"version": 2, "shots": [{
        "prompt": "@图片1 在雨中看向 @视频1",
        "duration_seconds": 5,
        "asset_mode": "仅本镜头",
        "assets": [
            {"kind": "image", "label": "人物", "file": {
                "name": "person.png", "subfolder": "Myang_node/director/shot_1"}},
            {"kind": "video", "role": "action", "label": "动作", "file": {
                "name": "motion.mp4", "subfolder": "Myang_node/director/shot_1"}},
        ],
    }]}, 22)
    segment = plan["segments"][0]
    check(segment["asset_mode"] == "仅本镜头", "shot asset mode was lost")
    check([asset["kind"] for asset in segment["assets"]] == ["image", "video"],
          "shot materials were not preserved in the execution plan")
    check(segment["assets"][1]["role"] == "action",
          "shot action-video role was not preserved")


def test_shot_material_path_cannot_escape_input():
    try:
        shot_media_module._safe_input_path({
            "file": {"name": "secret.mp4", "subfolder": "../../outside"}})
    except ValueError:
        return
    raise AssertionError("shot material path traversal was accepted")


def test_director_expands_existing_public_nodes():
    node = director.H3Director()
    timeline = json.dumps({
        "version": 3,
        "shots": [{"prompt": "已固定分镜", "duration_seconds": 5,
                   "fixed_from_plan": True}],
        "plan_snapshot": {"segments": [{"prompt": "旧的 LLM 快照"}]},
    }, ensure_ascii=False)
    result = node.run(
        h3=object(), model=object(), sampler=object(),
        source_mode=director.DIRECTOR_TIMELINE,
        timeline_json=timeline, script_fallback="", total_seconds=5,
        segment_seconds=5, llm_enabled=False,
        llm_service="未配置 LLM 服务", task_mode=legacy.TASK_FRESH,
        resolution="480P", aspect_ratio="16:9", width=864, height=480,
        steps=25, denoise=1.0, scheduler="simple", noise_seed=0,
        context_length="22", ref_image_size="匹配生成分辨率",
        save_segments=False, segment_prefix="video/test",
        save_raw_segments=False, unique_id="1776")
    graph = result["expand"]
    classes = [entry["class_type"] for entry in graph.values()]
    check("H3ScriptSplitter" not in classes,
          "manual storyboard rerun still created the LLM splitter")
    check(classes.count("H3LongVideo") == 1,
          "Director did not compose the existing long-video node")
    check(classes.count("H3DirectorPlanValue") == 1,
          "Director plan output is not linked")
    literal = next(entry for entry in graph.values()
                   if entry["class_type"] == "H3DirectorPlanValue")
    check(literal["inputs"]["progress_owner"] == "1776",
          "Director instance id does not reach its progress event pipeline")
    frozen_plan = json.loads(literal["inputs"]["plan_json"])
    check(frozen_plan["segments"][0]["prompt"] == "已固定分镜",
          "manual rerun used the saved LLM snapshot instead of the editable storyboard card")
    long_node = next(entry for entry in graph.values()
                     if entry["class_type"] == "H3LongVideo")
    check("llm_service" not in long_node["inputs"],
          "Director still forwards its LLM service into H3LongVideo")


def test_director_action_mode_builds_automatic_reference_plan():
    node = director.H3Director()
    ref_video = torch.zeros(500, 1, 1, 3)
    result = node.run(
        h3=object(), model=object(), sampler=object(),
        source_mode=director.DIRECTOR_TIMELINE,
        timeline_json='{"shots":[{"prompt":"全段一致"}]}',
        script_fallback="", total_seconds=999, segment_seconds=5,
        llm_enabled=True, llm_service="未配置 LLM 服务",
        task_mode=legacy.TASK_TRANSFER, resolution="480P",
        aspect_ratio="16:9", width=864, height=480,
        steps=25, denoise=1.0, scheduler="simple", noise_seed=0,
        context_length="22", ref_image_size="匹配生成分辨率",
        save_segments=False, segment_prefix="video/test",
        save_raw_segments=False, ref_video=ref_video)
    graph = result["expand"]
    check(not any(entry["class_type"] == "H3ScriptSplitter"
                  for entry in graph.values()),
          "action transfer still used script/timeline splitting")
    literal = next(entry for entry in graph.values()
                   if entry["class_type"] == "H3DirectorPlanValue")
    plan = json.loads(literal["inputs"]["plan_json"])
    check(plan["segment_count"] == 5,
          "reference video was not automatically split")
    check({segment["prompt"] for segment in plan["segments"]} == {"全段一致"},
          "automatic action segments did not share one prompt")
    long_node = next(entry for entry in graph.values()
                     if entry["class_type"] == "H3LongVideo")
    resize = next(entry for entry in graph.values()
                  if entry["class_type"] == "H3ReferenceResize")
    check(resize["inputs"]["image"] is ref_video
          and resize["inputs"]["resolution"] == shot_media_module.REFERENCE_VIDEO_ORIGINAL,
          "the action source did not reach reference resolution preprocessing")
    check(isinstance(long_node["inputs"]["ref_video"], list),
          "the resized action source did not reach H3LongVideo")


def test_director_action_mode_accepts_one_uploaded_video_without_external_input():
    node = director.H3Director()
    timeline = json.dumps({"shots": [{
        "prompt": "上传视频动作迁移",
        "assets": [{"kind": "video", "role": "action", "file": {
            "name": "motion.mp4", "subfolder": "Myang_node/director/shot_1"}}],
    }]})
    result = node.run(
        h3=object(), model=object(), sampler=object(),
        source_mode=director.DIRECTOR_TIMELINE,
        timeline_json=timeline, script_fallback="", total_seconds=5,
        segment_seconds=5, llm_enabled=False,
        llm_service="未配置 LLM 服务", task_mode=legacy.TASK_TRANSFER,
        resolution="480P", aspect_ratio="16:9", width=864, height=480,
        steps=25, denoise=1.0, scheduler="simple", noise_seed=0,
        context_length="22", ref_image_size="匹配生成分辨率",
        save_segments=False, segment_prefix="video/test",
        save_raw_segments=False)
    graph = result["expand"]
    sources = [entry for entry in graph.values()
               if entry["class_type"] == "H3DirectorActionSource"]
    check(len(sources) == 1, "Director did not create its uploaded action source")
    long_node = next(entry for entry in graph.values()
                     if entry["class_type"] == "H3LongVideo")
    check("ref_video" in long_node["inputs"] and "ref_audio" in long_node["inputs"],
          "uploaded action video/audio were not routed into H3LongVideo")


def test_director_action_mode_rejects_uploaded_and_external_video_conflict():
    node = director.H3Director()
    timeline = json.dumps({"shots": [{
        "prompt": "冲突测试",
        "assets": [{"kind": "video", "file": {"name": "motion.mp4"}}],
    }]})
    try:
        node.run(
            h3=object(), model=object(), sampler=object(),
            source_mode=director.DIRECTOR_TIMELINE,
            timeline_json=timeline, script_fallback="", total_seconds=5,
            segment_seconds=5, llm_enabled=False, llm_service="none",
            task_mode=legacy.TASK_TRANSFER, resolution="480P",
            aspect_ratio="16:9", width=864, height=480,
            steps=25, denoise=1.0, scheduler="simple", noise_seed=0,
            context_length="22", ref_image_size="匹配生成分辨率",
            save_segments=False, segment_prefix="video/test",
            save_raw_segments=False, ref_video=torch.zeros(240, 1, 1, 3))
    except ValueError as error:
        check("同时存在" in str(error), "wrong dual action-source error")
        return
    raise AssertionError("Director accepted uploaded and external action videos together")


def test_continuation_uses_previous_video_only_as_motion_context():
    plan = director._timeline_plan({"shots": [
        {"prompt": "续写第一段", "duration_seconds": 5},
        {"prompt": "续写第二段", "duration_seconds": 5},
    ]}, 22)
    ref_video = torch.zeros(60, 1, 1, 3)
    graph = legacy.H3LongVideo().run(
        h3=SimpleNamespace(video_vae=object(), audio_vae=object()),
        model=object(), sampler=object(), plan_json=json.dumps(plan),
        task_mode=legacy.TASK_CONTINUE, resolution="480P",
        aspect_ratio="16:9", width=864, height=480,
        steps=25, denoise=1.0, scheduler="simple", noise_seed=0,
        context_length="22", prompt_mode=legacy.MODE_DIRECT,
        media_prefix="", ref_image_size="匹配生成分辨率",
        save_segments=False, segment_prefix="video/test",
        save_raw_segments=False, ref_video=ref_video)["expand"]
    conditions = [entry for entry in graph.values()
                  if entry["class_type"] == "H3Condition"]
    check(all("ref_video" not in entry["inputs"] for entry in conditions),
          "continuation incorrectly sent its previous video as a reference")
    anchors = [entry for entry in graph.values()
               if entry["class_type"] == "H3AnchorContext"]
    check("context_frames" in anchors[0]["inputs"],
          "continuation first segment did not use pixel motion context")
    check("context_latent" in anchors[1]["inputs"],
          "continuation later segment did not use the previous latent")


def test_action_transfer_slices_video_and_audio_on_the_same_windows():
    plan = director._single_prompt_transfer_plan(
        {"shots": [{"prompt": "统一动作"}]}, 22, 5, 240)
    ref_video = torch.zeros(240, 1, 1, 3)
    ref_audio = {"waveform": torch.zeros(1, 1, 320000), "sample_rate": 32000}
    graph = legacy.H3LongVideo().run(
        h3=SimpleNamespace(video_vae=object(), audio_vae=object()),
        model=object(), sampler=object(), plan_json=json.dumps(plan),
        task_mode=legacy.TASK_TRANSFER, resolution="480P",
        aspect_ratio="16:9", width=864, height=480,
        steps=25, denoise=1.0, scheduler="simple", noise_seed=0,
        context_length="22", prompt_mode=legacy.MODE_DIRECT,
        media_prefix="", ref_image_size="匹配生成分辨率",
        save_segments=False, segment_prefix="video/test",
        save_raw_segments=False, ref_video=ref_video,
        ref_audio=ref_audio)["expand"]
    video_slices = [entry for entry in graph.values()
                    if entry["class_type"] == "H3ReferenceClip"]
    audio_slices = [entry for entry in graph.values()
                    if entry["class_type"] == "H3ReferenceAudioClip"]
    check(len(video_slices) == plan["segment_count"] == len(audio_slices),
          "action video/audio were not split once per generated segment")
    check([(entry["inputs"]["start_frame"], entry["inputs"]["frame_count"])
           for entry in video_slices]
          == [(entry["inputs"]["start_frame"], entry["inputs"]["frame_count"])
              for entry in audio_slices],
          "action video and audio segment windows diverged")
    conditions = [entry for entry in graph.values()
                  if entry["class_type"] == "H3Condition"]
    check(all("ref_video" in entry["inputs"] and "ref_audio" in entry["inputs"]
              for entry in conditions),
          "synchronized action video/audio did not reach H3 conditioning as a pair")


def test_action_transfer_resume_keeps_absolute_windows_and_numbering():
    full = director._single_prompt_transfer_plan(
        {"shots": [{"prompt": "统一动作"}]}, 22, 5, 700)
    resumed = director._single_prompt_transfer_plan(
        {"shots": [{"prompt": "统一动作"}]}, 22, 5, 700, start_segment=5)
    check(full["segment_count"] > 5, "test needs a reference longer than 5 segments")
    check(resumed["segment_count"] == full["segment_count"] - 4,
          "resuming did not drop the already generated head")
    check(resumed["resume_start_segment"] == 5
          and resumed["total_segments_planned"] == full["segment_count"],
          "resume plan lost its absolute segment bookkeeping")
    first = resumed["segments"][0]
    check(first["index"] == 5 and first["ref_start_frame"]
          == full["segments"][4]["ref_start_frame"],
          "resumed segment 5 did not keep the window it had in a full run")
    check([segment["frames"] for segment in resumed["segments"]]
          == [segment["frames"] for segment in full["segments"][4:]],
          "resuming changed the frame counts of the remaining segments")
    try:
        director._single_prompt_transfer_plan(
            {"shots": [{"prompt": "统一动作"}]}, 22, 5, 700,
            start_segment=full["segment_count"] + 1)
    except ValueError as error:
        check("起始段" in str(error), "wrong out-of-range start segment error")
    else:
        raise AssertionError("plan accepted a start segment past the last one")


def test_long_video_resume_anchors_context_video_without_using_it_as_reference():
    plan = director._single_prompt_transfer_plan(
        {"shots": [{"prompt": "统一动作"}]}, 22, 5, 700, start_segment=5)
    ref_video = torch.zeros(700, 1, 1, 3)
    context_video = torch.zeros(125, 1, 1, 3)
    graph = legacy.H3LongVideo().run(
        h3=SimpleNamespace(video_vae=object(), audio_vae=object()),
        model=object(), sampler=object(), plan_json=json.dumps(plan),
        task_mode=legacy.TASK_TRANSFER, resolution="480P",
        aspect_ratio="16:9", width=864, height=480,
        steps=25, denoise=1.0, scheduler="simple", noise_seed=0,
        context_length="22", prompt_mode=legacy.MODE_DIRECT,
        media_prefix="", ref_image_size="匹配生成分辨率",
        save_segments=True, segment_prefix="video/test",
        save_raw_segments=False, ref_video=ref_video,
        context_video=context_video)["expand"]

    clips = [entry for entry in graph.values()
             if entry["class_type"] == "H3ReferenceClip"]
    check(clips[0]["inputs"]["start_frame"]
          == plan["segments"][0]["ref_start_frame"] > 0,
          "the resumed first segment still sliced the reference from frame 0")
    check(all(entry["inputs"]["image"] is ref_video for entry in clips),
          "the previous cut leaked into the ref2va reference channel")

    anchors = [entry for entry in graph.values()
               if entry["class_type"] == "H3AnchorContext"]
    check(len(anchors) == plan["segment_count"],
          "the resumed first segment was generated without a seam anchor")
    tails = [entry for entry in graph.values()
             if entry["class_type"] == "ImageFromBatch"
             and entry["inputs"]["image"] is context_video]
    check(len(tails) == 1 and tails[0]["inputs"]["start_frame" if
          "start_frame" in tails[0]["inputs"] else "batch_index"] == 125 - 22,
          "the context video tail was not the anchor window")

    saves = [entry["inputs"]["filename_prefix"] for entry in graph.values()
             if entry["class_type"] == "SaveVideo"]
    check(any("第05段" in name for name in saves)
          and not any("第01段" in name for name in saves),
          "resumed segments were renumbered from 1 on disk")


def test_director_requires_previous_cut_when_resuming():
    node = director.H3Director()
    timeline = '{"shots":[{"prompt":"统一动作"}]}'
    common = dict(
        h3=object(), model=object(), sampler=object(),
        source_mode=director.DIRECTOR_TIMELINE, timeline_json=timeline,
        script_fallback="", total_seconds=30, segment_seconds=5,
        llm_enabled=False, llm_service="未配置 LLM 服务",
        task_mode=legacy.TASK_TRANSFER, resolution="480P",
        aspect_ratio="16:9", width=864, height=480, steps=25, denoise=1.0,
        scheduler="simple", noise_seed=0, context_length="22",
        ref_image_size="匹配生成分辨率", save_segments=False,
        segment_prefix="video/test", save_raw_segments=False,
        ref_video=torch.zeros(700, 1, 1, 3))
    try:
        node.run(**common, **{"起始段": 5})
    except ValueError as error:
        check("前段视频" in str(error), "wrong missing-context error: %s" % error)
    else:
        raise AssertionError("Director resumed without the previous cut")

    try:
        node.run(**common, **{"起始段": 1, "前段视频": torch.zeros(125, 1, 1, 3)})
    except ValueError as error:
        check("起始段" in str(error), "wrong stale-context error: %s" % error)
    else:
        raise AssertionError("Director accepted a context cut while starting at 1")

    graph = node.run(**common, **{
        "起始段": 5, "前段视频": torch.zeros(125, 1, 1, 3)})["expand"]
    long_node = next(entry for entry in graph.values()
                     if entry["class_type"] == "H3LongVideo")
    check("context_video" in long_node["inputs"],
          "the previous cut never reached H3LongVideo")
    check(long_node["inputs"]["context_video"]
          is not long_node["inputs"].get("ref_video"),
          "the previous cut was wired into the reference video channel")


GLOBAL_TIMELINE = json.dumps({
    "shots": [{"prompt": "镜头一", "duration_seconds": 5}],
    "global_assets": [
        {"kind": "image", "label": "女主角正面照",
         "file": {"name": "hero.png", "subfolder": "Myang_node/director/__global__"}},
        {"kind": "audio", "label": "主题曲",
         "file": {"name": "bgm.wav", "subfolder": "Myang_node/director/__global__"}},
    ],
})


def _director_script_run(node, timeline, **overrides):
    inputs = dict(
        h3=object(), model=object(), sampler=object(),
        source_mode=director.DIRECTOR_SCRIPT, timeline_json=timeline,
        script_fallback="一段长剧本", total_seconds=20, segment_seconds=5,
        llm_enabled=True, llm_service="未配置 LLM 服务",
        task_mode=legacy.TASK_FRESH, resolution="480P", aspect_ratio="16:9",
        width=864, height=480, steps=25, denoise=1.0, scheduler="simple",
        noise_seed=0, context_length="22", ref_image_size="匹配生成分辨率",
        save_segments=False, segment_prefix="video/test",
        save_raw_segments=False)
    inputs.update(overrides)
    return node.run(**inputs)["expand"]


def test_script_mode_feeds_director_uploads_to_the_splitter_and_the_loop():
    graph = _director_script_run(director.H3Director(), GLOBAL_TIMELINE)
    bundles = [entry for entry in graph.values()
               if entry["class_type"] == "H3ShotMedia"]
    check(len(bundles) == 1,
          "script mode built no shared media bundle for the Director uploads")
    assets = json.loads(bundles[0]["inputs"]["assets_json"])
    check([asset["kind"] for asset in assets] == ["image", "audio"],
          "Director global uploads did not survive normalization")
    check(bundles[0]["inputs"]["asset_mode"] == "叠加全局素材",
          "shared uploads would drop an upstream Media Agent bundle")
    check("media" not in bundles[0]["inputs"],
          "no Media Agent is connected, so none should be wired in")

    splitter = next(entry for entry in graph.values()
                    if entry["class_type"] == "H3ScriptSplitter")
    long_node = next(entry for entry in graph.values()
                     if entry["class_type"] == "H3LongVideo")
    check(isinstance(splitter["inputs"].get("media"), list),
          "the LLM splitter never saw the shared materials, so it cannot "
          "assign @图片N per segment")
    check(splitter["inputs"]["media"] == long_node["inputs"]["media"],
          "the splitter and the render loop disagreed about the material set")


def test_shared_uploads_stack_after_a_connected_media_agent():
    agent_bundle = object()
    graph = _director_script_run(
        director.H3Director(), GLOBAL_TIMELINE, media=agent_bundle)
    bundle = next(entry for entry in graph.values()
                  if entry["class_type"] == "H3ShotMedia")
    check(bundle["inputs"].get("media") is agent_bundle,
          "the Agent bundle was dropped instead of being extended")
    long_node = next(entry for entry in graph.values()
                     if entry["class_type"] == "H3LongVideo")
    check(long_node["inputs"]["media"] is not agent_bundle,
          "H3LongVideo still received the bare Agent bundle")


def test_manifest_numbering_matches_the_generator_and_survives_the_cache_key():
    bundle = media_catalog.MyangMediaCatalog(assets=(
        media_catalog.MyangMediaAsset(
            1, "image", torch.zeros(1, 8, 8, 3), "hero.png", "女主角正面照"),
        media_catalog.MyangMediaAsset(
            2, "video", torch.zeros(4, 8, 8, 3), "dance.mp4", "舞蹈动作"),
        media_catalog.MyangMediaAsset(
            3, "image", torch.zeros(1, 8, 8, 3), "street.png", "夜市街景"),
    ))
    rows = list(legacy.core.media_rows(bundle))
    check([(kind, ordinal) for kind, ordinal, _s, _f in rows]
          == [("image", 1), ("image", 2), ("video", 1)],
          "media_rows did not renumber per type the way H3Condition counts")
    manifest = legacy._format_media_manifest(bundle)
    check("@图片2：" in manifest and "夜市街景" in manifest,
          "the manifest hid the subject the LLM needs to match assets: %s" % manifest)
    check("@视频1" in manifest and "@视频3" not in manifest,
          "the manifest numbered the clip by catalog slot instead of by type")
    check(legacy._cache_key("剧本", 4, "svc", 0, manifest)
          != legacy._cache_key("剧本", 4, "svc", 0, ""),
          "changing the materials would reuse a split that names old assets")


def test_shared_video_uploads_are_rejected_for_continuation():
    timeline = json.dumps({
        "shots": [{"prompt": "镜头一", "duration_seconds": 5}],
        "global_assets": [{"kind": "video", "label": "前文",
                           "file": {"name": "clip.mp4"}}],
    })
    try:
        _director_script_run(
            director.H3Director(), timeline,
            task_mode=legacy.TASK_CONTINUE, ref_video=torch.zeros(60, 1, 1, 3))
    except ValueError as error:
        check("公共素材" in str(error), "wrong shared-video error: %s" % error)
        return
    raise AssertionError("continuation accepted a shared video upload")


def test_director_forwards_skill_and_vision_settings_to_the_splitter():
    graph = _director_script_run(
        director.H3Director(), GLOBAL_TIMELINE,
        **{"skill_preset": "h3-prompt-writing",
           "skill_text": "每段必须以 [Shot N] 开头",
           "vlm_service": "some-vlm"})
    splitter = next(entry for entry in graph.values()
                    if entry["class_type"] == "H3ScriptSplitter")
    check(splitter["inputs"].get("skill_preset") == "h3-prompt-writing"
          and splitter["inputs"].get("skill_text") == "每段必须以 [Shot N] 开头"
          and splitter["inputs"].get("vlm_service") == "some-vlm",
          "the Director kept the Skill/VLM settings to itself")

    required = director.H3Director.INPUT_TYPES()["required"]
    check(list(required)[-1] == "vlm_service",
          "Skill/VLM controls no longer form the stable tail of Director widgets")
    presets = required["skill_preset"][0]
    check(presets[0] == legacy.SKILL_PRESET_AUTO and "none" in presets,
          "the Skill dropdown lost auto/none routing")

    # Transfer mode writes its own single prompt and never calls the splitter,
    # so a Skill there would be dead weight.
    transfer = director.H3Director().run(
        h3=object(), model=object(), sampler=object(),
        source_mode=director.DIRECTOR_TIMELINE,
        timeline_json='{"shots":[{"prompt":"统一动作"}]}', script_fallback="",
        total_seconds=30, segment_seconds=5, llm_enabled=True,
        llm_service="未配置 LLM 服务", task_mode=legacy.TASK_TRANSFER,
        resolution="480P", aspect_ratio="16:9", width=864, height=480,
        steps=25, denoise=1.0, scheduler="simple", noise_seed=0,
        context_length="22", ref_image_size="匹配生成分辨率",
        save_segments=False, segment_prefix="video/test",
        save_raw_segments=False, ref_video=torch.zeros(300, 1, 1, 3),
        **{"skill_preset": "h3-prompt-writing"})["expand"]
    check(not any(entry["class_type"] == "H3ScriptSplitter"
                  for entry in transfer.values()),
          "action transfer should not route through the LLM splitter")


def test_resume_settings_are_ignored_outside_action_transfer():
    """A control the panel hides must never be able to block a run.

    `起始段` and `前段视频` only appear in the action-transfer panel, so after
    switching task modes their values are leftover state, not intent — and the
    user has no visible control to reset them with.
    """
    node = director.H3Director()
    for task in (legacy.TASK_FRESH, legacy.TASK_CONTINUE):
        graph = node.run(
            h3=object(), model=object(), sampler=object(),
            source_mode=director.DIRECTOR_TIMELINE,
            timeline_json='{"shots":[{"prompt":"镜头一","duration_seconds":5}]}',
            script_fallback="", total_seconds=20, segment_seconds=5,
            llm_enabled=False, llm_service="未配置 LLM 服务", task_mode=task,
            resolution="480P", aspect_ratio="16:9", width=864, height=480,
            steps=25, denoise=1.0, scheduler="simple", noise_seed=0,
            context_length="22", ref_image_size="匹配生成分辨率",
            save_segments=False, segment_prefix="video/test",
            save_raw_segments=False,
            ref_video=torch.zeros(300, 1, 1, 3) if task == legacy.TASK_CONTINUE else None,
            **{"起始段": 6, "前段视频": torch.zeros(125, 1, 1, 3)})["expand"]
        long_node = next(entry for entry in graph.values()
                         if entry["class_type"] == "H3LongVideo")
        check("context_video" not in long_node["inputs"],
              "%s leaked the stale resume context into the render loop" % task)

    source = (PACKAGE_DIR / "web" / "h3_director_ui.js").read_text("utf-8")
    check('by["起始段"].value = 1' in source,
          "the frontend never resets the stale start segment when leaving transfer")


def test_director_accepts_turbo_on_its_single_model_input():
    node = director.H3Director()
    timeline = '{"shots":[{"prompt":"镜头一","duration_seconds":5}]}'
    turbo_model = SimpleNamespace(model_options={turbo.TURBO_MARKER: {
        "profile": turbo.PROFILE_REF_4_V01,
        "recommended_steps": 4,
        "allowed_steps": (4,),
        "shift_video": 12.0,
        "shift_audio": 3.0,
        "task_family": "ref2va",
    }})
    result = node.run(
        h3=object(), model=turbo_model, sampler=object(),
        source_mode=director.DIRECTOR_TIMELINE,
        timeline_json=timeline, script_fallback="", total_seconds=5,
        segment_seconds=5, llm_enabled=False,
        llm_service="未配置 LLM 服务", task_mode=legacy.TASK_FRESH,
        resolution="480P", aspect_ratio="16:9", width=864, height=480,
        steps=25, denoise=0.5, scheduler="beta", noise_seed=0,
        context_length="22", ref_image_size="匹配生成分辨率",
        save_segments=False, segment_prefix="video/test",
        save_raw_segments=False)
    long_node = next(entry for entry in result["expand"].values()
                     if entry["class_type"] == "H3LongVideo")
    inputs = long_node["inputs"]
    check(inputs["model"] is turbo_model, "Director ignored its Turbo model input")
    check(inputs["steps"] == 25 and inputs["scheduler"] == "simple",
          "Director silently replaced the manually selected Turbo NFE")
    check(inputs["denoise"] == 1.0,
          "Director did not force full-denoise Turbo sampling")


def test_director_keeps_user_selected_step_inside_turbo_allowed_profile():
    turbo_model = SimpleNamespace(model_options={turbo.TURBO_MARKER: {
        "profile": turbo.PROFILE_8_V1,
        "recommended_steps": 8,
        "allowed_steps": (8, 4),
    }})
    result = director.H3Director().run(
        h3=object(), model=turbo_model, sampler=object(),
        source_mode=director.DIRECTOR_TIMELINE,
        timeline_json='{"shots":[{"prompt":"四步合法档","duration_seconds":5}]}',
        script_fallback="", total_seconds=5, segment_seconds=5,
        llm_enabled=False, llm_service="未配置 LLM 服务",
        task_mode=legacy.TASK_FRESH, resolution="480P", aspect_ratio="16:9",
        width=864, height=480, steps=4, denoise=0.5, scheduler="beta",
        noise_seed=0, context_length="22", ref_image_size="匹配生成分辨率",
        save_segments=False, segment_prefix="video/test", save_raw_segments=False)
    long_node = next(entry for entry in result["expand"].values()
                     if entry["class_type"] == "H3LongVideo")
    check(long_node["inputs"]["steps"] == 4,
          "Director overwrote a valid user-selected Turbo NFE")


def test_director_keeps_queued_legacy_recommended_steps_compatible():
    turbo_model = SimpleNamespace(model_options={turbo.TURBO_MARKER: {
        "profile": turbo.PROFILE_8_V1,
        "recommended_steps": 8,
        "allowed_steps": (8, 4),
    }})
    result = director.H3Director().run(
        h3=object(), model=object(), sampler=object(),
        source_mode=director.DIRECTOR_TIMELINE,
        timeline_json='{"shots":[{"prompt":"显式推荐步数","duration_seconds":5}]}',
        script_fallback="", total_seconds=5, segment_seconds=5,
        llm_enabled=False, llm_service="未配置 LLM 服务",
        task_mode=legacy.TASK_FRESH, resolution="480P", aspect_ratio="16:9",
        width=864, height=480, steps=25, denoise=1.0, scheduler="simple",
        noise_seed=0, context_length="22", ref_image_size="匹配生成分辨率",
        save_segments=False, segment_prefix="video/test", save_raw_segments=False,
        **{"Turbo联合模型": turbo_model, "Turbo推荐一采步数": 8})
    long_node = next(entry for entry in result["expand"].values()
                     if entry["class_type"] == "H3LongVideo")
    check(long_node["inputs"]["steps"] == 8,
          "queued legacy recommended_steps input did not remain compatible")


def test_director_plan_value_embeds_progress_owner():
    plan_json, fps = director.H3DirectorPlanValue().emit(
        '{"segment_count":1,"segments":[]}', "42")
    check(json.loads(plan_json)["progress_owner"] == "42" and fps == 24.0,
          "Director plan value did not embed the owner id")


def test_split_segment_prompts_are_pushed_to_the_panel():
    """Agent mode authors its prompts at runtime; the panel must receive them.

    Otherwise the only readout is one truncated line of whichever segment is
    currently sampling, and there is no way to check what the LLM wrote.
    """
    import server

    sent = []

    class Recorder:
        def send_sync(self, event, payload):
            sent.append((event, payload))

    previous = getattr(server.PromptServer, "instance", None)
    server.PromptServer.instance = Recorder()
    try:
        director.H3DirectorPlanValue().emit(json.dumps({
            "source": "myang_director_timeline",
            "frames_per_segment": 125, "segment_seconds_snapped": 5.0,
            "style_header": "夜市赛博风", "skill_source": "h3-prompt-writing",
            "segments": [
                {"index": 1, "brief": "开场", "prompt": "@图片1 少女走进夜市"},
                {"index": 2, "brief": "旋转", "prompt": "@图片1 旋转 <d>走吧</d>"},
            ],
        }, ensure_ascii=False), "8")
        # No owner id means no panel is listening; stay silent instead of
        # broadcasting to every Director on the canvas.
        director.H3DirectorPlanValue().emit('{"segments":[{"index":1}]}', "")
    finally:
        server.PromptServer.instance = previous

    events = [name for name, _payload in sent]
    check(events == ["myh3_director_plan"],
          "expected exactly one plan broadcast, got %s" % events)
    payload = sent[0][1]
    check(payload["owner_id"] == "8" and payload["segment_count"] == 2,
          "the plan broadcast lost its owner or its segments")
    check(payload["style_header"] == "夜市赛博风"
          and payload["skill_source"] == "h3-prompt-writing",
          "the panel cannot show which Skill and global style were used")
    first = payload["segments"][0]
    check(first["prompt"] == "@图片1 少女走进夜市" and first["brief"] == "开场",
          "segment prompts did not survive the broadcast")
    check(first["frames"] == 125 and first["duration_seconds"] == 5.0,
          "per-segment length fell back to nothing when the plan only has "
          "plan-level defaults")

    source = (PACKAGE_DIR / "web" / "h3_director_ui.js").read_text("utf-8")
    check('api.addEventListener("myh3_director_plan"' in source,
          "the Director panel never subscribes to the plan broadcast")
    check("renderPromptInto(body, segment.prompt, materials)" in source,
          "segment prompts are not rendered with material chips")
    check("node.__myangDirectorPlanList?.isConnected) updateSegmentPlan(node)" in source,
          "receiving a plan rebuilds the whole panel and destroys the caret")


def test_director_broadcast_never_breaks_a_run():
    import server

    class Broken:
        def send_sync(self, event, payload):
            raise RuntimeError("socket closed")

    previous = getattr(server.PromptServer, "instance", None)
    server.PromptServer.instance = Broken()
    try:
        plan_json, _fps = director.H3DirectorPlanValue().emit(
            '{"segments":[{"index":1,"prompt":"x"}]}', "8")
    finally:
        server.PromptServer.instance = previous
    check(json.loads(plan_json)["progress_owner"] == "8",
          "a failed preview push must not take the plan down with it")


def test_director_builds_integrated_optional_detail_pass():
    node = director.H3Director()
    detail_model = object()
    result = node.run(
        h3=object(), model=object(), sampler=object(),
        source_mode=director.DIRECTOR_TIMELINE,
        timeline_json='{"shots":[{"prompt":"镜头一","duration_seconds":5}]}',
        script_fallback="", total_seconds=5, segment_seconds=5,
        llm_enabled=False, llm_service="未配置 LLM 服务",
        task_mode=legacy.TASK_FRESH, resolution="480P", aspect_ratio="16:9",
        width=864, height=480, steps=25, denoise=1.0,
        scheduler="simple", noise_seed=0, context_length="22",
        ref_image_size="匹配生成分辨率", save_segments=True,
        segment_prefix="video/test", save_raw_segments=True,
        **{"二采开启": True, "二采模型": detail_model,
           "二采模式": "放大 + 二采（推荐）", "二采分辨率": "832P",
           "二采放大方式": "neural_3d (神经3D Latent放大·推荐)",
           "二采Latent模型": "minimax_h3_latent_upscaler_3d_fp16.safetensors"})
    graph = result["expand"]
    detail_nodes = [entry for entry in graph.values()
                    if entry["class_type"] == "H3DetailSettings"]
    check(len(detail_nodes) == 1, "Director did not build its integrated detail settings")
    check(detail_nodes[0]["inputs"]["二采模型"] is detail_model,
          "Director did not route the detail base model")
    long_node = next(entry for entry in graph.values()
                     if entry["class_type"] == "H3LongVideo")
    check("二采设置" in long_node["inputs"],
          "integrated detail settings did not reach H3LongVideo")
    check(long_node["inputs"]["save_raw_segments"] is True,
          "Director discarded raw-segment saving while detail pass is enabled")


def test_director_disables_raw_segment_copy_when_detail_is_off():
    node = director.H3Director()
    result = node.run(
        h3=object(), model=object(), sampler=object(),
        source_mode=director.DIRECTOR_TIMELINE,
        timeline_json='{"shots":[{"prompt":"镜头一","duration_seconds":5}]}',
        script_fallback="", total_seconds=5, segment_seconds=5,
        llm_enabled=False, llm_service="none", task_mode=legacy.TASK_FRESH,
        resolution="480P", aspect_ratio="16:9", width=864, height=480,
        steps=25, denoise=1.0, scheduler="simple", noise_seed=0,
        context_length="22", ref_image_size="匹配生成分辨率",
        save_segments=True, segment_prefix="video/test", save_raw_segments=True,
        **{"二采开启": False})
    long_node = next(entry for entry in result["expand"].values()
                     if entry["class_type"] == "H3LongVideo")
    check(long_node["inputs"]["save_raw_segments"] is False,
          "raw pre-detail segments remained enabled while detail pass was off")


def test_long_video_uses_variable_shot_windows():
    plan = director._timeline_plan({"shots": [
        {"prompt": "镜头一", "duration_seconds": 5},
        {"prompt": "镜头二", "duration_seconds": 6},
    ]}, 22)
    graph = legacy.H3LongVideo().run(
        h3=SimpleNamespace(video_vae=object(), audio_vae=object()),
        model=object(), sampler=object(), plan_json=json.dumps(plan),
        task_mode=legacy.TASK_TRANSFER, resolution="480P",
        aspect_ratio="16:9", width=864, height=480,
        steps=25, denoise=1.0, scheduler="simple", noise_seed=0,
        context_length="22", prompt_mode=legacy.MODE_DIRECT,
        media_prefix="", llm_service="none",
        ref_image_size="匹配生成分辨率", save_segments=False,
        segment_prefix="video/test", save_raw_segments=False,
        ref_video=torch.zeros(plan["ref_frames_needed"], 1, 1, 3),
    )["expand"]
    windows = [entry["inputs"] for entry in graph.values()
               if entry["class_type"] == "ImageFromBatch"]
    expected_frames = [item["frames"] for item in plan["segments"]]
    check([item["length"] for item in windows] == expected_frames,
          "transfer slices ignored per-shot frame lengths")
    check([item["batch_index"] for item in windows]
          == [0, expected_frames[0] - 22],
          "variable transfer cursor is wrong")


def test_long_video_routes_each_shot_action_material_lazily():
    shots = []
    for index in range(2):
        shots.append({
            "prompt": "第%d镜头 @视频1" % (index + 1),
            "duration_seconds": 5 + index,
            "assets": [{
                "kind": "video", "role": "action",
                "file": {"name": "motion_%d.mp4" % index,
                         "subfolder": "Myang_node/director/shot_%d" % index},
            }],
        })
    plan = director._timeline_plan({"version": 2, "shots": shots}, 22)
    graph = legacy.H3LongVideo().run(
        h3=SimpleNamespace(video_vae=object(), audio_vae=object()),
        model=object(), sampler=object(), plan_json=json.dumps(plan),
        task_mode=legacy.TASK_TRANSFER, resolution="480P",
        aspect_ratio="16:9", width=864, height=480,
        steps=25, denoise=1.0, scheduler="simple", noise_seed=0,
        context_length="22", prompt_mode=legacy.MODE_DIRECT,
        media_prefix="", llm_service="none",
        ref_image_size="匹配生成分辨率", save_segments=False,
        segment_prefix="video/test", save_raw_segments=False,
    )["expand"]
    shot_media = [entry for entry in graph.values()
                  if entry["class_type"] == "H3ShotMedia"]
    check(len(shot_media) == 2, "per-shot files were not lazily resolved per segment")
    check([entry["inputs"]["required_frames"] for entry in shot_media]
          == [segment["frames"] for segment in plan["segments"]],
          "action material did not receive its shot-specific frame window")


def test_neural_checkpoint_contract_and_temporal_chunking():
    model = neural.LatentResizer3D(
        in_channels=24, in_blocks=1, out_blocks=1,
        channels=32, dropout=0.0, temporal_every=0)
    state = model.state_dict()
    config = neural._architecture(state)
    check(config["in_channels"] == 24 and config["channels"] == 32,
          "checkpoint architecture detection changed")
    clone = neural.LatentResizer3D(**config)
    clone.load_state_dict(state, strict=True)
    source = torch.randn(1, 24, 5, 2, 2)
    with torch.inference_mode():
        output = neural._forward_bounded(
            clone.eval(), source, 2.0, 4, 4, chunk_steps=2, overlap=2)
    check(tuple(output.shape) == (1, 24, 5, 4, 4),
          "temporal chunking changed T or target canvas")


def test_director_example_workflow_is_compact_and_wired():
    path = PACKAGE_DIR / "example_workflows" / "Minimax_H3_Myang_Director_CN.json"
    workflow = json.loads(path.read_text(encoding="utf-8"))
    nodes = {node["id"]: node for node in workflow["nodes"]}
    directors = [node for node in nodes.values() if node["type"] == "H3Director"]
    check(len(directors) == 1 and len(nodes) == 8,
          "Director example is missing or no longer compact")
    director_node = directors[0]
    linked_inputs = {item["name"] for item in director_node["inputs"]
                     if item.get("link") is not None}
    check({"h3", "model", "sampler", "二采模型"}.issubset(linked_inputs),
          "Director example did not wire the integrated detail model")
    check(director_node["widgets_values_named"]["二采开启"] is False,
          "Director example must keep the expensive detail pass opt-in")
    for link in workflow["links"]:
        source, source_slot = nodes[link[1]], int(link[2])
        target, target_slot = nodes[link[3]], int(link[4])
        check(source_slot < len(source.get("outputs") or []), "bad workflow source slot")
        check(target_slot < len(target.get("inputs") or []), "bad workflow target slot")


if __name__ == "__main__":
    tests = [
        test_director_registration_and_variable_timeline,
        test_director_panel_tracks_node_selection_and_resize,
        test_director_storyboard_card_import_export_ui,
        test_action_transfer_plan_uses_one_prompt_and_covers_reference,
        test_external_action_transfer_rejects_embedded_video,
        test_director_action_source_loads_uploaded_video_and_soundtrack,
        test_reference_clip_and_audio_cover_unaligned_tail,
        test_reference_video_resolution_preserves_aspect_and_caps_1080p,
        test_director_preserves_per_shot_materials,
        test_shot_material_path_cannot_escape_input,
        test_director_expands_existing_public_nodes,
        test_director_action_mode_builds_automatic_reference_plan,
        test_director_action_mode_accepts_one_uploaded_video_without_external_input,
        test_director_action_mode_rejects_uploaded_and_external_video_conflict,
        test_continuation_uses_previous_video_only_as_motion_context,
        test_action_transfer_slices_video_and_audio_on_the_same_windows,
        test_action_transfer_resume_keeps_absolute_windows_and_numbering,
        test_long_video_resume_anchors_context_video_without_using_it_as_reference,
        test_director_requires_previous_cut_when_resuming,
        test_script_mode_feeds_director_uploads_to_the_splitter_and_the_loop,
        test_shared_uploads_stack_after_a_connected_media_agent,
        test_manifest_numbering_matches_the_generator_and_survives_the_cache_key,
        test_shared_video_uploads_are_rejected_for_continuation,
        test_director_forwards_skill_and_vision_settings_to_the_splitter,
        test_resume_settings_are_ignored_outside_action_transfer,
        test_director_accepts_turbo_on_its_single_model_input,
        test_director_keeps_user_selected_step_inside_turbo_allowed_profile,
        test_director_keeps_queued_legacy_recommended_steps_compatible,
        test_director_plan_value_embeds_progress_owner,
        test_split_segment_prompts_are_pushed_to_the_panel,
        test_director_broadcast_never_breaks_a_run,
        test_director_builds_integrated_optional_detail_pass,
        test_director_disables_raw_segment_copy_when_detail_is_off,
        test_long_video_uses_variable_shot_windows,
        test_long_video_routes_each_shot_action_material_lazily,
        test_neural_checkpoint_contract_and_temporal_chunking,
        test_director_example_workflow_is_compact_and_wired,
    ]
    for test in tests:
        test()
        print("PASS", test.__name__)
