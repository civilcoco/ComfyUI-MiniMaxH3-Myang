import importlib
import json
import sys
import tempfile
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
detail = importlib.import_module("ComfyUI-MiniMaxH3-Myang.detail")
neural = importlib.import_module("ComfyUI-MiniMaxH3-Myang.latent_upscale_3d")
legacy = importlib.import_module("ComfyUI-MiniMaxH3-Myang.nodes")
native = importlib.import_module("ComfyUI-MiniMaxH3-Myang.nodes")
core = importlib.import_module("ComfyUI-MiniMaxH3-Myang.core")
shot_media_module = importlib.import_module("ComfyUI-MiniMaxH3-Myang.media")
turbo = importlib.import_module("ComfyUI-MiniMaxH3-Myang.turbo")


def check(value, message):
    if not value:
        raise AssertionError(message)


def test_director_registration_and_variable_timeline():
    check("H3Director" in package.NODE_CLASS_MAPPINGS,
          "Director is not registered")
    check({"H3Pass1CheckpointSave", "H3Pass1CheckpointLoad",
           "H3Pass1VideoEncode"}.issubset(package.NODE_CLASS_MAPPINGS),
          "pass-1 recovery nodes are not registered")
    director_optional = director.H3Director.INPUT_TYPES()["optional"]
    redundant = {"script", "二采设置", "Turbo联合模型", "Turbo推荐一采步数"}
    check(redundant.isdisjoint(director_optional),
          "Director still exposes redundant compatibility inputs")
    check({"一采断点模式", "一采成片", "一采成片音频"}.issubset(
        director_optional), "Director does not expose pass-1 recovery controls")
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


def test_director_timeline_isolated_by_task_mode():
    transfer = {
        "shots": [{"prompt": "动作提示", "duration_seconds": 5}],
        "global_assets": [{"kind": "image", "label": "动作角色",
                            "file": {"name": "transfer.png"}}],
    }
    fresh = {
        "shots": [{"prompt": "纯生成提示", "duration_seconds": 5}],
        "global_assets": [{"kind": "image", "label": "生成角色",
                            "file": {"name": "fresh.png"}}],
    }
    envelope = json.dumps({"version": 4, "active_mode": legacy.TASK_FRESH,
                           "modes": {legacy.TASK_TRANSFER: transfer,
                                      legacy.TASK_FRESH: fresh}},
                          ensure_ascii=False)
    check(director._timeline_shots(envelope, legacy.TASK_TRANSFER)[0]["prompt"]
          == "动作提示", "动作迁移读取了其他模式的提示词")
    check(director._timeline_shots(envelope, legacy.TASK_FRESH)[0]["prompt"]
          == "纯生成提示", "纯生成读取了其他模式的提示词")
    check(director._timeline_globals(envelope, legacy.TASK_TRANSFER) == [],
          "动作迁移重新暴露了已移除的独立公共素材桶")
    check(director._timeline_globals(envelope, legacy.TASK_FRESH)[0]["file"]["name"]
          == "fresh.png", "纯生成读取了其他模式的公共素材")
    try:
        director._timeline_shots(envelope, legacy.TASK_CONTINUE)
    except ValueError:
        pass
    else:
        raise AssertionError("缺失模式不应借用其他模式的分镜")


def test_director_transition_controls_motion_context_boundaries():
    plan = director._timeline_plan({"shots": [
        {"prompt": "镜头一", "duration_seconds": 5.0, "transition": "切镜"},
        {"prompt": "镜头二", "duration_seconds": 5.0, "transition": "承接"},
        {"prompt": "镜头三", "duration_seconds": 5.0, "transition": "切镜"},
    ]}, 22)
    check([item["transition"] for item in plan["segments"]]
          == ["开场", "承接", "切镜"],
          "manual storyboard transitions were not normalized into the run plan")
    expected = sum(item["frames"] for item in plan["segments"]) - 22
    check(plan["ref_frames_needed"] == expected,
          "cut boundaries still subtract a MotionContext overlap")
    cut_tail = director._slice_plan_from_segment(
        plan, 3, 22, has_context=False)
    check(cut_tail["segment_count"] == 1
          and cut_tail["segments"][0]["index"] == 3
          and cut_tail["resume_context_required"] is False,
          "starting from a cut did not remain an independent absolute segment")

    source = (PACKAGE_DIR / "web" / "h3_director_ui.js").read_text("utf-8")
    check("function renderShotTransition(node, shot, index)" in source
          and 'for (const option of ["承接", "切镜"])' in source,
          "manual storyboard cards have no per-boundary transition control")
    check("MotionContext 无缝续接" in source
          and "独立生成，不参考上一段末尾" in source,
          "the transition choices do not explain their generation behaviour")
    check('control.setAttribute("aria-pressed", String(active))' in source,
          "the segmented transition control has no accessible selected state")


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
    check('add.textContent = "＋ 插入对话"' in source
          and "createDialogueToolbar(editor, sync)" in source
          and '`[Chinese] ${selected || "请输入台词"}`' in source,
          "Director ordinary prompt inputs have no usable dialogue-block control")
    check('语法：<d>[Chinese] 台词</d>' in source
          and 'block.contentEditable = "true"' in source
          and '?.closest?.(".myh3-chip")' in source,
          "Director dialogue syntax is hidden or rendered dialogue blocks are not editable")
    check("event.preventDefault();" in source[source.index("function createDialogueToolbar"):
                                               source.index("function closeDirectorMenu")],
          "Director dialogue button destroys the contenteditable selection before insertion")
    check("function promptSelectionText(editor)" in source
          and 'if (dialogue) return `<d>${dialogue.textContent}</d>`;' in source
          and 'event.clipboardData?.setData("text/plain", text)' in source
          and "promptFragmentFromText(text, materialList())" in source,
          "Director dialogue copy/paste does not preserve the structured block")
    check("function selectedDialogueBlock(editor)" in source
          and "if (wholeDialogue) wholeDialogue.remove();" in source,
          "cutting a dialogue leaves an empty styled shell in the prompt")
    check('event.stopPropagation();' in source[source.index('editor.addEventListener("keydown"'):
                                                source.index('editor.addEventListener("paste"')],
          "Director editor shortcuts still bubble to the ComfyUI canvas")
    check("label.textContent = entry?.subject" in source,
          "Director material chips still expose filenames instead of optional subject names")
    long_source = (PACKAGE_DIR / "web" / "h3_longvideo_ui.js").read_text("utf-8")
    check("text.textContent = entry?.subject" in long_source,
          "Long-video material chips still expose filenames instead of optional subject names")
    check('api.addEventListener("myh3_progress"' in source
          and "renderDirectorProgressPanel(node)" in source
          and 'title.textContent = "生成进度"' in source
          and "panel.append(heading, bar, text, prompt, preview);" in source
          and "if (state.previewFile)" in source,
          "Director progress card no longer shows the current sampling step")
    check("position:sticky;top:0;z-index:40" in source,
          "Director progress card no longer stays visible while scrolling")
    check("function directorTemplateSource(node, taskMode = currentTask(node))" in source
          and "if (taskMode === TRANSFER)" in source
          and "globalAssets: []," in source
          and "const source = directorTemplateSource(node, taskMode)" in source,
          "action-transfer templates can still capture the manual storyboard bucket")
    check("const availableTemplates = () => (node.__myangDirectorTemplates || [])" in source
          and ".filter((item) => item.task_mode === currentTask(node))" in source,
          "Director template choices are not isolated by task mode")
    check("function materialQuota(entries, allowedKinds, limits = {})" in source
          and "图片 4/9" not in source
          and 'parts.push(`总计 ${total}/${MEDIA_TOTAL_LIMIT}`)' in source,
          "Director material cards have no live per-kind/total quota counter")
    check('detailControl(node, "二采连续Sigma", "连续 Sigma 二采（实验）"' in source
          and '"二采连续Sigma", "detail"' in source
          and "实验轨迹共" in source,
          "Director detail card has no opt-in continuous-Sigma experiment")
    check('api.addEventListener("executing"' in source
          and 'prepare: {start: 0.00' in source
          and "function markDirectorPreparing(node, activity)" in source
          and 'if (state.status === "running") return;' in source
          and 'stage === "preparing"' in source,
          "Director preparation can rewind an active sample/refine run")
    check('assembling: {start: 0.94' in source
          and 'exporting: {start: 0.98' in source
          and 'stage === "assembling"' in source
          and 'stage === "assembled"' in source
          and 'state.phase = "assembling"' in source
          and 'state.status = "done"' not in source[
              source.index('else if (stage === "done")'):
              source.index('if (detail?.preview_file)')],
          "Director still completes before segment merge/video export")
    check('modeCaption.textContent = "生成长度模式"' in source
          and '"整段生成（自动匹配视频长度）"' in source
          and 'modeSelect.onchange = () => setNativeWidget' in source,
          "action transfer still uses an ambiguous segmentation checkbox")
    check('"关闭",' in source[source.index('function renderFirstPassMemoryPanel'):
                                      source.index('function renderDetailPanel')],
          "Director UI hides the legacy 一采显存策略 value")
    transfer_start = source.index("function renderTransferPanel(node, root)")
    transfer_end = source.index("const DETAIL_FIELDS", transfer_start)
    transfer_source = source[transfer_start:transfer_end]
    check(transfer_source.index('promptLabel.textContent = "全片统一提示词"')
          < transfer_source.index("const prompt = createPromptEditor"),
          "the full-video prompt label is detached from its editor")
    check('const NATIVE_VIDEO_WIDGET = "video-preview"' in source
          and "function mountNativeVideoPreview(node, output = null)" in source
          and 'video.style.cssText = "display:block;width:100%;height:100%' in source,
          "Director does not embed and contain ComfyUI's native video player")
    check("function videoResultFromOutput(output)" in source
          and "function nativeVideoElement(node)" in source
          and "scheduleNativeVideoMount(this, output)" in source
          and "node.videoContainer?.replaceChildren?.()" not in source,
          "Director can still destroy or fail to recover the native video player")
    check('"二采显存策略": "二采显存策略"' in source
          and '"自动平衡（16GB推荐）"' in source
          and '"显存优先（832P保底）"' in source,
          "Director detail card lost its VRAM balance selector")
    check("images: undefined" not in source
          and "gifs: undefined" not in source
          and "videos: undefined" not in source,
          "Director still discards the native player payload")
    check("function outputWithoutNativeMediaPreview(output)" in source
          and 'for (const key of ["images", "gifs", "videos", "files"])' in source
          and "filtered[key] = []" in source
          and 'for (const envelope of ["output", "ui"])' in source
          and "function clearNativeStillPreviewSoon(node)" in source
          and "function clearNativeStillPreview(node)" in source
          and "function clearNativeStillPreviewState(node)" in source
          and "function installCanvasPreviewGuard(node)" in source
          and 'Object.defineProperty(node, "imgs"' in source
          and "function hideNativeVideoWidget(node)" in source
          and "hideNativeVideoWidget(this);" in source
          and "nodeType.prototype.onDrawBackground = function ()" in source
          and "clearNativeStillPreviewState(this);" in source,
          "Director still lets temporary sampling images create a bottom canvas preview")
    check('const stableSize = [Number(this.size?.[0] || 0)' in source
          and "this.setSize?.(stableSize);" in source,
          "video output can still resize the Director node")
    check('stop.textContent = "停止并释放"' in source
          and "await api.interrupt();" in source
          and "await requestDirectorMemoryRelease();" in source,
          "Director progress panel has no safe stop-and-release action")
    init_source = (PACKAGE_DIR / "__init__.py").read_text("utf-8")
    check('queue.set_flag("unload_models", True)' in init_source
          and 'queue.set_flag("free_memory", True)' in init_source
          and "finish_myang_cleanup(queue)" in init_source
          and "_clear_whisper_cache()" in init_source,
          "interrupted Director cleanup does not release executor and auxiliary caches")
    check(source.count("root.appendChild(renderOutputVideoPanel(node))") == 3,
          "not every Director mode puts the output player inside its panel")
    check("overflow-y:auto;scrollbar-gutter:stable" in source
          and "max-height:340px;overflow-y:auto" not in source
          and "max-height:470px;overflow-y:auto" not in source,
          "Director still uses nested scrolling instead of one card-level scrollbar")
    check("nodeHeight - (Number.isFinite(panelTop)" in source
          and "this.__myangDirectorPanelWidget = panel" in source
          and "getMaxHeight: () => Number.MAX_SAFE_INTEGER" in source
          and "element.style.height = cssHeight" in source,
          "Director card height is still capped instead of filling the resized node")
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
    check(".myh3-progress-preview" in progress_css
          and "height: 170px" in progress_css
          and "max-height: 170px" not in progress_css,
          "Director's progress preview must use a fixed 170px box, not a max-height "
          "cap: with only width:100% the box height comes from the image's intrinsic "
          "ratio, so every frame swap relayouts and visibly flashes")
    check("__myangPreviewToken" in source and "warm.onload = swap" in source,
          "Director progress preview swaps src without warming the next frame first")
    check('root.className = "myh3-director-root"' in source
          and ".myh3-director-root > *" in progress_css
          and "flex-shrink: 0" in progress_css
          and source.count("flex:0 0 auto;min-height:38px") >= 2,
          "Director flex children can still collapse closed setting cards into border lines")
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
    check("active.plan_snapshot = normalizePlanSnapshot(node.__myangDirectorPlan)" in source
          and "node.__myangDirectorModeBuckets" in source
          and "loadModeBucket(this, modeTaskValue(this))" in source,
          "the latest LLM plan is not persisted/restored per task mode")
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
    check("const ENHANCEMENT_FIELDS" in source
          and "renderEnhancementPanel(node)" in source
          and '"H3 小脸精修"' in source
          and '"MAINodes 高速动作修复"' in source
          and '"角色五视图分镜"' in source,
          "Director has no compact opt-in enhancement card")
    timeline_start = source.index("function renderTimeline(node)")
    timeline_end = source.index("function refresh(node)", timeline_start)
    timeline_source = source[timeline_start:timeline_end]
    transfer_branch = timeline_source.index("if (transferring) {")
    check(0 <= timeline_source.index("root.appendChild(renderDetailPanel(node));") < transfer_branch
          and 0 <= timeline_source.index("root.appendChild(renderEnhancementPanel(node));") < transfer_branch,
          "second-pass or enhancement cards are still conditional on the source mode")
    detail_start = source.index("function renderDetailPanel(node)")
    enhancement_start = source.index("function renderEnhancementPanel(node)", detail_start)
    global_assets_start = source.index("function renderGlobalAssets(node", enhancement_start)
    detail_source = source[detail_start:enhancement_start]
    enhancement_source = source[enhancement_start:global_assets_start]
    check('panel.dataset.myangCollapsible = "detail"' in detail_source
          and 'panel.dataset.myangCollapsible = "enhancement"' in enhancement_source
          and 'document.createElement("summary")' not in detail_source
          and 'document.createElement("summary")' not in enhancement_source
          and 'min-height:38px' in detail_source
          and 'min-height:38px' in enhancement_source
          and 'setAttribute("aria-expanded"' in detail_source
          and 'setAttribute("aria-expanded"' in enhancement_source,
          "collapsed second-pass cards can still be flattened by global details/summary CSS")
    check('for (const name of ENHANCEMENT_FIELDS) hideWidget(by[name])' in source,
          "enhancement widgets are duplicated outside the Director card")
    check('width:118px;min-width:118px' in source,
          "enhancement switches can still be compressed by long option names")


def test_director_asset_preview_preserves_intrinsic_aspect_and_opens_images():
    source = (PACKAGE_DIR / "web" / "h3_director_ui.js").read_text("utf-8")
    asset_start = source.index("function renderShotAssets(node")
    asset_end = source.index("function modeNotice(", asset_start)
    asset_source = source[asset_start:asset_end]
    check("function openAssetPreview(asset, trigger = null)" in source,
          "Director materials have no shared image/video preview dialog")
    check('overlay.setAttribute("aria-modal", "true")' in source
          and 'event.key === "Escape"' in source,
          "material preview is not an accessible dismissible dialog")
    check('previewButton.onclick = () => openAssetPreview(asset, previewButton)' in asset_source,
          "image/video thumbnails do not open the full preview")
    check("object-fit:contain" in asset_source
          and "object-fit:cover" not in asset_source,
          "material thumbnails still crop portrait media into a landscape frame")
    check("preview.controls = false" in asset_source,
          "video thumbnail still competes with the full-size native player")


def test_director_widget_changes_preserve_focus_and_caret():
    source = (PACKAGE_DIR / "web" / "h3_director_ui.js").read_text("utf-8")
    callback_start = source.index("const syncedCallback = (...args)")
    callback_end = source.index("return value;", callback_start)
    callback_source = source[callback_start:callback_end]
    check("syncDirectorWidgetChange(this, item.name)" in callback_source
          and "refresh(this)" not in callback_source,
          "ordinary widget callbacks still rebuild the whole Director card")
    native_setter_start = source.index("function setNativeWidget(node, name, value)")
    native_setter_end = source.index("function detailControl", native_setter_start)
    check("syncDirectorWidgetChange(node, name)" in source[native_setter_start:native_setter_end]
          and "__myangDirectorSyncs" in source[native_setter_start:native_setter_end]
          and "refresh(node)" not in source[native_setter_start:native_setter_end],
          "mirrored controls still destroy focus or schedule duplicate refreshes")
    check("const DIRECTOR_SECTION_REFRESH" in source
          and "function captureDirectorView(node)" in source
          and "function restoreDirectorView(node, state)" in source
          and "root.replaceChildren();" in source,
          "Director has no scoped rendering or focus/caret preservation")
    detail_start = source.index("function renderDetailPanel(node)")
    detail_end = source.index("function renderAudioPanel(node)", detail_start)
    detail_source = source[detail_start:detail_end]
    check('"__myangDetailPanelExpanded", "block"' in detail_source
          and "bindDirectorCollapsible(" in detail_source
          and "renderTimeline(node);" not in detail_source,
          "folding the detail card still rebuilds and clears live progress")
    audio_start = source.index("function renderAudioPanel(node)")
    audio_end = source.index("function renderEnhancementPanel(node)", audio_start)
    audio_source = source[audio_start:audio_end]
    check('"__myangAudioPanelExpanded", "flex"' in audio_source
          and "bindDirectorCollapsible(" in audio_source
          and "renderTimeline(node);" not in audio_source,
          "folding the audio card still rebuilds and clears live progress")
    enhancement_start = source.index("function renderEnhancementPanel(node)")
    enhancement_end = source.index("function directorStatsText", enhancement_start)
    enhancement_source = source[enhancement_start:enhancement_end]
    check('"__myangEnhancementPanelExpanded", "flex"' in enhancement_source
          and "bindDirectorCollapsible(" in enhancement_source
          and "renderTimeline(node);" not in enhancement_source,
          "folding the enhancement card still rebuilds and clears live progress")
    timeline_start = source.index("function renderTimeline(node)")
    timeline_end = source.index("function refresh(node)", timeline_start)
    timeline_source = source[timeline_start:timeline_end]
    check("const progressPanel = node.__myangDirectorProgressEls?.panel || null;"
          in timeline_source
          and "const liveProgressPanel = progressPanel || renderDirectorProgressPanel(node);"
          in timeline_source
          and "root.appendChild(liveProgressPanel);"
          in timeline_source
          and "updateDirectorProgress(node);" in timeline_source,
          "a full Director form refresh still recreates and blanks live progress")


def test_director_storyboard_card_import_export_ui():
    source = (PACKAGE_DIR / "web" / "h3_director_ui.js").read_text("utf-8")
    schema = (PACKAGE_DIR / "web" / "h3_storyboard_cards.js").read_text("utf-8")
    check('from "./h3_storyboard_cards.js"' in source
          and "createStoryboardCardDocument" in source
          and "parseStoryboardCardDocument" in source,
          "Director does not import the structured storyboard card module")
    check('button("导入分镜卡"' in source and 'button("导出分镜卡"' in source,
          "Director manual-card toolbar has no structured import/export actions")
    check("createStoryboardCardDocument" in source
          and "parseStoryboardCardDocument" in source,
          "Director import/export buttons do not use the structured storyboard schema")
    check("node.__myangDirectorPlan = null" in source
          and "source.value = MANUAL" in source,
          "imported cards can still be overwritten by the previous LLM plan")
    check("active.storyboard_metadata = node.__myangStoryboardMetadata || null" in source
          and "node.__myangStoryboardMetadata = bucket.storyboard_metadata" in source,
          "imported storyboard provenance is not persisted with the workflow")
    check('const STORYBOARD_CARD_FORMAT = "minimax-h3-myang-director-storyboard"' in schema
          and "STORYBOARD_CARD_VERSION = 1" in schema,
          "storyboard files have no stable format identity or schema version")
    check("global_materials" in schema and "material_policy" in schema
          and "duration_seconds" in schema and "transition" in schema,
          "storyboard export drops structured card or material fields")
    check("imported_storyboard: true" in schema,
          "imported cards cannot be distinguished from copy/duplicate cards")


def test_director_card_folding_and_layer_persistence_ui():
    source = (PACKAGE_DIR / "web" / "h3_director_ui.js").read_text("utf-8")
    bucket = source.index("function normalizeModeBucket")
    whitelist = source[bucket:source.index("\nfunction ", bucket + 1)]
    # normalizeModeBucket rebuilds every shot from a fixed field list, so a key
    # missing here is dropped on reload. Layers were, which silently reverted a
    # layered card to its flat prompt on the next workflow load.
    check("normalizeSegmentLayers(shot?.layers)" in whitelist
          and "normalized.layers = layers" in whitelist,
          "card layers are dropped when the workflow reloads")
    check("collapsed: shot?.collapsed === true" in whitelist
          and "layers_open: shot?.layers_open === true" in whitelist,
          "card fold state does not survive a reload")
    check("__myangLayersOpen" not in source,
          "a transient __-prefixed flag is still being written into the workflow")

    check('button(shot.collapsed ? "▸" : "▾"' in source
          and "shot.collapsed = !shot.collapsed" in source,
          "storyboard cards have no per-card fold toggle")
    check('aria-expanded", shot.collapsed ? "false" : "true"' in source,
          "the fold toggle does not expose its state to assistive tech")
    check('button(anyOpen ? "全部折叠" : "全部展开"' in source
          and "for (const shot of shots) shot.collapsed = anyOpen;" in source,
          "there is no way to fold or unfold the whole storyboard at once")
    # A folded card must skip building its editor, not merely hide it: the cost
    # of a long storyboard is the contenteditable bodies and material menus.
    fold_guard = source.index("if (shot.collapsed) {")
    editor = source.index("const prompt = createPromptEditor(node, shot, {", fold_guard)
    check("renderCollapsedShotSummary(shot)" in source[fold_guard:editor]
          and "return;" in source[fold_guard:editor],
          "a folded card still builds its prompt editor")
    check("function renderCollapsedShotSummary(shot)" in source
          and "dialogueSeconds(entry)" in source,
          "the folded summary does not report dialogue against the shot length")


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


def test_action_transfer_uses_reference_duration_not_stale_template_total():
    plan = director._single_prompt_transfer_plan(
        {"shots": [{"prompt": "14秒动作迁移"}]}, overlap=22,
        segment_seconds=6, ref_frames=14 * 24, auto_segment=True)
    check(plan["segment_count"] == 3,
          "14-second action source with a 6-second ceiling did not create 3 segments")
    timeline = json.dumps({
        "shots": [{"prompt": "14秒动作迁移"}],
        "template_contract": {
            "active": True, "total_seconds": 5, "segment_seconds": 6,
        },
    }, ensure_ascii=False)
    contract = director._timeline_template_contract(timeline, legacy.TASK_TRANSFER)
    check(contract["total_seconds"] == 0 and contract["segment_seconds"] == 6,
          "legacy action template still overrides decoded reference-video duration")


def test_action_transfer_sampler_windows_never_exceed_the_time_ceiling():
    plan = director._single_prompt_transfer_plan(
        {"shots": [{"prompt": "14秒动作迁移，按7秒上限切分"}]},
        overlap=22, segment_seconds=7, ref_frames=14 * 24,
        auto_segment=True)
    ceiling = core.length_for(7, 24.0)
    check(plan["segment_count"] == 3,
          "the real sampler overlap was not included when enforcing the ceiling")
    check(max(item["frames"] for item in plan["segments"]) <= ceiling,
          "an action sampler window still exceeds the requested time ceiling")
    starts = [item["ref_start_frame"] for item in plan["segments"]]
    expected = [0]
    for item in plan["segments"][:-1]:
        expected.append(expected[-1] + item["frames"] - 22)
    check(starts == expected,
          "rebalanced action segments no longer begin on their overlap windows")
    check(plan["ref_frames_needed"] >= 14 * 24,
          "balanced action segments no longer cover the complete 14-second source")


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
    director._load_director_action_video = (
        lambda asset, **_options: (frames, soundtrack))
    try:
        plan_json, loaded_frames, loaded_audio = director.H3DirectorActionSource().load(
            json.dumps(timeline), "", 5, 22)
    finally:
        director._load_director_action_video = original
    plan = json.loads(plan_json)
    check(loaded_frames is frames and loaded_audio is soundtrack,
          "Director action source did not preserve the uploaded video/audio")
    check(plan["segment_count"] == 3 and plan["reference_tail_pad"] is True,
          "uploaded action video did not create an automatic plan within the hard window ceiling")
    check(all(not any(asset["kind"] == "video" for asset in segment["assets"])
              for segment in plan["segments"]),
          "uploaded action source leaked into per-segment reference assets")
    director._load_director_action_video = (
        lambda asset, **_options: (frames, soundtrack))
    try:
        whole_json, _, _ = director.H3DirectorActionSource().load(
            json.dumps(timeline), "", 5, 22, auto_segment="关闭")
    finally:
        director._load_director_action_video = original
    check(json.loads(whole_json)["segment_count"] == 1,
          "localized false value unexpectedly enabled automatic segmentation")
    director._load_director_action_video = (
        lambda asset, **_options: (frames, soundtrack))
    try:
        try:
            director.H3DirectorActionSource().load(
                json.dumps(timeline), "", 5, 22, target_frames=300)
        except ValueError as error:
            check("需要 300 帧" in str(error),
                  "short action reference produced the wrong rough-cut error")
        else:
            raise AssertionError("rough-cut action transfer accepted a short source")
    finally:
        director._load_director_action_video = original


def test_bounded_reference_decode_caps_the_canvas_before_the_stack():
    def canvas(width, height):
        return shot_media_module.reference_video_size(width, height, "1080P")

    # The budget check reads the machine's live free memory, so while ComfyUI
    # itself runs (20GB of models in pinned RAM), an affordable 9.8GiB decode
    # would "fail" — the test flaked on machine state, not on code. Pin a fixed
    # pot of memory for the whole test.
    class _FixedMemory:
        def __init__(self, available):
            self._available = int(available)

        def virtual_memory(self):
            return SimpleNamespace(available=self._available)

    real_psutil = sys.modules.get("psutil")
    sys.modules["psutil"] = _FixedMemory(64 * 1024 ** 3)
    try:
        _run_bounded_reference_decode_checks(canvas)
    finally:
        if real_psutil is None:
            sys.modules.pop("psutil", None)
        else:
            sys.modules["psutil"] = real_psutil


def _run_bounded_reference_decode_checks(canvas):
    stream = SimpleNamespace(
        frames=846, average_rate=60, duration=None, time_base=None)
    upright = SimpleNamespace(rotation=0, width=2160, height=3840)
    plan = core._bounded_plan(upright, canvas, stream, 2.5, "action.mp4")
    check(plan.canvas == (1080, 1920) and plan.turns == 0,
          "the reference decode did not adopt the selected 1080P canvas")
    check((plan.reformat["width"], plan.reformat["height"]) == (1080, 1920)
          and plan.reformat["format"] == "rgb24",
          "the decode still asks swscale for the source canvas, so a 4K clip is "
          "materialised at 4K before anything can shrink it")
    check(core._bounded_frame_estimate(stream, 2.5) == 338,
          "the 24fps decimation estimate disagrees with the decode loop")

    sideways = SimpleNamespace(rotation=90, width=3840, height=2160)
    turned = core._bounded_plan(sideways, canvas, stream, 2.5, "action.mp4")
    check(turned.canvas == (1080, 1920) and turned.turns == 1
          and (turned.reformat["width"], turned.reformat["height"]) == (1920, 1080),
          "a rotation-tagged clip was not scaled to the transposed canvas")

    unaffordable = SimpleNamespace(
        frames=100000, average_rate=60, duration=None, time_base=None)
    try:
        core._bounded_plan(upright, None, unaffordable, 2.5, "action.mp4")
    except ValueError as error:
        check("参考视频分辨率" in str(error),
              "an unaffordable decode failed without naming the control that "
              "fixes it")
    else:
        raise AssertionError(
            "an unaffordable reference decode was not refused up front")


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
        save_raw_segments=False, **{"参考视频分辨率": "1080P"})
    graph = result["expand"]
    sources = [entry for entry in graph.values()
               if entry["class_type"] == "H3DirectorActionSource"]
    check(len(sources) == 1, "Director did not create its uploaded action source")
    check(sources[0]["inputs"].get("resolution") == "1080P",
          "the reference-resolution cap never reaches the decoder, so it cannot "
          "bound peak RAM")
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
    graph = native.H3LongVideo().run(
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
    graph = native.H3LongVideo().run(
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
    graph = native.H3LongVideo().run(
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
    full = director._single_prompt_transfer_plan(
        {"shots": [{"prompt": "统一动作"}]}, 22, 5, 700)
    try:
        director._slice_plan_from_segment(full, 5, 22, has_context=False)
    except ValueError as error:
        check("前段视频" in str(error), "wrong missing-context error: %s" % error)
    else:
        raise AssertionError("Director resumed a seamless segment without the previous cut")

    graph = node.run(**common, **{
        "从指定段开始": True, "起始段": 5,
        "前段视频": torch.zeros(125, 1, 1, 3)})["expand"]
    slice_node = next(entry for entry in graph.values()
                      if entry["class_type"] == "H3DirectorPlanSlice")
    check(slice_node["inputs"]["start_segment"] == 5
          and "context_video" in slice_node["inputs"],
          "the runtime plan slicer did not receive the resume gate and context")
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
    bundle = SimpleNamespace(
        items=(
            SimpleNamespace(input_index=1, media_type="image", value=torch.zeros(1, 8, 8, 3)),
            SimpleNamespace(input_index=2, media_type="video", value=torch.zeros(4, 8, 8, 3)),
            SimpleNamespace(input_index=3, media_type="image", value=torch.zeros(1, 8, 8, 3)),
        ),
        links=(
            {"order": 1, "filename": "hero.png", "subject": "女主角正面照"},
            {"order": 2, "filename": "dance.mp4", "subject": "舞蹈动作"},
            {"order": 3, "filename": "street.png", "subject": "夜市街景"},
        ))
    rows = list(native.core.media_rows(bundle))
    check([(kind, ordinal) for kind, ordinal, _s, _f in rows]
          == [("image", 1), ("image", 2), ("video", 1)],
          "media_rows did not renumber per type the way H3Condition counts")
    manifest = legacy._format_media_manifest(bundle)
    check("@图片2：" in manifest and "夜市街景" in manifest,
          "the manifest hid the subject the LLM needs to match assets: %s" % manifest)
    check("@视频1" in manifest and "@视频3" not in manifest,
          "the manifest numbered the clip by input_index instead of by type")
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
    enhancement_tail = [
        "脸部精修开启", "脸部检测器", "脸部精修步数", "脸部精修重绘",
        "脸部裁剪倍率", "脸部身份图序号", "动作修复开启", "动作修复档位",
        "动作修复步数", "动作修复注入", "多视角分镜开启",
        "多视角角色图片序号", "多视角尺寸", "多视角步数", "多视角LoRA",
        "多视角LoRA强度", "二采复用一采条件",
        "音频精修开启", "音频精修步数", "音频去噪强度", "音频精修采样器",
        "音频精修调度器", "音频接缝平滑", "音频接缝时长",
        "从指定段开始",
        "二采显存策略", "二采自定义显存预留", "二采自定义预览间隔",
        "二采连续Sigma",
    ]
    check(list(required)[-len(enhancement_tail):] == enhancement_tail
          and list(required).index("vlm_service") < list(required).index("脸部精修开启"),
          "enhancement widgets were not appended after the existing saved values")
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


def test_resume_checkbox_gates_manual_cards_and_agent_always_starts_from_head():
    """The integer is inert until checked; Agent ignores even a stale check."""
    node = director.H3Director()
    common = dict(
        h3=object(), model=object(), sampler=object(),
        timeline_json=json.dumps({"shots": [
            {"prompt": "镜头一", "duration_seconds": 5},
            {"prompt": "镜头二", "duration_seconds": 5, "transition": "承接"},
            {"prompt": "镜头三", "duration_seconds": 5, "transition": "切镜"},
        ]}), script_fallback="长剧本", total_seconds=20, segment_seconds=5,
        llm_enabled=False, llm_service="未配置 LLM 服务",
        task_mode=legacy.TASK_FRESH, resolution="480P", aspect_ratio="16:9",
        width=864, height=480, steps=25, denoise=1.0, scheduler="simple",
        noise_seed=0, context_length="22", ref_image_size="匹配生成分辨率",
        save_segments=False, segment_prefix="video/test", save_raw_segments=False)

    unchecked = node.run(
        **common, source_mode=director.DIRECTOR_TIMELINE,
        **{"从指定段开始": False, "起始段": 2,
           "前段视频": torch.zeros(125, 1, 1, 3)})["expand"]
    check(not any(entry["class_type"] == "H3DirectorPlanSlice"
                  for entry in unchecked.values()),
          "unchecked manual start segment still cropped the plan")
    unchecked_long = next(entry for entry in unchecked.values()
                          if entry["class_type"] == "H3LongVideo")
    check("context_video" not in unchecked_long["inputs"],
          "unchecked manual resume leaked stale context")

    manual = node.run(
        **common, source_mode=director.DIRECTOR_TIMELINE,
        **{"从指定段开始": True, "起始段": 2,
           "前段视频": torch.zeros(125, 1, 1, 3)})["expand"]
    check(any(entry["class_type"] == "H3DirectorPlanSlice"
              for entry in manual.values()),
          "checked manual storyboard did not get a runtime plan slice")

    agent = node.run(
        **common, source_mode=director.DIRECTOR_SCRIPT,
        **{"从指定段开始": True, "起始段": 2,
           "前段视频": torch.zeros(125, 1, 1, 3)})["expand"]
    check(not any(entry["class_type"] == "H3DirectorPlanSlice"
                  for entry in agent.values()),
          "Agent mode accepted a stale resume selection")
    agent_long = next(entry for entry in agent.values()
                      if entry["class_type"] == "H3LongVideo")
    check("context_video" not in agent_long["inputs"],
          "Agent mode received a resume context")

    source = (PACKAGE_DIR / "web" / "h3_director_ui.js").read_text("utf-8")
    check('if (!transferring && manual) root.appendChild(renderResumePanel(node));'
          in source and 'hideWidget(by["从指定段开始"]);' in source,
          "the resume checkbox is not confined to transfer and manual cards")
    check('toggle.type = "checkbox"' in source
          and 'startInput.disabled = !enabled' in source,
          "the start selector is not gated by an actual checkbox")


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
           "一采断点模式": legacy.PASS1_CHECKPOINT_SAVE,
           "二采显存策略": detail.DETAIL_MEMORY_LOW,
           "二采放大方式": "neural_3d (神经3D Latent放大·推荐)",
           "二采Latent模型": "minimax_h3_latent_upscaler_3d_fp16.safetensors"})
    graph = result["expand"]
    detail_nodes = [entry for entry in graph.values()
                    if entry["class_type"] == "H3DetailSettings"]
    check(len(detail_nodes) == 1, "Director did not build its integrated detail settings")
    check(detail_nodes[0]["inputs"]["二采模型"] is detail_model,
          "Director did not route the detail base model")
    check(detail_nodes[0]["inputs"]["memory_profile"] == detail.DETAIL_MEMORY_LOW,
          "Director did not forward the detail VRAM profile")
    long_node = next(entry for entry in graph.values()
                     if entry["class_type"] == "H3LongVideo")
    check("二采设置" in long_node["inputs"],
          "integrated detail settings did not reach H3LongVideo")
    check(long_node["inputs"]["save_raw_segments"] is True,
          "Director discarded raw-segment saving while detail pass is enabled")
    check(long_node["inputs"]["一采断点模式"] == legacy.PASS1_CHECKPOINT_SAVE,
          "Director did not forward its pass-1 checkpoint mode")

    continuous = node.run(
        h3=object(), model=object(), sampler=object(),
        source_mode=director.DIRECTOR_TIMELINE,
        timeline_json='{"shots":[{"prompt":"镜头一","duration_seconds":5}]}',
        script_fallback="", total_seconds=5, segment_seconds=5,
        llm_enabled=False, llm_service="none",
        task_mode=legacy.TASK_FRESH, resolution="480P", aspect_ratio="16:9",
        width=864, height=480, steps=8, denoise=1.0,
        scheduler="simple", noise_seed=0, context_length="22",
        ref_image_size="匹配生成分辨率", save_segments=False,
        segment_prefix="video/test", save_raw_segments=False,
        **{"二采开启": True, "二采连续Sigma": True,
           "二采模式": "放大 + 二采（推荐）", "二采分辨率": "832P",
           "二采放大方式": "neural_3d (神经3D Latent放大·推荐)"})
    continuous_detail = next(
        entry for entry in continuous["expand"].values()
        if entry["class_type"] == "H3DetailSettings")
    check(continuous_detail["inputs"]["continuous_sigma"] is True
          and "二采模型" not in continuous_detail["inputs"],
          "Director continuous-Sigma profile still requires or routes a separate model")


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


def _optional_pack(name):
    """Import an optional sibling custom-node pack, or skip the caller."""
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as error:
        if error.name == name:
            raise SkipTest("%s is not installed in custom_nodes" % name) from error
        raise


class SkipTest(Exception):
    """Raised when an opt-in dependency is absent; the runner reports SKIP."""


def test_director_builds_installed_opt_in_enhancements():
    import nodes as comfy_nodes

    mainodes = _optional_pack("ComfyUI-MAINodes")
    face_refine = _optional_pack("ComfyUI-H3-FaceRefine")
    comfy_nodes.NODE_CLASS_MAPPINGS.update(mainodes.NODE_CLASS_MAPPINGS)
    comfy_nodes.NODE_CLASS_MAPPINGS.update(face_refine.NODE_CLASS_MAPPINGS)
    check(all(name in comfy_nodes.NODE_CLASS_MAPPINGS for name in (
        "H3ContactSheet", "H3JerkOracle", "H3FaceTrackCrop")),
        "installed enhancement packs did not register their public nodes")
    lora = COMFY_DIR / "models" / "loras" / \
        "minimax_h3_five_view_1024cont_s600.safetensors"
    check(lora.is_file() and lora.stat().st_size > 60_000_000,
          "turnaround LoRA is missing or truncated")

    result = director.H3Director().run(
        h3=SimpleNamespace(clip=object(), video_vae=object()),
        model=object(), sampler=object(),
        source_mode=director.DIRECTOR_TIMELINE,
        timeline_json=json.dumps({"shots": [
            {"prompt": "人物转身", "duration_seconds": 5},
        ]}), script_fallback="", total_seconds=5, segment_seconds=5,
        llm_enabled=False, llm_service="none", task_mode=legacy.TASK_FRESH,
        resolution="480P", aspect_ratio="16:9", width=864, height=480,
        steps=25, denoise=1.0, scheduler="simple", noise_seed=0,
        context_length="22", ref_image_size="匹配生成分辨率",
        save_segments=False, segment_prefix="video/test",
        save_raw_segments=False, media=object(),
        **{"脸部精修开启": True, "动作修复开启": True,
           "多视角分镜开启": True, "多视角角色图片序号": 1,
           "多视角尺寸": "512", "多视角步数": 28,
           "多视角LoRA": lora.name, "多视角LoRA强度": 0.75})
    graph = result["expand"]
    kinds = [entry["class_type"] for entry in graph.values()]
    for required in ("H3ContactSheet", "H3ContactSheetDecode",
                     "H3DirectorTurnaroundMedia",
                     "H3DirectorEnhancementSettings", "H3LongVideo"):
        check(required in kinds, "Director omitted %s" % required)
    long_node = next(entry for entry in graph.values()
                     if entry["class_type"] == "H3LongVideo")
    check("增强设置" in long_node["inputs"] and "media" in long_node["inputs"],
          "enhancement settings or generated sheet did not reach H3LongVideo")


def test_long_video_builds_motion_then_face_pipeline():
    import nodes as comfy_nodes

    mainodes = _optional_pack("ComfyUI-MAINodes")
    face_refine = _optional_pack("ComfyUI-H3-FaceRefine")
    comfy_nodes.NODE_CLASS_MAPPINGS.update(mainodes.NODE_CLASS_MAPPINGS)
    comfy_nodes.NODE_CLASS_MAPPINGS.update(face_refine.NODE_CLASS_MAPPINGS)
    plan = director._timeline_plan({"shots": [
        {"prompt": "人物快速转头", "duration_seconds": 5},
    ]}, 22)
    graph = native.H3LongVideo().run(
        h3=SimpleNamespace(video_vae=object(), audio_vae=object()),
        model=object(), sampler=object(), plan_json=json.dumps(plan),
        task_mode=legacy.TASK_FRESH, resolution="480P",
        aspect_ratio="16:9", width=864, height=480,
        steps=25, denoise=1.0, scheduler="simple", noise_seed=0,
        context_length="22", prompt_mode=legacy.MODE_DIRECT,
        media_prefix="", ref_image_size="匹配生成分辨率",
        save_segments=False, segment_prefix="video/test",
        save_raw_segments=False, **{"增强设置": {
            "model": object(),
            "motion": {"enabled": True, "preset": "balanced (default)",
                       "steps": 6, "inject": 0.70},
            "face": {"enabled": True, "detector": "bbox\\face_yolov8m.pt",
                     "steps": 4, "denoise": 0.45, "crop_factor": 2.5,
                     "identity_ordinal": 0},
        }})["expand"]
    kinds = [entry["class_type"] for entry in graph.values()]
    for required in (
            "H3JerkOracle", "H3TimeSmear", "H3InjectSchedule",
            "H3ExactRecover", "H3AudioRecover", "H3FramesToSeconds",
            "H3FaceTrackCrop", "H3InjectVideoLatent",
            "H3PerFrameDenoise", "H3FaceStitch"):
        check(required in kinds, "enhancement pipeline omitted %s" % required)
    stages = [entry["inputs"].get("stage") for entry in graph.values()
              if entry["class_type"] == "H3ProgressSignal"]
    check(stages.index("motion_refined") < stages.index("face_start"),
          "face refinement no longer runs after motion repair")


def test_face_detector_rejects_a_truncated_pytorch_archive():
    _optional_pack("ComfyUI-H3-FaceRefine")
    face_nodes = importlib.import_module("ComfyUI-H3-FaceRefine.nodes")
    with tempfile.TemporaryDirectory(prefix="h3_face_detector_") as directory:
        broken = Path(directory) / "face_yolov8m.pt"
        broken.write_bytes(b"PK\x03\x04" + b"truncated" * 32)
        try:
            face_nodes._validate_detector_archive(str(broken), broken.name)
        except ValueError as error:
            check("损坏或下载不完整" in str(error),
                  "truncated detector error is not actionable")
        else:
            raise AssertionError("truncated PyTorch detector passed preflight")


def test_long_video_stops_on_face_detector_preflight_before_graph_sampling():
    import nodes as comfy_nodes

    face_refine = _optional_pack("ComfyUI-H3-FaceRefine")
    comfy_nodes.NODE_CLASS_MAPPINGS.update(face_refine.NODE_CLASS_MAPPINGS)
    original = comfy_nodes.NODE_CLASS_MAPPINGS["H3FaceTrackCrop"]

    class BrokenDetector:
        @classmethod
        def validate_detector(cls, detector):
            raise ValueError("检测模型损坏")

    comfy_nodes.NODE_CLASS_MAPPINGS["H3FaceTrackCrop"] = BrokenDetector
    plan = director._timeline_plan({"shots": [
        {"prompt": "人物特写", "duration_seconds": 5},
    ]}, 22)
    try:
        try:
            native.H3LongVideo().run(
                h3=SimpleNamespace(video_vae=object(), audio_vae=object()),
                model=object(), sampler=object(), plan_json=json.dumps(plan),
                task_mode=legacy.TASK_FRESH, resolution="480P",
                aspect_ratio="16:9", width=864, height=480,
                steps=25, denoise=1.0, scheduler="simple", noise_seed=0,
                context_length="22", prompt_mode=legacy.MODE_DIRECT,
                media_prefix="", ref_image_size="匹配生成分辨率",
                save_segments=False, segment_prefix="video/test",
                save_raw_segments=False, **{"增强设置": {
                    "model": object(),
                    "face": {"enabled": True,
                             "detector": "bbox\\face_yolov8m.pt"},
                }})
        except ValueError as error:
            check("视频采样前停止" in str(error) and "检测模型损坏" in str(error),
                  "detector preflight failure lost its early-stop diagnosis")
        else:
            raise AssertionError("long-video graph accepted a broken face detector")
    finally:
        comfy_nodes.NODE_CLASS_MAPPINGS["H3FaceTrackCrop"] = original


def test_long_video_uses_variable_shot_windows():
    plan = director._timeline_plan({"shots": [
        {"prompt": "镜头一", "duration_seconds": 5},
        {"prompt": "镜头二", "duration_seconds": 6},
    ]}, 22)
    graph = native.H3LongVideo().run(
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
    graph = native.H3LongVideo().run(
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


def _second_pass_graph(**detail_overrides):
    """Build a one-shot long-video graph with the detail pass switched on."""
    pass1_mode = detail_overrides.pop(
        "pass1_checkpoint_mode", legacy.PASS1_CHECKPOINT_OFF)
    pass1_video = detail_overrides.pop("pass1_video", None)
    pass1_audio = detail_overrides.pop("pass1_audio", None)
    plan = director._timeline_plan(
        {"shots": [{"prompt": "镜头一", "duration_seconds": 5}]}, 22)
    settings = {
        "enabled": True,
        "mode": "放大 + 二采（推荐）",
        "resolution": "832P",
        "width": 1664, "height": 928,
        "steps": 4, "denoise": 0.2,
        "scheduler": "beta", "sampler_name": "res_multistep",
        "upscale_method": "neural_3d (神经3D Latent放大·推荐)",
        "chunk_frames": 4,
        "latent_upscale_model": "minimax_h3_latent_upscaler_3d_fp16.safetensors",
        "latent_precision": neural.PRECISIONS[0],
        "latent_chunk_steps": 0,
        "passes": 1,
        "seed_mode": "每轮沿用同一种子",
        "reuse_condition": True,
        "model": object(),
    }
    settings.update(detail_overrides)
    extra = {
        "二采设置": settings,
        "一采断点模式": pass1_mode,
    }
    if pass1_video is not None:
        extra["一采成片"] = pass1_video
    if pass1_audio is not None:
        extra["一采成片音频"] = pass1_audio
    return native.H3LongVideo().run(
        h3=SimpleNamespace(video_vae=object(), audio_vae=object()),
        model=object(), sampler=object(), plan_json=json.dumps(plan),
        task_mode=legacy.TASK_FRESH, resolution="480P",
        aspect_ratio="16:9", width=864, height=480,
        steps=8, denoise=1.0, scheduler="simple", noise_seed=0,
        context_length="22", prompt_mode=legacy.MODE_DIRECT,
        media_prefix="", llm_service="none",
        ref_image_size="匹配生成分辨率", save_segments=False,
        segment_prefix="video/test", save_raw_segments=False,
        **extra)["expand"]


def test_pass1_checkpoint_modes_save_or_skip_first_sampler():
    saved = _second_pass_graph(
        pass1_checkpoint_mode=legacy.PASS1_CHECKPOINT_SAVE)
    saved_kinds = [entry["class_type"] for entry in saved.values()]
    check(saved_kinds.count("H3Pass1CheckpointSave") == 1,
          "save mode did not persist the completed pass-1 latent")
    saved_samplers = [entry for entry in saved.values()
                      if entry["class_type"] == "H3SamplerAdvanced"]
    check({entry["inputs"].get("pass_label") for entry in saved_samplers}
          == {"sample1", "sample2"},
          "save mode changed the normal two-pass sampler chain")

    reused = _second_pass_graph(
        pass1_checkpoint_mode=legacy.PASS1_CHECKPOINT_REUSE)
    reused_kinds = [entry["class_type"] for entry in reused.values()]
    check(reused_kinds.count("H3Pass1CheckpointLoad") == 1,
          "reuse mode did not load the saved pass-1 latent")
    reused_samplers = [entry for entry in reused.values()
                       if entry["class_type"] == "H3SamplerAdvanced"]
    check([entry["inputs"].get("pass_label") for entry in reused_samplers]
          == ["sample2"],
          "reuse mode still expanded a first-pass sampler")

    original_isfile = native.os.path.isfile
    try:
        native.os.path.isfile = lambda _path: True
        resumed_existing = _second_pass_graph(
            pass1_checkpoint_mode=legacy.PASS1_CHECKPOINT_RESUME)
        native.os.path.isfile = lambda _path: False
        resumed_missing = _second_pass_graph(
            pass1_checkpoint_mode=legacy.PASS1_CHECKPOINT_RESUME)
    finally:
        native.os.path.isfile = original_isfile
    existing_kinds = [entry["class_type"]
                      for entry in resumed_existing.values()]
    missing_kinds = [entry["class_type"]
                     for entry in resumed_missing.values()]
    check("H3Pass1CheckpointLoad" in existing_kinds
          and not any(entry["inputs"].get("pass_label") == "sample1"
                      for entry in resumed_existing.values()
                      if entry["class_type"] == "H3SamplerAdvanced"),
          "resume mode did not skip an already completed first-pass segment")
    check("H3Pass1CheckpointSave" in missing_kinds
          and any(entry["inputs"].get("pass_label") == "sample1"
                  for entry in resumed_missing.values()
                  if entry["class_type"] == "H3SamplerAdvanced"),
          "resume mode did not generate and checkpoint a missing segment")


def test_single_pass1_video_can_enter_detail_without_sampling_again():
    graph = _second_pass_graph(
        pass1_checkpoint_mode=legacy.PASS1_VIDEO_REUSE,
        pass1_video=object(), pass1_audio=object())
    kinds = [entry["class_type"] for entry in graph.values()]
    check(kinds.count("H3Pass1VideoEncode") == 1,
          "finished pass-1 video was not re-encoded for detail")
    samplers = [entry for entry in graph.values()
                if entry["class_type"] == "H3SamplerAdvanced"]
    check([entry["inputs"].get("pass_label") for entry in samplers]
          == ["sample2"],
          "finished pass-1 video still triggered pass-1 sampling")
    sampled_signal = next(
        entry for entry in graph.values()
        if entry["class_type"] == "H3ProgressSignal"
        and entry["inputs"].get("stage") == "sampled")
    check(sampled_signal["inputs"]["save_preview"] is True,
          "direct pass-1 video reuse lost its compact progress-card preview")
    collector = next(entry for entry in graph.values()
                     if entry["class_type"] == "H3SegmentCollector")
    check(bool(collector["inputs"].get("run_id"))
          and "owner_id" in collector["inputs"]
          and collector["inputs"].get("total_segments") == 1,
          "final collector cannot publish merge progress to the Director")


def test_pass1_checkpoint_round_trip_preserves_av_streams_and_contract():
    import comfy.nested_tensor
    import folder_paths

    video = torch.randn(1, 24, 3, 4, 5)
    audio = torch.randn(1, 8, 6)
    samples = {"samples": comfy.nested_tensor.NestedTensor((video, audio))}
    original_output = folder_paths.get_output_directory
    with tempfile.TemporaryDirectory(
            prefix="h3_pass1_checkpoint_", dir=str(TEST_DIR)) as directory:
        folder_paths.get_output_directory = lambda: directory
        try:
            native.H3Pass1CheckpointSave().save(
                samples, "video/test", 3, 125)
            loaded = native.H3Pass1CheckpointLoad().load(
                "video/test", 3, 125)[0]
            streams = list(loaded["samples"].unbind())
            check(torch.equal(streams[0], video) and torch.equal(streams[1], audio),
                  "checkpoint round trip changed video/audio latent values")
            try:
                native.H3Pass1CheckpointLoad().load("video/test", 3, 126)
            except ValueError as error:
                check("帧数" in str(error), "frame mismatch returned an unclear error")
            else:
                raise AssertionError("checkpoint accepted a different storyboard duration")
        finally:
            folder_paths.get_output_directory = original_output


def test_second_pass_reuses_the_first_pass_conditioning():
    graph = _second_pass_graph()
    conditions = [entry for entry in graph.values()
                  if entry["class_type"] == "H3Condition"]
    check(len(conditions) == 1,
          "the detail pass still rebuilds conditioning at its own resolution")
    # H3 conditioning carries prompt tokens and minimax_refs only; the target
    # canvas comes from the latent.  Rebuilding it at the second-pass canvas
    # also re-fits every reference image to the larger area, which changes the
    # ref token layout the low-denoise pass is asked to converge to.
    check(conditions[0]["inputs"]["resolution"] == "480P",
          "the surviving conditioning is not the first pass's")

    rebuilt = _second_pass_graph(reuse_condition=False)
    resolutions = sorted(entry["inputs"]["resolution"]
                         for entry in rebuilt.values()
                         if entry["class_type"] == "H3Condition")
    check(resolutions == ["480P", "832P"],
          "opting out of reuse no longer rebuilds the second-pass conditioning")


def test_pixel_detail_path_receives_vram_headroom_and_preview_cadence():
    graph = _second_pass_graph(
        upscale_method="pixel (像素放大·自用版工作流方式)",
        memory_profile=detail.DETAIL_MEMORY_LOW,
        reserve_vram_gb=2.0,
        preview_interval=0)
    upscale = next(entry for entry in graph.values()
                   if entry["class_type"] == "H3LatentUpscale")
    check(upscale["inputs"]["reserve_vram_gb"] == 4.5
          and "vae" in upscale["inputs"],
          "pixel VAE projection did not receive its VRAM safety margin")
    second_sampler = next(
        entry for entry in graph.values()
        if entry["class_type"] == "H3SamplerAdvanced"
        and entry["inputs"].get("pass_label") == "sample2")
    check(second_sampler["inputs"]["reserve_vram_gb"] == 4.5
          and second_sampler["inputs"]["preview_interval"] == 0,
          "second-pass sampler ignored the selected VRAM profile")
    barriers = [entry for entry in graph.values()
                if entry["class_type"] == "H3RefineMemoryBarrier"]
    check(len(barriers) == 1
          and "conditioning" in barriers[0]["inputs"]
          and "latent" in barriers[0]["inputs"],
          "second pass does not clear VAE/upscaler residency after both inputs are ready")
    nodes_source = (PACKAGE_DIR / "nodes.py").read_text("utf-8")
    release_start = nodes_source.index("def _release_comfy_models(")
    release_end = nodes_source.index("class H3ConditionMemoryBarrier", release_start)
    release_source = nodes_source[release_start:release_end]
    check("comfy.model_prefetch.cleanup_prefetch_queues()" in release_source
          and "model_management.reset_cast_buffers()" in release_source,
          "second-pass barrier leaves AIMDO prefetch/cast allocations pinned")
    refine_start = next(entry for entry in graph.values()
                        if entry["class_type"] == "H3ProgressSignal"
                        and entry["inputs"].get("stage") == "refine_start")
    check(refine_start["inputs"]["save_preview"] is False,
          "second-pass preparation does not retain a visible progress frame")


def test_first_pass_memory_profiles_control_cleanup_and_clear_preview_decode():
    plan = director._timeline_plan(
        {"shots": [{"prompt": "镜头一", "duration_seconds": 5}]}, 22)

    def build(profile, task_mode=legacy.TASK_FRESH):
        return native.H3LongVideo().run(
            h3=SimpleNamespace(video_vae=object(), audio_vae=object()),
            model=object(), sampler=object(), plan_json=json.dumps(plan),
            task_mode=task_mode, resolution="640P",
            aspect_ratio="16:9", width=1152, height=640,
            steps=8, denoise=1.0, scheduler="simple", noise_seed=0,
            context_length="22", prompt_mode=legacy.MODE_DIRECT,
            media_prefix="", ref_image_size="匹配生成分辨率",
            save_segments=False, segment_prefix="video/test",
            save_raw_segments=False,
            ref_video=(torch.zeros(130, 1, 1, 3)
                       if task_mode == legacy.TASK_TRANSFER else None),
            **{"一采显存策略": profile})["expand"]

    balanced = build(legacy.FIRST_MEMORY_AUTO)
    kinds = [entry["class_type"] for entry in balanced.values()]
    check("H3PrepareSignal" in kinds,
          "the real execution graph has no pre-conditioning progress signal")
    check("H3PreConditionMemoryBarrier" in kinds,
          "16GB auto profile leaves the DiT resident during VAE conditioning")
    check("H3ConditionMemoryBarrier" in kinds,
          "16GB auto profile omitted the post-conditioning cleanup barrier")
    check("H3OutputMemoryRelease" in kinds,
          "16GB auto profile omitted the final model release")
    first_sampler = next(
        entry for entry in balanced.values()
        if entry["class_type"] == "H3SamplerAdvanced"
        and entry["inputs"].get("pass_label") == "sample1")
    check(first_sampler["inputs"]["preview_interval"] == 1
          and first_sampler["inputs"]["reserve_vram_gb"] == 1.25
          and first_sampler["inputs"]["preview_mode"] == "latent_rgb",
          "16GB auto profile does not provide zero-VRAM step previews")

    compatible = build(legacy.FIRST_MEMORY_STANDARD)
    compatible_kinds = [entry["class_type"] for entry in compatible.values()]
    check("H3PreConditionMemoryBarrier" not in compatible_kinds
          and "H3ConditionMemoryBarrier" not in compatible_kinds
          and "H3OutputMemoryRelease" not in compatible_kinds,
          "compatibility profile unexpectedly changed model residency")
    compatible_sampler = next(
        entry for entry in compatible.values()
        if entry["class_type"] == "H3SamplerAdvanced"
        and entry["inputs"].get("pass_label") == "sample1")
    check(compatible_sampler["inputs"]["preview_interval"] == 1
          and compatible_sampler["inputs"]["preview_mode"] == "vae",
          "compatibility profile no longer preserves per-step clear previews")

    low = build(legacy.FIRST_MEMORY_LOW)
    low_sampler = next(
        entry for entry in low.values()
        if entry["class_type"] == "H3SamplerAdvanced"
        and entry["inputs"].get("pass_label") == "sample1")
    check(low_sampler["inputs"]["preview_interval"] == 1
          and low_sampler["inputs"]["preview_mode"] == "latent_rgb"
          and low_sampler["inputs"]["reserve_vram_gb"] == 1.5,
          "low-memory profile does not use per-step zero-VRAM previews")

    for profile in (legacy.FIRST_MEMORY_AUTO,
                    legacy.FIRST_MEMORY_STANDARD,
                    legacy.FIRST_MEMORY_LOW):
        transfer = build(profile, legacy.TASK_TRANSFER)
        transfer_sampler = next(
            entry for entry in transfer.values()
            if entry["class_type"] == "H3SamplerAdvanced"
            and entry["inputs"].get("pass_label") == "sample1")
        expected_interval = 1 if profile == legacy.FIRST_MEMORY_STANDARD else 0
        expected_mode = "vae" if profile == legacy.FIRST_MEMORY_STANDARD else "latent_rgb"
        check(transfer_sampler["inputs"]["preview_interval"] == expected_interval
              and transfer_sampler["inputs"]["preview_mode"] == expected_mode,
              "action-transfer first-pass strategy is not selectable")


def test_precondition_memory_barrier_evicts_dit_before_vae():
    calls = []
    original = native._release_comfy_models
    native._release_comfy_models = lambda stage, keep_model=None, **kwargs: calls.append(
        (stage, keep_model, kwargs))
    bundle = object()
    model = object()
    try:
        result, = native.H3PreConditionMemoryBarrier().release(
            bundle, stage="model -> vae", loaded_model=model)
    finally:
        native._release_comfy_models = original
    check(result is bundle and calls == [(
              "model -> vae", model,
              {"preserve_dynamic_host_cache": True})],
          "pre-condition barrier did not evict GPU pages while preserving the H3 RAM cache")


def test_precondition_release_keeps_aimdo_ram_but_drops_gpu_pages():
    import comfy.model_management as model_management
    import comfy.model_prefetch as model_prefetch

    class DynamicPatcher:
        offload_device = "cpu"

        def __init__(self):
            self.loaded = 3 * 1024 ** 3
            self.partial_calls = []

        def is_dynamic(self):
            return True

        def loaded_size(self):
            return self.loaded

        def partially_unload(self, device, amount):
            self.partial_calls.append((device, amount))
            freed, self.loaded = self.loaded, 0
            return freed

    patcher = DynamicPatcher()
    loaded = SimpleNamespace(model=patcher, currently_used=True)
    calls = []
    names = (
        "current_loaded_models", "reset_cast_buffers", "get_torch_device",
        "get_free_memory", "get_all_torch_devices", "free_memory",
        "soft_empty_cache", "unload_all_models")
    originals = {name: getattr(model_management, name) for name in names}
    original_cleanup = model_prefetch.cleanup_prefetch_queues
    try:
        model_management.current_loaded_models = [loaded]
        model_management.reset_cast_buffers = lambda: calls.append("cast")
        model_management.get_torch_device = lambda: "cuda:0"
        model_management.get_free_memory = lambda _device: 1024 ** 3
        model_management.get_all_torch_devices = lambda: ["cuda:0"]
        model_management.free_memory = lambda amount, device, keep_loaded=[]: calls.append(
            ("free", amount, device, list(keep_loaded)))
        model_management.soft_empty_cache = lambda: calls.append("cache")
        model_management.unload_all_models = lambda: calls.append("full-unload")
        model_prefetch.cleanup_prefetch_queues = lambda: calls.append("prefetch")
        native._release_comfy_models(
            "model -> vae", keep_model=patcher,
            preserve_dynamic_host_cache=True)
    finally:
        for name, value in originals.items():
            setattr(model_management, name, value)
        model_prefetch.cleanup_prefetch_queues = original_cleanup
    check(patcher.loaded == 0 and patcher.partial_calls == [("cpu", 1e32)],
          "AIMDO GPU pages were not evicted by the pre-condition release")
    check("full-unload" not in calls and loaded.currently_used is False,
          "pre-condition release destroyed the dynamic model RAM cache")
    free_call = next(item for item in calls if isinstance(item, tuple))
    check(free_call[3] == [loaded],
          "regular ComfyUI cleanup was allowed to detach the preserved dynamic patcher")


def test_upscale_only_pixel_path_skips_every_vae_round_trip():
    graph = _second_pass_graph(
        mode="仅放大（不二采·最快）",
        upscale_method="nvidia_rtx_vsr (NVIDIA RTX 视频超分·实验)",
        model=None)
    kinds = [entry["class_type"] for entry in graph.values()]
    check("H3PixelUpscale" in kinds,
          "仅放大 + pixel/VSR did not take the pure-pixel path")
    check("H3LatentUpscale" not in kinds,
          "仅放大 + pixel/VSR still round-tripped through the latent upscaler")
    # One decode for the first pass is unavoidable (preview, drift, anchors).
    # The old path added decode -> VSR -> encode -> decode on top of it, which
    # re-softened exactly the edges VSR had just sharpened.
    check(kinds.count("VAEDecode") == 1 and "VAEEncode" not in kinds,
          "仅放大 + pixel/VSR still pays an extra VAE round trip")

    latent_graph = _second_pass_graph(mode="仅放大（不二采·最快）", model=None)
    latent_kinds = [entry["class_type"] for entry in latent_graph.values()]
    check("H3LatentUpscale" in latent_kinds
          and "H3PixelUpscale" not in latent_kinds,
          "仅放大 + neural_3d should still upscale in latent space")


def test_temporal_chunk_blending_is_seamless_and_optional():
    calls = []

    def stub(tensor, scale, target_size):
        # A per-frame spatial resize: whatever window it is given, a frame's
        # output is identical.  Overlapping windows must therefore agree, so a
        # partition-of-unity blend has to reproduce the full-context result.
        calls.append(int(tensor.shape[2]))
        return torch.nn.functional.interpolate(
            tensor, size=target_size, mode="trilinear", align_corners=False)

    source = torch.randn(1, 24, 21, 3, 3)
    with torch.inference_mode():
        full = neural._forward_bounded(stub, source, 2.0, 6, 6, chunk_steps=0)
        check(len(calls) == 1,
              "chunk_steps=0 did not run a single full-context pass")
        calls.clear()
        chunked = neural._forward_bounded(
            stub, source, 2.0, 6, 6, chunk_steps=5, overlap=2)
    check(len(calls) > 1, "a 21-step clip at chunk_steps=5 was not chunked")
    check(tuple(chunked.shape) == tuple(full.shape),
          "temporal chunking changed the output shape")
    check(torch.allclose(chunked, full, atol=1e-4),
          "chunk seams no longer reconstruct the full-context output")


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
    check(len(directors) == 1 and len(nodes) == 7,
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


def test_material_reorder_preserves_ordinal_prompt_slots():
    director_ui = (PACKAGE_DIR / "web" / "h3_director_ui.js").read_text(encoding="utf-8")
    agent_ui = (PACKAGE_DIR / "web" / "minimax_h3_myang_agent_ui.js").read_text(encoding="utf-8")
    check("text/x-myang-asset-id" in director_ui,
          "Director material cards no longer expose a drag identity")
    check("提示词中的序号文字保持不变" in director_ui,
          "Director reorder UI no longer documents ordinal slot semantics")
    check("movingAsset.kind !== asset.kind" in director_ui,
          "Director reorder must stay within one media type")
    check("remapPromptMentions" not in director_ui,
          "material reorder must not rewrite authored prompt mentions")
    check("Native links tell us which sources are still connected" in agent_ui,
          "Agent sync no longer documents preserving user material order")


if __name__ == "__main__":
    tests = [
        test_director_registration_and_variable_timeline,
        test_director_timeline_isolated_by_task_mode,
        test_director_transition_controls_motion_context_boundaries,
        test_director_panel_tracks_node_selection_and_resize,
        test_director_asset_preview_preserves_intrinsic_aspect_and_opens_images,
        test_director_widget_changes_preserve_focus_and_caret,
        test_director_storyboard_card_import_export_ui,
        test_director_card_folding_and_layer_persistence_ui,
        test_action_transfer_plan_uses_one_prompt_and_covers_reference,
        test_action_transfer_uses_reference_duration_not_stale_template_total,
        test_action_transfer_sampler_windows_never_exceed_the_time_ceiling,
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
        test_resume_checkbox_gates_manual_cards_and_agent_always_starts_from_head,
        test_director_accepts_turbo_on_its_single_model_input,
        test_director_keeps_user_selected_step_inside_turbo_allowed_profile,
        test_director_keeps_queued_legacy_recommended_steps_compatible,
        test_director_plan_value_embeds_progress_owner,
        test_split_segment_prompts_are_pushed_to_the_panel,
        test_director_broadcast_never_breaks_a_run,
        test_director_builds_integrated_optional_detail_pass,
        test_director_disables_raw_segment_copy_when_detail_is_off,
        test_pass1_checkpoint_modes_save_or_skip_first_sampler,
        test_single_pass1_video_can_enter_detail_without_sampling_again,
        test_pass1_checkpoint_round_trip_preserves_av_streams_and_contract,
        test_pixel_detail_path_receives_vram_headroom_and_preview_cadence,
        test_first_pass_memory_profiles_control_cleanup_and_clear_preview_decode,
        test_precondition_memory_barrier_evicts_dit_before_vae,
        test_precondition_release_keeps_aimdo_ram_but_drops_gpu_pages,
        test_director_builds_installed_opt_in_enhancements,
        test_long_video_builds_motion_then_face_pipeline,
        test_face_detector_rejects_a_truncated_pytorch_archive,
        test_long_video_stops_on_face_detector_preflight_before_graph_sampling,
        test_long_video_uses_variable_shot_windows,
        test_long_video_routes_each_shot_action_material_lazily,
        test_neural_checkpoint_contract_and_temporal_chunking,
        test_director_example_workflow_is_compact_and_wired,
        test_material_reorder_preserves_ordinal_prompt_slots,
    ]
    for test in tests:
        try:
            test()
        except SkipTest as reason:
            print("SKIP", test.__name__, "-", reason)
            continue
        print("PASS", test.__name__)
