import importlib
import json
import sys
import tempfile
from pathlib import Path

import torch


TEST_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = TEST_DIR.parent
CUSTOM_NODES_DIR = PACKAGE_DIR.parent
COMFY_DIR = CUSTOM_NODES_DIR.parent
for path in (str(COMFY_DIR), str(CUSTOM_NODES_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

package = importlib.import_module("ComfyUI-MiniMaxH3-Myang")
anchors = importlib.import_module("ComfyUI-MiniMaxH3-Myang.anchors")
director = importlib.import_module("ComfyUI-MiniMaxH3-Myang.director")
library = importlib.import_module("ComfyUI-MiniMaxH3-Myang.roughcut_library")
nodes = importlib.import_module("ComfyUI-MiniMaxH3-Myang.nodes")
roughcut = importlib.import_module("ComfyUI-MiniMaxH3-Myang.roughcut")


def check(value, message):
    if not value:
        raise AssertionError(message)


def _project_with_clip():
    project = roughcut.new_project()
    project["selection"].update({"in_frame": 40, "out_frame": 60})
    project["tracks"][0]["clips"] = [{
        "id": "base", "kind": "video", "name": "base.mp4",
        "timeline_start": 0, "timeline_end": 100,
        "source_in": 0, "source_out": 100, "source_fps": 24,
        "source": {"type": "library", "library_id": "lib_test",
                   "relative_path": "base.mp4"},
    }]
    return project


def test_roughcut_writeback_requires_current_project_authorization():
    project = roughcut.new_project()
    check(project["settings"]["writeback_enabled"] is False,
          "new rough-cut projects must start in edit-only mode")
    legacy = roughcut.normalize_project({
        **project,
        "settings": {
            key: value for key, value in project["settings"].items()
            if key != "writeback_enabled"
        },
    })
    check(legacy["settings"]["writeback_enabled"] is False,
          "legacy projects without authorization did not fail closed")
    project["settings"]["writeback_enabled"] = "true"
    normalized = roughcut.normalize_project(project)
    check(normalized["settings"]["writeback_enabled"] is False,
          "truthy string unexpectedly authorized rough-cut write-back")
    project["settings"]["writeback_enabled"] = True
    normalized = roughcut.normalize_project(project)
    check(normalized["settings"]["writeback_enabled"] is True,
          "an explicit project authorization was discarded")


def test_roughcut_overwrite_is_non_ripple_and_frame_accurate():
    project = _project_with_clip()
    updated = roughcut.overwrite_selection(project, {
        "type": "output", "filename": "bridge.mp4", "subfolder": "video",
    }, source_fps=24, source_frames=20)
    clips = updated["tracks"][0]["clips"]
    check([(item["timeline_start"], item["timeline_end"]) for item in clips]
          == [(0, 40), (40, 60), (60, 100)],
          "overwrite did not preserve the material outside I/O")
    generated = clips[1]
    check(generated["kind"] == "generated" and generated["source_out"] == 20,
          "generated source timing was not preserved")
    check(clips[2]["source_in"] == 60,
          "right-hand split did not advance its source in-point")


def test_roughcut_boundary_window_uses_material_outside_io():
    project = _project_with_clip()
    first = roughcut.boundary_window(project, "first", 22)
    last = roughcut.boundary_window(project, "last", 22)
    check(first is not None and first[1:] == (18, 22),
          "head MotionContext did not read the 22 frames before I")
    check(last is not None and last[1:] == (60, 22),
          "tail MotionContext did not read the 22 frames after O")
    project["selection"]["in_frame"] = 10
    check(roughcut.boundary_window(project, "first", 22) is None,
          "an undersized head context was silently accepted")


def test_roughcut_audio_window_uses_a1_at_the_same_io_boundary():
    project = _project_with_clip()
    project["tracks"][1]["clips"] = [{
        "id": "sound", "kind": "audio", "name": "sound.wav",
        "timeline_start": 10, "timeline_end": 90,
        "source_in": 240, "source_out": 1040, "source_fps": 240,
        "source": {"type": "library", "library_id": "lib_test",
                   "relative_path": "sound.wav"},
    }]
    head = roughcut.boundary_window(project, "first", 20, track_kind="audio")
    tail = roughcut.boundary_window(project, "last", 20, track_kind="audio")
    check(head is not None and head[1:] == (340, 200),
          "A1 head sound did not end exactly at I")
    check(tail is not None and tail[1:] == (740, 200),
          "A1 tail sound did not begin exactly at O")


def test_roughcut_rejects_media_on_the_wrong_track_kind():
    project = _project_with_clip()
    project["tracks"][1]["clips"] = [dict(
        project["tracks"][0]["clips"][0], id="wrong-track")]
    try:
        roughcut.normalize_project(project)
    except ValueError as error:
        check("音频轨" in str(error), "wrong-track error is not actionable")
    else:
        raise AssertionError("video material was accepted on A1")


def test_roughcut_library_is_allowlisted_and_contained():
    with tempfile.TemporaryDirectory(dir=TEST_DIR) as directory:
        root = Path(directory)
        (root / "roughcut_fixture.mp4").write_bytes(b"fixture")
        (root / "roughcut_fixture.txt").write_text("not media", encoding="utf-8")
        record = {"id": "lib_test", "name": "项目素材",
                  "path": str(root), "available": True}
        original = library.load_libraries
        library.load_libraries = lambda _path=None: [record]
        try:
            assets = library.scan_library(record["id"])
            names = [item["name"] for item in assets]
            check("roughcut_fixture.mp4" in names and "roughcut_fixture.txt" not in names,
                  "library scan included unsupported files")
            check(library.resolve_asset(record["id"], "roughcut_fixture.mp4").is_file(),
                  "registered media could not be resolved")
            try:
                library.resolve_asset(record["id"], "../README.md")
            except ValueError:
                pass
            else:
                raise AssertionError("library resolver accepted a traversal path")
        finally:
            library.load_libraries = original


def test_folder_library_imports_a_director_reference_without_moving_source():
    with tempfile.TemporaryDirectory(dir=TEST_DIR) as directory:
        input_root = Path(directory) / "input"
        input_root.mkdir()
        source = Path(directory) / "library" / "portrait.png"
        source.parent.mkdir()
        source.write_bytes(b"fake-image")
        record = {"id": "lib_test", "name": "项目素材",
                  "path": str(source.parent), "available": True}
        original_load = library.load_libraries
        original_input = library.folder_paths.get_input_directory
        library.load_libraries = lambda _path=None: [record]
        library.folder_paths.get_input_directory = lambda: str(input_root)
        try:
            file_info = library.import_asset("lib_test", "portrait.png")
            copied = input_root / file_info["subfolder"] / file_info["name"]
            check(copied.read_bytes() == b"fake-image",
                  "Director library import did not copy the selected file")
            check(source.is_file(), "Director library import moved or deleted the source")
            check(file_info["type"] == "input",
                  "Director library import did not return a ComfyUI input reference")
        finally:
            library.load_libraries = original_load
            library.folder_paths.get_input_directory = original_input


def test_director_catalogue_input_source_is_safe_and_timeline_compatible():
    with tempfile.TemporaryDirectory(dir=TEST_DIR) as directory:
        input_root = Path(directory) / "input"
        target = input_root / "Myang_node" / "director" / "hero.png"
        target.parent.mkdir(parents=True)
        target.write_bytes(b"fake-image")
        original_input = library.folder_paths.get_input_directory
        library.folder_paths.get_input_directory = lambda: str(input_root)
        try:
            check(library.resolve_input(
                "hero.png", "Myang_node/director") == target.resolve(),
                "Director catalogue media cannot be resolved from ComfyUI/input")
            project = _project_with_clip()
            project["tracks"][0]["clips"][0]["source"] = {
                "type": "input", "filename": "hero.png",
                "subfolder": "Myang_node/director",
            }
            normalized = roughcut.normalize_project(project)
            check(normalized["tracks"][0]["clips"][0]["source"]["type"] == "input",
                  "timeline rejected a Director catalogue source")
            try:
                library.resolve_input("hero.png", "../../outside")
            except ValueError:
                pass
            else:
                raise AssertionError("Director input resolver accepted path traversal")
        finally:
            library.folder_paths.get_input_directory = original_input


def test_director_exposes_working_roughcut_contract_append_only():
    inputs = director.H3Director.INPUT_TYPES()
    optional = inputs["optional"]
    names = list(optional)
    roughcut_index = names.index("粗剪时间轴开启")
    check(names[roughcut_index:roughcut_index + 2] == ["粗剪时间轴开启", "粗剪工程"],
          "rough-cut widgets changed order or stopped being append-only")
    project = roughcut.normalize_project(optional["粗剪工程"][1]["default"])
    check(project["format"] == roughcut.PROJECT_FORMAT,
          "Director rough-cut default is not a valid project")
    check("H3RoughCutBoundaryFrames" in package.NODE_CLASS_MAPPINGS
          and "H3RoughCutSave" in package.NODE_CLASS_MAPPINGS,
          "rough-cut execution nodes are not registered")
    long_optional = package.NODE_CLASS_MAPPINGS["H3LongVideo"].INPUT_TYPES()["optional"]
    check({"timeline_first_keyframe", "timeline_last_keyframe",
           "timeline_head_context", "timeline_tail_context",
           "roughcut_exact_duration"}.issubset(long_optional),
          "H3LongVideo cannot receive rough-cut boundary constraints")
    check("H3DirectorPlanLimit" in package.NODE_CLASS_MAPPINGS,
          "rough-cut duration limiter is not registered")
    action_optional = package.NODE_CLASS_MAPPINGS[
        "H3DirectorActionSource"].INPUT_TYPES()["optional"]
    check({"target_frames", "minimum_source_frame"}.issubset(action_optional),
          "action transfer cannot plan against the rough-cut I/O selection")


def test_director_rejects_truthy_legacy_strings_for_roughcut_gate():
    check(not director._input_enabled(False)
          and not director._input_enabled("False")
          and not director._input_enabled("关闭")
          and not director._input_enabled("自动平衡（16GB推荐）"),
          "a stale non-empty widget string can still enable rough-cut parsing")
    check(director._input_enabled(True)
          and director._input_enabled("true")
          and director._input_enabled("开启"),
          "valid rough-cut boolean values are no longer accepted")


def test_roughcut_io_state_and_duration_modes_are_explicit():
    project = roughcut.new_project()
    contract = roughcut.selection_contract(project)
    check(not contract["ready"] and not contract["in_set"] and not contract["out_set"],
          "a new timeline still opens with a fake I/O range")
    check(project["settings"]["auto_align"] and project["settings"]["snap_enabled"],
          "automatic alignment and snapping are not enabled by default")

    project["selection"].update({
        "in_frame": 48, "out_frame": 168, "in_set": True,
        "out_set": False, "anchor_side": "in", "duration_mode": "director",
    })
    check(roughcut.selection_contract(project)["ready"],
          "Director-duration mode does not accept its one authored point")
    project["selection"]["duration_mode"] = "timeline"
    check(not roughcut.selection_contract(project)["ready"],
          "timeline-duration mode accepted an incomplete I/O pair")
    project["selection"]["out_set"] = True
    contract = roughcut.selection_contract(project)
    check(contract["ready"] and contract["seconds"] == 5.0,
          "timeline-duration mode did not expose the selected range")


def test_segment_merge_progress_and_memory_barrier_are_reload_safe():
    source = Path(nodes.__file__).read_text("utf-8")
    check("from .progress import broadcast_progress" not in source,
          "segment collector still depends on a hot-reloaded progress module")
    check("if refining and index < count:" in source
          and '"H3SegmentMemoryBarrier"' in source,
          "second-pass residency is still carried into the next segment")

    events = []
    releases = []
    original_broadcast = nodes._broadcast_director_progress
    original_release = nodes._release_comfy_models
    nodes._broadcast_director_progress = lambda payload: events.append(payload) or True
    nodes._release_comfy_models = lambda stage, keep_model=None, **options: releases.append(
        (stage, keep_model, options))
    try:
        images = torch.zeros((1, 2, 2, 3))
        audio = {"waveform": torch.zeros((1, 2, 16)), "sample_rate": 16000}
        merged_images, _merged_audio = nodes.H3SegmentCollector().collect(
            1, run_id="run_test", owner_id="node_8", total_segments=1,
            images_1=images, audios_1=audio)
        check(merged_images is not None
              and [item["stage"] for item in events] == ["assembling", "assembled"],
              "final merge progress is missing or out of order")
        keep_model = object()
        h3 = object()
        import comfy.nested_tensor
        detail_latent = {"samples": comfy.nested_tensor.NestedTensor((
            torch.zeros((1, 24, 12, 2, 2)),
            torch.zeros((1, 64, 20)),
        ))}
        pass1_latent = {"samples": comfy.nested_tensor.NestedTensor((
            torch.zeros((1, 24, 12, 1, 1)),
            torch.zeros((1, 64, 20)),
        ))}
        (result_images, result_audio, result_h3,
         detail_context, pass1_context) = nodes.H3SegmentMemoryBarrier().release(
            images, audio, stage="segment 1 -> segment 2", h3=h3,
            keep_model=keep_model, detail_latent=detail_latent,
            pass1_latent=pass1_latent, context_length="22")
        check(result_images is images and result_audio is audio and result_h3 is h3,
              "the segment memory barrier copied or replaced media tensors")
        check(detail_context["samples"].device.type == "cpu"
              and detail_context["samples"].shape[2] == 7
              and detail_latent["samples"] is not detail_context["samples"],
              "the segment barrier truncated a latent the finished segment is "
              "still reachable through")
        check(pass1_context["samples"].device.type == "cpu"
              and pass1_context["samples"].shape[2] == 7
              and pass1_latent["samples"] is not pass1_context["samples"],
              "the segment barrier truncated the first-pass latent instead of "
              "just handing the next segment a compact tail")
        # The finished segment's pixels are still reachable by re-decoding these
        # latents, and the RAM pressure cache does evict and re-derive them, so
        # a shortened latent silently costs whole segments in the merged film.
        check(detail_latent["samples"].unbind()[0].shape[2] == 12
              and pass1_latent["samples"].unbind()[0].shape[2] == 12,
              "the segment barrier shortened a latent it does not own")
        check(releases == [("segment 1 -> segment 2", keep_model,
                            {"preserve_dynamic_host_cache": True})],
              "the segment barrier did not clear dynamic GPU pages while retaining RAM caches")
    finally:
        nodes._broadcast_director_progress = original_broadcast
        nodes._release_comfy_models = original_release


def test_anchor_trim_removes_hidden_tail_and_matches_audio_duration():
    images = torch.rand((20, 8, 8, 3))
    audio = {"waveform": torch.rand((1, 2, 32000)), "sample_rate": 32000}
    result_images, result_audio = anchors.H3AnchorTrim().trim(
        images, 3, audio=audio, fps=20, tail_trim_frames=2)
    check(int(result_images.shape[0]) == 15,
          "hidden tail context was not removed from the visible result")
    check(int(result_audio["waveform"].shape[-1]) == 24000,
          "audio was not trimmed to the visible bridge duration")


def test_roughcut_plan_is_fitted_before_generation_and_keeps_h3_grid():
    plan = {
        "fps": 24.0,
        "overlap_frames": 22,
        "segments": [
            {"index": 1, "transition": "开场", "frames": 192, "prompt": "A"},
            {"index": 2, "transition": "承接", "frames": 192, "prompt": "B"},
            {"index": 3, "transition": "切镜", "frames": 192, "prompt": "C"},
        ],
    }
    fitted = director._limit_plan_to_visible_frames(plan, 300, 22)
    check(fitted["segment_count"] == 2,
          "rough-cut limiter kept shots that lie completely after O")
    visible = (fitted["segments"][0]["frames"]
               + fitted["segments"][1]["frames"] - 22)
    check(300 <= visible < 317,
          "rough-cut plan did not cover I/O with at most one H3-grid surplus")
    check(all(int(item["frames"]) % 17 == 5 for item in fitted["segments"]),
          "rough-cut duration fitting produced an illegal H3 frame count")
    check([item["prompt"] for item in fitted["segments"]] == ["A", "B"],
          "rough-cut fitting changed or reordered authored prompts")

    legacy = {"segments": [
        {"frames": 39, "transition": "开场"},
        {"frames": 39, "transition": ""},
    ]}
    legacy_fitted = director._limit_plan_to_visible_frames(legacy, 50, 22)
    legacy_visible = (legacy_fitted["segments"][0]["frames"]
                      + legacy_fitted["segments"][1]["frames"] - 22)
    check(legacy_visible >= 50,
          "empty legacy transition no longer matches seamless LongVideo semantics")
    tiny = director._limit_plan_to_visible_frames(
        {"segments": [{"frames": 5, "transition": "开场"}]}, 1, 22)
    check(tiny["segments"][0]["frames"] > 22,
          "tiny I/O created a segment rejected by the context window")


def test_double_sided_motion_context_preserves_visible_io_duration():
    source = Path(nodes.__file__).read_text("utf-8")
    check("roughcut_exact_duration" in source,
          "long-video node cannot distinguish hidden I/O context")
    check("frames + head_motion_extension - 1" in source,
          "O-point keyframe is still attached to hidden alignment padding")
    check('"anchor_start_frame": hidden_layout["tail_start_frame"]' in source,
          "tail MotionContext does not start after the visible I/O range")
    check('tail_trim_frames = hidden_layout["tail_trim_frames"]' in source,
          "hidden H3-grid surplus is not cropped with I-side context")
    for head, tail in ((False, False), (True, False), (False, True), (True, True)):
        layout = nodes._roughcut_hidden_layout(
            192, 22, hidden_head=head, hidden_tail=tail)
        check(layout["condition_frames"]
              - layout["head_trim_frames"]
              - layout["tail_trim_frames"] == 192,
              "hidden MotionContext changed the visible I/O duration")
    check(anchors.tail_seed_step(214) is None,
          "off-phase exact O point was treated as safe for latent hard-seeding")
    check(anchors.tail_seed_step(221) == 65,
          "phase-aligned O point cannot use the zero-noise tail optimization")


def test_roughcut_frontend_is_integrated_without_refreshing_on_playhead_edits():
    director_source = (PACKAGE_DIR / "web" / "h3_director_ui.js").read_text("utf-8")
    editor_source = (PACKAGE_DIR / "web" / "h3_roughcut_ui.js").read_text("utf-8")
    backend_source = (PACKAGE_DIR / "roughcut_nodes.py").read_text("utf-8")
    director_backend = (PACKAGE_DIR / "director.py").read_text("utf-8")
    check("renderRoughCutCard(node," in director_source
          and 'api.addEventListener("myh3_roughcut_commit"' in director_source,
          "Director does not render or receive the rough-cut workspace")
    check("item.name !== TIMELINE_WIDGET" in director_source
          and "item.name !== ROUGHCUT_WIDGET" in director_source,
          "rough-cut JSON changes still rebuild the whole Director form")
    for required in ("素材百宝箱", "设置入点 I", "设置出点 O",
                     "沿用导演台时长", "使用时间轴 I/O 时长",
                     "自动对齐", "自动吸附", "MotionContext 窗口（推荐）",
                     "application/x-myang-asset", "导入工程 JSON",
                     "导出工程 JSON", "导出成片 MP4", "节目监视器",
                     "删除片段"):
        check(required in editor_source, f"rough-cut editor is missing {required}")
    check("browse-folder" in editor_source and "浏览并添加文件夹" in editor_source
          and "pathInput" not in editor_source,
          "rough-cut material library still requires users to type a folder path")
    style_source = (PACKAGE_DIR / "web" / "h3_prompt_editor.css").read_text("utf-8")
    check('open.className = "myh3-roughcut-open"' in editor_source
          and ".myh3-roughcut-open" in style_source
          and "white-space: nowrap" in style_source
          and "min-width: 118px" in style_source
          and ":focus-visible" in style_source,
          "rough-cut entry button can wrap or has no accessible focus state")
    check("+ editor.scroll.scrollLeft" not in editor_source,
          "horizontal scroll is still counted twice when placing clips")
    check("const TRACK_HEADER_WIDTH" in editor_source
          and "event.clientX - rect.left - TRACK_HEADER_WIDTH" in editor_source
          and "TRACK_HEADER_WIDTH + second * editor.zoom" in editor_source
          and "TRACK_HEADER_WIDTH + clip.timeline_start * editor.pixelsPerFrame" in editor_source
          and "TRACK_HEADER_WIDTH + frame * editor.pixelsPerFrame" in editor_source
          and "translate3d(${bounded * editor.pixelsPerFrame}px" in editor_source,
          "V1/A1 track header is not part of every timeline coordinate conversion")
    check("bindPlayheadPointer(editor, ruler)" in editor_source
          and "ruler.onpointermove" in editor_source
          and "setPointerCapture" in editor_source
          and "requestAnimationFrame" in editor_source
          and "myh3-rc-playhead-handle" in editor_source
          and "myh3-rc-playhead-time" in editor_source,
          "playhead is not a live, pointer-captured drag control with a visible handle")
    check("DIRECTOR_LIBRARY_ID" in editor_source
          and 'apiJson("/minimax-h3-myang/assets")' in editor_source
          and 'type: "input"' in editor_source
          and "assetUrl(asset)" in editor_source,
          "Director catalogue is not exposed as a draggable timeline material source")
    check('visualClip.kind === "image" ? "img"' in editor_source,
          "image clips are still previewed as videos")
    check("updateBoundaryStatus(editor)" in editor_source
          and "current.id === committed.id" in editor_source
          and "applyOverwrite(target" in editor_source,
          "boundary preflight or conflict-safe commit is missing")
    check("URL.createObjectURL(new Blob" in editor_source
          and "URL.revokeObjectURL" in editor_source
          and "`${MEDIA_ROUTE}/export`" in editor_source,
          "rough-cut JSON/MP4 export controls are not wired")
    check("Number(clip?.source_in) / sourceRate" in editor_source
          and "clipSourceSecond(editor" in editor_source,
          "program monitor ignores the clip source I/O offset")
    check('String(event.key || "").toLowerCase()' in editor_source
          and 'key === "i" || key === "o"' in editor_source,
          "timeline I/O keyboard shortcuts are missing")
    check("startTimelinePlayback(editor)" in editor_source
          and "syncTimelinePlaybackMedia(editor" in editor_source
          and "timelineClipAt(editor" in editor_source
          and "clipSourceSecond(editor" in editor_source
          and "播放时间轴（空格）" in editor_source
          and "停止并回到时间轴起点" in editor_source,
          "V1/A1 timeline playback transport or media synchronization is missing")
    check("deleteSelectedTimelineClip(editor)" in editor_source
          and "item.oncontextmenu" in editor_source
          and 'key === "delete" || key === "backspace"' in editor_source
          and "selectedTimelineClip(editor)" in editor_source,
          "timeline clips cannot be selected and deleted with visible controls or shortcuts")
    check("stepTimelineFrame(editor" in editor_source
          and "上一帧（左方向键）" in editor_source
          and "下一帧（右方向键）" in editor_source
          and "myh3-rc-program" in editor_source,
          "program monitor or frame stepping controls are missing")
    check('document.querySelectorAll(".myh3-rc-modal,.myh3-rc-backdrop")' in editor_source
          and 'event.stopPropagation()' in editor_source
          and '粗剪时间轴打开失败' in editor_source,
          "rough-cut editor can leak a modal or pass destructive events to the graph")
    check("project.selection.in_set" in editor_source
          and "project.selection.out_set" in editor_source
          and "markerSpecs" in editor_source,
          "timeline still renders fake I/O markers on first open")
    check("alignClipStart(editor" in editor_source
          and "snapPlayheadFrame(editor" in editor_source,
          "timeline alignment or snapping is not wired to editing")
    check("nodeWidget(node, ENABLED_WIDGET)?.value !== true" in editor_source
          and "readRoughCutProject(node).settings.writeback_enabled !== true"
          in editor_source
          and "writeback_enabled === true" in editor_source
          and "if (!applyRoughCutCommit(node, detail)) return" in director_source
          and "writeback_enabled is not True" in backend_source
          and '"writeback_enabled") is not True' in director_backend
          and "writeback_enabled=True" in director_backend,
          "disabled rough-cut write-back is not rejected at both frontend and backend")
    check("onDurationSeconds" in director_source
          and "syncTimelineDuration(editor)" in editor_source,
          "timeline I/O duration does not synchronize the Director")


if __name__ == "__main__":
    tests = [
        test_roughcut_writeback_requires_current_project_authorization,
        test_roughcut_overwrite_is_non_ripple_and_frame_accurate,
        test_roughcut_boundary_window_uses_material_outside_io,
        test_roughcut_audio_window_uses_a1_at_the_same_io_boundary,
        test_roughcut_rejects_media_on_the_wrong_track_kind,
        test_roughcut_library_is_allowlisted_and_contained,
        test_folder_library_imports_a_director_reference_without_moving_source,
        test_director_catalogue_input_source_is_safe_and_timeline_compatible,
        test_director_exposes_working_roughcut_contract_append_only,
        test_director_rejects_truthy_legacy_strings_for_roughcut_gate,
        test_roughcut_io_state_and_duration_modes_are_explicit,
        test_segment_merge_progress_and_memory_barrier_are_reload_safe,
        test_anchor_trim_removes_hidden_tail_and_matches_audio_duration,
        test_roughcut_plan_is_fitted_before_generation_and_keeps_h3_grid,
        test_double_sided_motion_context_preserves_visible_io_duration,
        test_roughcut_frontend_is_integrated_without_refreshing_on_playhead_edits,
    ]
    for test in tests:
        test()
        print("PASS", test.__name__)
