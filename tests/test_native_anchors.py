"""CPU-only regression checks for Myang's native H3 continuation."""

import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


PACKAGE_DIR = Path(__file__).resolve().parents[1]
CUSTOM_NODES = PACKAGE_DIR.parent
COMFY_ROOT = CUSTOM_NODES.parent
for path in (str(COMFY_ROOT), str(CUSTOM_NODES)):
    if path not in sys.path:
        sys.path.insert(0, path)

package = importlib.import_module("ComfyUI-MiniMaxH3-Myang")
anchors = importlib.import_module("ComfyUI-MiniMaxH3-Myang.anchors")
compat = importlib.import_module("ComfyUI-MiniMaxH3-Myang.anchor_compat")
core = importlib.import_module("ComfyUI-MiniMaxH3-Myang.core")
detail = importlib.import_module("ComfyUI-MiniMaxH3-Myang.detail")
legacy = importlib.import_module("ComfyUI-MiniMaxH3-Myang.nodes")
seam = importlib.import_module("ComfyUI-MiniMaxH3-Myang.seam")
turbo = importlib.import_module("ComfyUI-MiniMaxH3-Myang.turbo")


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def test_registration_and_schema():
    expected = {"H3AnchorContext", "H3AnchorKeyframe", "H3AnchorTrim",
                "H3Condition", "H3LongVideo", "H3TurboSchedule",
                "H3DetailRefine", "H3DetailSettings"}
    check(expected.issubset(package.NODE_CLASS_MAPPINGS),
          "native node mappings are incomplete")
    old_names = list(legacy._H3LongVideoInputs.INPUT_TYPES()["required"])
    new_names = list(legacy.H3LongVideo.INPUT_TYPES()["required"])
    # drift 校正与 anchor_schedule 已移除（回归 motion-context 全钉 + latent 无损）
    for gone in ("drift_method", "drift_strength", "anchor_schedule", "seam_blend_frames"):
        check(gone not in new_names, "H3LongVideo still exposes removed widget %s" % gone)
    expected_base = [
        "legacy_plan_padding" if n == "llm_service" else n
        for n in old_names if n not in ("drift_method", "drift_strength")]
    check(new_names[:len(expected_base)] == expected_base,
          "H3LongVideo positional widgets (minus drift/LLM migration) changed order")
    check("llm_service" not in new_names,
          "H3LongVideo still owns an LLM service input")
    check(legacy.H3LongVideo.INPUT_TYPES()["required"][
              "legacy_plan_padding"][0] == "STRING",
          "legacy LLM position is not an inert migration string")
    check(new_names[len(expected_base):] == ["detail_refinement", "save_raw_segments"],
          "native detail widgets were not appended after drift removal")
    optional = legacy.H3LongVideo.INPUT_TYPES()["optional"]
    check("二采设置" in optional and "refine_model" not in optional and
          "detail_settings" not in optional,
          "H3LongVideo did not collapse detail controls into one Chinese input")
    check("二采模型" in detail.H3DetailSettings.INPUT_TYPES()["optional"],
          "detail controller has no Chinese base-model input")
    check("ref_video" in core.H3Condition.INPUT_TYPES()["optional"],
          "H3Condition has no direct per-segment ref_video")


def test_layout_and_payload():
    check(compat.ensure_anchors(), "anchor patch failed")
    import comfy.ldm.minimax.model as mm
    import comfy.model_base as model_base

    check(getattr(mm.PackedLayout.__init__, anchors.LAYOUT_PATCH, False),
          "layout marker missing")
    check(getattr(model_base.MiniMaxH3.extra_conds,
                  anchors.PAYLOAD_PATCH, False), "payload marker missing")

    latent_t, frame_count = 7, anchors.pixel_frames(7)
    refs = [
        {"kind": "image", "latent_h": 8, "latent_w": 8},
        {"kind": "video_audio", "latent_t": 2,
         "latent_h": 8, "latent_w": 8, "ref_audio_t": 3},
        {"kind": "audio", "ref_audio_t": 4,
         anchors.AUDIO_END_FRAME: 4.8},
    ]
    positions = (0, 5, frame_count - 1)
    dummy_latent = torch.zeros(1, 16, 1, 8, 8)
    keyframes = [
        {"resolved_frame_index": 0, anchors.ANCHOR_FRAME: position, "latent": dummy_latent}
        for position in positions
    ]
    layout = mm.PackedLayout(
        11, latent_t, 8, 8, 12,
        keyframes=keyframes, refs=refs, frame_count=frame_count)
    origin = anchors.target_origin(layout)
    cond = [(a, b) for a, b, kind in layout.segments if kind == "cond"]
    check(len(cond) == 3, "wrong number of cond segments")
    check(float(layout.position_ids[cond[0][0], 0]) == origin,
          "first anchor missed shifted target origin")
    check(abs(float(layout.position_ids[cond[1][0], 0])
              - (origin + anchors.FRAME_RESCALE * 5)) < 1e-9,
          "interior anchor missed shifted target origin")
    if hasattr(mm, "_video_t_spans"):
        expected_last = (origin + sum(mm._video_t_spans(latent_t))
                         - mm.FRAME_RESCALE)
    else:
        expected_last = origin + anchors.FRAME_RESCALE * float(frame_count - 1)
    check(float(layout.position_ids[cond[2][0], 0]) == expected_last,
          "last anchor is not bit-equivalent to stock")

    audio_span = anchors._ref_segment_map(layout, refs)[2]["ref_audio"]
    actual_end = float(
        layout.position_ids[audio_span[0]:audio_span[1], 0].max()) + 1.0
    check(abs(actual_end - (origin + 8.0)) < 1e-9,
          "continuation audio is not end-aligned on target grid")

    old_original = anchors._payload_original
    try:
        anchors._payload_original = lambda _self, **_kwargs: {
            "minimax_payload": SimpleNamespace(cond={})}
        kf_latents = [object(), object()]
        ref_latents = [object(), object()]
        audio_latent = object()
        output = anchors._patched_payload(
            object(),
            minimax_keyframes=[
                {anchors.ANCHOR_FRAME: 0, "latent": kf_latents[0]},
                {anchors.ANCHOR_FRAME: 1, "latent": kf_latents[1]},
            ],
            minimax_refs=[
                {"kind": "image", "latent": ref_latents[0]},
                {"kind": "audio", "audio_latent": audio_latent,
                 anchors.AUDIO_END_FRAME: 1.2},
                {"kind": "video", "latent": ref_latents[1]},
            ],
            minimax_frame_count=124,
        )
        payload = output["minimax_payload"].cond
        check(payload["cond_video_latents"] == kf_latents + ref_latents,
              "payload video rows are not keyframes + refs")
        check(payload["cond_audio_latents"] == [audio_latent],
              "payload audio reference order changed")
        check(payload["frame_count"] == 124, "payload frame_count missing")
    finally:
        anchors._payload_original = old_original


def test_temporal_windows_and_audio_grid():
    video = torch.zeros(1, 16, 57, 2, 2)
    audio = torch.zeros(1, 32, 2, 320)
    latent = {"samples": [video, audio]}
    for frames, expected_steps in ((5, 2), (22, 7), (39, 12), (56, 17)):
        blocks, positions, covered = anchors._latent_blocks(latent, frames)
        check(len(blocks) == expected_steps, "%d-frame block count" % frames)
        check(covered == frames, "%d-frame coverage" % frames)
        check(positions == anchors.step_offsets(expected_steps),
              "%d-frame offsets" % frames)
        ref = anchors._latent_audio_ref(latent, frames)
        coordinate = ref[anchors.AUDIO_END_FRAME] * anchors.FRAME_RESCALE
        expected = int(frames * anchors.AUDIO_HZ // anchors.FPS)
        check(abs(coordinate - expected) < 1e-9,
              "%d-frame audio endpoint did not use the safe floor grid" % frames)

    # 124 frames produce 207 audio steps, but only 206 end before the video cut.
    latent_124 = {
        "samples": [torch.zeros(1, 16, 37, 2, 2),
                    torch.zeros(1, 32, 2, 207)]}
    ref = anchors._latent_audio_ref(latent_124, 22)
    check(abs(ref[anchors.AUDIO_END_FRAME] * anchors.FRAME_RESCALE - 36.0)
          < 1e-9, "audio guard included the step beyond the video cut")
    check(int(ref["audio_latent"].shape[-1]) == 36,
          "audio guard did not crop to complete 40Hz steps")


def test_overlap_latent_is_seeded_and_excluded_from_noise():
    target_video = torch.zeros(1, 16, 37, 2, 2)
    target_audio = torch.zeros(1, 32, 2, 207)
    source_video = torch.arange(37, dtype=torch.float32).view(
        1, 1, 37, 1, 1).expand(1, 16, 37, 2, 2).clone()
    source_audio = torch.zeros(1, 32, 2, 207)
    conditioning, trim, seeded = anchors.H3AnchorContext().apply(
        conditioning=[], vae=object(),
        latent={"samples": [target_video, target_audio]},
        context_length="22",
        context_latent={"samples": [source_video, source_audio]})
    check(conditioning == [] and trim == 22,
          "anchor metadata changed while adding the noise mask")
    video, audio = seeded["samples"].unbind()
    video_mask, audio_mask = seeded["noise_mask"].unbind()
    check(torch.equal(video[:, :, :7], source_video[:, :, -7:]),
          "overlap did not copy the previous segment's tail latent")
    check(torch.count_nonzero(video[:, :, 7:]) == 0,
          "new timeline was overwritten while seeding the overlap")
    check(torch.count_nonzero(video_mask[:, :, :7]) == 0,
          "overlap still receives random noise / denoising")
    check(torch.all(video_mask[:, :, 7:] == 1),
          "new frames no longer receive normal random noise")
    check(torch.all(audio_mask == 1) and torch.equal(audio, target_audio),
          "visual-only anchor masking unexpectedly froze the audio stream")


def test_seam_uses_matching_audio_window():
    prev_images = torch.zeros(6, 1, 1, 3)
    prev_images[-1] = 0.25
    next_images = torch.zeros(8, 1, 1, 3)
    next_images[:5] = torch.tensor(
        [0.10, 0.20, 0.30, 0.80, 0.90]).view(5, 1, 1, 1)
    next_images[5:] = 0.95

    rate, cut = 1000, 200
    prev_audio = {"waveform": torch.zeros(1, 1, 300), "sample_rate": rate}
    next_wave = torch.arange(400, dtype=torch.float32).view(1, 1, -1)
    next_audio = {"waveform": next_wave, "sample_rate": rate}
    prev_out, prev_audio_out, tail, tail_audio, _report = seam.H3SeamBlend().join(
        prev_images, next_images, trim_frames=5, blend_frames=2,
        curve=seam.CURVE_SMOOTH, fps=25.0,
        prev_audio=prev_audio, next_audio=next_audio)

    check(abs(float(prev_out[-1].mean()) - 0.90) < 1e-6,
          "visual seam used the beginning, not the end, of the pinned head")
    check(abs(float(prev_audio_out["waveform"][..., -1]) - (cut - 1)) < 1e-6,
          "audio seam repeated the beginning of the duplicate window")
    check(int(tail.shape[0]) == 3, "visual trim changed output frame count")
    check(int(tail_audio["waveform"].shape[-1]) == 120,
          "audio tail was not capped to delivered video duration")


def test_aimdo_headroom_does_not_raise_pin_budget():
    import comfy.model_management as mm
    old_fn = mm.pinned_hostbuf_size
    old_cap = mm.MAX_PINNED_MEMORY
    old_windows = mm.WINDOWS
    old_high_ram = mm.args.high_ram
    try:
        mm.MAX_PINNED_MEMORY = 100
        mm.WINDOWS = True
        mm.args.high_ram = False
        mm.pinned_hostbuf_size = lambda size: min(int(size), 100) * 2
        check(core._ensure_aimdo_hostbuf_headroom(),
              "aimdo hostbuf wrapper was not installed")
        check(mm.MAX_PINNED_MEMORY == 100,
              "aimdo workaround raised the registered pin budget")
        check(mm.pinned_hostbuf_size(50) == 100,
              "small model reservation changed")
        check(mm.pinned_hostbuf_size(200)
              == 200 + core.AIMDO_HOSTBUF_ALIGNMENT_SLACK,
              "large model virtual reservation does not cover the model")
    finally:
        mm.pinned_hostbuf_size = old_fn
        mm.MAX_PINNED_MEMORY = old_cap
        mm.WINDOWS = old_windows
        mm.args.high_ram = old_high_ram


def test_trim():
    images = torch.zeros(124, 2, 2, 3)
    sample_rate = 48000
    audio = {"waveform": torch.zeros(1, 2, 250000),
             "sample_rate": sample_rate}
    trimmed_images, trimmed_audio = anchors.H3AnchorTrim().trim(
        images, 22, audio=audio, fps=24.0)
    check(int(trimmed_images.shape[0]) == 102, "wrong image trim")
    expected_samples = round(102 / 24.0 * sample_rate)
    check(int(trimmed_audio["waveform"].shape[-1]) == expected_samples,
          "audio tail does not match delivered frames")
    short_audio = {
        "waveform": torch.ones(1, 2, round((22 / 24.0) * sample_rate) + 1000),
        "sample_rate": sample_rate,
    }
    _, padded_audio = anchors.H3AnchorTrim().trim(
        images, 22, audio=short_audio, fps=24.0)
    check(int(padded_audio["waveform"].shape[-1]) == expected_samples,
          "short audio tail was not padded to delivered video duration")
    check(torch.count_nonzero(padded_audio["waveform"][..., 1000:]) == 0,
          "short audio padding fabricated non-zero samples")


def _plan(count=2, overlap=22):
    frames = 124
    return json.dumps({
        "segment_count": count,
        "frames_per_segment": frames,
        "segment_seconds_snapped": frames / 24.0,
        "fps": 24.0,
        "overlap_frames": overlap,
        "total_seconds_actual": (frames + (count - 1) *
                                 (frames - overlap)) / 24.0,
    })


def _detail_settings(enabled=True, resolution="928P", model=None):
    return detail.H3DetailSettings().build(
        enabled, resolution, 1664, 928, 4, 0.2, "beta",
        "res_multistep", "bicubic", 4, **{"二采模型": model})[0]


def _expand(task, ref_video=None, overlap=22,
            detail_refinement=legacy.DETAIL_NATIVE, model=None, sampler=None,
            steps=8, scheduler="simple", denoise=1.0,
            drift_method="off", drift_strength=0.0, detail_settings=None):
    h3 = SimpleNamespace(video_vae=object(), audio_vae=object())
    return legacy.H3LongVideo().run(
        h3=h3,
        model=model if model is not None else object(),
        sampler=sampler if sampler is not None else object(),
        plan_json=_plan(overlap=overlap),
        task_mode=task,
        resolution="480P",
        aspect_ratio="16:9",
        width=864,
        height=480,
        steps=steps,
        denoise=denoise,
        scheduler=scheduler,
        noise_seed=1,
        context_length=str(overlap),
        prompt_mode=legacy.MODE_DIRECT,
        media_prefix="参考@视频1",
        llm_service="none",
        drift_method=drift_method,
        drift_strength=drift_strength,
        ref_image_size="匹配生成分辨率",
        detail_refinement=detail_refinement,
        **{"二采设置": detail_settings},
        save_segments=False,
        ref_video=ref_video,
        prompt="参考@视频1",
    )["expand"]


def test_dynamic_expansion():
    graph = _expand(legacy.TASK_FRESH)
    kinds = [node["class_type"] for node in graph.values()]
    check(kinds.count("H3Condition") == 2, "wrong H3Condition count")
    check(kinds.count("H3AnchorContext") == 1, "wrong anchor count")
    check(kinds.count("H3AnchorTrim") == 2, "wrong trim count")
    check(kinds.count("H3SeamBlend") == 0, "H3SeamBlend should be removed in favor of pure trim")
    check("H3SegmentPrompt" not in kinds,
          "H3LongVideo still creates an LLM prompt-refinement node")
    anchor_id, anchor_node = next((node_id, node) for node_id, node in graph.items()
                                  if node["class_type"] == "H3AnchorContext")
    check("context_latent" in anchor_node["inputs"],
          "latent continuation path was not used")
    check("context_audio" in anchor_node["inputs"],
          "decoded audio was not re-encoded onto the safe 40Hz grid")
    second_sampler = next(node for node in graph.values()
                          if node["class_type"] == "H3SamplerAdvanced"
                          and node["inputs"].get("segment_index") == 2
                          and node["inputs"].get("pass_label") == "sample1")
    check(second_sampler["inputs"]["latent_image"] == [anchor_id, 2],
          "segment sampler bypassed the anchor's zero-noise latent output")
    first_trim = next(node for node in graph.values()
                      if node["class_type"] == "H3AnchorTrim")
    check(first_trim["inputs"]["trim_frames"] == 0,
          "first segment audio was not normalised to video duration")
    check(not any("MotionContext" in kind for kind in kinds),
          "third-party MotionContext remains in expansion")
    check("ExtendIntermediateSigmas" not in kinds,
          "detail refinement must remain off by default")
    check("MiniMaxH3SigmaShift" not in kinds,
          "official H3 AV sigma shifts must not be rewritten")
    check("SplitSigmasDenoise" not in kinds,
          "dual-stage sampling must not be introduced")

    reference = torch.zeros(226, 1, 1, 3)
    graph = _expand(legacy.TASK_TRANSFER, ref_video=reference)
    conditions = [node for node in graph.values()
                  if node["class_type"] == "H3Condition"]
    check(len(conditions) == 2, "action transfer condition count")
    for node in conditions:
        link = node["inputs"].get("ref_video")
        check(isinstance(link, list), "action-transfer clip was dropped")
        check(graph[link[0]]["class_type"] == "ImageFromBatch",
              "H3Condition ref_video does not come from per-segment slice")


def test_low_sigma_refinement():
    graph = _expand(
        legacy.TASK_FRESH,
        detail_refinement=legacy.DETAIL_BALANCED)
    refiners = [node for node in graph.values()
                if node["class_type"] == "ExtendIntermediateSigmas"]
    samplers = [node for node in graph.values()
                if node["class_type"] == "H3SamplerAdvanced"]
    check(len(refiners) == 2, "each segment needs one sigma refiner")
    check(len(samplers) == 2, "refinement must not duplicate the sampler")
    for node in refiners:
        inputs = node["inputs"]
        check(inputs["steps"] == 2, "balanced subdivision changed")
        check(inputs["start_at_sigma"] == 0.8, "wrong refine start")
        check(inputs["end_at_sigma"] == 0.0, "wrong refine end")
        check(inputs["spacing"] == "cosine", "wrong refine spacing")
    for node in samplers:
        link = node["inputs"]["sigmas"]
        check(graph[link[0]]["class_type"] == "ExtendIntermediateSigmas",
              "sampler bypassed the refined sigma schedule")
    kinds = [node["class_type"] for node in graph.values()]
    check("MiniMaxH3SigmaShift" not in kinds, "H3 shift was overridden")
    check("SplitSigmasDenoise" not in kinds, "schedule was split in two")
    strong = legacy.detail_refinement_params(legacy.DETAIL_STRONG)
    check(strong == {"steps": 3, "start_at_sigma": 0.8,
                     "end_at_sigma": 0.0, "spacing": "cosine"},
          "strong refinement profile changed")


def test_second_pass_detail_refinement():
    check(any("nvidia_rtx_vsr" in str(m) for m in detail.DETAIL_UPSCALE_METHODS),
          "NVIDIA RTX VSR option was removed from the detail pass")
    settings = _detail_settings(True, "832P", object())
    check(settings["enabled"] and settings["resolution"] == "832P",
          "dedicated detail settings changed")

    source = torch.linspace(0, 1, 5 * 32 * 64 * 3).reshape(5, 32, 64, 3)
    width, height = detail.target_size(source, "自定义", 128, 64)
    check((width, height) == (128, 64), "custom second-pass size changed")
    resized = detail.resize_frames(source, width, height, "bicubic", 2)
    check(tuple(resized.shape) == (5, 64, 128, 3), "chunked resize shape")
    check(resized.dtype == torch.float16 and resized.device.type == "cpu",
          "second-pass staging tensor is not compact CPU storage")

    video = torch.zeros(1, 24, 2, 2, 4)
    audio = torch.zeros(1, 32, 2, 8)
    joined = detail.join_av_latent(video, audio)["samples"]
    check(joined.is_nested and len(joined.unbind()) == 2,
          "H3 AV latent was not combined in video/audio order")

    class VideoVAE:
        def encode(self, pixels):
            check(tuple(pixels.shape) == (5, 64, 128, 3),
                  "detail VAE did not receive the resized frames")
            return video

    class AudioVAE:
        audio_sample_rate = 32000

        def encode(self, waveform):
            check(tuple(waveform.shape) == (1, 6667, 2),
                  "detail audio VAE input changed")
            return audio

    h3 = SimpleNamespace(video_vae=VideoVAE(), audio_vae=AudioVAE())
    original_audio = {
        "waveform": torch.zeros(1, 2, 6667), "sample_rate": 32000}
    expanded = detail.H3DetailRefine().refine(
        h3=h3, model=object(), conditioning=[], images=source,
        audio=original_audio, resolution="自定义", width=128, height=64,
        upscale_method="bicubic", chunk_frames=2, steps=4, denoise=0.2,
        scheduler="beta", sampler_name="res_multistep", noise_seed=1)
    graph = expanded["expand"]
    kinds = [node["class_type"] for node in graph.values()]
    check(kinds.count("SamplerCustomAdvanced") == 1,
          "standalone detail node did not build one low-noise sampler")
    sampler = next(node for node in graph.values()
                   if node["class_type"] == "SamplerCustomAdvanced")
    joined = sampler["inputs"]["latent_image"]["samples"]
    check(joined.is_nested and len(joined.unbind()) == 2,
          "detail sampler lost the directly encoded joint AV latent")
    check("VAEEncode" not in kinds and "VAEEncodeAudio" not in kinds,
          "large resized frames leaked into the dynamic graph cache")

    graph = _expand(
        legacy.TASK_FRESH,
        detail_settings=_detail_settings(True, "928P", object()))
    kinds = [node["class_type"] for node in graph.values()]
    check(kinds.count("H3LatentUpscale") == 2,
          "long-video detail pass is not applied once per segment")
    refiners = [node for node in graph.values()
                if node["class_type"] == "H3LatentUpscale"]
    for node in refiners:
        check(node["inputs"]["resolution"] == "928P",
              "long-video detail target was not forwarded")
    second_pass_samplers = [node for node in graph.values()
                            if node["class_type"] == "H3SamplerAdvanced"
                            and node["inputs"].get("pass_label") == "sample2"]
    check(len(second_pass_samplers) == 2,
          "long-video detail conditioning did not reach one sampler per segment")

    graph = _expand(
        legacy.TASK_FRESH,
        detail_settings=_detail_settings(True, "928P", object()),
        drift_method="mean_std", drift_strength=0.6)
    anchor = next(node for node in graph.values()
                  if node["class_type"] == "H3AnchorContext")
    if "context_latent" in anchor["inputs"]:
        context_link = anchor["inputs"]["context_latent"]
        context_source = graph[context_link[0]]
        check(context_source["class_type"] == "H3SamplerAdvanced"
              and context_source["inputs"].get("pass_label") == "sample1",
              "refined high-resolution latent leaked into low-resolution anchors")
    else:
        context_link = anchor["inputs"]["context_frames"]
        context_trim = graph[context_link[0]]
        check(context_trim["class_type"] == "H3AnchorTrim",
              "refined high-resolution frames leaked into low-resolution anchors")
    trims = [node for node in graph.values()
             if node["class_type"] == "H3AnchorTrim"]
    check(len(trims) >= 2, "refined stream did not use H3AnchorTrim")

    try:
        _expand(legacy.TASK_FRESH,
                detail_settings=_detail_settings(True, "928P", None))
    except ValueError as error:
        check("二采模型" in str(error), "wrong missing detail-model error")
    else:
        raise AssertionError("second pass accepted a missing base model")

    graph = _expand(legacy.TASK_FRESH, detail_settings=settings)
    refiners = [node for node in graph.values()
                if node["class_type"] == "H3LatentUpscale"]
    check(len(refiners) == 2 and all(
        node["inputs"]["resolution"] == "832P" for node in refiners),
        "dedicated settings did not enable and override the detail pass")
    disabled = dict(settings, enabled=False)
    graph = _expand(legacy.TASK_FRESH, detail_settings=disabled)
    check(not any(node["class_type"] == "H3LatentUpscale"
                  for node in graph.values()),
          "dedicated switch did not disable the detail pass")


def test_turbo_schedule_contract():
    spec = turbo.resolve_profile(turbo.PROFILE_8_V1)
    check(spec["shift_video"] == 12.0 and spec["shift_audio"] == 3.0,
          "8-step v1 schedule changed")
    check(spec["allowed_steps"] == (8, 4),
          "8-step v1 official inference choices changed")
    spec_768 = turbo.resolve_profile(turbo.PROFILE_4_V1_768)
    check(spec_768["shift_video"] == 6.0 and
          spec_768["shift_audio"] == 3.0,
          "768p v1 schedule changed")

    import comfy_extras.nodes_minimax_h3 as official
    captured = {}
    original = official.MiniMaxH3SigmaShift

    class FakeShift:
        @classmethod
        def execute(cls, model, shift_video, shift_audio):
            captured.update(video=shift_video, audio=shift_audio)
            patched = SimpleNamespace(
                model_options={"transformer_options": {}})
            return SimpleNamespace(result=(patched,))

    official.MiniMaxH3SigmaShift = FakeShift
    try:
        result = turbo.H3TurboSchedule().apply(
            object(), turbo.PROFILE_8_V1)
    finally:
        official.MiniMaxH3SigmaShift = original
    check(captured == {"video": 12.0, "audio": 3.0},
          "adapter did not call the official AV shift")
    check(result[1:] == (8, 12.0, 3.0), "adapter outputs changed")
    marker = turbo.turbo_metadata(result[0])
    check(marker is not None and marker["recommended_steps"] == 8,
          "Turbo schedule marker missing")

    def sample_euler(*args, **kwargs):
        return None

    graph = _expand(
        legacy.TASK_FRESH, model=result[0],
        sampler=SimpleNamespace(sampler_function=sample_euler))
    check(any(node["class_type"] == "H3SamplerAdvanced"
              for node in graph.values()), "valid Turbo contract did not expand")
    graph = _expand(
        legacy.TASK_FRESH, model=result[0],
        detail_settings=_detail_settings(True, "832P", object()),
        sampler=SimpleNamespace(sampler_function=sample_euler))
    check(sum(node["class_type"] == "H3LatentUpscale"
              for node in graph.values()) == 2,
          "Turbo first pass plus base-model detail upscale did not expand")
    check(sum(node["class_type"] == "H3SamplerAdvanced"
              and node["inputs"].get("pass_label") == "sample2"
              for node in graph.values()) == 2,
          "Turbo detail branch did not build two base-model second passes")
    try:
        _expand(
            legacy.TASK_FRESH, model=result[0],
            detail_settings=_detail_settings(True, "832P", result[0]),
            sampler=SimpleNamespace(sampler_function=sample_euler))
    except ValueError as error:
        check("二采模型" in str(error), "wrong Turbo detail-branch error")
    else:
        raise AssertionError("detail pass accepted the Turbo model branch")

    def must_fail(needle, **kwargs):
        try:
            _expand(legacy.TASK_FRESH, model=result[0],
                    sampler=SimpleNamespace(sampler_function=sample_euler),
                    **kwargs)
        except ValueError as error:
            check(needle in str(error), "wrong Turbo validation error: %s" % error)
            return
        raise AssertionError("Turbo validation did not reject %s" % needle)

    manual_step_graph = _expand(
        legacy.TASK_FRESH, model=result[0], steps=6,
        sampler=SimpleNamespace(sampler_function=sample_euler))
    manual_scheduler = next(node for node in manual_step_graph.values()
                            if node["class_type"] == "BasicScheduler")
    check(manual_scheduler["inputs"]["steps"] == 6,
          "Turbo model still rejected or replaced manual NFE")
    must_fail("scheduler=simple", scheduler="beta")
    must_fail("denoise=1.0", denoise=0.8)
    graph = _expand(
        legacy.TASK_FRESH, model=result[0],
        sampler=SimpleNamespace(sampler_function=sample_euler),
        detail_refinement=legacy.DETAIL_BALANCED)
    check(not any(node["class_type"] == "ExtendIntermediateSigmas"
                  for node in graph.values()),
          "Turbo did not automatically suppress low-sigma refinement")

    def sample_res_multistep(*args, **kwargs):
        return None

    try:
        _expand(
            legacy.TASK_FRESH, model=result[0],
            sampler=SimpleNamespace(sampler_function=sample_res_multistep))
    except ValueError as error:
        check("Euler" in str(error), "wrong Turbo sampler error")
    else:
        raise AssertionError("Turbo contract accepted res_multistep")


def test_direct_reference_condition():
    captured = {}

    class FakeReference:
        @staticmethod
        def execute(**kwargs):
            captured.update(kwargs)
            return "conditioning", "latent"

    class FakeCore:
        MiniMaxH3ReferenceToVideo = FakeReference

        @staticmethod
        def align_frame_count(value):
            return int(value)

        @staticmethod
        def video_latent_t(value):
            return int(value)

    old_core = core._core
    core._core = lambda: FakeCore
    try:
        h3 = SimpleNamespace(clip=object(), video_vae=object(),
                             audio_vae=object())
        video = torch.zeros(5, 2, 2, 3)
        result = core.H3Condition().build(
            h3=h3,
            prompt="参考@视频1的动作",
            resolution="自定义",
            aspect_ratio="16:9",
            width=64,
            height=64,
            seconds=5.0,
            ref_image_size="匹配生成分辨率",
            ref_video=video,
        )
        check(result[0] == "conditioning", "condition result changed")
        check(captured["prompt"] == "参考<Video 1>的动作",
              "@视频1 was not translated")
        check(captured["ref_videos"]["ref_video_1"] is video,
              "direct ref_video did not reach official core")
    finally:
        core._core = old_core


if __name__ == "__main__":
    tests = [
        test_registration_and_schema,
        test_layout_and_payload,
        test_temporal_windows_and_audio_grid,
        test_overlap_latent_is_seeded_and_excluded_from_noise,
        test_seam_uses_matching_audio_window,
        test_aimdo_headroom_does_not_raise_pin_budget,
        test_trim,
        test_dynamic_expansion,
        test_low_sigma_refinement,
        test_second_pass_detail_refinement,
        test_turbo_schedule_contract,
        test_direct_reference_condition,
    ]
    for test in tests:
        test()
        print("PASS", test.__name__)
