"""CPU regressions for integrated H3 audio refinement and segment seams."""

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
audio = importlib.import_module("ComfyUI-MiniMaxH3-Myang.audio_refine")
director = importlib.import_module("ComfyUI-MiniMaxH3-Myang.director")
legacy = importlib.import_module("ComfyUI-MiniMaxH3-Myang.nodes")
native = importlib.import_module("ComfyUI-MiniMaxH3-Myang.nodes")


def check(value, message):
    if not value:
        raise AssertionError(message)


def _plan(count=2):
    return json.dumps({
        "segment_count": count,
        "frames_per_segment": 124,
        "segment_seconds_snapped": 124 / 24,
        "overlap_frames": 22,
        "fps": 24,
        "segments": [
            {"index": index, "prompt": "镜头%d" % index, "frames": 124}
            for index in range(1, count + 1)
        ],
    }, ensure_ascii=False)


def _long_video(**extra):
    return native.H3LongVideo().run(
        h3=SimpleNamespace(video_vae=object(), audio_vae=object(), names={}),
        model=object(), sampler=object(), plan_json=_plan(),
        task_mode=legacy.TASK_FRESH, resolution="480P",
        aspect_ratio="16:9", width=864, height=480,
        steps=25, denoise=1.0, scheduler="simple", noise_seed=7,
        context_length="22", prompt_mode=legacy.MODE_DIRECT,
        media_prefix="", ref_image_size="匹配生成分辨率",
        save_segments=False, segment_prefix="video/test",
        save_raw_segments=False, **extra)["expand"]


def test_registration_and_director_schema():
    expected = {
        "H3AudioRefineMask", "H3AudioRefineSampler",
        "H3AudioSettings", "H3AudioSeam",
    }
    check(expected.issubset(package.NODE_CLASS_MAPPINGS),
          "integrated audio nodes were not registered")
    required = list(director.H3Director.INPUT_TYPES()["required"])
    tail = [
        "音频精修开启", "音频精修步数", "音频去噪强度", "音频精修采样器",
        "音频精修调度器", "音频接缝平滑", "音频接缝时长",
    ]
    audio_start = required.index(tail[0])
    check(required[audio_start:audio_start + len(tail)] == tail,
          "audio widgets were not appended as a contiguous compatibility block")
    check(audio_start >= 1 and required[audio_start - 1] == "二采复用一采条件",
          "audio widgets moved ahead of the pre-existing Director controls")
    check("音频设置" in native.H3LongVideo.INPUT_TYPES()["optional"],
          "long-video node has no combined audio settings socket")


def test_native_mask_freezes_video_and_opens_audio():
    video = torch.zeros(1, 24, 3, 2, 2)
    sound = torch.zeros(1, 32, 2, 120)
    mask = audio._build_av_noise_mask(video, sound, 0.0)
    video_mask, audio_mask = mask.unbind()
    check(float(video_mask.max()) == 0.0,
          "audio refinement accidentally opened the video stream")
    check(float(audio_mask.min()) == 1.0,
          "audio refinement did not fully open the audio stream")


def test_seam_preserves_duration_and_matches_the_cut():
    rate, fps, trim, frames = 32000, 24.0, 22, 124
    previous = torch.zeros(1, 2, 16000)
    current = torch.ones(1, 2, 180000)
    images = torch.zeros(frames, 2, 2, 3)
    previous_out, current_out, report = audio.H3AudioSeam().smooth(
        {"waveform": previous, "sample_rate": rate},
        {"waveform": current, "sample_rate": rate},
        images, trim, 80.0, fps)
    expected = round((frames - trim) / fps * rate)
    check(previous_out["waveform"].shape[-1] == previous.shape[-1],
          "seam changed the previous segment duration")
    check(current_out["waveform"].shape[-1] == expected,
          "seam changed the current A/V duration")
    check(float(previous_out["waveform"][..., -1].mean()) > 0.70,
          "previous tail did not reach the duplicated anchor audio")
    check("总时长不变" in report, "seam report lost its duration guarantee")


def test_graph_adds_one_seam_per_boundary():
    graph = _long_video(**{"音频设置": {
        "refine_enabled": False,
        "seam_enabled": True,
        "seam_ms": 80,
    }})
    kinds = [entry["class_type"] for entry in graph.values()]
    check(kinds.count("H3AudioSeam") == 1,
          "two segments should create exactly one audio seam")
    collector = next(entry for entry in graph.values()
                     if entry["class_type"] == "H3SegmentCollector")
    seam_id, _seam = next((node_id, entry) for node_id, entry in graph.items()
                          if entry["class_type"] == "H3AudioSeam")
    check(collector["inputs"]["audios_1"] == [seam_id, 0],
          "collector still hard-concatenates the unmodified previous tail")


def test_graph_refines_audio_before_decode_and_continuation():
    base_model = object()
    graph = _long_video(**{"音频设置": {
        "refine_enabled": True, "steps": 4, "denoise": 0.5,
        "sampler_name": "euler", "scheduler": "simple",
        "seam_enabled": False, "model": base_model,
    }})
    kinds = [entry["class_type"] for entry in graph.values()]
    check(kinds.count("H3AudioRefineSampler") == 2,
          "audio refinement was not created once per segment")
    refiners = [entry for entry in graph.values()
                if entry["class_type"] == "H3AudioRefineSampler"]
    check(all(entry["inputs"]["model"] is base_model for entry in refiners),
          "audio refinement did not use the undistilled base model")
    decoders = [entry for entry in graph.values()
                if entry["class_type"] == "VAEDecodeAudio"]
    refiner_ids = {node_id for node_id, entry in graph.items()
                   if entry["class_type"] == "H3AudioRefineSampler"}
    check(any(entry["inputs"]["samples"][0] in refiner_ids for entry in decoders),
          "audio decoder still reads the Turbo latent before refinement")


def test_director_builds_integrated_audio_settings():
    result = director.H3Director().run(
        h3=object(), model=object(), sampler=object(),
        source_mode=director.DIRECTOR_TIMELINE,
        timeline_json='{"shots":[{"prompt":"镜头一","duration_seconds":5}]}',
        script_fallback="", total_seconds=5, segment_seconds=5,
        llm_enabled=False, llm_service="none", task_mode=legacy.TASK_FRESH,
        resolution="480P", aspect_ratio="16:9", width=864, height=480,
        steps=25, denoise=1.0, scheduler="simple", noise_seed=0,
        context_length="22", ref_image_size="匹配生成分辨率",
        save_segments=False, segment_prefix="video/test",
        save_raw_segments=False,
        **{"音频精修开启": True, "音频精修步数": 6,
           "音频去噪强度": 0.45, "音频接缝平滑": True,
           "音频接缝时长": 100.0})["expand"]
    settings = next(entry for entry in result.values()
                    if entry["class_type"] == "H3AudioSettings")
    long_video = next(entry for entry in result.values()
                      if entry["class_type"] == "H3LongVideo")
    check(settings["inputs"]["refine_steps"] == 6
          and settings["inputs"]["seam_ms"] == 100.0,
          "Director audio values did not reach the settings node")
    check("音频设置" in long_video["inputs"],
          "combined audio settings did not reach H3LongVideo")


if __name__ == "__main__":
    tests = [value for name, value in list(globals().items())
             if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
        print("PASS", test.__name__)
