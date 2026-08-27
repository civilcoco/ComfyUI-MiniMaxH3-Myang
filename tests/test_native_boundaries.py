"""Boundary regressions found during the independent native-node review."""

import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


PACKAGE_DIR = Path(__file__).resolve().parents[1]
for path in (str(PACKAGE_DIR.parent.parent), str(PACKAGE_DIR.parent)):
    if path not in sys.path:
        sys.path.insert(0, path)

core = importlib.import_module("ComfyUI-MiniMaxH3-Myang.core")
legacy = importlib.import_module("ComfyUI-MiniMaxH3-Myang.nodes")
media_catalog = importlib.import_module("ComfyUI-MiniMaxH3-Myang.media_catalog")


def check_raises(needle, function):
    try:
        function()
    except ValueError as error:
        if needle not in str(error):
            raise AssertionError("wrong error: %s" % error) from error
        return
    raise AssertionError("expected ValueError containing %r" % needle)


def plan(count, frames, overlap):
    return json.dumps({
        "segment_count": count,
        "frames_per_segment": frames,
        "segment_seconds_snapped": frames / 24.0,
        "fps": 24.0,
        "overlap_frames": overlap,
    })


def run(plan_json, task, overlap, ref_video=None,
        drift_method="off", drift_strength=0.0):
    h3 = SimpleNamespace(
        video_vae=object(), audio_vae=object(),
        names={"model": "minimax/minimax_h3_ref2va_int8.safetensors"})
    return legacy.H3LongVideo().run(
        h3=h3, model=object(), sampler=object(), plan_json=plan_json,
        task_mode=task, resolution="480P", aspect_ratio="16:9",
        width=864, height=480, steps=8, denoise=1.0,
        scheduler="simple", noise_seed=0, context_length=str(overlap),
        prompt_mode=legacy.MODE_DIRECT, media_prefix="参考@视频1",
        llm_service="none", drift_method=drift_method,
        drift_strength=drift_strength, ref_image_size="匹配生成分辨率",
        save_segments=False, ref_video=ref_video, prompt="test")


def test_short_transfer_is_rejected_early():
    # Two 124-frame segments with a 22-frame shared window need 226 source
    # frames.  A 225-frame source must fail before segment 1 samples.
    reference = torch.zeros(225, 1, 1, 3)
    check_raises(
        "需要 226 帧",
        lambda: run(plan(2, 124, 22), legacy.TASK_TRANSFER, 22,
                    ref_video=reference))


def test_modern_video_and_soundtrack_are_normalised():
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

    class ModernVideo:
        def get_components(self):
            return SimpleNamespace(
                images=torch.zeros(30, 2, 2, 3),
                audio={"waveform": torch.zeros(1, 2, 32000),
                       "sample_rate": 32000},
                frame_rate=30.0)

    media = media_catalog.MyangMediaCatalog(assets=(
        media_catalog.MyangMediaAsset(1, "video", ModernVideo()),
    ))
    old_core = core._core
    core._core = lambda: FakeCore
    try:
        h3 = SimpleNamespace(clip=object(), video_vae=object(),
                             audio_vae=object())
        core.H3Condition().build(
            h3=h3, prompt="参考@视频1和@音频1",
            resolution="自定义", aspect_ratio="16:9",
            width=64, height=64, seconds=5.0,
            ref_image_size="匹配生成分辨率", media=media)
        frames = captured["ref_videos"]["ref_video_1"]
        if int(frames.shape[0]) != 24:
            raise AssertionError("30fps video was not resampled to 24fps")
        if "ref_video_audio_1" not in captured["ref_video_audios"]:
            raise AssertionError("embedded soundtrack was dropped")
        if captured["prompt"] != "参考<Video 1>和<Audio 1>":
            raise AssertionError("video/audio mentions were not aligned")
    finally:
        core._core = old_core


if __name__ == "__main__":
    for test in (
        test_short_transfer_is_rejected_early,
        test_modern_video_and_soundtrack_are_normalised,
    ):
        test()
        print("PASS", test.__name__)
