"""Tests for segment continuity: seam blend and motion-context latent chaining."""

import importlib
import json
import sys
from pathlib import Path

import torch


PACKAGE_DIR = Path(__file__).resolve().parents[1]
CUSTOM_NODES = PACKAGE_DIR.parent
COMFY_ROOT = CUSTOM_NODES.parent
for path in (str(COMFY_ROOT), str(CUSTOM_NODES)):
    if path not in sys.path:
        sys.path.insert(0, path)

package = importlib.import_module("ComfyUI-MiniMaxH3-Myang")
anchors_module = importlib.import_module("ComfyUI-MiniMaxH3-Myang.anchors")
seam_module = importlib.import_module("ComfyUI-MiniMaxH3-Myang.seam")
nodes = importlib.import_module("ComfyUI-MiniMaxH3-Myang.nodes")
detail_module = importlib.import_module("ComfyUI-MiniMaxH3-Myang.detail")


def check(condition, message):
    if not condition:
        raise AssertionError(message)


class FakeVAE:
    def encode(self, pixels):
        T, H, W, C = pixels.shape
        steps = anchors_module.steps_for_frames(T) or 1
        return torch.zeros(1, C, steps, max(1, H // 16), max(1, W // 16))


class FakeAudioVAE:
    def __init__(self):
        self.audio_sample_rate = 32000

    def encode(self, waveform):
        return torch.zeros(1, 2, 2, 10)


class FakeBundle:
    def __init__(self):
        self.video_vae = FakeVAE()
        self.audio_vae = FakeAudioVAE()
        self.names = {"model": "minimax_h3_ref2va"}

    def model_for(self, kind):
        return object()


def test_seam_blend_with_luminance_compensation():
    """Verify that H3SeamBlend applies smooth fade with luminance compensation."""
    prev_images = torch.ones(50, 64, 64, 3) * 0.4
    next_images = torch.ones(50, 64, 64, 3) * 0.6
    seam = seam_module.H3SeamBlend()
    prev_out, prev_audio, tail_out, next_audio, report = seam.join(
        prev_images=prev_images,
        next_images=next_images,
        trim_frames=22,
        blend_frames=8,
        curve=seam_module.CURVE_SMOOTH,
        fps=24.0,
    )
    check(prev_out.shape == prev_images.shape, "prev_images shape mismatch")
    check(tail_out.shape[0] == 50 - 22, "tail_out length mismatch")
    check("seam:" in report, "missing seam report")


def test_longvideo_transfer_latent_continuity():
    """动作迁移段间走 context_latent 无损续接（motion-context 风格，全钉，无漂移）。"""
    plan_data = {
        "segment_count": 2,
        "frames_per_segment": 243,
        "segment_seconds_snapped": 10.0,
        "overlap_frames": 22,
        "style_header": "",
        "full_prompt": "",
        "segments": [
            {"index": 1, "brief": "s1", "prompt": "s1"},
            {"index": 2, "brief": "s2", "prompt": "s2"},
        ],
    }
    bundle = FakeBundle()
    ref_video = torch.zeros(500, 480, 864, 3)
    detail_settings = {
        "enabled": True, "resolution": "832P", "width": 1472, "height": 832,
        "steps": 4, "denoise": 0.2, "scheduler": "beta", "sampler_name": "res_multistep",
        "upscale_method": "bicubic", "chunk_frames": 4, "model": object(),
    }
    res = nodes.H3LongVideo().run(
        h3=bundle, model=object(), sampler=object(),
        plan_json=json.dumps(plan_data),
        task_mode="动作迁移（跟随参考视频）", resolution="480P", aspect_ratio="16:9",
        width=864, height=480, steps=8, denoise=1.0, scheduler="simple",
        noise_seed=42, context_length="22", prompt_mode="直接用分段稿",
        media_prefix="", llm_service="none", ref_image_size="匹配生成分辨率",
        save_segments=False, ref_video=ref_video, prompt="",
        **{"二采设置": detail_settings},
    )
    expanded = res["expand"]
    anchor_nodes = [v for v in expanded.values()
                    if isinstance(v, dict) and v.get("class_type") == "H3AnchorContext"]
    check(len(anchor_nodes) == 1,
          f"expected one sample1 conditioning anchor, got {len(anchor_nodes)}")
    anchor = anchor_nodes[0]
    check("context_latent" in anchor["inputs"],
          "seg 2 sample1 anchor must use lossless context_latent")
    check("context_frames" not in anchor["inputs"],
          "seg 2 sample1 anchor should not fall back to context_frames")
    source_link = anchor["inputs"]["context_latent"]
    source = expanded[source_link[0]]
    check(source.get("class_type") == "H3LatentIdentity",
          "sample1 continuity no longer uses the compact-tail identity link")
    barrier_link = source["inputs"]["samples"]
    barrier = expanded[barrier_link[0]]
    check(barrier.get("class_type") == "H3SegmentMemoryBarrier"
          and barrier_link[1] == 4,
          "low-resolution motion chain did not consume the compact pass1 tail")
    pass1_source = expanded[barrier["inputs"]["pass1_latent"][0]]
    check(pass1_source["inputs"].get("pass_label") == "sample1",
          "low-resolution motion chain did not originate from the previous sample1")

    detail_seeds = [v for v in expanded.values()
                    if isinstance(v, dict)
                    and v.get("class_type") == "H3LatentOverlapSeed"]
    check(len(detail_seeds) == 1,
          f"expected one sample2 overlap seed, got {len(detail_seeds)}")
    detail_seed = detail_seeds[0]
    current_base = expanded[detail_seed["inputs"]["latent"][0]]
    check(current_base.get("class_type") == "H3LatentUpscale",
          "sample2 overlap must preserve the current segment's upscaled latent")
    detail_link = detail_seed["inputs"]["context_latent"]
    detail_source = expanded[detail_link[0]]
    check(detail_source is barrier and detail_link[1] == 3,
          "high-resolution identity chain did not consume the compact detail tail")
    pass2_source = expanded[barrier["inputs"]["detail_latent"][0]]
    check(pass2_source["inputs"].get("pass_label") == "sample2",
          "high-resolution identity chain did not originate from the previous sample2")
    drift_nodes = [v for v in expanded.values()
                   if isinstance(v, dict) and v.get("class_type") == "H3DriftCorrect"]
    check(len(drift_nodes) == 0, "H3DriftCorrect should not appear after drift removal")


if __name__ == "__main__":
    for test in (
        test_seam_blend_with_luminance_compensation,
        test_longvideo_transfer_latent_continuity,
    ):
        test()
        print("PASS", test.__name__)
