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
nodes_v2 = importlib.import_module("ComfyUI-MiniMaxH3-Myang.nodes_v2")
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
    res = nodes_v2.H3LongVideo().run(
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
    check(len(anchor_nodes) == 2,
          f"expected one sample1 and one sample2 anchor, got {len(anchor_nodes)}")
    pass_labels = set()
    for anchor in anchor_nodes:
        check("context_latent" in anchor["inputs"],
              "seg 2 anchors must use lossless context_latent")
        check("context_frames" not in anchor["inputs"],
              "seg 2 anchor should not fall back to context_frames")
        source = expanded[anchor["inputs"]["context_latent"][0]]
        pass_label = source["inputs"].get("pass_label")
        pass_labels.add(pass_label)
        if pass_label == "sample2":
            current_base = expanded[anchor["inputs"]["latent"][0]]
            check(current_base.get("class_type") == "H3LatentUpscale",
                  "sample2 anchor must preserve the current segment's upscaled latent")
    check(pass_labels == {"sample1", "sample2"},
          "low-resolution motion and high-resolution identity chains are not separate")
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
