import importlib
import json
import sys
from pathlib import Path
import torch

TEST_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = TEST_DIR.parent
CUSTOM_NODES_DIR = PACKAGE_DIR.parent
COMFY_DIR = CUSTOM_NODES_DIR.parent

for p in (str(COMFY_DIR), str(CUSTOM_NODES_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import comfy.nested_tensor
pkg = importlib.import_module("ComfyUI-MiniMaxH3-Myang")
detail = importlib.import_module("ComfyUI-MiniMaxH3-Myang.detail")
nodes = importlib.import_module("ComfyUI-MiniMaxH3-Myang.nodes")
latent_upscale_3d = importlib.import_module("ComfyUI-MiniMaxH3-Myang.latent_upscale_3d")


def test_latent_upscaler_node():
    print("Testing H3LatentUpscale node...")
    upscaler = detail.H3LatentUpscale()

    # Create dummy 480P 16:9 latent: 864x480 -> 54x30 latent
    B, C, T, H, W = 1, 16, 25, 30, 54
    vid_lat = torch.randn(B, C, T, H, W)
    aud_lat = torch.randn(B, 8, 2, 40)
    samples_in = {"samples": comfy.nested_tensor.NestedTensor((vid_lat, aud_lat))}

    # Test Latent Bicubic to 832P (1472x832 -> 92x52 latent)
    out = upscaler.upscale(
        samples=samples_in,
        resolution="832P",
        aspect_ratio="16:9",
        width=1664,
        height=928,
        upscale_method="latent_bicubic (Latent极速双采·推荐)"
    )
    samples_out = out[0]
    nested_out = samples_out["samples"]
    tensors = nested_out.unbind()
    v_out, a_out = tensors[0], tensors[1]

    assert v_out.shape == (1, 16, 25, 52, 92), f"Expected shape (1, 16, 25, 52, 92), got {v_out.shape}"
    assert a_out.shape == aud_lat.shape, "Audio latent should be unchanged"

    # Test Custom resolution 1664x928 -> Latent: 104x58
    out_custom = upscaler.upscale(
        samples=samples_in,
        resolution="自定义",
        aspect_ratio="16:9",
        width=1664,
        height=928,
        upscale_method="latent_bicubic (Latent极速双采·推荐)"
    )
    v_custom = out_custom[0]["samples"].unbind()[0]
    assert v_custom.shape == (1, 16, 25, 58, 104), f"Expected shape (1, 16, 25, 58, 104), got {v_custom.shape}"
    print("PASS test_latent_upscaler_node")


def test_detail_settings_builder():
    print("Testing H3DetailSettings builder...")
    settings_node = detail.H3DetailSettings()
    res = settings_node.build(
        enabled=True,
        resolution="832P",
        width=1664,
        height=928,
        steps=4,
        denoise=0.2,
        scheduler="beta",
        sampler_name="res_multistep",
        upscale_method="lanczos (像素高保真超分·推荐·无伪影)",
        chunk_frames=4,
        二采模型="dummy_model"
    )
    cfg = res[0]
    assert cfg["enabled"] is True
    assert cfg["resolution"] == "832P"
    assert cfg["steps"] == 4
    assert cfg["denoise"] == 0.2
    assert "lanczos" in cfg["upscale_method"]
    print("PASS test_detail_settings_builder")


def test_learned_upscaler_cache_can_be_released():
    marker = torch.nn.Linear(1, 1)
    latent_upscale_3d._MODEL_CACHE["test-marker"] = marker
    assert latent_upscale_3d.clear_model_cache() == 1
    assert latent_upscale_3d._MODEL_CACHE == {}


if __name__ == "__main__":
    test_latent_upscaler_node()
    test_detail_settings_builder()
    test_learned_upscaler_cache_can_be_released()
    print("ALL LATENT UPSCALE TESTS PASSED!")
