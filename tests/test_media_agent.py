"""CPU-only ownership and compatibility checks for the built-in Media Agent."""

import importlib
import json
import os
import sys
import tempfile
from pathlib import Path

import torch


PACKAGE_DIR = Path(__file__).resolve().parents[1]
CUSTOM_NODES = PACKAGE_DIR.parent
COMFY_ROOT = CUSTOM_NODES.parent
for path in (str(COMFY_ROOT), str(CUSTOM_NODES)):
    if path not in sys.path:
        sys.path.insert(0, path)

package = importlib.import_module("ComfyUI-MiniMaxH3-Myang")
agent = importlib.import_module("ComfyUI-MiniMaxH3-Myang.agent_nodes")
media_types = importlib.import_module("ComfyUI-MiniMaxH3-Myang.media_catalog")
media_nodes = importlib.import_module("ComfyUI-MiniMaxH3-Myang.media")


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def test_agent_is_owned_and_registered_here():
    cls = package.NODE_CLASS_MAPPINGS.get("MiniMaxH3MediaAgent")
    check(cls is agent.MiniMaxH3MediaAgent, "Media Agent is not registered by Myang")
    check(cls.__module__.startswith("ComfyUI-MiniMaxH3-Myang."),
          "Media Agent still resolves to a foreign package")
    check(package.NODE_DISPLAY_NAME_MAPPINGS["MiniMaxH3MediaAgent"] ==
          "沐阳 H3 · Media Agent", "owned display name changed")
    check(cls.CATEGORY == "沐阳 H3", "Media Agent category is not owned by Myang")


def test_agent_bypass_builds_native_bundle():
    frames = torch.zeros(5, 2, 2, 3)
    links = [{"order": 1, "media_type": "video", "filename": "motion.mp4"}]
    result = agent.MiniMaxH3MediaAgent.plan(
        prompt="参考@视频1的动作",
        llm_service="none",
        skill_preset="none",
        skill_text="",
        agent_enabled=False,
        strict_media_check=True,
        ollama_auto_unload=True,
        seed=0,
        asset_1=frames,
        asset_manifest_json=json.dumps(links, ensure_ascii=False),
        时长=5.0,
    )["result"]
    bundle = result[3]
    check(isinstance(bundle, media_types.MyangMediaCatalog),
          "Agent did not emit its own media bundle")
    check(bundle.assets[0].kind == "video", "video slot was misclassified")
    check(bundle.assets[0].filename == "motion.mp4", "asset metadata was not bound to its payload")
    check("<Video 1>" in result[0], "agent prompt lost its canonical media tag")
    check("__MINIMAX" not in result[0], "agent prompt still contains a private wire placeholder")
    check("@视频1" in result[4], "editor prompt lost its readable media tag")


def test_segment_swap_is_catalog_native():
    original = torch.zeros(5, 2, 2, 3)
    replacement = torch.ones(5, 2, 2, 3)
    bundle = media_types.MyangMediaCatalog(assets=(
        media_types.MyangMediaAsset(1, "image", torch.zeros(1, 2, 2, 3)),
        media_types.MyangMediaAsset(2, "video", original),
    ))
    swapped, = media_nodes.H3MediaSwapClip().swap(bundle, replacement, 1)
    check(isinstance(swapped, media_types.MyangMediaCatalog),
          "swap returned a foreign media bundle")
    check(swapped.assets[1].payload is replacement, "segment clip was not replaced")


def test_catalog_order_and_normalizers_share_one_contract():
    image = torch.zeros(1, 4, 6, 3)
    video = torch.zeros(5, 4, 6, 3)
    waveform = torch.zeros(2, 160)
    catalog = media_types.MyangMediaCatalog(assets=(
        media_types.MyangMediaAsset(8, "audio", (waveform, 16000), label="对白"),
        media_types.MyangMediaAsset(3, "video", {"frames": video, "fps": 25}),
        media_types.MyangMediaAsset(7, "image", image),
    ))
    check([asset.kind for asset in catalog.ordered()] == ["image", "video", "audio"],
          "catalog ordering drifted from H3 reference numbering")
    frames, soundtrack, fps = media_types.video_stream({
        "images": video, "audio": {"waveform": waveform, "sample_rate": 16000}, "fps": 25,
    })
    check(frames is video and fps == 25 and soundtrack["waveform"].shape == (1, 2, 160),
          "video normalization lost frames, fps or embedded audio")
    check(media_types.audio_track((waveform, 16000))["waveform"].shape == (1, 2, 160),
          "tuple audio normalization did not produce ComfyUI AUDIO shape")
    check(media_types.classify_payload((waveform, 16000)) == "audio",
          "tuple audio was misclassified as image/video")


def test_skill_lookup_rejects_path_traversal():
    original_dir = agent.SKILL_DIR
    original_extra = agent.EXTRA_SKILLS_DIR
    with tempfile.TemporaryDirectory(dir=PACKAGE_DIR) as temp:
        root = Path(temp)
        skills = root / "skills"
        skills.mkdir()
        (skills / "safe.md").write_text("safe", encoding="utf-8")
        (root / "secret.md").write_text("secret", encoding="utf-8")
        agent.SKILL_DIR = skills
        agent.EXTRA_SKILLS_DIR = None
        try:
            check(agent._get_skill_target("safe.md") == (skills / "safe.md").resolve(),
                  "valid Skill leaf was rejected")
            for unsafe in ("../secret.md", "..\\secret.md", str(root / "secret.md")):
                check(agent._get_skill_target(unsafe) is None,
                      "Skill lookup accepted a path outside its library")
        finally:
            agent.SKILL_DIR = original_dir
            agent.EXTRA_SKILLS_DIR = original_extra


def test_directory_import_is_confined_to_opt_in_root():
    previous = os.environ.get("MINIMAX_H3_SKILLS_IMPORT_DIR")
    with tempfile.TemporaryDirectory(dir=PACKAGE_DIR) as temp:
        root = Path(temp)
        allowed = root / "allowed"
        inside = allowed / "incoming"
        outside = root / "outside"
        inside.mkdir(parents=True)
        outside.mkdir()
        os.environ["MINIMAX_H3_SKILLS_IMPORT_DIR"] = str(allowed)
        try:
            check(agent._resolve_skill_import_source("incoming") == inside.resolve(),
                  "relative import inside the allowlist was rejected")
            try:
                agent._resolve_skill_import_source(str(outside))
            except ValueError:
                pass
            else:
                raise AssertionError("directory import escaped its configured root")
        finally:
            if previous is None:
                os.environ.pop("MINIMAX_H3_SKILLS_IMPORT_DIR", None)
            else:
                os.environ["MINIMAX_H3_SKILLS_IMPORT_DIR"] = previous


if __name__ == "__main__":
    for test in (
        test_agent_is_owned_and_registered_here,
        test_agent_bypass_builds_native_bundle,
        test_segment_swap_is_catalog_native,
        test_catalog_order_and_normalizers_share_one_contract,
        test_skill_lookup_rejects_path_traversal,
        test_directory_import_is_confined_to_opt_in_root,
    ):
        test()
        print("PASS", test.__name__)
