"""CPU-only ownership and compatibility checks for the built-in Media Agent."""

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
agent = importlib.import_module("ComfyUI-MiniMaxH3-Myang.agent_nodes")
media_types = importlib.import_module("ComfyUI-MiniMaxH3-Myang.agent_media")
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
    check(cls.CATEGORY == "沐阳 H3", "Media Agent still appears under Easy")
    backend_source = (PACKAGE_DIR / "agent_nodes.py").read_text("utf-8")
    check("ComfyUI-MiniMaxH3-Easy" not in backend_source,
          "Agent still reads another node package at runtime")


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
        media_1=frames,
        media_links_json=json.dumps(links, ensure_ascii=False),
        时长=5.0,
    )["result"]
    bundle = result[3]
    check(isinstance(bundle, media_types.MiniMaxH3MediaBundle),
          "Agent did not emit its own media bundle")
    check(bundle.items[0].media_type == "video", "video slot was misclassified")
    # Bypass mode intentionally preserves readable editor syntax; the Myang
    # conditioning boundary translates it to <Video 1> when the model runs.
    check("@视频1" in result[0], "agent prompt lost its readable media tag")
    check("@视频1" in result[4], "editor prompt lost its readable media tag")


def test_agent_native_asset_protocol_and_legacy_compatibility():
    frames = torch.zeros(5, 2, 2, 3)
    metadata = [{
        "slot": 1,
        "order": 1,
        "media_type": "video",
        "filename": "native-motion.mp4",
        "label": "动作参考",
        "source_id": 42,
    }]
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
        asset_manifest_json=json.dumps(metadata, ensure_ascii=False),
        时长=5.0,
    )["result"]
    bundle = result[3]
    check(bundle.items[0].media_type == "video", "native asset type was lost")
    check(bundle.items[0].filename == "native-motion.mp4", "native metadata was lost")
    check("@视频1" in result[0], "native wire prompt is not readable")
    check("__MINIMAX_H3_REF_" not in result[0], "private runtime placeholder leaked")
    optional = agent.MiniMaxH3MediaAgent.INPUT_TYPES()["optional"]
    check("asset_1" in optional and "media_1" in optional,
          "native or legacy protocol is no longer accepted")


def test_segment_swap_needs_no_easy_package():
    original = torch.zeros(5, 2, 2, 3)
    replacement = torch.ones(5, 2, 2, 3)
    bundle = media_types.MiniMaxH3MediaBundle(items=(
        media_types._MediaInput(1, "image", torch.zeros(1, 2, 2, 3)),
        media_types._MediaInput(2, "video", original),
    ), links=())
    swapped, = media_nodes.H3MediaSwapClip().swap(bundle, replacement, 1)
    check(isinstance(swapped, media_types.MiniMaxH3MediaBundle),
          "swap returned a foreign media bundle")
    check(swapped.items[1].value is replacement, "segment clip was not replaced")


def test_agent_frontend_uses_native_catalog_transport():
    source = (PACKAGE_DIR / "web" / "minimax_h3_myang_agent_ui.js").read_text("utf-8")
    check("promptNode.inputs.asset_manifest_json" in source,
          "frontend does not serialize native asset metadata")
    check('promptNode.inputs[`asset_${index + 1}`]' in source,
          "frontend does not serialize native asset inputs")
    check("EASY_CLASS" not in source and "EASY_LINKS_PROP" not in source,
          "frontend still contains Easy node mirroring")
    check("myang_h3_asset_sources_v2" in source
          and "minimax_h3_agent_media_connections" in source,
          "saved frontend media links cannot migrate to the native property")


if __name__ == "__main__":
    for test in (
        test_agent_is_owned_and_registered_here,
        test_agent_bypass_builds_native_bundle,
        test_agent_native_asset_protocol_and_legacy_compatibility,
        test_segment_swap_needs_no_easy_package,
        test_agent_frontend_uses_native_catalog_transport,
    ):
        test()
        print("PASS", test.__name__)
