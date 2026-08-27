"""Unit tests for the new Agent -> H3ScriptSplitter -> H3LongVideo pipeline."""

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
agent_module = importlib.import_module("ComfyUI-MiniMaxH3-Myang.agent_nodes")
nodes_legacy = importlib.import_module("ComfyUI-MiniMaxH3-Myang.nodes")
nodes_v2 = importlib.import_module("ComfyUI-MiniMaxH3-Myang.nodes_v2")
media_types = importlib.import_module("ComfyUI-MiniMaxH3-Myang.media_catalog")


def check(condition, message):
    if not condition:
        raise AssertionError(message)


class FakeBundle:
    def __init__(self):
        self.video_vae = object()
        self.audio_vae = object()
        self.names = {"model": "minimax_h3_ref2va"}

    def model_for(self, kind):
        return object()


def test_agent_myang_prompt_generation():
    """Verify that MiniMaxH3MediaAgent outputs myang_prompt in slot 4."""
    frames = torch.zeros(5, 2, 2, 3)
    links = [{"order": 1, "media_type": "video", "filename": "action.mp4"}]
    prompt_text = "参考@视频1的动作，镜头平推靠近，少女微笑着挥手。"
    res = agent_module.MiniMaxH3MediaAgent.plan(
        prompt=prompt_text,
        llm_service="none",
        skill_preset="none",
        skill_text="",
        agent_enabled=False,
        strict_media_check=True,
        ollama_auto_unload=True,
        seed=0,
        asset_1=frames,
        asset_manifest_json=json.dumps(links, ensure_ascii=False),
        时长=10.0,
    )
    result = res["result"]
    myang_prompt = result[4]
    check(isinstance(myang_prompt, str), "myang_prompt must be a string")
    check("@视频1" in myang_prompt, "@视频1 tag should be preserved in myang_prompt")


def test_script_splitter_empty_script():
    """Verify that H3ScriptSplitter handles empty script cleanly without LLM."""
    splitter = nodes_v2.H3ScriptSplitter()
    out = splitter.split(
        script="",
        total_seconds=20.0,
        length_source="用填写的总时长",
        segment_seconds=10.0,
        overlap_frames=22,
        fps=24.0,
        llm_service="none",
        max_segments=12,
        ollama_auto_unload=True,
        use_cache=False,
        seed=0,
        llm_enabled=True,
    )
    plan_json, seg_count, seg_sec, frames, preview, ref_needed = out
    plan = json.loads(plan_json)
    check(plan["segment_count"] == 2, f"expected 2 segments, got {plan['segment_count']}")
    check(len(plan["segments"]) == 2, "segments array length mismatch")
    check("prompt" in plan["segments"][0], "segment must contain prompt field")


def test_script_splitter_manual_llm_toggle():
    """Verify that H3ScriptSplitter allows manual toggle (llm_enabled=False) with non-empty script."""
    splitter = nodes_v2.H3ScriptSplitter()
    script_content = "参考@视频1中的角色动作，保持环境风格不变。"
    out = splitter.split(
        script=script_content,
        total_seconds=20.0,
        length_source="用填写的总时长",
        segment_seconds=10.0,
        overlap_frames=22,
        fps=24.0,
        llm_service="none",  # Would fail if LLM was invoked
        max_segments=12,
        ollama_auto_unload=True,
        use_cache=False,
        seed=0,
        llm_enabled=False,  # Manually disabled!
    )
    plan_json, seg_count, seg_sec, frames, preview, ref_needed = out
    plan = json.loads(plan_json)
    check(plan["segment_count"] == 2, f"expected 2 segments, got {plan['segment_count']}")
    check(len(plan["segments"]) == 2, "segments array length mismatch")
    check(plan["segments"][0]["prompt"] == script_content, "segment 1 prompt should match script")
    check(plan["segments"][1]["prompt"] == script_content, "segment 2 prompt should match script")
    check("【LLM 切片已手动关闭】" in preview, "preview should indicate LLM manually disabled")


def test_script_splitter_plan_json_contains_prompts():
    """Verify that H3ScriptSplitter structure packages prompts and handles cached or mock payloads."""
    plan_data = {
        "segment_count": 2,
        "frames_per_segment": 245,
        "segment_seconds_snapped": 10.208,
        "overlap_frames": 22,
        "style_header": "4K电影质感，写实光影，胶片颗粒。",
        "full_prompt": "完整的两段式剧本，包含第1段打斗与第2段撤退。",
        "segments": [
            {
                "index": 1,
                "brief": "第1段：两人在雨夜小巷中交手，刀光闪烁。",
                "prompt": "4K电影质感，写实光影。两人在雨夜小巷中交手，动作迅猛，刀光闪烁，镜头快速推拉。<d>纳命来！</d>",
            },
            {
                "index": 2,
                "brief": "第2段：黑衣人掷出烟雾弹后翻身上墙撤退。",
                "prompt": "4K电影质感，写实光影。黑衣人落地翻滚，掷出烟雾弹，白色浓烟升腾，翻身上墙迅速撤离。",
            },
        ],
    }
    plan_json_str = json.dumps(plan_data, ensure_ascii=False)

    # Test H3SegmentPrompt retrieval
    seg_prompt_node = nodes_legacy.H3SegmentPrompt()
    p1, b1 = seg_prompt_node.build(
        plan_json=plan_json_str,
        segment_index=1,
        mode=nodes_legacy.MODE_DIRECT,
        media_prefix="",
        llm_service="none",
        carry_prev_tail=False,
        ollama_auto_unload=True,
        seed=0,
    )
    check("雨夜小巷" in p1, "H3SegmentPrompt failed to extract segment 1 prompt")

    p2, b2 = seg_prompt_node.build(
        plan_json=plan_json_str,
        segment_index=2,
        mode=nodes_legacy.MODE_DIRECT,
        media_prefix="",
        llm_service="none",
        carry_prev_tail=False,
        ollama_auto_unload=True,
        seed=0,
    )
    check("黑衣人" in p2, "H3SegmentPrompt failed to extract segment 2 prompt")


def test_longvideo_consumes_plan_json_prompts_without_separate_prompt():
    """Verify that H3LongVideo consumes plan_json prompts directly when prompt is empty."""
    plan_data = {
        "segment_count": 2,
        "frames_per_segment": 243,  # length_for(10.0, 24.0) = 243
        "segment_seconds_snapped": 10.0,
        "overlap_frames": 22,
        "style_header": "全局电影质感风格",
        "full_prompt": "完整提示词",
        "segments": [
            {
                "index": 1,
                "brief": "第1段动作",
                "prompt": "第1段提示词内容：主角登场",
            },
            {
                "index": 2,
                "brief": "第2段动作",
                "prompt": "第2段提示词内容：主角跳跃",
            },
        ],
    }
    plan_json_str = json.dumps(plan_data, ensure_ascii=False)

    long_video = nodes_v2.H3LongVideo()
    bundle = FakeBundle()
    ref_video = torch.zeros(500, 480, 864, 3)

    # Calling run with prompt="" (no external prompt connected)
    res = long_video.run(
        h3=bundle,
        model=object(),
        sampler=object(),
        plan_json=plan_json_str,
        task_mode=nodes_legacy.TASK_TRANSFER,
        resolution="480P",
        aspect_ratio="16:9",
        width=864,
        height=480,
        steps=8,
        denoise=1.0,
        scheduler="simple",
        noise_seed=0,
        context_length="22",
        prompt_mode=nodes_legacy.MODE_DIRECT,
        media_prefix="",
        llm_service="none",
        drift_method="off",
        drift_strength=0.6,
        ref_image_size="匹配生成分辨率",
        save_segments=False,
        ref_video=ref_video,
        prompt="",  # Not linked / empty
    )

    expanded_nodes = res["expand"]
    conditions = [v for v in expanded_nodes.values() if isinstance(v, dict) and v.get("class_type") == "H3Condition"]
    check(len(conditions) == 2, f"expected 2 H3Condition nodes, got {len(conditions)}")

    # Verify that H3Condition inputs received the pre-sliced prompt strings!
    cond1_prompt = conditions[0]["inputs"]["prompt"]
    cond2_prompt = conditions[1]["inputs"]["prompt"]
    check(cond1_prompt == "第1段提示词内容：主角登场", f"cond1 prompt mismatch: {cond1_prompt}")
    check(cond2_prompt == "第2段提示词内容：主角跳跃", f"cond2 prompt mismatch: {cond2_prompt}")


if __name__ == "__main__":
    for test in (
        test_agent_myang_prompt_generation,
        test_script_splitter_empty_script,
        test_script_splitter_manual_llm_toggle,
        test_script_splitter_plan_json_contains_prompts,
        test_longvideo_consumes_plan_json_prompts_without_separate_prompt,
    ):
        test()
        print("PASS", test.__name__)
