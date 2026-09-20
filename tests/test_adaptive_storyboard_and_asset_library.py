"""Regression tests for adaptive timing, subject state and reusable assets."""

import importlib
import sys
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parents[1]
CUSTOM_NODES = PACKAGE_DIR.parent
COMFY_ROOT = CUSTOM_NODES.parent
for value in (str(COMFY_ROOT), str(CUSTOM_NODES)):
    if value not in sys.path:
        sys.path.insert(0, value)

nodes = importlib.import_module("ComfyUI-MiniMaxH3-Myang.nodes")
agent_nodes = importlib.import_module("ComfyUI-MiniMaxH3-Myang.agent_nodes")
library = importlib.import_module("ComfyUI-MiniMaxH3-Myang.asset_library")


def test_adaptive_durations_respect_cap_and_total():
    storyboard = [
        {"index": 1, "transition": "开场", "duration_seconds": 4.5},
        {"index": 2, "transition": "承接", "duration_seconds": 7.5},
        {"index": 3, "transition": "切镜", "duration_seconds": 5.0},
    ]
    result = nodes._normalize_storyboard_durations(
        storyboard, total_seconds=17.0, maximum_seconds=8.0,
        overlap_frames=22, fps=24.0)
    assert len({item["frames"] for item in result}) > 1
    assert all(item["frames"] % 17 == 5 for item in result)
    assert all(item["duration_seconds"] <= 8.0 for item in result)
    visible = sum(item["frames"] for item in result) - 22
    assert abs(visible / 24.0 - 17.0) <= 17 / 48.0


def test_subject_ledger_is_segment_scoped_and_revisions_changes():
    manifest = (
        "- <Picture 1> 或 @图片1：静态图像，主体名：桃乐丝\n"
        "  画面内容：红发、红眼与黑色贝雷帽。\n"
        "- <Picture 2> 或 @图片2：静态图像，主体名：紫苑\n"
        "  画面内容：紫发、蓝色外套。")
    chunks = ["桃乐丝走进雨中。", "桃乐丝换上白色礼服。", "紫苑在室内读书。"]
    storyboard = [
        {"subjects": [{"name": "桃乐丝", "media": "@图片1",
                        "state_action": "首次", "declaration": "穿红色外套"}]},
        {"subjects": [{"name": "桃乐丝", "media": "@图片1",
                        "state_action": "变化", "declaration": "换成白色礼服"}]},
        {"subjects": [{"name": "紫苑", "media": "@图片2",
                        "state_action": "首次", "declaration": "穿蓝色外套"}]},
    ]
    result = nodes._apply_subject_ledger(chunks, storyboard, manifest, "")
    assert [subject["name"] for subject in result[0]["subjects"]] == ["桃乐丝"]
    assert result[1]["subjects"][0]["state_revision"] == 2
    assert [subject["name"] for subject in result[2]["subjects"]] == ["紫苑"]


def test_subject_ledger_recovers_pronoun_continuation_but_not_empty_cut():
    manifest = (
        "- <Picture 1> 或 @图片1：静态图像，主体名：桃乐丝角色立绘\n"
        "  画面内容：人物角色，红发、红眼与黑色贝雷帽。")
    chunks = [
        "桃乐丝参考@图片1走进房间。",
        "她换上白色礼服，继续向窗边走去。",
        "切到无人街道空镜。",
    ]
    storyboard = [
        {"transition": "开场", "subjects": [{
            "name": "桃乐丝", "media": "@图片1", "state_action": "首次",
            "declaration": "穿红色外套"}]},
        {"transition": "承接", "subjects": []},
        {"transition": "切镜", "subjects": []},
    ]
    result = nodes._apply_subject_ledger(chunks, storyboard, manifest, "")
    assert result[1]["subjects"][0]["name"] == "桃乐丝"
    assert result[1]["subjects"][0]["state_action"] == "变化"
    assert result[1]["subjects"][0]["state_revision"] == 2
    assert "白色礼服" in result[1]["subjects"][0]["declaration"]
    assert result[2]["subjects"] == []


def test_storyboard_title_is_card_metadata_not_a_truncated_script_header():
    source = (
        "================================================================================\n"
        "《异世界邻居》日常篇 01：台风天的停课通知与客厅城堡（上篇·75秒）\n"
        "【画面基准设定】角色与场景保持统一。\n"
        "桃乐丝推开窗户，接住被风吹来的纸飞机。")
    title = nodes._sanitize_storyboard_title(
        "================================================================================", source)
    assert title == "桃乐丝推开窗户"
    assert 2 <= len(title) <= 8
    assert "异世界" not in title and "=" not in title
    assert nodes._sanitize_storyboard_title("【雨夜相遇】", source) == "雨夜相遇"



def synthetic_skills(function):
    """Exercise discovery with redistributable fixtures, never personal skills."""
    from functools import wraps
    @wraps(function)
    def run():
        import tempfile
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            names = [
                "h3-prompt-writing", "minimax-h3-reference-video-prompt",
                "minimax-h3-text-video-prompt", "minimax-h3-prompt-reviewer",
                "minimax-h3-creative-director", "paper-collage-explainer-generator"]
            for name in names:
                (root / name).mkdir()
                (root / name / "SKILL.md").write_text(
                    "# " + name + "\nKeep subject and dialogue consistent.", encoding="utf-8")
            (root / "anime-pv-maker.md").write_text(
                "# Anime PV\nUse a clear opening shot.", encoding="utf-8")
            with patch.object(agent_nodes, "SKILL_DIR", root), patch.object(
                    agent_nodes, "EXTRA_SKILLS_DIR", None):
                return function()
    return run

@synthetic_skills
def test_per_segment_skill_plan_allows_bundles_but_keeps_one_structure_owner():
    names = agent_nodes._skill_names()
    raw = """
SEGMENT 1 | PRIMARY: h3-prompt-writing | OVERLAYS: anime-pv-maker.md | REVIEWERS: minimax-h3-prompt-reviewer | REASON: 动漫PV开场
SEGMENT 2 | PRIMARY: minimax-h3-reference-video-prompt | OVERLAYS: paper-collage-explainer-generator | REVIEWERS: none | REASON: 参考视频与纸艺场景
SEGMENT 3 | PRIMARY: minimax-h3-creative-director | OVERLAYS: none | REVIEWERS: none | REASON: 错误地把规划技能当写作技能
"""
    plan = agent_nodes._parse_skill_plan(raw, 3, names)
    assert len(plan) == 3
    assert plan[0]["primary"] == "h3-prompt-writing"
    assert plan[0]["overlays"] == ["anime-pv-maker.md"]
    assert plan[0]["reviewers"] == ["minimax-h3-prompt-reviewer"]
    assert plan[1]["primary"] == "minimax-h3-reference-video-prompt"
    assert plan[0]["primary"] != plan[1]["primary"]
    assert plan[2]["primary"] not in agent_nodes._PLANNING_ONLY_SKILLS
    demoted = agent_nodes._normalize_skill_plan_item(
        {"primary": "anime-pv-maker.md", "overlays": "none"}, 4, names)
    assert demoted["primary"] == "h3-prompt-writing"
    assert demoted["overlays"] == ["anime-pv-maker.md"]

    rules, source, used = agent_nodes.resolve_skill_bundle(
        plan[0]["primary"], plan[0]["overlays"], plan[0]["reviewers"],
        skill_text="用户要求：保留原台词。", budget=8000)
    assert len(rules) <= 8000
    assert rules.count("【PRIMARY 主技能：") == 1
    assert "【OVERLAY 辅助技能：" in rules
    assert "【REVIEWER 自检技能：" in rules
    assert source.startswith("用户规则；主=")
    assert used[0] == "h3-prompt-writing"


@synthetic_skills
def test_auto_skill_plan_routes_all_segments_in_one_advisory_call():
    calls = []
    original_expand = agent_nodes._expand_with_prompt_assistant
    agent_nodes._auto_skill_plan_cache.clear()

    def fake_expand(service, prompt, system, unload, seed, max_tokens=None):
        calls.append((service, prompt, system))
        return (
            "SEGMENT 1 | PRIMARY: h3-prompt-writing | OVERLAYS: anime-pv-maker.md | "
            "REVIEWERS: none | REASON: 动漫开场\n"
            "SEGMENT 2 | PRIMARY: minimax-h3-text-video-prompt | OVERLAYS: none | "
            "REVIEWERS: minimax-h3-prompt-reviewer | REASON: 纯文本空镜")

    agent_nodes._expand_with_prompt_assistant = fake_expand
    try:
        plan, source = agent_nodes.select_skill_plan_auto("stub", [
            {"index": 1, "brief": "角色跃入画面", "duration_seconds": 5,
             "selected_media_tags": ["@图片1"]},
            {"index": 2, "brief": "雨夜街道空镜", "duration_seconds": 7},
        ])
    finally:
        agent_nodes._expand_with_prompt_assistant = original_expand
        agent_nodes._auto_skill_plan_cache.clear()
    assert len(calls) == 1
    assert len(plan) == 2 and plan[0]["overlays"] == ["anime-pv-maker.md"]
    assert plan[1]["primary"] == "minimax-h3-text-video-prompt"
    assert "自动" in source


def test_auto_skill_bundles_reach_their_own_segment_writers():
    writer_systems = []
    original_call = nodes.call_llm
    original_select = agent_nodes.select_skill_plan_auto
    original_bundle = agent_nodes.resolve_skill_bundle

    def fake_call(service, user, system, unload, seed, max_tokens=None):
        if "长视频分镜规划师" in system:
            return (
                "[SEGMENT 1]\nTITLE: 雨夜相遇\nDURATION: 4.5\nTRANSITION: 开场\n"
                "GOAL: 少女推门进入\nSHOT: 0-4.5 || 门口中景 || 推门 || 缓推 || 雨声 || 无\n"
                "[SEGMENT 2]\nTITLE: 炉边交谈\nDURATION: 4.5\nTRANSITION: 承接\n"
                "GOAL: 两人在炉边交谈\nSHOT: 0-4.5 || 炉边双人景 || 交谈 || 固定 || 炉火声 || 无")
        writer_systems.append(system)
        return "integrated_multimodal_description: 本段动作完整展开。\noverall_soundscape: 环境声。\nnon_diegetic_music: 轻音乐。"

    def fake_select(service, briefs, unload=False):
        return ([
            {"index": 1, "primary": "h3-prompt-writing", "overlays": ["anime-pv-maker.md"],
             "reviewers": [], "reason": "动漫开场"},
            {"index": 2, "primary": "minimax-h3-text-video-prompt", "overlays": [],
             "reviewers": ["minimax-h3-prompt-reviewer"], "reason": "对白场景"},
        ], "逐镜头技能组合（测试）")

    def fake_bundle(primary, overlays=None, reviewers=None, skill_text="", budget=8000):
        marker = "RULE-A" if primary == "h3-prompt-writing" else "RULE-B"
        names = [primary] + list(overlays or []) + list(reviewers or [])
        return marker, " + ".join(names), names

    nodes.call_llm = fake_call
    agent_nodes.select_skill_plan_auto = fake_select
    agent_nodes.resolve_skill_bundle = fake_bundle
    try:
        payload, _notes = nodes._write_segments_with_media_agent(
            "少女在雨夜进入木屋。随后两人在炉边交谈。",
            ["少女在雨夜进入木屋。", "随后两人在炉边交谈。"],
            2, 5.0, 9.0, 22, 24.0, "", "auto-marker", "stub", False, 1,
            skill_preset="auto", skill_text="")
    finally:
        nodes.call_llm = original_call
        agent_nodes.select_skill_plan_auto = original_select
        agent_nodes.resolve_skill_bundle = original_bundle
    assert len(payload["segments"]) == 2
    assert len(writer_systems) == 2
    assert "RULE-A" in writer_systems[0] and "RULE-B" in writer_systems[1]
    assert payload["segments"][0]["skills"] != payload["segments"][1]["skills"]
    assert payload["skill_plan"][0]["primary"] == "h3-prompt-writing"
    assert payload["segments"][0]["title"] == "雨夜相遇"


def test_asset_catalogue_add_rename_and_remove():
    # The managed test shell is read-only even though the app itself can write
    # its user config. Exercise CRUD with an in-memory persistence boundary.
    store = []
    original_load, original_save = library.load_catalogue, library._save
    library.load_catalogue = lambda path=None: [dict(item) for item in store]
    def save_memory(assets, path=None):
        store[:] = [dict(item) for item in assets]
    library._save = save_memory
    try:
        created = library.add_asset({
            "name": "桃乐丝", "category": "character", "kind": "image",
            "subject_name": "桃乐丝", "identity": "红发红眼",
            "file": {"name": "dorothy.png", "subfolder": "Myang", "type": "input"},
        })
        assert library.load_catalogue()[0]["name"] == "桃乐丝"
        updated = library.update_asset(created["id"], {
            "name": "桃乐丝·礼服", "category": "character",
        })
        assert updated["name"] == "桃乐丝·礼服"
        assert library.remove_asset(created["id"])
        assert library.load_catalogue() == []
    finally:
        library.load_catalogue, library._save = original_load, original_save


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
        print("PASS", test.__name__)
