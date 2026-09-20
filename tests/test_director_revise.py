"""Storyboard revision helpers: AI layering, material rebinding, shot rewrite."""

import importlib
import sys
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parents[1]
CUSTOM_NODES = PACKAGE_DIR.parent
COMFY_ROOT = CUSTOM_NODES.parent
for path in (str(COMFY_ROOT), str(CUSTOM_NODES)):
    if path not in sys.path:
        sys.path.insert(0, path)

revise = importlib.import_module("ComfyUI-MiniMaxH3-Myang.director_revise")
nodes = importlib.import_module("ComfyUI-MiniMaxH3-Myang.nodes")
audit = importlib.import_module("ComfyUI-MiniMaxH3-Myang.dialogue_audit")

MATERIALS = [
    {"kind": "audio", "ordinal": 1, "label": "阿岚配音",
     "name": "lan_voice.wav", "subject": "阿岚"},
    {"kind": "image", "ordinal": 1, "label": "主角设定", "name": "hero.png"},
]
PROSE = "码头黄昏，阿岚站在集装箱前抬头看货轮。@图片1 阿岚<d>走吧。</d>"


def check(condition, message):
    if not condition:
        raise AssertionError(message)


class FakeLLM:
    """Answer each layer/revision call from a script keyed by a prompt marker."""

    def __init__(self, answers):
        self.answers = answers
        self.calls = []

    def __call__(self, service, system_prompt, user_prompt, seed):
        self.calls.append((system_prompt, user_prompt, seed))
        for marker, answer in self.answers.items():
            if marker in system_prompt:
                if isinstance(answer, Exception):
                    raise answer
                return answer
        return ""


def with_llm(fake, worker):
    original = revise._call
    revise._call = fake
    try:
        return worker()
    finally:
        revise._call = original


def test_layer_shot_splits_prose_without_touching_it():
    fake = FakeLLM({
        "主体": '{"subjects":[{"name":"阿岚","appearance":"黑色风衣","ref_tag":"@图片1"}],'
                '"timeline":[{"beat":"0-4s","camera":"中景推近","action":"抬头看货轮"}]}',
        "声音层": '{"ambient":"浪声","bgm":"","sfx":["汽笛","脚步声","风声","雨声","多余"]}',
        "台词层": '{"dialogue":[{"speaker":"阿岚","tone":"calm","text":"走吧。"}]}',
    })
    result = with_llm(fake, lambda: revise.layer_shot(
        PROSE, 8.0, "fake-service", seed=7, materials=MATERIALS))
    layers = result["layers"]
    check(layers["visual"] == PROSE,
          "layering altered the prose it was only supposed to split")
    check(layers["subjects_override"][0]["ref_tag"] == "@图片1",
          "the subject layer lost the material tag from the prose")
    check(layers["timeline"][0]["beat"] == "0-4s", "the timeline layer is missing")
    check(len(layers["sound"]["sfx"]) == 4,
          "the sound layer must cap sfx at four entries")
    check(layers["dialogue"][0]["text"] == "走吧。", "the dialogue layer is missing")
    check(result["budget_units"] == audit.budget_units(8.0),
          "the layering budget disagrees with dialogue_audit")
    check(not result["notes"], "a clean split should report no notes")
    check(len(fake.calls) == 3,
          "layering must stay three small calls, not one large one")
    for _system, user_prompt, _seed in fake.calls:
        check("@音频1" in user_prompt and "lan_voice.wav" in user_prompt,
              "the layer calls cannot see which materials exist")
    check(len({seed for _s, _u, seed in fake.calls}) == 3,
          "the three layer calls reuse one seed and can repeat a failure")


def test_layer_shot_survives_a_failed_or_silent_layer():
    fake = FakeLLM({
        "主体": RuntimeError("provider exploded"),
        "声音层": "not json at all",
        "台词层": '{"dialogue":[{"speaker":"阿岚","tone":"calm","text":"走吧。"}]}',
    })
    result = with_llm(fake, lambda: revise.layer_shot(
        PROSE, 8.0, "fake-service", materials=MATERIALS))
    layers = result["layers"]
    check(layers["visual"] == PROSE, "a failed layer cost the prose")
    check("subjects_override" not in layers and "sound" not in layers,
          "a failed layer was filled with invented content")
    check(layers["dialogue"], "one failed layer took the working layers with it")
    check(len(result["notes"]) == 2,
          "layer failures were not reported to the operator")

    try:
        revise.layer_shot("   ", 8.0, "fake-service")
    except ValueError as error:
        check("没有提示词" in str(error), "empty prose gave an unhelpful error")
    else:
        raise AssertionError("layering an empty shot should refuse")


def test_rebind_only_returns_bindings_and_drops_invented_targets():
    fake = FakeLLM({"绑定": (
        '{"bindings":['
        '{"kind":"audio","ordinal":1,"shots":[1,3,9],"subject":"阿岚","reason":"她的声音"},'
        '{"kind":"audio","ordinal":7,"shots":[1]},'
        '{"kind":"image","ordinal":1,"shots":[]},'
        '{"kind":"video","ordinal":1,"shots":[2]}],'
        '"unbound":[{"kind":"image","ordinal":1,"reason":"没有对应主体"}],'
        '"prompt":"我偷偷改了剧本"}')})
    shots = [
        {"index": 1, "brief": "码头对峙", "speakers": ["阿岚"]},
        {"index": 2, "brief": "快艇追逐", "speakers": []},
        {"index": 3, "brief": "船舱对话", "speakers": ["阿岚", "老陈"]},
    ]
    result = with_llm(fake, lambda: revise.rebind_materials(
        shots, MATERIALS, "fake-service"))
    check(set(result) == {"bindings", "unbound"},
          "the no-story-change mode returned something other than a binding")
    check(len(result["bindings"]) == 1,
          "bindings for unknown materials or empty targets were kept")
    binding = result["bindings"][0]
    check(binding["shots"] == [1, 3],
          "a binding onto a shot that does not exist was accepted")
    check(result["unbound"][0]["reason"] == "没有对应主体",
          "unbound materials lost their explanation")

    _system, user_prompt, _seed = fake.calls[0]
    check("说话人：阿岚、老陈" in user_prompt,
          "the model cannot see who speaks in which shot")
    check(PROSE not in user_prompt,
          "the no-story-change mode sent prose the model could rewrite")

    try:
        revise.rebind_materials(shots, [], "fake-service")
    except ValueError as error:
        check("没有可绑定的素材" in str(error), "empty material list gave a poor error")
    else:
        raise AssertionError("rebinding with no materials should refuse")


def test_rewrite_shot_is_bounded_by_duration_and_manifest():
    good = FakeLLM({"改写": '{"prompt":"码头夜色，阿岚转身离开。@图片1 阿岚<d>该走了。</d>",'
                            '"brief":"转身离开"}'})
    result = with_llm(good, lambda: revise.rewrite_shot(
        PROSE, 8.0, "把黄昏改成夜晚", "fake-service", materials=MATERIALS,
        previous_brief="上一段：抵达码头", transition="承接"))
    check(result["brief"] == "转身离开", "the rewrite lost its new title")
    # A short line leaves the shot quiet, which is advisory. Only an unspeakably
    # long line is a correctness problem, because that is the one the run-time
    # budget defers to the next segment.
    check(result["dialogue_ok"] is True,
          "a line that merely leaves silence was reported as a blocker")
    check("只需" in result["dialogue_report"],
          "the advisory about a quiet shot was dropped from the report")
    _system, user_prompt, _seed = good.calls[0]
    check("8.00 秒" in _system and str(audit.budget_units(8.0)) in _system,
          "the rewrite was not told the duration it has to fit")
    check("承接" in user_prompt and "抵达码头" in user_prompt,
          "the rewrite cannot see the seam it has to keep")

    invented = FakeLLM({"改写": '{"prompt":"阿岚看向 @图片4 的方向。"}'})
    try:
        with_llm(invented, lambda: revise.rewrite_shot(
            PROSE, 8.0, "随便改", "fake-service", materials=MATERIALS))
    except ValueError as error:
        check("@图片4" in str(error),
              "a reference to a material that does not exist was accepted")
    else:
        raise AssertionError("the rewrite must reject invented material tags")

    silent = FakeLLM({"改写": '{"brief":"只有标题"}'})
    try:
        with_llm(silent, lambda: revise.rewrite_shot(
            PROSE, 8.0, "随便改", "fake-service", materials=MATERIALS))
    except ValueError as error:
        check("没有返回" in str(error), "a silent rewrite was accepted as success")
    else:
        raise AssertionError("an answer without a prompt must not be accepted")

    try:
        revise.rewrite_shot(PROSE, 8.0, "  ", "fake-service")
    except ValueError as error:
        check("怎么改" in str(error), "an empty instruction gave a poor error")
    else:
        raise AssertionError("rewriting without an instruction should refuse")


def test_routes_and_error_status_mapping():
    source = (PACKAGE_DIR / "director_revise.py").read_text("utf-8")
    check('routes.post(LAYER_ROUTE)' in source
          and 'routes.post(REBIND_ROUTE)' in source
          and 'routes.post(REWRITE_ROUTE)' in source,
          "the revision helpers are not exposed to the card editor")
    check("asyncio.to_thread" in source,
          "a blocking LLM call would stall the ComfyUI event loop")
    check("_ROUTES_REGISTERED" in source,
          "routes can be registered twice on a module reload")
    init = (PACKAGE_DIR / "__init__.py").read_text("utf-8")
    check("_register_director_revise_routes()" in init,
          "the revision routes are never registered at startup")

    class Cancelled(Exception):
        pass
    Cancelled.__name__ = "InterruptProcessingException"
    check(revise._status_for(Cancelled())[0] == 409,
          "a cancelled request should not look like a provider failure")
    check(revise._status_for(RuntimeError("boom"))[0] == 502,
          "an unknown provider failure should surface as a bad gateway")


def test_card_editor_wires_the_three_revision_actions():
    ui = (PACKAGE_DIR / "web" / "h3_director_ui.js").read_text("utf-8")
    check('"/minimax-h3-myang/director/layer-shot"' in ui
          and '"/minimax-h3-myang/director/rebind-materials"' in ui
          and '"/minimax-h3-myang/director/rewrite-shot"' in ui,
          "the card editor does not call the revision endpoints")
    check('button("AI 分层"' in ui and "runShotLayering(node, shot, ai)" in ui,
          "a card still has to be layered by hand")
    check('button("手动分层"' in ui,
          "the hand-written layering path was removed instead of kept alongside")
    check("runShotRewrite(node, shot, index, revise)" in ui,
          "there is no per-shot AI rewrite action")
    check("runMaterialRebind(node, rebind)" in ui,
          "there is no way to rebind materials without touching the story")

    rebind = ui.index("async function runMaterialRebind")
    body = ui[rebind:ui.index("\n}", rebind)]
    # The no-story-change guarantee is that this path never assigns prose.
    check("shot.prompt" not in body and "layers.visual" not in body,
          "the material rebind path can overwrite a shot's prose")
    check("shot.assets = [" in body,
          "the rebind result is never applied to the cards")
    check("item.file?.name === asset.file?.name" in body,
          "rebinding twice would attach the same material again")

    rewrite = ui.index("async function runShotRewrite")
    rewrite_body = ui[rewrite:ui.index("\n}", rewrite)]
    check("previous_brief" in rewrite_body and "transition" in rewrite_body,
          "a single-shot rewrite is not told about the seam it must keep")
    check("data.dialogue_ok === false" in rewrite_body,
          "the operator is not told when a rewrite overruns its shot")


if __name__ == "__main__":
    for test in (
        test_layer_shot_splits_prose_without_touching_it,
        test_layer_shot_survives_a_failed_or_silent_layer,
        test_rebind_only_returns_bindings_and_drops_invented_targets,
        test_rewrite_shot_is_bounded_by_duration_and_manifest,
        test_routes_and_error_status_mapping,
        test_card_editor_wires_the_three_revision_actions,
    ):
        test()
        print("PASS", test.__name__)
