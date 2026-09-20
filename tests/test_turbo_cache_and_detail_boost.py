import importlib
import sys
import unittest
from pathlib import Path
import torch
from types import SimpleNamespace

TEST_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = TEST_DIR.parent
CUSTOM_NODES_DIR = PACKAGE_DIR.parent
COMFY_DIR = CUSTOM_NODES_DIR.parent

for p in (str(CUSTOM_NODES_DIR), str(COMFY_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

pkg = importlib.import_module("ComfyUI-MiniMaxH3-Myang")
turbo = importlib.import_module("ComfyUI-MiniMaxH3-Myang.turbo")
nodes = importlib.import_module("ComfyUI-MiniMaxH3-Myang.nodes")
nodes = importlib.import_module("ComfyUI-MiniMaxH3-Myang.nodes")


def test_turbo_speed_cache_is_legacy_only():
    print("Testing H3TurboSchedule legacy speed_cache compatibility...")
    sched = turbo.H3TurboSchedule()
    assert hasattr(sched, "apply")
    schema = turbo.H3TurboSchedule.INPUT_TYPES()
    assert "speed_cache" in schema["required"]
    options = schema["required"]["speed_cache"][0]
    # Legacy values remain accepted so old serialized widget positions/values
    # do not make workflow validation fail, but the frontend hides the field
    # and the runtime no longer imports or mounts either implementation.
    assert turbo.SPEED_CACHE_TESPEED in options
    assert turbo.SPEED_CACHE_SPECTRUM in options
    assert "LoRA文件" in schema["required"]
    assert "LoRA强度" in schema["required"]
    assert "手动覆盖Shift" in schema["required"]
    assert "附加LoRA开启" in schema["required"]
    for index in range(1, 4):
        assert "附加LoRA%d启用" % index in schema["required"]
        assert "附加LoRA%d文件" % index in schema["required"]
        assert "附加LoRA%d强度" % index in schema["required"]
    assert "附加LoRA4文件" not in schema["required"]
    ui_source = (PACKAGE_DIR / "web" / "h3_turbo_ui.js").read_text("utf-8")
    assert "makeShiftStatus(node)" in ui_source
    assert 'visible(by.shift_video, override)' in ui_source
    assert "hide(by.speed_cache)" in ui_source
    assert "index <= 3" in ui_source
    turbo_source = (PACKAGE_DIR / "turbo.py").read_text("utf-8")
    assert "TESpeedMiniMaxH3" not in turbo_source
    assert "SpectrumApplyMiniMaxH3" not in turbo_source
    print("PASS test_turbo_speed_cache_is_legacy_only")


def test_turbo_combines_model_only_lora_and_auto_profile():
    print("Testing combined model-only LoRA and AV schedule...")
    name = "minimax h3/minimax_h3_ref2va_turbo_4step_v0.1_comfyui_bf16.safetensors"
    assert turbo.infer_profile(name) == turbo.PROFILE_REF_4_V01
    captured = {}

    class FakeLoader:
        def load_lora_model_only(self, model, lora_name, strength_model):
            captured.update(base=model, lora=lora_name, strength=strength_model)
            return ("lora-model",)

    import comfy_extras.nodes_minimax_h3 as official
    original = official.MiniMaxH3SigmaShift

    class FakeShift:
        @classmethod
        def execute(cls, model, shift_video, shift_audio):
            captured.update(shift_model=model, video=shift_video, audio=shift_audio)
            return SimpleNamespace(result=(SimpleNamespace(model_options={}),))

    official.MiniMaxH3SigmaShift = FakeShift
    try:
        node = turbo.H3TurboSchedule()
        node._lora_loader = FakeLoader()
        result = node.apply(
            object(), turbo.PROFILE_AUTO,
            **{"LoRA文件": name, "LoRA强度": 0.8})
    finally:
        official.MiniMaxH3SigmaShift = original

    assert captured["lora"] == name and captured["strength"] == 0.8
    assert captured["shift_model"] == "lora-model"
    assert captured["video"] == 12.0 and captured["audio"] == 3.0
    marker = turbo.turbo_metadata(result[0])
    assert marker["task_family"] == "ref2va"
    assert marker["lora_loaded_here"] is True
    assert result[1:] == (4, 12.0, 3.0)
    print("PASS test_turbo_combines_model_only_lora_and_auto_profile")


def test_turbo_can_override_and_publish_actual_shift():
    print("Testing visible/custom Turbo Shift override...")
    captured = {}
    import comfy_extras.nodes_minimax_h3 as official
    original = official.MiniMaxH3SigmaShift

    class FakeShift:
        @classmethod
        def execute(cls, model, shift_video, shift_audio):
            captured.update(video=shift_video, audio=shift_audio)
            return SimpleNamespace(result=(SimpleNamespace(model_options={}),))

    official.MiniMaxH3SigmaShift = FakeShift
    try:
        result = turbo.H3TurboSchedule().apply(
            object(), turbo.PROFILE_8_V1,
            speed_cache=turbo.SPEED_CACHE_TESPEED,
            shift_video=9.5, shift_audio=2.25,
            **{"手动覆盖Shift": True})
    finally:
        official.MiniMaxH3SigmaShift = original

    assert captured == {"video": 9.5, "audio": 2.25}
    assert result[2:] == (9.5, 2.25)
    marker = turbo.turbo_metadata(result[0])
    assert marker["shift_overridden"] is True
    assert marker["speed_cache"] == turbo.SPEED_CACHE_OFF
    assert marker["official_shift_video"] == 12.0
    assert marker["official_shift_audio"] == 3.0
    print("PASS test_turbo_can_override_and_publish_actual_shift")


def test_turbo_stacks_up_to_three_effect_loras_after_official_lora():
    print("Testing ordered additional effect LoRA stacking...")
    calls = []

    class FakeLoader:
        def load_lora_model_only(self, model, lora_name, strength_model):
            calls.append((model, lora_name, strength_model))
            return ("%s>%s" % (model, lora_name),)

    import comfy_extras.nodes_minimax_h3 as official
    original = official.MiniMaxH3SigmaShift

    class FakeShift:
        @classmethod
        def execute(cls, model, shift_video, shift_audio):
            calls.append(("shift", model, shift_video, shift_audio))
            return SimpleNamespace(result=(SimpleNamespace(model_options={}),))

    official.MiniMaxH3SigmaShift = FakeShift
    try:
        node = turbo.H3TurboSchedule()
        node._lora_loader = FakeLoader()
        result = node.apply(
            "base", turbo.PROFILE_4_V1_768,
            **{
                "LoRA文件": "turbo_4step_768p.safetensors", "LoRA强度": 1.0,
                "附加LoRA开启": True,
                "附加LoRA1启用": True,
                "附加LoRA1文件": "style.safetensors", "附加LoRA1强度": 0.6,
                "附加LoRA2启用": False,
                "附加LoRA2文件": "ignored.safetensors", "附加LoRA2强度": 1.0,
                "附加LoRA3启用": True,
                "附加LoRA3文件": "character.safetensors", "附加LoRA3强度": -0.25,
            })
    finally:
        official.MiniMaxH3SigmaShift = original

    assert [entry[1] for entry in calls[:3]] == [
        "turbo_4step_768p.safetensors", "style.safetensors", "character.safetensors"]
    assert calls[1][0] == "base>turbo_4step_768p.safetensors"
    assert calls[2][0].endswith(">style.safetensors")
    assert calls[3][0] == "shift" and calls[3][1].endswith(">character.safetensors")
    marker = turbo.turbo_metadata(result[0])
    assert marker["additional_loras"] == [
        {"slot": 1, "name": "style.safetensors", "strength": 0.6},
        {"slot": 3, "name": "character.safetensors", "strength": -0.25},
    ]
    print("PASS test_turbo_stacks_up_to_three_effect_loras_after_official_lora")


def test_turbo_rejects_duplicate_lora_files_before_loading():
    node = turbo.H3TurboSchedule()
    try:
        node.apply(
            object(), turbo.PROFILE_4_V01,
            **{
                "LoRA文件": "same_4step.safetensors",
                "附加LoRA开启": True,
                "附加LoRA1启用": True,
                "附加LoRA1文件": "same_4step.safetensors",
            })
    except ValueError as error:
        assert "不能重复叠加" in str(error)
    else:
        raise AssertionError("duplicate Turbo/effect LoRA was loaded twice")
    print("PASS test_turbo_rejects_duplicate_lora_files_before_loading")


def test_splitter_schema_clean():
    print("Testing H3ScriptSplitter clean schema...")
    splitter = nodes.H3ScriptSplitter()
    schema = splitter.INPUT_TYPES()
    assert "detail_boost" not in schema["required"]
    assert "llm_enabled" in schema["required"]
    print("PASS test_splitter_schema_clean")


if __name__ == "__main__":
    test_turbo_speed_cache_is_legacy_only()
    test_turbo_combines_model_only_lora_and_auto_profile()
    test_turbo_can_override_and_publish_actual_shift()
    test_turbo_stacks_up_to_three_effect_loras_after_official_lora()
    test_turbo_rejects_duplicate_lora_files_before_loading()
    test_splitter_schema_clean()
    print("ALL TESTS PASSED!")
