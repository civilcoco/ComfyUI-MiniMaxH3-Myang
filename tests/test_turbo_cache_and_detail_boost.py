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
nodes_v2 = importlib.import_module("ComfyUI-MiniMaxH3-Myang.nodes_v2")


def test_turbo_speed_cache_integration():
    print("Testing H3TurboSchedule speed_cache integration...")
    sched = turbo.H3TurboSchedule()
    assert hasattr(sched, "apply")
    schema = turbo.H3TurboSchedule.INPUT_TYPES()
    assert "speed_cache" in schema["required"]
    options = schema["required"]["speed_cache"][0]
    assert turbo.SPEED_CACHE_TESPEED in options
    assert turbo.SPEED_CACHE_SPECTRUM in options
    assert "LoRA文件" in schema["required"]
    assert "LoRA强度" in schema["required"]
    assert "手动覆盖Shift" in schema["required"]
    ui_source = (PACKAGE_DIR / "web" / "h3_turbo_ui.js").read_text("utf-8")
    assert "makeShiftStatus(node)" in ui_source
    assert 'visible(by.shift_video, override)' in ui_source
    print("PASS test_turbo_speed_cache_integration")


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
            shift_video=9.5, shift_audio=2.25,
            **{"手动覆盖Shift": True})
    finally:
        official.MiniMaxH3SigmaShift = original

    assert captured == {"video": 9.5, "audio": 2.25}
    assert result[2:] == (9.5, 2.25)
    marker = turbo.turbo_metadata(result[0])
    assert marker["shift_overridden"] is True
    assert marker["official_shift_video"] == 12.0
    assert marker["official_shift_audio"] == 3.0
    print("PASS test_turbo_can_override_and_publish_actual_shift")


def test_splitter_schema_clean():
    print("Testing H3ScriptSplitter clean schema...")
    splitter = nodes_v2.H3ScriptSplitter()
    schema = splitter.INPUT_TYPES()
    assert "detail_boost" not in schema["required"]
    assert "llm_enabled" in schema["required"]
    print("PASS test_splitter_schema_clean")


if __name__ == "__main__":
    test_turbo_speed_cache_integration()
    test_turbo_combines_model_only_lora_and_auto_profile()
    test_turbo_can_override_and_publish_actual_shift()
    test_splitter_schema_clean()
    print("ALL TESTS PASSED!")
