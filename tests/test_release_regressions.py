"""Behavior regressions for issues found in the 0.2.0 release review."""
import importlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace, ModuleType
from unittest.mock import patch

import torch
import comfy.memory_management as memory
import comfy.model_management as management
import comfy.sample

agent = importlib.import_module("ComfyUI-MiniMaxH3-Myang.agent_nodes")
policy = importlib.import_module("ComfyUI-MiniMaxH3-Myang.memory_policy")
progress = importlib.import_module("ComfyUI-MiniMaxH3-Myang.progress")
GB = 1024 ** 3


def fake_aimdo(getter=True):
    calls = []
    lib = SimpleNamespace(set_simple_vram_headroom=calls.append,
                          set_nvml_pressure=lambda *_: (_ for _ in ()).throw(
                              AssertionError("NVML policy must stay unchanged")))
    if getter:
        lib.get_simple_vram_headroom = lambda: GB
    control = ModuleType("comfy_aimdo.control")
    control.lib = lib
    parent = ModuleType("comfy_aimdo")
    parent.control = control
    return {"comfy_aimdo": parent, "comfy_aimdo.control": control}, calls


def test_memory_restores_success_cancel_noise_error_and_retry_failure():
    from comfy.model_management import InterruptProcessingException
    for failure in (None, "cancel", "noise", "error", "oom"):
        modules, calls = fake_aimdo()
        attempts = []
        class Noise:
            seed = 1
            def generate_noise(self, latent):
                if failure == "noise":
                    raise RuntimeError("noise failed")
                return torch.zeros_like(latent["samples"])
        class Guider:
            model_patcher = object()
            def sample(self, *_args, **_kwargs):
                attempts.append(1)
                if failure == "cancel":
                    raise InterruptProcessingException()
                if failure == "error":
                    raise RuntimeError("sample failed")
                if failure == "oom":
                    raise torch.OutOfMemoryError("Allocation on device")
                return torch.zeros(1, 4, 1, 1, 1)
        before = management.EXTRA_RESERVED_VRAM
        with patch.dict("sys.modules", modules), patch.object(memory, "aimdo_enabled", True), patch.object(
                comfy.sample, "fix_empty_latent_channels", lambda _m, x, *_a: x), patch.object(
                management, "intermediate_device", lambda: torch.device("cpu")), patch.object(
                management, "soft_empty_cache", lambda: None):
            try:
                progress.H3SamplerAdvanced().execute(
                    noise=Noise(), guider=Guider(), sampler=object(),
                    sigmas=torch.tensor([1., 0.]),
                    latent_image={"samples": torch.zeros(1, 4, 1, 1, 1)},
                    vae=object(), run_id="regression", owner_id="regression",
                    segment_index=1, total_segments=1, pass_label="sample2",
                    reserve_vram_gb=4.5, preview_interval=0)
            except (RuntimeError, InterruptProcessingException):
                assert failure is not None
            else:
                assert failure is None
        assert calls[0] == int(4.5 * GB), (failure, calls)
        assert calls[-1] == GB, (failure, calls)
        assert management.EXTRA_RESERVED_VRAM == before
        if failure == "oom":
            assert len(attempts) == 2
            assert max(calls) > int(4.5 * GB)


def test_unknown_aimdo_baseline_is_not_overwritten():
    from comfy.cli_args import args
    modules, calls = fake_aimdo(getter=False)
    with patch.dict("sys.modules", modules), patch.object(memory, "aimdo_enabled", True), patch.object(
            args, "reserve_vram", None):
        with policy.MemoryReservation(4.5):
            assert not policy.increase_reservation(6)
    assert calls == []
    with patch.dict("sys.modules", modules), patch.object(memory, "aimdo_enabled", True), patch.object(
            args, "reserve_vram", 2.0):
        with policy.MemoryReservation(4.5):
            pass
    assert calls == [int(4.5 * GB), 2 * GB]


def test_static_memory_restores_and_nested_dynamic_scope_keeps_outer_target():
    with patch.object(memory, "aimdo_enabled", False), patch.object(
            management, "EXTRA_RESERVED_VRAM", GB):
        try:
            with policy.MemoryReservation(4):
                assert management.EXTRA_RESERVED_VRAM == 4 * GB
                raise RuntimeError("cancel")
        except RuntimeError:
            pass
        assert management.EXTRA_RESERVED_VRAM == GB
    modules, calls = fake_aimdo(getter=False)
    from comfy.cli_args import args
    with patch.dict("sys.modules", modules), patch.object(memory, "aimdo_enabled", True), patch.object(
            args, "reserve_vram", 1.0):
        with policy.MemoryReservation(4):
            with policy.MemoryReservation(6):
                pass
    assert calls == [4 * GB, 6 * GB, 4 * GB, GB]


def test_skill_add_edit_backup_and_legacy_migration_preserve_other_digests():
    with tempfile.TemporaryDirectory() as directory, patch.object(
            agent, "SKILL_DIR", Path(directory)), patch.object(agent, "EXTRA_SKILLS_DIR", None):
        root = Path(directory)
        for name in ("a.md", "b.md"):
            (root / name).write_text("# " + name, encoding="utf-8")
        records = {}
        for name in ("a.md", "b.md"):
            text, _ = agent._read_skill(name, budget=0)
            records[name] = {"full_text": text, "digest": "learned " + name, "learned_by": "llm"}
        cache = root / agent.SKILL_MEMORY_FILENAME
        # Legacy directory signatures must not invalidate unchanged texts.
        cache.write_text(json.dumps({"signature": "old", "skills": records}), encoding="utf-8")
        (root / "c.md").write_text("# C", encoding="utf-8")
        agent._learned_skill_text("c.md")
        assert agent._skill_memory()["a.md"]["digest"] == "learned a.md"
        assert agent._skill_memory()["b.md"]["digest"] == "learned b.md"
        (root / "a.md").write_text("# A revised", encoding="utf-8")
        agent._learned_skill_text("a.md")
        assert agent._skill_memory()["b.md"]["digest"] == "learned b.md"
        assert not agent._skill_memory()["a.md"]["digest"]
        cache.write_text("{broken", encoding="utf-8")
        assert agent._skill_memory()["b.md"]["digest"] == "learned b.md"
        assert cache.with_suffix(".json.bak").is_file()
        assert not list(root.glob("*.tmp"))


def test_native_attention_weight_changes_output_and_one_matches_original():
    from comfy.ldm.minimax.model import Attention
    torch.manual_seed(17)
    attention = Attention(8, 2, 4, 1e-6, operations=torch.nn)
    class Patcher:
        def __init__(self):
            self.object_patches = {}
            self.model_options = {}
        def get_model_object(self, _key):
            return SimpleNamespace(blocks=[SimpleNamespace(attn=attention)])
        def add_object_patch(self, key, value):
            self.object_patches[key] = value
    patcher = Patcher()
    assert progress._install_native_reference_weighting(patcher) == 1
    wrapped = next(iter(patcher.object_patches.values()))
    x = torch.randn(7, 8)
    def run(weight):
        layout = SimpleNamespace(reference_value_scales=((2, 5, weight),))
        return progress._reference_value_scales_wrapper(
            lambda value, _t, _c, transformer_options, **_kw: wrapped(
                value, transformer_options=transformer_options),
            x.clone(), None, None, transformer_options={},
            minimax_payload={"layout": layout})
    with torch.no_grad():
        original = attention(x.clone())
        one, two = run(1), run(2)
    torch.testing.assert_close(one, original)
    assert (one - two).abs().max().item() > 1e-5
    assert progress._install_native_reference_weighting(patcher) == 0
