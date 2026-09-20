import importlib
import sys
from pathlib import Path

import torch


TEST_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = TEST_DIR.parent
CUSTOM_NODES_DIR = PACKAGE_DIR.parent
COMFY_DIR = CUSTOM_NODES_DIR.parent

for path in (str(COMFY_DIR), str(CUSTOM_NODES_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

progress = importlib.import_module("ComfyUI-MiniMaxH3-Myang.progress")


def test_second_pass_dynamic_vram_limit_caps_weight_residency():
    import comfy.model_management as model_management

    class FakeVbar:
        def __init__(self):
            self.limit = None
            self.resident = 12 * 1024 ** 3
            self.freed = 0

        def set_watermark_limit(self, value):
            self.limit = value

        def loaded_size(self):
            return self.resident

        def free_memory(self, value):
            self.freed += value
            self.resident -= value
            return value

    class FakePatcher:
        load_device = torch.device("cuda")

        def __init__(self, vbar):
            self.vbar = vbar

        def _vbar_get(self, create=False):
            assert create is True
            return self.vbar

    class FakeGuider:
        def __init__(self, patcher):
            self.model_patcher = patcher

    vbar = FakeVbar()
    original_total = model_management.get_total_memory
    original_free = model_management.get_free_memory
    try:
        model_management.get_total_memory = lambda _device: 16 * 1024 ** 3
        model_management.get_free_memory = lambda _device: 2 * 1024 ** 3
        result = progress._apply_second_pass_vram_limit(
            FakeGuider(FakePatcher(vbar)), 6.0)
    finally:
        model_management.get_total_memory = original_total
        model_management.get_free_memory = original_free

    assert result is vbar
    assert vbar.limit == 8 * 1024 ** 3
    assert vbar.freed == 4 * 1024 ** 3


def test_import_does_not_override_comfy_memory_policy():
    # Import-time reservation calls were removed, including NVML changes.
    source = (PACKAGE_DIR / "__init__.py").read_text("utf-8")
    assert "_apply_reserved_vram" not in source
    assert "set_nvml_pressure" not in source


def test_first_pass_gets_no_watermark_under_dynamic_vram():
    import comfy.memory_management as memory_management
    import comfy.model_management as model_management
    import comfy.sample

    class FakeVbar:
        def __init__(self):
            self.limit = None

        def set_watermark_limit(self, value):
            self.limit = value

        def loaded_size(self):
            return 0

        def free_memory(self, value):
            return value

    class FakePatcher:
        load_device = torch.device("cuda")

        def __init__(self, vbar):
            self.vbar = vbar

        def _vbar_get(self, create=False):
            return self.vbar

        def model_size(self):
            return 32 * 1024 ** 3

    class FakeGuider:
        def __init__(self, patcher):
            self.model_patcher = patcher

        @staticmethod
        def sample(*_args, **_kwargs):
            return torch.zeros(1, 4, 1, 1, 1)

    class FakeNoise:
        seed = 3

        @staticmethod
        def generate_noise(latent):
            return torch.zeros_like(latent["samples"])

    vbar = FakeVbar()
    originals = {
        "fix": comfy.sample.fix_empty_latent_channels,
        "intermediate": model_management.intermediate_device,
        "total": model_management.get_total_memory,
        "reserve": model_management.EXTRA_RESERVED_VRAM,
        "aimdo": memory_management.aimdo_enabled,
    }
    comfy.sample.fix_empty_latent_channels = lambda _m, samples, *_a: samples
    model_management.intermediate_device = lambda: torch.device("cpu")
    model_management.get_total_memory = lambda _device: 16 * 1024 ** 3
    memory_management.aimdo_enabled = True
    try:
        progress.H3SamplerAdvanced().execute(
            noise=FakeNoise(), guider=FakeGuider(FakePatcher(vbar)),
            sampler=object(), sigmas=torch.tensor([1.0, 0.0]),
            latent_image={"samples": torch.zeros(1, 4, 1, 1, 1)},
            vae=object(), run_id="pass1", owner_id="test",
            segment_index=1, total_segments=1, pass_label="sample1",
            reserve_vram_gb=1.25, preview_interval=0)
    finally:
        comfy.sample.fix_empty_latent_channels = originals["fix"]
        model_management.intermediate_device = originals["intermediate"]
        model_management.get_total_memory = originals["total"]
        model_management.EXTRA_RESERVED_VRAM = originals["reserve"]
        memory_management.aimdo_enabled = originals["aimdo"]

    # A watermark limit is a permission, not a cap on demand. Handing pass 1
    # "you may hold total-reserve" made AIMDO fill straight to 14.74GB of a
    # 31.67GB staged model and left 1.25GB for activations, which is worse than
    # the self-regulating NVML pressure path. Pass 1 must stay unwatermarked
    # until the profiles carry a real measured activation budget.
    assert vbar.limit is None


def test_chunked_sage_attention_reuses_disposable_input_buffer():
    class FakeAttention(torch.nn.Module):
        heads = 4
        head_dim = 2

        def __init__(self):
            super().__init__()
            self.qkv_proj = torch.nn.Linear(8, 24, bias=False)
            self.q_norm = torch.nn.Identity()
            self.k_norm = torch.nn.Identity()
            self.out_proj = torch.nn.Identity()

    attention = FakeAttention()
    original = torch.randn(7, 8)
    expected = attention.qkv_proj(original).split(8, dim=-1)[2]
    disposable = original.clone()
    pointer = disposable.data_ptr()

    def helper(parts, _dtype):
        _q, _k, value = parts
        parts.clear()
        return value

    def forbidden_fallback(*_args, **_kwargs):
        raise AssertionError("chunked disposable input unexpectedly fell back")

    output = progress._minimax_sage_chunk_reuse_forward(
        attention, [disposable], None, {"minimax_head_chunks": 2},
        helper=helper, mm=object(), ck=object(),
        fallback=forbidden_fallback)

    assert output.data_ptr() == pointer
    assert torch.allclose(output, expected)


def test_chunked_sage_attention_supports_real_h3_width_mismatch():
    class FakeAttention(torch.nn.Module):
        # MiniMax H3 has the same relationship at full scale: hidden=5376,
        # heads*head_dim=7168. Keep the toy tensor small but unequal.
        heads = 4
        head_dim = 2

        def __init__(self):
            super().__init__()
            self.qkv_proj = torch.nn.Linear(6, 24, bias=False)
            self.q_norm = torch.nn.Identity()
            self.k_norm = torch.nn.Identity()
            self.out_proj = torch.nn.Linear(8, 6, bias=False)

    attention = FakeAttention()
    original = torch.randn(11, 6)
    value = attention.qkv_proj(original).split(8, dim=-1)[2]
    expected = attention.out_proj(value)
    disposable = original.clone()
    pointer = disposable.data_ptr()

    def helper(parts, _dtype):
        _q, _k, part_value = parts
        parts.clear()
        return part_value

    def forbidden_fallback(*_args, **_kwargs):
        raise AssertionError("real H3 width relationship unexpectedly fell back")

    output = progress._minimax_sage_chunk_reuse_forward(
        attention, [disposable], None, {"minimax_head_chunks": 2},
        helper=helper, mm=object(), ck=object(),
        fallback=forbidden_fallback)

    assert output.data_ptr() == pointer
    assert output.shape == original.shape
    assert torch.allclose(output, expected)


def test_chunked_sage_attention_applies_reference_value_weight():
    class FakeAttention(torch.nn.Module):
        heads = 4
        head_dim = 2

        def __init__(self):
            super().__init__()
            self.qkv_proj = torch.nn.Linear(8, 24, bias=False)
            self.q_norm = torch.nn.Identity()
            self.k_norm = torch.nn.Identity()
            self.out_proj = torch.nn.Identity()

    attention = FakeAttention()
    source = torch.randn(7, 8)
    expected = attention.qkv_proj(source).split(8, dim=-1)[2].clone()
    expected[2:5].mul_(1.75)

    def helper(parts, _dtype):
        _q, _k, value = parts
        parts.clear()
        return value

    def forbidden_fallback(*_args, **_kwargs):
        raise AssertionError("reference weighting unexpectedly fell back")

    with torch.no_grad():
        output = progress._minimax_sage_chunk_reuse_forward(
            attention, source.clone(), None, {
                "minimax_head_chunks": 2,
                "minimax_reference_value_scales": ((2, 5, 1.75),),
            }, helper=helper, mm=object(), ck=object(),
            fallback=forbidden_fallback)

    assert torch.allclose(output, expected)


def test_lowmem_optimized_attention_applies_reference_value_weight():
    class FakeAttention(torch.nn.Module):
        heads = 2
        head_dim = 2

        def __init__(self):
            super().__init__()
            self.qkv_proj = torch.nn.Linear(4, 12, bias=False)
            self.q_norm = torch.nn.Identity()
            self.k_norm = torch.nn.Identity()
            self.out_proj = torch.nn.Identity()

    attention = FakeAttention()
    source = torch.randn(6, 4)
    expected = attention.qkv_proj(source).split(4, dim=-1)[2].clone()
    expected[1:4].mul_(0.6)

    def optimized(_q, _k, value, chunk_heads, **_kwargs):
        return value.transpose(1, 2).reshape(
            1, source.shape[0], chunk_heads * attention.head_dim)

    def forbidden_fallback(*_args, **_kwargs):
        raise AssertionError("low-VRAM reference weighting unexpectedly fell back")

    with torch.no_grad():
        output = progress._minimax_lowmem_reference_forward(
            attention, source.clone(), None, {
                "minimax_head_chunks": 2,
                "minimax_reference_value_scales": ((1, 4, 0.6),),
            }, fallback=forbidden_fallback, mm=object(),
            comfy_module=object(), optimized_attention=optimized)

    assert torch.allclose(output, expected)


def test_latent_rgb_step_preview_never_calls_vae():
    import folder_paths
    from PIL import Image

    class ForbiddenVAE:
        @staticmethod
        def decode(_latent):
            raise AssertionError("latent preview must not load or call VAE")

    original_temp = folder_paths.get_temp_directory
    original_save = Image.Image.save
    saved = []
    folder_paths.get_temp_directory = lambda: str(PACKAGE_DIR)
    Image.Image.save = lambda _image, path, *_args, **_kwargs: saved.append(path)
    try:
        filename = progress._save_step_preview(
            torch.randn(1, 24, 3, 8, 12), ForbiddenVAE(),
            "preview-test", 1, 0, "sample1", preview_mode="latent_rgb")
    finally:
        folder_paths.get_temp_directory = original_temp
        Image.Image.save = original_save

    assert filename
    assert saved and saved[0].endswith(filename)


def test_attention_preflight_refuses_unfittable_qkv_before_allocating():
    """A QKV block that cannot fit must fail before the CUDA allocator is
    asked: a doomed allocation crawls through WDDM shared memory for many
    minutes before raising, which reads exactly like a hang."""
    import comfy.model_management as model_management

    class FakeCudaDevice:
        type = "cuda"

    class FakeAttention:
        heads = 56
        head_dim = 128

    original_free = model_management.get_free_memory
    original_evictable = progress._dynamic_evictable_bytes
    try:
        model_management.get_free_memory = lambda _device: 5 * 1024 ** 3
        progress._dynamic_evictable_bytes = lambda: 0

        # 234k tokens x 7168-wide fused QKV is ~10GB; 5GB free + nothing
        # evictable cannot hold it even with the 40% grace.
        try:
            progress.assert_attention_fits(
                FakeAttention(), 234_000, 2, FakeCudaDevice())
        except RuntimeError as error:
            message = str(error)
            assert "QKV" in message and "5.0GB" in message, message
            assert "out of memory" not in message.casefold(), (
                "the pre-flight message must not look like a CUDA OOM or the "
                "sampler's retry classifier will swallow it")
            assert "allocation on device" not in message.casefold(), message
        else:
            raise AssertionError("an un-fittable QKV was allowed through")

        # The same block with enough free memory (and recoverable weight
        # pages) must pass silently so streaming pass-1 runs keep working.
        model_management.get_free_memory = lambda _device: 14 * 1024 ** 3
        progress._dynamic_evictable_bytes = lambda: 2 * 1024 ** 3
        progress.assert_attention_fits(
            FakeAttention(), 234_000, 2, FakeCudaDevice())

        # Non-CUDA devices skip the check entirely (CPU regression tests).
        progress.assert_attention_fits(
            FakeAttention(), 234_000, 2, torch.device("cpu"))
    finally:
        model_management.get_free_memory = original_free
        progress._dynamic_evictable_bytes = original_evictable


def test_pass2_cuda_oom_retries_once_with_clean_residency():
    import comfy.model_management as model_management
    import comfy.memory_management as memory_management
    import comfy.sample

    calls = []
    cleanup = []

    class FakeGuider:
        model_patcher = object()

        def sample(self, *args, **kwargs):
            calls.append(model_management.EXTRA_RESERVED_VRAM)
            if len(calls) == 1:
                raise torch.OutOfMemoryError("Allocation on device")
            return torch.zeros(1, 4, 1, 1, 1)

    class FakeNoise:
        seed = 123

        @staticmethod
        def generate_noise(latent):
            return torch.zeros_like(latent["samples"])

    originals = {
        "fix": comfy.sample.fix_empty_latent_channels,
        "unload": model_management.unload_all_models,
        "empty": model_management.soft_empty_cache,
        "intermediate": model_management.intermediate_device,
        "reserve": model_management.EXTRA_RESERVED_VRAM,
        "aimdo": memory_management.aimdo_enabled,
    }
    comfy.sample.fix_empty_latent_channels = lambda _m, samples, *_a: samples
    model_management.unload_all_models = lambda: cleanup.append("unload")
    model_management.soft_empty_cache = lambda: cleanup.append("empty")
    model_management.intermediate_device = lambda: torch.device("cpu")
    memory_management.aimdo_enabled = False
    try:
        output = progress.H3SamplerAdvanced().execute(
            noise=FakeNoise(), guider=FakeGuider(), sampler=object(),
            sigmas=torch.tensor([1.0, 0.0]),
            latent_image={"samples": torch.zeros(1, 4, 1, 1, 1)},
            vae=object(), run_id="test", owner_id="test",
            segment_index=1, total_segments=1, pass_label="sample2",
            reserve_vram_gb=2.5, preview_interval=2)
    finally:
        comfy.sample.fix_empty_latent_channels = originals["fix"]
        model_management.unload_all_models = originals["unload"]
        model_management.soft_empty_cache = originals["empty"]
        model_management.intermediate_device = originals["intermediate"]
        model_management.EXTRA_RESERVED_VRAM = originals["reserve"]
        memory_management.aimdo_enabled = originals["aimdo"]

    assert len(calls) == 2
    assert calls[0] >= 2.5 * 1024 ** 3
    assert calls[1] >= 3.5 * 1024 ** 3
    assert cleanup == ["unload", "empty"]
    assert output[0]["samples"].shape == (1, 4, 1, 1, 1)


def test_dynamic_pass2_oom_evicts_pages_without_unloading_host_cache():
    import comfy.model_management as model_management
    import comfy.memory_management as memory_management
    import comfy.sample

    calls = []
    unloads = []
    headroom_calls = []

    class FakeVbar:
        def __init__(self):
            self.resident = 4 * 1024 ** 3
            self.limit = None
            self.freed = 0

        def set_watermark_limit(self, value):
            self.limit = value

        def loaded_size(self):
            return self.resident

        def free_memory(self, value):
            value = min(int(value), self.resident)
            self.resident -= value
            self.freed += value
            return value

    class FakePatcher:
        load_device = torch.device("cuda")

        def __init__(self, vbar):
            self.vbar = vbar

        def _vbar_get(self, create=False):
            return self.vbar

        @staticmethod
        def model_size():
            return 11 * 1024 ** 3

    class FakeGuider:
        def __init__(self, patcher):
            self.model_patcher = patcher

        def sample(self, *_args, **_kwargs):
            calls.append(1)
            if len(calls) == 1:
                raise torch.OutOfMemoryError("Allocation on device")
            return torch.zeros(1, 4, 1, 1, 1)

    class FakeNoise:
        seed = 13

        @staticmethod
        def generate_noise(latent):
            return torch.zeros_like(latent["samples"])

    vbar = FakeVbar()
    originals = {
        "fix": comfy.sample.fix_empty_latent_channels,
        "unload": model_management.unload_all_models,
        "empty": model_management.soft_empty_cache,
        "intermediate": model_management.intermediate_device,
        "total": model_management.get_total_memory,
        "free": model_management.get_free_memory,
        "device": model_management.get_torch_device,
        "reserve": model_management.EXTRA_RESERVED_VRAM,
        "aimdo": memory_management.aimdo_enabled,
    }
    original_headroom = progress.increase_reservation
    progress.increase_reservation = lambda value: headroom_calls.append(value) or True
    comfy.sample.fix_empty_latent_channels = lambda _m, samples, *_a: samples
    model_management.unload_all_models = lambda: unloads.append(True)
    model_management.soft_empty_cache = lambda: None
    model_management.intermediate_device = lambda: torch.device("cpu")
    model_management.get_total_memory = lambda _device: 16 * 1024 ** 3
    model_management.get_free_memory = lambda _device: 4 * 1024 ** 3
    model_management.get_torch_device = lambda: torch.device("cpu")
    memory_management.aimdo_enabled = True
    try:
        output = progress.H3SamplerAdvanced().execute(
            noise=FakeNoise(), guider=FakeGuider(FakePatcher(vbar)),
            sampler=object(), sigmas=torch.tensor([1.0, 0.0]),
            latent_image={"samples": torch.zeros(1, 4, 1, 1, 1)},
            vae=object(), run_id="dynamic-oom", owner_id="test",
            segment_index=1, total_segments=1, pass_label="sample2",
            reserve_vram_gb=0.0, preview_interval=0)
    finally:
        comfy.sample.fix_empty_latent_channels = originals["fix"]
        model_management.unload_all_models = originals["unload"]
        model_management.soft_empty_cache = originals["empty"]
        model_management.intermediate_device = originals["intermediate"]
        model_management.get_total_memory = originals["total"]
        model_management.get_free_memory = originals["free"]
        model_management.get_torch_device = originals["device"]
        model_management.EXTRA_RESERVED_VRAM = originals["reserve"]
        memory_management.aimdo_enabled = originals["aimdo"]
        progress.increase_reservation = original_headroom

    assert len(calls) == 2
    assert not unloads
    assert vbar.limit is not None and vbar.freed >= 512 * 1024 ** 2
    assert output[0]["samples"].shape == (1, 4, 1, 1, 1)
    assert headroom_calls == [2.5], headroom_calls


def test_dynamic_vram_does_not_double_count_global_reserve():
    import comfy.memory_management as memory_management
    import comfy.model_management as model_management
    import comfy.sample

    observed = []

    class FakeGuider:
        model_patcher = object()

        @staticmethod
        def sample(*_args, **_kwargs):
            observed.append(model_management.EXTRA_RESERVED_VRAM)
            return torch.zeros(1, 4, 1, 1, 1)

    class FakeNoise:
        seed = 7

        @staticmethod
        def generate_noise(latent):
            return torch.zeros_like(latent["samples"])

    originals = {
        "fix": comfy.sample.fix_empty_latent_channels,
        "intermediate": model_management.intermediate_device,
        "reserve": model_management.EXTRA_RESERVED_VRAM,
        "aimdo": memory_management.aimdo_enabled,
    }
    comfy.sample.fix_empty_latent_channels = lambda _m, samples, *_a: samples
    model_management.intermediate_device = lambda: torch.device("cpu")
    memory_management.aimdo_enabled = True
    try:
        progress.H3SamplerAdvanced().execute(
            noise=FakeNoise(), guider=FakeGuider(), sampler=object(),
            sigmas=torch.tensor([1.0, 0.0]),
            latent_image={"samples": torch.zeros(1, 4, 1, 1, 1)},
            vae=object(), run_id="dynamic", owner_id="test",
            segment_index=1, total_segments=1, pass_label="sample2",
            reserve_vram_gb=4.5, preview_interval=0)
    finally:
        comfy.sample.fix_empty_latent_channels = originals["fix"]
        model_management.intermediate_device = originals["intermediate"]
        model_management.EXTRA_RESERVED_VRAM = originals["reserve"]
        memory_management.aimdo_enabled = originals["aimdo"]

    assert observed == [originals["reserve"]]


def test_sampler_announces_second_pass_before_first_step():
    import comfy.model_management as model_management
    import comfy.sample
    import server

    events = []

    class FakeGuider:
        model_patcher = object()

        @staticmethod
        def sample(*args, **kwargs):
            kwargs["callback"](0, torch.zeros(1, 4, 1, 1, 1), None, 2)
            return torch.zeros(1, 4, 1, 1, 1)

    class FakeNoise:
        seed = 1

        @staticmethod
        def generate_noise(latent):
            return torch.zeros_like(latent["samples"])

    class FakeServer:
        @staticmethod
        def send_sync(name, payload):
            events.append((name, payload))

    originals = {
        "fix": comfy.sample.fix_empty_latent_channels,
        "intermediate": model_management.intermediate_device,
        "instance": getattr(server.PromptServer, "instance", None),
        "reserve": model_management.EXTRA_RESERVED_VRAM,
    }
    comfy.sample.fix_empty_latent_channels = lambda _m, samples, *_a: samples
    model_management.intermediate_device = lambda: torch.device("cpu")
    server.PromptServer.instance = FakeServer()
    try:
        progress.H3SamplerAdvanced().execute(
            noise=FakeNoise(), guider=FakeGuider(), sampler=object(),
            sigmas=torch.tensor([1.0, 0.5, 0.0]),
            latent_image={"samples": torch.zeros(1, 4, 1, 1, 1)},
            vae=object(), run_id="run", owner_id="owner",
            segment_index=1, total_segments=1, pass_label="sample2",
            reserve_vram_gb=5.0, preview_interval=0)
    finally:
        comfy.sample.fix_empty_latent_channels = originals["fix"]
        model_management.intermediate_device = originals["intermediate"]
        server.PromptServer.instance = originals["instance"]
        model_management.EXTRA_RESERVED_VRAM = originals["reserve"]

    progress_events = [payload for name, payload in events
                       if name == "myh3_progress"]
    assert progress_events[0]["pass_label"] == "sample2"
    assert progress_events[0]["step"] == 0
    assert progress_events[0]["step_total"] == 2


def test_sampler_can_return_clean_x0_for_continuous_sigma():
    import comfy.model_management as model_management
    import comfy.sample

    class FakeModel:
        @staticmethod
        def process_latent_out(value):
            return value + 10

    class FakePatcher:
        model = FakeModel()

    class FakeGuider:
        model_patcher = FakePatcher()

        @staticmethod
        def sample(*args, **kwargs):
            kwargs["callback"](
                0, torch.full((1, 4, 1, 1, 1), 2.0), None, 1)
            return torch.full((1, 4, 1, 1, 1), 3.0)

    class FakeNoise:
        seed = 1

        @staticmethod
        def generate_noise(latent):
            return torch.zeros_like(latent["samples"])

    originals = {
        "fix": comfy.sample.fix_empty_latent_channels,
        "intermediate": model_management.intermediate_device,
        "reserve": model_management.EXTRA_RESERVED_VRAM,
    }
    comfy.sample.fix_empty_latent_channels = lambda _m, samples, *_a: samples
    model_management.intermediate_device = lambda: torch.device("cpu")
    try:
        noisy, clean = progress.H3SamplerAdvanced().execute(
            noise=FakeNoise(), guider=FakeGuider(), sampler=object(),
            sigmas=torch.tensor([1.0, 0.0]),
            latent_image={"samples": torch.zeros(1, 4, 1, 1, 1)},
            vae=object(), run_id="continuous", owner_id="owner",
            segment_index=1, total_segments=1, pass_label="sample1",
            reserve_vram_gb=0.0, preview_interval=0,
            return_denoised=True)
    finally:
        comfy.sample.fix_empty_latent_channels = originals["fix"]
        model_management.intermediate_device = originals["intermediate"]
        model_management.EXTRA_RESERVED_VRAM = originals["reserve"]

    assert torch.all(noisy["samples"] == 3)
    assert torch.all(clean["samples"] == 12)


if __name__ == "__main__":
    for name, function in sorted(list(globals().items())):
        if name.startswith("test_") and callable(function):
            function()
