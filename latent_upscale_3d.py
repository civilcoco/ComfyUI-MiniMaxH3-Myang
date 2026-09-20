"""Learned MiniMax H3 24-channel latent upscaler.

SPDX-License-Identifier: Apache-2.0

This is an in-package adaptation of the Apache-2.0 implementation in
AIMixer/ComfyUI_MiniMaxH3_Director (audited commit
``a267324a9f88141ff4e4b0e8c1a6ed90b4e45db7``).  It consumes the Apache-2.0
LBH-123-AI 3D latent-upscaler weights but does not require either custom-node
package at runtime.  Myang adds bounded temporal execution, selectable
precision, strict canvas alignment and immediate GPU offload.
"""

from __future__ import annotations

import glob
import gc
import logging
import os
import re

import torch
import torch.nn as nn
import torch.nn.functional as F


logger = logging.getLogger(__name__)

MODEL_FOLDER = "latent_upscale_models"
MISSING_MODEL = "（请把 H3 3D 放大权重放入 models/latent_upscale_models）"
PRECISIONS = ["fp16（推荐·省显存）", "fp32（最高稳定性）", "bf16（实验）"]

# Published training statistics for the Apache-2.0 LBH H3 upscaler weights.
LATENTS_MEAN = [
    0.858090341091156, -0.9606591463088989, 1.0661640167236328,
    -0.5090325474739075, -0.2727581858634949, -1.3675414323806763,
    -0.2553254961967468, -0.26907554268836975, -0.5376840829849243,
    -0.0464097298681736, 0.6657370328903198, 0.19690127670764923,
    -0.5460608005523682, -0.4035342037677765, -0.23683024942874908,
    0.25928452610969543, -0.30133944749832153, 0.211341992020607,
    -1.1206848621368408, 0.3581933379173279, -0.04225143790245056,
    0.2604829967021942, 0.22864092886447906, 0.7056031823158264,
]
LATENTS_STD = [
    1.2223774194717407, 1.2767263650894165, 1.6831774711608887,
    1.7549455165863037, 1.5636216402053832, 2.194143533706665,
    0.9653137922286987, 1.0569885969161987, 0.841948926448822,
    0.7729952931404114, 1.8955937623977661, 0.946841835975647,
    0.7996809482574463, 0.44988900423049927, 0.7197399735450743,
    0.6936293244361877, 2.961095094680786, 2.7694199085235596,
    3.0496184825897217, 2.1088054180145264, 3.276226282119751,
    3.1627357006073, 2.2816812992095947, 2.6127843856811523,
]

_MODEL_CACHE: dict[str, nn.Module] = {}


def clear_model_cache() -> int:
    """Drop the learned-upscaler CPU cache and return the number removed.

    ComfyUI's regular ``unload_all_models`` only knows about ModelPatcher
    instances.  The small 3D detail model is intentionally cached by this
    module, so it needs an explicit release path when a task is cancelled.
    """
    cached = list(_MODEL_CACHE.values())
    _MODEL_CACHE.clear()
    for model in cached:
        try:
            model.to("cpu")
        except Exception:
            pass
    if cached:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("H3-Myang: 已释放 %d 个神经 3D Latent 放大模型缓存", len(cached))
    return len(cached)


def _folder() -> str | None:
    try:
        import folder_paths

        if MODEL_FOLDER not in folder_paths.folder_names_and_paths:
            folder_paths.add_model_folder_path(
                MODEL_FOLDER, os.path.join(folder_paths.models_dir, MODEL_FOLDER))
        paths = folder_paths.get_folder_paths(MODEL_FOLDER)
        return paths[0] if paths else None
    except Exception:
        return None


def model_names() -> list[str]:
    root = _folder()
    names: list[str] = []
    if root:
        try:
            os.makedirs(root, exist_ok=True)
        except OSError:
            pass
        for extension in ("*.safetensors", "*.pth"):
            names.extend(os.path.basename(path)
                         for path in glob.glob(os.path.join(root, extension)))
        try:
            import folder_paths

            names.extend(folder_paths.get_filename_list(MODEL_FOLDER) or [])
        except Exception:
            pass
    unique = sorted({name for name in names if name and not name.startswith("（")})
    preferred = [name for name in unique if "3d" in name.casefold()]
    return preferred or unique or [MISSING_MODEL]


def precision_dtype(value: str, device: torch.device) -> torch.dtype:
    text = str(value).casefold()
    if device.type != "cuda":
        return torch.float32
    if "bf16" in text:
        return torch.bfloat16
    if "fp32" in text:
        return torch.float32
    return torch.float16


def _norm(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(32, channels)


def _zero(module: nn.Module) -> nn.Module:
    for parameter in module.parameters():
        parameter.detach().zero_()
    return module


class _Residual3D(nn.Module):
    def __init__(self, channels: int, embedding_channels: int,
                 dropout: float = 0.0, out_channels: int | None = None):
        super().__init__()
        self.out_channels = out_channels or channels
        # Attribute names intentionally match the published checkpoint keys.
        self.in_layers = nn.Sequential(
            _norm(channels), nn.SiLU(),
            nn.Conv3d(channels, self.out_channels, 3, padding=1))
        self.emb_layers = nn.Sequential(
            nn.SiLU(), nn.Linear(embedding_channels, 2 * self.out_channels))
        self.out_norm = _norm(self.out_channels)
        self.out_layers = nn.Sequential(
            nn.SiLU(), nn.Dropout(dropout),
            _zero(nn.Conv3d(self.out_channels, self.out_channels, 3, padding=1)))
        self.skip = (nn.Conv3d(channels, self.out_channels, 1)
                     if channels != self.out_channels else nn.Identity())

    def forward(self, tensor, embedding):
        hidden = self.in_layers(tensor)
        embedded = self.emb_layers(embedding).to(hidden.dtype)
        while embedded.ndim < hidden.ndim:
            embedded = embedded[..., None]
        scale, shift = torch.chunk(embedded, 2, dim=1)
        hidden = self.out_norm(hidden) * (1 + scale) + shift
        return self.skip(tensor) + self.out_layers(hidden)


class _Temporal3D(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 5):
        super().__init__()
        padding = kernel_size // 2
        self.norm = _norm(channels)
        self.dwconv = nn.Conv3d(
            channels, channels, (kernel_size, 1, 1),
            padding=(padding, 0, 0), groups=channels)
        self.pwconv = nn.Conv3d(channels, channels, 1)
        nn.init.zeros_(self.pwconv.weight)
        nn.init.zeros_(self.pwconv.bias)

    def forward(self, tensor):
        hidden = F.silu(self.norm(tensor))
        return tensor + self.pwconv(self.dwconv(hidden))


class LatentResizer3D(nn.Module):
    def __init__(self, in_channels=24, in_blocks=12, out_blocks=12,
                 channels=512, dropout=0.1, temporal_every=2,
                 temporal_kernel=5):
        super().__init__()
        self.conv_in = nn.Conv3d(in_channels, channels, 3, padding=1)
        embed_dim = 64
        self.embed = nn.Sequential(
            nn.Linear(1, embed_dim), nn.SiLU(),
            nn.Linear(embed_dim, embed_dim))
        self.in_blocks = self._blocks(
            in_blocks, channels, embed_dim, dropout,
            temporal_every, temporal_kernel)
        self.out_blocks = self._blocks(
            out_blocks, channels, embed_dim, dropout,
            temporal_every, temporal_kernel)
        self.norm_out = _norm(channels)
        self.conv_out = nn.Conv3d(channels, in_channels, 3, padding=1)

    @staticmethod
    def _blocks(count, channels, embed_dim, dropout,
                temporal_every, temporal_kernel):
        blocks = nn.ModuleList()
        for index in range(count):
            blocks.append(_Residual3D(channels, embed_dim, dropout))
            if temporal_every > 0 and index % temporal_every == 0:
                blocks.append(_Temporal3D(channels, temporal_kernel))
        return blocks

    @staticmethod
    def _run(blocks, tensor, embedding):
        for block in blocks:
            if isinstance(block, _Residual3D):
                tensor = block(tensor, embedding.expand(tensor.shape[0], -1))
            else:
                tensor = block(tensor)
        return tensor

    def forward(self, tensor, scale: float, target_size: tuple[int, int, int]):
        if tuple(target_size) == tuple(tensor.shape[-3:]):
            return tensor
        scale_embedding = torch.tensor(
            [[float(scale) - 1.0]], dtype=tensor.dtype, device=tensor.device)
        embedding = self.embed(scale_embedding)
        hidden = self._run(self.in_blocks, self.conv_in(tensor), embedding)
        hidden = F.interpolate(
            hidden, size=target_size, mode="trilinear", align_corners=False)
        hidden = self._run(self.out_blocks, hidden, embedding)
        return self.conv_out(F.silu(self.norm_out(hidden)))


def _load_state(path: str) -> dict:
    if path.casefold().endswith(".safetensors"):
        from safetensors.torch import load_file
        state = load_file(path, device="cpu")
    else:
        state = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
    if any(key.startswith("upscaler.") for key in state):
        state = {key[len("upscaler."):]: value
                 for key, value in state.items() if key.startswith("upscaler.")}
    return {key: (value.to(torch.float16)
                  if getattr(value, "dtype", None) == torch.float8_e4m3fn else value)
            for key, value in state.items()}


def _architecture(state: dict) -> dict:
    config = dict(in_channels=24, in_blocks=12, out_blocks=12,
                  channels=512, dropout=0.1, temporal_every=2,
                  temporal_kernel=5)
    weight = state.get("conv_in.weight")
    if weight is not None:
        config["in_channels"] = int(weight.shape[1])
        config["channels"] = int(weight.shape[0])
    input_ids, output_ids = set(), set()
    for key in state:
        found = re.match(r"in_blocks\.(\d+)\.in_layers\.", key)
        if found:
            input_ids.add(int(found.group(1)))
        found = re.match(r"out_blocks\.(\d+)\.in_layers\.", key)
        if found:
            output_ids.add(int(found.group(1)))
    if input_ids:
        config["in_blocks"] = len(input_ids)
    if output_ids:
        config["out_blocks"] = len(output_ids)
    temporal = [value for key, value in state.items()
                if key.endswith("dwconv.weight")]
    if temporal:
        config["temporal_kernel"] = int(temporal[0].shape[2])
    else:
        config["temporal_every"] = 0
    return config


def _model_path(name: str) -> str:
    if not name or str(name).startswith("（"):
        raise ValueError(
            "未选择神经 3D Latent 放大权重；请下载 Apache-2.0 权重并放到 "
            f"ComfyUI/models/{MODEL_FOLDER}/")
    import folder_paths

    path = folder_paths.get_full_path(MODEL_FOLDER, name)
    if path:
        return path
    root = _folder()
    candidate = os.path.join(root, name) if root else ""
    if candidate and os.path.isfile(candidate):
        return candidate
    raise FileNotFoundError(f"找不到 H3 Latent 放大权重：{name}")


def _load_model(name: str, dtype: torch.dtype) -> nn.Module:
    key = f"{name}::{dtype}"
    cached = _MODEL_CACHE.get(key)
    if cached is not None:
        return cached
    state = _load_state(_model_path(name))
    if "resizer.conv_in.weight" in state and "conv_in.weight" not in state:
        raise ValueError("当前只支持 3D H3 Latent 权重；所选文件看起来是 2D 权重")
    config = _architecture(state)
    model = LatentResizer3D(**config).to(device="cpu", dtype=dtype).eval()
    model.load_state_dict(state, strict=True)
    _MODEL_CACHE[key] = model
    logger.info("H3-Myang: 加载神经 3D Latent 放大器 %s（%s 参数）",
                name, f"{sum(p.numel() for p in model.parameters()):,}")
    return model


def _blend_ramp(span: int, overlap: int, device, dtype) -> torch.Tensor:
    """Cross-fade weights: flat across the core, tapering to zero at both ends."""
    ramp = torch.ones(span, device=device, dtype=dtype)
    taper = min(int(overlap), span // 2)
    if taper > 0:
        edge = torch.arange(
            1, taper + 1, device=device, dtype=dtype) / (taper + 1)
        ramp[:taper] = edge
        ramp[span - taper:] = edge.flip(0)
    return ramp.view(1, 1, span, 1, 1)


def _forward_bounded(model, tensor, scale, target_h, target_w,
                     chunk_steps: int, overlap: int = 4):
    """Blend overlapping temporal windows instead of butt-joining their cores.

    The old behaviour ran each window with context on both sides and then hard
    cropped the core.  Two neighbouring cores are produced from different
    receptive-field content, so every chunk boundary carried a visible pulse --
    at the default 16-step chunk that is one flicker roughly every 2.7 seconds
    of output, which reads as "the second pass is worse than the first".

    Two fixes, both matching upstream: replicate the first/last step so the edge
    windows see as much context as the interior ones, and cross-fade the overlap
    region rather than cutting it.  ``chunk_steps`` <= 0 runs the whole clip in
    one full-context pass, which has no seams at all.
    """
    total = int(tensor.shape[2])
    chunk_steps = int(chunk_steps)
    if chunk_steps <= 0 or total <= chunk_steps:
        return model(tensor, scale, (total, target_h, target_w))

    overlap = max(1, min(int(overlap), chunk_steps))
    padded = torch.cat((
        tensor[:, :, :1].expand(-1, -1, overlap, -1, -1),
        tensor,
        tensor[:, :, -1:].expand(-1, -1, overlap, -1, -1)), dim=2)

    accumulated = None
    weights = None
    for core_start in range(0, total, chunk_steps):
        core_stop = min(total, core_start + chunk_steps)
        window = padded[:, :, core_start:core_stop + 2 * overlap]
        span = int(window.shape[2])
        result = model(window, scale, (span, target_h, target_w))
        if accumulated is None:
            accumulated = torch.zeros(
                (int(result.shape[0]), int(result.shape[1]),
                 total + 2 * overlap, target_h, target_w),
                device=result.device, dtype=torch.float32)
            weights = torch.zeros(
                (1, 1, total + 2 * overlap, 1, 1),
                device=result.device, dtype=torch.float32)
        ramp = _blend_ramp(span, overlap, result.device, torch.float32)
        accumulated[:, :, core_start:core_start + span] += (
            result.to(torch.float32) * ramp)
        weights[:, :, core_start:core_start + span] += ramp
    blended = accumulated / weights.clamp_min(1e-6)
    return blended[:, :, overlap:overlap + total].to(dtype=tensor.dtype)


def upscale_video_latent(video_latent: torch.Tensor, target_width: int,
                         target_height: int, model_name: str,
                         precision: str, chunk_steps: int = 16) -> torch.Tensor:
    if not torch.is_tensor(video_latent) or video_latent.ndim != 5:
        raise ValueError("神经 H3 Latent 放大需要 [B,C,T,H,W] 视频潜变量")
    if int(video_latent.shape[1]) != 24:
        raise ValueError("神经 H3 Latent 放大只支持 MiniMax H3 的 24 通道潜变量")

    source_h, source_w = int(video_latent.shape[-2]), int(video_latent.shape[-1])
    target_width = max(32, round(int(target_width) / 32) * 32)
    target_height = max(32, round(int(target_height) / 32) * 32)
    target_w = max(2, target_width // 16)
    target_h = max(2, target_height // 16)
    if target_w < source_w or target_h < source_h:
        raise ValueError("神经 Latent 节点只做放大，不做降采样")
    if target_w == source_w and target_h == source_h:
        return video_latent

    scale = max(target_w / source_w, target_h / source_h)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = precision_dtype(precision, device)
    model = _load_model(model_name, dtype).to(device=device, dtype=dtype).eval()
    original_dtype = video_latent.dtype
    mean = torch.tensor(
        LATENTS_MEAN, device=device, dtype=dtype).view(1, -1, 1, 1, 1)
    std = torch.tensor(
        LATENTS_STD, device=device, dtype=dtype).view(1, -1, 1, 1, 1)
    work = video_latent.to(device=device, dtype=dtype)
    work = (work - mean) / std
    try:
        with torch.inference_mode():
            output = _forward_bounded(
                model, work, scale, target_h, target_w, chunk_steps)
            output = output * std + mean
        output = output.to(device="cpu", dtype=original_dtype).contiguous()
    finally:
        model.to("cpu")
        del work
        if device.type == "cuda":
            torch.cuda.empty_cache()
    logger.info(
        "H3-Myang: 神经3D Latent %dx%d -> %dx%d，T=%d，chunk=%s，%s",
        source_w, source_h, target_w, target_h,
        int(video_latent.shape[2]),
        "全上下文" if int(chunk_steps) <= 0 else int(chunk_steps), precision)
    return output
