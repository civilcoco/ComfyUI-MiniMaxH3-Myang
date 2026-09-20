"""Memory-bounded pixel upscale followed by a native H3 low-noise pass."""

import logging
import os
import subprocess
import sys
import tempfile

import torch
import torchaudio

import comfy.nested_tensor
import comfy.utils
import folder_paths
from comfy_execution.graph_utils import GraphBuilder

from . import core
from .anchors import pixel_frames
from .latent_upscale_3d import (
    PRECISIONS as LATENT_PRECISIONS,
    model_names as latent_model_names,
    upscale_video_latent as learned_upscale_video_latent,
)
from .turbo import turbo_metadata


logger = logging.getLogger(__name__)

DETAIL_OFF = "关闭"
DETAIL_RESOLUTIONS = [
    DETAIL_OFF, "540P", "640P", "720P", "768P", "832P", "928P",
    "1024P", "1080P", "自定义",
]
# The old "latent (latent空间放大·jingchen573方式)" entry is deliberately gone.
# Bislerp on a temporally compressed H3 latent blends cells that each decode
# into a four-frame group, so two different motion states come back overlaid --
# the ghosting is inherent, not a tuning problem. Its documented cure (wire a
# VAE for the decode->encode projection) makes it identical to ``pixel``, so no
# configuration left it both correct and distinct. ``upscale_latent`` is still
# reachable on H3LatentUpscale by leaving ``vae`` unwired, as that input says.
DETAIL_UPSCALE_METHODS = [
    "neural_3d (神经3D Latent放大·推荐)",
    "pixel (像素放大·自用版工作流方式)",
    "nvidia_rtx_vsr (NVIDIA RTX 视频超分·实验)",
]
DETAIL_IMAGE_METHODS = [
    "pixel (像素放大·自用版工作流方式)",
    "nvidia_rtx_vsr (NVIDIA RTX 视频超分·实验)",
]
DETAIL_MODE_UPSCALE_REFINE = "放大 + 二采（推荐）"
DETAIL_MODE_REFINE = "同分辨率二采（不放大）"
DETAIL_MODE_UPSCALE_ONLY = "仅放大（不二采·最快）"
DETAIL_MODES = [
    DETAIL_MODE_UPSCALE_REFINE,
    DETAIL_MODE_REFINE,
    DETAIL_MODE_UPSCALE_ONLY,
]
DETAIL_SEED_INHERIT = "每轮沿用同一种子"
DETAIL_SEED_OFFSET = "每轮种子 +1"
DETAIL_SEED_MODES = [DETAIL_SEED_INHERIT, DETAIL_SEED_OFFSET]
DETAIL_SAMPLERS = ["res_multistep", "euler"]
DETAIL_SCHEDULERS = ["beta", "simple", "normal"]
DETAIL_MEMORY_AUTO = "自动平衡（16GB推荐）"
DETAIL_MEMORY_SPEED = "速度优先（关闭二采逐步预览）"
DETAIL_MEMORY_PREVIEW = "完整逐步预览（占显存）"
DETAIL_MEMORY_LOW = "显存优先（832P保底）"
DETAIL_MEMORY_CUSTOM = "自定义"
DETAIL_MEMORY_PROFILES = [
    DETAIL_MEMORY_AUTO,
    DETAIL_MEMORY_SPEED,
    DETAIL_MEMORY_PREVIEW,
    DETAIL_MEMORY_LOW,
    DETAIL_MEMORY_CUSTOM,
]


def detail_memory_policy(profile, custom_reserve_gb=1.25,
                         custom_preview_interval=2):
    """Return per-sampler VRAM headroom and VAE preview cadence.

    A second-pass step preview invokes the video VAE while H3 is still loaded.
    On 16 GB Windows cards that can push allocations into shared GPU memory and
    make every following diffusion step reload weights.  Keeping the final
    refined preview while sampling fewer intermediate previews is lossless for
    the generated video and substantially reduces that contention.
    """

    value = str(profile or DETAIL_MEMORY_AUTO)
    if value == DETAIL_MEMORY_SPEED:
        return 0.0, 0
    if value == DETAIL_MEMORY_PREVIEW:
        return 0.0, 1
    if value == DETAIL_MEMORY_LOW:
        # The in-place MiniMax attention path removes one full [tokens, hidden]
        # allocation, so 4.5GB is a safer and substantially faster 832P water-
        # mark than the old 6GB cap. Under AIMDO this is activation capacity,
        # not globally unused VRAM.
        return 4.5, 0
    if value == DETAIL_MEMORY_CUSTOM:
        return (max(0.0, min(8.0, float(custom_reserve_gb))),
                max(0, min(100, int(custom_preview_interval))))
    # AIMDO already evicts dynamic weight pages from NVML pressure.  A fixed
    # headroom value makes the 11GB Ref2VA model needlessly page from host RAM
    # before the first 720/832P step.  Automatic mode therefore installs no
    # watermark at all; an actual OOM triggers a targeted dynamic-page retry.
    # The finished second-pass segment is still decoded and previewed normally.
    return 0.0, 0


def target_size(images, resolution, width=1344, height=768):
    """Choose a 32-aligned canvas without cropping the source aspect."""
    if not isinstance(images, torch.Tensor) or images.ndim != 4:
        raise ValueError("H3 二采精修需要 [帧, 高, 宽, 通道] 的 IMAGE")
    source_h, source_w = int(images.shape[1]), int(images.shape[2])
    if source_h < 1 or source_w < 1:
        raise ValueError("H3 二采精修收到空画面")

    if str(resolution) == "自定义":
        target_w = max(32, int(width) // 32 * 32)
        target_h = max(32, int(height) // 32 * 32)
    else:
        short_edge = int(str(resolution).rstrip("P"))
        ratio = source_w / source_h
        if ratio >= 1.0:
            target_w, target_h = short_edge * ratio, float(short_edge)
        else:
            target_w, target_h = float(short_edge), short_edge / ratio
        target_w = max(32, round(target_w / 32) * 32)
        target_h = max(32, round(target_h / 32) * 32)

    if target_w < source_w or target_h < source_h:
        raise ValueError(
            "二采目标 %dx%d 小于一采 %dx%d；这是放大精修节点，不做降采样" %
            (target_w, target_h, source_w, source_h))
    return target_w, target_h


def _resize_nvidia_vsr(images, width, height, chunk_frames):
    try:
        import nvvfx
    except ImportError as error:
        raise ImportError(
            "NVIDIA RTX VSR 不可用：请安装 nvidia-vfx，并确认使用兼容的 NVIDIA GPU") from error
    if not torch.cuda.is_available():
        raise RuntimeError("NVIDIA RTX VSR 需要可用的 CUDA 显卡")
    if int(images.shape[-1]) != 3:
        raise ValueError("NVIDIA RTX VSR 只接受三通道 RGB 画面")

    frame_count = int(images.shape[0])
    output_width = max(8, round(int(width) / 8) * 8)
    output_height = max(8, round(int(height) / 8) * 8)
    worker = os.path.join(os.path.dirname(__file__), "rtx_vsr_worker.py")
    if not os.path.isfile(worker):
        raise RuntimeError("NVIDIA RTX VSR 隔离工作器缺失")

    temp_root = folder_paths.get_temp_directory()
    os.makedirs(temp_root, exist_ok=True)
    with tempfile.TemporaryDirectory(
            prefix="h3_rtx_vsr_", dir=temp_root) as temp_dir:
        import numpy as np

        input_path = os.path.join(temp_dir, "input.f16")
        output_path = os.path.join(temp_dir, "output.f16")
        source = np.memmap(
            input_path, mode="w+", dtype=np.float16,
            shape=(frame_count, int(images.shape[1]), int(images.shape[2]), 3))
        chunk_frames = max(1, int(chunk_frames))
        for start in range(0, frame_count, chunk_frames):
            stop = min(frame_count, start + chunk_frames)
            source[start:stop] = images[start:stop].detach().to(
                device="cpu", dtype=torch.float16).numpy()
        source.flush()
        del source

        command = [
            sys.executable, worker,
            "--input", input_path,
            "--output", output_path,
            "--frames", str(frame_count),
            "--input-width", str(int(images.shape[2])),
            "--input-height", str(int(images.shape[1])),
            "--output-width", str(output_width),
            "--output-height", str(output_height),
        ]
        timeout_seconds = max(120, 60 + frame_count * 2)
        try:
            completed = subprocess.run(
                command, capture_output=True, text=True,
                timeout=timeout_seconds, check=False)
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(
                "NVIDIA RTX VSR 超时（%d 秒）；隔离进程已终止，"
                "ComfyUI 主进程不会被卡死" % timeout_seconds) from error
        if completed.returncode != 0 or "RTX_VSR_OK" not in completed.stdout:
            details = (completed.stderr or completed.stdout or "未知原生错误").strip()
            raise RuntimeError("NVIDIA RTX VSR 执行失败：%s" % details[-2000:])

        mapped = np.memmap(
            output_path, mode="r", dtype=np.float16,
            shape=(frame_count, output_height, output_width, 3))
        output = torch.from_numpy(np.array(mapped, copy=True))
        del mapped
        return output


def resize_frames(images, width, height, method, chunk_frames):
    """Resize with bounded GPU staging and compact CPU output storage."""
    method_str = str(method).lower().strip()
    if "nvidia_rtx_vsr" in method_str or "rtx_vsr" in method_str or "vsr" in method_str:
        return _resize_nvidia_vsr(
            images, int(width), int(height), int(chunk_frames))
    frame_count = int(images.shape[0])
    chunk_frames = max(1, int(chunk_frames))
    target_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    output = torch.empty(
        (frame_count, int(height), int(width), int(images.shape[-1])),
        dtype=torch.float16, device="cpu")

    if "lanczos" in method_str:
        comfy_method = "lanczos"
    elif "bicubic" in method_str:
        comfy_method = "bicubic"
    elif "bilinear" in method_str:
        comfy_method = "bilinear"
    elif "nearest" in method_str:
        comfy_method = "nearest-exact"
    elif "area" in method_str:
        comfy_method = "area"
    elif "bislerp" in method_str:
        comfy_method = "bislerp"
    else:
        comfy_method = "lanczos"

    for start in range(0, frame_count, chunk_frames):
        stop = min(frame_count, start + chunk_frames)
        chunk = images[start:stop].detach().to(device=target_device, dtype=torch.float32)
        chunk = comfy.utils.common_upscale(
            chunk.movedim(-1, 1), int(width), int(height), comfy_method,
            "disabled").movedim(1, -1)
        output[start:stop].copy_(chunk.to(device="cpu", dtype=output.dtype))
    return output


def join_av_latent(video, audio):
    if not isinstance(video, torch.Tensor) or video.ndim != 5:
        raise ValueError("H3 视频潜变量必须是 [B,C,T,H,W]")
    if not isinstance(audio, torch.Tensor) or audio.ndim != 4:
        raise ValueError("H3 音频潜变量必须是 [B,C,声道,T]")
    if int(video.shape[0]) != int(audio.shape[0]):
        raise ValueError("H3 二采的视频与音频 batch 不一致")

    frames = pixel_frames(int(video.shape[2]))
    expected_audio = round(frames / 24.0 * 40.0)
    actual_audio = int(audio.shape[-1])
    if abs(actual_audio - expected_audio) > 1:
        raise ValueError(
            "H3 二采音画时长不一致：%d 帧应约 %d 个音频步，实际 %d" %
            (frames, expected_audio, actual_audio))
    return {"samples": comfy.nested_tensor.NestedTensor((video, audio))}


def upscale_latent(samples, resolution, aspect_ratio="16:9", width=1664, height=928, method="latent_bicubic"):
    """Directly spatial upscale H3 NestedTensor (video, audio) without intermediate VAE encode/decode.

    Uses comfy.utils.common_upscale with bislerp (smoothest in latent space) and
    jingchen573's alignment algorithm (preserves aspect ratio, even latent dims).
    """
    if isinstance(samples, dict):
        nested = samples.get("samples")
    else:
        nested = samples

    if hasattr(nested, "unbind"):
        tensors = nested.unbind()
        video_latent, audio_latent = tensors[0], tensors[1]
    elif isinstance(nested, (tuple, list)):
        video_latent, audio_latent = nested[0], nested[1]
    else:
        video_latent, audio_latent = nested, None

    if not isinstance(video_latent, torch.Tensor) or video_latent.ndim != 5:
        raise ValueError("MiniMax H3 视频 Latent 必须是 [B, C, T, H, W]")

    source_lat_h = int(video_latent.shape[3])
    source_lat_w = int(video_latent.shape[4])

    target_w, target_h = core.canvas_for(resolution, aspect_ratio, width, height)
    raw_target_lat_h = max(2, target_h // 16)
    raw_target_lat_w = max(2, target_w // 16)

    # jingchen573 alignment: short side floor-to-even first,
    # long side follows the short side's actual scale ratio (preserves aspect ratio).
    LATENT_ALIGN = 2

    def floor_even(v):
        return max(LATENT_ALIGN, (v // LATENT_ALIGN) * LATENT_ALIGN)

    if source_lat_w >= source_lat_h:
        long_in, short_in = source_lat_w, source_lat_h
        long_raw, short_raw = raw_target_lat_w, raw_target_lat_h
    else:
        long_in, short_in = source_lat_h, source_lat_w
        long_raw, short_raw = raw_target_lat_h, raw_target_lat_w

    short_out = floor_even(short_raw)
    short_scale = short_out / short_in
    ideal_long = long_in * short_scale
    long_cap = floor_even(long_raw)

    lower = floor_even(int(ideal_long))
    upper = lower + LATENT_ALIGN
    candidates = {c for c in (lower, upper, long_cap) if LATENT_ALIGN <= c <= long_cap}
    long_out = min(candidates, key=lambda c: (abs(c - ideal_long), c)) if candidates else long_cap

    if source_lat_w >= source_lat_h:
        target_lat_w, target_lat_h = long_out, short_out
    else:
        target_lat_h, target_lat_w = long_out, short_out

    # bislerp is the smoothest method for latent-space interpolation;
    # "latent" in method name -> default to bislerp (jingchen573 recommended).
    mode_str = str(method).lower()
    if "bislerp" in mode_str or "latent" in mode_str:
        comfy_method = "bislerp"
    elif "bicubic" in mode_str:
        comfy_method = "bicubic"
    elif "bilinear" in mode_str:
        comfy_method = "bilinear"
    elif "nearest" in mode_str:
        comfy_method = "nearest-exact"
    elif "area" in mode_str:
        comfy_method = "area"
    else:
        comfy_method = "bislerp"

    B, C, T, H, W = video_latent.shape
    device = video_latent.device
    dtype = video_latent.dtype

    flat_video = video_latent.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W).to(torch.float32)
    upscaled_flat = comfy.utils.common_upscale(
        flat_video, target_lat_w, target_lat_h, comfy_method, "disabled"
    ).to(dtype=dtype, device=device)
    upscaled_video = upscaled_flat.view(B, T, C, target_lat_h, target_lat_w).permute(0, 2, 1, 3, 4)

    actual_pixel_w = target_lat_w * 16
    actual_pixel_h = target_lat_h * 16
    logger.info("H3-Myang: Latent upscale (%dx%d -> %dx%d latent | %dx%d px) method=%s",
                W, H, target_lat_w, target_lat_h, actual_pixel_w, actual_pixel_h, comfy_method)

    if audio_latent is not None:
        return {"samples": comfy.nested_tensor.NestedTensor((upscaled_video, audio_latent))}
    else:
        return {"samples": upscaled_video}


def upscale_learned_latent(samples, resolution, aspect_ratio, width, height,
                           model_name, precision, chunk_steps):
    """Apply the learned 3D network to the video stream and preserve H3 audio."""
    nested = samples.get("samples") if isinstance(samples, dict) else samples
    if hasattr(nested, "unbind"):
        streams = list(nested.unbind())
    elif isinstance(nested, (tuple, list)):
        streams = list(nested)
    else:
        streams = [nested]
    video = streams[0]
    audio = streams[1] if len(streams) > 1 else None
    target_w, target_h = core.canvas_for(
        resolution, aspect_ratio, width, height)
    output_video = learned_upscale_video_latent(
        video, target_w, target_h, model_name, precision, chunk_steps)
    packed = dict(samples) if isinstance(samples, dict) else {}
    packed.pop("noise_mask", None)
    if audio is not None:
        packed["samples"] = comfy.nested_tensor.NestedTensor(
            (output_video, audio))
    else:
        packed["samples"] = output_video
    return packed


class H3LatentUpscale:
    CATEGORY = "沐阳 H3"
    FUNCTION = "upscale"
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("samples",)
    DESCRIPTION = (
        "直接在 Latent 潜在空间对 H3 视频进行空间插值放大（极速）。"
        "接 VAE 后自动做 decode→encode 投影，消除插值伪影（马赛克/色彩偏移）。")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "samples": ("LATENT",),
            "resolution": (DETAIL_RESOLUTIONS[1:], {"default": "832P"}),
            "aspect_ratio": (core.ASPECT_RATIOS, {"default": "16:9"}),
            "width": ("INT", {"default": 1664, "min": 32, "max": 8192, "step": 32}),
            "height": ("INT", {"default": 928, "min": 32, "max": 8192, "step": 32}),
            "upscale_method": (DETAIL_UPSCALE_METHODS, {"default": "neural_3d (神经3D Latent放大·推荐)"}),
            "chunk_frames": ("INT", {
                "default": 4, "min": 1, "max": 64,
                "tooltip": "像素/VSR 分块帧数；越小越省显存",
            }),
            "latent_upscale_model": (latent_model_names(), {
                "tooltip": "神经3D模式使用。权重放到 models/latent_upscale_models",
            }),
            "latent_precision": (LATENT_PRECISIONS, {
                "default": LATENT_PRECISIONS[0],
            }),
            "latent_chunk_steps": ("INT", {
                "default": 0, "min": 0, "max": 256,
                "tooltip": "神经3D按 latent 时间步分块；0=全上下文单次推理（无接缝，推荐）。"
                           "显存不够再往上调",
            }),
            "reserve_vram_gb": ("FLOAT", {
                "default": 0.0, "min": 0.0, "max": 8.0, "step": 0.05,
                "tooltip": "像素/VAE投影时为系统预留显存，避免16GB显卡进入共享显存",
            }),
        }, "optional": {
            "vae": ("VAE", {"tooltip": "接视频 VAE 做 decode→encode 投影，把插值后的 latent 拉回 VAE 流形，消除马赛克/色彩偏移。不接则纯 latent 插值（最快但可能有伪影）"}),
        }}

    def upscale(self, samples, resolution, aspect_ratio="16:9", width=1664, height=928,
                upscale_method="neural_3d (神经3D Latent放大·推荐)",
                chunk_frames=4,
                latent_upscale_model="", latent_precision=LATENT_PRECISIONS[0],
                latent_chunk_steps=0, vae=None, reserve_vram_gb=0.0):
        if "neural_3d" in str(upscale_method).casefold():
            return (upscale_learned_latent(
                samples, resolution, aspect_ratio, width, height,
                latent_upscale_model, latent_precision, latent_chunk_steps),)
        if vae is not None:
            # VAE 投影模式：decode → 像素放大 → encode（和自用版工作流一样，无伪影）
            return (_project_latent(
                samples, vae, resolution, aspect_ratio, width, height,
                upscale_method, chunk_frames,
                reserve_vram_gb=float(reserve_vram_gb)),)
        # 纯 latent 插值（最快，但 ViT decoder 可能产生伪影）
        return (upscale_latent(samples, resolution, aspect_ratio, width, height, upscale_method),)


from .memory_policy import scoped_reservation


@scoped_reservation
def _project_latent(latent_dict, vae, resolution, aspect_ratio, width, height,
                    upscale_method="pixel", chunk_frames=4,
                    reserve_vram_gb=0.0):
    """VAE decode -> pixel upscale -> VAE encode (same as self-use workflow).

    Decodes one-pass latent to pixels, upscales in pixel space (reliable,
    no ViT decoder artifacts from latent interpolation), then re-encodes
    to get a valid high-res latent on the VAE manifold.
    """
    nested = latent_dict.get("samples") if isinstance(latent_dict, dict) else latent_dict
    if hasattr(nested, "unbind"):
        tensors = list(nested.unbind())
    elif isinstance(nested, (tuple, list)):
        tensors = list(nested)
    else:
        tensors = [nested]

    video_latent = tensors[0]
    audio_latent = tensors[1] if len(tensors) > 1 else None

    # The reservation decorator restores the prior policy on every exit.
    # 1. VAE decode: latent -> pixels [B, T, H, W, C] in [0, 1]
    pixels = vae.decode(video_latent)
    pixels_4d = pixels[0]  # [T, H, W, C]

    # 2. Pixel upscale.  Use the same preset/aspect resolver as the pass-2
    # condition so every segment keeps exactly the same latent geometry.
    target_w, target_h = core.canvas_for(
        resolution, aspect_ratio, width, height)
    source_h = int(pixels_4d.shape[1])
    source_w = int(pixels_4d.shape[2])
    if target_w < source_w or target_h < source_h:
        raise ValueError(
            "二采目标 %dx%d 小于一采 %dx%d；这是放大精修节点，不做降采样" %
            (target_w, target_h, source_w, source_h))
    upscaled = resize_frames(
        pixels_4d, target_w, target_h, upscale_method, chunk_frames)

    logger.info(
        "H3-Myang: latent VAE projection decode->resize(%dx%d)->encode | %d frames",
        target_w, target_h, int(upscaled.shape[0]))

    # Pixel resize is CPU-backed, but the completed VAE decode and the last
    # interpolation chunk can leave several GB of reclaimable CUDA blocks.
    # Drop the low-resolution frame owners and return those blocks before
    # asking the VAE encoder for its high-resolution workspace.
    del pixels_4d, pixels
    if torch.cuda.is_available():
        import comfy.model_management
        comfy.model_management.soft_empty_cache()

    # 3. Keep the public IMAGE convention [T,H,W,C].  Adding another batch
    # dimension here makes ComfyUI crop the temporal axis as if it were a
    # spatial dimension and silently shortens the video.
    new_video_latent = vae.encode(upscaled)

    if audio_latent is not None:
        return {"samples": comfy.nested_tensor.NestedTensor((new_video_latent, audio_latent))}
    return {"samples": new_video_latent}


class H3DetailSettings:
    CATEGORY = "沐阳 H3"
    FUNCTION = "build"
    RETURN_TYPES = ("MYANG_H3_DETAIL",)
    RETURN_NAMES = ("二采设置",)
    DESCRIPTION = "长视频二采的独立总开关与全部参数；长视频节点只保留连接口。"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "enabled": ("BOOLEAN", {"default": False,
                                    "label_on": "开启二采",
                                    "label_off": "关闭二采"}),
            "mode": (DETAIL_MODES, {"default": DETAIL_MODE_UPSCALE_REFINE}),
            "resolution": (DETAIL_RESOLUTIONS[1:], {"default": "832P"}),
            "width": ("INT", {"default": 1664, "min": 32, "max": 8192,
                              "step": 32,
                              "tooltip": "仅在二采输出短边选『自定义』时使用"}),
            "height": ("INT", {"default": 928, "min": 32, "max": 8192,
                               "step": 32,
                               "tooltip": "仅在二采输出短边选『自定义』时使用"}),
            "steps": ("INT", {"default": 4, "min": 1, "max": 100,
                              "tooltip": "二采采样步数（推荐 4 步）"}),
            "denoise": ("FLOAT", {"default": 0.2, "min": 0.01,
                                  "max": 1.0, "step": 0.01,
                                  "tooltip": "二采重绘幅度（推荐 0.15~0.25）"}),
            "scheduler": (DETAIL_SCHEDULERS, {"default": "beta"}),
            "sampler_name": (DETAIL_SAMPLERS, {"default": "res_multistep"}),
            "upscale_method": (DETAIL_UPSCALE_METHODS, {"default": "neural_3d (神经3D Latent放大·推荐)"}),
            "chunk_frames": ("INT", {"default": 4, "min": 1, "max": 64,
                                    "tooltip": "放大分组帧数；RTX VSR 内部仍逐帧进出显卡"}),
            "latent_upscale_model": (latent_model_names(), {
                "tooltip": "神经3D模式的 Apache-2.0 权重；放到 models/latent_upscale_models",
            }),
            "latent_precision": (LATENT_PRECISIONS, {
                "default": LATENT_PRECISIONS[0],
                "tooltip": "fp16 更省显存；出现异常色块时改 fp32",
            }),
            "latent_chunk_steps": ("INT", {
                "default": 0, "min": 0, "max": 256,
                "tooltip": "神经3D时间分块；0=全上下文单次推理（无接缝，推荐）。"
                           "显存不够再往上调，8 最省显存",
            }),
            "passes": ("INT", {
                "default": 1, "min": 1, "max": 8,
                "tooltip": "二采轮数；只在前一轮放大，后续保持同分辨率精修",
            }),
            "seed_mode": (DETAIL_SEED_MODES, {"default": DETAIL_SEED_INHERIT}),
            # Append-only: new widgets go after the existing ones so saved
            # workflows keep their positional values.
            "reuse_condition": ("BOOLEAN", {
                "default": True,
                "label_on": "复用文本/素材条件（推荐）",
                "label_off": "按二采分辨率重建条件",
                "tooltip": "开：只复用文本token与已编码参考素材，不包含、不复制"
                           "一采成片；二采目标latent始终独立。"
                           "关：按二采分辨率重跑一次条件，参考图会被重采样到更大面积，"
                               "token 全变，低降噪几步收不过去，容易涂抹和轻微身份漂移",
            }),
            # Append-only memory block for positional workflow compatibility.
            "memory_profile": (DETAIL_MEMORY_PROFILES, {
                "default": DETAIL_MEMORY_AUTO,
                "tooltip": "控制二采模型权重驻留与激活空间，以及清晰逐步预览频率；"
                           "DynamicVRAM下激活空间仍会被计算使用，不是空置预留。"
                           "不改变832P输出、步数、重绘幅度或最终画质",
            }),
            "custom_reserve_gb": ("FLOAT", {
                "default": 1.25, "min": 0.0, "max": 8.0, "step": 0.05,
                "tooltip": "仅自定义档生效；DynamicVRAM下为二采激活空间，"
                           "非动态模式下为传统显存预留；0=沿用ComfyUI启动设置",
            }),
            "custom_preview_interval": ("INT", {
                "default": 2, "min": 0, "max": 100,
                "tooltip": "仅自定义档生效；0=关闭二采逐步清晰预览，1=每步，2=每2步。"
                           "二采完成后的最终预览始终保留",
            }),
            # Append-only experimental sampler path. Keep every preceding
            # widget position stable for existing workflows.
            "continuous_sigma": ("BOOLEAN", {
                "default": False,
                "label_on": "连续 Sigma（实验）",
                "label_off": "独立二采（默认）",
                "tooltip": "开启后把一采步数和二采步数组成一条 Sigma 轨迹："
                           "低分辨率执行前半段，放大 noisy latent 后继续剩余步数。"
                           "将复用一采模型、采样器、调度器与种子；不增加总步数，"
                           "但暂不兼容音频精修、小脸/动作修复、检查点直入和像素放大。",
            }),
            # Append-only: a 1:1 post-decode enhancement, independent of the
            # upscale method (which does nothing at all in 同分辨率 mode).
            "vsr_enhance": ("BOOLEAN", {
                "default": False,
                "label_on": "二采后 VSR 增强（原尺寸）",
                "label_off": "关闭 VSR 增强",
                "tooltip": "仅『同分辨率二采』生效：解码后按原尺寸跑一遍 NVIDIA RTX VSR，"
                           "只去噪锐化，不改分辨率。需要 nvvfx 与 NVIDIA 显卡；每段一次"
                           "隔离进程加一整段磁盘往返、逐帧推理。"
                           "『放大 + 二采』和『仅放大』请用放大方式里的 VSR，"
                           "否则同一批帧要过两遍",
            }),
        }, "optional": {
            "二采模型": ("MODEL", {
                "tooltip": "接 Turbo LoRA 之前的 Ref2VA 基模；开启二采时必须连接",
            }),
        }}

    def build(self, enabled, resolution, width, height, steps, denoise,
              scheduler, sampler_name, upscale_method, chunk_frames,
              mode=DETAIL_MODE_UPSCALE_REFINE, latent_upscale_model="",
              latent_precision=LATENT_PRECISIONS[0], latent_chunk_steps=0,
              passes=1, seed_mode=DETAIL_SEED_INHERIT, reuse_condition=True,
              memory_profile=DETAIL_MEMORY_AUTO, custom_reserve_gb=1.25,
              custom_preview_interval=2, continuous_sigma=False,
              vsr_enhance=False,
              **kwargs):
        reserve_gb, preview_interval = detail_memory_policy(
            memory_profile, custom_reserve_gb, custom_preview_interval)
        return ({
            "enabled": bool(enabled),
            "mode": str(mode),
            "resolution": str(resolution),
            "width": int(width),
            "height": int(height),
            "steps": int(steps),
            "denoise": float(denoise),
            "scheduler": str(scheduler),
            "sampler_name": str(sampler_name),
            "upscale_method": str(upscale_method),
            "chunk_frames": int(chunk_frames),
            "latent_upscale_model": str(latent_upscale_model),
            "latent_precision": str(latent_precision),
            "latent_chunk_steps": int(latent_chunk_steps),
            "passes": int(passes),
            "seed_mode": str(seed_mode),
            "reuse_condition": bool(reuse_condition),
            "memory_profile": str(memory_profile),
            "reserve_vram_gb": float(reserve_gb),
            "preview_interval": int(preview_interval),
            "continuous_sigma": bool(continuous_sigma),
            "vsr_enhance": bool(vsr_enhance),
            "model": kwargs.get("二采模型"),
        },)


class H3DetailRefine:
    CATEGORY = "沐阳 H3"
    FUNCTION = "refine"
    RETURN_TYPES = ("IMAGE", "LATENT")
    RETURN_NAMES = ("refined_images", "detail_latent")
    DESCRIPTION = (
        "低分辨率一采结果先在 CPU 分块放大，再用未挂 Turbo LoRA 的 H3 基模"
        "做低降噪二采；最终音频保持一采原音频。")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "h3": ("MYANG_H3",),
            "model": ("MODEL", {
                "tooltip": "必须接 Turbo LoRA 之前的 Ref2VA 基模",
            }),
            "conditioning": ("CONDITIONING",),
            "images": ("IMAGE",),
            "audio": ("AUDIO",),
            "resolution": (DETAIL_RESOLUTIONS[1:], {"default": "768P"}),
            "width": ("INT", {"default": 1664, "min": 32, "max": 8192,
                              "step": 32}),
            "height": ("INT", {"default": 928, "min": 32, "max": 8192,
                               "step": 32}),
            "upscale_method": (DETAIL_IMAGE_METHODS, {"default": DETAIL_IMAGE_METHODS[0]}),
            "chunk_frames": ("INT", {"default": 4, "min": 1, "max": 64}),
            "steps": ("INT", {"default": 4, "min": 1, "max": 100}),
            "denoise": ("FLOAT", {"default": 0.2, "min": 0.01,
                                  "max": 1.0, "step": 0.01}),
            "scheduler": (DETAIL_SCHEDULERS, {"default": "beta"}),
            "sampler_name": (DETAIL_SAMPLERS, {"default": "res_multistep"}),
            "noise_seed": ("INT", {"default": 0, "min": 0,
                                  "max": 0xffffffffffffffff}),
        }}

    def refine(self, h3, model, conditioning, images, audio, resolution,
               width, height, upscale_method, chunk_frames, steps, denoise,
               scheduler, sampler_name, noise_seed):
        if turbo_metadata(model) is not None:
            raise ValueError(
                "二采精修必须接 Turbo LoRA 之前的 H3 基模；"
                "Turbo 模型不能运行 beta/低降噪二采")
        frames = int(images.shape[0])
        if core.length_for(frames / 24.0, 24.0) != frames:
            raise ValueError("二采输入帧数 %d 不在 H3 的 17k+5 网格上" % frames)
        target_w, target_h = target_size(images, resolution, width, height)
        upscaled = resize_frames(
            images, target_w, target_h, upscale_method, chunk_frames)

        video_latent = h3.video_vae.encode(upscaled)
        del upscaled
        waveform = audio["waveform"]
        sample_rate = int(audio["sample_rate"])
        vae_rate = int(getattr(h3.audio_vae, "audio_sample_rate", 44100))
        if sample_rate != vae_rate:
            waveform = torchaudio.functional.resample(
                waveform, sample_rate, vae_rate)
        audio_latent = h3.audio_vae.encode(waveform.movedim(1, -1))
        latent = join_av_latent(video_latent, audio_latent)

        graph = GraphBuilder()
        guider = graph.node(
            "BasicGuider", model=model, conditioning=conditioning)
        sigmas = graph.node(
            "BasicScheduler", model=model, scheduler=scheduler,
            steps=steps, denoise=denoise)
        sampler = graph.node("KSamplerSelect", sampler_name=sampler_name)
        noise = graph.node("RandomNoise", noise_seed=noise_seed)
        sample = graph.node(
            "SamplerCustomAdvanced", noise=noise.out(0), guider=guider.out(0),
            sampler=sampler.out(0), sigmas=sigmas.out(0),
            latent_image=latent)
        decoded = graph.node(
            "VAEDecode", samples=sample.out(0), vae=h3.video_vae)

        logger.info(
            "H3-Myang: 二采 %dx%d -> %dx%d | %d步 denoise=%.2f | 原音频旁路",
            int(images.shape[2]), int(images.shape[1]), target_w, target_h,
            int(steps), float(denoise))
        return {"expand": graph.finalize(),
                "result": (decoded.out(0), sample.out(0))}


class H3PixelUpscale:
    CATEGORY = "沐阳 H3"
    FUNCTION = "upscale"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    DESCRIPTION = (
        "只在像素空间放大已解码的画面，不碰 VAE。『仅放大（不二采）』模式用它，"
        "RTX VSR / Lanczos 算出来的锐度不会再被一次 VAE 往返抹回去。")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "resolution": (DETAIL_RESOLUTIONS[1:], {"default": "832P"}),
            "aspect_ratio": (core.ASPECT_RATIOS, {"default": "16:9"}),
            "width": ("INT", {"default": 1664, "min": 32, "max": 8192,
                              "step": 32}),
            "height": ("INT", {"default": 928, "min": 32, "max": 8192,
                               "step": 32}),
            "upscale_method": (DETAIL_IMAGE_METHODS, {
                "default": DETAIL_IMAGE_METHODS[0]}),
            "chunk_frames": ("INT", {"default": 4, "min": 1, "max": 64}),
        }}

    def upscale(self, images, resolution, aspect_ratio, width, height,
                upscale_method, chunk_frames):
        # Same canvas resolver the sampling path uses, so switching between
        # 仅放大 and 放大+二采 cannot silently change the output size.
        target_w, target_h = core.canvas_for(
            resolution, aspect_ratio, width, height)
        source_h, source_w = int(images.shape[1]), int(images.shape[2])
        if target_w < source_w or target_h < source_h:
            raise ValueError(
                "二采目标 %dx%d 小于一采 %dx%d；这是放大节点，不做降采样" %
                (target_w, target_h, source_w, source_h))
        upscaled = resize_frames(
            images, target_w, target_h, upscale_method, chunk_frames)
        logger.info(
            "H3-Myang: 仅放大（纯像素）%dx%d -> %dx%d | %d 帧 | 0 次 VAE",
            source_w, source_h, target_w, target_h, int(images.shape[0]))
        return (upscaled.to(dtype=images.dtype),)


class H3VsrEnhance:
    CATEGORY = "沐阳 H3"
    FUNCTION = "enhance"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    DESCRIPTION = (
        "二采解码后按原尺寸跑一遍 NVIDIA RTX VSR：不改分辨率，只做去噪与锐化。"
        "『同分辨率二采』没有缩放步骤，放大方式对它是空转，这里是 VSR 唯一"
        "既能生效、又不会被后续 VAE 编码抹掉的位置。")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "chunk_frames": ("INT", {
                "default": 4, "min": 1, "max": 64,
                "tooltip": "只影响送进隔离进程的分批大小；VSR 内部始终逐帧过显卡",
            }),
        }}

    def enhance(self, images, chunk_frames=4):
        height, width = int(images.shape[1]), int(images.shape[2])
        # ``_resize_nvidia_vsr`` rounds its output to a multiple of 8, so an
        # unaligned canvas would come back a different size and this would
        # quietly become a resize instead of an enhancement. Every H3 canvas is
        # 32-aligned, so refuse rather than resample by accident.
        if width % 8 or height % 8:
            raise ValueError(
                "VSR 增强要求画面宽高是 8 的倍数，当前 %dx%d" % (width, height))
        logger.info(
            "H3-Myang: 二采后 VSR 增强 | %dx%d 原尺寸 | %d 帧",
            width, height, int(images.shape[0]))
        enhanced = resize_frames(
            images, width, height, "nvidia_rtx_vsr", chunk_frames)
        return (enhanced.to(dtype=images.dtype),)


NODE_CLASS_MAPPINGS = {
    "H3DetailSettings": H3DetailSettings,
    "H3DetailRefine": H3DetailRefine,
    "H3LatentUpscale": H3LatentUpscale,
    "H3PixelUpscale": H3PixelUpscale,
    "H3VsrEnhance": H3VsrEnhance,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3DetailSettings": "沐阳 H3 · 二采放大设置",
    "H3DetailRefine": "沐阳 H3 · 二采放大精修（像素路径）",
    "H3LatentUpscale": "沐阳 H3 · Latent 直接放大（极速双采）",
    "H3PixelUpscale": "沐阳 H3 · 纯像素放大（不经 VAE）",
    "H3VsrEnhance": "沐阳 H3 · 二采后 VSR 增强（原尺寸）",
}
