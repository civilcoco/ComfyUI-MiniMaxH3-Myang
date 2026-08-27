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

DETAIL_UPSCALE_METHODS = [
    "neural_3d (神经3D Latent放大·推荐)",
    "latent (latent空间放大·jingchen573方式)",
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

            "upscale_method": (DETAIL_UPSCALE_METHODS, {"default": "latent (latent空间放大·jingchen573方式)"}),
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
                "default": 16, "min": 1, "max": 128,
                "tooltip": "神经3D按 latent 时间步分块；越小越省显存，越大越快",
            }),
        }, "optional": {
            "vae": ("VAE", {"tooltip": "接视频 VAE 做 decode→encode 投影，把插值后的 latent 拉回 VAE 流形，消除马赛克/色彩偏移。不接则纯 latent 插值（最快但可能有伪影）"}),

        }}



    def upscale(self, samples, resolution, aspect_ratio="16:9", width=1664, height=928,
                upscale_method="latent (latent空间放大·jingchen573方式)",
                chunk_frames=4,
                latent_upscale_model="", latent_precision=LATENT_PRECISIONS[0],
                latent_chunk_steps=16, vae=None):
        if "neural_3d" in str(upscale_method).casefold():
            return (upscale_learned_latent(
                samples, resolution, aspect_ratio, width, height,
                latent_upscale_model, latent_precision, latent_chunk_steps),)
        if vae is not None:
            # VAE 投影模式：decode → 像素放大 → encode（和自用版工作流一样，无伪影）
            return (_project_latent(
                samples, vae, resolution, aspect_ratio, width, height,
                upscale_method, chunk_frames),)
        # 纯 latent 插值（最快，但 ViT decoder 可能产生伪影）

        return (upscale_latent(samples, resolution, aspect_ratio, width, height, upscale_method),)





def _project_latent(latent_dict, vae, resolution, aspect_ratio, width, height,
                    upscale_method="pixel", chunk_frames=4):
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



    # 1. VAE decode: latent -> pixels [B, T, H, W, C] in [0, 1]

    pixels = vae.decode(video_latent)

    pixels_4d = pixels[0]  # [T, H, W, C]



    # 2. Pixel upscale.  Use the exact same preset/aspect resolver as the
    # H3Condition node built for pass 2.  Deriving the canvas from the decoded
    # source aspect here used to turn a 1152x640 first pass into 1504x832,
    # while H3Condition correctly allocated 1472x832 for the selected 16:9
    # preset.  Pass 1 could sample, but its 1504-wide face-refined latent then
    # failed as the next segment's 1472-wide detail anchor.
    target_w, target_h = core.canvas_for(
        resolution, aspect_ratio, width, height)
    source_h, source_w = int(pixels_4d.shape[1]), int(pixels_4d.shape[2])
    if target_w < source_w or target_h < source_h:
        raise ValueError(
            "二采目标 %dx%d 小于一采 %dx%d；这是放大精修节点，不做降采样" %
            (target_w, target_h, source_w, source_h))
    upscaled = resize_frames(
        pixels_4d, target_w, target_h, upscale_method, chunk_frames)


    logger.info("H3-Myang: latent VAE projection decode->resize(%dx%d)->encode | %d frames",

                target_w, target_h, int(upscaled.shape[0]))



    # 3. VAE encode: upscaled pixels -> latent on manifold.
    #
    # ComfyUI's VAE wrapper expects a video IMAGE batch as [T,H,W,C].  It
    # adds the leading video batch itself *after* cropping H/W.  Supplying
    # [1,T,H,W,C] here makes ``vae_encode_crop_pixels`` treat T as another
    # spatial axis and crop it to a multiple of the spatial VAE ratio.  A
    # legal 175-frame H3 clip was therefore cropped to 160 before encoding,
    # producing 47 temporal tokens / 158 decoded frames instead of 52 / 175.
    # That silent clock change later crashed H3V2VInit and also shortened the
    # face-refine stream.  Keep the public IMAGE convention used by the stock
    # VAEEncode node; the wrapper will form [1,C,T,H,W] internally.
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
                "default": 16, "min": 1, "max": 128,
                "tooltip": "神经3D时间分块；8 更省显存，16 默认，32 更快",
            }),
            "passes": ("INT", {
                "default": 1, "min": 1, "max": 8,
                "tooltip": "二采轮数；只在前一轮放大，后续保持同分辨率精修",
            }),
            "seed_mode": (DETAIL_SEED_MODES, {"default": DETAIL_SEED_INHERIT}),
        }, "optional": {
            "二采模型": ("MODEL", {

                "tooltip": "接 Turbo LoRA 之前的 Ref2VA 基模；开启二采时必须连接",

            }),

        }}



    def build(self, enabled, resolution, width, height, steps, denoise,
              scheduler, sampler_name, upscale_method, chunk_frames,
              mode=DETAIL_MODE_UPSCALE_REFINE, latent_upscale_model="",
              latent_precision=LATENT_PRECISIONS[0], latent_chunk_steps=16,
              passes=1, seed_mode=DETAIL_SEED_INHERIT, **kwargs):
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





NODE_CLASS_MAPPINGS = {

    "H3DetailSettings": H3DetailSettings,

    "H3DetailRefine": H3DetailRefine,

    "H3LatentUpscale": H3LatentUpscale,

}

NODE_DISPLAY_NAME_MAPPINGS = {

    "H3DetailSettings": "沐阳 H3 · 二采放大设置",

    "H3DetailRefine": "沐阳 H3 · 二采放大精修（像素路径）",

    "H3LatentUpscale": "沐阳 H3 · Latent 直接放大（极速双采）",

}
