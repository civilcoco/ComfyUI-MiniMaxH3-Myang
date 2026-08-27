"""沐阳 H3 · 长视频执行进度信号节点。

这是一个**内部透传节点**，不出现在用户菜单里也没有实际计算意义。它唯一的
作用是被插入 ``H3LongVideo`` 展开的子图执行链中：当 ComfyUI 真正执行到子图里
这一刻，它把当前段号、阶段、提示词和一帧预览图通过 ``PromptServer`` 推给前端。

为什么需要它
------------
``H3LongVideo.run`` 是 GraphBuilder 展开节点：``run`` 方法在队列开始时**一次性**
把所有段的采样链都建好再返回。原本写在这里循环里的 ``send_sync`` 会在构建图
阶段瞬间全部发完，根本不反映真正的执行进度——前端只看到最后一段提示词一闪而
过，然后是漫长的空白等待。把这个信号挪到执行链里、由依赖关系驱动，前端拿到的
才是"此刻真正渲染到第几段、哪个阶段"。

它是透传节点（输入即输出，不拷贝 tensor），所以把它串进数据流不影响画面结果，
只保证执行顺序：上游先跑完，信号发出，下游再开始。
"""

import logging
import os
import time

logger = logging.getLogger(__name__)


def _save_preview_frame(images, run_id, segment_index, stage):
    """把一段画面的中间帧存成 temp 里的 PNG，返回文件名。

    temp 目录可覆盖写，同一 run/段/stage 反复执行会覆盖旧帧；前端 URL 自带时间
    戳破缓存，所以覆盖不会让浏览器继续显示旧图。
    """
    try:
        import torch  # noqa: F401  (确保环境里有 torch)
        from PIL import Image
        import numpy as np
        import folder_paths
    except Exception as exc:  # pragma: no cover - 环境缺失时降级
        logger.warning("H3ProgressSignal: 预览帧保存失败（缺依赖）: %s", exc)
        return None

    try:
        tensor = images
        if not hasattr(tensor, "shape") or len(tensor.shape) < 4:
            return None
        total = int(tensor.shape[0])
        if total <= 0:
            return None
        # 中间帧比首帧更能代表本段内容（首帧常是锚点过渡）
        mid = total // 2
        frame = tensor[mid].detach()
        if frame.is_cuda:
            frame = frame.cpu()
        arr = (frame.numpy() * 255.0).clip(0, 255).astype("uint8")
        img = Image.fromarray(arr)
        fname = "myh3_preview_%s_s%02d_%s.png" % (str(run_id)[:48],
                                                   int(segment_index),
                                                   str(stage))
        out_dir = folder_paths.get_temp_directory()
        os.makedirs(out_dir, exist_ok=True)
        img.save(os.path.join(out_dir, fname))
        return fname
    except Exception as exc:  # pragma: no cover - 预览失败不应阻断采样
        logger.warning("H3ProgressSignal: 预览帧保存异常: %s", exc)
        return None


class H3ProgressSignal:
    """透传 IMAGE/AUDIO，同时向前端广播一次执行进度。

    放在子图执行链的关键位置（采样后 / 漂移后 / 二采后 / 分段完成），它的
    ``FUNCTION`` 在 ComfyUI 真正执行到这一刻才被调用，所以发出的进度事件是
    实时的，而不是 ``run`` 构图阶段的一次性快闪。
    """

    CATEGORY = "沐阳 H3"
    FUNCTION = "signal"
    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("images", "audio")
    DESCRIPTION = "内部进度信号节点：透传画面/声音并广播当前段进度。请勿手动添加。"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "audio": ("AUDIO",),
                "segment_index": ("INT", {"default": 1, "min": 1, "max": 999}),
                "total_segments": ("INT", {"default": 1, "min": 1, "max": 999}),
                "stage": ("STRING", {"default": "sampled"}),
                "run_id": ("STRING", {"default": ""}),
                "owner_id": ("STRING", {"default": ""}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "brief": ("STRING", {"default": ""}),
                "save_preview": ("BOOLEAN", {"default": True}),
            },
        }

    def signal(self, images, audio, segment_index, total_segments, stage,
               run_id, owner_id, prompt, brief, save_preview):
        # 排查用：执行到这一行就证明 signal 节点真的被 ComfyUI 拉起了。
        # 若 ComfyUI 控制台见不到这条，说明节点没执行（多半是没被下游需要）。
        logger.info(
            "H3ProgressSignal 执行: seg=%s/%s stage=%s run=%s",
            segment_index, total_segments, stage, str(run_id or "")[:16])
        payload = {
            "run_id": str(run_id or ""),
            "owner_id": str(owner_id or ""),
            "segment_index": int(segment_index),
            "total_segments": int(total_segments),
            "stage": str(stage or ""),
            "prompt": str(prompt or ""),
            "brief": str(brief or ""),
        }
        if save_preview and images is not None:
            fname = _save_preview_frame(images, run_id, segment_index, stage)
            if fname:
                payload["preview_file"] = fname
                payload["preview_ts"] = int(time.time() * 1000)
        try:
            from server import PromptServer
            inst = getattr(PromptServer, "instance", None)
            if inst is not None and hasattr(inst, "send_sync"):
                inst.send_sync("myh3_progress", payload)
        except Exception as exc:  # pragma: no cover - 信号失败不能影响出图
            logger.debug("H3ProgressSignal: 事件发送失败: %s", exc)
        return (images, audio)


def _save_step_preview(x0, vae, run_id, segment_index, step, pass_label):
    """Decode one temporal token from x0 and save as a step-level preview frame."""
    try:
        from PIL import Image
        import numpy as np
        import folder_paths
    except Exception:
        return None

    try:
        if hasattr(x0, "unbind"):
            tensors = x0.unbind()
            video_latent = tensors[0]
        elif isinstance(x0, (tuple, list)):
            video_latent = x0[0]
        else:
            video_latent = x0

        if video_latent is None or video_latent.ndim != 5:
            return None

        T = int(video_latent.shape[2])
        if T <= 0:
            return None

        mid_t = T // 2
        single_token = video_latent[:, :, mid_t:mid_t + 1, :, :].clone()
        pixels = vae.decode(single_token)

        if pixels.ndim == 5:
            frame = pixels[0, 0]
        elif pixels.ndim == 4:
            frame = pixels[0]
        else:
            return None

        frame = frame.detach().cpu()
        arr = (frame.numpy() * 255.0).clip(0, 255).astype("uint8")
        img = Image.fromarray(arr)

        fname = "myh3_preview_%s_s%02d_%s_step%02d.png" % (
            str(run_id)[:48], int(segment_index), str(pass_label), int(step))
        out_dir = folder_paths.get_temp_directory()
        os.makedirs(out_dir, exist_ok=True)
        img.save(os.path.join(out_dir, fname))
        return fname
    except Exception as exc:
        logger.debug("H3Sampler: step preview failed: %s", exc)
        return None


class H3SamplerAdvanced:
    """SamplerCustomAdvanced replacement with step-level VAE preview.

    Instead of ComfyUI's latent2rgb blurry preview, each sampling step decodes
    one temporal token via VAE and pushes a clear frame to the frontend.
    """

    CATEGORY = "沐阳 H3"
    FUNCTION = "execute"
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("output",)
    DESCRIPTION = "内部采样器节点：逐步 VAE 解码预览。请勿手动添加。"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "noise": ("NOISE",),
                "guider": ("GUIDER",),
                "sampler": ("SAMPLER",),
                "sigmas": ("SIGMAS",),
                "latent_image": ("LATENT",),
                "vae": ("VAE",),
                "run_id": ("STRING", {"default": ""}),
                "owner_id": ("STRING", {"default": ""}),
                "segment_index": ("INT", {"default": 1, "min": 1, "max": 999}),
                "total_segments": ("INT", {"default": 1, "min": 1, "max": 999}),
                "pass_label": ("STRING", {"default": "sample1"}),
            },
        }

    def execute(self, noise, guider, sampler, sigmas, latent_image,
                vae, run_id, owner_id, segment_index, total_segments,
                pass_label="sample1"):
        import comfy
        import comfy.model_management

        latent = latent_image
        latent_samples = latent["samples"]
        latent = latent.copy()
        latent_samples = comfy.sample.fix_empty_latent_channels(
            guider.model_patcher, latent_samples,
            latent.get("downscale_ratio_spacial", None),
            latent.get("downscale_ratio_temporal", None))
        latent["samples"] = latent_samples

        noise_mask = latent.get("noise_mask", None)
        total_steps = sigmas.shape[-1] - 1

        def step_callback(step, x0, x, total):
            payload = {
                "run_id": str(run_id),
                "owner_id": str(owner_id or ""),
                "segment_index": int(segment_index),
                "total_segments": int(total_segments),
                "stage": "sampling",
                "pass_label": str(pass_label),
                "step": int(step + 1),
                "step_total": int(total),
            }
            if x0 is not None:
                preview_file = _save_step_preview(
                    x0, vae, run_id, segment_index, step, pass_label)
                if preview_file:
                    payload["preview_file"] = preview_file
                    payload["preview_ts"] = int(time.time() * 1000)
            try:
                from server import PromptServer
                inst = getattr(PromptServer, "instance", None)
                if inst is not None and hasattr(inst, "send_sync"):
                    inst.send_sync("myh3_progress", payload)
            except Exception:
                pass

        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
        samples = guider.sample(
            noise.generate_noise(latent), latent_samples, sampler, sigmas,
            denoise_mask=noise_mask, callback=step_callback,
            disable_pbar=disable_pbar, seed=noise.seed)
        samples = samples.to(comfy.model_management.intermediate_device())

        out = latent.copy()
        out.pop("downscale_ratio_spacial", None)
        out.pop("downscale_ratio_temporal", None)
        out["samples"] = samples
        return (out,)


NODE_CLASS_MAPPINGS = {
    "H3ProgressSignal": H3ProgressSignal,
    "H3SamplerAdvanced": H3SamplerAdvanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3ProgressSignal": "沐阳 H3 · 进度信号（内部）",
    "H3SamplerAdvanced": "沐阳 H3 · 采样器（内部·步级预览）",
}
