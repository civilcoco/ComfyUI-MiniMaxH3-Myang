"""MiniMax H3 audio refinement and duration-preserving seam smoothing.

The masked refinement implementation is adapted from ComfyUI-H3-AudioRefine
at revision d0ed019b6f1c4ceb0caf3d69c502a313d1f6da9d (MIT, Copyright (c)
2026 Adudeguyman).  Myang integrates the exact, uncached path only and adds a
separate long-video boundary processor.  See THIRD_PARTY_NOTICES.md.
"""

import logging

import torch
import torch.nn.functional as F

import comfy.nested_tensor
import comfy.sample
import comfy.samplers
import comfy.utils
import latent_preview


logger = logging.getLogger(__name__)


def _require_av_nested(samples):
    """Validate an H3 packed AV latent and return its video/audio streams."""
    if not getattr(samples, "is_nested", False):
        raise ValueError(
            "H3 音频精修需要一采完成后的 MiniMax H3 联合音画 latent，"
            "当前收到的是普通 latent")
    streams = samples.unbind()
    if len(streams) < 2:
        raise ValueError("H3 音频精修需要视频、音频两个 latent 流")
    video, audio = streams[0], streams[1]
    if video.ndim != 5 or audio.ndim != 4:
        raise ValueError(
            "H3 音频精修收到异常 latent 形状：video=%s audio=%s" %
            (list(video.shape), list(audio.shape)))
    return video, audio


def _build_av_noise_mask(video, audio, video_denoise=0.0):
    """Build native per-stream masks: frozen video, generated audio."""
    video_mask = torch.full(
        (1, 1, video.shape[2], video.shape[3], video.shape[4]),
        float(video_denoise), dtype=torch.float32)
    audio_mask = torch.ones(
        (1, 1, audio.shape[2], audio.shape[3]), dtype=torch.float32)
    return comfy.nested_tensor.NestedTensor((video_mask, audio_mask))


def _send_progress(run_id, owner_id, segment_index, total_segments,
                   step, total):
    try:
        from server import PromptServer
        server = getattr(PromptServer, "instance", None)
        if server is not None and hasattr(server, "send_sync"):
            server.send_sync("myh3_progress", {
                "run_id": str(run_id or ""),
                "owner_id": str(owner_id or ""),
                "segment_index": int(segment_index),
                "total_segments": int(total_segments),
                "stage": "sampling",
                "pass_label": "audio_refine",
                "step": int(step),
                "step_total": int(total),
            })
    except Exception:
        # Progress is diagnostic UI only and must never stop a render.
        pass


class H3AudioRefineMask:
    CATEGORY = "沐阳 H3/音频"
    FUNCTION = "apply"
    RETURN_TYPES = ("LATENT",)
    DESCRIPTION = "冻结 H3 视频 latent，只开放音频流供后续采样精修。"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"latent": ("LATENT",)}, "optional": {
            "video_denoise": ("FLOAT", {
                "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
        }}

    def apply(self, latent, video_denoise=0.0):
        video, audio = _require_av_nested(latent["samples"])
        out = latent.copy()
        out["noise_mask"] = _build_av_noise_mask(
            video, audio, video_denoise)
        return (out,)


class H3AudioRefineSampler:
    """Exact native masked refinement; no large frozen-video cache."""

    CATEGORY = "沐阳 H3/音频"
    FUNCTION = "refine"
    RETURN_TYPES = ("LATENT",)
    DESCRIPTION = (
        "用基模附加去噪 H3 音频流，视频流完全冻结。每一步仍接近一次完整 H3 "
        "前向计算；本节点不启用大内存缓存。")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "positive": ("CONDITIONING",),
            "negative": ("CONDITIONING",),
            "latent": ("LATENT",),
            "seed": ("INT", {
                "default": 0, "min": 0, "max": 0xffffffffffffffff}),
            "steps": ("INT", {"default": 4, "min": 1, "max": 100}),
            "cfg": ("FLOAT", {
                "default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1}),
            "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {
                "default": "euler"}),
            "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {
                "default": "simple"}),
            "audio_denoise": ("FLOAT", {
                "default": 0.5, "min": 0.01, "max": 1.0, "step": 0.01}),
        }, "optional": {
            "video_denoise": ("FLOAT", {
                "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            "run_id": ("STRING", {"default": ""}),
            "owner_id": ("STRING", {"default": ""}),
            "segment_index": ("INT", {"default": 1, "min": 1, "max": 999}),
            "total_segments": ("INT", {"default": 1, "min": 1, "max": 999}),
        }}

    def refine(self, model, positive, negative, latent, seed, steps, cfg,
               sampler_name, scheduler, audio_denoise, video_denoise=0.0,
               run_id="", owner_id="", segment_index=1, total_segments=1):
        video, audio = _require_av_nested(latent["samples"])
        latent_image = latent["samples"]
        noise_mask = _build_av_noise_mask(video, audio, video_denoise)
        noise = comfy.sample.prepare_noise(
            latent_image, seed, latent.get("batch_index"))
        preview_callback = latent_preview.prepare_callback(model, steps)

        def callback(step, x0, x, total):
            preview_callback(step, x0, x, total)
            _send_progress(run_id, owner_id, segment_index, total_segments,
                           int(step) + 1, total)

        samples = comfy.sample.sample(
            model, noise, steps, cfg, sampler_name, scheduler,
            positive, negative, latent_image,
            denoise=audio_denoise, noise_mask=noise_mask,
            callback=callback,
            disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED, seed=seed)
        out = latent.copy()
        out.pop("noise_mask", None)
        out["samples"] = samples
        return (out,)


class H3AudioSettings:
    """Transport Director audio options through one stable graph socket."""

    CATEGORY = "沐阳 H3/导演台"
    FUNCTION = "build"
    RETURN_TYPES = ("MYANG_H3_AUDIO",)
    RETURN_NAMES = ("音频设置",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "refine_enabled": ("BOOLEAN", {"default": False}),
            "refine_steps": ("INT", {"default": 4, "min": 1, "max": 100}),
            "refine_denoise": ("FLOAT", {
                "default": 0.5, "min": 0.01, "max": 1.0, "step": 0.01}),
            "refine_sampler": (comfy.samplers.KSampler.SAMPLERS, {
                "default": "euler"}),
            "refine_scheduler": (comfy.samplers.KSampler.SCHEDULERS, {
                "default": "simple"}),
            "seam_enabled": ("BOOLEAN", {"default": True}),
            "seam_ms": ("FLOAT", {
                "default": 80.0, "min": 0.0, "max": 500.0, "step": 5.0}),
        }, "optional": {"model": ("MODEL",)}}

    def build(self, refine_enabled, refine_steps, refine_denoise,
              refine_sampler, refine_scheduler, seam_enabled, seam_ms,
              model=None):
        return ({
            "refine_enabled": bool(refine_enabled),
            "steps": int(refine_steps),
            "denoise": float(refine_denoise),
            "sampler_name": str(refine_sampler),
            "scheduler": str(refine_scheduler),
            "seam_enabled": bool(seam_enabled),
            "seam_ms": float(seam_ms),
            "model": model,
        },)


def _resample_waveform(waveform, source_rate, target_rate):
    if int(source_rate) == int(target_rate):
        return waveform
    samples = max(1, int(round(
        int(waveform.shape[-1]) * float(target_rate) / float(source_rate))))
    shape = waveform.shape
    flat = waveform.reshape(-1, 1, shape[-1]).to(torch.float32)
    result = F.interpolate(flat, size=samples, mode="linear", align_corners=False)
    return result.reshape(*shape[:-1], samples).to(waveform.dtype)


def _match_audio_shape(previous, current):
    """Make batches/channels compatible without pulling in an audio package."""
    pb, pc = int(previous.shape[0]), int(previous.shape[1])
    cb, cc = int(current.shape[0]), int(current.shape[1])
    if pb != cb:
        if cb == 1:
            current = current.repeat(pb, 1, 1)
        elif pb == 1:
            previous = previous.repeat(cb, 1, 1)
        else:
            raise ValueError("H3 音频接缝的 batch 数不一致")
    if pc != cc:
        if cc == 1:
            current = current.repeat(1, pc, 1)
        elif pc == 1:
            previous = previous.repeat(1, cc, 1)
        else:
            raise ValueError("H3 音频接缝的声道数不一致")
    return previous, current


class H3AudioSeam:
    """Rewrite only the previous tail, then trim the duplicated anchor audio."""

    CATEGORY = "沐阳 H3/音频"
    FUNCTION = "smooth"
    RETURN_TYPES = ("AUDIO", "AUDIO", "STRING")
    RETURN_NAMES = ("previous_audio", "next_audio", "report")
    DESCRIPTION = (
        "利用新段被裁掉的锚点音频，把上一段尾部平滑过渡到新段；不增加重复声音，"
        "不改变成片时长或画面对齐。")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "previous_audio": ("AUDIO",),
            "next_audio": ("AUDIO",),
            "next_images": ("IMAGE",),
            "trim_frames": ("INT", {"default": 22, "min": 0, "max": 4096}),
            "fade_ms": ("FLOAT", {
                "default": 80.0, "min": 0.0, "max": 500.0, "step": 5.0}),
            "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0}),
        }}

    def smooth(self, previous_audio, next_audio, next_images, trim_frames,
               fade_ms=80.0, fps=24.0):
        previous = previous_audio.get("waveform")
        current = next_audio.get("waveform")
        previous_rate = int(previous_audio.get("sample_rate") or 32000)
        current_rate = int(next_audio.get("sample_rate") or previous_rate)
        if previous is None or current is None:
            raise ValueError("H3 音频接缝没有收到有效 waveform")
        current = _resample_waveform(current, current_rate, previous_rate)
        previous, current = _match_audio_shape(previous, current)
        current = current.to(device=previous.device, dtype=previous.dtype)

        trim = max(0, min(int(trim_frames), int(next_images.shape[0]) - 1))
        cut = min(int(current.shape[-1]), int(round(
            trim / max(float(fps), 1.0) * previous_rate)))
        wanted_frames = max(0, int(next_images.shape[0]) - trim)
        wanted = int(round(wanted_frames / max(float(fps), 1.0) * previous_rate))
        current_out = current[..., cut:cut + wanted]
        if int(current_out.shape[-1]) < wanted:
            current_out = F.pad(current_out, (0, wanted - int(current_out.shape[-1])))

        fade = min(
            int(round(max(0.0, float(fade_ms)) / 1000.0 * previous_rate)),
            cut, int(previous.shape[-1]))
        previous_out = previous
        if fade > 0:
            # cut-fade:cut is the duplicate anchor audio immediately before the
            # exact visual trim point.  Rewriting the old tail against it removes
            # clicks and ambience jumps without repeating or shifting samples.
            t = torch.linspace(
                0.0, 1.0, fade, device=previous.device,
                dtype=torch.float32).reshape(1, 1, -1)
            weight = t * t * (3.0 - 2.0 * t)
            old_tail = previous[..., -fade:].to(torch.float32)
            anchor_tail = current[..., cut - fade:cut].to(torch.float32)
            # A conservative level match avoids a volume step while clamping
            # protects transients from aggressive automatic gain changes.
            eps = 1e-6
            old_rms = old_tail.square().mean(dim=-1, keepdim=True).sqrt()
            new_rms = anchor_tail.square().mean(dim=-1, keepdim=True).sqrt()
            gain = (old_rms / (new_rms + eps)).clamp(0.75, 1.33)
            anchor_tail = anchor_tail * gain
            mixed = old_tail * (1.0 - weight) + anchor_tail * weight
            previous_out = torch.cat(
                [previous[..., :-fade], mixed.to(previous.dtype)], dim=-1)

            # Begin the retained part at the same matched level, then relax to
            # unity. This prevents a second loudness edge exactly at the cut.
            relax = min(fade, int(current_out.shape[-1]))
            if relax > 0:
                ramp = torch.linspace(
                    0.0, 1.0, relax, device=previous.device,
                    dtype=torch.float32).reshape(1, 1, -1)
                correction = gain * (1.0 - ramp) + ramp
                current_out = current_out.clone()
                current_out[..., :relax] = (
                    current_out[..., :relax].to(torch.float32) * correction
                ).to(current_out.dtype)

        report = "音频接缝：%d ms（%d samples），总时长不变" % (
            int(round(float(fade_ms))), fade)
        logger.info("H3-Myang: %s", report)
        return (
            {"waveform": previous_out, "sample_rate": previous_rate},
            {"waveform": current_out, "sample_rate": previous_rate},
            report,
        )


NODE_CLASS_MAPPINGS = {
    "H3AudioRefineMask": H3AudioRefineMask,
    "H3AudioRefineSampler": H3AudioRefineSampler,
    "H3AudioSettings": H3AudioSettings,
    "H3AudioSeam": H3AudioSeam,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3AudioRefineMask": "沐阳 H3 · 音频精修遮罩",
    "H3AudioRefineSampler": "沐阳 H3 · 音频精修采样",
    "H3AudioSettings": "沐阳 H3 · 导演台音频设置",
    "H3AudioSeam": "沐阳 H3 · 音频接缝平滑",
}
