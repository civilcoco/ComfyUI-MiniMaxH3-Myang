"""接缝交叉淡化 —— 把硬切换成一段过渡。

The pinned head of a segment and the tail of the one before it depict the same
moments: that is what the anchors are for. The join therefore throws the pinned
head away and butts the two clips together, which leaves a visible pop wherever
the two renders disagree slightly -- a touch of colour, a little more grain, a
detail that resolved differently.

Both versions exist, so instead of discarding one, cross-fade them. The previous
clip's tail is rewritten as a ramp from itself into the new clip's pinned head,
and the new clip still starts after the trim. Nothing is duplicated, no frame is
added or lost, and the handoff into the new segment's free-running content
becomes continuous instead of instantaneous.

Sitting on top of drift correction rather than replacing it: correction removes
the accumulated bias, this removes the step at the boundary.
"""

import logging

import torch

logger = logging.getLogger(__name__)

CURVE_SMOOTH = "平滑（推荐）"
CURVE_LINEAR = "线性"
CURVES = [CURVE_SMOOTH, CURVE_LINEAR]
AUDIO_FADE_MS = 20.0


def _weights(n: int, curve: str, device, dtype) -> torch.Tensor:
    """0 -> 1 across n samples, including both endpoints."""
    if n <= 1:
        return torch.ones(max(n, 0), device=device, dtype=dtype)
    t = torch.linspace(0.0, 1.0, n, device=device, dtype=dtype)
    if str(curve) == CURVE_LINEAR:
        return t
    # smoothstep: zero slope at both ends, so the fade has no visible corner
    return t * t * (3.0 - 2.0 * t)


def _blend_images(prev, head, curve):
    n = int(head.shape[0])
    w = _weights(n, curve, prev.device, torch.float32).view(-1, 1, 1, 1)
    tail = prev[-n:].to(torch.float32)
    head_aligned = head[:n].to(torch.float32)
    # Compensate for micro luminance/chroma drift across the blend window to eliminate exposure steps
    diff_mean = tail.mean(dim=(1, 2, 3), keepdim=True) - head_aligned.mean(dim=(1, 2, 3), keepdim=True)
    head_comp = head_aligned + diff_mean * (1.0 - w) * 0.5
    mixed = tail * (1.0 - w) + head_comp * w
    return torch.cat([prev[:-n], mixed.clamp(0.0, 1.0).to(prev.dtype)], dim=0)


def _blend_audio(prev, head, cut, curve, fade_ms=AUDIO_FADE_MS):
    """Blend the matching waveform immediately before the exact trim point."""
    if prev is None or head is None:
        return prev
    pw, hw = prev.get("waveform"), head.get("waveform")
    rate = int(prev.get("sample_rate") or 32000)
    if pw is None or hw is None or int(head.get("sample_rate") or rate) != rate:
        return prev
    cut = max(0, min(int(cut), int(hw.shape[-1])))
    n = min(int(round(float(fade_ms) / 1000.0 * rate)),
            cut, int(pw.shape[-1]))
    if n <= 0:
        return prev
    w = _weights(n, curve, pw.device, torch.float32)
    tail = pw[..., -n:].to(torch.float32)
    # The duplicated window ends at ``cut``.  Taking its beginning here repeats
    # older music by (trim - fade) frames; the matching samples are cut-n:cut.
    mixed = tail * (1.0 - w) + hw[..., cut - n:cut].to(torch.float32) * w
    return {"waveform": torch.cat([pw[..., :-n], mixed.to(pw.dtype)], dim=-1),
            "sample_rate": rate}


class H3SeamBlend:
    CATEGORY = "沐阳 H3"
    FUNCTION = "join"
    RETURN_TYPES = ("IMAGE", "AUDIO", "IMAGE", "AUDIO", "STRING")
    RETURN_NAMES = ("prev_images", "prev_audio", "next_images", "next_audio", "report")
    DESCRIPTION = ("把段间的硬切换成交叉淡化。上一段的尾巴改写成"
                   "「自己 → 新段钉入帧」的渐变，新段照常裁掉钉入的部分。"
                   "总帧数不变，也不会重复画面。")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prev_images": ("IMAGE", {"tooltip": "上一段（已在成片里的那份）"}),
                "next_images": ("IMAGE", {"tooltip": "新段，未裁剪，开头还带着钉入的帧"}),
                "trim_frames": ("INT", {"default": 22, "min": 0, "max": 240,
                                        "tooltip": "钉入了多少帧，接 H3AnchorContext 的输出"}),
                "blend_frames": ("INT", {"default": 8, "min": 0, "max": 240,
                                         "tooltip": "用其中多少帧做淡化。0 = 关闭，退回硬切。"
                                                    "超过钉入帧数会被自动收窄"}),
                "curve": (CURVES, {"default": CURVE_SMOOTH}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0}),
            },
            "optional": {
                "prev_audio": ("AUDIO",),
                "next_audio": ("AUDIO",),
            },
        }

    def join(self, prev_images, next_images, trim_frames, blend_frames, curve, fps,
             prev_audio=None, next_audio=None):
        trim = max(0, int(trim_frames))
        head = next_images[:trim]
        tail_out = next_images[trim:]
        audio_out = next_audio

        if next_audio is not None and trim > 0:
            wf = next_audio.get("waveform")
            rate = int(next_audio.get("sample_rate") or 32000)
            if wf is not None:
                cut = min(int(wf.shape[-1]), int(round(trim / max(fps, 1.0) * rate)))
                wanted = int(round(int(tail_out.shape[0]) / max(fps, 1.0) * rate))
                audio_out = {"waveform": wf[..., cut:cut + wanted],
                             "sample_rate": rate}

        n = min(int(blend_frames), trim, int(prev_images.shape[0]), int(head.shape[0]))
        if n <= 0:
            report = "seam: 硬切（未淡化）"
            return (prev_images, prev_audio, tail_out, audio_out, report)

        if prev_images.shape[1:] != head.shape[1:]:
            raise ValueError("H3-Myang: 两段画面尺寸不同，无法交叉淡化")

        before = float((prev_images[-1].to(torch.float32)
                        - head[min(n, head.shape[0]) - 1].to(torch.float32)).abs().mean())
        # Only the last n pinned frames correspond to the previous clip's last
        # n frames.  Blending the whole head would rewrite all trim frames.
        prev_out = _blend_images(prev_images, head[-n:], curve)
        after = float((prev_out[-1].to(torch.float32)
                       - tail_out[0].to(torch.float32)).abs().mean()) if tail_out.shape[0] else 0.0
        audio_cut = 0
        if next_audio is not None:
            rate = int(next_audio.get("sample_rate") or 32000)
            audio_cut = int(round(trim / max(fps, 1.0) * rate))
        prev_audio_out = _blend_audio(
            prev_audio, next_audio, audio_cut, curve)

        report = ("seam: %d 帧 %s 淡化 | 接缝处逐像素差 %.4f -> %.4f"
                  % (n, str(curve).split("（")[0], before, after))
        logger.info("H3-Myang: %s", report)
        return (prev_out, prev_audio_out, tail_out, audio_out, report)


NODE_CLASS_MAPPINGS = {"H3SeamBlend": H3SeamBlend}
NODE_DISPLAY_NAME_MAPPINGS = {"H3SeamBlend": "H3 接缝淡化"}
