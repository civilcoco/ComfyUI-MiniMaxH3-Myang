"""Drift correction for chained MiniMax H3 clips.

Autoregressive extension conditions each clip on the previous clip's *generated*
output, so any bias in the model compounds: measured across four chained clips,
mean saturation climbed 25.1% -> 36.0% while mean brightness fell ~108 -> 91.9.

The fix is to renormalise every clip back onto a fixed anchor before it is fed
forward, which turns compounding drift into bounded drift.

One transform is fitted per clip and applied to every frame of it. Per-frame
matching would track the anchor more tightly but introduce temporal flicker,
which is worse than the drift it removes.
"""

import logging

import torch

logger = logging.getLogger(__name__)

METHOD_MKL = "mkl (推荐·全协方差)"
METHOD_MEANSTD = "mean_std (逐通道)"
METHOD_OFF = "off"
METHODS = [METHOD_MKL, METHOD_MEANSTD, METHOD_OFF]

ALIGN_SEAM = "衔接处（推荐）"
ALIGN_WHOLE = "整段对齐锚点"
ALIGNMENTS = [ALIGN_SEAM, ALIGN_WHOLE]


def _sample_pixels(images: torch.Tensor, max_frames: int, max_pixels: int,
                   window: str = "all") -> torch.Tensor:
    """Flatten a subset of frames to [N, 3] float64.

    `window` picks *which* frames: "head" and "tail" take the frames either side
    of a cut, "all" spreads evenly over the clip.
    """
    n = images.shape[0]
    if window == "head":
        images = images[:max(1, min(max_frames, n))]
    elif window == "tail":
        images = images[-max(1, min(max_frames, n)):]
    elif n > max_frames:
        idx = torch.linspace(0, n - 1, max_frames).round().long()
        images = images[idx]
    flat = images[..., :3].reshape(-1, 3).to(torch.float64)
    if flat.shape[0] > max_pixels:
        step = flat.shape[0] // max_pixels
        flat = flat[::step][:max_pixels]
    return flat


def _sqrtm_psd(mat: torch.Tensor) -> torch.Tensor:
    """Symmetric PSD matrix square root via eigendecomposition."""
    vals, vecs = torch.linalg.eigh(mat)
    vals = vals.clamp_min(0.0).sqrt()
    return (vecs * vals.unsqueeze(0)) @ vecs.transpose(-1, -2)


def _invsqrtm_psd(mat: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    vals, vecs = torch.linalg.eigh(mat)
    vals = vals.clamp_min(eps).rsqrt()
    return (vecs * vals.unsqueeze(0)) @ vecs.transpose(-1, -2)


def _mkl_transform(src: torch.Tensor, ref: torch.Tensor):
    """Monge-Kantorovich linear colour transport (Pitie & Kokaram).

    Matches the full 3x3 colour covariance, not just per-channel spread, so a
    cast that lives in the channel *correlations* — which is what the H3 chain
    actually accumulates — is corrected rather than merely rescaled.
    """
    mu_s, mu_r = src.mean(0), ref.mean(0)
    a = torch.cov(src.T) + torch.eye(3, dtype=src.dtype) * 1e-8
    b = torch.cov(ref.T) + torch.eye(3, dtype=ref.dtype) * 1e-8
    a_half = _sqrtm_psd(a)
    a_inv_half = _invsqrtm_psd(a)
    middle = _sqrtm_psd(a_half @ b @ a_half)
    t = a_inv_half @ middle @ a_inv_half
    return mu_s, mu_r, t


def _meanstd_transform(src: torch.Tensor, ref: torch.Tensor):
    mu_s, mu_r = src.mean(0), ref.mean(0)
    scale = ref.std(0).clamp_min(1e-8) / src.std(0).clamp_min(1e-8)
    return mu_s, mu_r, torch.diag(scale)


def _stats(images: torch.Tensor) -> tuple[float, float]:
    """Mean saturation (%) and mean brightness (0-255), matching common tools."""
    rgb = images[..., :3].to(torch.float32)
    mx = rgb.max(dim=-1).values
    mn = rgb.min(dim=-1).values
    sat = torch.where(mx > 1e-6, (mx - mn) / mx.clamp_min(1e-6), torch.zeros_like(mx))
    lum = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    return float(sat.mean()) * 100.0, float(lum.mean()) * 255.0


class H3DriftCorrect:
    CATEGORY = "沐阳 H3"
    FUNCTION = "correct"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "report")
    DESCRIPTION = ("把本段画面的色彩统计对齐到锚点段，抵消链式续写的累积漂移。"
                   "整段拟合一个变换后统一应用，不会引入逐帧闪烁。")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "anchor": ("IMAGE", {"tooltip": "对齐目标。衔接处模式接上一段，整段模式接第 1 段"}),
                "method": (METHODS, {"default": METHOD_MEANSTD}),
                "align": (ALIGNMENTS, {
                    "default": ALIGN_SEAM,
                    "tooltip": "衔接处：只比对切口两侧的少量帧——那里内容几乎相同，"
                               "差异就是漂移本身，镜头和光线的真实变化不会被压掉。\n"
                               "整段：把整段的色彩分布拉成锚点段的样子，"
                               "场景一变就会被改得很重"}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                                       "tooltip": "1.0 完全对齐；0.5 只走一半，"
                                                  "想保留一点段间自然变化时用"}),
                "sample_frames": ("INT", {"default": 8, "min": 1, "max": 512,
                                          "tooltip": "衔接处模式下比对切口两侧各多少帧"}),
            },
        }

    def correct(self, images, anchor, method, strength, sample_frames, align=ALIGN_SEAM):
        before = _stats(images)
        if str(method) == METHOD_OFF or strength <= 0.0:
            report = "drift_correct: off | 饱和度 %.1f%% 亮度 %.1f" % before
            return (images, report)

        # Fitting on the seam is what keeps this a *drift* correction rather
        # than a palette transplant: the last frames of the previous clip and
        # the first frames of this one show almost the same moment, so whatever
        # differs between them is the model's accumulated bias. Comparing whole
        # clips instead would read an intended change of scene as drift and
        # undo it.
        seam = str(align) != ALIGN_WHOLE
        src = _sample_pixels(images, sample_frames, 400_000, "head" if seam else "all")
        ref = _sample_pixels(anchor, sample_frames, 400_000, "tail" if seam else "all")
        if str(method) == METHOD_MKL:
            mu_s, mu_r, t = _mkl_transform(src, ref)
        else:
            mu_s, mu_r, t = _meanstd_transform(src, ref)

        dtype, device = images.dtype, images.device
        t = t.to(torch.float32).to(device)
        mu_s = mu_s.to(torch.float32).to(device)
        mu_r = mu_r.to(torch.float32).to(device)

        rgb = images[..., :3].to(torch.float32)
        flat = rgb.reshape(-1, 3)
        moved = (flat - mu_s) @ t + mu_r
        if strength < 1.0:
            moved = flat + (moved - flat) * float(strength)
        out = moved.clamp(0.0, 1.0).reshape(rgb.shape).to(dtype)
        if images.shape[-1] > 3:
            out = torch.cat([out, images[..., 3:]], dim=-1)

        after = _stats(out)
        report = ("drift_correct: %s strength=%.2f\n"
                  "  饱和度 %.1f%% -> %.1f%% (锚点 %.1f%%)\n"
                  "  亮度   %.1f  -> %.1f  (锚点 %.1f)"
                  % (str(method).split(" ")[0], strength,
                     before[0], after[0], _stats(anchor)[0],
                     before[1], after[1], _stats(anchor)[1]))
        logger.info("H3-Myang: %s", report.replace("\n", " "))
        return (out, report)


NODE_CLASS_MAPPINGS = {"H3DriftCorrect": H3DriftCorrect}
NODE_DISPLAY_NAME_MAPPINGS = {"H3DriftCorrect": "沐阳 H3 · 段间漂移校正"}
