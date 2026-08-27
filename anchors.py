"""Native MiniMax H3 temporal anchors for ComfyUI-MiniMaxH3-Myang.

SPDX-License-Identifier: GPL-3.0-only

Copyright (C) 2026 NikoDemon80
Copyright (C) 2026 Myang
Modified by Myang on 2026-08-26.

Portions of the layout/payload, temporal-latent and synchronized-trim logic
are adapted from NikoDemon80/ComfyUI-H3-Motion-Context (GPL-3.0).  Myang's
changes add arbitrary-position anchors, marker-gated composition, seam
scheduling and the long-video integration.  See THIRD_PARTY_NOTICES.md.

The stock H3 layout accepts only the first and last output frame as a
keyframe.  Myang keeps the stock constructor, lets it allocate the rows, then
moves only marked keyframe rows onto the target video's timeline.  This makes
arbitrary-position, multi-keyframe conditioning possible without copying the
model's layout implementation.

The patch is lazy and marker-gated: installing this module does not alter a
normal H3 graph.  It also fixes the upstream keyframe + reference payload
collision only for Myang-marked graphs.
"""

import logging
import math

import torch
import comfy.utils
import node_helpers


logger = logging.getLogger(__name__)

ANCHOR_FRAME = "myang_h3_anchor_frame_v2"
# Rides on the keyframe itself rather than arriving as a separate argument.
# comfy/model_base.py builds PackedLayout with only `keyframes` and `refs`, and
# it builds it *inside* extra_conds -- so a payload key written after that call
# is always too late, and there is no parameter to pass the total through.
ANCHOR_TOTAL = "myang_h3_anchor_total_v2"
AUDIO_END_FRAME = "myang_h3_audio_end_frame_v2"
LAYOUT_PATCH = "_myang_h3_anchor_layout_v2"
PAYLOAD_PATCH = "_myang_h3_anchor_payload_v2"

FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
FRAME_RESCALE = 5.0 / 3.0
FPS = 24.0
AUDIO_HZ = 40.0
CONTEXT_LENGTHS = ["22", "5", "39", "56"]

_layout_original = None
_payload_original = None


def pixel_frames(latent_t):
    return sum(FRAME_PER_TOKEN[k % 5] for k in range(int(latent_t)))


def step_offsets(latent_t):
    offsets, cursor = [], 0
    for k in range(int(latent_t)):
        offsets.append(cursor)
        cursor += FRAME_PER_TOKEN[k % 5]
    return offsets


def steps_for_frames(frame_count):
    covered = 0
    for steps in range(1, int(frame_count) + 1):
        covered += FRAME_PER_TOKEN[(steps - 1) % 5]
        if covered >= int(frame_count):
            return steps if covered == int(frame_count) else None
    return None


def streams(latent):
    samples = latent["samples"]
    if hasattr(samples, "unbind"):
        return list(samples.unbind())
    if isinstance(samples, (tuple, list)):
        return list(samples)
    raise ValueError("H3-Myang: 需要 MiniMax H3 的画面+声音 latent")


def video_stream(latent):
    parts = streams(latent)
    if not parts:
        raise ValueError("H3-Myang: H3 latent 里没有画面流")
    video = parts[0]
    if video.ndim == 4:
        video = video.unsqueeze(0)
    if video.ndim != 5:
        raise ValueError(
            "H3-Myang: 画面 latent 应为 [B,C,T,H,W]，实际 %s" %
            (tuple(video.shape),))
    return video


def audio_stream(latent):
    parts = streams(latent)
    if len(parts) < 2:
        return None
    audio = parts[1]
    if audio.ndim == 3:
        audio = audio.unsqueeze(0)
    if audio.ndim != 4:
        raise ValueError(
            "H3-Myang: 声音 latent 应为 [B,C,2,T]，实际 %s" %
            (tuple(audio.shape),))
    return audio


def target_origin(layout):
    segments = getattr(layout, "segments", None) or ()
    if not segments or segments[-1][2] != "video":
        raise RuntimeError("H3-Myang: PackedLayout 的最后一段不再是目标 video")
    return float(layout.position_ids[int(segments[-1][0]), 0])


def _has_anchor(keyframes):
    return bool(keyframes) and any(ANCHOR_FRAME in item for item in keyframes)


def _expected_ref_segments(ref):
    kind = ref.get("kind")
    if kind == "image":
        return ("ref_img",)
    if kind == "audio":
        return ("ref_audio",) if int(ref.get("ref_audio_t", 0)) > 0 else ()
    if kind in ("video", "video_audio"):
        if int(ref.get("ref_audio_t", 0)) > 0:
            return ("ref_audio", "ref_img")
        return ("ref_img",)
    raise RuntimeError("H3-Myang: 未知参考素材类型 %r" % kind)


def _ref_segment_map(layout, refs):
    actual = [(a, b, kind) for a, b, kind in layout.segments
              if kind in ("ref_img", "ref_audio")]
    expected = [(index, kind) for index, ref in enumerate(refs or ())
                for kind in _expected_ref_segments(ref)]
    if len(actual) != len(expected):
        raise RuntimeError(
            "H3-Myang: 参考素材应产生 %d 段，实际 %d 段" %
            (len(expected), len(actual)))
    result = {}
    for (index, want), (start, stop, got) in zip(expected, actual):
        if want != got:
            raise RuntimeError(
                "H3-Myang: 第 %d 个参考素材应产生 %s，实际 %s" %
                (index + 1, want, got))
        result.setdefault(index, {})[got] = (start, stop)
    return result


def _move_audio_anchor(layout, refs):
    marked = [i for i, ref in enumerate(refs or ())
              if AUDIO_END_FRAME in ref]
    if not marked:
        return
    if len(marked) != 1:
        raise RuntimeError("H3-Myang: 一次续接只能有一个声音锚点")
    index = marked[0]
    ref = refs[index]
    if ref.get("kind") != "audio":
        raise RuntimeError("H3-Myang: 声音时间标记只能用于 audio reference")
    audio_t = int(ref.get("ref_audio_t", 0))
    if audio_t <= 0:
        return
    segment = _ref_segment_map(layout, refs).get(index, {}).get("ref_audio")
    if segment is None:
        raise RuntimeError("H3-Myang: 标记的声音参考没有产生 ref_audio 段")
    start, stop = segment
    if stop - start != audio_t * 2:
        raise RuntimeError(
            "H3-Myang: %d 个声音 step 应占 %d 行，实际 %d" %
            (audio_t, audio_t * 2, stop - start))
    current_start = float(layout.position_ids[start, 0])
    desired_end = (target_origin(layout)
                   + FRAME_RESCALE * float(ref[AUDIO_END_FRAME]))
    desired_start = desired_end - float(audio_t)
    layout.position_ids[start:stop, 0] += desired_start - current_start


def _call_stock_layout(stock, obj, text_len, latent_t, latent_h, latent_w, audio_t,
                       keyframes=None, refs=None, frame_count=None):
    try:
        stock(obj, text_len, latent_t, latent_h, latent_w, audio_t,
              keyframes=keyframes, refs=refs, frame_count=frame_count)
    except TypeError:
        stock(obj, text_len, latent_t, latent_h, latent_w, audio_t,
              keyframes=keyframes, refs=refs)


def _patched_layout(self, text_len, latent_t, latent_h, latent_w, audio_t,
                    keyframes=None, refs=None, frame_count=None):
    has_anchor = _has_anchor(keyframes)
    if has_anchor and refs and any(
            ANCHOR_FRAME not in keyframe for keyframe in keyframes):
        raise ValueError(
            "H3-Myang: 有参考素材时不能混用普通关键帧与 Myang 任意位置锚点")

    safe_keyframes = keyframes
    if has_anchor:
        safe_keyframes = []
        for keyframe in keyframes:
            item = keyframe.copy()
            if ANCHOR_FRAME in item:
                item["resolved_frame_index"] = 0
            safe_keyframes.append(item)

    _call_stock_layout(
        _layout_original, self, text_len, latent_t, latent_h, latent_w, audio_t,
        keyframes=safe_keyframes, refs=refs, frame_count=frame_count)

    if has_anchor:
        if frame_count is None:
            # Recover it from the anchors: nothing upstream can hand it over as
            # an argument, so H3AnchorContext stamps it onto every anchor.
            totals = {int(k[ANCHOR_TOTAL]) for k in keyframes if ANCHOR_TOTAL in k}
            if len(totals) == 1:
                frame_count = totals.pop()
        if frame_count is None:
            raise ValueError(
                "H3-Myang: 任意位置锚点需要目标总帧数，但锚点上没有记录。"
                "请让 H3AnchorContext 生成锚点（它会写入总帧数），"
                "不要手工拼 minimax_keyframes。")
        cond_segments = [(a, b) for a, b, kind in self.segments
                         if kind == "cond"]
        if len(cond_segments) != len(keyframes):
            raise RuntimeError(
                "H3-Myang: %d 个锚点只找到 %d 个 cond 段" %
                (len(keyframes), len(cond_segments)))
        origin = target_origin(self)
        last_coordinate = None
        for (start, stop), keyframe in zip(cond_segments, keyframes):
            if ANCHOR_FRAME not in keyframe:
                continue
            frame = int(keyframe[ANCHOR_FRAME])
            if frame < 0 or frame >= int(frame_count):
                raise ValueError(
                    "H3-Myang: 锚点帧 %d 超出目标范围 0..%d" %
                    (frame, int(frame_count) - 1))
            import comfy.ldm.minimax.model as mm
            if hasattr(mm, "_video_t_spans") and frame == int(frame_count) - 1:
                if last_coordinate is None:
                    last_coordinate = (origin + sum(mm._video_t_spans(latent_t))
                                       - mm.FRAME_RESCALE)
                coordinate = last_coordinate
            else:
                coordinate = origin + FRAME_RESCALE * float(frame)
            self.position_ids[start:stop, 0] = coordinate

    _move_audio_anchor(self, refs)


setattr(_patched_layout, LAYOUT_PATCH, True)


def _patched_payload(self, **kwargs):
    output = _payload_original(self, **kwargs)
    keyframes = kwargs.get("minimax_keyframes")
    refs = kwargs.get("minimax_refs")
    marked = (_has_anchor(keyframes)
              or bool(refs) and any(AUDIO_END_FRAME in ref for ref in refs))
    if not (marked and keyframes and refs):
        return output

    cond = output.get("minimax_payload")
    payload = getattr(cond, "cond", None) if cond is not None else None
    if not isinstance(payload, dict):
        raise RuntimeError(
            "H3-Myang: 无法读取 H3 payload，不能安全合并关键帧与参考素材")

    payload["cond_video_latents"] = (
        [item["latent"] for item in keyframes if "latent" in item]
        + [item["latent"] for item in refs if "latent" in item])
    payload["cond_audio_latents"] = [
        item["audio_latent"] for item in refs
        if item.get("audio_latent") is not None]
    frame_count = kwargs.get("minimax_frame_count")
    if frame_count is not None:
        payload["frame_count"] = frame_count
    return output


setattr(_patched_payload, PAYLOAD_PATCH, True)


def _layout_self_test(mm, stock):
    global _layout_original
    _layout_original = stock
    text_len, latent_t, latent_h, latent_w, audio_t = 11, 7, 8, 8, 12
    frame_count = pixel_frames(latent_t)

    dummy_latent = torch.zeros(1, 16, 1, 8, 8)
    for frame in (0, frame_count - 1):
        expected = mm.PackedLayout.__new__(mm.PackedLayout)
        _call_stock_layout(
            stock, expected, text_len, latent_t, latent_h, latent_w, audio_t,
            keyframes=[{"resolved_frame_index": frame, "latent": dummy_latent}],
            frame_count=frame_count)
        actual = mm.PackedLayout.__new__(mm.PackedLayout)
        _patched_layout(
            actual, text_len, latent_t, latent_h, latent_w, audio_t,
            keyframes=[{"resolved_frame_index": 0, ANCHOR_FRAME: frame, "latent": dummy_latent}],
            frame_count=frame_count)
        if not torch.allclose(actual.position_ids, expected.position_ids, atol=1e-7):
            raise RuntimeError(
                "H3-Myang: 官方首尾锚点等价检查失败（frame=%d）" % frame)

    # One image ref moves the target origin.  The marked video keyframe and
    # marked audio ref must both follow that shifted target, not text_len.
    end_frame = 4.8  # 5/3 * 4.8 == an exact target audio coordinate (8).
    refs = [
        {"kind": "image", "latent_h": 8, "latent_w": 8},
        {"kind": "audio", "ref_audio_t": 4,
         AUDIO_END_FRAME: end_frame},
    ]
    layout = mm.PackedLayout.__new__(mm.PackedLayout)
    _patched_layout(
        layout, text_len, latent_t, latent_h, latent_w, audio_t,
        keyframes=[{"resolved_frame_index": 0, ANCHOR_FRAME: 5, "latent": dummy_latent}],
        refs=refs, frame_count=frame_count)
    origin = target_origin(layout)
    cond_start = next(a for a, _, kind in layout.segments if kind == "cond")
    if abs(float(layout.position_ids[cond_start, 0])
           - (origin + FRAME_RESCALE * 5)) > 1e-9:
        raise RuntimeError("H3-Myang: refs 下的内部锚点没有跟随目标时间轴")
    audio_start, audio_stop = _ref_segment_map(layout, refs)[1]["ref_audio"]
    actual_end = float(layout.position_ids[audio_start:audio_stop, 0].max()) + 1.0
    expected_end = origin + FRAME_RESCALE * end_frame
    if abs(actual_end - expected_end) > 1e-9:
        raise RuntimeError("H3-Myang: 声音锚点没有对齐目标声音网格")


def _validate_patch_owner(current, owner_module, ours, old_motion_context, label):
    if getattr(current, ours, False):
        return "ours"
    if getattr(current, old_motion_context, False):
        raise RuntimeError(
            "H3-Myang: 当前进程已经启用了旧 Motion-Context 的 %s 补丁；"
            "两者不能同时占用同一入口。请重启 ComfyUI 后只使用 Myang 节点。" % label)
    where = getattr(current, "__module__", "") or ""
    if hasattr(current, "__wrapped__") or (owner_module and where != owner_module):
        raise RuntimeError(
            "H3-Myang: %s 已被未知插件包装（%s），为防止静默错帧而拒绝叠放" %
            (label, where or "?"))
    return "stock"


def install_patches():
    """Install the marker-gated patches after validating both owners."""
    global _layout_original, _payload_original
    import comfy.ldm.minimax.model as mm
    import comfy.model_base as model_base

    layout_current = mm.PackedLayout.__init__
    payload_current = model_base.MiniMaxH3.extra_conds
    layout_state = _validate_patch_owner(
        layout_current, mm.PackedLayout.__module__, LAYOUT_PATCH,
        "_h3_motion_context_layout_patch", "PackedLayout")
    payload_state = _validate_patch_owner(
        payload_current, model_base.MiniMaxH3.__module__, PAYLOAD_PATCH,
        "_h3_motion_context_payload_patch", "MiniMaxH3.extra_conds")

    if layout_state == "stock":
        _layout_original = layout_current
        _layout_self_test(mm, layout_current)
        mm.PackedLayout.__init__ = _patched_layout
    if payload_state == "stock":
        _payload_original = payload_current
        model_base.MiniMaxH3.extra_conds = _patched_payload

    logger.info("H3-Myang: 任意位置多关键帧与 keyframe/ref 合并已启用")
    return True


def rollback_patches():
    """Best-effort rollback used if a recognised observer cannot be replayed."""
    import comfy.ldm.minimax.model as mm
    import comfy.model_base as model_base
    if getattr(mm.PackedLayout.__init__, LAYOUT_PATCH, False) and _layout_original:
        mm.PackedLayout.__init__ = _layout_original
    if (getattr(model_base.MiniMaxH3.extra_conds, PAYLOAD_PATCH, False)
            and _payload_original):
        model_base.MiniMaxH3.extra_conds = _payload_original


def _resize(image, width, height):
    samples = image[..., :3].movedim(-1, 1)
    samples = comfy.utils.common_upscale(
        samples, int(width), int(height), "lanczos", "disabled")
    return samples.movedim(1, -1)


def _latent_blocks(context_latent, frame_count):
    video = video_stream(context_latent)
    steps = steps_for_frames(frame_count)
    if steps is None:
        raise ValueError(
            "H3-Myang: %d 帧不能完整落在 H3 temporal latent 网格上" %
            frame_count)
    total = int(video.shape[2])
    if steps > total:
        raise ValueError(
            "H3-Myang: 上一段只有 %d 个 latent step，需要 %d 个" %
            (total, steps))
    start = total - steps
    if start % 5:
        raise ValueError("H3-Myang: 上一段 latent 尾部没有落在 H3 时间相位起点")
    blocks = [video[:1, :, start + k:start + k + 1].clone()
              for k in range(steps)]
    return blocks, step_offsets(steps), pixel_frames(steps)


def _pixel_blocks(vae, context_frames, frame_count, width, height):
    available = int(context_frames.shape[0])
    if available < int(frame_count):
        raise ValueError(
            "H3-Myang: 需要 %d 帧上下文，实际只有 %d 帧" %
            (frame_count, available))
    tail = _resize(context_frames[available - int(frame_count):], width, height)
    encoded = vae.encode(tail)
    if getattr(encoded, "ndim", 0) != 5:
        raise ValueError("H3-Myang: 视频 VAE 输出应为 [B,C,T,H,W]")
    steps = int(encoded.shape[2])
    covered = pixel_frames(steps)
    if covered != int(frame_count):
        raise RuntimeError(
            "H3-Myang: %d 帧编码成 %d step，只覆盖 %d 帧" %
            (frame_count, steps, covered))
    blocks = [encoded[:, :, k:k + 1].clone() for k in range(steps)]
    return blocks, step_offsets(steps), covered


def _latent_audio_ref(context_latent, trim_frames):
    video = video_stream(context_latent)
    audio = audio_stream(context_latent)
    if audio is None:
        raise ValueError("H3-Myang: context_latent 没有声音流，无法连续钉住声音")
    total_t = int(audio.shape[-1])
    source_frames = pixel_frames(int(video.shape[2]))
    # Video ends between 40 Hz audio steps for most legal H3 frame counts.
    # Only pin complete audio steps strictly before that boundary.  Rounding up
    # includes the padded step beyond the visual cut and repeats it next segment.
    source_end = min(total_t, int(math.floor(
        float(source_frames) / FPS * AUDIO_HZ + 1e-9)))
    count = min(_audio_guard_steps(trim_frames), source_end)
    if count <= 0:
        raise ValueError("H3-Myang: 声音锚点窗短到没有完整的 40Hz step")
    start = source_end - count
    end_coordinate = count
    return {
        "kind": "audio",
        "ref_audio_t": count,
        "audio_latent": audio[:1, ..., start:source_end].clone(),
        AUDIO_END_FRAME: float(end_coordinate) / FRAME_RESCALE,
    }


def _encoded_audio_ref(audio_vae, context_audio, trim_frames):
    waveform = context_audio.get("waveform")
    if waveform is None:
        raise ValueError("H3-Myang: context_audio 没有 waveform")
    source_rate = int(context_audio.get("sample_rate", 0))
    vae_rate = int(getattr(audio_vae, "audio_sample_rate", 32000))
    if source_rate <= 0:
        raise ValueError("H3-Myang: context_audio 的 sample_rate 无效")
    if source_rate != vae_rate:
        try:
            import torchaudio
        except Exception as exc:
            raise RuntimeError(
                "H3-Myang: 声音需要从 %dHz 重采样到 %dHz，但 torchaudio 不可用" %
                (source_rate, vae_rate)) from exc
        waveform = torchaudio.functional.resample(
            waveform, source_rate, vae_rate)
    safe_steps = _audio_guard_steps(trim_frames)
    wanted = max(1, int(round(float(safe_steps) / AUDIO_HZ * vae_rate)))
    available = int(waveform.shape[-1])
    if available < wanted:
        logger.warning(
            "H3-Myang: 声音只有 %.3fs，短于 %.3fs 锚点窗",
            available / float(vae_rate), wanted / float(vae_rate))
    else:
        waveform = waveform[..., available - wanted:]
    latent = audio_vae.encode(waveform[:1].movedim(1, -1))
    audio_t = int(latent.shape[-1])
    if audio_t > safe_steps:
        latent = latent[..., :safe_steps]
        audio_t = safe_steps
    end_coordinate = safe_steps
    return {
        "kind": "audio",
        "ref_audio_t": audio_t,
        "audio_latent": latent,
        AUDIO_END_FRAME: float(end_coordinate) / FRAME_RESCALE,
    }


def _audio_guard_steps(trim_frames):
    """Complete 40 Hz steps before a 24 fps video cut (never round upward)."""
    return max(1, int(math.floor(
        float(trim_frames) / FPS * AUDIO_HZ + 1e-9)))


def _masked_anchor_latent(latent, blocks):
    """Seed the overlap into the target and denoise only the new timeline."""
    import comfy.nested_tensor

    parts = streams(latent)
    if not parts:
        raise ValueError("H3-Myang: 目标 H3 latent 没有画面流")
    target_video = video_stream(latent).clone()
    overlap_steps = len(blocks)
    if overlap_steps <= 0 or overlap_steps >= int(target_video.shape[2]):
        raise ValueError(
            "H3-Myang: 锚点 latent 步数 %d 无法写入目标 %d 步" %
            (overlap_steps, int(target_video.shape[2])))
    for index, block in enumerate(blocks):
        value = block.to(device=target_video.device, dtype=target_video.dtype)
        if int(value.shape[0]) != int(target_video.shape[0]):
            value = value[:1].expand(target_video.shape[0], -1, -1, -1, -1)
        target_video[:, :, index:index + 1] = value

    seeded_parts = [target_video]
    for part in parts[1:]:
        seeded_parts.append(part.unsqueeze(0) if part.ndim == 3 else part)
    video_mask = torch.ones(
        (target_video.shape[0], 1, target_video.shape[2], 1, 1),
        device=target_video.device, dtype=torch.float32)
    video_mask[:, :, :overlap_steps] = 0.0
    masks = [video_mask]
    # Audio continuity keeps using H3's aligned audio reference. It remains
    # fully denoised so the joint AV pass can synthesize lipsync/foley instead
    # of preserving empty target-audio rows.
    for part in seeded_parts[1:]:
        masks.append(torch.ones(
            (part.shape[0], 1, *part.shape[2:]),
            device=part.device, dtype=torch.float32))

    output = dict(latent)
    output["samples"] = comfy.nested_tensor.NestedTensor(tuple(seeded_parts))
    output["noise_mask"] = comfy.nested_tensor.NestedTensor(tuple(masks))
    return output




class H3AnchorContext:
    CATEGORY = "沐阳 H3"
    FUNCTION = "apply"
    RETURN_TYPES = ("CONDITIONING", "INT", "LATENT")
    RETURN_NAMES = ("conditioning", "trim_frames", "masked_latent")
    DESCRIPTION = (
        "上一段尾部钉到新段开头；重叠 latent 不加随机噪声，只有新增"
        "画面参与去噪。22 帧是稳定基线；5 帧仅产生两个时序 latent 锚点。")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "vae": ("VAE",),
                "latent": ("LATENT",),
                "context_length": (CONTEXT_LENGTHS, {
                    "default": "22",
                    "tooltip": "22=稳定连续窗；5=实验速度锚点（2个 temporal blocks）",
                }),
            },
            "optional": {
                "context_latent": ("LATENT",),
                "context_frames": ("IMAGE",),
                "context_audio": ("AUDIO",),
                "audio_vae": ("VAE",),
            },
        }

    def apply(self, conditioning, vae, latent, context_length,
              context_latent=None, context_frames=None,
              context_audio=None, audio_vae=None):
        from .anchor_compat import ensure_anchors
        ensure_anchors()

        trim = int(context_length)
        target_video = video_stream(latent)
        target_frames = pixel_frames(int(target_video.shape[2]))
        width = int(target_video.shape[4]) * 16
        height = int(target_video.shape[3]) * 16

        if context_latent is not None and context_frames is None:
            source_video = video_stream(context_latent)
            source_size = (int(source_video.shape[4]) * 16,
                           int(source_video.shape[3]) * 16)
            if source_size != (width, height):
                raise ValueError(
                    "H3-Myang: 上一段与新段分辨率不同，latent 不能直接续接")
            blocks, positions, trim = _latent_blocks(context_latent, trim)
            if context_audio is not None:
                if audio_vae is None:
                    raise ValueError(
                        "H3-Myang: context_audio 已连接，但没有连接 audio_vae")
                audio_ref = _encoded_audio_ref(
                    audio_vae, context_audio, trim)
            else:
                audio_ref = _latent_audio_ref(context_latent, trim)
            source = "latent"
        elif context_frames is not None:
            blocks, positions, trim = _pixel_blocks(
                vae, context_frames, trim, width, height)
            audio_ref = None
            if context_audio is not None:
                if audio_vae is None:
                    raise ValueError(
                        "H3-Myang: context_audio 已连接，但没有连接 audio_vae")
                audio_ref = _encoded_audio_ref(
                    audio_vae, context_audio, trim)
            source = "pixels"
        else:
            raise ValueError("H3-Myang: 需要 context_latent 或 context_frames")

        if trim >= target_frames:
            raise ValueError(
                "H3-Myang: 锚点占用 %d 帧，目标段只有 %d 帧" %
                (trim, target_frames))

        # motion-context 风格：整窗全钉，不做步级调度。钉入帧随后由
        # H3AnchorTrim 裁掉，拼接总帧数不变。
        keyframes = [
            {"resolved_frame_index": 0,
             ANCHOR_FRAME: positions[k],
             ANCHOR_TOTAL: target_frames,
             "latent": blocks[k]}
            for k in range(len(blocks))
        ]
        output = node_helpers.conditioning_set_values(
            conditioning, {"minimax_keyframes": keyframes}, append=True)
        output = node_helpers.conditioning_set_values(
            output, {"minimax_frame_count": target_frames})
        if audio_ref is not None:
            output = node_helpers.conditioning_set_values(
                output, {"minimax_refs": [audio_ref]}, append=True)
        masked_latent = _masked_anchor_latent(latent, blocks)

        logger.info(
            "H3-Myang: %s -> 全钉 %d/%d 步 @帧%s | trim %d | "
            "锚点区零噪声，新画面正常加噪",
            source, len(keyframes), len(blocks), positions, trim)
        return (output, trim, masked_latent)


class H3AnchorKeyframe:
    CATEGORY = "沐阳 H3"
    FUNCTION = "apply"
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    DESCRIPTION = (
        "把一张指定姿势钉在目标视频的任意帧。可串联多个节点，"
        "也可与段首续接锚点组合。")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "conditioning": ("CONDITIONING",),
            "vae": ("VAE",),
            "latent": ("LATENT",),
            "image": ("IMAGE",),
            "frame_index": ("INT", {
                "default": 0, "min": 0, "max": 4095,
                "tooltip": "从 0 开始；必须小于本段总帧数",
            }),
        }}

    def apply(self, conditioning, vae, latent, image, frame_index):
        from .anchor_compat import ensure_anchors
        ensure_anchors()
        target = video_stream(latent)
        frame_count = pixel_frames(int(target.shape[2]))
        index = int(frame_index)
        if index < 0 or index >= frame_count:
            raise ValueError(
                "H3-Myang: frame_index=%d，目标范围是 0..%d" %
                (index, frame_count - 1))
        width, height = int(target.shape[4]) * 16, int(target.shape[3]) * 16
        encoded = vae.encode(_resize(image[:1], width, height))
        if getattr(encoded, "ndim", 0) != 5 or int(encoded.shape[2]) != 1:
            raise RuntimeError("H3-Myang: 单帧关键图必须编码成一个 temporal block")
        keyframe = {
            "resolved_frame_index": 0,
            ANCHOR_FRAME: index,
            "latent": encoded[:, :, :1].clone(),
        }
        output = node_helpers.conditioning_set_values(
            conditioning, {"minimax_keyframes": [keyframe]}, append=True)
        output = node_helpers.conditioning_set_values(
            output, {"minimax_frame_count": frame_count})
        return (output,)


class H3AnchorTrim:
    CATEGORY = "沐阳 H3"
    FUNCTION = "trim"
    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("images", "audio")
    DESCRIPTION = "同步裁掉锚点占用的开头画面和声音，并按画面时长截齐声音尾部。"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "trim_frames": ("INT", {
                    "default": 0, "min": 0, "max": 4096}),
            },
            "optional": {
                "audio": ("AUDIO",),
                "fps": ("FLOAT", {
                    "default": 24.0, "min": 1.0, "max": 240.0}),
            },
        }

    def trim(self, images, trim_frames, audio=None, fps=24.0):
        trim = max(0, int(trim_frames))
        total = int(images.shape[0])
        if trim >= total:
            raise ValueError(
                "H3-Myang: 不能从 %d 帧里裁掉 %d 帧" % (total, trim))
        images = images[trim:]
        if audio is None:
            return (images, None)
        waveform = audio["waveform"]
        sample_rate = int(audio["sample_rate"])
        head = int(round(trim / float(fps) * sample_rate))
        waveform = waveform[..., head:]
        wanted = int(round(int(images.shape[0]) / float(fps) * sample_rate))
        if int(waveform.shape[-1]) > wanted:
            waveform = waveform[..., :wanted]
        elif int(waveform.shape[-1]) < wanted:
            waveform = torch.nn.functional.pad(
                waveform, (0, wanted - int(waveform.shape[-1])))
        return (images, {
            "waveform": waveform,
            "sample_rate": sample_rate,
        })


NODE_CLASS_MAPPINGS = {
    "H3AnchorContext": H3AnchorContext,
    "H3AnchorKeyframe": H3AnchorKeyframe,
    "H3AnchorTrim": H3AnchorTrim,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3AnchorContext": "沐阳 H3 · 段间多关键帧",
    "H3AnchorKeyframe": "沐阳 H3 · 任意位置关键帧",
    "H3AnchorTrim": "沐阳 H3 · 锚点同步裁剪",
}
