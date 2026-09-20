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
import types

from .memory_policy import scoped_reservation, increase_reservation

logger = logging.getLogger(__name__)

def broadcast_progress(payload):
    """Publish a Director progress event without ever blocking generation.

    Non-image stages such as final segment assembly do not need a passthrough
    node, so the collector calls this small helper directly. Keep the same
    event channel as ``H3ProgressSignal`` and treat a missing frontend/server
    as an optional UI condition rather than a generation failure.
    """
    try:
        from server import PromptServer
        inst = getattr(PromptServer, "instance", None)
        if inst is not None and hasattr(inst, "send_sync"):
            inst.send_sync("myh3_progress", dict(payload or {}))
            return True
    except Exception as exc:  # pragma: no cover - UI transport is optional
        logger.debug("H3 progress broadcast failed: %s", exc)
    return False


def _dynamic_evictable_bytes():
    """GPU weight bytes AIMDO could still hand back for a big allocation.

    A VBAR's resident pages are evictable on demand: the pressure controller
    drops them back to the pinned host cache when an activation allocation
    needs the room. Current free memory alone therefore understates what the
    next allocation can actually use, so the attention pre-flight adds every
    dynamic model's ``loaded_size`` on top.
    """
    total = 0
    try:
        import comfy.model_management as model_management
        for loaded in list(model_management.current_loaded_models):
            patcher = getattr(loaded, "model", None)
            get_vbar = getattr(patcher, "_vbar_get", None)
            if not callable(get_vbar):
                continue
            try:
                vbar = get_vbar()
            except Exception:
                continue
            loaded_size = getattr(vbar, "loaded_size", None)
            if callable(loaded_size):
                total += max(0, int(loaded_size()))
    except Exception:  # pragma: no cover - diagnostic accounting is optional
        return 0
    return total


def _attention_qkv_bytes(tokens, inner, element_size):
    """Size of the fused [tokens, 3*inner] QKV allocation every block makes."""
    return int(tokens) * int(inner) * 3 * int(element_size)


def assert_attention_fits(attention, tokens, element_size, device):
    """Refuse an attention block whose QKV cannot fit before asking CUDA.

    When the fused QKV is larger than everything the allocator could free,
    the allocation request still enters the driver first: on this WDDM box
    that request crawls through shared GPU memory for many minutes before
    torch finally raises, which reads exactly like a hang (one 720P run sat
    12.8 minutes inside the failed allocation). Compare the block's real
    numbers against free VRAM plus evictable dynamic weight pages and fail
    immediately with the diagnosis instead.

    The 1.4x budget grace keeps the check conservative: a block is only
    refused when it could not fit even with 40% slack, so pass-1 runs that
    legitimately stream weights keep working.
    """
    if getattr(device, "type", "") != "cuda":
        return
    heads = int(getattr(attention, "heads", 0) or 0)
    head_dim = int(getattr(attention, "head_dim", 0) or 0)
    if heads <= 0 or head_dim <= 0 or int(tokens) <= 0:
        return
    try:
        import comfy.model_management as model_management
        free_bytes = int(model_management.get_free_memory(device))
    except Exception:  # pragma: no cover - accounting is best effort
        return
    whale = _attention_qkv_bytes(tokens, heads * head_dim, element_size)
    evictable = _dynamic_evictable_bytes()
    if whale <= (free_bytes + evictable) * 1.4:
        return
    raise RuntimeError(
        "H3-Myang: 本段注意力需要约 %.1fGB 连续显存（%d token × %d 宽 QKV），"
        "当前可用 %.1fGB + 可换出权重页 %.1fGB 也不够。二采前请降低二采分辨率"
        "（如降到 540P）、缩短单段时长，或换更短的动作参考视频；这是在进入"
        "显存分配器之前的预检，避免卡在共享显存里十几分钟才报错。" % (
            whale / 1024 ** 3, int(tokens), heads * head_dim,
            free_bytes / 1024 ** 3, evictable / 1024 ** 3))


def _minimax_sage_chunk_reuse_forward(
        attention, x, rope_freqs, transformer_options, *, helper, mm, ck,
        fallback):
    """Run KJ's chunked H3 SageAttention without a second full output buffer.

    ``MiniMaxLowVRAMAttention`` deliberately passes the normalized block input
    in a one-item list: that tensor has no remaining consumer after QKV has
    been projected. KJ's Sage path used to delete it and then allocate a new
    ``[tokens, hidden]`` output while the 3x-larger fused QKV allocation was
    still alive. At 832P that extra allocation commonly crosses a 16GB Windows
    card into shared GPU memory (or raises OOM).

    Real H3 uses a 5376-wide block input but a 7168-wide attention stream, so
    that input cannot directly hold the head results.  The Q third of fused
    QKV is already 7168-wide and disposable after each head group has been
    consumed; write the attention result back over Q, then run ``out_proj`` in
    token chunks and copy its 5376-wide result into the handed-over block input.
    This removes both of KJ's additional full-sequence output allocations while
    keeping the math identical. Plain tensor callers normally stay on KJ's
    implementation; when per-reference weighting is active this function also
    handles them with a separate output buffer, because KJ's override otherwise
    bypasses the native H3 attention hook that applies those weights.
    """
    options = transformer_options if isinstance(transformer_options, dict) else {}
    disposable = isinstance(x, list) and len(x) == 1
    value_scales = tuple(options.get("minimax_reference_value_scales", ()))
    if isinstance(x, list) and not disposable:
        return fallback(x, rope_freqs=rope_freqs,
                        transformer_options=transformer_options)
    if not disposable and not value_scales:
        return fallback(x, rope_freqs=rope_freqs,
                        transformer_options=transformer_options)
    reusable = x[0] if disposable else x
    heads = int(getattr(attention, "heads", 0) or 0)
    head_dim = int(getattr(attention, "head_dim", 0) or 0)
    chunks = min(max(1, int(options.get("minimax_head_chunks", 1))), heads)
    inner = heads * head_dim
    qkv_projection = getattr(attention, "qkv_proj", None)
    out_projection = getattr(attention, "out_proj", None)
    hidden = int(reusable.shape[-1]) if reusable.ndim == 2 else 0
    if (heads <= 0 or head_dim <= 0
            or reusable.ndim != 2 or not reusable.is_contiguous()
            or not callable(qkv_projection) or not callable(out_projection)
            or int(getattr(qkv_projection, "in_features", hidden)) != hidden
            or int(getattr(qkv_projection, "out_features", inner * 3)) != inner * 3
            or int(getattr(out_projection, "in_features", inner)) != inner
            or int(getattr(out_projection, "out_features", hidden)) != hidden):
        return fallback(x, rope_freqs=rope_freqs,
                        transformer_options=transformer_options)

    if disposable:
        reusable = x.pop()
    dtype = reusable.dtype
    device = reusable.device
    tokens = int(reusable.shape[0])
    # Refuse a provably un-fittable QKV before the allocator enters the
    # driver; a doomed request crawls through WDDM shared memory for many
    # minutes before raising, which looks like a hang.
    assert_attention_fits(attention, tokens, reusable.element_size(), device)
    # ``Tensor.split`` returns several sibling views which PyTorch forbids
    # modifying in-place when autograd metadata exists. ``narrow`` creates
    # ordinary slice views over the same fused allocation and remains safe in
    # both the real inference context and CPU regression tests.
    fused_qkv = qkv_projection(reusable)
    for start, stop, scale in value_scales:
        start = max(0, min(tokens, int(start)))
        stop = max(start, min(tokens, int(stop)))
        scale = float(scale)
        if stop > start and scale != 1.0:
            fused_qkv[start:stop, inner * 2:].mul_(scale)
    q = fused_qkv.narrow(-1, 0, inner)
    k = fused_qkv.narrow(-1, inner, inner)
    v = fused_qkv.narrow(-1, inner * 2, inner)
    del fused_qkv
    q = q.view(1, tokens, heads, head_dim)
    k = k.view(1, tokens, heads, head_dim)
    v = v.view(1, tokens, heads, head_dim)
    if rope_freqs is not None:
        qw = mm.cast_to(attention.q_norm.weight, device=device)
        kw = mm.cast_to(attention.k_norm.weight, device=device)
        ck.rms_rope_split_half_(
            q, k, rope_freqs, qw, kw, epsilon=attention.q_norm.eps,
            rot_dim=rope_freqs.shape[-3] * 2)
    else:
        q = attention.q_norm(q)
        k = attention.k_norm(k)

    # When hidden == inner the original input is the cheapest destination and
    # QKV can be released before projection.  Real H3 has hidden != inner, so
    # overwrite Q itself; each later chunk reads disjoint Q/K/V heads.
    if disposable:
        destination = (reusable.view(1, tokens, heads, head_dim)
                       if hidden == inner else q)
    else:
        attention_result = reusable.new_empty((tokens, inner))
        destination = attention_result.view(1, tokens, heads, head_dim)
    head_start = 0
    for index in range(chunks):
        head_end = (head_start + heads // chunks
                    + (1 if index < heads % chunks else 0))
        piece = helper([
            q[:, :, head_start:head_end],
            k[:, :, head_start:head_end],
            v[:, :, head_start:head_end],
        ], dtype)
        destination[:, :, head_start:head_end].copy_(piece)
        del piece
        head_start = head_end
    if not disposable:
        del q, k, v, destination
        return out_projection(attention_result)
    if hidden == inner:
        del q, k, v, destination
        attention_result = reusable
        q = k = v = None
    else:
        # ``q`` now contains every attention head. K/V are views into the same
        # fused allocation, therefore it remains live only until out_proj has
        # consumed the final token chunk; no separate [tokens, inner] tensor is
        # created.
        attention_result = q.reshape(tokens, inner)

    token_chunks = min(tokens, max(4, chunks))
    token_start = 0
    for index in range(token_chunks):
        token_end = (token_start + tokens // token_chunks
                     + (1 if index < tokens % token_chunks else 0))
        projected = out_projection(attention_result[token_start:token_end])
        reusable[token_start:token_end].copy_(projected)
        del projected
        token_start = token_end
    del q, k, v, attention_result
    return reusable


def _minimax_lowmem_reference_forward(
        attention, x, rope_freqs, transformer_options, *, fallback, mm,
        comfy_module, optimized_attention):
    """Add reference Value scaling to KJ's optimized-attention H3 forward."""
    options = transformer_options if isinstance(transformer_options, dict) else {}
    value_scales = tuple(options.get("minimax_reference_value_scales", ()))
    if not value_scales:
        return fallback(x, rope_freqs=rope_freqs,
                        transformer_options=transformer_options)
    if isinstance(x, list):
        if len(x) != 1:
            return fallback(x, rope_freqs=rope_freqs,
                            transformer_options=transformer_options)
        x = x.pop()
    tokens = int(x.shape[0])
    device = x.device
    inner = int(attention.heads) * int(attention.head_dim)
    qkv = attention.qkv_proj(x)
    del x
    for start, stop, scale in value_scales:
        start = max(0, min(tokens, int(start)))
        stop = max(start, min(tokens, int(stop)))
        scale = float(scale)
        if stop > start and scale != 1.0:
            qkv[start:stop, inner * 2:].mul_(scale)
    q, k, v = qkv.split(inner, dim=-1)
    v = v.view(tokens, attention.heads, attention.head_dim)
    if rope_freqs is not None:
        q = q.view(1, tokens, attention.heads, attention.head_dim)
        k = k.view(1, tokens, attention.heads, attention.head_dim)
        qw = mm.cast_to(attention.q_norm.weight, device=device)
        kw = mm.cast_to(attention.k_norm.weight, device=device)
        rot = rope_freqs.shape[-3] * 2
        if mm.in_training:
            q, k = comfy_module.quant_ops.ck.rms_rope_split_half(
                q, k, rope_freqs, qw, kw, epsilon=attention.q_norm.eps,
                rot_dim=rot)
        else:
            comfy_module.quant_ops.ck.rms_rope_split_half_(
                q, k, rope_freqs, qw, kw, epsilon=attention.q_norm.eps,
                rot_dim=rot)
        q, k = q[0], k[0]
    else:
        q = attention.q_norm(q.view(
            tokens, attention.heads, attention.head_dim))
        k = attention.k_norm(k.view(
            tokens, attention.heads, attention.head_dim))
    q = q.transpose(0, 1).unsqueeze(0)
    k = k.transpose(0, 1).unsqueeze(0)
    v = v.transpose(0, 1).unsqueeze(0)
    chunks = min(max(1, int(options.get("minimax_head_chunks", 1))),
                 int(attention.heads))
    if chunks <= 1:
        out = optimized_attention(
            q, k, v, attention.heads, mask=None, skip_reshape=True,
            transformer_options=transformer_options).squeeze(0)
    else:
        out = qkv.new_empty((tokens, inner))
        head_start = 0
        for index in range(chunks):
            head_end = (head_start + attention.heads // chunks
                        + (1 if index < attention.heads % chunks else 0))
            piece = optimized_attention(
                q[:, head_start:head_end], k[:, head_start:head_end],
                v[:, head_start:head_end], head_end - head_start,
                mask=None, skip_reshape=True,
                transformer_options=transformer_options)
            out[:, head_start * attention.head_dim:
                head_end * attention.head_dim] = piece.squeeze(0)
            head_start = head_end
    del q, k, v, qkv
    return attention.out_proj(out)


def _install_minimax_sage_buffer_reuse(patcher):
    """Upgrade compatible KJ H3 attention patches on the active model clone."""
    patches = getattr(patcher, "object_patches", None)
    if not isinstance(patches, dict):
        return 0
    installed = 0
    sage_installed = 0
    lowmem_installed = 0
    skipped = 0
    for key, patched in list(patches.items()):
        function = getattr(patched, "__func__", patched)
        function_name = getattr(function, "__name__", "")
        if function_name not in {
                "minimax_sageattn_forward", "minimax_attn_lowmem_forward"}:
            continue
        if (getattr(function, "_myang_reuses_h3_input", False)
                or getattr(function, "_myang_reference_weighting", False)):
            continue
        attention = getattr(patched, "__self__", None)
        namespace = getattr(function, "__globals__", {})
        mm = namespace.get("mm")
        if attention is None or mm is None:
            continue
        # Validate the real MiniMax projection geometry.  Hidden and attention
        # widths deliberately differ in H3 (5376 -> 7168); the Q-buffer path
        # above supports that shape and projects token chunks back to hidden.
        projection = getattr(attention, "qkv_proj", None)
        output = getattr(attention, "out_proj", None)
        hidden = getattr(projection, "in_features", None)
        inner = int(getattr(attention, "heads", 0) or 0) * int(
            getattr(attention, "head_dim", 0) or 0)
        if (hidden is None or inner <= 0
                or int(getattr(projection, "out_features", inner * 3)) != inner * 3
                or int(getattr(output, "in_features", inner)) != inner
                or int(getattr(output, "out_features", hidden)) != int(hidden)):
            skipped += 1
            continue

        original = patched
        if function_name == "minimax_sageattn_forward":
            helper = namespace.get("_sageattn_int8_fp8_nhd")
            ck = namespace.get("_ck")
            if not callable(helper) or ck is None:
                continue

            def reuse_forward(self, x, rope_freqs=None, transformer_options={},
                              _helper=helper, _mm=mm, _ck=ck,
                              _fallback=original):
                return _minimax_sage_chunk_reuse_forward(
                    self, x, rope_freqs, transformer_options, helper=_helper,
                    mm=_mm, ck=_ck, fallback=_fallback)

            reuse_forward._myang_reuses_h3_input = True
            patches[key] = types.MethodType(reuse_forward, attention)
            sage_installed += 1
        else:
            comfy_module = namespace.get("comfy")
            optimized = namespace.get("optimized_attention")
            if comfy_module is None or not callable(optimized):
                continue

            def weighted_forward(
                    self, x, rope_freqs=None, transformer_options={},
                    _fallback=original, _mm=mm, _comfy=comfy_module,
                    _optimized=optimized):
                return _minimax_lowmem_reference_forward(
                    self, x, rope_freqs, transformer_options,
                    fallback=_fallback, mm=_mm, comfy_module=_comfy,
                    optimized_attention=_optimized)

            weighted_forward._myang_reference_weighting = True
            weighted_forward._uses_optimized_attention = True
            patches[key] = types.MethodType(weighted_forward, attention)
            lowmem_installed += 1
        installed += 1

    model_options = getattr(patcher, "model_options", {}) or {}
    transformer_options = (model_options.get("transformer_options", {})
                           if isinstance(model_options, dict) else {})
    if not isinstance(transformer_options, dict):
        transformer_options = {}
    delegate = transformer_options.get("sol_take_forward")
    if (callable(delegate)
            and getattr(delegate, "__name__", "") == "minimax_attn_lowmem_forward"
            and not getattr(delegate, "_myang_reference_weighting", False)):
        namespace = getattr(delegate, "__globals__", {})
        mm = namespace.get("mm")
        comfy_module = namespace.get("comfy")
        optimized = namespace.get("optimized_attention")
        if mm is not None and comfy_module is not None and callable(optimized):
            def weighted_delegate(
                    self, x, rope_freqs=None, transformer_options={},
                    _fallback=delegate, _mm=mm, _comfy=comfy_module,
                    _optimized=optimized):
                def bound_fallback(value, rope_freqs=None,
                                   transformer_options={}):
                    return _fallback(
                        self, value, rope_freqs=rope_freqs,
                        transformer_options=transformer_options)
                return _minimax_lowmem_reference_forward(
                    self, x, rope_freqs, transformer_options,
                    fallback=bound_fallback, mm=_mm,
                    comfy_module=_comfy,
                    optimized_attention=_optimized)

            weighted_delegate._myang_reference_weighting = True
            weighted_delegate._uses_optimized_attention = True
            transformer_options["sol_take_forward"] = weighted_delegate
            installed += 1
            lowmem_installed += 1
    if sage_installed:
        logger.info(
            "H3-Myang: 已为 %d 个 MiniMax SageAttention 块启用Q缓冲复用与分块输出，"
            "避免832P额外整块显存分配", sage_installed)
    if lowmem_installed:
        logger.info(
            "H3-Myang: 已为 %d 个 MiniMax 低显存注意力入口启用参考权重兼容",
            lowmem_installed)
    if skipped:
        logger.info(
            "H3-Myang: %d 个 H3 注意力入口投影结构不兼容；保持 KJ 原实现",
            skipped)
    return installed


def _install_native_reference_weighting(patcher):
    """Patch only this model's joint H3 blocks; leave other models untouched."""
    getter = getattr(patcher, "get_model_object", None)
    patches = getattr(patcher, "object_patches", None)
    if not callable(getter) or not isinstance(patches, dict):
        return 0
    import comfy
    import comfy.model_management as management
    from comfy.ldm.minimax import model as minimax
    try:
        model = getter("diffusion_model")
    except (AttributeError, KeyError):
        return 0
    installed = 0
    unsupported = []
    for index, block in enumerate(getattr(model, "blocks", ())):
        attention = getattr(block, "attn", None)
        if not isinstance(attention, minimax.Attention):
            continue
        key = f"diffusion_model.blocks.{index}.attn.forward"
        original = patches.get(key, attention.forward)
        function = getattr(original, "__func__", original)
        if (getattr(function, "_myang_reference_weighting", False)
                or getattr(function, "_myang_reuses_h3_input", False)):
            continue
        if function is not minimax.Attention.forward:
            unsupported.append(key)
            continue
        def weighted_forward(self, x, rope_freqs=None, transformer_options=None,
                             _fallback=original):
            return _minimax_lowmem_reference_forward(
                self, x, rope_freqs, transformer_options or {},
                fallback=_fallback, mm=management, comfy_module=comfy,
                optimized_attention=minimax.optimized_attention)
        weighted_forward._myang_reference_weighting = True
        patcher.add_object_patch(key, types.MethodType(weighted_forward, attention))
        installed += 1
    options = patcher.model_options.setdefault("transformer_options", {})
    options["myang_unsupported_reference_attention"] = tuple(unsupported)
    return installed


REFERENCE_WEIGHT_WRAPPER_KEY = "myang_h3_reference_value_scales"


def _reference_value_scales_wrapper(executor, x, timestep, context,
                                    transformer_options={},
                                    minimax_payload=None, **kwargs):
    """Copy the layout's weighted reference rows into transformer_options.

    ``PackedLayout`` (patched by ``anchors``) records ``reference_value_scales``
    for every ref block that carries a non-unit ``reference_weight``.  The
    attention forwards read ``minimax_reference_value_scales`` from
    ``transformer_options`` because that dict is the only per-call channel
    they receive; the payload never reaches an attention block.  This wrapper
    runs around ``MiniMaxH3Model._forward`` on every step and bridges the two.
    A layout rebuilt inside ``_forward`` (signature mismatch) would not carry
    the scales, so only the extra_conds-built layout is consulted; in that case
    weighting silently stays off, which is the pre-existing behaviour.
    """
    payload = minimax_payload or {}
    layout = payload.get("layout") if isinstance(payload, dict) else None
    scales = tuple(getattr(layout, "reference_value_scales", ()) or ())
    if isinstance(transformer_options, dict):
        if scales:
            if transformer_options.get("myang_unsupported_reference_attention"):
                raise ValueError("当前 H3 注意力补丁不支持参考权重；请使用原生注意力或受支持的 KJ 注意力，或关闭参考权重。")
            transformer_options["minimax_reference_value_scales"] = scales
        else:
            transformer_options.pop("minimax_reference_value_scales", None)
    return executor(x, timestep, context, transformer_options=transformer_options,
                    minimax_payload=minimax_payload, **kwargs)


def _install_reference_weight_bridge(patcher):
    """Register native attention and the layout bridge on the model clone."""
    _install_native_reference_weighting(patcher)
    add_wrapper = getattr(patcher, "add_wrapper_with_key", None)
    get_wrappers = getattr(patcher, "get_wrappers", None)
    if not callable(add_wrapper) or not callable(get_wrappers):
        return False
    try:
        import comfy.patcher_extension as patcher_extension
        wrapper_type = patcher_extension.WrappersMP.DIFFUSION_MODEL
    except Exception:  # pragma: no cover - very old ComfyUI
        return False
    try:
        if get_wrappers(wrapper_type, REFERENCE_WEIGHT_WRAPPER_KEY):
            return True
        add_wrapper(wrapper_type, REFERENCE_WEIGHT_WRAPPER_KEY,
                    _reference_value_scales_wrapper)
        return True
    except Exception as error:  # noqa: BLE001 - never block sampling
        logger.debug("H3-Myang: 无法安装参考权重桥接：%s", error)
        return False


def _staged_model_size(patcher):
    """Bytes the dynamic patcher stages into its VBAR, 0 when unavailable."""
    getter = getattr(patcher, "model_size", None)
    if not callable(getter):
        return 0
    try:
        return max(0, int(getter()))
    except Exception:  # pragma: no cover - patcher without a usable size
        return 0


def _apply_dynamic_vram_limit(guider, reserve_vram_gb, pass_label="",
                              force=False):
    """Give AIMDO real free headroom without imposing a fictitious model cap.

    A VBAR watermark limits *dynamic pages*, not total GPU use.  The previous
    ``total VRAM - reserve`` calculation ignored fixed weights and existing
    latents/conditions.  Worse, when that value exceeded the staged model it
    still wrote a full-model watermark, encouraging AIMDO to fill VRAM before
    the first high-resolution activation.

    Use the current free-memory shortfall instead: normal execution only evicts
    enough resident pages to reach the small requested cushion and otherwise
    leaves AIMDO's NVML pressure controller alone.  The H3 barriers flush all
    dynamic pages before a new phase, so installing a permanent/future VBAR
    watermark here would defeat AIMDO's own page-fault policy and turn every
    layer into a host round-trip.  After a real OOM, ``force`` additionally
    drops a bounded fraction of resident pages while preserving their pinned
    host cache for a fast deterministic retry.
    """
    try:
        reserve = max(0.0, float(reserve_vram_gb))
    except (TypeError, ValueError):
        return None
    if reserve <= 0.0:
        return None
    label = str(pass_label) or "采样"
    try:
        import comfy.model_management as model_management

        patcher = getattr(guider, "model_patcher", None)
        get_vbar = getattr(patcher, "_vbar_get", None)
        if not callable(get_vbar):
            return None
        vbar = get_vbar(create=True)
        if vbar is None or not hasattr(vbar, "set_watermark_limit"):
            return None
        device = getattr(patcher, "load_device", None)
        if device is None:
            device = model_management.get_torch_device()
        total = int(model_management.get_total_memory(device))
        free_now = int(model_management.get_free_memory(device))
        reserve_bytes = int(reserve * 1024 ** 3)
        staged = _staged_model_size(patcher)
        loaded_size = getattr(vbar, "loaded_size", None)
        free_memory = getattr(vbar, "free_memory", None)
        if not callable(loaded_size) or not callable(free_memory):
            return None
        resident = max(0, int(loaded_size()))
        shortfall = max(0, reserve_bytes - free_now)
        if not force and shortfall <= 0:
            logger.info(
                "H3-Myang: %s AIMDO自动调度 | 当前可用 %.2fGB ≥ 启动缓冲 %.2fGB；"
                "不设置硬水位，允许模型尽量驻留",
                label, free_now / 1024 ** 3, reserve)
            return vbar

        # A forced retry must change residency even though the failed
        # activation has already been released and free-memory now looks high.
        # Keep at least 2GB of dynamic pages resident to avoid turning every
        # layer into a disk/host fault.
        minimum_resident = min(resident, 2 * 1024 ** 3)
        forced_drop = 0
        if force and resident > minimum_resident:
            forced_drop = min(2 * 1024 ** 3,
                              max(512 * 1024 ** 2, resident // 4))
        to_free = max(shortfall, forced_drop)
        limit = max(minimum_resident, resident - to_free)
        if resident <= 0 or limit >= resident:
            logger.info(
                "H3-Myang: %s 当前没有可换出的动态权重页 | 可用 %.2fGB / 缓冲 %.2fGB",
                label, free_now / 1024 ** 3, reserve)
            return vbar

        vbar.set_watermark_limit(limit)
        freed = int(free_memory(resident - limit) or 0)
        logger.info(
            "H3-Myang: %s 动态页换出%s | %.2fGB → %.2fGB（释放 %.2fGB） | "
            "当前可用 %.2fGB / 启动缓冲 %.2fGB / 模型总量 %.2fGB / 显存 %.2fGB",
            label, "（OOM重试）" if force else "",
            resident / 1024 ** 3, limit / 1024 ** 3,
            freed / 1024 ** 3, free_now / 1024 ** 3, reserve,
            staged / 1024 ** 3, total / 1024 ** 3)
        return vbar
    except Exception as exc:  # pragma: no cover - AIMDO unavailable/optional
        logger.debug("H3-Myang: 无法设置 %s 动态权重驻留上限: %s", label, exc)
        return None


def _apply_second_pass_vram_limit(guider, reserve_vram_gb):
    """Backwards-compatible alias for the pass-agnostic watermark helper."""
    return _apply_dynamic_vram_limit(guider, reserve_vram_gb, "sample2")


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


class H3PrepareSignal:
    """Put preparation progress on the real execution path before H3Condition."""

    CATEGORY = "沐阳 H3/内部"
    FUNCTION = "signal"
    RETURN_TYPES = ("MYANG_H3",)
    RETURN_NAMES = ("h3",)
    DESCRIPTION = "内部准备信号：在条件编码实际开始前更新导演台。"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "h3": ("MYANG_H3",),
            "segment_index": ("INT", {"default": 1, "min": 1, "max": 999}),
            "total_segments": ("INT", {"default": 1, "min": 1, "max": 999}),
            "run_id": ("STRING", {"default": ""}),
            "owner_id": ("STRING", {"default": ""}),
            "activity": ("STRING", {"default": "准备提示词、素材与条件编码"}),
        }}

    def signal(self, h3, segment_index, total_segments, run_id, owner_id,
               activity):
        payload = {
            "run_id": str(run_id or ""),
            "owner_id": str(owner_id or ""),
            "segment_index": int(segment_index),
            "total_segments": int(total_segments),
            "stage": "preparing",
            "activity": str(activity or "准备提示词、素材与条件编码"),
        }
        logger.info(
            "H3-Myang: 导演台准备 | seg=%s/%s | %s",
            segment_index, total_segments, payload["activity"])
        try:
            from server import PromptServer
            inst = getattr(PromptServer, "instance", None)
            if inst is not None and hasattr(inst, "send_sync"):
                inst.send_sync("myh3_progress", payload)
        except Exception as exc:  # pragma: no cover - UI signal is optional
            logger.debug("H3PrepareSignal: 事件发送失败: %s", exc)
        return (h3,)


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


def _video_latent_from_x0(x0):
    """Return the video stream without materialising the packed audio stream."""
    if bool(getattr(x0, "is_nested", False)):
        tensors = x0.unbind()
        return tensors[0]
    if isinstance(x0, (tuple, list)):
        return x0[0]
    return x0


def _latent_rgb_preview(video_latent):
    """Project one H3 latent token to RGB on CPU without loading the video VAE.

    ComfyUI already ships the official MiniMax H3 24-channel RGB projection.
    Moving just one temporal token to CPU first keeps this path independent of
    CUDA residency: a preview can no longer evict the DiT or allocate a second
    several-gigabyte model while sampling.
    """
    from comfy.latent_formats import MiniMaxH3Video
    import latent_preview
    import torch
    from PIL import Image

    mid_t = int(video_latent.shape[2]) // 2
    token = video_latent[:, :, mid_t:mid_t + 1, :, :]
    token = token.detach().to(device="cpu", dtype=torch.float32)
    previewer = latent_preview.Latent2RGBPreviewer(
        MiniMaxH3Video.latent_rgb_factors,
        MiniMaxH3Video.latent_rgb_factors_bias)
    image = previewer.decode_latent_to_preview(token)

    # Latent spatial resolution is 1/16 of the video.  Upscale only the small
    # PIL image on CPU so the Director card remains legible without GPU work.
    longest = max(image.size)
    if 0 < longest < 512:
        scale = min(8.0, 512.0 / float(longest))
        target = (max(1, round(image.width * scale)),
                  max(1, round(image.height * scale)))
        image = image.resize(target, resample=Image.Resampling.BILINEAR)
    return image


def _save_step_preview(x0, vae, run_id, segment_index, step, pass_label,
                       preview_mode="vae"):
    """Save a step preview using either VAE decode or zero-VRAM latent RGB."""
    try:
        from PIL import Image
        import numpy as np
        import folder_paths
    except Exception:
        return None

    try:
        video_latent = _video_latent_from_x0(x0)

        if video_latent is None or video_latent.ndim != 5:
            return None

        T = int(video_latent.shape[2])
        if T <= 0:
            return None

        if str(preview_mode) == "latent_rgb":
            img = _latent_rgb_preview(video_latent)
        else:
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
    """SamplerCustomAdvanced replacement with selectable step previews."""

    CATEGORY = "沐阳 H3"
    FUNCTION = "execute"
    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("output", "denoised_output")
    DESCRIPTION = "内部采样器节点：支持零显存 latent 与清晰 VAE 步级预览。请勿手动添加。"

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
                "reserve_vram_gb": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 8.0, "step": 0.05}),
                "preview_interval": ("INT", {
                    "default": 1, "min": 0, "max": 100}),
            },
            "optional": {
                "preview_mode": (["latent_rgb", "vae"], {
                    "default": "vae",
                    "tooltip": "latent_rgb 不加载 VAE；vae 为旧版清晰预览"}),
                "return_denoised": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "仅实验性连续 Sigma 使用：返回分段点预测的干净 x0；"
                               "普通采样关闭，避免额外复制 latent"}),
            },
        }

    @scoped_reservation
    def execute(self, noise, guider, sampler, sigmas, latent_image,
                vae, run_id, owner_id, segment_index, total_segments,
                pass_label="sample1", reserve_vram_gb=0.0,
                preview_interval=1, preview_mode="vae",
                return_denoised=False):
        import comfy
        import comfy.memory_management
        import comfy.model_management
        import comfy.nested_tensor

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

        active_preview_interval = max(0, int(preview_interval))
        runtime_memory_reported = False
        latest_x0 = None
        # A 100k-token action-transfer pass can spend minutes inside its first
        # diffusion step while the log stays silent, which reads exactly like a
        # hang and has triggered more than one manual kill of a healthy run.
        # Log every completed step with its wall time so slow and stuck are
        # distinguishable from the log alone.
        sampling_started = time.monotonic()
        step_marks: list[float] = []

        def step_callback(step, x0, x, total):
            nonlocal runtime_memory_reported, latest_x0
            if return_denoised and x0 is not None:
                latest_x0 = x0
            now = time.monotonic()
            if step + 1 > len(step_marks):
                previous = step_marks[-1] if step_marks else 0.0
                step_marks.append(now - sampling_started)
                logger.info(
                    "H3-Myang: %s 段%d 步 %d/%d | 本步 %.1fs | 累计 %.1fs",
                    pass_label, int(segment_index), step + 1, int(total),
                    (now - sampling_started) - previous,
                    now - sampling_started)
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
            if (not runtime_memory_reported
                    and str(pass_label).startswith("sample2")):
                runtime_memory_reported = True
                try:
                    free_gb = (comfy.model_management.get_free_memory(
                        comfy.model_management.get_torch_device()) / 1024 ** 3)
                    payload["free_vram_gb"] = round(float(free_gb), 2)
                    logger.info(
                        "H3-Myang: 二采采样显存实测 | 当前可用 %.2fGB | 配置加载预留 %.2fGB",
                        free_gb, float(reserve_vram_gb))
                except Exception:
                    pass
            interval = active_preview_interval
            if (x0 is not None and interval > 0
                    # The finished segment is decoded immediately after the
                    # sampler. Avoid loading the VAE redundantly while the DiT
                    # is still resident on the final callback.
                    and step + 1 < total
                    and (step + 1) % interval == 0):
                preview_file = _save_step_preview(
                    x0, vae, run_id, segment_index, step, pass_label,
                    preview_mode=preview_mode)
                if preview_file:
                    payload["preview_file"] = preview_file
                    payload["preview_ts"] = int(time.time() * 1000)
                    payload["preview_kind"] = str(preview_mode)
            try:
                from server import PromptServer
                inst = getattr(PromptServer, "instance", None)
                if inst is not None and hasattr(inst, "send_sync"):
                    inst.send_sync("myh3_progress", payload)
            except Exception:
                pass

        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
        # ComfyUI normally aims to fill almost all VRAM.  At 832P on a 16GB
        # Windows card that can cross into shared GPU memory, where paging is
        # dramatically slower than leaving a modest safety margin.  Apply the
        # requested margin only for this sampler call and always restore the
        # process-wide setting afterwards.
        dynamic_vram = bool(getattr(
            comfy.memory_management, "aimdo_enabled", False))

        _install_minimax_sage_buffer_reuse(
            getattr(guider, "model_patcher", None))
        _install_reference_weight_bridge(
            getattr(guider, "model_patcher", None))

        # Evict bounded dynamic pages only for the high-resolution pass;
        # the scoped reservation restores any reversible backend target.
        if dynamic_vram and str(pass_label).startswith("sample2"):
            _apply_dynamic_vram_limit(
                guider, reserve_vram_gb, pass_label)

        # The first callback arrives only after a complete diffusion step. At
        # 768/832P that can take minutes, so publish step zero before the model
        # is prepared and let the Director enter the second-pass phase now.
        start_payload = {
            "run_id": str(run_id),
            "owner_id": str(owner_id or ""),
            "segment_index": int(segment_index),
            "total_segments": int(total_segments),
            "stage": "sampling",
            "pass_label": str(pass_label),
            "step": 0,
            "step_total": int(total_steps),
            "reserve_vram_gb": float(reserve_vram_gb),
        }
        try:
            from server import PromptServer
            inst = getattr(PromptServer, "instance", None)
            if inst is not None and hasattr(inst, "send_sync"):
                inst.send_sync("myh3_progress", start_payload)
        except Exception:
            pass
        generated_noise = noise.generate_noise(latent)

        def run_sampler():
            return guider.sample(
                generated_noise, latent_samples, sampler, sigmas,
                denoise_mask=noise_mask, callback=step_callback,
                disable_pbar=disable_pbar, seed=noise.seed)

        retry_after_oom = False
        aimdo_retry_headroom_gb = None
        try:
            samples = run_sampler()
        except Exception as exc:
            # A pass-2 832P activation can need a large contiguous output
            # tensor even with head-chunked SageAttention. If the first
            # attempt inherited fragmented VAE/upscaler/DiT residency,
            # restart deterministically once after a hard ComfyUI offload.
            import torch
            message = str(exc).casefold()
            is_cuda_oom = (isinstance(exc, torch.OutOfMemoryError)
                           or "allocation on device" in message
                           or "out of memory" in message)
            if not is_cuda_oom or not str(pass_label).startswith("sample2"):
                raise
            retry_after_oom = True
        # Do not retry inside the except block: its traceback can retain
        # the first attempt's q/k/v/output tensors until the handler exits,
        # defeating CUDA cache cleanup on the real device.
        if retry_after_oom:
            retry_headroom_gb = min(
                8.0, max(float(reserve_vram_gb) + 1.0, 2.5))
            logger.warning(
                "H3-Myang: %s 首次采样显存不足；临时换出动态权重页并以 %.2fGB "
                "启动缓冲、关闭中途预览后按相同种子重试一次",
                pass_label, retry_headroom_gb)
            active_preview_interval = 0
            try:
                retry_device = comfy.model_management.get_torch_device()
                free_before_retry = int(
                    comfy.model_management.get_free_memory(retry_device))
            except Exception:  # pragma: no cover - reporting only
                retry_device, free_before_retry = None, 0
            import gc
            gc.collect()
            if dynamic_vram:
                # The retry must actually change the conditions, not just
                # log a promise. On this stack a VBAR watermark alone never
                # bounded weight residency -- set_simple_vram_headroom is
                # the only knob that does -- and a failed forward leaves
                # _prefetch/_v_block_faulted markers plus cast buffers and
                # prefetch queues that keep GPU pages mapped. Apply all of
                # it, then report what the recovery actually achieved.
                try:
                    if increase_reservation(retry_headroom_gb):
                        aimdo_retry_headroom_gb = retry_headroom_gb
                except Exception:  # noqa: BLE001 - optional aimdo build
                    pass
                try:
                    import comfy.model_prefetch
                    comfy.model_prefetch.cleanup_prefetch_queues()
                except Exception:  # noqa: BLE001 - core API varies by build
                    pass
                try:
                    comfy.model_management.reset_cast_buffers()
                except Exception:  # noqa: BLE001 - core API varies by build
                    pass
                try:
                    from . import nodes as director_nodes
                    director_nodes._clear_dynamic_transient_pins(
                        getattr(guider, "model_patcher", None))
                except Exception:  # noqa: BLE001 - best-effort cleanup
                    pass
                comfy.model_management.soft_empty_cache()
                _apply_dynamic_vram_limit(
                    guider, retry_headroom_gb, pass_label, force=True)
            else:
                comfy.model_management.unload_all_models()
                comfy.model_management.soft_empty_cache()
                comfy.model_management.EXTRA_RESERVED_VRAM = max(
                    comfy.model_management.EXTRA_RESERVED_VRAM,
                    retry_headroom_gb * 1024 ** 3)
            if retry_device is not None:
                try:
                    free_after_retry = int(
                        comfy.model_management.get_free_memory(retry_device))
                    evictable = _dynamic_evictable_bytes()
                    logger.info(
                        "H3-Myang: %s OOM重试现场 | 可用显存 %.2fGB → %.2fGB | "
                        "可换出动态权重页 %.2fGB | AIMDO缓冲 %s",
                        pass_label, free_before_retry / 1024 ** 3,
                        free_after_retry / 1024 ** 3,
                        evictable / 1024 ** 3,
                        "已升至 %.2fGB" % aimdo_retry_headroom_gb
                        if aimdo_retry_headroom_gb is not None else "未变更")
                except Exception:  # pragma: no cover - reporting only
                    pass
            samples = run_sampler()
        samples = samples.to(comfy.model_management.intermediate_device())

        out = latent.copy()
        out.pop("downscale_ratio_spacial", None)
        out.pop("downscale_ratio_temporal", None)
        out["samples"] = samples
        denoised = out
        if return_denoised and latest_x0 is not None:
            x0 = latest_x0
            if (getattr(samples, "is_nested", False)
                    and not getattr(x0, "is_nested", False)):
                latent_shapes = [value.shape for value in samples.unbind()]
                x0 = comfy.nested_tensor.NestedTensor(
                    comfy.utils.unpack_latents(x0, latent_shapes))
            x0 = guider.model_patcher.model.process_latent_out(x0.cpu())
            denoised = latent.copy()
            denoised.pop("downscale_ratio_spacial", None)
            denoised.pop("downscale_ratio_temporal", None)
            denoised["samples"] = x0
        return (out, denoised)


NODE_CLASS_MAPPINGS = {
    "H3PrepareSignal": H3PrepareSignal,
    "H3ProgressSignal": H3ProgressSignal,
    "H3SamplerAdvanced": H3SamplerAdvanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3PrepareSignal": "沐阳 H3 · 准备进度（内部）",
    "H3ProgressSignal": "沐阳 H3 · 进度信号（内部）",
    "H3SamplerAdvanced": "沐阳 H3 · 采样器（内部·步级预览）",
}
