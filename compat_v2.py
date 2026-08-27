"""Install Myang's H3 anchors beneath recognised read-only observers."""

import logging
import sys

from . import anchors


logger = logging.getLogger(__name__)

_OBSERVERS = (
    {
        "module_hint": "_morton_h3",
        "freevar": "original_init",
        "registry": "_PATCHED_LAYOUTS",
        "installer": "_patch_packed_layout",
        "name": "Sol-Attn Morton",
    },
)


def _identify(wrapper):
    module_name = getattr(wrapper, "__module__", "") or ""
    return next(
        (item for item in _OBSERVERS
         if item["module_hint"] in module_name),
        None,
    )


def _closure_value(wrapper, name):
    names = getattr(getattr(wrapper, "__code__", None), "co_freevars", ()) or ()
    cells = getattr(wrapper, "__closure__", None) or ()
    if name not in names or len(names) != len(cells):
        return None
    return cells[names.index(name)].cell_contents


def ensure_anchors():
    """Settle the global wrapper order before the first segment samples."""
    import comfy.ldm.minimax.model as mm

    outer = mm.PackedLayout.__init__
    if getattr(outer, anchors.LAYOUT_PATCH, False):
        return True

    observers = []
    inner = outer
    while True:
        spec = _identify(inner)
        if spec is None:
            break
        unwrapped = _closure_value(inner, spec["freevar"])
        if unwrapped is None:
            raise RuntimeError(
                "H3-Myang: %s 的包装结构已变化，不能安全解开" % spec["name"])
        module_name = getattr(inner, "__module__", "") or ""
        module = sys.modules.get(module_name)
        if module is None:
            raise RuntimeError(
                "H3-Myang: 找不到 %s 模块 %s" %
                (spec["name"], module_name or "?"))
        if not callable(getattr(module, spec["installer"], None)):
            raise RuntimeError(
                "H3-Myang: %s 不再提供 %s" %
                (spec["name"], spec["installer"]))
        observers.append((spec, module))
        inner = unwrapped

    # Already in the desired observer(Myang(stock)) order.
    if getattr(inner, anchors.LAYOUT_PATCH, False):
        return True
    if getattr(inner, "_h3_motion_context_layout_patch", False):
        raise RuntimeError(
            "H3-Myang: 当前 ComfyUI 进程已经运行过旧 Motion-Context。"
            "请重启 ComfyUI，并只运行 Myang 自有锚点工作流。")

    owner = getattr(mm.PackedLayout, "__module__", "") or ""
    where = getattr(inner, "__module__", "") or ""
    if owner and where and where != owner:
        raise RuntimeError(
            "H3-Myang: PackedLayout 已被未知插件修改（%s），拒绝叠放" % where)

    mm.PackedLayout.__init__ = inner
    try:
        anchors.install_patches()
        for spec, module in reversed(observers):
            registry = getattr(module, spec["registry"], None)
            if isinstance(registry, set):
                registry.discard(id(mm.PackedLayout))
            getattr(module, spec["installer"])(mm)
    except Exception:
        anchors.rollback_patches()
        mm.PackedLayout.__init__ = outer
        raise

    if observers:
        logger.info(
            "H3-Myang: 自有锚点已安装在 %s 内层",
            "、".join(spec["name"] for spec, _ in observers))
    return True
