"""Make H3 Motion Context and Sol-Attn coexist.

Both packs wrap ``comfy.ldm.minimax.model.PackedLayout.__init__``:

* Motion-Context *replaces the behaviour* — it lets keyframes anchor at
  interior frames instead of only first/last.
* Sol-Attn's ``_morton_h3`` only *observes* — it calls the original and then
  records the video span so the Morton reorder and the attention sink can find
  it. It changes nothing about the layout.

Motion-Context cannot tell those apart. Its guard compares
``PackedLayout.__init__.__module__`` against the class's own module, sees a
stranger, and refuses rather than wrapping something that might be a rival
implementation. That is the right default; it is just wrong in this one case.

Order is what actually matters, and the order is decided by the graph:
``SolAttnPatch`` sits on the model chain and runs before anything that uses
Motion-Context, so Sol-Attn always wins the race and Motion-Context always
refuses. Had it gone the other way both would have worked, because Sol-Attn
wrapping Motion-Context composes fine.

So this module puts them back in the order that works: unwrap the observers,
let Motion-Context patch the stock constructor, then reinstall the observers on
top. Result::

    PackedLayout.__init__ = solattn_observer( motion_context( stock ) )

Only wrappers positively recognised as observers are unwound. Anything else and
this bails out with the original error intact — a rival implementation of the
same patch really cannot be composed, and guessing would produce joins neither
pack intended.
"""

import logging
import sys

logger = logging.getLogger(__name__)

_done = False

# Wrappers known to be side-effect-free observers of PackedLayout.__init__.
#   module_hint: substring matched against the wrapper's __module__
#   freevar:     closure cell holding the constructor it wraps
#   registry:    module-level set of already-wrapped class ids, cleared so the
#                pack's own installer can be re-run
#   installer:   the pack's installer, called with the model module
_OBSERVERS = (
    {"module_hint": "_morton_h3", "freevar": "original_init",
     "registry": "_PATCHED_LAYOUTS", "installer": "_patch_packed_layout",
     "name": "Sol-Attn Morton"},
)


def _motion_context_module():
    """Motion-Context's nodes module, found through the node registry.

    Going through NODE_CLASS_MAPPINGS rather than a path means the folder can
    be named anything and a Manager install, a manual clone and a fork all
    resolve the same way.
    """
    try:
        import nodes as comfy_nodes
    except Exception:
        return None
    cls = comfy_nodes.NODE_CLASS_MAPPINGS.get("MiniMaxH3MotionContext")
    if cls is None:
        return None
    return sys.modules.get(getattr(cls, "__module__", ""))


def _identify(init):
    """Which observer wrapped this, if any."""
    where = getattr(init, "__module__", "") or ""
    for spec in _OBSERVERS:
        if spec["module_hint"] in where:
            return spec
    return None


def _inner(init, freevar):
    """The constructor an observer wrapper closed over."""
    names = getattr(getattr(init, "__code__", None), "co_freevars", ()) or ()
    cells = getattr(init, "__closure__", None) or ()
    if freevar not in names or len(names) != len(cells):
        return None
    return cells[names.index(freevar)].cell_contents


def ensure_motion_context_layout():
    """Install Motion-Context's layout patch, re-composing observers if needed.

    Returns True when interior anchors are available. Raises with an
    actionable message when they are not, because continuing would sample a
    whole segment before Motion-Context refuses on its own.
    """
    global _done
    if _done:
        return True

    mc = _motion_context_module()
    if mc is None or not hasattr(mc, "_apply_layout_patch"):
        # No Motion-Context installed. H3LongVideo only calls this when it is
        # about to chain segments, so this is worth saying plainly.
        raise RuntimeError(
            "H3LongVideo 需要 ComfyUI-H3-Motion-Context 做段间衔接，但没找到它。"
            "装上并重启，或把段数降到 1 段。")

    if mc._layout_patch_applied():
        _done = True
        return True
    if mc._apply_layout_patch():
        _done = True
        return True

    # Refused. If everything sitting on top of the constructor is a known
    # observer, unwind, patch the real thing, and put them back.
    import comfy.ldm.minimax.model as mm

    stack, init = [], mm.PackedLayout.__init__
    while True:
        spec = _identify(init)
        if spec is None:
            break
        inner = _inner(init, spec["freevar"])
        if inner is None:
            logger.warning("H3-Myang: %s 的包装结构不认识，无法解开", spec["name"])
            break
        stack.append(spec)
        init = inner

    if not stack:
        raise RuntimeError(
            "PackedLayout.__init__ 已经被另一个包改写，而且它不是可以叠放的观察者，"
            "所以 Motion-Context 拒绝再包一层。停用其中一个再重启。"
            "（当前来自 %r）" % (getattr(mm.PackedLayout.__init__, "__module__", "?"),))

    restore = mm.PackedLayout.__init__
    mm.PackedLayout.__init__ = init          # back to stock
    try:
        ok = mc._apply_layout_patch()
    except Exception:
        mm.PackedLayout.__init__ = restore
        raise
    if not ok:
        mm.PackedLayout.__init__ = restore
        raise RuntimeError(
            "已经把观察者补丁临时摘掉，Motion-Context 仍然拒绝安装它的 layout 补丁。"
            "原因在上面的日志里（通常是 custom_nodes 下装了不止一份 Motion-Context）。")

    # Reinstall the observers outermost-first so the original nesting order is
    # preserved: the one that was on top goes back on top.
    for spec in reversed(stack):
        mod = sys.modules.get(_source_module(restore, spec))
        if mod is None:
            continue
        registry = getattr(mod, spec["registry"], None)
        if isinstance(registry, set):
            registry.discard(id(mm.PackedLayout))
        getattr(mod, spec["installer"])(mm)

    logger.info("H3-Myang: 已把 %s 与 Motion-Context 重新叠放（观察者在外，钉帧在内），两者现在可以共存",
                "、".join(s["name"] for s in stack))
    _done = True
    return True


def _source_module(outermost, spec):
    """Module name of the observer described by `spec`, from the live chain."""
    init = outermost
    while init is not None:
        where = getattr(init, "__module__", "") or ""
        if spec["module_hint"] in where:
            return where
        init = _inner(init, spec["freevar"])
    return ""
