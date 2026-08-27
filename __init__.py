"""ComfyUI-MiniMaxH3-Myang: native long-form MiniMax H3 workflow."""

__version__ = "0.1.0"

from .nodes_v2 import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from .drift import (
    NODE_CLASS_MAPPINGS as _DRIFT_CLASSES,
    NODE_DISPLAY_NAME_MAPPINGS as _DRIFT_NAMES,
)
from .media import (
    NODE_CLASS_MAPPINGS as _MEDIA_CLASSES,
    NODE_DISPLAY_NAME_MAPPINGS as _MEDIA_NAMES,
)
from .core_v2 import (
    NODE_CLASS_MAPPINGS as _CORE_CLASSES,
    NODE_DISPLAY_NAME_MAPPINGS as _CORE_NAMES,
)
from .anchors import (
    NODE_CLASS_MAPPINGS as _ANCHOR_CLASSES,
    NODE_DISPLAY_NAME_MAPPINGS as _ANCHOR_NAMES,
)
from .seam import (
    NODE_CLASS_MAPPINGS as _SEAM_CLASSES,
    NODE_DISPLAY_NAME_MAPPINGS as _SEAM_NAMES,
)
from .turbo import (
    NODE_CLASS_MAPPINGS as _TURBO_CLASSES,
    NODE_DISPLAY_NAME_MAPPINGS as _TURBO_NAMES,
)
from .detail import (
    NODE_CLASS_MAPPINGS as _DETAIL_CLASSES,
    NODE_DISPLAY_NAME_MAPPINGS as _DETAIL_NAMES,
)
from .agent_nodes import (
    NODE_CLASS_MAPPINGS as _AGENT_CLASSES,
    NODE_DISPLAY_NAME_MAPPINGS as _AGENT_NAMES,
)
from .progress import (
    NODE_CLASS_MAPPINGS as _PROGRESS_CLASSES,
    NODE_DISPLAY_NAME_MAPPINGS as _PROGRESS_NAMES,
)
from .director import (
    NODE_CLASS_MAPPINGS as _DIRECTOR_CLASSES,
    NODE_DISPLAY_NAME_MAPPINGS as _DIRECTOR_NAMES,
)
from .latent_upscale_3d import clear_model_cache as _clear_latent_upscale_cache


def _register_memory_cleanup_route():
    """Expose package-aware cleanup for the browser interrupt event."""
    try:
        from aiohttp import web
        from server import PromptServer

        prompt_server = getattr(PromptServer, "instance", None)
        if prompt_server is None:
            return

        @prompt_server.routes.post("/minimax-h3-myang/free-memory")
        async def free_myang_memory(request):
            cleared = _clear_latent_upscale_cache()
            queue = getattr(prompt_server, "prompt_queue", None)
            if queue is not None:
                # The worker consumes these flags after the interrupted stack has
                # unwound: reset execution tensors, unload patchers, GC, empty VRAM.
                queue.set_flag("unload_models", True)
                queue.set_flag("free_memory", True)
            return web.json_response({"ok": True, "upscaler_models_cleared": cleared})
    except Exception:
        # Route registration must never prevent the node package from loading.
        return


_register_memory_cleanup_route()


NODE_CLASS_MAPPINGS.update(_DRIFT_CLASSES)
NODE_CLASS_MAPPINGS.update(_MEDIA_CLASSES)
NODE_CLASS_MAPPINGS.update(_CORE_CLASSES)
NODE_CLASS_MAPPINGS.update(_ANCHOR_CLASSES)
NODE_CLASS_MAPPINGS.update(_SEAM_CLASSES)
NODE_CLASS_MAPPINGS.update(_TURBO_CLASSES)
NODE_CLASS_MAPPINGS.update(_DETAIL_CLASSES)
NODE_CLASS_MAPPINGS.update(_AGENT_CLASSES)
NODE_CLASS_MAPPINGS.update(_PROGRESS_CLASSES)
NODE_CLASS_MAPPINGS.update(_DIRECTOR_CLASSES)
NODE_DISPLAY_NAME_MAPPINGS.update(_DRIFT_NAMES)
NODE_DISPLAY_NAME_MAPPINGS.update(_MEDIA_NAMES)
NODE_DISPLAY_NAME_MAPPINGS.update(_CORE_NAMES)
NODE_DISPLAY_NAME_MAPPINGS.update(_ANCHOR_NAMES)
NODE_DISPLAY_NAME_MAPPINGS.update(_SEAM_NAMES)
NODE_DISPLAY_NAME_MAPPINGS.update(_TURBO_NAMES)
NODE_DISPLAY_NAME_MAPPINGS.update(_DETAIL_NAMES)
NODE_DISPLAY_NAME_MAPPINGS.update(_AGENT_NAMES)
NODE_DISPLAY_NAME_MAPPINGS.update(_PROGRESS_NAMES)
NODE_DISPLAY_NAME_MAPPINGS.update(_DIRECTOR_NAMES)

WEB_DIRECTORY = "./web"

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
    "__version__",
]
