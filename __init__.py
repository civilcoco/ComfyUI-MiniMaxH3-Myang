"""ComfyUI-MiniMaxH3-Myang: native long-form MiniMax H3 workflow."""

import asyncio

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from .drift import (
    NODE_CLASS_MAPPINGS as _DRIFT_CLASSES,
    NODE_DISPLAY_NAME_MAPPINGS as _DRIFT_NAMES,
)
from .media import (
    NODE_CLASS_MAPPINGS as _MEDIA_CLASSES,
    NODE_DISPLAY_NAME_MAPPINGS as _MEDIA_NAMES,
)
from .core import (
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
from .audio_refine import (
    NODE_CLASS_MAPPINGS as _AUDIO_CLASSES,
    NODE_DISPLAY_NAME_MAPPINGS as _AUDIO_NAMES,
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
    clear_whisper_cache as _clear_whisper_cache,
)
from .progress import (
    NODE_CLASS_MAPPINGS as _PROGRESS_CLASSES,
    NODE_DISPLAY_NAME_MAPPINGS as _PROGRESS_NAMES,
)
from .roughcut_nodes import (
    NODE_CLASS_MAPPINGS as _ROUGHCUT_CLASSES,
    NODE_DISPLAY_NAME_MAPPINGS as _ROUGHCUT_NAMES,
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

        cleanup_state = {"task": None}

        async def finish_myang_cleanup(queue):
            # Wait until the interrupted execution has dropped its live tensor
            # references. Moving an active auxiliary model here can race CUDA.
            if queue is not None:
                for _ in range(1200):
                    getter = getattr(queue, "get_current_queue_volatile", None)
                    running = getter()[0] if getter is not None else queue.get_current_queue()[0]
                    if not running:
                        break
                    await asyncio.sleep(0.05)
                # Let prompt_worker consume unload_models/free_memory first.
                await asyncio.sleep(0.25)
            _clear_latent_upscale_cache()
            _clear_whisper_cache()

        @prompt_server.routes.post("/minimax-h3-myang/free-memory")
        async def free_myang_memory(request):
            queue = getattr(prompt_server, "prompt_queue", None)
            if queue is not None:
                # Queue cleanup as soon as stop is pressed. The worker performs
                # it safely after the interrupted stack has unwound.
                queue.set_flag("unload_models", True)
                queue.set_flag("free_memory", True)
            try:
                from . import llm_service
                stopped_requests = llm_service.cancel_active_http_requests()
            except Exception:
                stopped_requests = 0
            task = cleanup_state["task"]
            if task is None or task.done():
                cleanup_state["task"] = asyncio.create_task(
                    finish_myang_cleanup(queue))
            return web.json_response({
                "ok": True,
                "cleanup_queued": True,
                "stopped_network_requests": stopped_requests,
            })
    except Exception:
        # Route registration must never prevent the node package from loading.
        return


_register_memory_cleanup_route()

try:
    from .roughcut_library import register_routes as _register_roughcut_routes
    _register_roughcut_routes()
except Exception:
    # An unavailable web server must not prevent command-line node discovery.
    pass

try:
    from .roughcut_export import register_routes as _register_roughcut_export_route
    _register_roughcut_export_route()
except Exception:
    # Export is a local UI service and must not break node discovery.
    pass

try:
    from .asset_library import register_routes as _register_asset_library_routes
    _register_asset_library_routes()
except Exception:
    # The semantic catalogue is an optional local UI service.
    pass

try:
    from .director_templates import register_routes as _register_director_template_routes
    _register_director_template_routes()
except Exception:
    # Reusable Director templates are an optional local UI service.
    pass

try:
    from .director_revise import register_routes as _register_director_revise_routes
    _register_director_revise_routes()
except Exception:
    # Storyboard layering/rebinding/rewriting are optional editor-side helpers;
    # losing them must not stop the nodes themselves from loading.
    pass


NODE_CLASS_MAPPINGS.update(_DRIFT_CLASSES)
NODE_CLASS_MAPPINGS.update(_MEDIA_CLASSES)
NODE_CLASS_MAPPINGS.update(_CORE_CLASSES)
NODE_CLASS_MAPPINGS.update(_ANCHOR_CLASSES)
NODE_CLASS_MAPPINGS.update(_SEAM_CLASSES)
NODE_CLASS_MAPPINGS.update(_AUDIO_CLASSES)
NODE_CLASS_MAPPINGS.update(_TURBO_CLASSES)
NODE_CLASS_MAPPINGS.update(_DETAIL_CLASSES)
NODE_CLASS_MAPPINGS.update(_AGENT_CLASSES)
NODE_CLASS_MAPPINGS.update(_PROGRESS_CLASSES)
NODE_CLASS_MAPPINGS.update(_ROUGHCUT_CLASSES)
NODE_CLASS_MAPPINGS.update(_DIRECTOR_CLASSES)
NODE_DISPLAY_NAME_MAPPINGS.update(_DRIFT_NAMES)
NODE_DISPLAY_NAME_MAPPINGS.update(_MEDIA_NAMES)
NODE_DISPLAY_NAME_MAPPINGS.update(_CORE_NAMES)
NODE_DISPLAY_NAME_MAPPINGS.update(_ANCHOR_NAMES)
NODE_DISPLAY_NAME_MAPPINGS.update(_SEAM_NAMES)
NODE_DISPLAY_NAME_MAPPINGS.update(_AUDIO_NAMES)
NODE_DISPLAY_NAME_MAPPINGS.update(_TURBO_NAMES)
NODE_DISPLAY_NAME_MAPPINGS.update(_DETAIL_NAMES)
NODE_DISPLAY_NAME_MAPPINGS.update(_AGENT_NAMES)
NODE_DISPLAY_NAME_MAPPINGS.update(_PROGRESS_NAMES)
NODE_DISPLAY_NAME_MAPPINGS.update(_ROUGHCUT_NAMES)
NODE_DISPLAY_NAME_MAPPINGS.update(_DIRECTOR_NAMES)

# ComfyUI uses the same class mapping for node-menu discovery and for dynamic
# graph execution. Removing these mappings would break the all-in-one Director
# when it expands its internal nodes. The official deprecated flag hides nodes
# from the normal menu/search while preserving old workflows and runtime use.
_DIRECTOR_HIDDEN_NODE_IDS = frozenset({
    # Former public orchestration chain, now owned by H3Director.
    "H3ScriptSplitter", "H3SegmentPrompt", "H3SegmentCollector", "H3LongVideo",
    "H3DriftCorrect", "H3MediaSwapClip", "H3SeamBlend",
    "H3AnchorContext", "H3AnchorKeyframe", "H3AnchorTrim",
    "H3AudioRefineMask", "H3AudioRefineSampler", "H3AudioSettings", "H3AudioSeam",
    "H3DetailSettings", "H3DetailRefine", "H3LatentUpscale", "H3PixelUpscale",
    "H3VsrEnhance",
    # Runtime-only helpers which should never appear as user choices.
    "H3ShotMedia", "H3ReferenceClip", "H3ReferenceAudioClip", "H3ReferenceResize",
    "H3TailAnchorContext", "H3PrepareSignal", "H3ProgressSignal", "H3SamplerAdvanced",
    "H3RoughCutBoundaryFrames", "H3RoughCutSave",
    "H3Pass1CheckpointSave", "H3Pass1CheckpointLoad", "H3Pass1VideoEncode",
    "H3PreConditionMemoryBarrier", "H3ConditionMemoryBarrier",
    "H3RefineMemoryBarrier", "H3OutputMemoryRelease",
    "H3SegmentMemoryBarrier", "H3FramesToSeconds", "H3LatentIdentity",
    "H3DirectorPlanValue", "H3DirectorPlanSlice", "H3DirectorPlanLimit",
    "H3DirectorActionSource", "H3DirectorMediaImage", "H3DirectorTurnaroundMedia",
    "H3DirectorEnhancementSettings",
})
for _node_id in _DIRECTOR_HIDDEN_NODE_IDS:
    _node_class = NODE_CLASS_MAPPINGS.get(_node_id)
    if _node_class is not None:
        _node_class.DEPRECATED = True

WEB_DIRECTORY = "./web"

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]
