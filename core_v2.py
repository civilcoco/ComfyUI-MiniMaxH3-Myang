"""Compatibility shim.

The reference/keyframe handling now lives in one place, `core`. Keeping a second
`H3Condition` here is what let a fix land in one copy and not the other -- the
resolution cap, the five reference-area budgets and the filename mention mode
were all repaired in the base and then shadowed by the override that used to be
here. Everything is re-exported so existing imports keep working.
"""

from .core import (  # noqa: F401
    H3Bundle, H3Condition, H3Loader, H3Model,
    NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS,
    _to_24fps, canvas_for, length_for, resolve_mentions, iter_media,
)
from .media_catalog import audio_track, image_batch, video_stream  # noqa: F401
