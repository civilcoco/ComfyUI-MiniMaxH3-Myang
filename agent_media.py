"""Compatibility facade for the Myang media catalog.

SPDX-License-Identifier: GPL-3.0-only

New code should import :mod:`media_catalog` directly. These names remain only
so older workflows and third-party integrations can load without migration.
"""

from __future__ import annotations

from .core import _audio
from .media_catalog import (
    MAX_ASSETS, MyangMediaAsset, MyangMediaCatalog, audio_track, classify_payload,
    image_batch, parse_media_links, video_stream,
)

MAX_MEDIA = MAX_ASSETS
_MediaInput = MyangMediaAsset
MiniMaxH3MediaBundle = MyangMediaCatalog


def _parse_media_links(value: str) -> list[dict]:
    return parse_media_links(value)


def _infer_media_type(value, hint: str = "") -> str:
    return classify_payload(value, hint)


def _extract_image_tensor(value):
    return image_batch(value)


def _extract_audio_dict(value):
    try:
        audio = _audio(value)
    except ValueError:
        audio = None
    return audio if audio is not None else audio_track(value)


def _video_parts(value):
    return video_stream(value)


__all__ = [
    "MAX_MEDIA",
    "MiniMaxH3MediaBundle",
    "_MediaInput",
    "_extract_audio_dict",
    "_extract_image_tensor",
    "_infer_media_type",
    "_parse_media_links",
    "_video_parts",
]
