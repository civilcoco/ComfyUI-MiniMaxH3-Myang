"""Myang-native media catalog and payload normalization.

SPDX-License-Identifier: GPL-3.0-only

Copyright (C) 2026 Myang

The catalog is the single contract shared by the Agent, Director and H3
conditioning nodes.  Each asset owns both its runtime payload and descriptive
metadata, so consumers never have to align two parallel lists or translate
private prompt placeholders.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from typing import Any, Iterator

import torch


MAX_ASSETS = 15
MEDIA_KINDS = frozenset({"image", "video", "audio"})
_KIND_ORDER = {"image": 0, "video": 1, "audio": 2}
_VIDEO_EXTENSIONS = frozenset({".mp4", ".mov", ".webm", ".avi", ".mkv", ".flv", ".wmv", ".m4v"})
_AUDIO_EXTENSIONS = frozenset({".mp3", ".wav", ".m4a", ".flac", ".aac", ".ogg", ".opus"})
_IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tiff"})


def _kind(value: str) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"picture", "图片"}:
        normalized = "image"
    if normalized not in MEDIA_KINDS:
        raise ValueError(f"H3-Myang: 不支持的素材类型 {value!r}")
    return normalized


@dataclass(frozen=True, slots=True)
class MyangMediaAsset:
    """One catalog entry with its payload and identity kept together."""

    slot: int
    kind: str
    payload: Any
    filename: str = ""
    label: str = ""
    origin: str = ""

    def __post_init__(self):
        object.__setattr__(self, "slot", int(self.slot))
        object.__setattr__(self, "kind", _kind(self.kind))
        object.__setattr__(self, "filename", str(self.filename or "").strip())
        object.__setattr__(self, "label", str(self.label or "").strip())
        object.__setattr__(self, "origin", str(self.origin or "").strip())

    def carrying(self, payload: Any) -> "MyangMediaAsset":
        """Return the same logical asset carrying a different runtime value."""
        return replace(self, payload=payload)


@dataclass(frozen=True, slots=True)
class MyangMediaCatalog:
    """Immutable, deterministically ordered media catalog."""

    assets: tuple[MyangMediaAsset, ...] = ()

    def __post_init__(self):
        assets = tuple(self.assets or ())
        slots = [asset.slot for asset in assets]
        if len(slots) != len(set(slots)):
            raise ValueError("H3-Myang: 素材目录里存在重复插槽")
        object.__setattr__(self, "assets", assets)

    def ordered(self) -> tuple[MyangMediaAsset, ...]:
        return tuple(sorted(self.assets, key=lambda asset: (_KIND_ORDER[asset.kind], asset.slot)))

    def next_slot(self) -> int:
        return max((asset.slot for asset in self.assets), default=0) + 1

    def appended(self, asset: MyangMediaAsset) -> "MyangMediaCatalog":
        return MyangMediaCatalog(self.assets + (asset,))

    def replacing_video(self, ordinal: int, payload: Any) -> tuple["MyangMediaCatalog", bool]:
        target = int(ordinal)
        seen = 0
        updated: list[MyangMediaAsset] = []
        replaced = False
        for asset in self.assets:
            if asset.kind == "video":
                seen += 1
                if seen == target:
                    asset = asset.carrying(payload)
                    replaced = True
            updated.append(asset)
        return MyangMediaCatalog(tuple(updated)), replaced


def parse_asset_manifest(value: str) -> dict[int, dict[str, Any]]:
    """Parse the browser's descriptive records, indexed by source slot."""
    try:
        decoded = json.loads(str(value or "[]"))
    except (TypeError, json.JSONDecodeError):
        return {}
    if not isinstance(decoded, list):
        return {}
    records: dict[int, dict[str, Any]] = {}
    for fallback, record in enumerate(decoded, 1):
        if not isinstance(record, dict):
            continue
        try:
            slot = int(record.get("slot") or record.get("order") or fallback)
        except (TypeError, ValueError):
            continue
        records[slot] = record
    return records


def classify_payload(value: Any, hint: str = "") -> str:
    """Classify a ComfyUI value without relying on another custom node pack."""
    if value is None:
        return ""
    suggested = str(hint or "").strip().lower()
    if suggested == "picture":
        suggested = "image"

    if hasattr(value, "get_components"):
        return "video"

    if isinstance(value, torch.Tensor):
        if value.ndim == 4:
            if suggested in {"image", "video"}:
                return suggested
            return "video" if int(value.shape[0]) > 1 else "image"
        if value.ndim == 3:
            return "image"
        return suggested if suggested in MEDIA_KINDS else "image"

    if isinstance(value, dict):
        if value.get("waveform") is not None:
            return "audio"
        frame_value = next((value.get(key) for key in ("images", "frames", "video", "samples")
                            if value.get(key) is not None), None)
        if frame_value is not None:
            timeline_keys = {"fps", "frame_rate", "framerate", "audio", "soundtrack"}
            if (isinstance(frame_value, torch.Tensor) and frame_value.ndim == 4
                    and int(frame_value.shape[0]) == 1 and not timeline_keys.intersection(value)):
                return suggested if suggested in {"image", "video"} else "image"
            return "video"

    if isinstance(value, (tuple, list)):
        if (len(value) >= 2 and isinstance(value[0], torch.Tensor)
                and value[0].ndim <= 3 and isinstance(value[1], (int, float))
                and float(value[1]) >= 1000):
            return "audio"
        has_audio = False
        has_still = False
        has_video = False
        has_fps = False
        for part in value:
            if isinstance(part, dict) and part.get("waveform") is not None:
                has_audio = True
            elif isinstance(part, torch.Tensor):
                if part.ndim == 4 and int(part.shape[0]) > 1:
                    has_video = True
                elif part.ndim in {3, 4}:
                    has_still = True
                elif part.ndim <= 2:
                    has_audio = True
            elif isinstance(part, (int, float)) and not isinstance(part, bool):
                has_audio = has_audio or float(part) >= 1000
                has_fps = has_fps or 1 <= float(part) <= 240
            elif isinstance(part, dict) and any(
                    part.get(key) is not None for key in ("images", "frames", "video", "samples")):
                has_video = True
        if has_video:
            return "video"
        if has_still and (has_audio or has_fps):
            return "video"
        if has_audio:
            return "audio"
        if has_still:
            return "image"

    if isinstance(value, (str, os.PathLike)):
        suffix = os.path.splitext(os.fspath(value))[1].lower()
        if suffix in _VIDEO_EXTENSIONS:
            return "video"
        if suffix in _AUDIO_EXTENSIONS:
            return "audio"
        if suffix in _IMAGE_EXTENSIONS:
            return "image"

    return suggested if suggested in MEDIA_KINDS else "image"


def image_batch(value: Any) -> torch.Tensor:
    """Return an IMAGE-compatible NHWC batch from supported ComfyUI values."""
    if isinstance(value, torch.Tensor):
        if value.ndim == 3:
            return value.unsqueeze(0)
        if value.ndim == 4:
            return value
        raise ValueError(f"H3-Myang: 图像张量应为 3D/4D，实际为 {value.ndim}D")
    if hasattr(value, "get_components"):
        return image_batch(value.get_components().images)
    if isinstance(value, dict):
        for key in ("images", "image", "frames", "video", "samples"):
            if value.get(key) is not None:
                return image_batch(value[key])
    if isinstance(value, (tuple, list)):
        batches = []
        for part in value:
            try:
                batches.append(image_batch(part))
            except (AttributeError, TypeError, ValueError):
                continue
        if batches:
            return torch.cat(batches, dim=0)
    raise ValueError(f"H3-Myang: 无法从 {type(value).__name__} 读取图像帧")


def audio_track(value: Any) -> dict[str, Any] | None:
    """Return a normalized ComfyUI AUDIO dictionary, or None for no audio."""
    if value is None:
        return None
    if hasattr(value, "audio"):
        return audio_track(value.audio)
    if isinstance(value, dict) and value.get("waveform") is not None:
        waveform = value["waveform"]
        if not isinstance(waveform, torch.Tensor):
            raise ValueError("H3-Myang: 音频 waveform 必须是张量")
        if waveform.ndim == 1:
            waveform = waveform[None, None, :]
        elif waveform.ndim == 2:
            waveform = waveform[None, :, :]
        rate = int(value.get("sample_rate") or value.get("samplerate") or value.get("sampler_rate") or 32000)
        return {"waveform": waveform, "sample_rate": rate}
    if isinstance(value, (tuple, list)):
        if value and isinstance(value[0], torch.Tensor):
            rate = value[1] if len(value) > 1 and isinstance(value[1], (int, float)) else 32000
            return audio_track({"waveform": value[0], "sample_rate": rate})
        for part in value:
            try:
                track = audio_track(part)
            except (TypeError, ValueError):
                continue
            if track is not None:
                return track
    raise ValueError(f"H3-Myang: 无法从 {type(value).__name__} 读取音频")


def video_stream(value: Any) -> tuple[torch.Tensor, dict[str, Any] | None, float]:
    """Return ``(frames, optional audio, fps)`` for supported video carriers."""
    if value is None:
        raise ValueError("H3-Myang: 参考视频为空")
    if hasattr(value, "get_components"):
        components = value.get_components()
        frames = image_batch(components.images)
        sound = audio_track(getattr(components, "audio", None))
        rate = float(getattr(components, "frame_rate", 24.0) or 24.0)
        return frames, sound, rate
    if isinstance(value, torch.Tensor):
        return image_batch(value), None, 24.0
    if isinstance(value, dict):
        frames = next((value.get(key) for key in ("images", "frames", "video", "samples")
                       if value.get(key) is not None), None)
        if frames is not None:
            sound = value.get("audio") if value.get("audio") is not None else value.get("soundtrack")
            rate = float(value.get("fps") or value.get("frame_rate") or value.get("framerate") or 24.0)
            return image_batch(frames), audio_track(sound), rate
    if isinstance(value, (tuple, list)):
        frames = None
        sound = None
        rate = 24.0
        for part in value:
            if frames is None:
                try:
                    frames = image_batch(part)
                    continue
                except (AttributeError, TypeError, ValueError):
                    pass
            if isinstance(part, (int, float)) and not isinstance(part, bool) and 1 <= float(part) <= 240:
                rate = float(part)
            elif sound is None:
                try:
                    sound = audio_track(part)
                except (TypeError, ValueError):
                    pass
        if frames is not None:
            return frames, sound, rate
    raise ValueError(f"H3-Myang: 不支持的视频载体 {type(value).__name__}")


def asset_from_input(slot: int, payload: Any, metadata: dict[str, Any] | None = None) -> MyangMediaAsset:
    details = metadata or {}
    hint = str(details.get("kind") or details.get("media_type") or "")
    return MyangMediaAsset(
        slot=slot,
        kind=classify_payload(payload, hint),
        payload=payload,
        filename=str(details.get("filename") or ""),
        label=str(details.get("label") or details.get("subject") or ""),
        origin=str(details.get("origin") or details.get("source") or "agent"),
    )


def catalog_rows(catalog: MyangMediaCatalog | None) -> Iterator[tuple[str, int, str, str]]:
    counts = {kind: 0 for kind in MEDIA_KINDS}
    for asset in catalog.ordered() if isinstance(catalog, MyangMediaCatalog) else ():
        counts[asset.kind] += 1
        yield asset.kind, counts[asset.kind], asset.label, asset.filename


def iter_catalog(catalog: MyangMediaCatalog | None) -> Iterator[tuple[str, Any, str]]:
    for asset in catalog.ordered() if isinstance(catalog, MyangMediaCatalog) else ():
        yield asset.kind, asset.payload, asset.filename or asset.label


__all__ = [
    "MAX_ASSETS",
    "MyangMediaAsset",
    "MyangMediaCatalog",
    "asset_from_input",
    "audio_track",
    "catalog_rows",
    "classify_payload",
    "image_batch",
    "iter_catalog",
    "parse_asset_manifest",
    "video_stream",
]
