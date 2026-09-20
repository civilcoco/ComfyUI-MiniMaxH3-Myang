"""Verify that Myang runtime code stays on the native media catalog path.

This audit intentionally does not inspect legal attribution or the small
``agent_media.py`` compatibility facade.  It checks executable runtime files:
old saved workflows may still be read, but new runs must not emit or mirror the
retired cross-package protocol.
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN = {
    "agent_nodes.py": (
        "ComfyUI-MiniMaxH3-Easy",
        "from .agent_media",
        "import agent_media",
        'return f"__MINIMAX_H3_REF_',
    ),
    "director.py": ("from .agent_media", "import agent_media"),
    "media.py": ("from .agent_media", "import agent_media"),
    "nodes.py": ("from .agent_media", "import agent_media"),
    "web/minimax_h3_myang_agent_ui.js": (
        "EASY_CLASS",
        "EASY_LINKS_PROP",
        "function isEasy(",
        "replaceAgentTagsWithEasyRefs",
        "mirrorAgentConnectionsToEasy",
        "connectionsFromEasy",
        "promptNode.inputs.media_links_json =",
    ),
}

REQUIRED = {
    "media_catalog.py": (
        "class MyangMediaAsset",
        "class MyangMediaCatalog",
        "def parse_asset_manifest",
    ),
    "agent_nodes.py": (
        'optional["asset_manifest_json"]',
        'kwargs.get(f"asset_{index}")',
        "MyangMediaCatalog(items=tuple(items)",
        '"wire_format": "readable_media_tags"',
    ),
    "web/minimax_h3_myang_agent_ui.js": (
        "promptNode.inputs.asset_manifest_json",
        "promptNode.inputs[`asset_${index + 1}`]",
        "myang_h3_asset_sources_v2",
    ),
}


def main() -> int:
    errors: list[str] = []
    for relative, needles in FORBIDDEN.items():
        text = (ROOT / relative).read_text(encoding="utf-8")
        for needle in needles:
            if needle in text:
                errors.append(f"retired runtime protocol {needle!r} found in {relative}")
    for relative, needles in REQUIRED.items():
        text = (ROOT / relative).read_text(encoding="utf-8")
        for needle in needles:
            if needle not in text:
                errors.append(f"native media contract {needle!r} missing from {relative}")

    if errors:
        print("NATIVE MEDIA AUDIT FAILED")
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print("NATIVE MEDIA AUDIT PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
