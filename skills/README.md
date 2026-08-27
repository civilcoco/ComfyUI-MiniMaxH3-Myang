# Optional Agent Skills

This directory is intentionally distributed without third-party Skill content.

- The Media Agent works without an external Skill and falls back to its built-in
  media-reference and prompt-writing rules.
- Install only Skills whose license permits redistribution and use.
- A personal Skill directory can be supplied with `MINIMAX_H3_SKILLS_DIR`.
- Local directory import is disabled by default. To enable it, set
  `MINIMAX_H3_SKILLS_IMPORT_DIR` to a dedicated import-only directory; the HTTP
  endpoint will reject paths outside that directory.
- `_skill_index.json` and `_skill_memory.json` are runtime caches and must not be
  committed or included in release archives.

MiniMax Hub and MiniMax H3 documentation are not bundled by this project. Obtain
them from their official source and review their current terms before use.
