# Release checklist

## Repository

- [ ] Replace placeholder GitHub URLs and publisher identifiers.
- [ ] Confirm `LICENSE`, `THIRD_PARTY_NOTICES.md` and source headers.
- [ ] Run `python tools/release_audit.py --strict-local` in a clean checkout.
- [ ] Run `tools/run_tests.ps1 -ComfyRoot <path-to-ComfyUI>` and keep every Python/JS audit green.
- [ ] Confirm local learned Skills and `docs/research` artifacts were not force-added.
- [ ] Open the sanitized example workflow in a clean ComfyUI installation.
- [ ] Confirm missing optional nodes are documented; test with “角色五视图” both off and on.
- [ ] Commit on `main`, create an annotated `v0.2.0` tag and publish release notes.
- [ ] Enable Issues, private vulnerability reporting and branch protection.

## Comfy Registry (optional, after GitHub release)

- [ ] Create a publisher at https://registry.comfy.org/.
- [ ] Put the immutable publisher id in `pyproject.toml`.
- [ ] Validate metadata with `comfy node init`/Registry tooling.
- [ ] Store `REGISTRY_ACCESS_TOKEN` only as a GitHub Actions secret.
- [ ] Publish manually first; automate after the initial package is verified.

Official instructions:
https://docs.comfy.org/registry/publishing

## Demo video

- [ ] Re-read `LEGAL.md` and the current MiniMax H3 model license.
- [ ] Confirm rights/consent for every reference asset and music track.
- [ ] Use a clean capture with no local paths, account names or API settings.
- [ ] Show exact version/tag and tested hardware/settings.
- [ ] Label AI-generated footage where required.
- [ ] Put repository and upstream credits in the description.
- [ ] Do not upload globally until H3 output-territory permission is clear.
