# Release checklist

## Repository

- [x] Replace placeholder GitHub URLs and publisher identifiers.
- [x] Confirm `LICENSE`, `THIRD_PARTY_NOTICES.md` and source headers.
- [x] Run secret/path/media filename scans.
- [x] Run CPU regression suites and import/registration smoke tests.
- [ ] Open the sanitized example workflow in a clean ComfyUI installation.
- [x] Confirm the release has no undeclared external custom-node dependency.
- [ ] Commit on `main`, create an annotated `v0.1.0` tag and publish release notes.
- [ ] Enable Issues, private vulnerability reporting and branch protection.

## Comfy Registry (optional, after GitHub release)

- [x] Create a publisher at https://registry.comfy.org/.
- [x] Put the immutable publisher id (`civilcoco`) in `pyproject.toml`.
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
