# Code, model and media publication notes

This is a practical compliance summary, not legal advice.

## Repository code

The node package and example workflows in this repository are licensed under
GPL-3.0-only. Source provenance and retained third-party notices are recorded
in `THIRD_PARTY_NOTICES.md`.

The repository does **not** include MiniMax H3 weights, text encoders, VAEs,
LightX2V LoRAs, NVIDIA SDK components, input media or generated videos. Those
items have separate licenses and terms.

The optional LBH-123-AI 3D latent-upscaler checkpoints are also not bundled.
Their model card currently marks the weights Apache-2.0; the compatible Myang
runtime attribution and the Apache-2.0 license copy are recorded in
`THIRD_PARTY_NOTICES.md` and `LICENSES/Apache-2.0.txt`.

## MiniMax H3 model and outputs

Before downloading weights, generating a demo, or publishing a generated
video, read the current official terms:

- Model license: https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE
- License Q&A: https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/QA-about-License.md

The license dated 2026-08-02 defines the European Union, United Kingdom,
Republic of Korea and United States as “Excluded Territories”. It also states
that H3 works and their outputs/results may not be used, distributed or
displayed outside the “Applicable Territory”. The license may change; preserve
a dated copy of the terms you relied on.

A globally accessible social-media upload may be viewable in excluded
territories. Obtain written clarification or an appropriate license from
MiniMax before treating broad international publication as authorized. The
license directs excluded-territory deployment enquiries to `api@minimax.io`.

For commercial products/services, the current license also contains branding
and revenue-based authorization requirements. Read the actual agreement rather
than relying only on this summary.

## Input and output rights checklist

Before publishing a demo video, confirm that:

1. You own or have permission for every input image, reference video, voice,
   music track, logo and font.
2. Identifiable people consent to likeness and voice use; do not present a
   synthetic performance as authentic.
3. The prompt does not request removal of third-party watermarks or unauthorized
   reuse of protected characters, performances or brands.
4. The platform's current synthetic-media disclosure is enabled and the video
   is visibly labeled when required.
5. Credits include this repository and applicable upstream projects, without
   implying that upstream authors endorse Myang.
6. The repository release contains no model weights, personal media, API keys,
   local absolute paths or private prompts.
