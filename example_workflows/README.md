# Example workflows

Author and maintainer: **沐阳Myang**<br>
Bilibili: [**沐阳Myang**](https://space.bilibili.com/506587111)<br>
GitHub: [@civilcoco](https://github.com/civilcoco)

## Director example

`Minimax_H3_Myang_Director_CN.json` is the compact starting point. It uses
Myang nodes and ComfyUI core nodes only, so no other custom-node package is
required for its graph.

## Full long-video example

`Minimax_H3_Myang_LongVideo_CN.json` keeps the stages individually wired and
demonstrates optional acceleration and utility nodes around Myang. Local input
filenames, personal prompts, output names and seed values have been replaced.

After opening either example, select your own reference media, H3/CLIP/VAE/LoRA
files, prompts, output name, and seed.

The full graph uses these optional custom-node packages:

- [ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite)
  for `VHS_LoadVideo` and `VHS_LoadAudioUpload`;
- [ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes) for
  `MiniMaxChunkFeedForward`, `MiniMaxH3MemoryEfficientSageAttentionPatch`, and
  `MiniMaxLowVRAMAttention`;
- [ComfyUI-SolAttn_triton](https://github.com/kijai/ComfyUI-SolAttn_triton);
- [ComfyUI-Spectrum-MiniMax-H3](https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3);
- [ComfyUI-ReservedVRAM](https://github.com/Windecay/ComfyUI-ReservedVRAM);
- [ComfyUI-Easy-Use](https://github.com/yolain/ComfyUI-Easy-Use) for `easy seed`.

`EasyCache`, `PreviewAny`, `ResolutionSelector`, `CreateVideo`, and `SaveVideo`
in the current example are ComfyUI core nodes, not dependencies on the packages
previously associated with similar node names.

Most acceleration and cache nodes are bypassable. They are not Python import
dependencies of Myang itself. If an optional optimizer changes quality or audio
continuity, bypass it and establish a plain Myang two-segment baseline first.

Model weights and media are intentionally not included. Read
[`../LEGAL.md`](../LEGAL.md) before downloading H3 weights or publishing
generated output.
