# Example workflow

Author and maintainer: **沐阳Myang**
Bilibili: **沐阳Myang**
GitHub: [@civilcoco](https://github.com/civilcoco)

`Minimax_H3_Myang_LongVideo_CN.json` is a sanitized copy of the author's full
long-video workflow. It preserves the optional nodes added around Myang; local
input filenames, personal prompts, output names and seed values were replaced.

After opening it, select your own:

- character/reference images;
- motion-reference video;
- optional reference audio;
- installed H3/CLIP/VAE/LoRA filenames.

The workflow demonstrates optional integrations. ComfyUI Manager can identify
most missing nodes automatically. The current full graph uses:

- [ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite)
  for video/audio upload and final video nodes;
- the Media Agent is included in ComfyUI-MiniMaxH3-Myang itself;
- [ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes) for H3 memory
  patches, EasyCache and PreviewAny;
- [ComfyUI-SolAttn_triton](https://github.com/kijai/ComfyUI-SolAttn_triton);
- [ComfyUI-Spectrum-MiniMax-H3](https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3);
- [ComfyUI-ReservedVRAM](https://github.com/Windecay/ComfyUI-ReservedVRAM);
- [ComfyUI-Easy-Use](https://github.com/yolain/ComfyUI-Easy-Use) for `easy seed`;
- [Bjornulf custom nodes](https://github.com/justUmen/Bjornulf_custom_nodes)
  for the optional resolution selector.

Most acceleration/cache nodes are bypassable. They are not Python import
dependencies of Myang itself. If an optional optimizer changes quality or
audio continuity, bypass it and establish a plain Myang baseline first.

Model weights and media are intentionally not included. Read `../LEGAL.md`
before downloading H3 weights or publishing generated output.
