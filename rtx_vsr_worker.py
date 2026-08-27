"""Isolated NVIDIA VFX worker.

Some Windows builds of nvvfx 0.1.0.1 block indefinitely while destroying a
VideoSuperRes effect.  This helper deliberately ends with os._exit so that the
OS releases the CUDA/VFX resources without blocking the ComfyUI process.
Frames are exchanged through disk-backed float16 arrays to keep GPU staging to
one RGB frame.
"""

import argparse
import os
import sys
import traceback


def run(args):
    import numpy as np
    import torch
    import nvvfx

    source = np.memmap(
        args.input, mode="r", dtype=np.float16,
        shape=(args.frames, args.input_height, args.input_width, 3))
    target = np.memmap(
        args.output, mode="w+", dtype=np.float16,
        shape=(args.frames, args.output_height, args.output_width, 3))

    effect = nvvfx.VideoSuperRes(nvvfx.effects.QualityLevel.ULTRA)
    effect.output_width = args.output_width
    effect.output_height = args.output_height
    effect.load()
    for index in range(args.frames):
        # Copying the mapped slice avoids retaining file-backed memory in CUDA.
        frame = torch.from_numpy(
            np.array(source[index], dtype=np.float32, copy=True)
        ).movedim(-1, 0).cuda().contiguous()
        result = torch.from_dlpack(effect.run(frame).image).clone()
        target[index] = result.movedim(0, -1).to(
            device="cpu", dtype=torch.float16).numpy()
        del frame, result
    target.flush()
    sys.stdout.write("RTX_VSR_OK\n")
    sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--frames", required=True, type=int)
    parser.add_argument("--input-width", required=True, type=int)
    parser.add_argument("--input-height", required=True, type=int)
    parser.add_argument("--output-width", required=True, type=int)
    parser.add_argument("--output-height", required=True, type=int)
    args = parser.parse_args()
    exit_code = 0
    try:
        run(args)
    except BaseException:
        traceback.print_exc()
        sys.stderr.flush()
        exit_code = 1
    # Do not run nvvfx.VideoSuperRes.__del__: it hangs on affected Windows
    # installations even after the output has been cloned and synchronized.
    os._exit(exit_code)


if __name__ == "__main__":
    main()
