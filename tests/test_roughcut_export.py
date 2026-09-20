import importlib
import math
import shutil
import sys
import uuid
import wave
from fractions import Fraction
from pathlib import Path

import av
import numpy as np


TEST_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = TEST_DIR.parent
CUSTOM_NODES_DIR = PACKAGE_DIR.parent
COMFY_DIR = CUSTOM_NODES_DIR.parent
for path in (str(COMFY_DIR), str(CUSTOM_NODES_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

roughcut = importlib.import_module("ComfyUI-MiniMaxH3-Myang.roughcut")
exporter = importlib.import_module("ComfyUI-MiniMaxH3-Myang.roughcut_export")


def check(value, message):
    if not value:
        raise AssertionError(message)


def _write_video(path, color, frames=4, fps=8):
    with av.open(str(path), mode="w", format="mp4") as output:
        stream = output.add_stream("h264", rate=fps)
        stream.width = 64
        stream.height = 48
        stream.pix_fmt = "yuv420p"
        for index in range(frames):
            image = np.zeros((48, 64, 3), dtype=np.uint8)
            image[:] = color
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            frame.pts = index
            frame.time_base = Fraction(1, fps)
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode(None):
            output.mux(packet)


def _write_image(path, color):
    with av.open(str(path), mode="w", format="image2") as output:
        stream = output.add_stream("png", rate=1)
        stream.width = 64
        stream.height = 48
        stream.pix_fmt = "rgb24"
        image = np.zeros((48, 64, 3), dtype=np.uint8)
        image[:] = color
        frame = av.VideoFrame.from_ndarray(image, format="rgb24")
        for packet in stream.encode(frame):
            output.mux(packet)
        for packet in stream.encode(None):
            output.mux(packet)


def _write_audio(path, seconds=1.25, sample_rate=48000):
    frames = round(seconds * sample_rate)
    samples = np.asarray([
        round(math.sin(index * math.tau * 440 / sample_rate) * 8000)
        for index in range(frames)
    ], dtype=np.int16)
    stereo = np.column_stack((samples, samples))
    with wave.open(str(path), "wb") as output:
        output.setnchannels(2)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(stereo.tobytes())


def _clip(identifier, kind, start, end, name):
    return {
        "id": identifier,
        "kind": kind,
        "name": name,
        "timeline_start": start,
        "timeline_end": end,
        "source_in": 0,
        "source_out": end - start,
        "source_fps": 8,
        "source": {
            "type": "library",
            "library_id": "lib_test",
            "relative_path": name,
        },
    }


def test_export_streams_v1_a1_and_blank_regions(tmp_path=None):
    root = (TEST_DIR / f"myang_roughcut_export_{uuid.uuid4().hex}"
            if tmp_path is None else tmp_path)
    root.mkdir(parents=True)
    video_path = root / "red.mp4"
    image_path = root / "blue.png"
    audio_path = root / "tone.wav"
    _write_video(video_path, (240, 10, 10))
    _write_image(image_path, (10, 10, 240))
    _write_audio(audio_path, seconds=1.5)

    project = roughcut.new_project(fps=8, width=64, height=48)
    project["tracks"][0]["clips"] = [
        _clip("red", "video", 0, 4, video_path.name),
        _clip("blue", "image", 6, 10, image_path.name),
    ]
    project["tracks"][1]["clips"] = [
        _clip("tone", "audio", 0, 12, audio_path.name),
    ]
    paths = {item.name: item for item in (video_path, image_path, audio_path)}
    original_resolve = exporter.resolve_asset
    original_output = exporter.folder_paths.get_output_directory
    exporter.resolve_asset = lambda _library, relative: paths[relative]
    exporter.folder_paths.get_output_directory = lambda: str(root / "output")
    try:
        result = exporter.export_project(project, "roughcut/test", crf=22)
    finally:
        exporter.resolve_asset = original_resolve
        exporter.folder_paths.get_output_directory = original_output

    output_path = root / "output" / result["video"]["subfolder"] / result["video"]["filename"]
    check(result["ok"] and result["frames"] == 12 and output_path.is_file(),
          "export did not return a valid local output record")
    with av.open(str(output_path), mode="r") as container:
        frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
    check(len(frames) == 12, "exported video does not match the timeline frame count")
    check(float(frames[0][24, 32, 0]) > 180, "V1 video was not rendered")
    check(float(frames[4].mean()) < 12, "empty V1 region was not rendered as black")
    check(float(frames[9][24, 32, 2]) > 180, "V1 image was not held on the timeline")
    check(float(frames[-1].mean()) < 12,
          "A1-only tail did not preserve the V1 blank region")
    with av.open(str(output_path), mode="r") as container:
        check(len(container.streams.audio) == 1, "A1 was not exported as an audio stream")
        audio_frames = list(container.decode(audio=0))
        check(sum(frame.samples for frame in audio_frames) >= 70000,
              "A1 duration was truncated")
    if tmp_path is None:
        shutil.rmtree(root)


def test_export_rejects_output_path_traversal():
    try:
        exporter._validate_prefix("../outside")
    except ValueError:
        pass
    else:
        raise AssertionError("export accepted a path outside ComfyUI/output")


def test_track_spans_preserve_gaps_and_last_overlap_wins():
    track = {"clips": [
        {"id": "a", "timeline_start": 0, "timeline_end": 8},
        {"id": "b", "timeline_start": 3, "timeline_end": 5},
    ]}
    spans = exporter.track_spans(track, 10)
    check([(item.start, item.end, item.clip and item.clip["id"]) for item in spans]
          == [(0, 3, "a"), (3, 5, "b"), (5, 8, "a"), (8, 10, None)],
          "timeline span planning lost an overlap or blank region")


if __name__ == "__main__":
    test_export_streams_v1_a1_and_blank_regions()
    print("PASS test_export_streams_v1_a1_and_blank_regions")
    test_export_rejects_output_path_traversal()
    print("PASS test_export_rejects_output_path_traversal")
    test_track_spans_preserve_gaps_and_last_overlap_wins()
    print("PASS test_track_spans_preserve_gaps_and_last_overlap_wins")
