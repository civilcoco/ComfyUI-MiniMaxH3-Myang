"""Measure face continuity in a finished H3 long video.

The probe uses only local checkpoints. It samples the video, detects faces,
normalizes each crop, and compares adjacent segment distributions with HOG,
HSV, and an optional local CLIP-ViT-H image encoder.
"""

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


def _cosine(left, right):
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denom) if denom else 0.0


def _square_crop(image, box, padding=0.16):
    height, width = image.shape[:2]
    x1, y1, x2, y2 = (float(value) for value in box)
    size = max(x2 - x1, y2 - y1) * (1.0 + 2.0 * padding)
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    left, top = max(0, round(cx - size * 0.5)), max(0, round(cy - size * 0.5))
    right, bottom = min(width, round(cx + size * 0.5)), min(height, round(cy + size * 0.5))
    return image[top:bottom, left:right]


def _classic_descriptor(crop):
    resized = cv2.resize(crop, (128, 128), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [24, 16], [0, 180, 0, 256]).ravel()
    hist /= max(float(np.linalg.norm(hist)), 1e-8)
    hog = cv2.HOGDescriptor((128, 128), (32, 32), (16, 16), (16, 16), 9)
    texture = hog.compute(resized).ravel()
    texture /= max(float(np.linalg.norm(texture)), 1e-8)
    return hist.astype(np.float32), texture.astype(np.float32)


def _clip_embeddings(crops, checkpoint, device):
    if not checkpoint or not crops:
        return None
    import torch
    from safetensors.torch import load_file
    from transformers import CLIPVisionConfig, CLIPVisionModelWithProjection

    config = CLIPVisionConfig(
        hidden_size=1280, intermediate_size=5120, num_hidden_layers=32,
        num_attention_heads=16, image_size=224, patch_size=14,
        projection_dim=1024, hidden_act="quick_gelu")
    with torch.device("meta"):
        model = CLIPVisionModelWithProjection(config)
    incompatible = model.load_state_dict(
        load_file(str(checkpoint), device="cpu"), strict=False, assign=True)
    unexpected = set(incompatible.unexpected_keys) - {
        "vision_model.embeddings.position_ids"}
    if incompatible.missing_keys or unexpected:
        raise ValueError(
            "CLIP checkpoint does not match ViT-H: missing=%s unexpected=%s" %
            (incompatible.missing_keys, sorted(unexpected)))
    embeddings = model.vision_model.embeddings
    embeddings.position_ids = torch.arange(
        embeddings.num_positions, dtype=torch.long).expand((1, -1))
    meta = [name for name, tensor in model.named_buffers()
            if tensor is not None and tensor.is_meta]
    if meta:
        raise ValueError("CLIP model has unresolved meta buffers: %s" % meta)
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    model = model.to(device=device, dtype=dtype).eval()

    mean = torch.tensor((0.48145466, 0.4578275, 0.40821073), device=device, dtype=dtype)
    std = torch.tensor((0.26862954, 0.26130258, 0.27577711), device=device, dtype=dtype)
    output = []
    for start in range(0, len(crops), 12):
        batch = []
        for crop in crops[start:start + 12]:
            rgb = cv2.cvtColor(cv2.resize(crop, (224, 224), interpolation=cv2.INTER_AREA),
                               cv2.COLOR_BGR2RGB)
            batch.append(torch.from_numpy(rgb).permute(2, 0, 1))
        pixels = torch.stack(batch).to(device=device, dtype=dtype) / 255.0
        pixels = (pixels - mean[:, None, None]) / std[:, None, None]
        vectors = model(pixel_values=pixels).image_embeds.float()
        vectors /= vectors.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        output.extend(vectors.detach().cpu().numpy())
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return output


def _segment_of(frame, cuts):
    return sum(frame >= cut for cut in cuts) + 1


def _median_similarity(records, key):
    values = []
    for left, right in zip(records, records[1:]):
        values.append(_cosine(left[key], right[key]))
    return float(np.median(values)) if values else None


def probe(args):
    from ultralytics import YOLO

    capture = cv2.VideoCapture(str(args.video))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    cuts = sorted({cut for cut in args.cuts if 0 < cut < frame_count})
    sample_ids = set(range(0, frame_count, args.stride))
    for cut in cuts:
        sample_ids.update(range(max(0, cut - 48), min(frame_count, cut + 97), 6))
    sample_ids = sorted(sample_ids)

    frames = []
    valid_ids = []
    for frame_id in sample_ids:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ok, frame = capture.read()
        if ok:
            valid_ids.append(frame_id)
            frames.append(frame)
    capture.release()

    detector = YOLO(str(args.face_model))
    predictions = detector.predict(
        frames, device="cpu", conf=args.confidence, verbose=False)
    records = []
    crops = []
    for frame_id, frame, prediction in zip(valid_ids, frames, predictions):
        boxes = prediction.boxes
        if boxes is None or len(boxes) == 0:
            continue
        candidates = []
        for confidence, box in zip(boxes.conf.cpu().numpy(), boxes.xyxy.cpu().numpy()):
            area = max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))
            candidates.append((float(confidence) * math.pow(max(area, 1.0), 0.08),
                               float(confidence), box))
        _, confidence, box = max(candidates, key=lambda item: item[0])
        crop = _square_crop(frame, box)
        if min(crop.shape[:2]) < 24:
            continue
        color, texture = _classic_descriptor(crop)
        records.append({
            "frame": frame_id, "segment": _segment_of(frame_id, cuts),
            "confidence": confidence, "box": [float(value) for value in box],
            "color": color, "texture": texture,
        })
        crops.append(crop)

    clip_vectors = _clip_embeddings(crops, args.clip_model, args.device)
    if clip_vectors is not None:
        for record, vector in zip(records, clip_vectors):
            record["clip"] = vector

    segment_rows = []
    for segment in range(1, len(cuts) + 2):
        rows = [record for record in records if record["segment"] == segment]
        summary = {"segment": segment, "detected_samples": len(rows)}
        for key in ("color", "texture", "clip"):
            vectors = [row[key] for row in rows if key in row]
            if vectors:
                center = np.mean(vectors, axis=0)
                center /= max(float(np.linalg.norm(center)), 1e-8)
                summary[key] = center
        segment_rows.append(summary)

    adjacent = []
    for left, right in zip(segment_rows, segment_rows[1:]):
        row = {"from": left["segment"], "to": right["segment"]}
        for key in ("color", "texture", "clip"):
            if key in left and key in right:
                row[key + "_cosine"] = _cosine(left[key], right[key])
        adjacent.append(row)

    result = {
        "video": str(Path(args.video).resolve()), "fps": fps,
        "frame_count": frame_count, "cuts": cuts,
        "detected_samples": len(records),
        "successive_detected_face_similarity": {
            key: _median_similarity(records, key)
            for key in ("color", "texture", "clip")
            if any(key in record for record in records)
        },
        "adjacent_segment_centroid_similarity": adjacent,
        "segments": [
            {key: value for key, value in row.items()
             if key not in ("color", "texture", "clip")}
            for row in segment_rows],
        "samples": [
            {key: value for key, value in record.items()
             if key not in ("color", "texture", "clip")}
            for record in records],
    }
    Path(args.output).write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--cuts", type=lambda value: [int(x) for x in value.split(",")],
                        default=[])
    parser.add_argument("--face-model", type=Path, required=True)
    parser.add_argument("--clip-model", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stride", type=int, default=24)
    parser.add_argument("--confidence", type=float, default=0.05)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = probe(args)
    print(json.dumps({key: result[key] for key in (
        "frame_count", "cuts", "detected_samples",
        "successive_detected_face_similarity",
        "adjacent_segment_centroid_similarity")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
