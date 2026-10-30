from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

import video_privacy

try:
    from .replay_to_mlflow import log_result
except ImportError:
    from replay_to_mlflow import log_result

CLIP_BUILDER_VERSION = "2.0"
SEED = 20261025
ANNOTATION_REVISION = "db171f1b7fedf4d3453e81297ff02f9915356d19"
IMAGE_REVISION = "04cab2dc3e18be03e2935b731c1dfa9eb29a31b5"
REQUIRED_CATEGORIES = ("face_size_small", "face_size_medium", "face_size_large",
                       "occlusion", "pose_profile", "low_light", "motion_blur", "scene_transition")


@dataclass
class Face:
    box: list[float]
    blur: int = 0
    illumination: int = 0
    occlusion: int = 0
    pose: int = 0


@dataclass
class SourceImage:
    source_id: str
    path: Path
    faces: list[Face]


@dataclass
class ClipFrame:
    image: np.ndarray
    faces: list[Face]
    low_light: bool = False
    motion_blur: bool = False
    scene_transition: bool = False


@dataclass
class Clip:
    clip_id: str
    source_ids: tuple[str, ...]
    frames: list[ClipFrame]


def source_split(source_ids: Iterable[str], seed: int = SEED,
                 tune_fraction: float = 0.7) -> dict[str, str]:
    if not 0.0 < tune_fraction < 1.0:
        raise ValueError("tune_fraction must be between zero and one")
    identifiers = sorted(set(source_ids))
    if len(identifiers) < 2:
        raise ValueError("At least two distinct source images are required for a held-out split")
    ranked = sorted(identifiers,
                    key=lambda source_id: hashlib.sha256(f"{seed}:{source_id}".encode()).digest())
    tune_count = min(len(ranked) - 1, max(1, int(round(len(ranked) * tune_fraction))))
    return {source_id: ("tune" if index < tune_count else "heldout")
            for index, source_id in enumerate(ranked)}


def read_wider_annotations(annotation_file: Path, image_root: Path, limit: int = 0,
                           seed: int = SEED) -> list[SourceImage]:
    lines = annotation_file.read_text(encoding="utf-8-sig").splitlines()
    records, cursor = [], 0
    while cursor < len(lines):
        name = lines[cursor].strip(); cursor += 1
        if not name:
            continue
        if cursor >= len(lines):
            break
        try:
            count = int(lines[cursor].strip())
        except ValueError as error:
            raise ValueError(f"Malformed WIDER annotation at line {cursor}: {name}") from error
        cursor += 1
        if count == 0 and cursor < len(lines):
            placeholder = lines[cursor].split()
            if len(placeholder) >= 10 and all(value == "0" for value in placeholder[:10]):
                cursor += 1
        faces = []
        for _ in range(count):
            if cursor >= len(lines):
                raise ValueError(f"Truncated WIDER annotation: {name}")
            values = [int(value) for value in lines[cursor].split()]
            cursor += 1
            if len(values) < 10 or values[7] == 1 or values[2] <= 0 or values[3] <= 0:
                continue
            faces.append(Face([float(value) for value in values[:4]], values[4], values[6], values[8], values[9]))
        image_path = image_root / name
        if image_path.is_file():
            records.append(SourceImage(name, image_path, faces))
    if limit and len(records) > limit:
        records = random.Random(seed).sample(records, limit)
    return sorted(records, key=lambda record: record.source_id)


def _transform_face(face: Face, matrix: np.ndarray, width: int, height: int) -> Face | None:
    x, y, box_width, box_height = face.box
    corners = np.asarray([[x, y, 1], [x + box_width, y + box_height, 1]], dtype=np.float32)
    mapped = corners @ matrix.T
    x0, y0 = np.maximum(mapped[0], [0, 0])
    x1, y1 = np.minimum(mapped[1], [width, height])
    if x1 <= x0 or y1 <= y0:
        return None
    return Face([float(x0), float(y0), float(x1 - x0), float(y1 - y0)],
                face.blur, face.illumination, face.occlusion, face.pose)


def build_clip(source: SourceImage, clip_id: str, frames: int = 3,
               variant: str = "normal") -> Clip:
    if frames < 1 or variant not in {"normal", "low_light", "motion_blur"}:
        raise ValueError("Invalid clip frame count or variant")
    image = cv2.imread(str(source.path))
    if image is None:
        raise ValueError(f"Could not read image {source.path}")
    height, width = image.shape[:2]
    rendered = []
    for index in range(frames):
        scale = 1.0 + 0.025 * index
        tx, ty = -0.0125 * width * index, -0.006 * height * index
        matrix = np.asarray([[scale, 0.0, (1.0 - scale) * width / 2 + tx],
                             [0.0, scale, (1.0 - scale) * height / 2 + ty]], dtype=np.float32)
        frame = cv2.warpAffine(image, matrix, (width, height), borderMode=cv2.BORDER_REFLECT)
        low_light, motion_blur = variant == "low_light", variant == "motion_blur"
        if low_light:
            frame = np.clip(frame.astype(np.float32) * 0.35, 0, 255).astype(np.uint8)
        if motion_blur:
            kernel = np.zeros((7, 7), dtype=np.float32); kernel[3, :] = 1 / 7
            frame = cv2.filter2D(frame, -1, kernel)
        faces = [mapped for face in source.faces
                 if (mapped := _transform_face(face, matrix, width, height)) is not None]
        rendered.append(ClipFrame(frame, faces, low_light, motion_blur))
    return Clip(clip_id, (source.source_id,), rendered)


def build_transition_clip(first: Clip, second: Clip, clip_id: str) -> Clip:
    if set(first.source_ids) & set(second.source_ids):
        raise ValueError("Hard-cut clips must use distinct source images")
    frames = list(first.frames) + list(second.frames)
    if len(first.frames) < 1 or len(second.frames) < 1:
        raise ValueError("Both transition clips must have frames")
    frames[len(first.frames)].scene_transition = True
    return Clip(clip_id, tuple(sorted(first.source_ids + second.source_ids)), frames)


def build_proxy_clips(sources: list[SourceImage], assignments: dict[str, str],
                      split: str, frames: int) -> list[Clip]:
    eligible = [source for source in sources if assignments[source.source_id] == split]
    variants = [build_clip(source, f"{source.source_id}:{variant}", frames, variant)
                for source in eligible for variant in ("normal", "low_light", "motion_blur")]
    normal = [build_clip(source, f"{source.source_id}:transition", frames, "normal") for source in eligible]
    transitions = [build_transition_clip(normal[index], normal[index + 1],
                                         f"transition:{index}")
                   for index in range(0, len(normal) - 1, 2)]
    return variants + transitions


def mask_rect(box: Iterable[float], width: int, height: int,
              inflation: float = 0.0) -> tuple[int, int, int, int]:
    x, y, box_width, box_height = (float(value) for value in list(box)[:4])
    extra = max(0.0, inflation)
    mx = int(round(box_width * (video_privacy.FACE_MARGIN_X + extra)))
    mt = int(round(box_height * (video_privacy.FACE_MARGIN_TOP + extra)))
    mb = int(round(box_height * (video_privacy.FACE_MARGIN_BOTTOM + extra)))
    return (max(0, int(round(x)) - mx), max(0, int(round(y)) - mt),
            min(width, int(round(x + box_width)) + mx),
            min(height, int(round(y + box_height)) + mb))


def coverage_ratio(face_box: Iterable[float], mask: np.ndarray) -> float:
    height, width = mask.shape[:2]
    x, y, w, h = (int(round(value)) for value in list(face_box)[:4])
    x0, y0, x1, y1 = max(0, x), max(0, y), min(width, x + w), min(height, y + h)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return float(mask[y0:y1, x0:x1].mean())


def _rate_ci(values: dict[str, tuple[int, int]], seed: int, count: int) -> list[float | None]:
    keys = sorted(values)
    if not keys:
        return [None, None]
    rng = random.Random(seed)
    rates = []
    for _ in range(count):
        sample = [rng.choice(keys) for _ in keys]
        exposed = sum(values[key][0] for key in sample)
        total = sum(values[key][1] for key in sample)
        if total:
            rates.append(exposed / total)
    return ([float(np.percentile(rates, 2.5)), float(np.percentile(rates, 97.5))]
            if rates else [None, None])


def _categories(face: Face, frame: ClipFrame) -> list[str]:
    height = face.box[3]
    categories = ["face_size_small" if height < 32 else "face_size_medium" if height <= 64 else "face_size_large"]
    if face.occlusion > 0: categories.append("occlusion")
    if face.pose > 0: categories.append("pose_profile")
    if frame.low_light or face.illumination > 0: categories.append("low_light")
    if frame.motion_blur or face.blur > 0: categories.append("motion_blur")
    if frame.scene_transition: categories.append("scene_transition")
    return categories


class Detector:
    def __init__(self, kind: str, weights_dir: Path, score: float, nms: float):
        self.kind, self.score, self.nms = kind, score, nms
        self._yunet = {}
        if kind == "yunet":
            self.weights_dir = weights_dir
            self.weights_dir.mkdir(parents=True, exist_ok=True)
            self.model_path = video_privacy.ensure_yunet_model(self.weights_dir)
        elif kind == "scrfd":
            from insightface.app import FaceAnalysis
            self._root = weights_dir / "insightface"
            self._app = FaceAnalysis(name="buffalo_sc", root=str(self._root),
                                     allowed_modules=["detection"], providers=["CPUExecutionProvider"])
            self._app.prepare(ctx_id=-1, det_size=(640, 640))
            self._model = self._app.models["detection"]
            self.model_path = Path(self._model.model_file)
            self._model.det_thresh = score
            if hasattr(self._model, "nms_thresh"):
                self._model.nms_thresh = nms
        else:
            raise ValueError(f"Unknown detector: {kind}")

    def detect(self, frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        if self.kind == "yunet":
            key = (width, height)
            detector = self._yunet.get(key)
            if detector is None:
                detector = cv2.FaceDetectorYN_create(str(self.model_path), "", key,
                                                      self.score, self.nms, 5000)
                self._yunet[key] = detector
            _, faces = detector.detect(frame)
            if faces is None:
                return np.empty((0, 5), dtype=np.float32)
            return np.column_stack((faces[:, :4], faces[:, 14])).astype(np.float32)
        boxes, _ = self._model.detect(frame, max_num=0)
        if boxes is None or not len(boxes):
            return np.empty((0, 5), dtype=np.float32)
        return np.asarray(boxes, dtype=np.float32)


def _positions(length: int, tile_length: int, grid: int, overlap: float) -> list[int]:
    if grid <= 1:
        return [max(0, (length - tile_length) // 2)]
    if grid == 2:
        return [0, max(0, length - tile_length)]
    step = max(1, int(tile_length * (1.0 - overlap)))
    positions = list(range(0, max(1, length - tile_length + 1), step))
    positions.append(max(0, length - tile_length))
    positions = sorted(set(positions))
    if len(positions) > grid:
        positions = [positions[int(round(i * (len(positions) - 1) / (grid - 1))) ] for i in range(grid)]
    return positions


def detect_multiscale(detector: Detector, frame: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    height, width = frame.shape[:2]
    threshold = config["threshold"]
    candidates = [box for box in detector.detect(frame) if box[4] >= threshold]
    if cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean() < video_privacy.LOW_LIGHT_LUMA:
        enhanced = video_privacy._enhance_low_light(frame)
        candidates.extend(box for box in detector.detect(enhanced) if box[4] >= threshold)
    scale, grid = config["tile_scale"], config["tile_grid"]
    if grid > 0:
        tile_w, tile_h = max(1, int(width * scale)), max(1, int(height * scale))
        xs = _positions(width, tile_w, grid, config["tile_overlap"])
        ys = _positions(height, tile_h, grid, config["tile_overlap"])
        for y0 in ys:
            for x0 in xs:
                tile = cv2.resize(frame[y0:y0 + tile_h, x0:x0 + tile_w], (width, height),
                                  interpolation=cv2.INTER_CUBIC)
                for detection in detector.detect(tile):
                    if detection[4] >= threshold:
                        mapped = detection.copy()
                        mapped[:4] *= np.asarray([tile_w / width, tile_h / height,
                                                  tile_w / width, tile_h / height], dtype=np.float32)
                        mapped[0] += x0; mapped[1] += y0
                        candidates.append(mapped)
    if not candidates:
        return np.empty((0, 5), dtype=np.float32)
    boxes = [list(map(int, np.round(row[:4]))) for row in candidates]
    indices = cv2.dnn.NMSBoxes(boxes, [float(row[4]) for row in candidates],
                               threshold, config["nms"])
    if len(indices) == 0:
        return np.empty((0, 5), dtype=np.float32)
    return np.asarray([candidates[int(index)] for index in np.asarray(indices).reshape(-1)], dtype=np.float32)


class BoxTracker:
    def __init__(self, mode: str, hold_frames: int):
        self.mode, self.hold_frames, self.tracks = mode, hold_frames, []

    def update(self, detections: np.ndarray) -> list[tuple[np.ndarray, int]]:
        unmatched = set(range(len(self.tracks)))
        if self.mode == "kalman":
            for track in self.tracks:
                track["predicted"] = track["filter"].predict()[:4, 0]
        for detection in detections:
            box = detection[:4]
            def predicted(track):
                return track.get("predicted", track["box"] + (track["velocity"] if self.mode == "velocity" else 0))

            best = max(unmatched, key=lambda i: video_privacy._box_iou(predicted(self.tracks[i]), box), default=None)
            if best is not None and video_privacy._box_iou(predicted(self.tracks[best]), box) >= 0.2:
                track = self.tracks[best]
                delta = box - track["box"]
                if self.mode == "velocity":
                    track["velocity"] = 0.5 * track["velocity"] + 0.5 * delta
                    track["box"] = box.copy()
                elif self.mode == "kalman":
                    track["box"] = track["filter"].correct(box.astype(np.float32).reshape(4, 1))[:4, 0]
                else:
                    track["box"] = box.copy()
                track["missed"] = 0; unmatched.remove(best)
            else:
                track = {"box": box.copy(), "velocity": np.zeros(4, dtype=np.float32), "missed": 0}
                if self.mode == "kalman":
                    filt = cv2.KalmanFilter(8, 4)
                    filt.transitionMatrix = np.eye(8, dtype=np.float32)
                    filt.transitionMatrix[:4, 4:] = np.eye(4, dtype=np.float32)
                    filt.measurementMatrix = np.zeros((4, 8), dtype=np.float32)
                    filt.measurementMatrix[:, :4] = np.eye(4, dtype=np.float32)
                    filt.processNoiseCov = np.eye(8, dtype=np.float32) * 1e-2
                    filt.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1e-1
                    filt.errorCovPost = np.eye(8, dtype=np.float32)
                    filt.statePost = np.vstack((box.reshape(4, 1), np.zeros((4, 1), dtype=np.float32)))
                    track["filter"] = filt
                self.tracks.append(track)
        active, retained = [], []
        for index, track in enumerate(self.tracks):
            if index in unmatched:
                if self.mode == "velocity": track["box"] = track["box"] + track["velocity"]
                if self.mode == "kalman": track["box"] = track["predicted"]
                track["missed"] += 1
            if track["missed"] <= self.hold_frames:
                retained.append(track); active.append((track["box"].copy(), track["missed"]))
        self.tracks = retained
        return active


def _tracker(mode: str, hold: int):
    if mode == "current":
        return video_privacy._FaceTracker(hold)
    if mode in {"iou", "velocity", "kalman"}:
        return BoxTracker(mode, hold)
    if mode == "none":
        return None
    raise ValueError(f"Unknown tracker: {mode}")


def _tracker_update(tracker: Any, detections: np.ndarray, mode: str):
    if mode == "none":
        return [(row[:4], 0) for row in detections]
    if mode == "current":
        rows = np.zeros((len(detections), 15), dtype=np.float32)
        if len(detections):
            rows[:, :4] = detections[:, :4]; rows[:, 14] = detections[:, 4]
        boxes = tracker.update(rows if len(rows) else None)
        return [(box, track.missed_frames) for box, track in zip(boxes, tracker._tracks)]
    return tracker.update(detections)


def evaluate(detector_kind: str, config: dict[str, Any], clips: list[Clip], weights_dir: Path,
             seed: int, bootstraps: int, failure_dir: Path | None = None) -> dict[str, Any]:
    detector = Detector(detector_kind, weights_dir, config["threshold"], config["nms"])
    counts: dict[str, dict[str, list[int]]] = {
        category: {str(c): [0, 0] for c in (0.5, 0.9, 1.0)}
        for category in ("overall", *REQUIRED_CATEGORIES)}
    clip_counts: dict[str, dict[str, dict[str, list[int]]]] = {}
    masked_area = false_area = frame_area = false_frames = empty_frames = 0
    elapsed = 0.0; frame_count = 0; failures = []; failure_images = []
    for clip in clips:
        tracker = _tracker(config["tracker"], config["hold_frames"])
        for frame_index, item in enumerate(clip.frames):
            started = time.perf_counter()
            detections = detect_multiscale(detector, item.image, config)
            tracked = _tracker_update(tracker, detections, config["tracker"])
            elapsed += time.perf_counter() - started
            height, width = item.image.shape[:2]
            output = item.image.copy(); mask = np.zeros((height, width), dtype=np.uint8)
            for box, missed in tracked:
                rect = mask_rect(box, width, height, config.get("inflation_per_miss", 0.0) * missed)
                x0, y0, x1, y1 = rect
                if x1 > x0 and y1 > y0:
                    mask[y0:y1, x0:x1] = 1
                region = output[y0:y1, x0:x1]
                if region.size:
                    reduced = (max(1, min(video_privacy.PIXEL_MAX_BLOCKS, region.shape[1] // video_privacy.PIXEL_BLOCK_SIZE)),
                               max(1, min(video_privacy.PIXEL_MAX_BLOCKS, region.shape[0] // video_privacy.PIXEL_BLOCK_SIZE)))
                    pixelated = cv2.resize(region, reduced, interpolation=cv2.INTER_AREA)
                    output[y0:y1, x0:x1] = cv2.resize(pixelated, (region.shape[1], region.shape[0]),
                                                       interpolation=cv2.INTER_NEAREST)
            gt = np.zeros_like(mask)
            for face in item.faces:
                x, y, w, h = (int(round(v)) for v in face.box)
                x0, y0, x1, y1 = max(0, x), max(0, y), min(width, x+w), min(height, y+h)
                if x1 <= x0 or y1 <= y0: continue
                gt[y0:y1, x0:x1] = 1
                ratio = coverage_ratio(face.box, mask)
                categories = ["overall"] + _categories(face, item)
                for category in categories:
                    counts.setdefault(category, {str(c): [0, 0] for c in (0.5, 0.9, 1.0)})
                    clip_counts.setdefault(category, {}).setdefault(clip.clip_id, {str(c): [0, 0] for c in (0.5, 0.9, 1.0)})
                    for cutoff in (0.5, 0.9, 1.0):
                        key = str(cutoff); exposed = int(ratio < cutoff)
                        counts[category][key][0] += exposed; counts[category][key][1] += 1
                        clip_counts[category][clip.clip_id][key][0] += exposed
                        clip_counts[category][clip.clip_id][key][1] += 1
                if ratio < 0.9:
                    failures.append({"clip_id_hash": hashlib.sha256(clip.clip_id.encode()).hexdigest()[:12],
                                     "frame": frame_index, "coverage": ratio,
                                     "categories": _categories(face, item)})
                    if failure_dir is not None:
                        visual = output.copy()
                        cv2.rectangle(visual, (x0, y0), (x1 - 1, y1 - 1), (0, 0, 255), 2)
                        cv2.putText(visual, f"GT coverage={ratio:.2f}", (max(0, x0), max(20, y0-4)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                        failure_images.append((ratio, clip.clip_id, frame_index, visual))
            masked_area += int(mask.sum()); false_area += int(np.logical_and(mask > 0, gt == 0).sum())
            frame_area += width * height; frame_count += 1
            if not item.faces:
                empty_frames += 1; false_frames += int(mask.sum() > 0)
    if failure_dir is not None:
        failure_dir.mkdir(parents=True, exist_ok=True)
        for saved, (_coverage, clip_id, frame_index, visual) in enumerate(sorted(failure_images, key=lambda row: row[0])[:100]):
            safe_clip_id = clip_id.replace(":", "_").replace("/", "_")
            cv2.imwrite(str(failure_dir / f"failure_{saved:03d}_{safe_clip_id}_{frame_index}.jpg"), visual)
    saved = min(100, len(failure_images))
    metrics = {}
    for category, category_counts in counts.items():
        metrics[category] = {}
        for cutoff in (0.5, 0.9, 1.0):
            key = str(cutoff); exposed, total = category_counts[key]
            cluster = {clip_id: (value[key][0], value[key][1])
                       for clip_id, value in clip_counts.get(category, {}).items()}
            metrics[category][key] = {"r_exposed": exposed / total if total else None,
                                      "exposed": exposed, "visible_face_frames": total,
                                      "ci95": _rate_ci(cluster, seed + int(cutoff*100), bootstraps)}
    model_path = detector.model_path
    weight_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    return {"metrics": metrics, "over_masking_area_fraction": false_area / frame_area if frame_area else None,
            "false_mask_rate": false_frames / empty_frames if empty_frames else None,
            "false_mask_frames": false_frames, "empty_frames": empty_frames,
            "masked_area_fraction": masked_area / frame_area if frame_area else None,
            "ms_per_frame": elapsed * 1000 / frame_count if frame_count else None,
            "frame_count": frame_count, "failure_count_r90": len(failures),
            "failure_frames_saved": saved, "failures": failures,
            "weights": {"source": video_privacy.YUNET_MODEL_URL if detector_kind == "yunet" else
                        "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_sc.zip (det_500m.onnx)",
                        "sha256": weight_hash, "path_name": model_path.name}}


def config_base(detector: str) -> dict[str, Any]:
    return {"detector": detector, "threshold": 0.35, "nms": 0.35,
            "tile_scale": video_privacy.FACE_TILE_SCALE, "tile_overlap": 1 - (1 - video_privacy.FACE_TILE_SCALE) / video_privacy.FACE_TILE_SCALE,
            "tile_grid": 2, "tracker": "current", "fps": 12,
            "hold_seconds": video_privacy.FACE_TRACK_HOLD_SECONDS,
            "hold_frames": max(video_privacy.FACE_TRACK_MIN_HOLD_FRAMES,
                                int(round(12 * video_privacy.FACE_TRACK_HOLD_SECONDS))),
            "inflation_per_miss": 0.0, "margins": {"x": video_privacy.FACE_MARGIN_X,
            "top": video_privacy.FACE_MARGIN_TOP, "bottom": video_privacy.FACE_MARGIN_BOTTOM}}


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return os.environ.get("GIT_COMMIT", "not-included-in-container-copy")


def save_trial(step: str, split: str, config: dict[str, Any], result: dict[str, Any],
               clips: list[Clip], results_dir: Path, seed: int, commit: str,
               dataset_info: dict[str, Any]) -> Path:
    payload = {"schema_version": 1, "step": step, "split": split, "proxy_data": True,
               "seed": seed, "clip_builder_version": CLIP_BUILDER_VERSION,
               "git_commit": commit, "config": config,
               "dataset": {"source": "WIDER FACE train annotations and public still-image mirror; synthetic clips (proxy)",
                           "clips": len(clips), "frames": sum(len(clip.frames) for clip in clips),
                           "source_images": len({source for clip in clips for source in clip.source_ids}),
                           **dataset_info},
               **result}
    results_dir.mkdir(parents=True, exist_ok=True)
    config_id = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]
    path = results_dir / f"face_blur_{step}_{split}_{config_id}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    log_result(path)
    return path


def _all_results(results_dir: Path, step: str, split: str) -> list[dict[str, Any]]:
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(results_dir.glob(f"face_blur_{step}_{split}_*.json"))]


def _score(row: dict[str, Any]) -> tuple[float, float, float]:
    rate = row["metrics"]["overall"]["0.9"]["r_exposed"]
    return (float("inf") if rate is None else rate,
            row.get("over_masking_area_fraction") or 0.0,
            row.get("ms_per_frame") or float("inf"))


def run_step(step: str, sources: list[SourceImage], assignments: dict[str, str], frames: int,
             weights_dir: Path, results_dir: Path, failure_dir: Path, bootstraps: int,
             seed: int, commit: str, dataset_info: dict[str, Any]) -> list[Path]:
    all_paths = []
    if step == "baseline":
        trials = [(kind, config_base(kind)) for kind in ("yunet", "scrfd")]
        for kind, config in trials:
            for split in ("tune", "heldout"):
                clips = build_proxy_clips(sources, assignments, split, frames)
                result = evaluate(kind, config, clips, weights_dir, seed, bootstraps,
                                  failure_dir if step == "baseline" and split == "heldout" else None)
                all_paths.append(save_trial(step, split, config, result, clips, results_dir, seed, commit, dataset_info))
    elif step == "tune":
        baseline = _all_results(results_dir, "baseline", "tune")
        if not baseline:
            raise FileNotFoundError("Run baseline before threshold tuning")
        selected_detector = min(baseline, key=_score)["config"]["detector"]
        candidates = [(selected_detector, {**config_base(selected_detector), "threshold": threshold, "nms": nms})
                      for threshold in (0.25, 0.35, 0.45) for nms in (0.25, 0.35, 0.45)]
        tune_clips = build_proxy_clips(sources, assignments, "tune", frames)
        for kind, config in candidates:
            result = evaluate(kind, config, tune_clips, weights_dir, seed, bootstraps)
            all_paths.append(save_trial(step, "tune", config, result, tune_clips,
                                        results_dir, seed, commit, dataset_info))
        tuned = sorted(_all_results(results_dir, step, "tune"), key=_score)
        best = tuned[0]
        summary_path = results_dir / "face_blur_tune_grid_summary.json"
        summary = {"step": "tune", "split": "tune", "proxy_data": True,
                   "selection_rule": "lowest tuning R_exposed_90; ties by lower over-mask area then lower ms/frame",
                   "detector": selected_detector, "grid_trial_count": len(tuned),
                   "tune_results": [{"config": row["config"], "metrics": row["metrics"],
                                     "over_masking_area_fraction": row["over_masking_area_fraction"],
                                     "false_mask_rate": row["false_mask_rate"],
                                     "ms_per_frame": row["ms_per_frame"]} for row in tuned],
                   "heldout_top_n": min(3, len(tuned)),
                   "heldout_configs": [row["config"] for row in tuned[:3]]}
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        for row in tuned[:3]:
            config = row["config"]
            heldout_clips = build_proxy_clips(sources, assignments, "heldout", frames)
            result = evaluate(selected_detector, config, heldout_clips, weights_dir, seed, bootstraps)
            all_paths.append(save_trial(step, "heldout", config, result, heldout_clips,
                                        results_dir, seed, commit, dataset_info))
        (results_dir / "face_blur_selected_config.json").write_text(json.dumps(best["config"], indent=2), encoding="utf-8")
    else:
        selection_path = results_dir / "face_blur_selected_config.json"
        if not selection_path.is_file():
            raise FileNotFoundError("Run baseline and tune before later experiment steps")
        base = json.loads(selection_path.read_text(encoding="utf-8"))
        if step == "tiles":
            candidates = [{**base, "tile_scale": .6, "tile_overlap": 1 - .4/.6, "tile_grid": 2},
                          {**base, "tile_scale": .7, "tile_overlap": .25, "tile_grid": 3},
                          {**base, "tile_scale": .5, "tile_overlap": .3, "tile_grid": 3},
                          {**base, "tile_scale": .6, "tile_overlap": .3, "tile_grid": 3}]
        elif step == "track":
            candidates = [{**base, "tracker": mode, "hold_frames": base["hold_frames"],
                           "inflation_per_miss": 0.0} for mode in ("current", "none", "iou", "kalman")]
        elif step == "conservative":
            hold = base["hold_frames"]
            candidates = [{**base, "threshold": base["threshold"], "hold_frames": hold, "inflation_per_miss": 0.0},
                          {**base, "threshold": max(.15, base["threshold"] - .15), "hold_frames": hold, "inflation_per_miss": 0.0},
                          {**base, "threshold": base["threshold"], "hold_frames": hold * 3, "inflation_per_miss": .05},
                          {**base, "threshold": max(.15, base["threshold"] - .15), "hold_frames": hold * 3, "inflation_per_miss": .10}]
        elif step == "final":
            candidates = [base]
        else:
            raise ValueError(f"Unknown step {step}")
        for config in candidates:
            for split in ("tune", "heldout"):
                clips = build_proxy_clips(sources, assignments, split, frames)
                result = evaluate(config["detector"], config, clips, weights_dir, seed, bootstraps,
                                  failure_dir if step == "final" and split == "heldout" else None)
                all_paths.append(save_trial(step, split, config, result, clips, results_dir, seed, commit, dataset_info))
        selected = min(_all_results(results_dir, step, "tune"), key=_score)
        if step != "final":
            selection_path.write_text(json.dumps(selected["config"], indent=2), encoding="utf-8")
    return all_paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--weights-dir", type=Path, required=True, help="External cache directory; never stored in repo")
    parser.add_argument("--results-dir", type=Path, default=Path("tools/results"))
    parser.add_argument("--failure-dir", type=Path, required=True, help="External directory for failure frames")
    parser.add_argument("--step", choices=("baseline", "tune", "tiles", "track", "conservative", "final"), required=True)
    parser.add_argument("--image-limit", type=int, default=32)
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--bootstraps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--git-commit", default=None)
    args = parser.parse_args()
    sources = read_wider_annotations(args.annotations, args.images_root, args.image_limit, args.seed)
    if len(sources) < 4:
        raise SystemExit(f"Need at least four labeled source images; found {len(sources)}")
    assignments = source_split([source.source_id for source in sources], args.seed)
    if set(assignments.values()) != {"tune", "heldout"}:
        raise SystemExit("The sample must include source images in both tune and heldout splits")
    commit = args.git_commit or _git_commit()
    manifest = hashlib.sha256()
    for source in sources:
        manifest.update(hashlib.sha256(source.source_id.encode("utf-8")).digest())
        manifest.update(hashlib.sha256(source.path.read_bytes()).digest())
    dataset_info = {"annotation_sha256": hashlib.sha256(args.annotations.read_bytes()).hexdigest(),
                    "source_manifest_sha256": manifest.hexdigest(),
                    "annotation_revision": ANNOTATION_REVISION,
                    "image_revision": IMAGE_REVISION,
                    "dataset_version": "WIDER FACE train bounding-box/attribute annotations; public HF mirror"}
    paths = run_step(args.step, sources, assignments, args.frames, args.weights_dir,
                     args.results_dir, args.failure_dir, args.bootstraps, args.seed, commit, dataset_info)
    print(json.dumps({"step": args.step, "source_images": len(sources),
                      "tune_sources": sum(value == "tune" for value in assignments.values()),
                      "heldout_sources": sum(value == "heldout" for value in assignments.values()),
                      "results": [str(path) for path in paths]}, indent=2))


if __name__ == "__main__":
    main()
