import json
from pathlib import Path

import numpy as np

from tools.face_blur_benchmark import (
    Clip,
    ClipFrame,
    Face,
    SourceImage,
    build_clip,
    build_transition_clip,
    coverage_ratio,
    mask_rect,
    source_split,
)


def test_coverage_metric_uses_covered_gt_face_area():
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5:15, 5:15] = 1
    assert coverage_ratio([5, 5, 10, 10], mask) == 1.0
    assert coverage_ratio([0, 0, 10, 10], mask) == 0.25


def test_mask_rect_matches_production_margin_geometry():
    assert mask_rect([10, 10, 20, 20], 100, 100) == (6, 5, 34, 35)


def test_source_split_is_deterministic_and_source_level():
    first = source_split(["a", "b", "c", "d"], seed=4)
    assert first == source_split(["d", "c", "b", "a"], seed=4)
    assert set(first.values()) == {"tune", "heldout"}
    split = source_split([f"source-{i}" for i in range(12)], seed=20261025)
    assert sum(value == "heldout" for value in split.values()) == 4


def test_clip_builder_transforms_gt_for_each_frame(tmp_path: Path):
    image = np.full((80, 100, 3), 127, dtype=np.uint8)
    path = tmp_path / "still.jpg"
    import cv2
    cv2.imwrite(str(path), image)
    clip = build_clip(SourceImage("source", path, [Face([20, 20, 20, 20])]), "clip", frames=3)
    assert len(clip.frames) == 3
    assert all(len(frame.faces) == 1 for frame in clip.frames)
    assert clip.frames[-1].faces[0].box[2] > clip.frames[0].faces[0].box[2]


def test_transition_requires_disjoint_source_images():
    frame = ClipFrame(np.zeros((4, 4, 3), dtype=np.uint8), [])
    first, second = Clip("a", ("same",), [frame]), Clip("b", ("same",), [frame])
    try:
        build_transition_clip(first, second, "cut")
    except ValueError:
        pass
    else:
        raise AssertionError("same source must not be treated as a hard cut")


def test_transition_marks_first_post_cut_frame():
    frame_a = ClipFrame(np.zeros((4, 4, 3), dtype=np.uint8), [])
    frame_b = ClipFrame(np.zeros((4, 4, 3), dtype=np.uint8), [])
    merged = build_transition_clip(Clip("a", ("a",), [frame_a]), Clip("b", ("b",), [frame_b]), "cut")
    assert merged.frames[0].scene_transition is False
    assert merged.frames[1].scene_transition is True


def test_replay_flattens_required_metric_names():
    from tools.replay_to_mlflow import flatten_metrics
    metrics = flatten_metrics({"metrics": {"overall": {"0.9": {
        "r_exposed": 0.2, "ci95": [0.1, 0.3]}}, "low_light": {"0.9": {
        "r_exposed": 0.4, "ci95": [0.2, 0.6]}}},
        "over_masking_area_fraction": 0.05, "false_mask_rate": 0.1, "ms_per_frame": 20})
    assert metrics["r_exposed_90"] == 0.2
    assert metrics["r_exposed_90_ci95_lower"] == 0.1
    assert metrics["low_light_r_exposed_90"] == 0.4


def test_replay_logs_local_run_without_network(tmp_path: Path, monkeypatch):
    from tools.replay_to_mlflow import log_result
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    uri = (tmp_path / "mlruns").resolve().as_uri()
    result = {"schema_version": 1, "step": "unit", "split": "tune", "proxy_data": True,
              "seed": 1, "clip_builder_version": "test", "git_commit": "abc",
              "config": {"detector": "yunet", "threshold": .35, "nms": .35,
                         "tile_scale": .6, "tile_overlap": .3, "tile_grid": 2,
                         "tracker": "none", "hold_frames": 0, "inflation_per_miss": 0},
              "weights": {"source": "local-test", "sha256": "0" * 64},
              "dataset": {"source_images": 1, "clips": 1, "frames": 1},
              "metrics": {"overall": {"0.9": {"r_exposed": .5, "ci95": [.5, .5],
                  "exposed": 1, "visible_face_frames": 2}}},
              "over_masking_area_fraction": 0.0, "false_mask_rate": 0.0,
              "ms_per_frame": 1.0, "failures": []}
    path = tmp_path / "result.json"
    path.write_text(json.dumps(result), encoding="utf-8")
    run_id = log_result(path, tracking_uri=uri, experiment_name="unit-test-face-blur")
    assert run_id
    import mlflow
    mlflow.set_tracking_uri(uri)
    experiment = mlflow.get_experiment_by_name("unit-test-face-blur")
    assert experiment is not None
    runs = mlflow.search_runs([experiment.experiment_id])
    assert len(runs) == 1
    assert runs.iloc[0]["metrics.r_exposed_90"] == .5
    client = mlflow.tracking.MlflowClient(tracking_uri=uri)
    artifacts = {item.path for item in client.list_artifacts(run_id)}
    assert {"result.json", "config.json", "failure_summary.json", "git_commit.txt"} <= artifacts

def test_wider_reader_keeps_empty_negative_images(tmp_path: Path):
    import cv2

    from tools.face_blur_benchmark import read_wider_annotations
    root = tmp_path / "images"
    for name in ("negative.jpg", "positive.jpg"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), np.zeros((20, 20, 3), dtype=np.uint8))
    annotations = tmp_path / "wider.txt"
    annotations.write_text(
        "negative.jpg\n0\n0 0 0 0 0 0 0 0 0 0\n"
        "positive.jpg\n1\n2 3 10 12 0 0 0 0 0 0\n", encoding="utf-8")
    records = read_wider_annotations(annotations, root)
    assert len(records) == 2
    assert records[0].faces == []
    assert len(records[1].faces) == 1



def test_current_tracker_reports_missed_frames_for_conservative_inflation():
    from tools.face_blur_benchmark import _tracker, _tracker_update
    tracker = _tracker("current", 3)
    detection = np.asarray([[10, 10, 20, 20, 0.9]], dtype=np.float32)
    assert _tracker_update(tracker, detection, "current")[0][1] == 0
    held = _tracker_update(tracker, np.empty((0, 5), dtype=np.float32), "current")
    assert held[0][1] == 1


def test_kalman_tracker_matches_and_holds_a_moving_box():
    from tools.face_blur_benchmark import BoxTracker
    tracker = BoxTracker("kalman", 2)
    first = np.asarray([[10, 10, 20, 20, .9]], dtype=np.float32)
    moved = np.asarray([[11, 10, 20, 20, .9]], dtype=np.float32)
    assert len(tracker.update(first)) == 1
    matched = tracker.update(moved)
    assert len(matched) == 1
    assert matched[0][1] == 0
    missed = tracker.update(np.empty((0, 5), dtype=np.float32))
    assert len(missed) == 1
    assert missed[0][1] == 1
