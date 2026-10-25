"""Replay saved face-blur benchmark JSON runs into an MLflow tracking store."""
from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

EXPERIMENT_NAME = "face-blur-privacy"


def _mlflow():
    try:
        import mlflow
    except ImportError as error:
        raise RuntimeError("Install mlflow-skinny==3.17.2 to replay benchmark results") from error
    return mlflow


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", value).strip("_").lower()


def flatten_metrics(payload: dict[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for threshold, value in payload.get("metrics", {}).get("overall", {}).items():
        label = str(round(float(threshold) * 100))
        if value.get("r_exposed") is not None:
            metrics[f"r_exposed_{label}"] = float(value["r_exposed"])
        metrics[f"visible_face_frames_{label}"] = float(value.get("visible_face_frames", 0))
        ci = value.get("ci95") or [None, None]
        for bound, number in zip(("ci95_lower", "ci95_upper"), ci):
            if number is not None:
                metrics[f"r_exposed_{label}_{bound}"] = float(number)
    for category, thresholds in payload.get("metrics", {}).items():
        if category == "overall":
            continue
        name = _safe_name(category)
        for threshold, value in thresholds.items():
            label = str(round(float(threshold) * 100))
            if value.get("r_exposed") is not None:
                metrics[f"{name}_r_exposed_{label}"] = float(value["r_exposed"])
            metrics[f"{name}_visible_face_frames_{label}"] = float(value.get("visible_face_frames", 0))
            ci = value.get("ci95") or [None, None]
            for bound, number in zip(("ci95_lower", "ci95_upper"), ci):
                if number is not None:
                    metrics[f"{name}_r_exposed_{label}_{bound}"] = float(number)
    for name in ("over_masking_area_fraction", "false_mask_rate", "ms_per_frame"):
        value = payload.get(name)
        if value is not None:
            metrics[name] = float(value)
    return metrics


def _params(payload: dict[str, Any]) -> dict[str, str | float | int | bool]:
    config = payload.get("config", {})
    weights = payload.get("weights", {})
    params: dict[str, str | float | int | bool] = {
        "detector": str(config.get("detector", "unknown")),
        "weights_source": str(weights.get("source", "unknown")),
        "weights_sha256": str(weights.get("sha256", "unknown")),
        "threshold": float(config.get("threshold", 0)),
        "nms": float(config.get("nms", 0)),
        "tile_scale": float(config.get("tile_scale", 0)),
        "tile_overlap": float(config.get("tile_overlap", 0)),
        "tile_grid": int(config.get("tile_grid", 0)),
        "tracker": str(config.get("tracker", "unknown")),
        "fps": int(config.get("fps", 0)),
        "hold_seconds": float(config.get("hold_seconds", 0)),
        "hold_frames": int(config.get("hold_frames", 0)),
        "inflation_per_miss": float(config.get("inflation_per_miss", 0)),
        "margins": json.dumps(config.get("margins", {}), sort_keys=True),
        "split": str(payload.get("split", "unknown")),
        "seed": int(payload.get("seed", 0)),
        "clip_builder_version": str(payload.get("clip_builder_version", "unknown")),
        "source_images": int(payload.get("dataset", {}).get("source_images", 0)),
        "clips": int(payload.get("dataset", {}).get("clips", 0)),
        "frames": int(payload.get("dataset", {}).get("frames", 0)),
    }
    return params


def log_result(result_path: Path, tracking_uri: str | None = None,
               experiment_name: str = EXPERIMENT_NAME) -> str:
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow = _mlflow()
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    elif not os.environ.get("MLFLOW_TRACKING_URI"):
        local_store = Path(tempfile.gettempdir()) / "face-blur-mlflow"
        mlflow.set_tracking_uri(local_store.resolve().as_uri())
    mlflow.set_experiment(experiment_name)
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    config = payload.get("config", {})
    tags = {"step": str(payload.get("step", "unknown")),
            "split": str(payload.get("split", "unknown")),
            "proxy_data": "true",
            "clip_builder_version": str(payload.get("clip_builder_version", "unknown")),
            "git_commit": str(payload.get("git_commit", "unknown"))}
    with mlflow.start_run(run_name=f"{tags['step']}-{tags['split']}-{result_path.stem}") as run:
        mlflow.log_params(_params(payload))
        mlflow.set_tags(tags)
        metrics = flatten_metrics(payload)
        if metrics:
            mlflow.log_metrics(metrics)
        mlflow.log_dict(payload, "result.json")
        mlflow.log_dict(config, "config.json")
        mlflow.log_dict({"failure_count_r90": payload.get("failure_count_r90", 0),
                         "failure_frames_saved": payload.get("failure_frames_saved", 0),
                         "failures": payload.get("failures", [])}, "failure_summary.json")
        mlflow.log_text(tags["git_commit"] + "\n", "git_commit.txt")
        summary_path = result_path.parent / "face_blur_tune_grid_summary.json"
        if tags["step"] == "tune" and summary_path.is_file():
            mlflow.log_artifact(str(summary_path), artifact_path=None)
    return run.info.run_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("tools/results"))
    parser.add_argument("--glob", default="face_blur_*.json")
    parser.add_argument("--tracking-uri", default=None)
    parser.add_argument("--experiment-name", default=EXPERIMENT_NAME)
    args = parser.parse_args()
    paths = []
    for path in sorted(args.results_dir.glob(args.glob)):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("schema_version") == 1 and "metrics" in payload and "config" in payload:
            paths.append(path)
    if not paths:
        raise SystemExit(f"No benchmark result JSON files matched {args.results_dir / args.glob}")
    ids = [log_result(path, args.tracking_uri, args.experiment_name) for path in paths]
    print(json.dumps({"logged": len(ids), "run_ids": ids}, indent=2))


if __name__ == "__main__":
    main()
