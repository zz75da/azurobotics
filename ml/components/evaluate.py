"""Pipeline step 3: score the model on the held-out test.csv and enforce the F1 quality gate.

Writes metrics.json (F1, ROC AUC, precision, recall, gate result) to --metrics-out. If F1 is below
--f1-threshold the step exits non-zero, which fails the Azure ML pipeline, so nothing is registered.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import joblib  # noqa: E402
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score  # noqa: E402

from mlops import PROMOTION_THRESHOLDS, _read_csv, _to_training_data  # noqa: E402


class QualityGateError(RuntimeError):
    """The model's F1 on the held-out split is below the threshold."""


def run_evaluate(model_path: Path, test_csv: Path, metrics_dir: Path, f1_threshold: float) -> Dict[str, Any]:
    artifact = joblib.load(model_path)
    model, columns = artifact["model"], tuple(artifact["features"])
    records = _read_csv(test_csv.read_text(encoding="utf-8"), require_labels=True)
    features, labels = _to_training_data(records, columns)
    predicted = (model.predict(features) == -1).astype(int)
    metrics = {
        "f1": round(float(f1_score(labels, predicted, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(labels, -model.decision_function(features))), 4),
        "precision": round(float(precision_score(labels, predicted, zero_division=0)), 4),
        "recall": round(float(recall_score(labels, predicted, zero_division=0)), 4),
        "test_rows": len(features),
        "f1_threshold": f1_threshold,
    }
    metrics["gate_passed"] = metrics["f1"] >= f1_threshold
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _log_to_mlflow(metrics)
    if not metrics["gate_passed"]:
        raise QualityGateError("F1 {} is below the threshold {}.".format(metrics["f1"], f1_threshold))
    return metrics


def _log_to_mlflow(metrics: Dict[str, Any]) -> None:
    if not (os.environ.get("MLFLOW_TRACKING_URI") or os.environ.get("AZUREML_RUN_ID")):
        return
    import mlflow

    mlflow.log_metrics({key: float(value) for key, value in metrics.items() if key != "gate_passed"})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model-dir", required=True, help="output folder of the train step")
    parser.add_argument("--prepared", required=True, help="output folder of the prep step")
    parser.add_argument("--metrics-out", required=True)
    parser.add_argument("--f1-threshold", type=float, default=PROMOTION_THRESHOLDS["f1"])
    args = parser.parse_args()
    try:
        metrics = run_evaluate(
            Path(args.model_dir) / "model.joblib", Path(args.prepared) / "test.csv", Path(args.metrics_out), args.f1_threshold
        )
    except QualityGateError as error:
        print("QUALITY GATE FAILED: {}".format(error), file=sys.stderr)
        return 1
    print(json.dumps(metrics))
    return 0


if __name__ == "__main__":
    sys.exit(main())
