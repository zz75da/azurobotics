"""Pipeline step 2: train the anomaly model on train.csv with the app's own model code.

Uses mlops.py (feature columns, hyperparameter validation, model builders) and writes the model in
the artifact format the app loads: a joblib dict with "model", "reference" and "features". The
default hyperparameters are the app's tuned AI4I preset. Inside an Azure ML job, MLflow logs to the
workspace automatically; elsewhere it logs only when MLFLOW_TRACKING_URI is set.
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

from mlops import (  # noqa: E402
    FEATURES,
    OPTIONAL_FEATURES,
    REFERENCE_MAX_ROWS,
    _build_model,
    _read_csv,
    _sample_rows,
    _to_training_data,
    _validate_hyperparameters,
)
from pipelines import PRESETS, TUNED_PRESET  # noqa: E402


def run_train(train_csv: Path, model_dir: Path, hyperparameters: Dict[str, Any] | None = None) -> Dict[str, Any]:
    hp = _validate_hyperparameters(hyperparameters or PRESETS[TUNED_PRESET]["hyperparameters"])
    if hp["algorithm"] == "auto":
        raise ValueError("Choose one algorithm: model selection needs the test split, which this step must not see.")
    records = _read_csv(train_csv.read_text(encoding="utf-8"), require_labels=True)
    columns = FEATURES + tuple(name for name in OPTIONAL_FEATURES if name in records[0])
    features, labels = _to_training_data(records, columns)
    model = _build_model(hp["algorithm"], hp, float(labels.mean()))
    model.fit(features, labels) if hp["algorithm"] == "random_forest" else model.fit(features)

    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "reference": _sample_rows(features, REFERENCE_MAX_ROWS), "features": columns},
        model_dir / "model.joblib",
        compress=3,
    )
    info = {"algorithm": hp["algorithm"], "features": list(columns), "train_rows": len(features), "hyperparameters": hp}
    (model_dir / "train_info.json").write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
    _log_to_mlflow(hp, info)
    return info


def _log_to_mlflow(hp: Dict[str, Any], info: Dict[str, Any]) -> None:
    if not (os.environ.get("MLFLOW_TRACKING_URI") or os.environ.get("AZUREML_RUN_ID")):
        return
    import mlflow

    mlflow.log_params({key: str(value) for key, value in hp.items()})
    mlflow.log_metric("train_rows", info["train_rows"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--prepared", required=True, help="output folder of the prep step")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--hyperparameters", default="", help="JSON object; the tuned AI4I preset when empty")
    args = parser.parse_args()
    hp = json.loads(args.hyperparameters) if args.hyperparameters.strip() else None
    print(json.dumps(run_train(Path(args.prepared) / "train.csv", Path(args.model_dir), hp), default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
