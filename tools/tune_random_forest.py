"""Tune the supervised Random Forest option on a labelled telemetry CSV.

Usage:
    python tools/tune_random_forest.py --input ai4i_telemetry.csv --output sweep_results.json

The decision threshold is chosen by 5-fold cross-validation on the training split only, and the
configurations are ranked by that CV F1. The held-out test columns are reported for information
and are never used for selection. The split matches the app (test_size 0.25, random_state 42,
stratified). The JSON can be sent to POST /api/ml/train as "tuning_results" so the sweep is
stored with the MLflow run.
"""
from __future__ import annotations

import argparse
import itertools
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

FEATURES = ["duration_ms", "battery_pct", "x_m", "y_m"]
OPTIONAL_FEATURES = ["air_temp_k", "temp_delta_k", "power_w", "wear_torque", "type_code"]
GRID = {
    "n_estimators": [100, 300],
    "max_depth": [None, 8, 14],
    "min_samples_leaf": [1, 3, 5],
    "class_weight": ["balanced", "balanced_subsample", None],
    "max_features": [0.5, 1.0],
}


def tune(frame: pd.DataFrame) -> list[dict]:
    columns = FEATURES + [name for name in OPTIONAL_FEATURES if name in frame.columns]
    x, y = frame[columns], frame["is_anomaly"]
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.25, random_state=42, stratify=y)
    folds = StratifiedKFold(5, shuffle=True, random_state=42)
    results = []
    for values in itertools.product(*GRID.values()):
        params = dict(zip(GRID, values))
        estimator = dict(params, random_state=42, n_jobs=-1)
        oof = cross_val_predict(RandomForestClassifier(**estimator), train_x, train_y, cv=folds, method="predict_proba")[:, 1]
        threshold, cv_f1 = max(
            ((float(t), f1_score(train_y, (oof >= t).astype(int), zero_division=0)) for t in np.arange(0.1, 0.9, 0.02)),
            key=lambda item: item[1],
        )
        proba = RandomForestClassifier(**estimator).fit(train_x, train_y).predict_proba(test_x)[:, 1]
        predicted = (proba >= threshold).astype(int)
        results.append(
            {
                **params,
                "threshold": round(threshold, 2),
                "cv_f1": round(float(cv_f1), 4),
                "test_f1": round(float(f1_score(test_y, predicted, zero_division=0)), 4),
                "test_precision": round(float(precision_score(test_y, predicted, zero_division=0)), 4),
                "test_recall": round(float(recall_score(test_y, predicted, zero_division=0)), 4),
                "test_auc": round(float(roc_auc_score(test_y, proba)), 4),
            }
        )
    return sorted(results, key=lambda item: item["cv_f1"], reverse=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="sweep_results.json")
    args = parser.parse_args()
    results = tune(pd.read_csv(args.input))
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=1)
    print("{} configurations; best by CV F1: {}".format(len(results), results[0]))


if __name__ == "__main__":
    main()
