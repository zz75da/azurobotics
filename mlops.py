from __future__ import annotations

import json
import logging
import math
import os
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset
from evidently.ui.workspace import Workspace
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from robot_data import make_sample_telemetry_csv
from experiment_tracking import digest_text, log_training_run


FEATURES = ("duration_ms", "battery_pct", "x_m", "y_m")
MODEL_NAME = "robot-telemetry-isolation-forest"
EVIDENTLY_PROJECT_NAME = "Marso Robotics - Telemetry Drift"
MIN_TRAINING_ROWS = 40
MAX_PREDICTION_ROWS = 200_000
PREDICTION_CHUNK_ROWS = 10_000
DRIFT_SAMPLE_ROWS = 20_000
REFERENCE_MAX_ROWS = 20_000
MODEL_RETENTION = 3
MODEL_HISTORY_RETENTION = 100
EVIDENTLY_RUN_RETENTION = 500
PROMOTION_THRESHOLDS = {"roc_auc": 0.75, "f1": 0.5}
LOG = logging.getLogger("robotics-demo.mlops")
ALGORITHMS = ("isolation_forest", "random_forest", "auto")
DEFAULT_HYPERPARAMETERS: Dict[str, Any] = {
    "algorithm": "isolation_forest",
    "n_estimators": 150,
    "max_samples": "auto",
    "max_features": 1.0,
    "test_size": 0.25,
    "random_state": 42,
}


class RandomForestAnomalyDetector(RandomForestClassifier):
    """Supervised classifier exposing the IsolationForest contract.

    predict() returns -1 for an anomaly and 1 otherwise; decision_function() is higher for
    normal rows. Prediction, threshold selection and MLflow code can treat both models alike.
    """

    def _anomaly_probability(self, X):
        proba = self.predict_proba(X)
        return proba[:, list(self.classes_).index(1)]

    def predict(self, X):
        return np.where(self._anomaly_probability(X) >= 0.5, -1, 1)

    def decision_function(self, X):
        return -self._anomaly_probability(X)


def _build_model(algorithm: str, hp: Dict[str, Any], contamination: float):
    if algorithm == "random_forest":
        return RandomForestAnomalyDetector(
            n_estimators=hp["n_estimators"],
            max_features=hp["max_features"],
            class_weight="balanced",
            random_state=hp["random_state"],
            n_jobs=1,
        )
    return IsolationForest(
        n_estimators=hp["n_estimators"],
        max_samples=hp["max_samples"],
        max_features=hp["max_features"],
        contamination=contamination,
        random_state=hp["random_state"],
        n_jobs=1,
    )


def _select_threshold(model: IsolationForest, train_x, train_y, test_x, test_y) -> Dict[str, float]:
    """Pick the F1-optimal score threshold on the training split, then report it on the test split.

    Informational only: the promotion gate still uses the model's default decision rule.
    """
    try:
        train_scores = -model.decision_function(train_x)
        precision, recall, thresholds = precision_recall_curve(train_y, train_scores)
        if len(thresholds) == 0:
            return {}
        f1_curve = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
        threshold = float(thresholds[int(np.argmax(f1_curve))])
        predicted = (-model.decision_function(test_x) >= threshold).astype(int)
        return {
            "decision_threshold": round(threshold, 4),
            "f1_tuned": round(float(f1_score(test_y, predicted, zero_division=0)), 4),
            "precision_tuned": round(float(precision_score(test_y, predicted, zero_division=0)), 4),
            "recall_tuned": round(float(recall_score(test_y, predicted, zero_division=0)), 4),
        }
    except Exception:
        LOG.exception("threshold_selection_failed")
        return {}


def _validate_hyperparameters(overrides: Dict[str, Any] | None) -> Dict[str, Any]:
    hp = dict(DEFAULT_HYPERPARAMETERS)
    if not overrides:
        return hp
    if not isinstance(overrides, dict):
        raise ValueError("hyperparameters doit être un objet JSON.")
    unknown = set(overrides) - set(hp)
    if unknown:
        raise ValueError("Hyperparamètres inconnus : {}.".format(", ".join(sorted(unknown))))
    hp.update(overrides)
    if hp["algorithm"] not in ALGORITHMS:
        raise ValueError("algorithm doit valoir : {}.".format(", ".join(ALGORITHMS)))
    for name in ("n_estimators", "random_state"):
        if isinstance(hp[name], bool) or not isinstance(hp[name], int):
            raise ValueError("{} doit être un entier.".format(name))
    if not 10 <= hp["n_estimators"] <= 1000:
        raise ValueError("n_estimators doit être compris entre 10 et 1000.")
    if not 0 <= hp["random_state"] <= 2**31 - 1:
        raise ValueError("random_state est hors limites.")
    if isinstance(hp["test_size"], bool) or not isinstance(hp["test_size"], (int, float)) or not 0.1 <= hp["test_size"] <= 0.5:
        raise ValueError("test_size doit être compris entre 0.1 et 0.5.")
    if isinstance(hp["max_features"], bool) or not isinstance(hp["max_features"], (int, float)) or not 0 < hp["max_features"] <= 1:
        raise ValueError("max_features doit être compris entre 0 (exclu) et 1.")
    ms = hp["max_samples"]
    if ms != "auto" and (isinstance(ms, bool) or not isinstance(ms, (int, float)) or ms <= 0):
        raise ValueError("max_samples doit valoir 'auto' ou un nombre positif.")
    return hp


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class MLOpsService:
    def __init__(self, database_path: Path, artifacts_path: Path, evidently_path: Path) -> None:
        self.database_path = database_path
        self.artifacts_path = artifacts_path
        self.evidently_path = evidently_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.artifacts_path.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._prediction_counts = {"normal": 0, "anomaly": 0}
        self._drift_reports = 0
        self._initialize()
        self._workspace = Workspace.create(str(self.evidently_path))
        projects = self._workspace.search_project(EVIDENTLY_PROJECT_NAME)
        if projects:
            self._evidently_project = projects[0]
        else:
            self._evidently_project = self._workspace.create_project(EVIDENTLY_PROJECT_NAME)
            self._evidently_project.description = (
                "Rapports de dérive de la télémétrie synthétique ou importée. "
                "Les données brutes ne sont pas conservées dans les snapshots."
            )
            self._evidently_project.save()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.database_path), timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 10000")
        return connection

    def _initialize(self) -> None:
        paths_to_remove: List[Path] = []
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS ml_models (
                    version TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    artifact_path TEXT,
                    created_at TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    training_rows INTEGER NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS ml_prediction_batches (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    records_processed INTEGER NOT NULL,
                    anomalies_detected INTEGER NOT NULL,
                    evidently_snapshot_id TEXT NOT NULL
                )
                """
            )

    def train(
        self,
        csv_content: str | None = None,
        hyperparameters: Dict[str, Any] | None = None,
        description: str | None = None,
        run_tags: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        hp = _validate_hyperparameters(hyperparameters)
        data_source = "synthetic-sample" if csv_content is None else "uploaded-csv"
        source = make_sample_telemetry_csv() if csv_content is None else csv_content
        records = _read_csv(source, require_labels=True)
        if len(records) < MIN_TRAINING_ROWS:
            raise ValueError(
                "L'entraînement nécessite au moins {} lignes valides.".format(MIN_TRAINING_ROWS)
            )
        features, labels = _to_training_data(records)
        class_counts = labels.value_counts()
        if len(class_counts) != 2:
            raise ValueError("La colonne is_anomaly doit contenir les deux classes 0 et 1.")
        if int(class_counts.min()) < 2:
            raise ValueError("Chaque classe de is_anomaly doit contenir au moins deux lignes.")
        contamination = float(labels.mean())
        if contamination > 0.5:
            raise ValueError("Les anomalies annotées ne peuvent pas représenter plus de 50 % des lignes.")
        train_x, test_x, train_y, test_y = train_test_split(
            features,
            labels,
            test_size=hp["test_size"],
            random_state=hp["random_state"],
            stratify=labels,
        )
        algorithms = ("isolation_forest", "random_forest") if hp["algorithm"] == "auto" else (hp["algorithm"],)
        evaluated = []
        for algorithm in algorithms:
            model = _build_model(algorithm, hp, contamination)
            model.fit(train_x, train_y) if algorithm == "random_forest" else model.fit(train_x)
            model_predicted = (model.predict(test_x) == -1).astype(int)
            evaluated.append(
                (
                    float(f1_score(test_y, model_predicted, zero_division=0)),
                    float(roc_auc_score(test_y, -model.decision_function(test_x))),
                    algorithm,
                    model,
                    model_predicted,
                )
            )
        _, _, selected_algorithm, candidate, predicted = max(evaluated, key=lambda item: item[:2])
        anomaly_scores = -candidate.decision_function(test_x)
        metrics = {
            "roc_auc": round(float(roc_auc_score(test_y, anomaly_scores)), 4),
            "f1": round(float(f1_score(test_y, predicted, zero_division=0)), 4),
            "precision": round(float(precision_score(test_y, predicted, zero_division=0)), 4),
            "recall": round(float(recall_score(test_y, predicted, zero_division=0)), 4),
            "train_rows": int(len(train_x)),
            "test_rows": int(len(test_x)),
        }
        threshold_info = _select_threshold(candidate, train_x, train_y, test_x, test_y)
        metrics.update(threshold_info)
        promoted = all(
            metrics[name] >= threshold for name, threshold in PROMOTION_THRESHOLDS.items()
        )
        version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
        snapshot_id = self._log_drift(_sample_rows(test_x, DRIFT_SAMPLE_ROWS), _sample_rows(train_x, REFERENCE_MAX_ROWS))

        artifact_path: Path | None = None
        temporary_path: Path | None = None
        registered = False
        with self._lock:
            try:
                if promoted:
                    artifact_path = self.artifacts_path / "{}.joblib".format(version)
                    temporary_path = artifact_path.with_suffix(".joblib.tmp")
                    joblib.dump(
                        {"model": candidate, "reference": _sample_rows(train_x, REFERENCE_MAX_ROWS)},
                        temporary_path,
                        compress=3,
                    )
                    os.replace(temporary_path, artifact_path)
                paths_to_remove: List[Path] = []
                with self._connect() as connection:
                    connection.execute(
                        """
                        INSERT INTO ml_models(
                            version, name, status, artifact_path, created_at, metrics_json, training_rows
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            version,
                            MODEL_NAME,
                            "promoted" if promoted else "rejected",
                            artifact_path.name if artifact_path else None,
                            _utc_now(),
                            json.dumps(metrics, sort_keys=True),
                            len(records),
                        ),
                    )
                    if promoted:
                        paths_to_remove = self._prune_models(connection)
                    connection.execute(
                        """
                        DELETE FROM ml_models WHERE version IN (
                            SELECT version FROM ml_models
                            ORDER BY created_at DESC, version DESC LIMIT -1 OFFSET ?
                        )
                        """,
                        (MODEL_HISTORY_RETENTION,),
                    )
                registered = True
                for path in paths_to_remove:
                    if path.is_file():
                        path.unlink()
            except Exception:
                if artifact_path and not registered and artifact_path.exists():
                    artifact_path.unlink()
                raise
            finally:
                if temporary_path and temporary_path.exists():
                    temporary_path.unlink()
        LOG.info(
            "ml_model_evaluated",
            extra={
                "model_version": version,
                "model_status": "promoted" if promoted else "rejected",
                "records_processed": len(records),
            },
        )
        mlflow_run_id = log_training_run(
            version=version,
            model=candidate,
            model_name=MODEL_NAME,
            params={
                **{"model__" + k: v for k, v in candidate.get_params().items()},
                "algorithm": type(candidate).__name__,
                "algorithm_requested": hp["algorithm"],
                "algorithm_candidates": ",".join(item[2] for item in evaluated),
                "contamination": round(contamination, 4),
                "test_size": hp["test_size"],
                "split_random_state": hp["random_state"],
                "features": ",".join(FEATURES),
                "feature_count": len(FEATURES),
                "stratified_split": True,
            },
            metrics=metrics,
            promoted=promoted,
            dataset_digest=digest_text(source),
            dataset_rows=len(records),
            thresholds=PROMOTION_THRESHOLDS,
            data_source=data_source,
            hyperparameters=hp,
            description=description,
            extra_tags=run_tags,
            class_counts={int(k): int(v) for k, v in class_counts.items()},
            train_sample=train_x if data_source == "synthetic-sample" else None,
        )
        return {
            "version": version,
            "status": "promoted" if promoted else "rejected",
            "metrics": metrics,
            "thresholds": PROMOTION_THRESHOLDS,
            "evidently_snapshot_id": snapshot_id,
            "mlflow_run_id": mlflow_run_id,
        }

    def predict(self, csv_content: str) -> Dict[str, Any]:
        records = _read_csv(csv_content, require_labels=False)
        if not records:
            raise ValueError("Le fichier CSV ne contient aucune ligne.")
        if len(records) > MAX_PREDICTION_ROWS:
            raise ValueError(
                "La prédiction est limitée à {} lignes par lot.".format(MAX_PREDICTION_ROWS)
            )
        features = _to_feature_data(records)
        drift_features = _sample_rows(features, DRIFT_SAMPLE_ROWS)
        with self._lock:
            model_row = self._latest_promoted_model()
            if model_row is None:
                raise ValueError("Aucun modèle promu. Entraînez d'abord le modèle.")
            if not model_row["artifact_path"]:
                raise RuntimeError("Le modèle promu ne référence aucun artefact.")
            if Path(model_row["artifact_path"]).name != model_row["artifact_path"]:
                raise RuntimeError("Le chemin de l'artefact du modèle est invalide.")
            artifact_path = self.artifacts_path / model_row["artifact_path"]
            if not artifact_path.is_file():
                raise RuntimeError(
                    "L'artefact du modèle promu est absent : {}.".format(artifact_path)
                )
            artifact = joblib.load(artifact_path)
            model: IsolationForest = artifact["model"]
            reference: pd.DataFrame = artifact["reference"]
            snapshot_id = self._log_drift(drift_features, reference)
            predictions = np.concatenate(
                [
                    (model.predict(features.iloc[start : start + PREDICTION_CHUNK_ROWS]) == -1).astype(int)
                    for start in range(0, len(features), PREDICTION_CHUNK_ROWS)
                ]
            )
            anomaly_count = int(predictions.sum())
            batch_id = uuid.uuid4().hex
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO ml_prediction_batches(
                        id, created_at, model_version, records_processed,
                        anomalies_detected, evidently_snapshot_id
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        batch_id,
                        _utc_now(),
                        model_row["version"],
                        len(records),
                        anomaly_count,
                        snapshot_id,
                    ),
                )
                connection.execute(
                    """
                    DELETE FROM ml_prediction_batches WHERE id IN (
                        SELECT id FROM ml_prediction_batches
                        ORDER BY created_at DESC LIMIT -1 OFFSET 500
                    )
                    """
                )
            self._prediction_counts["anomaly"] += anomaly_count
            self._prediction_counts["normal"] += len(records) - anomaly_count
        LOG.info(
            "ml_prediction_batch_completed",
            extra={
                "model_version": model_row["version"],
                "run_id": batch_id,
                "records_processed": len(records),
            },
        )
        return {
            "batch_id": batch_id,
            "model_version": model_row["version"],
            "records_processed": len(records),
            "anomalies_detected": anomaly_count,
            "anomaly_rate_pct": round(100 * anomaly_count / len(records), 2),
            "evidently_snapshot_id": snapshot_id,
            "predictions": [
                {
                    "robot_id": str(record.get("robot_id", "")),
                    "timestamp": str(record.get("timestamp", "")),
                    "is_anomaly": int(prediction),
                }
                for record, prediction in zip(records[:100], predictions[:100])
            ],
            "predictions_truncated": len(records) > 100,
        }

    def _log_drift(self, current: pd.DataFrame, reference: pd.DataFrame) -> str:
        with self._lock:
            definition = DataDefinition(numerical_columns=list(FEATURES))
            current_dataset = Dataset.from_pandas(current, data_definition=definition)
            reference_dataset = Dataset.from_pandas(reference, data_definition=definition)
            snapshot = Report([DataDriftPreset()]).run(
                current_data=current_dataset,
                reference_data=reference_dataset,
            )
            run = self._workspace.add_run(
                self._evidently_project.id,
                snapshot,
                include_data=False,
            )
            run_ids = sorted(
                self._workspace.list_runs(self._evidently_project.id),
                key=str,
            )
            for old_run_id in run_ids[:-EVIDENTLY_RUN_RETENTION]:
                self._workspace.delete_run(self._evidently_project.id, old_run_id)
            self._drift_reports += 1
            return str(run.id)

    def _latest_promoted_model(self) -> sqlite3.Row | None:
        with self._connect() as connection:
            return connection.execute(
                """
                SELECT version, artifact_path, metrics_json, created_at
                FROM ml_models WHERE status = 'promoted'
                ORDER BY created_at DESC, version DESC LIMIT 1
                """
            ).fetchone()

    def _prune_models(self, connection: sqlite3.Connection) -> List[Path]:
        old_models = connection.execute(
            """
            SELECT version, artifact_path FROM ml_models
            WHERE status = 'promoted'
            ORDER BY created_at DESC, version DESC LIMIT -1 OFFSET ?
            """,
            (MODEL_RETENTION,),
        ).fetchall()
        paths_to_remove = [
            self.artifacts_path / old_model["artifact_path"]
            for old_model in old_models
            if old_model["artifact_path"]
        ]
        for old_model in old_models:
            connection.execute(
                "UPDATE ml_models SET status = 'archived', artifact_path = NULL WHERE version = ?",
                (old_model["version"],),
            )
        return paths_to_remove

    def status(self) -> Dict[str, Any]:
        with self._connect() as connection:
            models = connection.execute(
                """
                SELECT version, status, created_at, metrics_json, training_rows
                FROM ml_models ORDER BY created_at DESC, version DESC LIMIT 10
                """
            ).fetchall()
            batches = connection.execute(
                """
                SELECT id, created_at, model_version, records_processed,
                    anomalies_detected, evidently_snapshot_id
                FROM ml_prediction_batches ORDER BY created_at DESC, id DESC LIMIT 10
                """
            ).fetchall()
        active_model = next((model for model in models if model["status"] == "promoted"), None)
        return {
            "model_name": MODEL_NAME,
            "active_model": _model_to_dict(active_model) if active_model else None,
            "models": [_model_to_dict(model) for model in models],
            "prediction_batches": [dict(batch) for batch in batches],
            "evidently_project": EVIDENTLY_PROJECT_NAME,
            "evidently_workspace": str(self.evidently_path),
            "promotion_thresholds": PROMOTION_THRESHOLDS,
        }

    def prometheus_metrics(self) -> str:
        model_row = self._latest_promoted_model()
        lines = [
            "# HELP robotics_ml_model_promoted Whether a model is currently promoted.",
            "# TYPE robotics_ml_model_promoted gauge",
            "robotics_ml_model_promoted {}".format(int(model_row is not None)),
            "# HELP robotics_ml_drift_reports_total Evidently drift reports stored locally.",
            "# TYPE robotics_ml_drift_reports_total counter",
            "robotics_ml_drift_reports_total {}".format(self._drift_reports),
            "# HELP robotics_ml_predictions_total Predictions returned by this process.",
            "# TYPE robotics_ml_predictions_total counter",
        ]
        for outcome, count in sorted(self._prediction_counts.items()):
            lines.append(
                'robotics_ml_predictions_total{{outcome="{}"}} {}'.format(outcome, count)
            )
        lines.extend(
            [
                "# HELP robotics_ml_model_info Metadata for the currently promoted model.",
                "# TYPE robotics_ml_model_info gauge",
            ]
        )
        if model_row is not None:
            metrics = json.loads(model_row["metrics_json"])
            lines.append(
                'robotics_ml_model_info{{version="{}"}} 1'.format(model_row["version"])
            )
            for metric_name in ("roc_auc", "f1", "precision", "recall"):
                lines.extend(
                    [
                        "# HELP robotics_ml_{} Latest promoted model evaluation metric.".format(
                            metric_name
                        ),
                        "# TYPE robotics_ml_{} gauge".format(metric_name),
                        "robotics_ml_{} {}".format(metric_name, metrics[metric_name]),
                    ]
                )
        return "\n".join(lines) + "\n"


def _read_csv(content: str, require_labels: bool) -> List[Dict[str, str]]:
    import csv
    import io

    reader = csv.DictReader(io.StringIO(content))
    if not reader.fieldnames:
        raise ValueError("Le CSV doit contenir une ligne d'en-tête.")
    missing = [column for column in FEATURES if column not in reader.fieldnames]
    if missing:
        raise ValueError("Colonnes numériques requises : {}.".format(", ".join(missing)))
    if require_labels and "is_anomaly" not in reader.fieldnames:
        raise ValueError("Pour entraîner le modèle, le CSV doit contenir la colonne is_anomaly (0/1).")
    records: List[Dict[str, str]] = []
    for row_number, row in enumerate(reader, start=2):
        if len(records) >= 200_000:
            raise ValueError("Limite de 200000 lignes dépassée.")
        if any(value is None or not value.strip() for value in (row.get(name) for name in FEATURES)):
            raise ValueError("Valeur manquante parmi les colonnes numériques à la ligne {}.".format(row_number))
        records.append(row)
    return records


def _sample_rows(frame: pd.DataFrame, limit: int) -> pd.DataFrame:
    if len(frame) <= limit:
        return frame
    return frame.sample(n=limit, random_state=42).reset_index(drop=True)


def _to_feature_data(records: List[Dict[str, str]]) -> pd.DataFrame:
    values = []
    for row_number, record in enumerate(records, start=2):
        row = []
        for feature in FEATURES:
            try:
                value = float(record[feature])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "Valeur numérique invalide pour {} à la ligne {}.".format(feature, row_number)
                ) from error
            if not math.isfinite(value):
                raise ValueError(
                    "Valeur numérique non finie pour {} à la ligne {}.".format(feature, row_number)
                )
            row.append(value)
        values.append(row)
    return pd.DataFrame(values, columns=FEATURES)


def _to_training_data(records: List[Dict[str, str]]) -> Tuple[pd.DataFrame, pd.Series]:
    features = _to_feature_data(records)
    labels = []
    for row_number, record in enumerate(records, start=2):
        try:
            label_value = float(record["is_anomaly"])
        except (TypeError, ValueError) as error:
            raise ValueError("is_anomaly doit valoir 0 ou 1 (ligne {}).".format(row_number)) from error
        if label_value not in (0.0, 1.0):
            raise ValueError("is_anomaly doit valoir 0 ou 1 (ligne {}).".format(row_number))
        labels.append(int(label_value))
    return features, pd.Series(labels, dtype="int64")


def _model_to_dict(model: sqlite3.Row) -> Dict[str, Any]:
    value = dict(model)
    value["metrics"] = json.loads(value.pop("metrics_json"))
    return value
