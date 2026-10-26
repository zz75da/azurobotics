"""Lightweight Airflow-style pipelines (small DAGs of tasks) built on the existing services.

Nothing here schedules or distributes work: a PipelineRunner executes one named pipeline at a time
in a background thread, task by task in dependency order, and records per-task status.
Run history is kept in memory (bounded) and persisted in SQLite; container storage is ephemeral.
"""
from __future__ import annotations

import copy
import json
import logging
import os
import re
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from experiment_tracking import EXPERIMENT_NAME, tracking_enabled
from mlops import (
    ALGORITHMS,
    DEFAULT_HYPERPARAMETERS,
    MIN_TRAINING_ROWS,
    PROMOTION_THRESHOLDS,
    _read_csv,
    _to_feature_data,
    _to_training_data,
    _validate_hyperparameters,
)
from robot_data import analyze_text, make_sample_telemetry_csv

LOG = logging.getLogger("robotics-demo.pipelines")

MAX_CONTENT_BYTES = 25 * 1024 * 1024
MAX_DESCRIPTION_CHARS = 500
MEMORY_HISTORY = 50
DB_HISTORY = 200
RUN_TIMEOUT_SECONDS = 1800
STALE_GRACE_SECONDS = 300
ANOMALY_RATE_WARNING_PCT = 20.0
SWEEP_PATH = Path(__file__).resolve().parent / "tools" / "results" / "ai4i_random_forest_sweep.json"
RUN_ID_PATTERN = re.compile(r"^[0-9a-f]{12}$")
DURATION_BOUNDS = (1.0, 5.0, 15.0, 60.0, 300.0, float("inf"))
FINAL_STATUSES = ("success", "failed")
TUNED_PRESET = "ai4i_random_forest_tuned"

# Mirrors the best row of tools/results/ai4i_random_forest_sweep.json (checked by the tests).
PRESETS: Dict[str, Dict[str, Any]] = {
    "default": {
        "label": "Isolation Forest par défaut",
        "description": "Hyperparamètres par défaut du service MLOps.",
        "hyperparameters": {},
    },
    TUNED_PRESET: {
        "label": "Random Forest ajusté (AI4I)",
        "description": "Meilleure configuration du balayage AI4I ; à utiliser avec un CSV étiqueté.",
        "hyperparameters": {
            "algorithm": "random_forest",
            "n_estimators": 300,
            "max_depth": 8,
            "min_samples_leaf": 5,
            "class_weight": None,
            "max_features": 0.5,
            "decision_threshold": 0.32,
        },
    },
}


class PipelineBusy(Exception):
    """Another run is already active."""


class TaskFailure(Exception):
    """A task failed with a message that is safe to show to users."""


@dataclass
class Context:
    params: Dict[str, Any]
    csv: Optional[str]
    mlops: Any
    run_id: str
    tuning_results: Optional[List[Dict[str, Any]]] = None
    state: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Task:
    id: str
    label: str
    fn: Callable[[Context], Dict[str, Any]]
    depends_on: Tuple[str, ...] = ()


@dataclass(frozen=True)
class Pipeline:
    name: str
    title: str
    description: str
    tasks: Tuple[Task, ...]
    kind: str  # "train" | "score" | "logs"
    suffixes: Tuple[str, ...] = (".csv",)
    summary: Tuple[Tuple[str, str], ...] = ()


def topological_order(tasks: Tuple[Task, ...] | List[Task]) -> List[Task]:
    by_id = {task.id: task for task in tasks}
    if len(by_id) != len(tasks):
        raise ValueError("Identifiants de tâches dupliqués.")
    for task in tasks:
        for dependency in task.depends_on:
            if dependency not in by_id:
                raise ValueError("Dépendance inconnue : {}.".format(dependency))
    ordered: List[Task] = []
    done: set = set()
    remaining = list(tasks)
    while remaining:
        ready = [task for task in remaining if all(dep in done for dep in task.depends_on)]
        if not ready:
            raise ValueError("Le graphe de tâches contient un cycle.")
        for task in ready:
            ordered.append(task)
            done.add(task.id)
            remaining.remove(task)
    return ordered


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_message(error: BaseException) -> str:
    text = re.sub(r"[\x00-\x1f\x7f]+", " ", str(error)).strip()
    return text[:300] or "Échec de la tâche."


def _clean_outputs(outputs: Dict[str, Any]) -> Dict[str, Any]:
    cleaned: Dict[str, Any] = {}
    for key, value in list(outputs.items())[:30]:
        if isinstance(value, bool) or value is None or isinstance(value, int):
            cleaned[str(key)] = value
        elif isinstance(value, float):
            cleaned[str(key)] = round(value, 4) if value == value and abs(value) != float("inf") else None
        else:
            cleaned[str(key)] = str(value)[:300]
    return cleaned


def load_tuning_results() -> Optional[List[Dict[str, Any]]]:
    try:
        data = json.loads(SWEEP_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, list) or len(data) > 500 or len(json.dumps(data)) > 200_000:
        return None
    return [item for item in data if isinstance(item, dict)]


# ---- tasks -------------------------------------------------------------------------------------

def _validate_training_data(ctx: Context) -> Dict[str, Any]:
    records = _read_csv(ctx.csv if ctx.csv is not None else make_sample_telemetry_csv(), require_labels=True)
    if len(records) < MIN_TRAINING_ROWS:
        raise TaskFailure("Au moins {} lignes valides sont nécessaires.".format(MIN_TRAINING_ROWS))
    _, labels = _to_training_data(records)
    counts = labels.value_counts()
    if len(counts) != 2 or int(counts.min()) < 2:
        raise TaskFailure("is_anomaly doit contenir au moins deux lignes de chaque classe (0 et 1).")
    return {
        "rows": len(records),
        "anomalies": int(counts.get(1, 0)),
        "anomaly_rate_pct": 100 * float(labels.mean()),
        "source": "synthetic-sample" if ctx.csv is None else "uploaded-csv",
    }


def _train(ctx: Context) -> Dict[str, Any]:
    options: Dict[str, Any] = {
        "hyperparameters": ctx.params["hyperparameters"],
        "description": ctx.params.get("description") or None,
        "run_tags": {"pipeline": ctx.params["pipeline"], "pipeline_run": ctx.run_id, "trigger": ctx.params["trigger"]},
    }
    if ctx.tuning_results:
        options["tuning_results"] = ctx.tuning_results
    result = ctx.mlops.train(ctx.csv, **options)
    ctx.state["train"] = result
    return {
        "model_version": result.get("version"),
        "algorithm": ctx.params["hyperparameters"]["algorithm"],
        "mlflow_run_id": result.get("mlflow_run_id"),
        "tuning_results_embedded": bool(ctx.tuning_results),
    }


def _evaluate(ctx: Context) -> Dict[str, Any]:
    result = ctx.state["train"]
    metrics = result.get("metrics", {})
    outputs: Dict[str, Any] = {
        name: metrics.get(name) for name in ("f1", "roc_auc", "precision", "recall", "test_rows") if name in metrics
    }
    thresholds = result.get("thresholds", PROMOTION_THRESHOLDS)
    outputs["thresholds_met"] = all(
        isinstance(metrics.get(name), (int, float)) and metrics[name] >= limit for name, limit in thresholds.items()
    )
    return outputs


def _promote(ctx: Context) -> Dict[str, Any]:
    result = ctx.state["train"]
    outputs: Dict[str, Any] = {
        "promotion": result.get("status"),
        "model_version": result.get("version"),
        "mlflow_run_id": result.get("mlflow_run_id"),
    }
    if result.get("mlflow_run_id"):
        outputs["mlflow_experiment"] = EXPERIMENT_NAME
    return outputs


def _validate_scoring_data(ctx: Context) -> Dict[str, Any]:
    records = _read_csv(ctx.csv if ctx.csv is not None else make_sample_telemetry_csv(), require_labels=False)
    if not records:
        raise TaskFailure("Le fichier CSV ne contient aucune ligne.")
    _to_feature_data(records)
    return {"rows": len(records), "source": "synthetic-sample" if ctx.csv is None else "uploaded-csv"}


def _predict(ctx: Context) -> Dict[str, Any]:
    result = ctx.mlops.predict(ctx.csv if ctx.csv is not None else make_sample_telemetry_csv())
    ctx.state["predict"] = result
    return {
        "model_version": result.get("model_version"),
        "records": result.get("records_processed"),
        "anomalies": result.get("anomalies_detected"),
        "anomaly_rate_pct": result.get("anomaly_rate_pct"),
    }


def _drift_check(ctx: Context) -> Dict[str, Any]:
    result = ctx.state["predict"]
    rate = float(result.get("anomaly_rate_pct") or 0)
    return {
        "evidently_snapshot_id": result.get("evidently_snapshot_id"),
        "anomaly_rate_pct": rate,
        "anomaly_rate_warning": rate > ANOMALY_RATE_WARNING_PCT,
    }


def _parse_logs(ctx: Context) -> Dict[str, Any]:
    suffix = ctx.params["suffix"]
    content = ctx.csv if ctx.csv is not None else make_sample_telemetry_csv()
    analysis = analyze_text(content, suffix)
    ctx.state["analysis"] = analysis
    return {
        "events_total": analysis.get("events_total"),
        "robots_total": analysis.get("robots_total"),
        "errors_total": analysis.get("errors_total"),
    }


def _incidents_kpis(ctx: Context) -> Dict[str, Any]:
    analysis = ctx.state["analysis"]
    kpis = analysis.get("kpis") or {}
    return {
        "incident_groups": len(analysis.get("incident_groups") or []),
        "error_rate_pct": analysis.get("error_rate_pct"),
        "anomaly_rate_pct": kpis.get("anomaly_rate_pct"),
        "mean_time_between_incidents_min": kpis.get("mean_time_between_incidents_min"),
    }


def default_pipelines() -> Dict[str, Pipeline]:
    definitions = (
        Pipeline(
            name="train_anomaly_model",
            title="Entraîner le modèle d’anomalies",
            description="Valide les données, entraîne (suivi MLflow dans l’entraînement), évalue puis promeut si les seuils sont atteints.",
            kind="train",
            tasks=(
                Task("validate_data", "Valider les données", _validate_training_data),
                Task("train", "Entraîner", _train, ("validate_data",)),
                Task("evaluate", "Évaluer", _evaluate, ("train",)),
                Task("promote", "Promouvoir / enregistrer", _promote, ("evaluate",)),
            ),
            summary=(("evaluate", "f1"), ("evaluate", "roc_auc"), ("promote", "promotion"),
                     ("promote", "model_version"), ("promote", "mlflow_run_id")),
        ),
        Pipeline(
            name="score_and_drift",
            title="Prédire et contrôler la dérive",
            description="Valide le CSV, prédit avec le modèle promu puis rapporte la dérive Evidently.",
            kind="score",
            tasks=(
                Task("validate_data", "Valider les données", _validate_scoring_data),
                Task("predict", "Prédire", _predict, ("validate_data",)),
                Task("drift_check", "Contrôler la dérive", _drift_check, ("predict",)),
            ),
            summary=(("predict", "records"), ("predict", "anomaly_rate_pct"),
                     ("drift_check", "evidently_snapshot_id")),
        ),
        Pipeline(
            name="analyze_logs",
            title="Analyser les journaux de télémétrie",
            description="Analyse un CSV/JSONL puis extrait incidents récurrents et KPI.",
            kind="logs",
            suffixes=(".csv", ".jsonl", ".ndjson"),
            tasks=(
                Task("parse", "Analyser le fichier", _parse_logs),
                Task("incidents_kpis", "Incidents et KPI", _incidents_kpis, ("parse",)),
            ),
            summary=(("parse", "events_total"), ("parse", "errors_total"),
                     ("incidents_kpis", "incident_groups")),
        ),
    )
    for pipeline in definitions:
        topological_order(pipeline.tasks)
    return {pipeline.name: pipeline for pipeline in definitions}


# ---- request validation -----------------------------------------------------------------------

def _clean_text(value: Any, name: str, limit: int) -> str:
    if not isinstance(value, str):
        raise ValueError("{} doit être du texte.".format(name))
    value = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", "", value).strip()
    if len(value) > limit:
        raise ValueError("{} est limité à {} caractères.".format(name, limit))
    return value


def validate_run_request(pipeline: Pipeline, body: Any) -> Tuple[Dict[str, Any], Optional[str], Optional[str]]:
    """Return (normalised params, filename, csv text). Raises ValueError on anything unexpected."""
    if not isinstance(body, dict):
        raise ValueError("Le corps doit être un objet JSON.")
    unknown = set(body) - {"params", "filename", "content"}
    if unknown:
        raise ValueError("Champs inconnus : {}.".format(", ".join(sorted(map(str, unknown)))))
    raw = body.get("params")
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("params doit être un objet JSON.")
    allowed = {"dataset"}
    if pipeline.kind == "train":
        allowed |= {"preset", "algorithm", "hyperparameters", "description"}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError("Paramètres inconnus : {}.".format(", ".join(sorted(map(str, unknown)))))

    content = body.get("content")
    filename = body.get("filename")
    if content is not None:
        if not isinstance(content, str) or not isinstance(filename, str):
            raise ValueError("filename et content doivent être du texte.")
        filename = Path(filename.replace("\\", "/")).name
        if Path(filename).suffix.lower() not in pipeline.suffixes:
            raise ValueError("Formats acceptés : {}.".format(", ".join(pipeline.suffixes)))
        if len(content.encode("utf-8")) > MAX_CONTENT_BYTES:
            raise ValueError("Fichier de données trop volumineux (limite : 25 Mo).")
        filename = re.sub(r"[^A-Za-z0-9_.-]", "_", filename)[:100]
    elif filename is not None:
        raise ValueError("filename nécessite content.")

    dataset = raw.get("dataset", "uploaded" if content is not None else "synthetic")
    if dataset not in ("synthetic", "uploaded"):
        raise ValueError("dataset doit valoir 'synthetic' ou 'uploaded'.")
    if dataset == "uploaded" and content is None:
        raise ValueError("Le jeu de données importé nécessite un fichier CSV.")
    if dataset == "synthetic" and content is not None:
        raise ValueError("Un fichier a été envoyé alors que dataset vaut 'synthetic'.")
    params: Dict[str, Any] = {"dataset": dataset}

    if pipeline.kind == "train":
        preset = raw.get("preset", "default")
        if preset not in PRESETS:
            raise ValueError("Préréglage inconnu.")
        overrides = dict(PRESETS[preset]["hyperparameters"])
        supplied = raw.get("hyperparameters")
        if supplied is not None:
            if not isinstance(supplied, dict):
                raise ValueError("hyperparameters doit être un objet JSON.")
            overrides.update(supplied)
        if "algorithm" in raw:
            overrides["algorithm"] = raw["algorithm"]
        params["preset"] = preset
        params["hyperparameters"] = _validate_hyperparameters(overrides)
        params["description"] = _clean_text(raw.get("description") or "", "description", MAX_DESCRIPTION_CHARS)
    return params, filename, content


# ---- runner -----------------------------------------------------------------------------------

class PipelineRunner:
    def __init__(
        self,
        database_path: Path,
        mlops: Any,
        pipelines: Optional[Dict[str, Pipeline]] = None,
        timeout_seconds: float = RUN_TIMEOUT_SECONDS,
    ) -> None:
        self.database_path = database_path
        self.mlops = mlops
        self.pipelines = pipelines or default_pipelines()
        self.timeout_seconds = timeout_seconds
        self._lock = threading.RLock()
        self._active: Optional[Dict[str, Any]] = None
        self._mono: Dict[str, float] = {}
        self._history: List[Dict[str, Any]] = []
        self._counts: Dict[Tuple[str, str], int] = {}
        self._durations: Dict[str, List[float]] = {}
        self._sums: Dict[str, float] = {}
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.database_path), timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 10000")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS pipeline_runs (
                    id TEXT PRIMARY KEY,
                    pipeline TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    record_json TEXT NOT NULL
                )
                """
            )
            rows = connection.execute(
                "SELECT record_json FROM pipeline_runs WHERE status IN ('pending', 'running')"
            ).fetchall()
            for row in rows:
                record = json.loads(row["record_json"])
                self._close_interrupted(record)
                self._store(connection, record)
            connection.execute(
                "DELETE FROM pipeline_runs WHERE id IN "
                "(SELECT id FROM pipeline_runs ORDER BY created_at DESC LIMIT -1 OFFSET ?)",
                (DB_HISTORY,),
            )
            recent = connection.execute(
                "SELECT record_json FROM pipeline_runs ORDER BY created_at DESC LIMIT ?", (MEMORY_HISTORY,)
            ).fetchall()
        self._history = [json.loads(row["record_json"]) for row in recent]

    @staticmethod
    def _close_interrupted(record: Dict[str, Any]) -> None:
        for task in record["tasks"]:
            if task["status"] in ("pending", "running"):
                task["status"] = "skipped"
                task["error"] = "interrupted"
        record["status"] = "failed"
        record["error"] = "Arrêt inattendu du processus pendant l'exécution."
        record["finished_at"] = record.get("finished_at") or _utc_now()

    @staticmethod
    def _store(connection: sqlite3.Connection, record: Dict[str, Any]) -> None:
        connection.execute(
            "INSERT OR REPLACE INTO pipeline_runs(id, pipeline, status, created_at, record_json) VALUES (?, ?, ?, ?, ?)",
            (record["id"], record["pipeline"], record["status"], record["created_at"], json.dumps(record)),
        )

    def _persist(self, record: Dict[str, Any]) -> None:
        try:
            with self._connect() as connection:
                self._store(connection, record)
        except Exception:
            LOG.exception("pipeline_persist_failed", extra={"run_id": record["id"]})

    # -- public API --

    def definitions(self, mlflow_url: Optional[str] = None) -> Dict[str, Any]:
        with self._lock:
            latest = {}
            for record in self._history:
                latest.setdefault(record["pipeline"], self._summary(record))
            active = self._summary(self._active) if self._active else None
        return {
            "pipelines": [
                {
                    "name": p.name,
                    "title": p.title,
                    "description": p.description,
                    "kind": p.kind,
                    "accepted_suffixes": list(p.suffixes),
                    "tasks": [{"id": t.id, "label": t.label, "depends_on": list(t.depends_on)} for t in p.tasks],
                    "latest_run": latest.get(p.name),
                }
                for p in self.pipelines.values()
            ],
            "presets": [
                {"id": key, "label": preset["label"], "description": preset["description"],
                 "hyperparameters": preset["hyperparameters"]}
                for key, preset in PRESETS.items()
            ],
            "algorithms": list(ALGORITHMS),
            "default_hyperparameters": DEFAULT_HYPERPARAMETERS,
            "max_file_mb": MAX_CONTENT_BYTES // (1024 * 1024),
            "active_run": active,
            "mlflow": {
                "enabled": tracking_enabled(),
                "experiment": EXPERIMENT_NAME,
                "url": mlflow_url,
            },
            "history_note": "Historique conservé dans SQLite sur le stockage du conteneur, qui est éphémère.",
        }

    def start(self, name: str, body: Any, trigger: str = "manual") -> Dict[str, Any]:
        pipeline = self.pipelines.get(name) if isinstance(name, str) else None
        if pipeline is None:
            raise KeyError(name)
        params, filename, content = validate_run_request(pipeline, body)
        params.update({"pipeline": pipeline.name, "trigger": trigger})
        if pipeline.kind == "logs":
            params["suffix"] = Path(filename).suffix.lower() if filename else ".csv"
        tuning = None
        if pipeline.kind == "train" and params["preset"] == TUNED_PRESET and content is not None:
            tuning = load_tuning_results()
        run_id = uuid.uuid4().hex[:12]
        record = {
            "id": run_id,
            "pipeline": pipeline.name,
            "status": "pending",
            "trigger": trigger,
            "created_at": _utc_now(),
            "started_at": None,
            "finished_at": None,
            "duration_seconds": None,
            "error": None,
            "params": {
                **{k: v for k, v in params.items() if k not in ("pipeline", "trigger", "suffix")},
                "filename": filename,
                "content_bytes": len(content.encode("utf-8")) if content is not None else 0,
            },
            "summary": {},
            "tasks": [
                {"id": t.id, "label": t.label, "depends_on": list(t.depends_on), "status": "pending",
                 "started_at": None, "finished_at": None, "duration_seconds": None, "error": None, "outputs": {}}
                for t in topological_order(pipeline.tasks)
            ],
        }
        with self._lock:
            self._release_if_stale()
            if self._active is not None:
                raise PipelineBusy("Un pipeline est déjà en cours d'exécution.")
            self._active = record
            self._history.insert(0, record)
            del self._history[MEMORY_HISTORY:]
        self._persist(record)
        context = Context(params=params, csv=content, mlops=self.mlops, run_id=run_id, tuning_results=tuning)
        threading.Thread(
            target=self._execute, args=(pipeline, record, context), name="pipeline-" + run_id, daemon=True
        ).start()
        return self.get_run(run_id) or {}

    def list_runs(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [self._summary(record) for record in self._history]

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        if not isinstance(run_id, str) or not RUN_ID_PATTERN.match(run_id):
            return None
        with self._lock:
            for record in self._history:
                if record["id"] == run_id:
                    return copy.deepcopy(record)
        with self._connect() as connection:
            row = connection.execute("SELECT record_json FROM pipeline_runs WHERE id = ?", (run_id,)).fetchone()
        return json.loads(row["record_json"]) if row else None

    def is_busy(self) -> bool:
        with self._lock:
            self._release_if_stale()
            return self._active is not None

    def wait_idle(self, timeout: float = 60.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not self.is_busy():
                return True
            time.sleep(0.05)
        return False

    # -- execution --

    def _release_if_stale(self) -> None:
        active = self._active
        started = self._mono.get(active["id"]) if active else None
        if started is None:
            return
        if time.monotonic() - started >  self.timeout_seconds + STALE_GRACE_SECONDS:
            self._finish(active, "failed", "Délai maximal dépassé ; l'exécution a été abandonnée.")

    def _finish(self, record: Dict[str, Any], status: str, error: Optional[str]) -> None:
        if record["status"] in FINAL_STATUSES:
            return
        for task in record["tasks"]:
            if task["status"] in ("pending", "running"):
                task["status"] = "skipped"
                task["error"] = "timeout" if error and "Délai" in error else "run_aborted"
        record["status"] = status
        record["error"] = error
        record["finished_at"] = _utc_now()
        started = self._mono.pop(record["id"], None)
        record["duration_seconds"] = round(time.monotonic() - started, 3) if started is not None else 0.0
        key = (record["pipeline"], status)
        self._counts[key] = self._counts.get(key, 0) + 1
        self._durations.setdefault(record["pipeline"], []).append(record["duration_seconds"])
        self._sums[record["pipeline"]] = self._sums.get(record["pipeline"], 0.0) + record["duration_seconds"]
        if self._active is record:
            self._active = None
        self._persist(record)

    def _execute(self, pipeline: Pipeline, record: Dict[str, Any], context: Context) -> None:
        by_id = {task.id: task for task in pipeline.tasks}
        with self._lock:
            record["status"] = "running"
            record["started_at"] = _utc_now()
            self._mono[record["id"]] = time.monotonic()
            deadline = self._mono[record["id"]] + self.timeout_seconds
        failed = False
        timed_out = False
        try:
            for entry in record["tasks"]:
                task = by_id[entry["id"]]
                with self._lock:
                    if record["status"] in FINAL_STATUSES:
                        return
                    upstream_failed = any(
                        next(t for t in record["tasks"] if t["id"] == dep)["status"] != "success"
                        for dep in task.depends_on
                    )
                    if upstream_failed:
                        entry["status"] = "skipped"
                        entry["error"] = "upstream_failed"
                        failed = True
                        continue
                    if time.monotonic() > deadline:
                        timed_out = True
                        break
                    entry["status"] = "running"
                    entry["started_at"] = _utc_now()
                started = time.monotonic()
                outputs: Dict[str, Any] = {}
                error: Optional[str] = None
                try:
                    outputs = _clean_outputs(task.fn(context) or {})
                except (TaskFailure, ValueError) as exc:
                    error = _safe_message(exc)
                except Exception:
                    LOG.exception("pipeline_task_failed", extra={"run_id": record["id"], "task": task.id})
                    error = "Échec inattendu de la tâche. Consultez les journaux locaux."
                with self._lock:
                    if record["status"] in FINAL_STATUSES:
                        return
                    entry["duration_seconds"] = round(time.monotonic() - started, 3)
                    entry["finished_at"] = _utc_now()
                    entry["status"] = "failed" if error else "success"
                    entry["error"] = error
                    entry["outputs"] = outputs
                    failed = failed or bool(error)
                self._persist(record)
            with self._lock:
                summary: Dict[str, Any] = {}
                for task_id, key in pipeline.summary:
                    value = next(t for t in record["tasks"] if t["id"] == task_id)["outputs"].get(key)
                    if value is not None:
                        summary[key] = value
                record["summary"] = summary
                if timed_out:
                    self._finish(record, "failed", "Délai maximal dépassé.")
                else:
                    self._finish(record, "failed" if failed else "success",
                                 "Une tâche a échoué." if failed else None)
        except Exception:
            LOG.exception("pipeline_run_crashed", extra={"run_id": record["id"]})
            with self._lock:
                self._finish(record, "failed", "Erreur interne du pipeline.")

    # -- helpers --

    @staticmethod
    def _summary(record: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if record is None:
            return {}
        return {
            "id": record["id"],
            "pipeline": record["pipeline"],
            "status": record["status"],
            "trigger": record["trigger"],
            "created_at": record["created_at"],
            "started_at": record["started_at"],
            "finished_at": record["finished_at"],
            "duration_seconds": record["duration_seconds"],
            "summary": dict(record["summary"]),
            "tasks": [{"id": t["id"], "status": t["status"]} for t in record["tasks"]],
        }

    def prometheus_metrics(self) -> str:
        with self._lock:
            counts = dict(self._counts)
            durations = {k: list(v) for k, v in self._durations.items()}
            sums = dict(self._sums)
            active = int(self._active is not None)
        lines = [
            "# HELP robotics_pipeline_runs_total Pipeline runs finished by this process.",
            "# TYPE robotics_pipeline_runs_total counter",
        ]
        for (name, status), count in sorted(counts.items()):
            lines.append('robotics_pipeline_runs_total{{pipeline="{}",status="{}"}} {}'.format(name, status, count))
        lines.extend([
            "# HELP robotics_pipeline_run_duration_seconds Pipeline run duration in seconds.",
            "# TYPE robotics_pipeline_run_duration_seconds histogram",
        ])
        for name, values in sorted(durations.items()):
            for bound in DURATION_BOUNDS:
                label = "+Inf" if bound == float("inf") else str(bound)
                lines.append(
                    'robotics_pipeline_run_duration_seconds_bucket{{pipeline="{}",le="{}"}} {}'.format(
                        name, label, sum(1 for value in values if value <= bound)
                    )
                )
            lines.append('robotics_pipeline_run_duration_seconds_sum{{pipeline="{}"}} {:.6f}'.format(name, sums[name]))
            lines.append('robotics_pipeline_run_duration_seconds_count{{pipeline="{}"}} {}'.format(name, len(values)))
        lines.extend([
            "# HELP robotics_pipeline_active Whether a pipeline run is currently active.",
            "# TYPE robotics_pipeline_active gauge",
            "robotics_pipeline_active {}".format(active),
        ])
        return "\n".join(lines) + "\n"


def public_mlflow_url() -> Optional[str]:
    value = os.environ.get("MLFLOW_PUBLIC_URL", "").strip()
    return value if re.match(r"^https?://[^\s\"'<>]+$", value) else None
