"""Generate the example upload files in samples/, one set per pipeline of the application.

Everything is synthetic and seeded, so running it again gives the same files.

    python tools/make_samples.py
"""
from __future__ import annotations

import csv
import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

SAMPLES = Path(__file__).resolve().parent.parent / "samples"
START = datetime(2026, 3, 2, 6, 0, 0)
ROBOTS = ["AMR-{:02d}".format(number) for number in range(1, 9)]
EVENTS = ("move", "pick", "drop", "charge", "stop")
EVENT_WEIGHTS = (48, 23, 20, 6, 3)
COLUMNS = ("timestamp", "robot_id", "event", "duration_ms", "battery_pct", "x_m", "y_m")
INCIDENTS = (
    ("E101", "error", "Obstacle détecté, trajectoire bloquée"),
    ("E204", "error", "Défaut de préhension"),
    ("W310", "warning", "Batterie faible"),
    ("F500", "critical", "Arrêt d'urgence déclenché"),
)


def _stamp(index: int, step_seconds: int = 15) -> str:
    return (START + timedelta(seconds=step_seconds * index)).isoformat() + "Z"


def _normal_row(rng: random.Random, index: int) -> Dict[str, Any]:
    return {
        "timestamp": _stamp(index),
        "robot_id": rng.choice(ROBOTS),
        "event": rng.choices(EVENTS, weights=EVENT_WEIGHTS, k=1)[0],
        "duration_ms": rng.randint(200, 12_000),
        "battery_pct": round(rng.uniform(18, 100), 1),
        "x_m": round(rng.uniform(0, 120), 2),
        "y_m": round(rng.uniform(0, 80), 2),
    }


def _anomalous_row(rng: random.Random, index: int) -> Dict[str, Any]:
    row = _normal_row(rng, index)
    row.update(
        event="error",
        duration_ms=rng.randint(60_000, 200_000),
        battery_pct=round(rng.uniform(-15, 5), 1),
        x_m=round(rng.uniform(-60, -10), 2),
        y_m=round(rng.uniform(110, 170), 2),
    )
    return row


def _drifted_row(rng: random.Random, index: int) -> Dict[str, Any]:
    """Worn fleet: slower missions, emptier batteries, activity moved to another zone."""
    row = _normal_row(rng, index)
    row.update(
        duration_ms=rng.randint(14_000, 40_000),
        battery_pct=round(rng.uniform(5, 45), 1),
        x_m=round(rng.uniform(90, 200), 2),
        y_m=round(rng.uniform(60, 140), 2),
    )
    return row


def _telemetry(rng: random.Random, rows: int, labelled: bool) -> List[Dict[str, Any]]:
    records = []
    for index in range(rows):
        anomalous = rng.random() < 0.05
        row = _anomalous_row(rng, index) if anomalous else _normal_row(rng, index)
        if labelled:
            row["is_anomaly"] = int(anomalous)
        records.append(row)
    return records


def _journal(rng: random.Random, rows: int) -> List[Dict[str, Any]]:
    """Operating log with recurring incidents and a few deliberate quality defects."""
    records = []
    for index in range(rows):
        row = _normal_row(rng, index)
        row["timestamp"] = _stamp(index, step_seconds=120)
        row.update(severity="info", code="", message="")
        if rng.random() < 0.08:
            # AMR-03 is deliberately over-represented so that it tops the worst robots KPI.
            row["robot_id"] = "AMR-03" if rng.random() < 0.5 else row["robot_id"]
            code, severity, message = rng.choices(INCIDENTS, weights=(50, 25, 20, 5), k=1)[0]
            row.update(event="error", severity=severity, code=code, message=message)
        records.append(row)
    records[40]["robot_id"] = ""
    records[85]["battery_pct"] = 137.0
    records[130]["duration_ms"] = -250
    records[210]["timestamp"] = ""
    return records


def _write_csv(path: Path, records: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def _write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as output:
        for record in records:
            kept = {key: value for key, value in record.items() if value != ""}
            output.write(json.dumps(kept, ensure_ascii=False) + "\n")


def main() -> None:
    SAMPLES.mkdir(exist_ok=True)
    _write_csv(SAMPLES / "1_entrainement.csv", _telemetry(random.Random(101), 1200, labelled=True))
    _write_csv(SAMPLES / "2_prediction_sans_derive.csv", _telemetry(random.Random(102), 400, labelled=False))
    rng = random.Random(103)
    _write_csv(SAMPLES / "2_prediction_avec_derive.csv", [_drifted_row(rng, index) for index in range(400)])
    journal = _journal(random.Random(104), 500)
    _write_csv(SAMPLES / "3_journaux.csv", journal)
    _write_jsonl(SAMPLES / "3_journaux.jsonl", journal)


if __name__ == "__main__":
    main()
