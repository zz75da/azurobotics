from __future__ import annotations

import csv
import io
import json
import random
import tempfile
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

from video_privacy import create_sample_video


SAMPLE_ROWS = 600
MAX_ROWS = 200_000


def make_sample_telemetry_csv() -> str:
    rng = random.Random(42)
    start = datetime(2026, 1, 15, 8, 0, 0)
    events = ("move", "pick", "drop", "charge", "stop", "error")
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(("timestamp", "robot_id", "event", "duration_ms", "battery_pct", "x_m", "y_m", "is_anomaly"))
    for index in range(SAMPLE_ROWS):
        event = rng.choices(events, weights=(45, 22, 20, 5, 6, 2), k=1)[0]
        timestamp = start + timedelta(seconds=15 * index)
        is_anomaly = index % 20 == 0
        row = {
            "timestamp": timestamp.isoformat() + "Z",
            "robot_id": "AMR-{:02d}".format(rng.randint(1, 8)),
            "event": event,
            "duration_ms": 180_000 if is_anomaly else rng.randint(200, 12_000),
            "battery_pct": -10.0 if is_anomaly else round(rng.uniform(18, 100), 1),
            "x_m": -40.0 if is_anomaly else round(rng.uniform(0, 120), 2),
            "y_m": 150.0 if is_anomaly else round(rng.uniform(0, 80), 2),
            "is_anomaly": int(is_anomaly),
        }
        writer.writerow(row.values())
    return output.getvalue()


def make_sample_bundle() -> bytes:
    telemetry = make_sample_telemetry_csv()
    incidents = []
    for row in csv.DictReader(io.StringIO(telemetry)):
        if row["event"] in ("error", "stop"):
            incidents.append(
                {
                    "timestamp": row["timestamp"],
                    "robot_id": row["robot_id"],
                    "severity": "error" if row["event"] == "error" else "warning",
                    "code": "TEST-{}".format(row["event"].upper()),
                    "message": "Incident synthétique pour le robot {}".format(row["robot_id"]),
                }
            )
    incident_text = "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in incidents)

    with tempfile.TemporaryDirectory(prefix="robotics-demo-") as temporary:
        video_path = Path(temporary) / "warehouse_test.avi"
        create_sample_video(video_path)
        bundle = io.BytesIO()
        with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            archive.writestr("data/telemetry.csv", telemetry)
            archive.writestr("data/incidents.jsonl", incident_text)
            archive.write(str(video_path), "video/warehouse_test.avi")
            archive.writestr(
                "README.txt",
                "Jeu entièrement synthétique, généré localement. Aucune donnée personnelle.\n"
                "telemetry.csv : événements AMR, horodatage, batterie, position, durée et label is_anomaly synthétique.\n"
                "incidents.jsonl : incidents de test associés aux événements stop/error.\n"
                "video/warehouse_test.avi : scène dessinée, sans personne réelle.\n"
                "Les métriques sont illustratives et ne représentent pas un SLA réel.\n",
            )
        return bundle.getvalue()


def analyze_text(content: str, suffix: str) -> Dict[str, Any]:
    if suffix == ".csv":
        records = []
        for record in csv.DictReader(io.StringIO(content)):
            if len(records) >= MAX_ROWS:
                raise ValueError("Limite de {} lignes dépassée.".format(MAX_ROWS))
            records.append(record)
    else:
        records = []
        for line_number, line in enumerate(content.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError("JSONL invalide à la ligne {}.".format(line_number)) from error
            if not isinstance(record, dict):
                raise ValueError("Chaque ligne JSONL doit être un objet.")
            if len(records) >= MAX_ROWS:
                raise ValueError("Limite de {} lignes dépassée.".format(MAX_ROWS))
            records.append(record)
    if not records:
        raise ValueError("Le fichier ne contient aucune ligne exploitable.")

    event_counts: Dict[str, int] = {}
    robots = set()
    batteries: List[float] = []
    durations: List[float] = []
    error_count = 0
    quality_issues = {
        "missing_timestamp": 0,
        "missing_robot_id": 0,
        "missing_event": 0,
        "invalid_battery_pct": 0,
        "invalid_duration_ms": 0,
    }
    for row in records:
        for field in ("timestamp", "robot_id", "event"):
            value = row.get(field, row.get("type") if field == "event" else None)
            if value is None or (isinstance(value, str) and not value.strip()):
                quality_issues["missing_{}".format(field)] += 1
        event = str(row.get("event", row.get("type", "inconnu"))).strip().lower() or "inconnu"
        event_counts[event] = event_counts.get(event, 0) + 1
        robot_id = row.get("robot_id")
        if robot_id not in (None, ""):
            robots.add(str(robot_id))
        if event in ("error", "fault", "failure"):
            error_count += 1
        battery = _number(row.get("battery_pct"))
        if battery is not None:
            batteries.append(battery)
            if not 0 <= battery <= 100:
                quality_issues["invalid_battery_pct"] += 1
        elif row.get("battery_pct") not in (None, ""):
            quality_issues["invalid_battery_pct"] += 1
        duration = _number(row.get("duration_ms"))
        if duration is not None:
            durations.append(duration)
            if duration < 0:
                quality_issues["invalid_duration_ms"] += 1
        elif row.get("duration_ms") not in (None, ""):
            quality_issues["invalid_duration_ms"] += 1

    count = len(records)
    return {
        "events_total": count,
        "robots_total": len(robots),
        "errors_total": error_count,
        "error_rate_pct": round(100 * error_count / count, 2),
        "battery_avg_pct": round(sum(batteries) / len(batteries), 1) if batteries else None,
        "duration_avg_ms": round(sum(durations) / len(durations)) if durations else None,
        "quality_issues": quality_issues,
        "event_counts": dict(sorted(event_counts.items(), key=lambda item: (-item[1], item[0]))),
        "sample": records[:5],
    }


def _number(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result == result and abs(result) != float("inf") else None
