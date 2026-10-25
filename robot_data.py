from __future__ import annotations

import csv
import io
import json
import random
import tempfile
import zipfile
from datetime import datetime, timedelta, timezone
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
        "incident_groups": _incident_groups(records),
        "kpis": _kpis(records),
        "sample": records[:5],
    }


ERROR_EVENTS = ("error", "fault", "failure")
ERROR_SEVERITIES = ("error", "critical", "fatal", "warning")


def _is_incident(row: Dict[str, Any]) -> bool:
    event = str(row.get("event", row.get("type", ""))).strip().lower()
    severity = str(row.get("severity", "")).strip().lower()
    return event in ERROR_EVENTS or severity in ERROR_SEVERITIES or bool(row.get("code"))


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _incident_groups(records: List[Dict[str, Any]], limit: int = 10) -> List[Dict[str, Any]]:
    """Group incidents by code (or message, or event) to show the most recurrent problems first."""
    groups: Dict[str, Dict[str, Any]] = {}
    for row in records:
        if not _is_incident(row):
            continue
        key = str(row.get("code") or row.get("message") or row.get("event") or row.get("type") or "inconnu").strip()[:120]
        group = groups.setdefault(key, {"key": key, "count": 0, "robots": set(), "first_seen": None, "last_seen": None})
        group["count"] += 1
        if row.get("robot_id") not in (None, ""):
            group["robots"].add(str(row["robot_id"]))
        stamp = _parse_timestamp(row.get("timestamp"))
        if stamp is not None:
            if group["first_seen"] is None or stamp < group["first_seen"]:
                group["first_seen"] = stamp
            if group["last_seen"] is None or stamp > group["last_seen"]:
                group["last_seen"] = stamp
    ranked = sorted(groups.values(), key=lambda item: (-item["count"], item["key"]))[:limit]
    return [
        {
            "key": item["key"],
            "count": item["count"],
            "robots_affected": len(item["robots"]),
            "first_seen": item["first_seen"].isoformat() if item["first_seen"] else None,
            "last_seen": item["last_seen"].isoformat() if item["last_seen"] else None,
        }
        for item in ranked
    ]


def _kpis(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Service KPIs computed only from the provided rows (not an SLA)."""
    per_robot: Dict[str, List[int]] = {}
    per_day: Dict[str, List[int]] = {}
    incident_times: List[datetime] = []
    labelled = 0
    anomalies = 0
    for row in records:
        incident = _is_incident(row)
        robot = row.get("robot_id")
        if robot not in (None, ""):
            stats = per_robot.setdefault(str(robot), [0, 0])
            stats[0] += 1
            stats[1] += int(incident)
        stamp = _parse_timestamp(row.get("timestamp"))
        if stamp is not None:
            day = per_day.setdefault(stamp.date().isoformat(), [0, 0])
            day[0] += 1
            day[1] += int(incident)
            if incident:
                incident_times.append(stamp)
        label = _number(row.get("is_anomaly"))
        if label is not None:
            labelled += 1
            anomalies += int(label == 1)
    incident_times.sort()
    gaps = [(b - a).total_seconds() / 60 for a, b in zip(incident_times, incident_times[1:])]
    robots = sorted(
        (
            {"robot_id": name, "events": s[0], "incidents": s[1], "incident_rate_pct": round(100 * s[1] / s[0], 2)}
            for name, s in per_robot.items()
        ),
        key=lambda item: (-item["incident_rate_pct"], -item["incidents"], item["robot_id"]),
    )[:10]
    return {
        "mean_time_between_incidents_min": round(sum(gaps) / len(gaps), 1) if gaps else None,
        "anomaly_rate_pct": round(100 * anomalies / labelled, 2) if labelled else None,
        "worst_robots": robots,
        "daily": [
            {"day": day, "events": s[0], "incidents": s[1]} for day, s in sorted(per_day.items())[-14:]
        ],
    }


def _number(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result == result and abs(result) != float("inf") else None
