"""Convert the UCI AI4I 2020 predictive-maintenance dataset (CC BY 4.0) to the app's telemetry CSV.

Usage:
    python tools/convert_ai4i.py [--input ai4i2020.csv] [--output ai4i_telemetry.csv] [--limit N]

Without --input the file is downloaded from UCI. The app's model reads four generic numeric
columns, so the real sensor values are stored in them as follows:

    duration_ms  <- Rotational speed [rpm]
    battery_pct  <- Torque [Nm]
    x_m          <- Process temperature [K]
    y_m          <- Tool wear [min]
    is_anomaly   <- Machine failure (0/1)

The column names are only containers: the values are real machine data, not robot battery or
position. Dataset: https://archive.ics.uci.edu/dataset/601 (Matzka, 2020), licence CC BY 4.0.
"""
from __future__ import annotations

import argparse
import csv
import io
import sys
import urllib.request
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

URL = "https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip"
HEADER = ("timestamp", "robot_id", "event", "duration_ms", "battery_pct", "x_m", "y_m", "is_anomaly")


def load_source(path: str | None) -> str:
    if path:
        return Path(path).read_text(encoding="utf-8-sig")
    with urllib.request.urlopen(URL, timeout=60) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    name = next(n for n in archive.namelist() if n.lower().endswith(".csv"))
    return archive.read(name).decode("utf-8-sig")


def convert(source: str, limit: int | None = None) -> str:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(HEADER)
    for index, row in enumerate(csv.DictReader(io.StringIO(source))):
        if limit is not None and index >= limit:
            break
        timestamp = (start + timedelta(minutes=int(row["UDI"]))).strftime("%Y-%m-%dT%H:%M:%SZ")
        writer.writerow(
            (
                timestamp,
                "MACH-" + row["Type"],
                "failure" if row["Machine failure"] == "1" else "cycle",
                row["Rotational speed [rpm]"],
                row["Torque [Nm]"],
                row["Process temperature [K]"],
                row["Tool wear [min]"],
                int(row["Machine failure"]),
            )
        )
    return output.getvalue()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--input", help="local ai4i2020.csv (downloaded from UCI if omitted)")
    parser.add_argument("--output", default="ai4i_telemetry.csv")
    parser.add_argument("--limit", type=int, help="maximum number of rows")
    args = parser.parse_args()
    text = convert(load_source(args.input), args.limit)
    with open(args.output, "w", encoding="utf-8", newline="") as handle:
        handle.write(text)
    print("{} lignes écrites dans {}".format(text.count("\n") - 1, args.output))
    return 0


if __name__ == "__main__":
    sys.exit(main())
