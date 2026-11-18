"""Pipeline step 1: AI4I 2020 raw CSV -> app telemetry schema, split into train and test files.

Reuses tools/convert_ai4i.py so the Azure ML pipeline and the app see the same columns. Writes
train.csv, test.csv and dataset.json (SHA-256 of the raw file, row and failure counts) to --output.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sklearn.model_selection import train_test_split  # noqa: E402

from tools.convert_ai4i import convert  # noqa: E402

FAILURE_COLUMNS = ("TWF", "HDF", "PWF", "OSF", "RNF")


def run_prep(raw_text: str, output_dir: Path, test_size: float = 0.25, seed: int = 42) -> Dict[str, Any]:
    converted = list(csv.reader(io.StringIO(convert(raw_text))))
    header, rows = converted[0], converted[1:]
    labels = [row[header.index("is_anomaly")] for row in rows]
    if len(set(labels)) != 2:
        raise ValueError("The raw file must contain both failure and non-failure rows.")
    train_rows, test_rows = train_test_split(rows, test_size=test_size, random_state=seed, stratify=labels)

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, part in (("train.csv", train_rows), ("test.csv", test_rows)):
        with (output_dir / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(part)

    raw_rows = list(csv.DictReader(io.StringIO(raw_text)))
    summary = {
        "dataset": "AI4I 2020 predictive maintenance (UCI 601, CC BY 4.0)",
        "dataset_sha256": hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
        "rows": len(rows),
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "failures": sum(int(label) for label in labels),
        "failure_types": {name: sum(int(row.get(name, 0) or 0) for row in raw_rows) for name in FAILURE_COLUMNS},
        "test_size": test_size,
        "seed": seed,
    }
    (output_dir / "dataset.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--raw", required=True, help="AI4I 2020 CSV as published by UCI")
    parser.add_argument("--output", required=True, help="folder for train.csv, test.csv and dataset.json")
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    summary = run_prep(Path(args.raw).read_text(encoding="utf-8-sig"), Path(args.output), args.test_size, args.seed)
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
