"""Retrieval evaluation: 10 questions with the note each one should find, scored as recall@3.

Runs against a deployed search API (SEARCH_API_URL), so it measures the real path: ONNX query
embedding, hybrid search in Azure AI Search, grouping by note.

    SEARCH_API_URL=http://127.0.0.1:8899 python -m search.evaluate
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import time
import urllib.parse
import urllib.request
from typing import Dict, List, Tuple

# Questions are phrased the way an operator would ask, not copied from the notes.
QUESTIONS: List[Tuple[str, str]] = [
    ("How often should I check the cutting tool for wear?", "twf-01-tool-wear-inspection"),
    ("steps to swap a worn tool and reset the wear counter", "twf-02-tool-replacement"),
    ("what must I check after a tool snapped during machining", "twf-04-after-tool-breakage"),
    ("the cooling fans and the cabinet filter need a check", "hdf-01-cooling-inspection"),
    ("how to remove dust from the heat exchanger fins", "hdf-02-heat-exchanger-cleaning"),
    ("coolant level is low and the pump makes a strange noise", "hdf-04-coolant-pump-check"),
    ("the spindle stalls because it does not get enough power", "pwf-02-low-power-stall"),
    ("power is over 9000 W, should I slow the feed?", "pwf-03-overpower-load"),
    ("what is the torque limit for product variant H", "osf-02-torque-limits-by-variant"),
    ("the machine stopped and nobody knows why", "rnf-01-unexplained-stop"),
]


def ask(base_url: str, question: str, k: int = 3) -> Tuple[List[str], float]:
    url = "{}/search?{}".format(base_url.rstrip("/"), urllib.parse.urlencode({"q": question, "k": k}))
    started = time.perf_counter()
    with urllib.request.urlopen(url, timeout=60) as response:
        body = json.load(response)
    return [r["note_id"] for r in body["results"]], (time.perf_counter() - started) * 1000


def evaluate(base_url: str, k: int = 3) -> Dict[str, object]:
    rows, latencies = [], []
    for question, expected in QUESTIONS:
        found, latency = ask(base_url, question, k)
        latencies.append(latency)
        rows.append({"question": question, "expected": expected, "found": found,
                     "rank": found.index(expected) + 1 if expected in found else None})
    hits = sum(1 for row in rows if row["rank"])
    return {"recall_at_k": hits / len(rows), "k": k, "rows": rows,
            "client_latency_ms_p50": round(statistics.median(latencies), 1)}


if __name__ == "__main__":
    report = evaluate(os.environ["SEARCH_API_URL"])
    for row in report["rows"]:
        print("{:>4}  {:<55} {}".format(row["rank"] or "-", row["question"][:55], row["expected"]))
    print("recall@{}: {:.2f}".format(report["k"], report["recall_at_k"]))
    sys.exit(0)
