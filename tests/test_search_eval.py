"""Recall@3 of the live search API on 10 questions. Skipped unless SEARCH_API_URL is set."""
import os

import pytest

from search.build_index import load_chunks
from search.evaluate import QUESTIONS, evaluate

SEARCH_API_URL = os.environ.get("SEARCH_API_URL")


def test_every_expected_note_exists():
    note_ids = {chunk["note_id"] for chunk in load_chunks()}
    assert len(QUESTIONS) == 10
    assert all(expected in note_ids for _, expected in QUESTIONS)


@pytest.mark.skipif(not SEARCH_API_URL, reason="set SEARCH_API_URL to a running search API")
def test_recall_at_3_on_the_live_index():
    report = evaluate(SEARCH_API_URL, k=3)
    misses = [row for row in report["rows"] if not row["rank"]]
    print("recall@3 = {:.2f}; misses: {}".format(report["recall_at_k"], misses))
    assert report["recall_at_k"] >= 0.8
