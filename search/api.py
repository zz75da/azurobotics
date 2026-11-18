"""Search API for the synthetic maintenance procedures (personal project).

GET  /search       hybrid keyword + vector query, top-k procedures, optional failure_type filter
POST /procedures   AI4I machine reading -> likely failure types (documented AI4I rules) -> procedures
GET  /health

Queries are embedded with the ONNX MiniLM embedder. Each query writes one JSON log line (latency,
top-k, failure type, result count; never the query text), which reaches Log Analytics through the
Container Apps environment. No answer is generated: the API returns the retrieved passages.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from search.failure_types import Reading, likely_failure_types

LOG = logging.getLogger("robotics-search")
FAILURE_TYPES = ("TWF", "HDF", "PWF", "OSF", "RNF", "GENERAL")
SELECT = ["note_id", "title", "failure_type", "section", "content", "source"]
# What /procedures searches for each type. TWF is a risk window, not a predicted failure (see README).
TYPE_QUERIES = {
    "TWF": ("tool-wear risk window (200-240 min)", "inspect and replace a worn cutting tool"),
    "HDF": ("heat dissipation failure condition", "cooling inspection when the temperature difference is low at low speed"),
    "PWF": ("power failure condition", "spindle power outside the 3500 to 9000 W band"),
    "OSF": ("overstrain failure condition", "overstrain recovery and tool wear times torque limit"),
}


class ReadingIn(BaseModel):
    product_type: str = Field(pattern="^[LMH]$", description="AI4I product variant: L, M or H")
    air_temperature_k: float = Field(gt=0)
    process_temperature_k: float = Field(gt=0)
    rotational_speed_rpm: float = Field(ge=0)
    torque_nm: float = Field(ge=0)
    tool_wear_min: float = Field(ge=0)


def _log(event: str, **fields: Any) -> None:
    LOG.info(json.dumps({"event": event, **fields}))


def _group_by_note(hits: List[Dict[str, Any]], top: int) -> List[Dict[str, Any]]:
    """Several chunks can come from one note; keep each note once, at its best rank."""
    notes: Dict[str, Dict[str, Any]] = {}
    for hit in hits:
        note = notes.setdefault(hit["note_id"], {
            "note_id": hit["note_id"], "title": hit["title"], "failure_type": hit["failure_type"],
            "source": hit["source"], "score": hit.get("@search.score"), "passages": [],
        })
        note["passages"].append({"section": hit["section"], "content": hit["content"]})
    return list(notes.values())[:top]


def create_app(search_client: Any, embedder: Any) -> FastAPI:
    app = FastAPI(title="Robotics maintenance search (synthetic notes)", version="1.0")

    def run_search(text: str, top: int, failure_type: Optional[str]) -> List[Dict[str, Any]]:
        from azure.search.documents.models import VectorizedQuery

        vector = embedder.embed([text])[0]
        hits = search_client.search(
            search_text=text,
            vector_queries=[VectorizedQuery(vector=vector, k_nearest_neighbors=top * 3, fields="vector")],
            filter="failure_type eq '{}'".format(failure_type) if failure_type else None,
            select=SELECT,
            top=top * 3,  # chunks; grouped to `top` notes below
        )
        return _group_by_note(list(hits), top)

    @app.get("/health")
    def health() -> Dict[str, str]:
        return {"status": "ok"}

    @app.get("/search")
    def search(
        q: str = Query(min_length=2, max_length=500),
        k: int = Query(3, ge=1, le=10),
        failure_type: Optional[str] = Query(None),
    ) -> Dict[str, Any]:
        if failure_type and failure_type not in FAILURE_TYPES:
            raise HTTPException(422, "failure_type must be one of " + ", ".join(FAILURE_TYPES))
        started = time.perf_counter()
        results = run_search(q, k, failure_type)
        _log("search_query", endpoint="search", latency_ms=round((time.perf_counter() - started) * 1000, 1),
             top_k=k, failure_type=failure_type, results=len(results), query_chars=len(q))
        return {"query": q, "top_k": k, "failure_type": failure_type, "results": results}

    @app.post("/procedures")
    def procedures(reading: ReadingIn, k: int = Query(2, ge=1, le=5)) -> Dict[str, Any]:
        started = time.perf_counter()
        types = likely_failure_types(Reading(**reading.model_dump()))
        matches = []
        for kind in types:
            label, query = TYPE_QUERIES[kind]
            matches.append({
                "failure_type": kind,
                "label": label,
                "kind": "risk window" if kind == "TWF" else "documented failure condition met",
                "procedures": run_search(query, k, kind),
            })
        _log("search_query", endpoint="procedures", latency_ms=round((time.perf_counter() - started) * 1000, 1),
             top_k=k, failure_type=",".join(types) or None, results=sum(len(m["procedures"]) for m in matches))
        message = None if types else "No documented AI4I failure condition is met for this reading."
        return {"failure_types": types, "matches": matches, "message": message,
                "note": "Synthetic procedures for a personal project; not real maintenance instructions."}

    return app


def _production_app() -> FastAPI:
    from azure.core.credentials import AzureKeyCredential
    from azure.search.documents import SearchClient

    from search.embedder import MiniLMEmbedder

    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
    client = SearchClient(
        os.environ["SEARCH_ENDPOINT"],
        os.environ.get("SEARCH_INDEX", "maintenance-procedures"),
        AzureKeyCredential(os.environ["SEARCH_QUERY_KEY"]),
    )
    return create_app(client, MiniLMEmbedder(Path(os.environ.get("MODEL_DIR", "/models/minilm"))))


app = _production_app() if os.environ.get("SEARCH_ENDPOINT") else None
