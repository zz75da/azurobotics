"""The ONNX embedder used by the search app must produce the same vectors as sentence-transformers,
which embeds the documents at indexing time. Downloads both models, so it runs only when
RUN_EMBEDDING_PARITY=1 and sentence-transformers is installed (see search/requirements-index.txt)."""
import os
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(os.environ.get("RUN_EMBEDDING_PARITY") != "1", reason="set RUN_EMBEDDING_PARITY=1")

SENTENCES = [
    "Heat dissipation failure: process minus air temperature below 8.6 K at low speed.",
    "How do I replace a worn cutting tool?",
    "Power above 9000 W during cutting",
    "lockout tagout",
    "Overstrain: tool wear multiplied by torque exceeds the limit of the product variant, which depends on "
    "whether the variant is L, M or H, and requires recovering the machine and inspecting the spindle bearings. " * 8,
]


def test_onnx_vectors_match_sentence_transformers(tmp_path_factory):
    sentence_transformers = pytest.importorskip("sentence_transformers")
    from search.embedder import DIMENSIONS, MODEL_REPO, MODEL_REVISION, MiniLMEmbedder, download_model

    onnx = MiniLMEmbedder(download_model(Path(tmp_path_factory.mktemp("minilm"))))
    reference = sentence_transformers.SentenceTransformer(MODEL_REPO, revision=MODEL_REVISION, device="cpu")
    expected = reference.encode(SENTENCES, normalize_embeddings=True)
    actual = np.array(onnx.embed(SENTENCES))
    assert actual.shape == (len(SENTENCES), DIMENSIONS)
    cosine = (actual * expected).sum(axis=1)
    assert cosine.min() > 0.9999, cosine
    assert np.abs(actual - expected).max() < 1e-3
