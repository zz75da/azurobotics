"""Build the maintenance-procedures index in Azure AI Search from docs/maintenance/*.md.

Each note is split into chunks along its "## " sections; every chunk carries the note's title and
failure type. Chunks are embedded locally with sentence-transformers all-MiniLM-L6-v2 (384 dims,
same pinned revision as the app's ONNX embedder), then uploaded with azure-search-documents.

The admin key is read from SEARCH_ADMIN_KEY or, with --key-vault, from the Key Vault secret
search-admin-key (DefaultAzureCredential: your az login). It is never written anywhere.

    python -m search.build_index --endpoint https://<service>.search.windows.net --key-vault <vault>
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Dict, List

REPO = Path(__file__).resolve().parents[1]
NOTES = REPO / "docs" / "maintenance"
INDEX_NAME = "maintenance-procedures"
FRONT_MATTER = re.compile(r"^---\n(.*?)\n---\n", re.S)


def parse_note(path: Path) -> Dict[str, str]:
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    match = FRONT_MATTER.match(text)
    if not match:
        raise ValueError("{} has no front matter".format(path.name))
    meta = dict(line.split(": ", 1) for line in match.group(1).splitlines() if ": " in line)
    return {"id": meta["id"], "title": meta["title"], "failure_type": meta["failure_type"],
            "body": text[match.end():], "source": "docs/maintenance/" + path.name}


def chunk_note(note: Dict[str, str]) -> List[Dict[str, str]]:
    """One chunk per "## " section; the synthetic-note banner and the H1 title are not chunked."""
    chunks = []
    for number, part in enumerate(re.split(r"^## ", note["body"], flags=re.M)[1:], start=1):
        heading, _, content = part.partition("\n")
        content = content.strip()
        if not content:
            continue
        chunks.append({
            "id": "{}-{}".format(note["id"], number),
            "note_id": note["id"],
            "title": note["title"],
            "failure_type": note["failure_type"],
            "section": heading.strip(),
            "content": content,
            "source": note["source"],
            # Text that is embedded: the title gives each section its context.
            "embedding_text": "{}. {}: {}".format(note["title"], heading.strip(), content),
        })
    return chunks


def load_chunks(folder: Path = NOTES) -> List[Dict[str, str]]:
    notes = [parse_note(path) for path in sorted(folder.glob("*.md")) if path.name != "README.md"]
    return [chunk for note in notes for chunk in chunk_note(note)]


def index_definition(name: str = INDEX_NAME):
    from azure.search.documents.indexes.models import (
        HnswAlgorithmConfiguration,
        HnswParameters,
        SearchableField,
        SearchField,
        SearchFieldDataType,
        SearchIndex,
        SimpleField,
        VectorSearch,
        VectorSearchProfile,
    )

    from search.embedder import DIMENSIONS

    return SearchIndex(
        name=name,
        fields=[
            SimpleField(name="id", type=SearchFieldDataType.String, key=True),
            SimpleField(name="note_id", type=SearchFieldDataType.String, filterable=True),
            SearchableField(name="title", type=SearchFieldDataType.String, analyzer_name="en.lucene"),
            SimpleField(name="failure_type", type=SearchFieldDataType.String, filterable=True, facetable=True),
            SimpleField(name="section", type=SearchFieldDataType.String, filterable=True),
            SearchableField(name="content", type=SearchFieldDataType.String, analyzer_name="en.lucene"),
            SimpleField(name="source", type=SearchFieldDataType.String),
            SearchField(
                name="vector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=DIMENSIONS,
                vector_search_profile_name="minilm",
            ),
        ],
        vector_search=VectorSearch(
            algorithms=[HnswAlgorithmConfiguration(name="hnsw", parameters=HnswParameters(metric="cosine"))],
            profiles=[VectorSearchProfile(name="minilm", algorithm_configuration_name="hnsw")],
        ),
    )


def _admin_key(key_vault: str | None) -> str:
    if os.environ.get("SEARCH_ADMIN_KEY"):
        return os.environ["SEARCH_ADMIN_KEY"]
    if not key_vault:
        raise SystemExit("set SEARCH_ADMIN_KEY or pass --key-vault")
    from azure.identity import DefaultAzureCredential
    from azure.keyvault.secrets import SecretClient

    client = SecretClient("https://{}.vault.azure.net".format(key_vault), DefaultAzureCredential())
    return client.get_secret("search-admin-key").value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--endpoint", default=os.environ.get("SEARCH_ENDPOINT"), required=not os.environ.get("SEARCH_ENDPOINT"))
    parser.add_argument("--key-vault", help="vault holding the search-admin-key secret")
    parser.add_argument("--index", default=INDEX_NAME)
    args = parser.parse_args()

    from azure.core.credentials import AzureKeyCredential
    from azure.search.documents import SearchClient
    from azure.search.documents.indexes import SearchIndexClient
    from sentence_transformers import SentenceTransformer

    from search.embedder import MODEL_REPO, MODEL_REVISION

    chunks = load_chunks()
    model = SentenceTransformer(MODEL_REPO, revision=MODEL_REVISION, device="cpu")
    vectors = model.encode([c["embedding_text"] for c in chunks], normalize_embeddings=True, batch_size=16)
    documents = [
        {**{k: v for k, v in chunk.items() if k != "embedding_text"}, "vector": vector.tolist()}
        for chunk, vector in zip(chunks, vectors)
    ]
    credential = AzureKeyCredential(_admin_key(args.key_vault))
    SearchIndexClient(args.endpoint, credential).create_or_update_index(index_definition(args.index))
    results = SearchClient(args.endpoint, args.index, credential).upload_documents(documents)
    failed = [r.key for r in results if not r.succeeded]
    notes = len({c["note_id"] for c in chunks})
    print("indexed {} chunks from {} notes into {} ({} failed)".format(len(documents) - len(failed), notes, args.index, len(failed)))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.path.insert(0, str(REPO))
    sys.exit(main())
