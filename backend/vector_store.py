"""
vector_store.py — Qdrant local vector store

Collection schema:
  - vector: 384-dim cosine (multilingual-MiniLM)
  - payload: doc_id, filename, doc_type, language, page_number,
             chunk_index, text, upload_date, total_chunks

Metadata filtering works by combining Qdrant's Filter (must/should/must_not)
with vector similarity search in a single query — O(log n) not O(n).
"""

import uuid
import logging
from datetime import date
from typing import List, Optional, Dict, Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    Range,
    PointIdsList,
)
import numpy as np

from embedder import VECTOR_DIM

logger = logging.getLogger(__name__)

COLLECTION_NAME = "document_chunks"
QDRANT_PATH     = "./qdrant_storage"

# ---------------------------------------------------------------------------
# Singleton Qdrant client
# ---------------------------------------------------------------------------
_client: Optional[QdrantClient] = None


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(path=QDRANT_PATH)
        _ensure_collection(_client)
    return _client


def _ensure_collection(client: QdrantClient):
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_DIM,
                distance=Distance.COSINE,
            ),
        )
        logger.info(f"Created Qdrant collection: {COLLECTION_NAME}")
    else:
        logger.info(f"Using existing Qdrant collection: {COLLECTION_NAME}")


# ---------------------------------------------------------------------------
# Write operations
# ---------------------------------------------------------------------------

def add_document(
    doc_id: str,
    filename: str,
    doc_type: str,
    language: str,
    chunks: List[str],
    embeddings: np.ndarray,
    page_numbers: List[int],
    upload_date: Optional[str] = None,
) -> int:
    """
    Store all chunks of a document in Qdrant.

    Returns: number of points stored
    """
    client = get_client()
    today  = upload_date or date.today().isoformat()
    points = []

    for i, (chunk_text, embedding, page_num) in enumerate(
        zip(chunks, embeddings, page_numbers)
    ):
        point_id = str(uuid.uuid4())
        points.append(
            PointStruct(
                id=point_id,
                vector=embedding.tolist(),
                payload={
                    "doc_id"        : doc_id,
                    "filename"      : filename,
                    "doc_type"      : doc_type,
                    "language"      : language,
                    "page_number"   : page_num,
                    "chunk_index"   : i,
                    "text"          : chunk_text,
                    "upload_date"   : today,
                    "total_chunks"  : len(chunks),
                    "char_count"    : len(chunk_text),
                },
            )
        )

    # Batch upsert
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    logger.info(f"Stored {len(points)} chunks for doc_id={doc_id}")
    return len(points)


# ---------------------------------------------------------------------------
# Search operations
# ---------------------------------------------------------------------------

def search(
    query_vector: np.ndarray,
    language    : Optional[str] = None,
    doc_type    : Optional[str] = None,
    date_from   : Optional[str] = None,
    date_to     : Optional[str] = None,
    doc_id      : Optional[str] = None,
    limit       : int = 5,
) -> List[Dict[str, Any]]:
    """
    Hybrid search: metadata filter + vector similarity.

    The filter narrows the search space to matching documents.
    Qdrant then ranks those by cosine similarity to the query vector.

    This is NOT a two-stage post-filter — Qdrant applies both simultaneously,
    so ranking quality is preserved within the filtered subset.
    """
    client   = get_client()
    must     = _build_filters(language, doc_type, date_from, date_to, doc_id)
    qfilter  = Filter(must=must) if must else None

    if hasattr(client, "query_points"):
        response = client.query_points(
            collection_name = COLLECTION_NAME,
            query           = query_vector.tolist(),
            query_filter    = qfilter,
            limit           = limit,
            with_payload    = True,
        )
        results = response.points
    else:
        results = client.search(
            collection_name = COLLECTION_NAME,
            query_vector    = query_vector.tolist(),
            query_filter    = qfilter,
            limit           = limit,
            with_payload    = True,
        )

    return [
        {
            "score"     : hit.score,
            "text"      : hit.payload["text"],
            "filename"  : hit.payload["filename"],
            "doc_id"    : hit.payload["doc_id"],
            "language"  : hit.payload["language"],
            "doc_type"  : hit.payload["doc_type"],
            "page_number": hit.payload["page_number"],
            "chunk_index": hit.payload["chunk_index"],
            "upload_date": hit.payload["upload_date"],
        }
        for hit in results
    ]


def _build_filters(language, doc_type, date_from, date_to, doc_id):
    """Construct Qdrant filter conditions from user-supplied metadata criteria."""
    conditions = []

    if language and language != "all":
        conditions.append(
            FieldCondition(key="language", match=MatchValue(value=language))
        )
    if doc_type and doc_type != "all":
        conditions.append(
            FieldCondition(key="doc_type", match=MatchValue(value=doc_type))
        )
    if doc_id:
        conditions.append(
            FieldCondition(key="doc_id", match=MatchValue(value=doc_id))
        )
    # Date range filter
    date_range = {}
    if date_from:
        date_range["gte"] = date_from
    if date_to:
        date_range["lte"] = date_to
    if date_range:
        conditions.append(
            FieldCondition(key="upload_date", range=Range(**date_range))
        )

    return conditions


# ---------------------------------------------------------------------------
# Document management
# ---------------------------------------------------------------------------

def list_documents() -> List[Dict[str, Any]]:
    """
    Return a deduplicated list of all indexed documents (not chunks).
    Uses scroll to retrieve all points and deduplicates by doc_id.
    """
    client = get_client()
    docs   = {}

    offset = None
    while True:
        result, next_offset = client.scroll(
            collection_name = COLLECTION_NAME,
            limit           = 200,
            offset          = offset,
            with_payload    = True,
            with_vectors    = False,
        )
        for point in result:
            p = point.payload
            did = p["doc_id"]
            if did not in docs:
                docs[did] = {
                    "doc_id"      : did,
                    "filename"    : p["filename"],
                    "doc_type"    : p["doc_type"],
                    "language"    : p["language"],
                    "upload_date" : p["upload_date"],
                    "total_chunks": p["total_chunks"],
                }

        if next_offset is None:
            break
        offset = next_offset

    return list(docs.values())


def delete_document(doc_id: str) -> int:
    """
    Delete all chunks belonging to a document.
    Returns number of points deleted.
    """
    client = get_client()

    # Find all point IDs for this doc_id
    point_ids = []
    offset = None
    while True:
        result, next_offset = client.scroll(
            collection_name = COLLECTION_NAME,
            scroll_filter   = Filter(
                must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
            ),
            limit           = 200,
            offset          = offset,
            with_payload    = False,
            with_vectors    = False,
        )
        point_ids.extend([str(p.id) for p in result])
        if next_offset is None:
            break
        offset = next_offset

    if point_ids:
        client.delete(
            collection_name = COLLECTION_NAME,
            points_selector = PointIdsList(points=point_ids),
        )

    logger.info(f"Deleted {len(point_ids)} chunks for doc_id={doc_id}")
    return len(point_ids)


def get_stats() -> Dict[str, Any]:
    """Return collection statistics."""
    client = get_client()
    info   = client.get_collection(COLLECTION_NAME)
    return {
        "total_chunks" : info.points_count,
        "vector_size"  : VECTOR_DIM,
        "collection"   : COLLECTION_NAME,
    }
