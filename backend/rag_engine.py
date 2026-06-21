"""
rag_engine.py — Retrieval-Augmented Generation pipeline

Flow:
  1. Embed the user query (multilingual-MiniLM)
  2. Search Qdrant with optional metadata filters
  3. Build a context prompt from top-k chunks
  4. Send to Ollama (local LLM) for answer generation
  5. Return answer + source citations

Graceful degradation: if Ollama is not running, returns retrieved chunks
without LLM generation (still useful for testing).
"""

import os
import logging
from typing import List, Optional, Dict, Any

from embedder import embed
import vector_store as vs

logger = logging.getLogger(__name__)

# Smallest viable bilingual (bn+en) model. ~400 MB, fastest local option.
# Override with env var, e.g. OLLAMA_MODEL=qwen2.5:1.5b for better answer quality.
OLLAMA_MODEL   = os.getenv("OLLAMA_MODEL", "qwen2.5:0.5b")
MAX_CONTEXT    = 3000              # max characters fed to LLM context
TOP_K_RETRIEVE = 5


def search_and_answer(
    query     : str,
    language  : Optional[str] = None,
    doc_type  : Optional[str] = None,
    date_from : Optional[str] = None,
    date_to   : Optional[str] = None,
    doc_id    : Optional[str] = None,
    top_k     : int = TOP_K_RETRIEVE,
) -> Dict[str, Any]:
    """
    Full RAG pipeline.

    Args:
        query    : Natural language question (Bangla or English)
        language : Optional metadata filter ('bn', 'en', 'mixed', 'all')
        doc_type : Optional filter ('pdf', 'image', 'all')
        date_from: Optional ISO date string (inclusive lower bound)
        date_to  : Optional ISO date string (inclusive upper bound)
        doc_id   : Optional — restrict search to a specific document
        top_k    : Number of chunks to retrieve

    Returns dict with keys:
        answer   : str — LLM-generated answer (or chunk summary if no Ollama)
        sources  : list of source chunk dicts
        query    : echo of the original query
        filters  : echo of applied filters
        model    : LLM model used (or "no-llm")
    """
    # ── Step 1: Embed query ──────────────────────────────────────────────────
    logger.info(f"RAG query: '{query}' | filters: lang={language} type={doc_type}")
    query_vector = embed(query)[0]

    # ── Step 2: Retrieve relevant chunks ─────────────────────────────────────
    hits = vs.search(
        query_vector = query_vector,
        language     = language,
        doc_type     = doc_type,
        date_from    = date_from,
        date_to      = date_to,
        doc_id       = doc_id,
        limit        = top_k,
    )

    if not hits:
        return {
            "answer"  : "No relevant documents found. Please upload documents first or adjust your filters.",
            "sources" : [],
            "query"   : query,
            "filters" : _filter_summary(language, doc_type, date_from, date_to),
            "model"   : "no-results",
        }

    # ── Step 3: Build context ────────────────────────────────────────────────
    context_parts = []
    char_count    = 0

    for i, hit in enumerate(hits):
        chunk_text = hit["text"]
        if char_count + len(chunk_text) > MAX_CONTEXT:
            # Truncate last chunk to fit budget
            remaining = MAX_CONTEXT - char_count
            if remaining > 100:
                chunk_text = chunk_text[:remaining] + "..."
            else:
                break
        context_parts.append(
            f"[Source {i+1}: {hit['filename']}, page {hit['page_number']}]\n{chunk_text}"
        )
        char_count += len(chunk_text)

    context = "\n\n---\n\n".join(context_parts)

    # ── Step 4: Generate answer via Ollama ────────────────────────────────────
    answer, model_used = _generate_answer(query, context)

    return {
        "answer"  : answer,
        "sources" : hits,
        "query"   : query,
        "filters" : _filter_summary(language, doc_type, date_from, date_to),
        "model"   : model_used,
    }


def _generate_answer(query: str, context: str) -> tuple[str, str]:
    """Call Ollama for answer generation. Falls back gracefully if unavailable."""
    system_prompt = (
        "You are a helpful assistant that answers questions based strictly on the provided context. "
        "The context may contain text in Bengali (Bangla) or English. "
        "Answer in the same language as the question. "
        "If the answer is not in the context, say so clearly. "
        "Do not make up information. Cite source numbers when relevant."
    )
    user_message = f"Context:\n{context}\n\nQuestion: {query}"

    try:
        import ollama
        response = ollama.chat(
            model    = OLLAMA_MODEL,
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_message},
            ],
            options  = {
                "temperature": 0.2,   # deterministic, fewer tokens wasted
                "num_predict": 512,   # cap answer length → faster response
                "num_ctx"    : 4096,  # enough for context + query
            },
        )
        answer = response["message"]["content"]
        logger.info(f"Ollama answered ({len(answer)} chars) via {OLLAMA_MODEL}")
        return answer, OLLAMA_MODEL

    except Exception as e:
        logger.warning(f"Ollama unavailable ({e}). Returning retrieved chunks as answer.")
        # Graceful degradation: summarise what was found
        filenames = list({h["filename"] for h in []})
        fallback  = (
            f"[Ollama not available — showing retrieved context]\n\n"
            f"Found {len(context.split(chr(10))) } lines from your documents "
            f"relevant to: '{query}'\n\n"
            f"{context[:1500]}..."
        )
        return fallback, "no-llm"


def _filter_summary(language, doc_type, date_from, date_to) -> Dict[str, Any]:
    return {
        "language" : language or "all",
        "doc_type" : doc_type or "all",
        "date_from": date_from,
        "date_to"  : date_to,
    }
