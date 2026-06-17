"""
chunker.py — Bilingual text chunker (Bangla + English)

Bangla sentences end with '।' (daari).
English sentences end with '.', '!', '?'.
"""

import re
from typing import List


def chunk_text(
    text: str,
    chunk_size: int = 400,
    overlap: int = 80,
) -> List[str]:
    """
    Split multilingual text (Bangla + English) into overlapping chunks.

    Strategy:
    1. Split on sentence boundaries first (respects both scripts)
    2. Accumulate sentences until chunk_size is reached
    3. Keep last N characters as overlap for next chunk

    Args:
        text: Raw extracted text (may contain Bangla and/or English)
        chunk_size: Target chunk size in characters
        overlap: Characters to carry over into the next chunk

    Returns:
        List of chunk strings
    """
    if not text or not text.strip():
        return []

    text = _clean_text(text)

    # Split on Bangla daari (।), and English sentence endings (. ! ?)
    # Positive lookbehind keeps the delimiter attached to the sentence
    sentences = re.split(r'(?<=[।.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return [text] if text.strip() else []

    chunks: List[str] = []
    current_parts: List[str] = []
    current_len: int = 0

    for sentence in sentences:
        sentence_len = len(sentence)

        # If a single sentence exceeds chunk_size, hard-split it
        if sentence_len > chunk_size:
            if current_parts:
                chunks.append(" ".join(current_parts))
            hard_chunks = _hard_split(sentence, chunk_size, overlap)
            chunks.extend(hard_chunks[:-1])
            # Keep the last hard chunk as the start of next accumulation
            current_parts = [hard_chunks[-1]] if hard_chunks else []
            current_len = len(current_parts[0]) if current_parts else 0
            continue

        # Normal accumulation
        if current_len + sentence_len + 1 > chunk_size and current_parts:
            # Flush current chunk
            chunks.append(" ".join(current_parts))

            # Overlap: carry last sentences whose total length ≤ overlap
            overlap_parts: List[str] = []
            overlap_len = 0
            for part in reversed(current_parts):
                if overlap_len + len(part) <= overlap:
                    overlap_parts.insert(0, part)
                    overlap_len += len(part) + 1
                else:
                    break
            current_parts = overlap_parts
            current_len = overlap_len

        current_parts.append(sentence)
        current_len += sentence_len + 1

    # Flush remaining
    if current_parts:
        chunks.append(" ".join(current_parts))

    return [c for c in chunks if c.strip()]


def _clean_text(text: str) -> str:
    """Normalise whitespace and remove junk characters from OCR output."""
    # Collapse multiple spaces/newlines
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s{2,}', ' ', text)
    # Remove non-printable characters except Bangla Unicode block (U+0980–U+09FF)
    text = re.sub(r'[^\x20-\x7E\u0980-\u09FF।\s]', '', text)
    return text.strip()


def _hard_split(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Fallback: split a single very-long string by character count."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start = end - overlap if end - overlap > start else end
    return chunks
