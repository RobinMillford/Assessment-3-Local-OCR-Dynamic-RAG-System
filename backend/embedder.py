"""
embedder.py — Singleton embedding model wrapper

Model: paraphrase-multilingual-MiniLM-L12-v2
  - 384 dimensions
  - ~420 MB download on first run
  - Natively supports Bengali (bn) + English (en) in the same vector space
  - Runs fully locally on CPU (slow but works) or GPU
"""

import numpy as np
from typing import List, Union
import logging

logger = logging.getLogger(__name__)

_model = None
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
VECTOR_DIM = 384


def get_model():
    """Lazy-load the embedding model (loaded once, reused across requests)."""
    global _model
    if _model is None:
        logger.info(f"Loading embedding model: {MODEL_NAME}")
        logger.info("This takes ~30s on first run (model download + load)...")
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME)
        logger.info("Embedding model loaded.")
    return _model


def embed(texts: Union[str, List[str]]) -> np.ndarray:
    """
    Embed one or more texts.

    Args:
        texts: A single string or list of strings (Bangla, English, or mixed)

    Returns:
        numpy array of shape (n_texts, 384)
    """
    if isinstance(texts, str):
        texts = [texts]

    model = get_model()
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True,   # cosine similarity ≡ dot product
    )
    return embeddings.astype(np.float32)
