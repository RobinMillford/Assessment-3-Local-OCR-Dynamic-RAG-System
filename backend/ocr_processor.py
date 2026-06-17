"""
ocr_processor.py — Local OCR pipeline

Primary  : Surya (best Bangla accuracy, specifically handles complex Bengali scripts)
Fallback : pytesseract with 'ben+eng' language pack

Why Surya over Tesseract for Bangla?
- Surya uses a transformer-based recognition model trained on South Asian scripts
- Tesseract's Bengali (ben) model struggles with conjunct consonants (যুক্তাক্ষর)
  and diacritics (মাত্রা) in handwritten or low-resolution scanned text
- Surya processes the full line visually without needing a font model

Trade-off: Surya requires ~2GB of model weights vs Tesseract's ~10MB Bengali data.
For a secure local pipeline, this is acceptable.
"""

import logging
import os
from pathlib import Path
from typing import List, Tuple
from PIL import Image
import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Surya model loader (lazy, loaded once per process)
# ---------------------------------------------------------------------------
_surya_models = None
SURYA_AVAILABLE = False


def _try_load_surya():
    global _surya_models, SURYA_AVAILABLE
    if _surya_models is not None:
        return _surya_models

    try:
        from surya.model.detection.model import load_model as load_det_model
        from surya.model.detection.processor import load_processor as load_det_processor
        from surya.model.recognition.model import load_model as load_rec_model
        from surya.model.recognition.processor import load_processor as load_rec_processor

        logger.info("Loading Surya OCR models (first-time download may take a few minutes)...")
        det_model      = load_det_model()
        det_processor  = load_det_processor()
        rec_model      = load_rec_model()
        rec_processor  = load_rec_processor()

        _surya_models = (det_model, det_processor, rec_model, rec_processor)
        SURYA_AVAILABLE = True
        logger.info("Surya OCR models loaded successfully.")
        return _surya_models

    except Exception as e:
        logger.warning(f"Surya unavailable ({e}). Will fallback to pytesseract.")
        SURYA_AVAILABLE = False
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_text_from_file(file_path: str) -> Tuple[List[str], str]:
    """
    Extract text from a PDF or image file locally.

    Returns:
        (page_texts, ocr_engine_used)
        page_texts: list of strings, one per page/image
        ocr_engine_used: "pymupdf_native" | "surya" | "tesseract"
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return _process_pdf(file_path)
    elif suffix in {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"}:
        text, engine = _ocr_image(Image.open(file_path))
        return [text], engine
    else:
        raise ValueError(f"Unsupported file type: {suffix}")


def _process_pdf(pdf_path: str) -> Tuple[List[str], str]:
    """
    Process a PDF file page by page.

    Strategy:
    1. Try PyMuPDF native text extraction (works for digital/searchable PDFs instantly)
    2. If a page is blank (scanned), rasterise it and run OCR
    """
    doc = fitz.open(pdf_path)
    page_texts = []
    engines_used = set()

    logger.info(f"Processing PDF: {pdf_path} ({len(doc)} pages)")

    for page_num, page in enumerate(doc):
        native_text = page.get_text().strip()

        if native_text and len(native_text) > 30:
            # Digital PDF — text layer present, no OCR needed
            page_texts.append(native_text)
            engines_used.add("pymupdf_native")
            logger.info(f"  Page {page_num + 1}: native text extraction ({len(native_text)} chars)")
        else:
            # Scanned page — rasterise at 300 DPI and OCR
            logger.info(f"  Page {page_num + 1}: blank native text, running OCR...")
            pix = page.get_pixmap(dpi=300)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            text, engine = _ocr_image(img)
            page_texts.append(text)
            engines_used.add(engine)
            logger.info(f"  Page {page_num + 1}: OCR via {engine} ({len(text)} chars)")

    doc.close()
    engine_label = "+".join(sorted(engines_used)) or "unknown"
    return page_texts, engine_label


def _ocr_image(image: Image.Image) -> Tuple[str, str]:
    """
    Run OCR on a single PIL Image.
    Tries Surya first, falls back to pytesseract.
    """
    # Try Surya
    models = _try_load_surya()
    if models is not None:
        try:
            text = _run_surya(image, models)
            return text, "surya"
        except Exception as e:
            logger.warning(f"Surya OCR failed ({e}), trying tesseract...")

    # Fallback: pytesseract
    return _run_tesseract(image), "tesseract"


def _run_surya(image: Image.Image, models) -> str:
    """
    Run Surya OCR on a single image.
    Langs: Bengali (bn) + English (en)
    """
    from surya.ocr import run_ocr

    det_model, det_processor, rec_model, rec_processor = models

    # Surya expects a list of images and a list of language lists
    predictions = run_ocr(
        [image],
        [["bn", "en"]],
        det_model,
        det_processor,
        rec_model,
        rec_processor,
    )

    # Concatenate all recognised text lines
    lines = []
    if predictions and predictions[0].text_lines:
        for line in predictions[0].text_lines:
            if line.text and line.text.strip():
                lines.append(line.text.strip())

    return "\n".join(lines)


def _run_tesseract(image: Image.Image) -> str:
    """
    Fallback OCR via pytesseract.
    Requires: sudo apt-get install tesseract-ocr tesseract-ocr-ben
    """
    try:
        import pytesseract
        # ben = Bengali, eng = English
        text = pytesseract.image_to_string(image, lang="ben+eng")
        return text.strip()
    except Exception as e:
        logger.error(f"pytesseract also failed: {e}")
        return ""
