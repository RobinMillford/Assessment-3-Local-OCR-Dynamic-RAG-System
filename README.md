# Assessment 3: Local OCR & Dynamic RAG System

A secure, fully localized document processing pipeline with hybrid RAG search
supporting Bangla (Bengali) and English text.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Browser (frontend/index.html)                              │
│  Upload │ Document List │ Search + Metadata Filters         │
└────────────────────┬────────────────────────────────────────┘
                     │ HTTP
┌────────────────────▼────────────────────────────────────────┐
│  FastAPI Backend (backend/main.py)                          │
│                                                             │
│  POST /upload                                               │
│    → ocr_processor.py  (Surya OCR / PyMuPDF native)        │
│    → chunker.py        (Bangla + English sentence split)    │
│    → embedder.py       (multilingual-MiniLM, 384-dim)       │
│    → vector_store.py   (Qdrant local, with metadata)        │
│                                                             │
│  POST /search                                               │
│    → embedder.py       (embed query)                        │
│    → vector_store.py   (metadata filter + vector search)    │
│    → rag_engine.py     (Ollama qwen2.5:3b)                  │
└─────────────────────────────────────────────────────────────┘
         │ local file               │ local gRPC
┌────────▼────────┐        ┌────────▼────────┐
│  Qdrant (local) │        │  Ollama (local) │
│  qdrant_storage/│        │  qwen2.5:3b     │
└─────────────────┘        └─────────────────┘
```

**All processing is fully local. No data leaves the machine.**

---

## Tech Stack

| Component | Technology | Why |
|---|---|---|
| OCR (primary) | Surya | Transformer-based, best accuracy on Bengali conjunct consonants |
| OCR (fallback) | pytesseract + ben | Lightweight, always available |
| PDF processing | PyMuPDF | Native text layer for digital PDFs (no OCR needed) |
| Embeddings | `paraphrase-multilingual-MiniLM-L12-v2` | 384-dim, supports bn+en in same vector space, ~420 MB |
| Vector store | Qdrant (local mode) | Metadata filtering runs at DB level (not post-filter) |
| LLM | Ollama / qwen2.5:3b | Fully local, 2.3 GB, handles bilingual RAG |
| Backend | FastAPI | REST API + serves frontend HTML |

---

## Setup

### 1. Clone and install

```bash
git clone <your-repo-url>
cd assessment_3
bash setup.sh
```

Or manually:

```bash
uv venv .venv
uv pip install --python .venv/bin/python -r requirements.txt

# Tesseract (fallback OCR) with Bengali support
sudo apt-get install tesseract-ocr tesseract-ocr-ben   # Ubuntu
brew install tesseract && brew install tesseract-lang   # macOS

# Ollama (local LLM)
# setup.sh installs Ollama automatically on Linux, or with Homebrew on macOS.
# Manual Linux install:
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5:3b
```

### 2. Start the server

```bash
source .venv/bin/activate
cd backend
python main.py
```

### 3. Open the UI

```
http://localhost:8000         ← UI
http://localhost:8000/docs    ← Swagger API docs
```

> **Note:** On first upload, Surya models (~2 GB) download automatically.
> Subsequent uploads are fast.

### 4. Cleanup

Preview what would be removed:

```bash
./cleanup.sh --dry-run
```

Remove project files, model caches, the `qwen2.5:3b` Ollama model, and Tesseract packages:

```bash
./cleanup.sh --yes
```

Remove only files inside this project folder:

```bash
./cleanup.sh --yes --project-only
```

Also uninstall the Ollama app/service:

```bash
./cleanup.sh --yes --remove-ollama-app
```

---

## Quick API Reference

```bash
# Upload a document
curl -X POST http://localhost:8000/upload \
  -F "file=@document.pdf" \
  -F "language=auto"

# Search with metadata filters
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "কাগজটির বিষয় কি?", "language": "bn", "doc_type": "pdf"}'

# List all documents
curl http://localhost:8000/documents

# Delete a document
curl -X DELETE http://localhost:8000/documents/{doc_id}
```

---

## Must-Explain Answers

### 1. OCR model choice, trade-offs, and Bangla accuracy

**Primary: Surya OCR**

Surya is a transformer-based OCR engine (vision encoder + text recognition decoder)
specifically designed for multilingual document recognition, including South Asian scripts.

**Why Surya over Tesseract for Bengali:**

| | Surya | Tesseract (ben) |
|---|---|---|
| Architecture | Transformer (visual recognition) | LSTM + HMM (font-based) |
| Bengali conjuncts (যুক্তাক্ষর) | ✅ Handles well | ❌ Often fails |
| Bengali diacritics (হসন্ত, কার) | ✅ Strong | ⚠️ Moderate |
| Low-res scanned docs | ✅ Robust | ❌ Poor |
| Model size | ~2 GB | ~10 MB |
| Inference speed | Slower (GPU recommended) | Fast (CPU) |

**Trade-off accepted:** Larger model size in exchange for significantly better accuracy on
complex Bengali scripts. For a local deployment where accuracy is the priority over
speed, this is the correct choice.

**Fallback strategy:** If Surya fails to load (memory constraints, import error), the
system automatically falls back to pytesseract with `lang='ben+eng'`. This is handled
transparently in `ocr_processor.py`.

**Digital PDFs:** For PDFs with a text layer (non-scanned), PyMuPDF extracts text
natively — no OCR needed. The system detects this per-page: if native extraction yields
>30 characters, OCR is skipped.

---

### 2. Chunking strategy and embedding model selection

**Chunking (`backend/chunker.py`):**

Standard fixed-size character chunking fails on bilingual text because English and
Bengali have different sentence lengths and punctuation patterns.

The chunker uses a **sentence-boundary-aware split** that handles both scripts:
- Bengali sentence boundary: `।` (daari, U+0964)
- English sentence boundary: `.`, `!`, `?`

Algorithm:
1. Split text on `[।.!?]\s+` (positive lookbehind preserves delimiter)
2. Accumulate sentences until `chunk_size=400` characters is reached
3. Carry `overlap=80` characters into the next chunk to preserve cross-boundary context
4. Hard-split single sentences > chunk_size (very long OCR lines)

The overlap is critical for RAG: a question about a concept that spans two sentences
won't be split across non-overlapping chunks.

**Embedding model (`paraphrase-multilingual-MiniLM-L12-v2`):**

| Property | Value |
|---|---|
| Architecture | MiniLM-L12 (12-layer) |
| Vector dimensions | 384 |
| Languages | 50+ including Bengali (bn) and English (en) |
| Model size | ~420 MB |
| Bilingual semantic space | ✅ Bengali and English queries map to nearby vectors |

The key property is **cross-lingual alignment**: a Bengali chunk and a semantically
equivalent English query will have high cosine similarity. This means a user can ask
a question in English about a Bangla document and still retrieve relevant chunks.

---

### 3. System architecture: metadata filtering + vector similarity

**The core question:** How do metadata filters and vector search interact?

**Implementation in Qdrant:**

Qdrant's `Filter` is applied **during** the HNSW graph traversal, not as a post-filter.
This means:

```
User query: "ব্যবসায়িক পরিকল্পনা"
Filters:    language="bn", date_from="2026-01-01"

Qdrant execution:
  1. Embed query → 384-dim vector
  2. Navigate HNSW graph, skipping nodes that don't satisfy the filter
  3. Return top-k by cosine similarity WITHIN the filtered subset

NOT:
  1. Embed query
  2. Get top-1000 by similarity
  3. Post-filter to language="bn"  ← this would degrade recall
```

This is O(log n) over the filtered subset, not O(n) over all documents.

**Filter dimensions available:**

| Filter | Qdrant field | Type |
|---|---|---|
| Language | `language` | MatchValue ("bn", "en", "mixed") |
| Document type | `doc_type` | MatchValue ("pdf", "image") |
| Upload date range | `upload_date` | Range (gte/lte ISO string) |
| Specific document | `doc_id` | MatchValue (UUID string) |

**Example: user searches only Bangla PDFs uploaded after 2026-06-01:**

```python
Filter(must=[
    FieldCondition(key="language",    match=MatchValue(value="bn")),
    FieldCondition(key="doc_type",    match=MatchValue(value="pdf")),
    FieldCondition(key="upload_date", range=Range(gte="2026-06-01")),
])
```

This filter is passed as `query_filter` to `client.search()`, executing as a single
DB operation. The LLM then receives only the filtered, ranked chunks as context.

---

## Project Structure

```
assessment_3/
├── backend/
│   ├── main.py              ← FastAPI app, all endpoints
│   ├── ocr_processor.py     ← Surya + pytesseract OCR pipeline
│   ├── embedder.py          ← multilingual-MiniLM singleton
│   ├── vector_store.py      ← Qdrant local client + metadata ops
│   ├── rag_engine.py        ← search + Ollama answer generation
│   └── chunker.py           ← bilingual sentence-aware chunking
├── frontend/
│   └── index.html           ← Single-page UI (no build step)
├── uploads/                 ← Uploaded files stored here
├── qdrant_storage/          ← Qdrant data (auto-created)
├── requirements.txt
├── setup.sh
└── README.md
```
