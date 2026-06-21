#!/usr/bin/env bash
# =============================================================================
# setup.sh — One-command setup for Assessment 3: Local OCR + RAG System
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
OLLAMA_MODEL="${OLLAMA_MODEL:-qwen2.5:0.5b}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$SCRIPT_DIR/.uv-cache}"

install_ollama() {
    echo "⚠️  Ollama not found. Installing Ollama..."

    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if ! command -v curl &> /dev/null; then
            if command -v apt-get &> /dev/null; then
                echo "📦 Installing curl for Ollama installer..."
                sudo apt-get update -q
                sudo apt-get install -y curl
            else
                echo "❌ curl is required to install Ollama automatically."
                echo "   Install curl, then rerun this script."
                return 1
            fi
        fi

        local installer
        installer="$(mktemp)"
        curl -fsSL https://ollama.com/install.sh -o "$installer"
        sh "$installer"
        rm -f "$installer"
    elif [[ "$OSTYPE" == "darwin"* ]] && command -v brew &> /dev/null; then
        brew install ollama
    else
        echo "❌ Could not auto-install Ollama on this OS."
        echo "   Install manually from: https://ollama.com/download"
        return 1
    fi
}

start_ollama() {
    if ! command -v ollama &> /dev/null; then
        return 1
    fi

    if command -v curl &> /dev/null && curl -fsS http://127.0.0.1:11434/api/tags &> /dev/null; then
        return 0
    fi

    if command -v systemctl &> /dev/null && systemctl list-unit-files ollama.service &> /dev/null; then
        sudo systemctl enable --now ollama || true
        sleep 2
    elif command -v brew &> /dev/null; then
        brew services start ollama || true
        sleep 2
    fi

    if command -v curl &> /dev/null && curl -fsS http://127.0.0.1:11434/api/tags &> /dev/null; then
        return 0
    fi

    if command -v pgrep &> /dev/null && pgrep -x ollama &> /dev/null; then
        return 0
    fi

    echo "🚀 Starting Ollama server in the background..."
    nohup ollama serve > "$SCRIPT_DIR/ollama.log" 2>&1 &
    sleep 3
}

pull_ollama_model() {
    echo "✅ Ollama found. Pulling $OLLAMA_MODEL model (~400 MB for qwen2.5:0.5b)..."
    if ! ollama pull "$OLLAMA_MODEL"; then
        echo "⚠️  Could not pull $OLLAMA_MODEL."
        echo "   Try manually after starting Ollama:"
        echo "     ollama serve"
        echo "     ollama pull $OLLAMA_MODEL"
        return 1
    fi
    echo "✅ Ollama model ready"
}

echo "============================================================"
echo "  Assessment 3: Local OCR + RAG System — Setup"
echo "============================================================"

# ── Python virtual environment + dependencies ─────────────────────────────
echo ""
echo "📦 Creating local Python environment with uv..."
if ! command -v uv &> /dev/null; then
    echo "❌ uv is not installed."
    echo "   Install it first: https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
fi

cd "$SCRIPT_DIR"
uv venv "$VENV_DIR"

echo ""
echo "📦 Installing Python dependencies with uv..."
uv pip install --python "$VENV_DIR/bin/python" -r "$SCRIPT_DIR/requirements.txt"
echo "✅ Python environment ready at .venv"

# ── Tesseract (fallback OCR) ──────────────────────────────────────────────
echo ""
echo "🔤 Installing Tesseract + Bengali language pack (fallback OCR)..."
if command -v apt-get &> /dev/null; then
    sudo apt-get update -q
    sudo apt-get install -y tesseract-ocr tesseract-ocr-ben
    echo "✅ Tesseract installed with Bengali support"
elif command -v brew &> /dev/null; then
    brew install tesseract
    brew install tesseract-lang
    echo "✅ Tesseract installed (macOS)"
else
    echo "⚠️  Could not auto-install Tesseract. Install manually:"
    echo "   Ubuntu: sudo apt-get install tesseract-ocr tesseract-ocr-ben"
    echo "   macOS:  brew install tesseract && brew install tesseract-lang"
fi

# ── Ollama ────────────────────────────────────────────────────────────────
echo ""
echo "🤖 Checking Ollama (local LLM)..."
if ! command -v ollama &> /dev/null; then
    install_ollama
fi

start_ollama
pull_ollama_model

# ── Create directories ────────────────────────────────────────────────────
echo ""
echo "📁 Creating directories..."
mkdir -p "$SCRIPT_DIR/uploads" "$SCRIPT_DIR/qdrant_storage"
echo "✅ Directories created"

# ── Done ─────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "✅  Setup complete!"
echo ""
echo "  To start the server:"
echo "    source .venv/bin/activate"
echo "    cd backend && python main.py"
echo ""
echo "  Or without activating:"
echo "    .venv/bin/python backend/main.py"
echo ""
echo "  Then open: http://localhost:8000"
echo "  API docs : http://localhost:8000/docs"
echo ""
echo "  Note: Surya OCR models (~2GB) download automatically"
echo "  on the first document upload."
echo "============================================================"
