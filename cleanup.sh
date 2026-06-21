#!/usr/bin/env bash
# =============================================================================
# cleanup.sh - Remove Assessment 3 local artifacts and optional system deps
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OLLAMA_MODEL="${OLLAMA_MODEL:-qwen2.5:0.5b}"

DRY_RUN=false
ASSUME_YES=false
PROJECT_ONLY=false
REMOVE_OLLAMA_APP=false
REMOVE_ALL_OLLAMA_MODELS=false

usage() {
    cat <<EOF
Usage: ./cleanup.sh [OPTIONS]

Removes files and tools downloaded/created for this project.

Options:
  --dry-run                  Show what would be removed, but do nothing
  -y, --yes                  Do not ask for confirmation
  --project-only             Remove only files inside this project folder
  --remove-ollama-app        Also uninstall the Ollama application/service
  --all-ollama-models        Remove every local Ollama model, not only $OLLAMA_MODEL
  -h, --help                 Show this help

Examples:
  ./cleanup.sh --dry-run
  ./cleanup.sh --yes
  ./cleanup.sh --yes --remove-ollama-app
EOF
}

log() {
    echo "$*"
}

run_cmd() {
    if "$DRY_RUN"; then
        printf '[dry-run] '
        printf '%q ' "$@"
        printf '\n'
    else
        "$@"
    fi
}

remove_path() {
    local path="$1"
    if [[ -e "$path" || -L "$path" ]]; then
        run_cmd rm -rf "$path"
    else
        log "skip missing: $path"
    fi
}

confirm() {
    if "$ASSUME_YES" || "$DRY_RUN"; then
        return 0
    fi

    echo ""
    echo "This will remove project data, caches, OCR/LLM model downloads, and system packages used by this project."
    echo "Use --project-only if you only want to clean this folder."
    read -r -p "Continue? Type 'yes' to clean: " answer
    [[ "$answer" == "yes" ]]
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --dry-run)
                DRY_RUN=true
                ;;
            -y|--yes)
                ASSUME_YES=true
                ;;
            --project-only)
                PROJECT_ONLY=true
                ;;
            --remove-ollama-app)
                REMOVE_OLLAMA_APP=true
                ;;
            --all-ollama-models)
                REMOVE_ALL_OLLAMA_MODELS=true
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                echo "Unknown option: $1" >&2
                usage
                exit 2
                ;;
        esac
        shift
    done
}

clean_project_files() {
    log ""
    log "== Cleaning project files =="

    remove_path "$SCRIPT_DIR/.venv"
    remove_path "$SCRIPT_DIR/.uv-cache"
    remove_path "$SCRIPT_DIR/uploads"
    remove_path "$SCRIPT_DIR/qdrant_storage"
    remove_path "$SCRIPT_DIR/backend/uploads"
    remove_path "$SCRIPT_DIR/backend/qdrant_storage"
    remove_path "$SCRIPT_DIR/ollama.log"
    remove_path "$SCRIPT_DIR/.pytest_cache"
    remove_path "$SCRIPT_DIR/.mypy_cache"
    remove_path "$SCRIPT_DIR/.ruff_cache"

    while IFS= read -r path; do
        remove_path "$path"
    done < <(
        find "$SCRIPT_DIR" \
            \( -path "$SCRIPT_DIR/.venv" \
            -o -path "$SCRIPT_DIR/.uv-cache" \
            -o -path "$SCRIPT_DIR/uploads" \
            -o -path "$SCRIPT_DIR/qdrant_storage" \
            -o -path "$SCRIPT_DIR/backend/uploads" \
            -o -path "$SCRIPT_DIR/backend/qdrant_storage" \) -prune \
            -o -type d -name "__pycache__" -print 2> /dev/null
    )
}

clean_python_model_caches() {
    log ""
    log "== Cleaning Python/model caches for this project =="

    remove_path "$HOME/.cache/torch/sentence_transformers/sentence-transformers_paraphrase-multilingual-MiniLM-L12-v2"
    remove_path "$HOME/.cache/sentence_transformers/sentence-transformers_paraphrase-multilingual-MiniLM-L12-v2"

    local hf_hub="$HOME/.cache/huggingface/hub"
    if [[ -d "$hf_hub" ]]; then
        while IFS= read -r path; do
            remove_path "$path"
        done < <(
            find "$hf_hub" -maxdepth 1 -type d \
                \( -name "models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2" \
                -o -iname "*surya*" \) 2> /dev/null
        )
    else
        log "skip missing: $hf_hub"
    fi
}

clean_ollama_models() {
    log ""
    log "== Cleaning Ollama model downloads =="

    if ! command -v ollama &> /dev/null; then
        log "skip: ollama command not found"
        if "$REMOVE_ALL_OLLAMA_MODELS"; then
            remove_path "$HOME/.ollama/models"
            run_cmd sudo rm -rf /usr/share/ollama/.ollama/models
        fi
        return 0
    fi

    if "$REMOVE_ALL_OLLAMA_MODELS"; then
        local models
        models="$(ollama list 2> /dev/null | awk 'NR > 1 {print $1}' || true)"
        if [[ -z "$models" ]]; then
            log "skip: no Ollama models found"
        else
            while IFS= read -r model; do
                [[ -n "$model" ]] && run_cmd ollama rm "$model"
            done <<< "$models"
        fi
        remove_path "$HOME/.ollama/models"
        run_cmd sudo rm -rf /usr/share/ollama/.ollama/models
    else
        run_cmd ollama rm "$OLLAMA_MODEL" || true
    fi
}

remove_tesseract() {
    log ""
    log "== Removing Tesseract packages =="

    if command -v apt-get &> /dev/null; then
        run_cmd sudo apt-get remove -y tesseract-ocr tesseract-ocr-ben || true
        run_cmd sudo apt-get autoremove -y || true
    elif command -v brew &> /dev/null; then
        run_cmd brew uninstall tesseract tesseract-lang || true
    else
        log "skip: no supported package manager found for Tesseract removal"
    fi
}

remove_ollama_app() {
    log ""
    log "== Removing Ollama application/service =="

    if command -v systemctl &> /dev/null; then
        run_cmd sudo systemctl stop ollama || true
        run_cmd sudo systemctl disable ollama || true
        run_cmd sudo rm -f /etc/systemd/system/ollama.service
        run_cmd sudo systemctl daemon-reload || true
    fi

    if command -v brew &> /dev/null; then
        run_cmd brew services stop ollama || true
        run_cmd brew uninstall ollama || true
    fi

    run_cmd sudo rm -f /usr/local/bin/ollama
    run_cmd sudo rm -rf /usr/share/ollama
    run_cmd sudo rm -rf /var/lib/ollama
    remove_path "$HOME/.ollama"
}

main() {
    parse_args "$@"

    echo "============================================================"
    echo "  Assessment 3: Full Cleanup"
    echo "============================================================"

    if ! confirm; then
        echo "Cleanup cancelled."
        exit 1
    fi

    clean_project_files

    if ! "$PROJECT_ONLY"; then
        clean_python_model_caches
        clean_ollama_models
        remove_tesseract

        if "$REMOVE_OLLAMA_APP"; then
            remove_ollama_app
        else
            log ""
            log "Keeping Ollama app installed. Use --remove-ollama-app to uninstall it too."
        fi
    fi

    echo ""
    echo "============================================================"
    echo "Cleanup complete."
    echo "============================================================"
}

main "$@"
