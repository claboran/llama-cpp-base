#!/usr/bin/env bash
# Download GGUF models from Hugging Face into ./models (resumable, sha256-verified).
# Usage: scripts/download-models.sh [chat|embed|all]   (default: all)
set -euo pipefail

MODELS_DIR="${MODELS_DIR:-$(cd "$(dirname "$0")/.." && pwd)/models}"
HF_BASE="${HF_ENDPOINT:-https://huggingface.co}"
mkdir -p "$MODELS_DIR"

# repo | file
CHAT=( "bartowski/google_gemma-4-26B-A4B-it-GGUF" "google_gemma-4-26B-A4B-it-Q4_K_M.gguf" )
EMBED=( "ggml-org/bge-m3-Q8_0-GGUF" "bge-m3-q8_0.gguf" )

fetch() {
  local repo="$1" file="$2" dst="$MODELS_DIR/$2"
  local sha
  sha=$(curl -sf -m 60 "$HF_BASE/api/models/$repo/tree/main" \
        | python3 -c "import sys,json; print(next((f.get('lfs',{}).get('oid','') for f in json.load(sys.stdin) if f['path']=='$file'),''))")
  if [[ -f "$dst" && -n "$sha" ]] && echo "$sha  $dst" | sha256sum -c --status; then
    echo "[ok]   $file already present and verified"; return
  fi
  echo "[get]  $repo/$file -> $dst"
  curl -fL --retry 10 --retry-all-errors --retry-delay 5 -C - \
       ${HF_TOKEN:+-H "Authorization: Bearer $HF_TOKEN"} \
       -o "$dst.part" "$HF_BASE/$repo/resolve/main/$file"
  if [[ -n "$sha" ]]; then
    echo "$sha  $dst.part" | sha256sum -c --status || { echo "[fail] sha256 mismatch for $file" >&2; exit 1; }
  fi
  mv "$dst.part" "$dst"
  echo "[done] $file"
}

case "${1:-all}" in
  chat)  fetch "${CHAT[@]}" ;;
  embed) fetch "${EMBED[@]}" ;;
  all)   fetch "${EMBED[@]}"; fetch "${CHAT[@]}" ;;
  *) echo "usage: $0 [chat|embed|all]" >&2; exit 2 ;;
esac
