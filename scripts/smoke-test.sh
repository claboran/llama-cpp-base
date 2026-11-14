#!/usr/bin/env bash
# Test drive for the llama.cpp images, plain `docker run` (no compose).
# Usage: scripts/smoke-test.sh [IMAGE_TAG]    e.g. v1.0.0-cuda13.0.3
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TAG="${1:-v1.0.0-cuda13.0.3}"
BASE="llama-cpp-base:$TAG"
SERVER="llama-cpp-server:$TAG"
MODELS="${MODELS_DIR:-$ROOT/models}"
CHAT_MODEL="${CHAT_MODEL:-google_gemma-4-26B-A4B-it-Q4_K_M.gguf}"
EMBED_MODEL="${EMBED_MODEL:-bge-m3-q8_0.gguf}"
CHAT_CTX="${CHAT_CTX:-65536}"
PASS=0; FAIL=0

step() { printf '\n=== %s\n' "$*"; }
ok()   { echo "PASS: $*"; PASS=$((PASS+1)); }
bad()  { echo "FAIL: $*"; FAIL=$((FAIL+1)); }
run()  { docker run --rm --gpus all -v "$MODELS:/models:ro" "$@"; }

wait_health() {  # container port
  for _ in $(seq 1 180); do
    curl -fsS "http://127.0.0.1:$2/health" >/dev/null 2>&1 && return 0
    docker inspect -f '{{.State.Running}}' "$1" 2>/dev/null | grep -q true || return 1
    sleep 2
  done; return 1
}

step "1. GPU visible in container"
if run "$BASE" nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader; then ok nvidia-smi; else bad nvidia-smi; fi

step "2. build info + version"
run "$BASE" cat /opt/llama/BUILD_INFO
if run "$BASE" llama-cli --version; then ok version; else bad version; fi

step "3. CUDA device listed by llama.cpp"
devs=$(run "$BASE" llama-server --list-devices 2>&1); echo "$devs"
if grep -q CUDA0 <<<"$devs"; then ok list-devices; else bad "no CUDA0 device"; fi

if [[ -f "$MODELS/$CHAT_MODEL" ]]; then
  step "4. llama-bench ($CHAT_MODEL)"
  if run "$BASE" llama-bench -m "/models/$CHAT_MODEL" -ngl 999 -fa 1 -p 512 -n 128 -d 0,16384; then ok bench; else bad bench; fi

  step "5. chat server ($CHAT_MODEL, ctx $CHAT_CTX)"
  docker rm -f smoke-chat >/dev/null 2>&1
  docker run -d --name smoke-chat --gpus all -v "$MODELS:/models:ro" -p 127.0.0.1:18080:8080 "$SERVER" \
    -m "/models/$CHAT_MODEL" -c "$CHAT_CTX" -ngl 999 -fa on --jinja --alias gemma >/dev/null
  if wait_health smoke-chat 18080; then
    resp=$(curl -fsS http://127.0.0.1:18080/v1/chat/completions -H 'Content-Type: application/json' \
      -d '{"model":"gemma","messages":[{"role":"user","content":"Reply with exactly: the capital of France is Paris."}],"max_tokens":1024,"temperature":0}')
    echo "$resp" | python3 -c 'import sys,json; r=json.load(sys.stdin); m=r["choices"][0]["message"]; print("answer:", m["content"]); print("reasoning chars:", len(m.get("reasoning_content") or "")); print("timings:", {k:round(v,1) for k,v in r.get("timings",{}).items() if "per_second" in k})'
    # Gemma 4 thinks by default: check the final answer, not the reasoning text
    if python3 -c 'import sys,json; sys.exit(0 if "paris" in json.load(sys.stdin)["choices"][0]["message"]["content"].lower() else 1)' <<<"$resp"; then ok chat; else bad "chat answer"; fi
  else bad "chat server not healthy"; docker logs --tail 50 smoke-chat; fi
  docker rm -f smoke-chat >/dev/null
else
  echo "skip chat tests: $MODELS/$CHAT_MODEL missing"
fi

if [[ -f "$MODELS/$EMBED_MODEL" ]]; then
  step "6. embedding server ($EMBED_MODEL)"
  docker rm -f smoke-embed >/dev/null 2>&1
  docker run -d --name smoke-embed --gpus all -v "$MODELS:/models:ro" -p 127.0.0.1:18081:8080 "$SERVER" \
    -m "/models/$EMBED_MODEL" --embeddings --pooling cls -c 8192 -b 8192 -ub 8192 -ngl 999 --alias bge-m3 >/dev/null
  if wait_health smoke-embed 18081; then
    resp=$(curl -fsS http://127.0.0.1:18081/v1/embeddings -H 'Content-Type: application/json' \
      -d '{"model":"bge-m3","input":["The cat sits on the mat.","Eine Katze sitzt auf der Matte.","Quarterly revenue grew by 12%."]}')
    if echo "$resp" | python3 -c '
import sys, json, math
d = [e["embedding"] for e in json.load(sys.stdin)["data"]]
norm = [math.sqrt(sum(x*x for x in v)) for v in d]
cos = lambda a, b: sum(x*y for x, y in zip(a, b)) / (math.sqrt(sum(x*x for x in a)) * math.sqrt(sum(x*x for x in b)))
print("dims:", [len(v) for v in d], "norms:", [round(n, 3) for n in norm])
print("cos(en,de)=%.3f  cos(en,finance)=%.3f" % (cos(d[0], d[1]), cos(d[0], d[2])))
assert all(len(v) == 1024 for v in d)
assert cos(d[0], d[1]) > cos(d[0], d[2])
'; then ok embeddings; else bad embeddings; fi
  else bad "embed server not healthy"; docker logs --tail 50 smoke-embed; fi
  docker rm -f smoke-embed >/dev/null
else
  echo "skip embedding tests: $MODELS/$EMBED_MODEL missing"
fi

printf '\n=== result: %d passed, %d failed\n' "$PASS" "$FAIL"
[[ $FAIL -eq 0 ]]
