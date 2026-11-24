#!/usr/bin/env bash
# Checks the monitoring stack (compose.monitoring.yaml): Prometheus and Grafana are up, every
# running llama service is scraped, the llamacpp:* series the dashboard uses exist, and the
# dashboard is provisioned. Run after `make up-monitoring` and at least one llama service.
# Usage: scripts/monitoring-check.sh            (reads ports/credentials from .env if present)
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
[[ -f "$ROOT/.env" ]] && set -a && . "$ROOT/.env" && set +a
PROM="http://127.0.0.1:${PROMETHEUS_PORT:-9090}"
GRAFANA="http://127.0.0.1:${GRAFANA_PORT:-3000}"
GF_AUTH="${GRAFANA_ADMIN_USER:-admin}:${GRAFANA_ADMIN_PASSWORD:-admin}"
PASS=0; FAIL=0

ok()  { echo "PASS: $*"; PASS=$((PASS+1)); }
bad() { echo "FAIL: $*"; FAIL=$((FAIL+1)); }
query() {  # PromQL -> JSON result list
  curl -fsS -G "$PROM/api/v1/query" --data-urlencode "query=$1" | python3 -c 'import sys,json; print(json.dumps(json.load(sys.stdin)["data"]["result"]))'
}

if curl -fsS "$PROM/-/ready" >/dev/null 2>&1; then ok "prometheus ready ($PROM)"; else bad "prometheus not reachable at $PROM"; fi
# Grafana needs ~30 s for database migrations on its first start
for _ in $(seq 1 30); do curl -fsS "$GRAFANA/api/health" >/dev/null 2>&1 && break; sleep 2; done
if curl -fsS "$GRAFANA/api/health" >/dev/null 2>&1; then ok "grafana healthy ($GRAFANA)"; else bad "grafana not reachable at $GRAFANA"; fi

# Scrape targets: a running service must be up; a service that is not running is only reported.
targets=$(curl -fsS "$PROM/api/v1/targets?scrapePool=llama" 2>/dev/null | python3 -c '
import sys, json
for t in json.load(sys.stdin)["data"]["activeTargets"]:
    print(t["labels"]["service"], t["health"], t.get("lastError", ""))')
sed 's/^/  target: /' <<<"$targets"
for svc in chat embed qwen; do
  running=$(docker ps --filter "label=com.docker.compose.service=llm-$svc" -q)
  if grep -q "^$svc up" <<<"$targets"; then
    ok "llm-$svc scraped"
    n=$(query "count({__name__=~\"llamacpp:.+\",service=\"$svc\"})" | python3 -c 'import sys,json; r=json.load(sys.stdin); print(r[0]["value"][1] if r else 0)')
    if [[ "$n" -ge 15 ]]; then ok "llm-$svc exports $n llamacpp:* series"; else bad "llm-$svc exports only $n llamacpp:* series (metric names changed? check the dashboard)"; fi
  elif [[ -n "$running" ]]; then
    bad "llm-$svc is running but not scraped (401 = METRICS_API_KEY wrong; network llm-metrics missing = recreate the service)"
  else
    echo "skip: llm-$svc not running"
  fi
done

if curl -fsS -u "$GF_AUTH" "$GRAFANA/api/dashboards/uid/llama-cpp" >/dev/null 2>&1; then
  ok "grafana dashboard 'llama.cpp' provisioned"
else bad "grafana dashboard 'llama-cpp' missing (or GRAFANA_ADMIN_* wrong)"; fi

printf '\n=== result: %d passed, %d failed\n' "$PASS" "$FAIL"
[[ $FAIL -eq 0 ]]
