#!/usr/bin/env bash
# Benchmark + equivalence-gate each CPU runtime variant of encoder-cpu.
#   bench_variants.sh <models_dir> <ref-embed.json> <ref-rerank.json> <equivalence.py> [variant...]
# Variants: torch-fp32 ov-fp32 ov-int8w ov-int8 ort-int8 (default: all present).
# Each runs the real server on 127.0.0.1:8091, so the numbers are the served path.
set -uo pipefail
M=$1 REFE=$2 REFR=$3 EQ=$4; shift 4
PY=/opt/fast/venvs/encoder-cpu/bin/python
HERE="$(cd "$(dirname "$0")" && pwd)"
variants=("$@"); [ ${#variants[@]} -gt 0 ] || variants=(torch-fp32 ov-fp32 ov-int8w ov-int8 ort-int8)

for v in "${variants[@]}"; do
    [ -d "$M/$v" ] || { echo "skip $v (not exported)"; continue; }
    case $v in torch-*) rt=torch ;; ov-*) rt=ov ;; ort-*) rt=ort ;; esac
    ENCODER_RUNTIME=$rt ENCODER_MODEL_DIR=$M/$v ENCODER_IDLE_UNLOAD_S=0 \
        $PY -m uvicorn --app-dir "$HERE" server:app --host 127.0.0.1 --port 8091 --log-level warning &
    pid=$!
    for _ in $(seq 1 60); do $PY -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8091/healthz')" 2>/dev/null && break; sleep 1; done
    echo "===== $v ($rt)"
    $PY - "$REFE" <<'PY'
import json, statistics, sys, time, urllib.request, resource, os
B = "http://127.0.0.1:8091"
def post(p, b):
    r = urllib.request.Request(B + p, json.dumps(b).encode(), {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=600).read())
texts = json.load(open(sys.argv[1]))["texts"]
q = texts[3].split(".")[0][:180]
t = time.monotonic(); post("/v1/embeddings", {"input": "warm"}); le = time.monotonic() - t
t = time.monotonic(); post("/rerank", {"query": "warm", "docs": ["a"]}); lr = time.monotonic() - t
def med(f, n):
    xs = []
    for _ in range(n):
        t = time.monotonic(); f(); xs.append(time.monotonic() - t)
    return statistics.median(xs)
qe = med(lambda: post("/v1/embeddings", {"input": q}), 7)
r20 = med(lambda: post("/rerank", {"query": q, "docs": texts[10:30]}), 3)
r50 = med(lambda: post("/rerank", {"query": q, "docs": texts[10:60]}), 2)
t = time.monotonic(); post("/v1/embeddings", {"input": texts[:64]}); bulk = 64 / (time.monotonic() - t)
print(json.dumps({"load_embed_s": round(le, 1), "load_rerank_s": round(lr, 1),
                  "query_embed_ms": round(qe * 1000, 1), "rerank_n20_s": round(r20, 2),
                  "rerank_n50_s": round(r50, 2), "bulk_chunks_per_s": round(bulk, 1),
                  "search_n20_s_est": round(qe + r20, 2)}))
PY
    echo "server RSS MiB: $(( $(ps -o rss= -p $pid) / 1024 ))"
    $PY "$EQ" --reference-file "$REFE" --candidate http://127.0.0.1:8091 2>&1 | grep -E 'PASS|FAIL|min |kept|top-1'
    $PY "$EQ" --mode rerank --rerank-dialect broker --reference-file "$REFR" --candidate http://127.0.0.1:8091 2>&1 | grep -E 'PASS|FAIL|top-1|top-5|pairwise'
    kill $pid; wait $pid 2>/dev/null
done
