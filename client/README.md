# gpu-broker client

Reference Python client for the [gpu-broker](../README.md) HTTP API.

Single runtime dependency: `requests`. Vendor the module (`gpu_broker_client.py`) into
consumer repos, or `pip install -e ../` once the broker repo ships as a package.

---

## Quick start

```python
from client.gpu_broker_client import GpuBrokerClient

broker = GpuBrokerClient(client_id="my-service")
with broker.acquire("rerank") as h:
    # h.endpoint is e.g. "http://127.0.0.1:8082"
    resp = requests.post(f"{h.endpoint}/rerank", json={"query": q, "docs": docs}).json()
# released on exit, even on exception
```

The broker handles eviction, cold-start, `implies` warming, idle-shutdown, and
batch-size pinning. Callers just declare the role and use the returned endpoint.

---

## Role recipes

### `embed` — bge-m3 1024-dim embeddings (OpenAI-compatible `/v1/embeddings`)

```python
broker = GpuBrokerClient(client_id="brain-raglite-ingest")
with broker.acquire("embed") as h:
    for batch in batches:
        r = requests.post(f"{h.endpoint}/v1/embeddings", json={"input": batch}).json()
        vectors = [d["embedding"] for d in r["data"]]
```

For long-running ingest where a stall mid-batch could exceed
`idle_shutdown_seconds: 300`, pass `hold_seconds`:

```python
with broker.acquire("embed", hold_seconds=600) as h:
    ...
```

### `rerank` — bge-reranker-v2-m3 cross-encoder (`/rerank`)

Acquiring `rerank` implicitly warms `embed` (the role declares `implies: [embed]`
in `config/roles.yaml`). One acquire covers a query-and-rerank flow:

```python
broker = GpuBrokerClient(client_id="brain-raglite-mcp")
with broker.acquire("rerank") as h:
    # embed worker is already warm (and listens on its own port — :8081)
    emb = requests.post("http://127.0.0.1:8081/v1/embeddings", json={"input": query}).json()
    # ... pgvector + BM25 candidate retrieval ...
    rerank = requests.post(f"{h.endpoint}/rerank", json={"query": query, "docs": candidates}).json()
```

After acquire, inspect `h.warmed_implies` to confirm the broker also warmed `embed`:

```python
with broker.acquire("rerank") as h:
    assert "embed" in h.warmed_implies or worker_already_loaded
```

### `transcribe` — whisperx large-v3 + diarize (`/transcribe`)

The broker evicts `rerank` automatically (`evict_on_acquire: [rerank]` in
`roles.yaml`) and pins `batch_size=4` server-side via
`default_extra_args: ["--batch_size", "4"]`. The Pascal-compat torch pin
(`torch==2.4.0+cu118`) and the bundled VAD-model SHA are frozen inside the
`whisperx-server:0.1.0` image. Callers do not need to manage any of this.

```python
broker = GpuBrokerClient(client_id="transcribe-skill")
with broker.acquire("transcribe") as h:
    print(f"cold_start_ms={h.cold_start_ms}, evicted={h.evicted}")
    result = requests.post(f"{h.endpoint}/transcribe", json={
        "audio_path": "/path/to/clip.mp3",
        "language": "en",
        "diarize": True,
    }, timeout=600).json()
```

Latency expectations (GTX 1070, batch_size=4):
- Cold whisperx acquire: ~30s + alignment-model bake
- 12-min audio: ~86s

### `llm` (future)

Currently `enabled: false` in `roles.yaml`. Acquiring it raises a 503 with
`reason="role disabled"`. Wired up when the LLM worker lands per
[self-hosted-brain-sovereignty](../../../Mind/projects/pending/self-hosted-brain-sovereignty.md).

---

## Error handling

### `InfeasibleError` (HTTP 503)

The broker rejects an acquire when VRAM cannot be reserved AND no eviction plan
can free enough. Surface `reason` + `would_need_evict` to the user so they can
unblock manually (typically by releasing a pinned role):

```python
from client.gpu_broker_client import GpuBrokerClient, InfeasibleError

try:
    with broker.acquire("transcribe") as h:
        ...
except InfeasibleError as e:
    log.error("broker said no: %s (free=%d MiB, would_evict=%s)",
              e.reason, e.free_mb, e.would_need_evict)
    # e.g. ask the user to release a held handle, or /admin/unpin a pinned role
```

### `BrokerError` (transport, 4xx, 5xx)

Base class for everything broker-side. Catches both connection errors and any
non-2xx that isn't a 503. The acquire context manager always attempts release on
exit — even when the caller's code raises — and downgrades release failures to a
log warning so the original exception propagates.

---

## Read-only / admin

```python
# Full /status payload (broker_version, gpu_used_mb, workers list, pinned_roles, ...)
s = broker.status()

# Just the typed workers list
for w in broker.workers():
    print(f"{w.role:10s} loaded={w.loaded_mb_estimate}MiB active={w.active_handles} pinned={w.pinned}")

# Pin a role eviction-immune (used by the bridge state: embed pinned so brain-rag
# MCP queries hit a warm worker without traversing /acquire).
broker.pin("embed")
broker.unpin("embed")

# True if broker process answers /health with 2xx
if not broker.health():
    raise RuntimeError("gpu-broker not running")
```

---

## CLI

```bash
# From the broker venv (or any venv with requests installed):
python -m client status
python -m client health
python -m client acquire rerank --dwell 5
python -m client pin embed
python -m client --base-url http://fateburn.lan:8090 status
```

Mostly useful for ops sanity checks. Production consumers integrate via the
`GpuBrokerClient` class.

---

## Constructor reference

```python
GpuBrokerClient(
    base_url: str = "http://127.0.0.1:8090",
    client_id: str = "anonymous",       # REQUIRED non-empty; broker logs/metrics use this
    session: requests.Session | None = None,
)
```

Pass `session=` if your consumer already pools connections.

For deployments where the broker is on a different host (e.g. 55places
production hitting `http://fateburn.lan:8090`), expose a `GPU_BROKER_URL` env var:

```python
import os
broker = GpuBrokerClient(
    base_url=os.environ.get("GPU_BROKER_URL", "http://127.0.0.1:8090"),
    client_id="55places-rag",
)
```

---

## API contract source of truth

| Source | Use when |
|---|---|
| `broker/app.py` — pydantic models | Reading the canonical request/response shapes |
| `http://<broker>/openapi.json` | Generating clients in other languages |
| `config/roles.yaml` | Looking up which roles exist + their eviction policy + idle timers |
