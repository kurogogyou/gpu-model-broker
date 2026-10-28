# encoder-cpu

RAGLite's CPU fallback: bge-m3 embeddings and bge-reranker-v2-m3 reranking on a host with no GPU (MIKASA), in the same wire formats as `tei-embed` (`POST /v1/embeddings`, OpenAI-compatible) and `bge-reranker` (`POST /rerank`, broker dialect). A client switches between the GPU and CPU tiers by base URL alone. Models load on first request and unload after `ENCODER_IDLE_UNLOAD_S` (default 30 min).

Not a broker role: the broker's ledger is GPU-only, and a fallback must work when the broker is unreachable. Why, the routing ladder and the benchmark: [`../../docs/host-routing.md`](../../docs/host-routing.md).

## Run (MIKASA)

```bash
python3 -m venv /opt/fast/venvs/encoder-cpu
/opt/fast/venvs/encoder-cpu/bin/pip install -r requirements.txt -c constraints.txt
cp ../../systemd/encoder-cpu.service ~/.config/systemd/user/
systemctl --user daemon-reload && systemctl --user enable --now encoder-cpu
```

`ENCODER_MODEL_DIR` holds `embed/` and `rerank/`: for the torch runtime, links to the pinned snapshots (`/opt/fast/models/encoder-cpu/snapshots/`); for OpenVINO/ORT, an `export_variants.py` output.

## Re-benchmark

`export_variants.py` builds the OpenVINO and ONNX Runtime variants (fp32, int8 weights-only, int8 calibrated, ORT int8); `bench_variants.sh` runs each through the real server and gates it with `encoder_equivalence.py` (brain vault `Mind/tools/host/`) against reference bundles dumped from a GPU broker. Run both before changing the runtime or the CPU.
