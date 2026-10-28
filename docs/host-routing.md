# Host routing: which machine serves which encoder work

Written 2026-10-28; revise at the Triton adoption spike. One place for the routing ladder, the measurements behind it and where each piece of logic lives, so a later redesign can see why it is shaped this way.

## The ladder

| Tier | Host | Serves | When | Selected by |
|---|---|---|---|---|
| 1 | **Scully**, RTX 3090 (5060 Ti joining for Strata-style engines) | embed, rerank, transcribe, LLM roles | Scully is on | `gpu-broker-tunnel@scully` (MIKASA's default) |
| 2 | **SASHA**, RTX 3060 12 GB, off between jobs | embed, rerank, transcribe | Batch jobs while Scully is off | `gpu_job.sh` wakes it and switches to `gpu-broker-tunnel@sasha` |
| 3 | **MIKASA CPU**, i7-10700 (AVX2, no VNNI) | embed (queries), rerank (reduced candidate pool) | No broker answers: interactive search only, never ingest | RAGLite MCP server per search, on a failed broker `/health` |

Interactive search never wakes a host: a wake costs ~35 s before the broker answers plus a cold model load (SASHA: embed ~23 s, rerank ~19 s), longer than the CPU tier. Batch work never runs on the CPU tier: bulk embedding there is ~3 chunks/s against ~130 (3060) and ~320 (3090).

Scully now powers off and wakes on command like SASHA (remote `poweroff` via a sudoers rule scoped to that one command; WoL to SSH ~40 s), so "Scully is on" is a choice. With the planned dual-GPU Scully the routing can go by job weight: wake Scully for heavy or LLM work, SASHA for light batch work, and keep both off otherwise. That decision waits for the wall-meter comparison.

## Measurements (2026-10-28)

Same corpus and queries on every host (an `encoder_equivalence.py` bundle of 200 vault chunks).

| | Scully 3090 (280 W cap) | SASHA 3060 | MIKASA CPU |
|---|---|---|---|
| embed bs=1×160 tokens | 10.3 ms | 27.5 ms | see § CPU tier |
| embed bs=96×160 | 221 ms | 548 ms | |
| rerank N=20, server time | 47.5 ms | 121 ms | |
| Equivalence vs the 3090 | reference | min cosine 0.999997, top-10 30/30, rerank identical | |
| Idle, models resident (GPU board + CPU package, + DRAM on SASHA) | 36 W | 18 W | |
| Energy, embed | 971 J / 1000 chunks | 1,252 J | |
| Energy, rerank N=50 | 74 J / call | 98 J | |

Per unit of work Scully is ~25 % cheaper and 2.4× faster; SASHA wins only on idle hours. These are lower bounds without motherboard, fans and PSU loss; a wall-meter pass is pending and decides the always-on question.

## CPU tier

`containers/encoder-cpu/` on MIKASA (i7-10700, 8 threads), `encoder-cpu.service` on `127.0.0.1:8091`, **torch fp32** from the pinned snapshots (bge-m3 `5617a9f`, bge-reranker-v2-m3 `953dc6f`). Benchmarked 2026-10-28 against the 3090's fp16 outputs on the same 200-chunk bundle (`bench_variants.sh`):

| Variant | query embed | rerank N=20 | rerank N=50 | RSS | embed: min cosine / top-10 kept | rerank top-5 |
|---|---|---|---|---|---|---|
| **torch fp32 (chosen)** | 82 ms | **6.4 s** | 16.1 s | 3.7 GB | 0.99845 / 10 of 10 | identical |
| OpenVINO fp32 | 53 ms | 6.4 s | 16.8 s | 8.6 GB | 0.99845 / 10 | identical |
| OpenVINO int8 weights-only | 41 ms | 6.7 s | 17.1 s | 3.3 GB | 0.99780 / 9 | identical |
| OpenVINO int8 full (calibrated on vault text) | 35 ms | 7.0 s | 17.6 s | 6.1 GB | 0.99203 / 8 ✗ | 4.6 of 5 |
| ONNX Runtime int8 dynamic | 38 ms | 5.1 s | 12.8 s | 2.9 GB | 0.97253 / 7 ✗ | 4.6 of 5 |

- **int8 buys nothing on this CPU.** It has AVX2 but no VNNI, so int8 GEMM gains little; full-int8 OpenVINO is slower than fp32, and the one faster variant (ORT, −20 %) fails retrieval. Reassess on a VNNI CPU (Intel 11th-gen desktop or 12th+, AMD Zen 4+); the OptiPlex 7090 also takes 11th-gen Rocket Lake. The int8 exports are kept in `/opt/fast/models/encoder-cpu/` for that.
- **The cosine sub-gate (0.999) fails for fp32 by precision, not error.** sentence-transformers gives the same 0.99845; the reference is fp16 from the GPU, the median chunk is 0.99969 and 8 of 200 fall below 0.999 (worst: a Markdown table row). Retrieval agreement, the test that decides interchangeability, is perfect: 30 of 30 queries keep their top-10 and top-1.
- **Rerank dominates** a CPU search (~6.4 s of ~6.5 s at 20 candidates), so the fallback reranks 20 (oversample 4 at top-k 5) instead of the GPU path's 50. Oversample 4 vs 10, on the RAGLite MCP search path over 40 queries (brain vault `raglite_oversample_compare.py --cpu`): **a different result set, not a worse one by the only labelled measure.** Top-5 chunk-span overlap 2.75 of 5, top-1 the same in 29 of 40; on the 10 labelled queries, 4 hits 6 (MRR 0.342) and 10 hits 5 (MRR 0.308). End-to-end search on the CPU tier: **10.4 s at 4 vs 24.6 s at 10** (embedding, pgvector search and span expansion included); 9.1 s measured live through the MCP server. The GPU path keeps 10 until a larger labelled set separates them.

## Where the logic lives

| Piece | Location | Why there |
|---|---|---|
| VRAM budgets, placement, cold start, idle unload | this repo, `broker/` | per host; one checkout and one `config/roles.yaml` for every host, each activating the `cards:` entry for its card; `gpu_selector` lists are ordered preferences |
| Host selection for batch jobs | brain vault `Mind/tools/host/gpu_job.sh` + `gpu-broker-tunnel@.service` (on MIKASA) | neither host's broker can see the other; the choice is made where the job runs |
| CPU fallback server | this repo, `containers/encoder-cpu/` (runs on MIKASA) | same wire formats as the GPU workers, so the client switches by base URL |
| CPU fallback trigger | brain vault `Mind/tools/brain-rag/raglite_mcp_server.py` | the fallback must work exactly when the broker path is down, so it cannot depend on a broker |
| Remote power | `/etc/sudoers.d/90-mario-poweroff` on Scully and SASHA, WoL from MIKASA `~/bin/wake` | |

## Decisions

- **The CPU fallback is not a brokered tier (2026-10-28).** The ledger is GPU-only by design (refuses to start without `nvidia-smi` cards, budgets VRAM), a local broker would contend for the tunnel's fixed ports, and a fallback must not depend on another service being healthy. The fallback server lives here as a worker image so the design sits in one repo; its trigger lives in the client. **Revisit at the Triton spike:** if Triton replaces the serving containers, MIKASA becomes one more host running OpenVINO CPU instances behind the same API, and this split may collapse into a brokered tier.
- **Cold starts run outside the broker lock** (`a43006b`): a warm acquire no longer waits behind another role's 8–27 s cold start.
