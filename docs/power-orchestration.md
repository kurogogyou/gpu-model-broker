# Power orchestration across several clients

Written 2026-11-05, out of a concrete two-pass transcription case. Companion to [`host-routing.md`](host-routing.md), which decides *which* host serves a role; this one is about *who is allowed to turn that host off*. Open design, not a decision: the current mechanisms are described because they are what a redesign has to replace, not because they are right.

## The short version

Power is decided by clients, each holding private state about a shared remote resource. That works while there is one client. It has no answer for two, and the guard that looks like it covers the gap (`active_handles`) is an instantaneous sample of a value that legitimately drops to zero mid-job.

None of this is urgent: every host is owned by one person who can power it off by hand, and the planned always-on box removes most of the question. It is written down because the failure is silent, and because the next person to add a client will reasonably assume the broker handles this.

## Where power decisions live today

| Piece | Where | Knows about |
|---|---|---|
| `transcribe_remote.sh` wake + poweroff + boot-id lease | `ai-transcriber` repo, runs on the client | its own chain only |
| `gpu_job.sh` host selection, wake, poweroff (busy check since 2026-11-05, below) | brain vault `Mind/tools/host/`, runs on MIKASA | its own job, plus a `/status` sample |
| the scoped `poweroff` sudoers rule | `/etc/sudoers.d/90-mario-poweroff` on each GPU host | nothing |
| **the broker** | this repo | **every client on its host, and nothing about power** |

The last row is the whole problem. The broker is the only process that can see all clients on a host, and it is the only one with no opinion about whether the host should be on. Every component that *does* have an opinion can see exactly one client: its own.

## What the clients actually do

`transcribe_remote.sh` is the more developed of the two. It probes ssh; if the host is down it sends a WoL packet, and on exit it powers the host off again. Chained runs share one boot through a lease file keyed on the host's `/proc/sys/kernel/random/boot_id`, so a lease cannot survive a reboot or be replayed from a stale file, and a lease is only ever taken by the run that did the waking: a host that was already up belongs to whoever turned it on.

That last rule is what makes it safe today, and it is also exactly where it stops scaling. "Whoever turned it on" is answered from `$XDG_RUNTIME_DIR` on one client. A second client's lease directory is invisible, on another machine, under another user. **There is no place where the question "who turned this host on" has an authoritative answer.** With one client the client's private belief and the truth coincide, which is why nothing has gone wrong yet.

**`gpu_job.sh` had no guard at all until 2026-11-05.** It powered off any host it woke without asking the broker, so a `transcribe_remote.sh` run from COLETTE on a host a MIKASA job had woken died when that job ended. Unlike the race below, that needed no unlucky timing, only overlap. It now samples `/status` before restoring its tunnel and powers off only after two idle samples 15 s apart, waiting up to 30 min, else it leaves the host on. Two samples narrow hole 1 below without closing it. Nightly work on MIKASA goes through `gpu_nightly.sh`, which runs every step in one `gpu_job.sh` run, so a night has one wake and one power decision. Verified live on SASHA 2026-11-05 with a simulated second client.

## The guard that exists, and its two holes

Before powering a host off, `transcribe_remote.sh` asks the broker whether anybody else is using it:

```python
s = json.load(urllib.request.urlopen("http://127.0.0.1:8090/status"))
sys.exit(0 if s.get("active_handles", 0) > 0 else 1)
```

### 1. `active_handles` samples the wrong thing

It is a correct guard against *someone is mid-inference* and an incorrect guard against *someone has a job in progress*. A batch embedder that acquires per batch and releases in between is idle, by this measure, in every gap between batches. The window is small per occurrence and the job is long, so over a bulk ingest the odds of sampling a gap are not small at all.

Concretely, with a transcription chain on one client and a RAG ingest on another:

| | client A (transcribe) | client B (ingest) | host |
|---|---|---|---|
| t0 | wakes host, takes lease, pass 1 | | on |
| t1 | `KEEP_ON`, exits, host stays up | | on |
| t2 | | `/acquire` embed, batch 1 | on |
| t3 | | releases between batches | on, **`active_handles` = 0** |
| t4 | pass 2 with `RELEASE`, lease matches, sees 0 handles | | **poweroff sent** |
| t5 | | `/acquire` → connection refused, mid-ingest | off |

Client A did everything right. It woke the host, it held a valid lease for the current boot, and it checked for other users. The check just happened to land in a gap.

### 2. `hold_until` is written and never read

> ✅ **Fixed 2026-11-05 (option B below).** A handle with `hold_seconds` is now a lease: the idle sweep and `/status` drop it once `hold_until` passes, `POST /renew` extends it, and the reference client takes a 300 s lease by default and renews it in the background while its `with` block runs. A killed client's handle now lapses within one lease. `/status` also lists each handle's `client_id` and `hold_until`. Tests: `tests/test_handle_lease.py`. Still open: hole 1 for clients that acquire per batch, and handles taken with no lease (`hold_seconds=0`, or raw `/acquire` callers that omit it), which keep the old until-`/release` contract.

`ClientHandle.hold_until` is computed in `state.create_handle()` from a `hold_seconds` the client may send, and **no code path ever reads it**. Verified 2026-11-05: the only occurrences in `broker/` are the field declaration and the assignment. There is no reaper and no expiry check; `candidates_to_stop_for_idle()` skips any role with a handle, of any age.

Two consequences, pulling in opposite directions:

- A client **cannot declare intent**. `hold_seconds` is the natural way to say "I am a batch job, treat me as present for the next ten minutes even between acquisitions", which is precisely the fix for hole 1, and the field is inert.
- A **leaked handle is permanent**. If a client dies between `/acquire` and `/release` (SIGKILL, host reboot, network partition; the context manager covers ordinary exceptions and ordinary exits), the handle stays in `state.handles` forever. That pins the worker container resident against the idle sweep *and* makes `active_handles > 0` true for all time, so no client will ever power that host off again. The client's docstring says "broker-side cleanup catches up via idle sweep", which is not true for this case: the idle sweep is the thing the stale handle blocks.

The leak fails safe for power (host stays on) and unsafe for VRAM (a role pinned resident can force evictions). Reading `hold_until` would close both holes with the same change.

## Design options

Ordered roughly by cost. None is chosen.

**A. Do nothing; power off by hand.** Legitimate, and the current answer. One owner, every host reachable, the cost of a mistake is watts. The reason to move off it is that the failure in the table above is silent on both sides: client A reports success and client B reports a connection error that looks like a flaky tunnel.

**B. Read `hold_until`.** Smallest real change. Expire handles past their hold, and let a batch client declare a window and renew it. Closes the stale-handle leak and narrows the sampling race to the length of one hold. Does not answer "who owns the power", so clients still decide, and a client that never sets `hold_seconds` is no better off than today.

**C. Move the lease into the broker.** `/power/lease` with a refcount and a grace period, replacing the client-local file. The broker is the only process that sees every client on its host, so this is where the question has an answer at all. Two things make it harder than it looks: the broker would be deciding to kill the machine it runs on, so the poweroff has to be handed to something that survives it (a systemd timer armed on the way out, or a one-shot `shutdown -h +1` that a later acquire cancels); and the broker binds `127.0.0.1`, so every client already reaches it through its own tunnel, which is fine for this but means the lease is only as reliable as each client's tunnel.

**D. An external supervisor on an always-on host.** It owns power for every GPU host: polls each broker's `/status` over a tunnel, powers a host off after a grace period with no handles, wakes it on request. Decouples the decision from the machine being decided about, which is the cleanest property of the lot, and it is the natural home for the wake side too (clients stop needing WoL at all, they ask the supervisor). Cost is a new always-on service and a new single point of failure for *starting* work, though not for finishing it.

Option D only makes sense if there is an always-on host to put it on, which is the next section.

## What the planned always-on box changes

Mario's plan as of 2026-11-05: move the **RTX 3060 currently in SASHA** into a machine kept on ~24/7, parts pending. The 3060 is enough for transcription and embeddings.

⚠️ This differs from what [`host-routing.md`](host-routing.md) records, which has the 3060 in SASHA as a wake-on-demand tier 2 and a "planned dual-GPU Scully" with the 5060 Ti joining. Treat that doc's hardware layout as the older plan until the parts land and the routing table is rewritten.

If it happens, most of this document stops mattering, and it is worth being clear about why: the roles that several clients contend for (embed, rerank, transcribe) all fit on a 12 GB card that is never off. Nobody has to decide whether to power it down, so there is nothing to orchestrate. What remains is the narrow case the current lease already handles well: **wake a big-VRAM host for a heavy or LLM job, power it off after.** One client at a time, by job weight rather than by contention.

That argues against building C or D now. It argues for B, which is cheap, is correct in either world, and fixes a stale-handle leak that is a problem on an always-on host too, where a pinned role costs VRAM indefinitely and no reboot comes along to clear it.

The wall-meter comparison that `host-routing.md` says decides the always-on question is still pending, and it decides this one too.

## Open questions

1. Should a client be able to declare a job window, or should the broker infer it? Reading `hold_until` picks the first; a grace period after the last release picks the second and needs no client change.
2. What is the grace period, if any? It trades idle watts against wake latency for the next job, and `host-routing.md` already has both numbers for the two cards.
3. Who wakes a host under option D, and what does a client do when the supervisor is down? The CPU-fallback precedent in `host-routing.md` says a fallback must not depend on the service it is a fallback for.
4. Does the 5060 Ti still go into Scully, and if so does SASHA retire entirely once the 3060 moves out?
5. Should `transcribe_remote.sh`'s lease be retired if B, C or D lands, or kept as the no-broker path? It is the only mechanism that works when the broker is down, which is also when it is least able to check whether anyone else is there.

## Status of the claims here

Measured or read off the code on 2026-11-05: `hold_until` has no reader; `candidates_to_stop_for_idle()` ignores handle age; `active_handles` is an unfiltered count of `state.handles`; the transcribe lease is keyed on boot id and only taken on wake; WoL to ssh took 41 s and 37 s on two consecutive Scully wakes, and the worker cold start was 12.68 s both times.

**Not measured:** the two-client race in the table above has never been observed. It is read off the code, and the gap between a batch client's releases has not been timed against anything. Nobody has run two clients against one host concurrently.
