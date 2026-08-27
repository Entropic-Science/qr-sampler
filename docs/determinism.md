# Determinism in the qr-sampler stack

*Written 2026-08-19 against qr-sampler `4d476d9`, qr-llm-owui `23b42a7`, and the live
qr-server deployment (vLLM 0.24.0, torch 2.11.0+cu130, 4× RTX 5060 Ti, PP=4/TP=1).
Line numbers reference those working trees.*

> **Status update (2026-08-27).** The §6 spec is IMPLEMENTED: tasks T1–T4 and
> T7 landed in qr-sampler (`seeded_prng` source, `seed` field + cross-field
> validation, `deterministic_prng` preset, adapter override with
> preemption-resume, tests) and T5–T6 in qr-llm-owui (Filter preset +
> compare-mode seeded lane). T8 is reduced to the Tier-2 runbook
> (`deployments/qr-server/README.md` + `tier2-replay.conf.example`): the
> checkpoint swapped to `qwen3.8-27b-prismaaqua` (`efae774`), whose live
> `config.json` was verified to declare 48/64 `linear_attention` (GDN) layers
> (`full_attention_interval: 4`) — so the A9 blocker CARRIES OVER and
> `VLLM_BATCH_INVARIANT=1` still refuses to start (upstream: vLLM #42960).
> References to `prismaquant-qwen` as "current" below are historical; the A9
> verdict applies unchanged to the new checkpoint. Implementation deviations
> from the spec (helper widening for `output_tok_ids`, a code-level
> `_PER_REQUEST_ONLY_SOURCES` allowlist union, validation as a
> `@model_validator` rather than in `resolve.py`) are documented in the
> landing commit.
>
> **Audit addendum (2026-08-27, post-landing ultracode review):**
>
> 1. **Discarded partial-prefill rows (new B8, fixed).** vLLM V1 samples
>    chunked-prefill rows and DISCARDS the token (`discard_request_mask` in
>    `gpu_model_runner`; vLLM rewinds its own per-request generators by 4 for
>    them — verified in the deployed 0.24.0). A consumed-order counter would
>    therefore drift with batch-mates' share of the token budget. Fix: the
>    adapter re-syncs the seeded source's counter to `len(output_tok_ids)`
>    (the LIVE list the `BatchUpdate` contract guarantees) before every draw,
>    so the block index is a pure function of token POSITION — a discarded
>    row re-draws the same block, which a pure function tolerates. This also
>    makes preemption resume automatic. Caveat pinned: never enable vLLM
>    `--async-scheduling` for seeded requests (placeholder tokens can pad the
>    live list); the deployment does not use it.
> 2. **Footgun #5 deviation.** The stateless-row fallback logs ERROR and
>    routes to the process default instead of raising (a per-row raise in the
>    engine worker would itself be EngineDeadError, and the adapter cannot
>    know what the lost state promised). Any `entropy.request.stateless_row`
>    occurrence during a replay/verification run invalidates that run.
> 3. **API-side validation hardened.** `validate_params` now dry-runs the
>    full `resolve_config` so every value-level and cross-field rejection
>    (seed bounds, envelope pairing) surfaces as a clean per-request error in
>    the API server, never as an engine-worker raise; the worker-side
>    resolution is additionally wrapped (degrades to a loud native bypass).
> 4. **Live incident (2026-08-27) — preset-expanded process defaults vs the
>    seeded envelope.** The engine adapter pre-expands ``QR_PRESET`` into
>    its process defaults (deployment: ``qthought_think`` ⇒
>    ``signal_amplifier_type=server``), and per-request merges run over
>    those defaults — so a seeded envelope riding a preset that does not
>    pin its own amplifier (``normal_t1``, ``creative_sampling``) inherited
>    the server amplifier and failed the deterministic cross-field
>    validation. Because the API-side dry-run merged over RAW defaults, it
>    accepted what the worker rejected, and the worker's defense degraded
>    those requests to native random sampling — replays that "diverged"
>    while a loud ERROR sat in the journal. (An earlier revision of this
>    note misread that as engine run-to-run logit noise; retracted.)
>    Fixes: the dry-run and ``__init__`` now share
>    ``_process_default_config()`` so API acceptance and worker resolution
>    cannot disagree, and the composition recipe pins
>    ``qr_signal_amplifier_type=zscore_mean``. Lesson for operators: any
>    bypass-degradation ERROR in the journal means the affected replies
>    were NATIVE-sampled — never read their variability as an engine
>    determinism measurement.
>    **Post-fix measurement (same day, normal serving, prefix caching on):**
>    seeded `normal_t1` and `chat_light` replayed hash-identical ×3 (and
>    identical to each other — they are sampling-equivalent configs);
>    seeded `creative_sampling` replayed identical ×5 on a warm prefix
>    cache, with only the COLD first run differing — the documented A4
>    effect (cached-KV vs recomputed-KV differs at the last ulp on
>    non-invariant kernels), which the hot lane's dense probability tail
>    amplifies into a token flip while T≈1 lanes absorb it. Consecutive
>    replays — the compare UI's ↻ use case — are identical on every lane.
> 5. **Semantic notes for lane users:** (a) `n>1`/`best_of` children share
>    one `qr_seed` ⇒ n identical completions — use one request per sample;
>    (b) every conversation TURN restarts the counter at block 0 with the
>    same seed, so the control arm's u-stream is identical across turns and
>    across users on a shared seed — fine for determinism, but cross-turn or
>    cross-user analyses see a CORRELATED control, so vary seeds per run in
>    studies; (c) guided decoding / `bad_words`-style vocab masks run after
>    the one-hot force and can veto the forced token — keep them off this
>    lane (pre-existing property of all qr lanes).

This document maps every source of nondeterminism in the serving stack **other than
the intentional one** (physical entropy driving token selection), and lays out the
most straightforward path to a chatbot lane that is **fully deterministic**: same
conversation + same seed ⇒ bitwise-identical reply, *regardless of what else the
shared engine is doing at the time*. The engineering background is
[Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)
(Thinking Machines, 2025), whose batch-invariant kernels are upstreamed in the vLLM
version already running on the box.

---

## TL;DR

A deterministic lane needs exactly three properties, one per layer of the stack:

1. **Deterministic logits** — the engine's forward pass must be *batch-invariant*:
   a token's logits may depend only on its own prefix, never on which other requests
   share the GPU step. vLLM 0.24 provides this via `VLLM_BATCH_INVARIANT=1` — but
   the current `prismaquant-qwen` checkpoint (Qwen 3.5 hybrid with gated-delta-net
   `linear_attn` layers) **cannot start with that flag**; the GDN attention backend
   does not support invariance. Either serve a full-attention model for the
   deterministic lane, or fall back to determinism-by-serialization (Tier 2, §5.5).
2. **Deterministic randomness** — qr-sampler currently has **no seed plumbing at
   all** (no config field, no env var, no per-request key), and even a seeded PRNG
   consumed as a *shared stream* is not deterministic under concurrency, because
   stream position depends on batching, row-thread scheduling, prefetch misses, and
   circuit-breaker wall-clock state. The fix mirrors the batch-invariance idea:
   random bytes must be a **pure function of (request seed, token index)** — a
   counter-based per-request generator, not a shared stream. This requires a small,
   well-contained code change (§6, task T1–T3).
3. **Deterministic prompts** — the chatbot layer must send a byte-identical request
   for the same conversation. qr-llm-owui is already close: no system prompt is
   seeded, task generations are disabled, and the request body the Filter builds is
   a pure function of the conversation. The dormant hazards are OWUI's
   `{{CURRENT_DATE}}`-family template variables (active the moment anyone sets a
   system prompt containing them) and the boot-probe watchdog (§4.3).

Everything else — silent entropy fallback, calibration draws, the coherence gate,
the server-side amplifier, prefetch — must be pinned off in the deterministic
preset; the full checklist is §5 and the implementation spec for coding agents is §6.

---

## 1. What "deterministic" means here

**Goal (Tier 1, load-invariant):** for a fixed *determinism envelope*, replaying an
identical chat completion request (same messages, same `qr_*` args, same seed)
produces a bitwise-identical token stream, **even while other requests — QRNG
lanes, compare panes, qthought, batch jobs — run concurrently on the shared engine.**

The *determinism envelope* is everything that must be held fixed for the guarantee
to be meaningful. It is not a weakness of the design; it is the definition of "same
system":

| Pinned | Why |
|---|---|
| Model weights (checkpoint hash) | trivially |
| vLLM + torch + CUDA versions | kernel selection and numerics change across releases |
| numpy version + BLAS build | `argsort` tie order, NEP-50 promotion, pairwise-sum blocking (`pyproject.toml` allows `numpy>=1.26,<3` — a real 1.x↔2.x straddle) |
| GPU model + count + parallel layout (PP=4/TP=1) | reduction layouts differ per architecture |
| libc/libm version | `math.erf` in the z-score amplifier is libm, ulp-level platform-dependent |
| qr-sampler version + full resolved `QRSamplerConfig` | every knob below |
| Engine flags that shape computation (`--max-model-len`, dtype, quantization) | change kernels/numerics |

Cross-machine or cross-version bitwise reproducibility is explicitly **out of
scope** — the Thinking Machines work makes the same restriction. Within the
envelope, the guarantee is bitwise; across envelope changes, the repo's own
equivalence class applies ("identical up to last-ulp summation order",
`selector.py:28-32`, `LEARNINGS.md` §perf-tranche).

Wall-clock fields in telemetry (`TokenSamplingRecord.timestamp_ns`, latency
fields, `entropy_nonce`) are exempt: they differ every run by design and feed
nothing back into sampling (verified — §3.2.6).

---

## 2. Background: why inference is nondeterministic even with a PRNG

Condensed from the Thinking Machines post, because every design choice below
borrows from it:

- **Floating-point addition is not associative.** `(a+b)+c ≠ a+(b+c)` once
  magnitudes differ; any change in reduction order changes the last ulp(s).
- **GPU kernels are nevertheless run-to-run deterministic.** The popular
  "concurrency + floating point" explanation is wrong for the forward pass: LLM
  forward kernels don't use atomics; the same kernel on the same input tensor
  gives bitwise-identical output every launch.
- **The real culprit is lack of *batch invariance*.** `torch.mm(a[:1], b)` and
  `torch.mm(a, b)[:1]` differ, because kernels choose different tiling/split
  strategies at different batch sizes. Attention adds a second axis: how the KV
  cache is chunked (prefill vs decode, splits along the KV dimension) changes
  reduction order for the *same* sequence position.
- **Server load converts that into user-visible nondeterminism.** Your request's
  batch-mates are other people's traffic; batch size and composition vary; so the
  logits for your token vary; at temperature 0 they measured 80 unique completions
  in 1000 identical greedy requests, first divergence at token 103. The completion
  is deterministic *given the batch*, but the batch is not deterministic given
  your request.
- **The fix is batch-invariant kernels:** consistent matmul tiling at every batch
  size (no split-K), data-parallel norms, and fixed-*split-size* attention with the
  KV cache/page table updated before the kernel — so a token's reduction order
  depends only on its own prefix length, never on batch-mates. Cost measured at
  ~1.6–2× latency in their unoptimized vLLM integration; their released
  `batch_invariant_ops` was subsequently upstreamed into vLLM proper.

The transfer to this stack: **replace "logits" with "random bytes" and the same
theorem applies to qr-sampler.** A seeded PRNG consumed as a shared stream is the
moral equivalent of a batch-*variant* kernel: reproducible in isolation,
nondeterministic under load, because *which* bytes a token receives depends on
global consumption order. §3.2 inventories exactly where that order leaks in.

---

## 3. Where nondeterminism enters, layer by layer

The serving path is:

```
browser → owui (:8810) → OWUI middleware + qr_sampler_parameters Filter
        → vLLM OpenAI server (:8000) → forward pass (PP=4) → logits
        → VLLMAdapter.apply() [V1 logits processor]
        → SamplingPipeline: temperature → entropy fetch → amplify(u) → select → one-hot
        → vLLM native sampler (inert: logits are one-hot) → SSE back up
```

Verdicts: **VERIFIED** = differs across runs by construction today;
**CONDITIONAL** = differs under specific load/config/timing; **SAFE** = checked
deterministic within the envelope.

### 3.1 Layer A — the engine (logits)

| # | Source | Verdict | Detail |
|---|---|---|---|
| A1 | **Batch composition** | VERIFIED | `--max-num-seqs 4` + owui chat + compare panes + qthought + boot probes share one engine; kernels are not batch-invariant without the flag, so concurrent traffic changes your logits. This is the article's headline effect, live on this box. |
| A2 | **Priority scheduling** | CONDITIONAL | `--scheduling-policy priority` (commit `77de583`) + `QR_CHAT_PRIORITY`: admission order changes batch composition. Irrelevant once kernels are batch-invariant; amplifies A1 until then. |
| A3 | **Chunked prefill / decode mixing** | CONDITIONAL | How a prompt is chunked depends on what else is scheduled; non-invariant attention kernels reduce in different orders per chunking. Covered by the batch-invariant attention design (fixed split size, KV cache updated pre-kernel). |
| A4 | **Prefix caching** | CONDITIONAL | A cache hit *reuses* KV written under some past batch composition instead of recomputing it. With batch-variant kernels, cached-KV ≠ recomputed-KV bitwise ⇒ history-dependence. With invariant kernels KV is identical either way (that is the point of invariance); still, the strict lane disables it (cheap) rather than proving it. |
| A5 | **Preemption / recompute** | CONDITIONAL | KV-block pressure can evict and recompute a request. Bitwise-safe under invariant kernels for *logits*, but it resets qr-sampler per-request state (see B10 — this constrains the temperature strategy and the seed counter design). |
| A6 | **PP=4 / TP=1 topology** | SAFE | Favorable: pipeline parallelism moves activations point-to-point (NCCL send/recv, no reductions); TP=1 means no all-reduce in the forward pass at all. `VLLM_BATCH_INVARIANT=1` additionally pins NCCL algo/channels and disables symm-mem all-reduce (`batch_invariant.py::override_envs_for_invariance`). |
| A7 | **cudagraphs (PIECEWISE)** | SAFE | Replaying a captured graph re-runs the same kernels; batch-invariant mode leaves cudagraphs on (it only forces `VLLM_USE_AOT_COMPILE=0`). |
| A8 | **TF32 / reduced-precision reductions** | CONDITIONAL | Nondeterministic *rounding mode selection* across kernel variants; batch-invariant mode forces IEEE fp32 and disables fp16/bf16 reduced-precision reductions. |
| A9 | **The model itself: GDN hybrid** | **BLOCKER** | `prismaquant-qwen` (Qwen 3.5) has `linear_attn` gated-delta-net layers → `GDNAttentionBackend`, which inherits `supports_batch_invariance() → False` (`vllm/v1/attention/backend.py:272`); the selector **raises RuntimeError** at startup with `VLLM_BATCH_INVARIANT=1` (`vllm/v1/attention/selector.py:161`). Model-selection criteria in §5.2. |
| A10 | **Speculative decoding / multi-step** | SAFE (unused) | Not configured; keep it off in the deterministic lane — acceptance/rejection couples tokens to draft-model numerics. |
| A11 | **Tokenizer, chat template, tool parser** | SAFE | Jinja chat template, `qwen3_xml` parsing, detokenization: pure functions of the text. `chat_template_kwargs={"enable_thinking": false}` is pinned constant by the owui Filter. |

### 3.2 Layer B — qr-sampler (the sampling pipeline)

The pipeline per token: temperature strategy → entropy fetch → amplification
(bytes → u ∈ (0,1)) → selector (top-k → softmax → min-p → top-p → CDF search at u)
→ one-hot force. The math stages are clean; the entropy *plumbing* is where
determinism dies today.

#### 3.2.1 Seed plumbing: there is none (VERIFIED)

- `QRSamplerConfig` has **no seed field**; no `QR_SEED` env var, no `qr_seed`
  extra-arg exists (exhaustive grep).
- The only seedable object in the package is
  `MockUniformSource(mean=127.5, seed=None)` (`entropy/mock.py:35-38`) — and the
  factory can never seed it: `build_entropy_source` passes a config only to
  classes whose first ctor param is named/typed `config`
  (`core/pipeline.py:98-128`); `MockUniformSource`'s first param is `mean`, so it
  is always constructed bare (`pipeline.py:165`) ⇒ `np.random.default_rng(None)`
  ⇒ OS-entropy-seeded. The same applies to `fallback_mode="mock_uniform"`
  (`pipeline.py:178-181`).
- Note `mock_uniform` is also *distributionally* wrong as a PRNG control: it emits
  clipped N(127.5, 40) bytes, while the z-score amplifier's default population
  parameters (127.5, 73.612 = 255/√12) assume uniform bytes. A proper PRNG lane
  wants uniform bytes (§6, T1).
- The only injection hooks are constructor seams (`SamplingPipeline(entropy_source=…)`,
  `QthoughtRoller(entropy_source=…)`) that the vLLM adapter never uses.
- Server-side seeded lanes exist (`qbert0g` `prng_uniform` / `prng_markov`
  controls, 128-bit fixed seeds, regenerable offline from
  `(id, seed, stream_offset_bytes)`) — but they are **shared streams**: see next.

#### 3.2.2 Stream-position dependence — the qr-sampler analog of batch invariance (VERIFIED under concurrency)

Even with a perfectly seeded PRNG behind the gRPC socket, *which* block of the
stream a given token consumes depends on global ordering:

| # | Mechanism | Where |
|---|---|---|
| B1 | Concurrent requests interleave draws on the shared source; batch composition and admission order (A1/A2) decide the interleaving | engine-wide |
| B2 | Row-parallel sampling: `apply()` samples batch rows on a thread pool (`adapter.py:882-891`, cap = `apply_parallel_rows or os.cpu_count()`); locks serialise the RNG but not the row↔block assignment order | `engines/vllm/adapter.py` |
| B3 | Prefetch redeem miss: on ticket-timeout the prefetched block is *discarded* and a fresh serial fetch issued (`qgrpc/source.py:530-538`) — timing decides which bytes a token gets. The `entropy_prefetch` docstring claims "timing-only, does not affect the sampled distribution" (`config/model.py:230-231`) — **false for any stateful/seeded server** | qgrpc |
| B4 | Calibration draws consume stream bytes at unpredictable times: per-pipeline at engine start, plus **re-calibration at request-add** whenever per-request config differs from the pipeline default (`adapter.py:687-690`; 200×10 000 bytes on qthought presets); zscore calibration is cached process-wide **first-writer-wins** per source instance (`amplification/zscore.py:46-47,123-127`) | amplification |
| B5 | `QuantumGrpcSource.warmup()` really draws 8 bytes at startup (`qgrpc/source.py:308`) | qgrpc |
| B6 | qthought runs in a separate process against the same qbert0g device/stream, advancing it independently | qthought.py |
| B7 | Bidi-streaming FIFO correlation can mismatch responses to requests if the server doesn't echo `sequence_id` (`transport.py:245-249`); `unary` mode (deployed) is safe | qgrpc |

**Consequence:** a shared seeded stream can make a run *reproducible after the
fact* (given recorded offsets — that is what the research PRNG lanes are for) but
can never make a chat lane *deterministic under load*. Determinism requires bytes
= f(request seed, token index), independent of consumption order (§5.3).

#### 3.2.3 Wall-clock-driven source switching (CONDITIONAL, always armed)

The identity of the RNG itself can silently change mid-generation:

- `FallbackEntropySource` swaps to the fallback leg **per token** on any
  `EntropyUnavailableError` (`fallback.py:278-311`), logging at most once per
  60 s. Deployed config: `QR_FALLBACK_MODE=system` ⇒ silent per-token
  `os.urandom` substitution.
- Whether the primary "fails" is itself wall-clock-driven: the adaptive circuit
  breaker's timeout is a function of observed latency P99
  (`qgrpc/breaker.py:80-125`), open/half-open transitions use `time.monotonic()`
  (`breaker.py:73,143,172`), and the TCP pre-probe fast-fails during a 5 s backoff
  window without touching the socket (`preprobe.py:106-110`).
- Draw-path degradation substitutes fallback bytes + an **uncalibrated** local
  z-score amplifier (`core/pipeline.py:360-369, 590-610`) — different math, same
  token position.
- Quota exhaustion (`RESOURCE_EXHAUSTED`, `source.py:702-729`) also lands in the
  fallback.

For the deterministic lane all of this must be structurally unreachable: a local
in-process PRNG source never raises `EntropyUnavailableError`, so the breaker,
pre-probe, prefetch, and fallback machinery simply never engage (§5.3).

#### 3.2.4 Amplification (mostly SAFE)

- `zscore_mean` (`amplification/zscore.py:159-197`): `u = ½(1+erf(z/√2))` from the
  float64 mean of the byte block. **Same bytes + same (population_mean,
  population_std, ε) ⇒ same u.** Stateless. The only caveats: calibration state
  (B4 — pin `zscore_calibration_samples=0`) and libm `erf` (envelope).
- `zscore_thought`: identical `u`; its side accumulator is exact integer math.
- `ecdf`: **not usable** — `u` depends on a 2 000-block calibration array drawn
  from the live source per build, never cached; same bytes give different `u`
  across processes.
- `server` (`ServerDrawAmplifier`): **not usable** — `u` is computed by qbert0g
  from a 100 KiB integrated block; not reproducible client-side by construction.

#### 3.2.5 Temperature strategies (per-request SAFE, two exclusions, one trap)

- Pure per-token functions of `(logits, config)`: `fixed`, `edt`, `edt_paper`,
  `gdt`, `dynatemp`, `belltemp`, `tt_exchange`, `evdt_tt`, `mix_temperatures` — no
  RNG, no clock, no I/O. SAFE.
- Per-request state, deterministic given identical inputs: `hvh_drift`
  (H/VH EMAs), `ring_buffer_ar` (selected-token history). Fresh instance per
  request (`adapter.py:674`, invariant 14). **Trap (B10):** vLLM preemption
  removes/re-adds the request, rebuilding `_RequestState` ⇒ EMA state is lost
  mid-generation at a wall-clock-dependent point. `ring_buffer_ar`'s history is
  reconstructible from `output_ids`; `hvh_drift`'s EMA is not. The deterministic
  preset therefore uses a **stateless** strategy (§5.3).
- `coherence_gate`: **excluded** — boost driven by server-computed live
  cross-device coherence (`DrawMeta.coherence_z/r`), unreproducible; it is active
  in every `qthought*` preset and in the engine default `QR_PRESET=qthought_think`.
- Stateless-row leak: a row with no `_RequestState` falls back to the
  **process-shared** default pipeline strategy instance (`adapter.py:908-945`) —
  cross-request state contamination for stateful strategies. Loud WARNING; the
  triggering batch-update bug was fixed in `40e67d6`. Watch for the warning in the
  deterministic lane's verification runs.

#### 3.2.6 Selector & math numerics (SAFE within the envelope)

- Dtype flow: logits stay float32 end-to-end; `_stable_softmax` preserves dtype;
  `T == 1.0` is an exact no-op; `np.sum`/`cumsum`/`searchsorted` are
  deterministic for fixed shape/dtype/build. One-hot force uses exactly `-inf` /
  `0.0`, and vLLM's downstream sampler provably returns the forced token (point
  mass survives temperature/top-k/top-p; `-inf + finite = -inf` defeats
  logit_bias). vLLM's own `seed`/`temperature` params are **inert** on non-bypass
  rows — no `--seed` flag needed.
- Tie order in `np.argsort(...)[::-1]` / `argpartition` is unspecified but
  deterministic per numpy build (`selector.py:28-32,466,526,545`); the CDF
  fast-head and compact-top-k paths change summation order within the documented
  last-ulp equivalence class. All envelope-pinned.
- **One real knob:** the entropy/varentropy helpers use BLAS `np.dot` over the
  full vocab (`temperature/base.py:98,149,159`); OpenBLAS may vary its reduction
  blocking with thread count. Pin `OPENBLAS_NUM_THREADS=1` and `OMP_NUM_THREADS=1`
  in the engine unit for the strict claim.
- Telemetry/logging: verified write-only — nothing (status files, perf
  aggregator, gate status, `/health/entropy`) feeds back into sampling.
- `_RequestState.prefetch_salt = os.urandom(16)` (`adapter.py:180`) randomises the
  commitment *nonce* every run — it rides `sequence_id` and the log record only,
  never the bytes, **unless** the entropy server derives content from
  `sequence_id` (qbert0g does not). Harmless for tokens; noted for completeness.

### 3.3 Layer C — the chatbot (qr-llm-owui)

The good news first: the deployment is unusually deterministic-friendly by
construction. No system prompt is seeded (`Model.params={}`), so OWUI's entire
template-variable machinery is dormant; chat history is rebuilt server-side from
the DB (client can't perturb it); the Filter's body mutation is a pure function of
the user's stored preset; RAG/web-search/memory are off; and OWUI's five auxiliary
task generations (title/tags/follow-up/search-query/retrieval-query) are
force-disabled at every boot (`bootstrap_connections.py:293-299,351-352`).

What remains:

| # | Source | Verdict | Detail |
|---|---|---|---|
| C1 | **`{{CURRENT_DATE}}` family** | CONDITIONAL (dormant) | `open_webui/utils/task.py:86-105` substitutes date/time/weekday/timezone + user fields (`{{USER_AGE}}` calls `datetime.now()` too); the browser computes a parallel set and ships it in `metadata.variables`. Gated entirely on a non-empty system prompt (`payload.py:23-24`). **Rule: the deterministic lane uses either no system prompt or a static one with no `{{…}}` tokens.** |
| C2 | **Boot-probe watchdog** | VERIFIED (first 30 min after boot) | `setup_orchestrator.py:640-648,722-850` fires real completions (temp 0.0, no `vllm_xargs` ⇒ engine default preset = `qthought_think` with QRNG + coherence gate) every 20 s while any probe warns — occupying engine slots and, on a non-invariant engine, perturbing every concurrent chat (A1). |
| C3 | **Task generations if re-enabled** | CONDITIONAL | They are DB config rows an admin can flip in the UI; re-asserted only at boot. Critically they bypass the Filter (`routers/tasks.py` uses the direct dispatcher), so they'd run with **no `qr_*` args** on the engine default preset. Keep disabled. |
| C4 | **User-set sampling params** | SAFE today, trap tomorrow | OWUI Chat Controls blanket-copies any advanced param (`temperature`, `seed`, …) onto the body (`middleware.py:2096-2099`) — all **inert** against the one-hot force. A user setting "Seed" today gets *nothing*. This is the natural hook for the deterministic lane (§6, T5): forward `body.seed → qr_seed` when (and only when) the deterministic preset is selected. |
| C5 | **Compare panes** | VERIFIED (their own lane) | Wall-clock behaviors mutate pane *conversation state*: busy panes silently skip turns (`compare.js:408-411`), 800 ms send-dedupe, 30 s pending window, failed turns popped from the transcript. Determinism of the main lane is unaffected; a deterministic *pane* additionally needs these removed or accepted. |
| C6 | **Reasoning-inline transform** | SAFE, coupling noted | The Filter's `stream()` moves `reasoning_content` → `content` per chunk (pure), which then persists and re-enters the next turn's prompt. Deterministic — but toggling the Filter off changes replayed history. Envelope-pinned. |
| C7 | **Per-inlet/outlet `/health/entropy` probes** | SAFE | Drive banners only; never touch the body. Latency-only. |
| C8 | **`open-webui` version** | envelope | Prompt assembly is OWUI-internal; pin `open-webui==0.10.2` (already pinned) inside the envelope. |

---

## 4. The design, in one sentence

**Make every stage a pure function of (conversation, seed, token index):**
logits = f(prefix) via batch-invariant kernels; bytes = g(seed, step) via a
counter-based per-request PRNG; u = h(bytes) via the stateless z-score amplifier;
T = t(logits) via a stateless strategy; token = CDF(logits, T, u). Nothing depends
on batch-mates, thread order, wall-clock, or shared mutable state — so concurrency
cannot perturb it, which is precisely the property the Thinking Machines post
calls batch invariance, extended one layer up.

---

## 5. The straightforward path (Tier 1: load-invariant)

### 5.1 Engine: turn on batch invariance

In the systemd drop-in for `qr-sampler-vllm.service`:

```ini
Environment=VLLM_BATCH_INVARIANT=1
Environment=OMP_NUM_THREADS=1
Environment=OPENBLAS_NUM_THREADS=1
```

and add `--no-enable-prefix-caching` to `ExecStart` for the strict tier (A4).
What the flag does in vLLM 0.24 (verified in the installed tree,
`vllm/model_executor/layers/batch_invariant.py`):

- On Hopper/Blackwell: disables cuBLAS split-K via workspace config; on Ampere:
  installs Triton persistent-matmul overrides for `mm/addmm/matmul/linear`.
- Overrides `softmax`/`log_softmax`/`mean`/`bmm` with batch-invariant kernels;
  forces cuBLASLt; disables fp16/bf16 reduced-precision reductions and TF32
  (IEEE fp32 everywhere).
- Attention: FlashAttention forced to `num_splits=1`, FlexAttention pinned to
  16×16 tiles, Triton unified attention in invariant mode, cascade attention
  auto-disabled; unsupported backends refuse to start rather than silently vary.
- Distributed: custom all-reduce and NCCL symm-mem off, NCCL pinned to
  `tree`/`Simple`/1 channel — inert for this box's PP-only topology (A6) but
  harmless.

Everything else stays: PP=4, cudagraphs PIECEWISE, priority scheduling,
`max_num_seqs` at whatever throughput wants — under invariance, batch composition
no longer touches numerics, so none of these need to be sacrificed. Expect a
throughput cost in the tens of percent (the article measured 1.6–2.1× latency in
the unoptimized integration; 0.24's kernels are better — measure it).

### 5.2 Model-selection criteria (model choice is open)

The deterministic lane needs a checkpoint whose every layer has a batch-invariant
kernel path in vLLM 0.24:

1. **Attention: full attention only.** No mamba/linear-attention/GDN hybrids —
   `GDNAttentionBackend`, `Mamba*`, `LinearAttention` backends do not implement
   `supports_batch_invariance()` (this is what blocks `prismaquant-qwen`).
   Check `config.json` for `linear_attn`, `mamba`, `gdn`, `ssm` layer types.
   Sliding-window full attention is fine.
2. **Backend coverage:** FlashAttention, Triton unified, FlexAttention, and the
   MLA family (`triton_mla`, `flashattn_mla`) all support invariance — any normal
   dense model (Qwen3 dense, Llama, DeepSeek-MLA) lands on a supported backend.
3. **MoE is supported** (Triton MoE experts declare invariance) but dense is the
   simpler first target.
4. **Quantization:** bf16 is the zero-risk choice. Quantized paths have invariant
   implementations (NVFP4 forced to a batch-invariant linear backend, marlin /
   flashinfer scaled-mm guarded), but each scheme is another surface to verify —
   if using one, the startup either succeeds under the flag or refuses loudly;
   then run the §5.6 harness before trusting it.
5. **Sizing for the box:** 4× 16 GB (PP=4, `VLLM_PP_LAYER_PARTITION` retuned to
   the new layer count). A dense ~14B bf16 (≈28 GB weights) leaves ample KV; a
   dense 30B-class needs FP8/INT4 (criterion 4 applies).
6. Keep speculative decoding off; multimodal off (already
   `--limit-mm-per-prompt.*=0`).

The startup check is self-enforcing: with `VLLM_BATCH_INVARIANT=1`, an
unsupported combination raises at boot instead of serving variant numerics.

### 5.3 qr-sampler: the seeded, keyed PRNG lane

The code change (spec in §6). The shape of it:

- **New entropy source `seeded_prng`** — uniform bytes from a counter-based
  generator: block for token *k* of a request = `Philox(key=seed, counter=k)`.
  Order-invariant by construction: threads, batching, prefetch, and other
  requests cannot change what token *k* receives. One instance **per request**,
  held in `_RequestState`, its counter initialized from `len(output_ids)` so
  vLLM preemption/re-add resumes at the right block (A5/B10).
- **New per-request config field `seed`** (`qr_seed` extra-arg). No process-wide
  `QR_SEED` default — a deterministic lane must be *opted into per request*, or
  identical concurrent conversations would collide on identical streams silently.
- **New preset `deterministic_prng`** pinning the whole lane:

  | Field | Value | Why |
  |---|---|---|
  | `entropy_source_type` | `seeded_prng` | the keyed source |
  | `signal_amplifier_type` | `zscore_mean` | stateless; `ecdf`/`server` excluded (§3.2.4) |
  | `zscore_calibration_samples` | `0` | no calibration draws, no first-writer cache (B4) |
  | `temperature_strategy` | `edt` (or `fixed`) | stateless ⇒ preemption-proof (§3.2.5); mirrors the creative lane's entropy-adaptive character without `hvh_drift`'s unreconstructible EMA |
  | `entropy_prefetch` | `false` | moot for a local source, pinned for clarity (B3) |
  | `top_k` / `top_p` / `min_p_base` | mirror the lane being controlled | selector math is deterministic for any values |

- Fallback/breaker/pre-probe: structurally unreachable — the source is local and
  infallible; `fallback_mode` never engages. No silent RNG swaps (§3.2.3).
- Sampler-side settings that need **no** change once bytes are keyed:
  `apply_parallel_rows` (thread order no longer matters), `max_num_seqs`,
  scheduling policy.

### 5.4 Chatbot lane rules (qr-llm-owui)

1. Add `deterministic_prng` to the Filter's `UserValves.preset` choices; when it
   is selected, forward the user's Chat-Controls `seed` (today inert, C4) as
   `vllm_xargs["qr_seed"]`, defaulting to a fixed documented seed when unset.
   Never forward `seed` for other presets — it must not silently flip a QRNG
   lane's science.
2. System prompt: none, or static text with no `{{…}}` tokens (C1). This is a
   usage rule, not code — worth a line in the UI description of the preset.
3. Keep task generations disabled (already re-asserted per boot; C3), keep RAG
   and web search off.
4. Compare panes: optionally add a `seeded_prng` entropy axis + seed box to the
   pane config (`compare_proxy` is `extra="forbid"` — needs the field added);
   determinism of *panes* additionally inherits their wall-clock transcript
   quirks (C5).
5. Pin `open-webui==0.10.2` (done) inside the envelope.

### 5.5 Tier 2: serialized replay (works with `prismaquant-qwen` today)

Until the model question is settled (or for the hybrid checkpoint specifically),
the weaker guarantee — deterministic when the engine processes the request under
the same batch conditions — is achievable now by *making the batch composition
deterministic* instead of making kernels invariant to it:

- `SHARED_MAX_NUM_SEQS=1` (the request is always alone in the batch),
- `--no-enable-prefix-caching` (no history-dependence via cached KV),
- uniform priority (unset `QR_CHAT_PRIORITY`),
- quiesce other engine clients during the run (qthought, compare panes; wait out
  the boot-probe watchdog, C2),
- `QR_APPLY_PARALLEL_ROWS=1`, `QR_ENTROPY_PREFETCH=0`,
- plus the same seeded-source work from §5.3 (Tier 2 changes nothing about the
  entropy layer's requirements).

Kernels are run-to-run deterministic (the article's own measurement), so a fixed
batch of one, a deterministic scheduler, and keyed bytes give bitwise replays.
Cost: total serialization — this is a research/verification mode, not a serving
mode. This is also exactly the regime in which the existing seeded `qr_bypass`
smoke test produces identical outputs today.

### 5.6 Verification protocol

Determinism is a property you *measure*, not declare:

1. **Replay under load (the acceptance test for Tier 1):** send the same
   deterministic-lane request N=20 times — half solo, half concurrent with a
   synthetic mixed load (QRNG chat requests + a long batch job). All 20 token-id
   streams and per-token logprobs must be identical. This directly reproduces the
   article's 1000×"Tell me about Richard Feynman" experiment, on this stack.
2. **Cross-boot replay:** restart `qr-sampler-vllm`, replay, compare — catches
   hidden startup state (calibration, warmup) and cudagraph capture effects.
3. **Token-level triage when a mismatch appears:** log `u`, `T`, and the top-8
   logits at the first divergent step (the sampling logger already records
   `u_value`/temperature per token). `u` differs ⇒ entropy layer bug; `T` or
   logits differ with same `u` ⇒ engine invariance bug.
4. **Watch for `entropy.request.stateless_row` warnings** during runs — any
   occurrence means a row sampled with shared default state (§3.2.5).
5. **CI:** the qr-sampler unit tests in §6 T7 pin the pure-Python half (same
   seed ⇒ same bytes ⇒ same u ⇒ same token for fixed logits) forever.

---

## 6. For coding agents: implementation spec

Scope: qr-sampler (T1–T4, T7), qr-llm-owui (T5–T6), deployment (T8). Follow
`AGENTS.md` invariants throughout — notably: no hardcoded values (every constant a
`QRSamplerConfig` field), registry pattern for the new source, per-request fields
via `json_schema_extra={"per_request": True}`, frozen result types,
`python scripts/check.py` green before commit. The new per-request field and
preset touch the qthought contract surface only if exported via `contract.py` —
export `PRESET_DETERMINISTIC_PRNG` and bump `CONTRACT_VERSION` only if qthought
needs it by name; otherwise leave it out of `contract.__all__` and no bump is
required.

### T1 — `SeededPrngSource` (`src/qr_sampler/entropy/seeded.py`)

```python
class SeededPrngSource(EntropySource):
    """Counter-based deterministic PRNG source (uniform bytes).

    Block k is Philox(key=seed, counter=[0,0,0,k]) — a pure function of
    (seed, k), independent of draw order across requests, threads, and
    prefetch. Instances are PER-REQUEST (one per _RequestState); the internal
    draw index makes consecutive get_random_bytes() calls consume consecutive
    counter blocks, and starts at `initial_step` so preemption re-adds resume
    correctly.
    """

    def __init__(self, seed: int, initial_step: int = 0) -> None:
        self._seed = seed
        self._step = initial_step
        self._lock = threading.Lock()   # ABC thread-safety contract

    @property
    def name(self) -> str: return "seeded_prng"
    def is_available(self) -> bool: return True

    def get_random_bytes(self, n: int) -> bytes:
        with self._lock:
            step = self._step
            self._step += 1
        gen = np.random.Generator(
            np.random.Philox(key=self._seed, counter=[0, 0, 0, step]))
        return gen.bytes(n)

    def close(self) -> None: pass
```

Notes:
- Uniform bytes match the amplifier's default population parameters
  (`population_mean=127.5`, `population_std=255/√12≈73.6122`). (Footnote: the
  exact discrete-uniform σ is √(65535/12)≈73.90; the 0.4 % delta scales every z
  identically and is shared with the QRNG lanes' assumption — do not "fix" it in
  this preset; consistency with existing lanes matters more.)
- Philox raw streams are algorithmically fixed; `Generator.bytes` is stable in
  practice and numpy is envelope-pinned regardless. Use the 4×64-bit counter's
  high word for the step so each token has 2^128 blocks of headroom — no overlap
  by construction.
- Do **not** implement `prefetch`/`get_draw` — return the ABC defaults (`None` /
  raise), so the pipeline's serial path is always taken.
- Register in `EntropySourceRegistry._BUILTINS`
  (`"seeded_prng": "qr_sampler.entropy.seeded:SeededPrngSource"`), add
  `profiles/entropy/seeded_prng.yaml`, and mirror the entry point in
  `pyproject.toml` per AGENTS.md §"New entropy source".
- **Important:** `build_entropy_source` must *reject* `entropy_source_type=
  "seeded_prng"` without a resolved per-request seed (raise
  `ConfigValidationError`) — a process-default seeded source shared by all
  requests would resurrect the shared-stream problem (§3.2.2). Constructing it
  is the adapter's job (T3), not the factory's.

### T2 — config: the `seed` field and the preset (`config/model.py`, `config/presets.py`, `config/resolve.py`)

- `QRSamplerConfig.seed: int | None = Field(default=None, ge=0, lt=2**63,
  json_schema_extra={"per_request": True}, description=...)`. The derived
  `PER_REQUEST_FIELDS` set picks it up automatically; `qr_seed` extra-arg and
  `QR_SEED` env work for free — **but** add a validator rejecting a process-level
  `QR_SEED` (seed is per-request-only by design; see T1 note). Simplest: in
  `resolve_config`, error if `defaults.seed is not None`.
- Cross-field validation in `resolve.py` (single validation point, invariant 7),
  applied when `seed is not None`:
  - `entropy_source_type` must be `"seeded_prng"` (and vice versa: `seeded_prng`
    requires `seed`),
  - `signal_amplifier_type ∈ {"zscore_mean", "zscore_thought"}`,
  - `zscore_calibration_samples == 0`,
  - `temperature_strategy` not in `{"coherence_gate"}` (stateless strongly
    recommended; hard-error only on the gate, warn on `hvh_drift`/
    `ring_buffer_ar`),
  - `entropy_prefetch` is False,
  - `bypass` is False.
- `BUILTIN_PRESETS["deterministic_prng"]` per the §5.3 table (inner keys without
  the `qr_` prefix; no `seed` inside the preset — the seed always arrives as its
  own `qr_seed`). Matching `profiles/presets/deterministic_prng.yaml` (the sync
  test iterates automatically).
- Fix the `entropy_prefetch` docstring while here: it is timing-only **for
  stateless servers**; against a stateful/seeded stream a redeem miss changes
  which bytes a token receives (§3.2.2 B3).

### T3 — adapter: per-request source override (`engines/vllm/adapter.py`, `core/pipeline.py`)

- `_RequestState.__slots__` += `entropy_override`. In `_update_state_impl`'s
  added-branch (after `resolve_config`, before pipeline routing): if
  `req_config.seed is not None`, build
  `SeededPrngSource(seed=req_config.seed, initial_step=len(output_ids or []))`
  and store it; route the request to the **default** pipeline (the override
  replaces the source per call, so no per-source pipeline pre-init is involved).
  Add `"seeded_prng"` to the `validate_params` allowlist so API-server-side
  validation accepts it (mirror of `adapter.py:491` + `_allowed_source_names`).
  The `initial_step` from `output_ids` is what makes preemption re-adds (A5)
  resume at the correct counter — add a comment pointing at this document.
- `SamplingPipeline.sample_token` gains `entropy_source: EntropySource | None =
  None` (mirroring the existing per-request `config`/`amplifier`/`strategy`
  override pattern); `active_source = entropy_source or self._entropy_source`
  used in all three fetch branches; skip the prefetch-ticket handoff and the
  `FallbackEntropySource.last_source_used` fallback-detection when an override is
  present (`isinstance` guard already exists).
- `_sample_row` passes `entropy_source=state.entropy_override`.
- Skip step-0 prefetch (`adapter.py:716-720`) for override requests.
- Amplifier: the reuse-or-rebuild branch (`adapter.py:681-691`) with
  `zscore_calibration_samples=0` makes `calibrate()` a no-op — assert in tests
  that no entropy is drawn at request-add for the deterministic preset.

### T4 — (optional, correctness) seed the mock factory gap

`build_entropy_source` silently constructs `MockUniformSource()` unseeded because
`accepts_config` inspects only the first ctor param (§3.2.1). Independent of the
deterministic lane, make this loud or configurable — at minimum a docstring on
`mock_uniform` stating it is *never* seeded via config. Do not widen
`accepts_config` heuristics; prefer an explicit branch.

### T5 — owui Filter (`qr-llm-owui/src/qr_llm_owui/functions/_sources/qr_sampler_filter.py` + envelope)

- Extend `UserValves.preset` `Literal` with `"deterministic_prng"` (also in
  `functions/qr_sampler_filter.json` meta and `_build.py`; the envelope content
  hash drives re-seeding, so rebuild via the repo's `_build.py` flow).
- In `inlet`, when the resolved preset is `deterministic_prng`:
  `xargs["qr_seed"] = int(body.get("seed") or DEFAULT_DETERMINISTIC_SEED)` —
  OWUI's params-merge runs before the Filter inlet, so a user-set Chat-Controls
  seed is already on the body. Never forward `seed` for any other preset (it
  would silently flip a QRNG lane to PRNG).
- No guard-xargs analog needed unless the preset mirrors `creative_sampling`'s
  `top_p` guard — decide with the preset definition in T2.

### T6 — compare proxy (optional)

`compare_proxy.py`: add `"seeded_prng"` to `_ENTROPY_SOURCES`, add an optional
`seed: int | None` to `PaneConfig` (`extra="forbid"` models — the field must be
declared), map it to `qr_seed` in `build_vllm_body` only when entropy is
`seeded_prng`. UI: one input in `compare.js`'s pane popover.

### T7 — tests (qr-sampler)

- `tests/test_entropy/test_seeded.py`: same `(seed, step)` ⇒ same bytes; distinct
  steps ⇒ distinct blocks; `initial_step` resume equals uninterrupted sequence;
  thread-hammer N threads drawing from one instance ⇒ the *set* of blocks equals
  steps 0..N−1 (order-free property); no `prefetch` implemented.
- `tests/test_config`: seed cross-field validation matrix (each forbidden combo
  raises `ConfigValidationError`); `QR_SEED` process-default rejected; preset
  expands correctly; `PER_REQUEST_FIELDS` includes `seed`.
- `tests/test_engines/test_seed_override.py` (pattern-match `test_bypass.py`):
  two requests, same seed, same logit rows, mixed into one batch with a QRNG-mock
  row and processed with `apply_parallel_rows` unset (threaded) ⇒ identical
  selected tokens for the seeded pair across 50 repetitions; re-add with
  `output_ids` of length k resumes at counter k (assert via recorded `u` values).
- Pipeline-level replay: full `sample_token` loop over fixed synthetic logits,
  same seed twice ⇒ identical `SamplingResult.token_id` and `u_value` sequences.
- Frozen-gate note: do **not** touch `tests/test_statistical_properties.py`
  expectations; the seeded source must pass the same KS-uniformity gate as
  `mock_uniform` — add it to that suite's parametrization.

### T8 — deployment (`deployments/qr-server/`, live drop-in)

- Unit drop-in: `VLLM_BATCH_INVARIANT=1`, `OMP_NUM_THREADS=1`,
  `OPENBLAS_NUM_THREADS=1`, `--no-enable-prefix-caching` (strict tier), model
  path/name swap per §5.2, `VLLM_PP_LAYER_PARTITION` retuned for the new layer
  count. Mirror in `deployments/qr-server/qr-sampler-vllm.service` + README.
- `/etc/qr-server/qr-server.env`: no `QR_SEED` (per-request only). No change to
  `QR_PREINIT_ENTROPY_SOURCES` (seeded_prng is not pre-initialized — it is
  per-request by design; ensure the allowlist path in T3 covers it).
- Verification runbook: §5.6 steps 1–4, plus the existing deploy smoke
  (`/health/entropy` quantum lane intact — the deterministic lane must not
  disturb QRNG serving).
- Tier-2 variant (hybrid model): the §5.5 env set instead of
  `VLLM_BATCH_INVARIANT`.

### Known footguns (read before implementing)

1. `accepts_config` first-param heuristic (§3.2.1) — do not route seeding
   through it (T1's explicit rejection + T3's adapter construction avoid it).
2. Preemption re-adds rebuild `_RequestState` — the `initial_step=len(output_ids)`
   line in T3 is load-bearing; without it, replays diverge only under memory
   pressure, the worst kind of bug.
3. The batched one-hot force fills the whole tensor with `-inf` — seeded rows are
   normal sampled rows (unlike bypass rows) and need no partition changes.
4. `entropy_prefetch` docstring currently over-promises (T2 fixes it).
5. A stateless-row fallback (`adapter.py:908-945`) silently uses the process
   default pipeline — for a seeded request this would sample from the *QRNG*
   default. After T3, assert loudly (raise, don't warn) when a stateless row
   carries no state but the batch's default config has `seed` unset and the
   request expected one. At minimum, treat any `entropy.request.stateless_row`
   during verification as a failure.
6. vLLM's native sampler params (`temperature`, `seed`, `top_p`) remain inert on
   seeded rows exactly as on QRNG rows (one-hot force). Do not also set
   `qr_bypass` — bypass wins over everything and would hand the row to vLLM's
   sampler instead.

---

## 7. What stays nondeterministic even after all of this

- Anything outside the envelope (§1): version bumps, hardware changes, libm,
  numpy builds. Re-run §5.6 after any envelope change and re-baseline.
- Telemetry: timestamps, latencies, `entropy_nonce` (fresh `os.urandom` salt per
  request), status files — different every run, sampling-inert by verified
  construction.
- The QRNG lanes, on purpose — that is the product. The deterministic lane is the
  control arm beside them, and under batch-invariant kernels both lanes finally
  share bitwise-identical logits for identical prefixes, which makes the
  comparison cleaner science too: any divergence between a QRNG run and a seeded
  run with the same prompt is then attributable to the entropy stream alone, not
  to batch noise.
