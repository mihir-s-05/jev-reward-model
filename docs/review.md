# Implementation review — 2026-09-16

## Verified locally

`python -m compileall -q src tests`, `bash -n scripts/sweep.sh`, and **10 CPU pytest tests pass**. Tests use deterministic simulator fixtures, small PyTorch tensors/a tiny causal stand-in, and `httpx.MockTransport`. They do not download model weights or contact external evaluators.

The limited test surface targets research-critical invariants: explicit finish versus timeout; permanent forbidden-action failure; no oracle labels in public state; prerequisite retry/multiline behavior; dataset separation and solvability; discounted shaping telescoping and terminal zero; no oracle fallback for Jev-only rewards; GAE versus Monte Carlo; PPO ratio/clipping/sign; sampled-token likelihood alignment and temperature/prefix value; documented HTTP fields, probability parsing, cache accounting and secret exclusion. Several invariants share a test rather than expanding a production-style suite.

## Changes from the earlier scaffold

Removed combined train/evaluation loading, implicit success on timeout, top-k/top-p versus likelihood mismatch, first-line output repair, zero-valued placeholder entropy, and one-transition optimizer updates. Added structural/length splits, opaque operation IDs without topological hints, factual receipts, both alternative-judge controls, actual checkpoints/evaluation loading, reproducible settings/data/RNG capture, concurrent validated evaluator requests, usage/cost accounting, independent audits and reporting.

Reviewed PPO prefix-value causality, old-log-probability detachment, response/EOS boundaries, complete-episode GAE, minibatch accumulation, value clipping, optional frozen-reference behavior, shaping boundary reuse and terminal handling. The actor's text-only path through the actual conditional-generation model was checked against the pinned Transformers source. Jev schema was checked directly against the supplied API documentation, including the returned level legend.

## Not verified by execution

**The Qwen model was not loaded or trained on a GPU. Real Jev and vLLM requests were not made. No learning, throughput, cost, memory, or calibration results are reported.** The exact pinned Transformers/PEFT runtime was source-reviewed, not exercised with weights here.

Before the full study, run the documented actor preflight on the target hardware and device (`configs/grounded.yaml` on CUDA, or `configs/grounded_cpu.yaml` on CPU). It compares cached batched sampling with uncached training likelihoods and checks actor/critic backward gradients, without an optimizer update. Then run each judge's small counterexample audit to verify live authentication, schema, context budget, structured-output support, serving revision and usage. Finally perform a small grounded training run and checkpoint reload, followed by one run per evaluator arm, before scaling seeds/updates. These are runtime acceptance checks, not claims of work already performed. CPU actor runs execute the same rollout/PPO/eval loop; they are expected to be much slower. Qwen-judge arms still need a separate frozen server even when the actor is on CPU.

Potential model-specific risks include hybrid recurrent-kernel backward support, low-precision differences between cached and uncached paths, and gradient-checkpointing behavior. The preflight intentionally fails on significant joint-likelihood mismatch; do not hide it by widening tolerance. Fix the dependency/kernel/precision configuration consistently across arms.

Hosted Jev may update behind `jev-latest`; model-string checking cannot establish immutable weights. The local judge revision must match its actual server. Structured JSON does not guarantee valid probability mass, semantic correctness, or resistance to reward hacking. HTTP/schema errors halt training rather than silently alter rewards. Inspect all of these before interpreting experimental output.

This is a single-device research baseline, not a promise of maximum hardware throughput, a distributed RL implementation, or a production service. The chosen simulator is narrow and may be too easy for the untrained checkpoint; validation pilots determine whether a learning comparison is informative. Only observed results can establish that Jev works as a reward source.
