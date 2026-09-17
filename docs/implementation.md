# Implementation notes

## Why a small custom PPO implementation

The experimental unit is a tool transition containing several generated tokens,
with actor-only masking and boundary-specific discounting. Keeping these operations
in `ppo.py` makes the comparison auditable without relying on a rapidly changing
trainer API or coercing a multimodal model into an unsupported reward-model wrapper.
This is not a high-throughput distributed RL framework.

## Actor/critic and memory

`Policy` loads `Qwen3_5ForConditionalGeneration` with its native checkpoint keys. It
then retains `model.language_model` and the LM head, deletes the surrounding object,
and moves the retained modules to one GPU. This avoids loading the full multimodal
checkpoint into a text-only class with incorrectly remapped or randomly initialized
weights. Missing language-model keys and shape mismatches cause a hard failure;
unused checkpoint keys are preserved in metadata for inspection.

LoRA targets actual `nn.Linear` projections in the text backbone, including linear
attention/DeltaNet projections as well as full attention and MLP projections. The
frozen vocabulary projection is outside those targets. The actor and value head
share the adapted backbone; value gradients may also update its LoRA parameters.
There is no independently trained, large critic model. Disabling the adapter supplies
the frozen initial reference policy without storing another 4B model.

The value head remains float32; the default backbone is bfloat16. Native text
forward returns only final hidden states, not every layer's hidden states. Vocabulary
logits are materialized only at sampled-token prediction positions and in small
chunks. Gradient checkpointing recomputes those projection chunks during backward,
rather than retaining a full prompt-length by vocabulary tensor. Backbone gradient
checkpointing is configurable. Adapter/head/optimizer/RNG checkpoints omit base weights.

## Sampling and policy-ratio integrity

Generation uses the same text backbone as training, one action at a time, with a
cache within the action. It does not use top-p, top-k, repetition penalties, hidden
logit processors, or constrained decoding. Temperature is included in BOTH sampling
and recomputed probabilities. The official tokenizer chat template disables thinking
with `enable_thinking=False`; `/nothink` is not inserted as an invented switch.

Each turn stores exact prompt/response token IDs and actual sampling log probabilities.
The trainer does not decode and retokenize a sampled action to calculate its probability.
Decoded action text is only for the simulator's JSON parser and subsequent observed
history. EOS is retained in the action's training tokens when actually sampled.

Before each PPO batch, the first turn is checked for agreement between cached
sampling and a fresh full forward pass. The tolerance is configurable because
bfloat16/kernels can differ numerically. A failure should trigger inspection of model
loading, positional handling, kernels, and precision, NOT automatic tolerance inflation.
All dropout is disabled; training mode is used only to activate checkpointing.

## Deliberate efficiency tradeoffs

Actor rollouts are currently sequential. Judge evaluations are concurrent and pooled.
Microbatching processes one variable-length turn at a time and accumulates gradients
over a token-weighted minibatch, avoiding padding and hybrid recurrent-state ambiguity.
Context prefixes are recomputed between tool actions and during PPO epochs. This
favors readability and a clean sampling/update match over maximum GPU utilization.
It does not claim vLLM-level throughput, cross-turn KV reuse, multi-GPU scaling,
4-bit fitting, or overlap between actor generation and optimization.

Potential next optimizations are batched same-policy rollouts, length-bucketed padded
training, and a vLLM actor worker. Each needs sampling-probability/cache consistency
checks before use in PPO. Do not mix a quantized inference actor with unquantized
training log probabilities and call the samples exactly on-policy.

Qwen's hybrid stack can use optional `causal_conv1d` and `fla` kernels. Without suitable
kernels the reference implementation can be slower and use more memory. Install
compatible versions only after the basic smoke check, repeat forward/gradient/cache
checks, and record the chosen stack. No kernel speed or memory claim was measured here.

## TypeSafe HTTP contract

`judges.py` posts to `https://api.typesafe.ai/v1/systemone` with bearer authorization
and `model`, `state`, and a named `questions` map. It reads `answers`, not OpenAI-style
`choices`. Binary questions use `type: noul` and return `noul`; progress uses `score`
with ordered `criteria` and a string-indexed probability map. The normalized expected
utility is computed from that map and checked against the returned numeric score.

The comparator posts to a separately configured `/chat/completions` endpoint with the
same semantic rubric and evidence, requesting JSON mode. For the default Qwen3.5
comparator it disables thinking through chat-template arguments. Returned probabilities
are validated; truncated, malformed or inconsistent replies abort rather than becoming
rewards. The harness does not claim those self-reported values are calibrated.

Rate limits/overload/transient server or transport errors receive bounded exponential
backoff with jitter and `Retry-After` handling. Authentication/validation/schema errors
fail rather than silently changing providers, dropping trajectories, returning zero,
or falling back to oracle labels. Retry randomness is separate from research RNGs.

The conservative request-byte guard is NOT an exact TypeSafe token counter. Provider
context rejection still fails explicitly. No request is silently truncated. Credentials
are excluded from cache keys and logs; only synthetic task evidence is sent remotely.

The run-local SQLite cache keys on provider, endpoint, requested model, rubric version,
and exact payload. Raw attempt records include requests, responses, usage, status,
latency, and estimated cost. Cache reuse cannot make a moving model alias reproducible;
a frozen identifier and archived responses are still necessary.

## Artifacts

Each run writes `manifest.json`, `config.json`, `metrics.jsonl`, training/evaluation
trajectory JSONL, per-provider request/cache directories, checkpoints, and a final
`summary.json`. Failure writes `failure.json` and propagates; no successful result is
invented. A checkpoint directory is published by rename only after its files are written.

Resume restores only completed updates into a new directory and requires the same
configuration. The initial reference remains the original frozen Qwen base. The
ancestor's traces/caches must be retained separately. Reports refuse duplicate
arm/seed runs and mixed protocols, preventing accidental pooling of continuations
or different context ablations as independent evidence.
