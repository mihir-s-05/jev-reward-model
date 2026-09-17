# Experimental protocol

## Primary hypotheses

1. **Reward substitution:** `jev_terminal` improves independently verified success
   relative to its own untrained policy, and approaches `oracle_terminal` at a
   specified compute/data budget.
2. **Useful contextual shaping:** `jev_shaping` improves sample efficiency over
   `oracle_terminal`, not merely the reported scalar reward.
3. **Provider efficiency:** compare `jev_terminal` with `llm_terminal` at equal
   learning budgets and measured costs. Compare shaping providers only when the
   optional `llm_shaping` arm is included.

There are no claimed results. All hyperparameters are initial experimental choices,
not values selected from successful runs. Start by checking initial success and
learning on a single-order oracle configuration shared by every arm. Zero oracle
learning could indicate exploration, protocol, or implementation failure; it would
not by itself show that Jev is an ineffective evaluator.

## What is held constant

Actor checkpoint commit; LoRA rank/target modules; initial seed/value head; PPO
hyperparameters; exact task index schedule for each seed; maximum action budget;
full-history actor representation; sampling temperature; rollout batch size;
frozen-base KL regularization; and held-out cases. Initial adapters are no-ops and
the value head is zero-initialized. Neither judge supplies demonstrations or actions.

Every training seed sees identical task IDs across arms. Validation and test tasks
use a distinct fixed evaluation seed. Judge retries use independent entropy and
cannot perturb actor sampling or task construction. Evaluation is greedy and does
not consume the dedicated categorical-sampling RNG.

Equal updates/episodes are NOT equal compute: different policies can produce
longer trajectories. Plot independent success against generated tokens and wall
time as well as update count. The harness does not automatically stop every arm at
an identical FLOP or wall-clock budget.

## Task and evaluation boundaries

`train`, `validation`, and `test` contain standard, fragile, and budget cases.
`composition` combines fragile and budget constraints unseen jointly in training.
`long` uses this combined family with a larger order count. These are within-domain
compositional/horizon tests, not unrelated held-out domains. The task is synthetic;
its purpose is isolation and cheap ground truth, not a claim of real warehouse fidelity.

The private verifier checks all target shipments, exact quantities/SKUs/destinations,
required carriers, fragile packaging, protected-order mutations over the entire
history, total spend, and remaining reservations. Success is binary. Extra private
metrics diagnose failures but never become shaping rewards.

Only mechanical preconditions are enforced by tools. The tool can report an
accepted shipment that violates the goal. Both actor and evaluator must consult
requirements rather than mistake API acceptance for completion.

Final test/composition/long evaluation uses the final checkpoint, not a checkpoint
selected by inspecting final test scores. Periodic validation is logged but does not
automatically change hyperparameters, choose checkpoints, or enter the PPO update.
Three seeds are a starting point; five or more provide more useful uncertainty.
Bootstrap intervals and paired differences resample training seeds. With one seed,
no seed-uncertainty interval is manufactured.

## Rewards and PPO semantics

For a terminal-only judge, the last environment transition receives the judge's
probability that **all** goal conditions are met. Do not multiply marginal success
and compliance probabilities; independence has not been established. The compliance
and progress outputs are retained as diagnostics.

For shaping, the fixed judge produces a potential `Phi(h_t)` in [0,1] at every
nonterminal observed-history boundary. The probability distribution over five
ordered descriptions is mapped to utilities `[0,.25,.5,.75,1]`. This is a specified
heuristic utility, not an empirically calibrated distance to completion.

```
r_shaped[t] = r_oracle[t] + alpha * (gamma * Phi[t+1] - Phi[t])
Phi[T] = 0
```

A boundary is evaluated once and its value reused in both adjacent differences.
The final potential is forced to zero, including unsuccessful closure. Consequently:

```
sum_t gamma**t * shaping[t] = -alpha * Phi[0]
```

This does not create new terminal outcome information or automatically solve causal
credit assignment. With complete Monte Carlo returns and an exact consistently
shifted value function, potential shaping does not change the policy advantage.
Any learning benefit here must come through approximate value learning and finite
optimization. With default `gamma=1`, the total shaped return differs from the
oracle return only by a task-initial offset. A larger total training reward alone
is therefore not evidence of an improvement.

The implementation performs token-level PPO, not a clipped likelihood ratio for an
entire JSON action. Generated tokens have an autoregressive state/action transition;
the deterministic tool result then leads to the next observed-history prompt. Prompt
and tool-result tokens receive no actor/value loss. Values are taken from the hidden
state **before** each sampled token, not from the token after observing the action.

Inside one JSON action, discount and GAE continuation equal one. At a tool boundary,
they equal configured `gamma` and `lambda`. The final bootstrap is zero. This prevents
arbitrary extra discounting of longer spellings. KL uses sampled old-policy log ratios
against the frozen initial base model and is fixed during PPO epochs. It is a sample
estimate and can be negative on individual tokens/rollouts.

Advantages are standardized across the full on-policy batch; losses are weighted
by generated-token count in each minibatch. Old probabilities/values stay frozen
for all PPO epochs. Value clipping, gradient clipping, and approximate-KL early
stopping are applied. KL regularization is the same in every arm.

## Closure and truncation

`finish`, exhausting the finite action budget, and exhausting the configured context
budget all close the task; there is no promised continuation beyond that budget.
The original action budget is visible to the policy/judge. A context-budget closure
is additionally visible in the final snapshot. A prompt that does not fit before any
action is a configuration error, not an episode silently removed from the dataset.

Reaching the per-action generation-token cap does not manufacture an EOS. All sampled
tokens are retained. Malformed/truncated JSON produces a mechanical tool error and
consumes an action; a complete valid JSON call may execute even without EOS.

Changing closure conventions changes the task. A future implementation that truncates
only for collector convenience MUST bootstrap its value and handle nonzero remaining
potential; it cannot reuse these terminal conventions unchanged.

## Context and uncertainty ablations

`judge.context` can be `full`, `recent`, or `ledger`; the actor always sees full observed
history. Every mode keeps the original goal and tool contract. `recent` intentionally
loses old evidence. `ledger` deterministically retains observed requirements, every
accepted mutation (including later-undone changes), shipments, and outstanding work.
It never reads hidden requirements or oracle verdicts. Run these as separate,
otherwise-matched experiments; the report rejects accidental mixed configurations.

No confidence-weighted reward is used. TypeSafe's confidence statistic does not prove
correctness or out-of-distribution awareness. The ordinary LLM's probabilities are
self-reported numerical judgments, not necessarily calibrated predictive probabilities.

## Required measurements

Report independent success, failure categories, invalid actions, episode length,
actor generated tokens, rollout/reference/optimization/judge wall time, request counts,
cache hits, observed token usage, known estimated API spend, and unknown-cost requests.
Track held-out judge probability/Brier score alongside exact success. Increasing
surrogate reward without increasing actual success is the central warning signal.

Use `score-traces` to score actual current-policy behavior, not only scripted examples.
The scorer extracts `trajectory` and never passes the accompanying `oracle` record.
Raw request/response logs allow manual auditing of confident wrong scores and drift.

The initial audit's five trajectory types deliberately change outcome prevalence.
Its Brier/ECE measurements apply only to that mixture. They cannot establish universal
calibration, and optimizing against the model can change the trajectory distribution.

## Costs and reproducibility

Training and evaluation use separate judge instances, caches, and logs. Training
wall time includes waits while the actor GPU remains allocated; initial/final evaluation
is separate. Cache hits issue no new HTTP request. Raw failed attempts are logged;
missing usage or unknown prices are marked unknown, never silently zeroed. Recorded
`known_cost_usd` is a configured-price estimate, not an invoice.

A complete estimated training-cost curve requires an actor hourly rate and known
request costs. LLM arms additionally require a separate judge-server hourly rate;
set that rate explicitly to zero for a fully hosted comparator whose costs are
already captured by token pricing. For self-hosting, set token prices explicitly to
zero and supply the server rental rate. Server time is conservatively modeled as
reserved throughout training, not just generation calls. Loading, checkpoint I/O,
evaluation, external idle time, and retry billing without usage are not silently
included in this training-only estimate. Preserve these exclusions when reporting.

The actor is pinned once per sweep. The public `jev-latest` alias may move even when
its response identifier does not. The client detects identifier changes within a run,
but cannot detect undisclosed backend updates. The remote LLM likewise needs a pinned
model/server revision. Archive exact dependency versions, all configuration, code
revision, request logs, timestamps, and checkpoint lineage. No training credentials
are written to these artifacts.
