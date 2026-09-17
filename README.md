# Jev as a Reward Model

Research harness for testing whether TypeSafe's **Jev** can act as a reusable reward source for PPO training of **Qwen3.5-4B**.

The experiment separates four conditions:

1. **Grounded terminal reward + PPO** — simulator reward only.
2. **Jev terminal reward + PPO** — Jev is the only terminal reward source.
3. **Grounded reward + Jev potential shaping + PPO** — exact task reward plus dense Jev progress shaping.
4. **Frozen Qwen judge + PPO** — a local general-purpose judge using the same base checkpoint as an alternative evaluator.

All four use the same actor architecture, task distribution, PPO implementation, critic, seeds, and rollout budget. The exact simulator reward is always retained for evaluation, even when it is hidden from training.

## Task

The environment is a deterministic long-horizon constraint-following workflow. Each episode gives the agent an ordered list of 3–5 milestones, persistent constraints, and an exact terminal completion string. The policy emits one textual action per turn. `DO <milestone>` performs a milestone. Milestones only count in order. Constraint violations are irreversible. A hidden simulator therefore gives us an exact success signal while Jev sees only the public task specification and action history.

This task is intentionally controlled rather than realistic. It tests the core reward-model claim with minimal confounding: can a general evaluator recognize successful long-horizon behavior from trajectory context, and does optimizing that evaluator improve actual task success?

## Jev integration

The client follows TypeSafe's System One HTTP API:

```text
POST https://api.typesafe.ai/v1/systemone
Authorization: Bearer <API_KEY>
```

Requests contain `state`, `model: "jev-latest"`, and a map of typed questions. This repo uses a **noul** question for terminal success and a five-level **score** question for progress. The documented score is a probability-weighted level index, so the `[0, 4]` progress score is normalized to `[0, 1]`.

```bash
export TYPESAFE_API_KEY=...
```

Identical Jev requests are cached by SHA-256 under `.cache/jev/`. The client retries only TypeSafe's documented transient `429` and `529` responses with exponential backoff.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

Qwen3.5-4B is loaded from `Qwen/Qwen3.5-4B`. The policy uses LoRA adapters; the critic is a one-layer scalar value head over the actor's final prompt hidden state.

## Generate one fixed dataset

```bash
python -m jev_reward_model.data \
  --out data/tasks.jsonl \
  --n-train 512 \
  --n-eval 128 \
  --seed 7
```

The final 128 tasks are reserved by the evaluation script. For a publication-quality follow-up, hold out whole task templates as well as random seeds.

## Train

```bash
python -m jev_reward_model.train --config configs/grounded.yaml
python -m jev_reward_model.train --config configs/jev_terminal.yaml
python -m jev_reward_model.train --config configs/jev_shaping.yaml
python -m jev_reward_model.train --config configs/qwen_judge.yaml
```

The implementation keeps PPO explicit so reward semantics remain inspectable:

1. collect complete episodes;
2. freeze rollout log-probabilities and critic values;
3. compute generalized advantage estimates (GAE);
4. normalize advantages across the rollout batch;
5. apply the clipped PPO objective to the LoRA actor and MSE value loss to the learned critic.

The critic is deliberately policy-specific. Jev supplies reward; it is **not** treated as the PPO value function.

### Jev potential shaping

The shaping condition uses

```text
r'_t = r_t + alpha * (gamma * Phi(s_{t+1}) - Phi(s_t))
```

where `Phi` is Jev's normalized progress score. Terminal potential is forced to zero. This tests dense contextual evaluation without repeatedly rewarding the policy merely for occupying a high-scoring partial state.

## Evaluate

```bash
python -m jev_reward_model.evaluate --run-dir runs/grounded
```

Evaluation ignores the training reward and uses simulator truth. It records exact success, milestone progress, irreversible-violation rate, and complete action traces. The most informative failures are trajectories with high learned/Jev reward but simulator failure.

## Recommended first experiment

Run 3–5 seeds per condition and keep every parameter except reward source identical. Plot simulator success versus PPO update, simulator success versus evaluator cost, evaluator reward versus simulator success, and constraint-violation rate versus update.

If Jev terminal reward rises together with grounded simulator success, that supports Jev as a reusable reward source. If Jev reward rises while simulator success plateaus or falls, the primary finding is reward overoptimization. The shaping condition tests whether frequent contextual judgments improve sample efficiency beyond terminal Jev evaluation.

## Caveats

- The environment is synthetic. Positive results justify moving to code/tool-use environments; they do not establish universal reward modeling.
- The frozen-Qwen judge is an alternative evaluator baseline, not a perfectly cost-matched hosted-service comparison.
- Jev and the actor see textual evidence. This measures semantic trajectory judging, not verification against external systems.
- Any fixed judge can eventually be exploited. Evaluation must remain grounded in simulator truth.

## Tests

```bash
pytest -q
```

Tests intentionally cover only high-value invariants: ordered milestones, irreversible violations, and the potential-shaping equation. This is an experimentation repository, not production infrastructure.
