# Jev as a Reward Model

Research harness for testing whether TypeSafe's **Jev** can act as a reusable reward source for PPO training of **Qwen3.5-4B**.

The experiment intentionally separates four questions:

1. **Grounded terminal reward + PPO** — simulator reward only.
2. **Jev terminal reward + PPO** — Jev is the only terminal reward source.
3. **Grounded reward + Jev potential shaping + PPO** — exact task reward plus dense Jev progress shaping.
4. **Alternative judge + PPO** — same interface as Jev, implemented locally with Qwen as a cost-comparable judge baseline.

The environment is a deterministic, long-horizon constraint-following workflow. An episode gives the model a goal, persistent constraints, and a sequence of required milestones. The policy emits one textual action at a time. The simulator parses those actions and tracks irreversible violations, completed milestones, and terminal success. This provides an exact hidden reward for evaluation while letting training use any reward source.

## Why this task

The environment is deliberately simple enough to audit but difficult enough to test the hypothesis. A reward model must recognize progress, remember constraints introduced earlier in the trajectory, and distinguish superficially plausible actions from actions that actually satisfy the task. The hidden simulator reward lets us measure reward hacking directly.

## Jev API

The client calls `POST https://api.typesafe.ai/v1/systemone` with `model="jev-latest"`, a structured `state`, and typed questions. Jev supports `noul`, `choice`, and `score` questions; this repo uses `score` for progress and `noul` for terminal success. Set:

```bash
export TYPESAFE_API_KEY=...
```

Jev calls are cached on disk by request hash so rerunning an experiment does not repay for identical judgments.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

For Qwen3.5-4B, use a recent Transformers release. The code defaults to LoRA PPO to keep the experiment practical on a single modern GPU.

## Run

Generate a fixed dataset:

```bash
python -m jev_reward_model.data --out data/tasks.jsonl --n-train 512 --n-eval 128 --seed 7
```

Train one condition:

```bash
python -m jev_reward_model.train --config configs/grounded.yaml
python -m jev_reward_model.train --config configs/jev_terminal.yaml
python -m jev_reward_model.train --config configs/jev_shaping.yaml
python -m jev_reward_model.train --config configs/qwen_judge.yaml
```

Evaluate all checkpoints with the exact simulator reward:

```bash
python -m jev_reward_model.evaluate --run-dir runs/grounded
```

Results are written as JSONL plus a compact summary JSON. The key metrics are exact task success, irreversible-violation rate, reward-model score, and the correlation/gap between model reward and exact simulator success.

## Experimental discipline

- Use the same generated tasks, seeds, rollout count, and PPO hyperparameters across conditions.
- Keep the Jev rubric fixed during a run.
- Never expose simulator internals or exact reward fields to Jev or the actor.
- Evaluate every checkpoint with the hidden simulator reward.
- Inspect trajectories with high Jev reward but low simulator reward; those are the most informative failures.
- For a stronger follow-up, hold out whole task templates rather than only seeds.

## Repository layout

- `env.py`: deterministic workflow simulator and parser.
- `data.py`: reproducible task generation.
- `jev.py`: minimal TypeSafe HTTP client, retry, and cache.
- `rewards.py`: four reward conditions.
- `rollout.py`: Qwen policy interaction with the environment.
- `train.py`: PPO loop using TRL primitives and LoRA adapters.
- `evaluate.py`: independent simulator-grounded evaluation.
- `tests/`: small tests for simulator semantics and reward shaping.

This is a research repo, not a production service. The code favors inspectability and experimental control over abstraction.