# Jev as a reward model: Qwen3.5-4B + PPO

A from-scratch research harness for one question: **when an agent learns to maximize
Jev's reward, does independently verified task success improve?**

This branch replaces the working tree with a new implementation; it does not reuse
or modify the implementation on `main`. No training results are claimed. The code
has had static review and CPU-only invariant tests; Qwen loading, CUDA kernels,
live TypeSafe calls, comparator serving, and end-to-end learning remain unverified.

## Experiment

The actor is **`Qwen/Qwen3.5-4B`**, adapted with LoRA. All arms use the same clipped
PPO, shared-backbone learned value head, frozen-base reference policy, task schedule,
action budget, sampling distribution, and evaluation cases.

| Arm | Training reward | Question |
|---|---|---|
| `oracle_terminal` | Exact binary terminal success | Can ordinary grounded PPO learn the task? |
| `jev_terminal` | Jev's terminal probability of joint success | Can Jev replace the ground-truth reward? |
| `jev_shaping` | Exact terminal success + Jev potential differences | Does contextual intermediate feedback help learning? |
| `llm_terminal` | Another judge's terminal probability, same rubric | Is Jev competitive with an ordinary LLM judge? |
| `llm_shaping` | Exact terminal success + other judge's potential differences | Optional matched provider comparison for shaping |

The first four are the default sweep. Include the fifth for a fully matched
provider-by-reward-structure comparison. Do **not** compare `jev_shaping` against
`llm_terminal` and attribute the difference entirely to judge quality.

### Task: simulated order fulfillment

The agent inspects, reserves, packs, and ships 1–3 orders. It must retain destination,
quantity, fragile-packaging, shipping-budget, carrier, and protected-order constraints
across tool interactions. Wrong shipments can execute successfully; only the private
verifier decides whether the user's goal was satisfied. No actual purchases or
shipments occur. The generator guarantees a feasible scripted solution.

Each order normally takes four actions, followed by `finish`. Tools provide mechanical
observations, **not reward labels**. The actor and judges see only the original goal
and observed evidence. The verifier's success/failure labels, partial-credit metrics,
and private order requirements never appear in judge requests.

Training mixes standard, fragile, and budget cases. Evaluation separates new cases,
held-out fragile+budget compositions, and longer 5–7-order episodes. These are
**constraint/horizon shifts within one environment**, not proof of broad cross-domain
reward-model generality.

## Setup

Use Python 3.11+ and a CUDA-enabled PyTorch installation compatible with the GPU.
Create a fresh environment rather than installing into an existing training stack.

```bash
git clone --branch research/jev-qwen35-ppo-from-scratch \
  https://github.com/mihir-s-05/jev-reward-model.git
cd jev-reward-model
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
# Install the appropriate CUDA PyTorch build first when your platform requires it.
python -m pip install -e '.[train,analysis,dev]'
python -m pip freeze > environment.lock.txt
jev-reward doctor --config configs/default.yaml
```

Dependency ranges are compatibility constraints, **not a GPU-validated lockfile**.
The run manifest records exact installed versions, CUDA version, GPU, source commit,
rubric hash, and resolved actor revision. Archive `environment.lock.txt` with results.

The native checkpoint is loaded on CPU; its unused vision tower is discarded before
moving the text backbone to the GPU. Start with a 24–48 GB GPU and adequate host RAM,
then measure actual memory; this is planning guidance, not a tested fit guarantee.
There is no automatic device sharding, 4-bit quantization, or distributed trainer.
Optional Qwen DeltaNet kernels can materially affect speed/memory; see
[implementation notes](docs/implementation.md) before changing kernels mid-experiment.

### Credentials and comparator

```bash
export TYPESAFE_API_KEY='your-typesafe-key'
export JUDGE_BASE_URL='http://localhost:8001/v1'
export JUDGE_MODEL='Qwen/Qwen3.5-9B'
export JUDGE_API_KEY='unused-for-local-server'
```

`.env.example` documents variables but `.env` is **not automatically loaded**. Never
commit keys. The alternative is any compatible chat-completions endpoint supporting
JSON mode; the default comparison uses a separate **Qwen3.5-9B** judge. Its weights
are never updated by this harness. A hosted compatible endpoint avoids needing a
second local GPU. For a separate vLLM environment/GPU, an illustrative server command is:

```bash
CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen3.5-9B \
  --port 8001 --max-model-len 32768 --language-model-only
```

Check the installed server's Qwen3.5 support. Do not colocate it on the actor's GPU
and assume the benchmark still measures equal actor resources. Token prices default
to unknown for the comparator; set both in YAML for hosted pricing, or set both to
zero **and account for a separate server GPU** when self-hosting.

## Suggested execution order

### 1. Inspect the environment; then test the plumbing

```bash
jev-reward demo
jev-reward demo --fault address
python -m pytest

# This next command DOES load the 4B actor and train two small PPO updates on a GPU.
jev-reward train --config configs/smoke.yaml --arm oracle_terminal \
  --seed 0 --out runs/smoke-oracle
```

The smoke configuration is only an integration check. A successful CPU test suite
is not evidence that Transformers, PEFT, kernels, and GPU training work together.

### 2. Audit each judge before paying for full training

```bash
# 10 task instances x 5 controlled trajectories = up to 50 uncached requests/judge.
jev-reward audit --provider jev --cases 10 --out runs/audit-jev
jev-reward audit --provider llm --cases 10 --out runs/audit-llm

# Then check the actual Jev/GPU path on the same small training configuration.
jev-reward train --config configs/smoke.yaml --arm jev_terminal \
  --seed 0 --out runs/smoke-jev
```

Audits include successful, incomplete, wrong-address, protected-order, and unsupported
success-claim trajectories. They report Brier score, AUROC, threshold accuracy, and
calibration bins. Calibration refers only to this deliberately constructed mixture.
Inspect systematic errors; do not tune the rubric on final test cases.

### 3. Run the controlled comparison

```bash
# Four primary arms, each in a fresh subprocess, same three actor seeds.
jev-reward sweep --config configs/default.yaml --seeds 0 1 2 \
  --out runs/comparison

# Optional complete five-arm comparison, preferably 5 seeds for final evidence.
jev-reward sweep --config configs/default.yaml --seeds 0 1 2 3 4 \
  --arms oracle_terminal jev_terminal jev_shaping llm_terminal llm_shaping \
  --out runs/comparison-five
```

`sweep` resolves the actor to one immutable Hub commit before starting and rotates
arm order by seed. The public Jev alias is `jev-latest`; pin a provider-supplied
snapshot identifier when available. A stable alias/response model string does not
prove stable backend weights. Save dates and raw responses and acknowledge that
limitation in results.

By default, validation/final trajectories in learned-reward arms are also scored by
that arm's judge, under a **separate evaluation cost ledger**. `--eval-judge none`
skips these API calls; `--eval-judge jev` measures every arm using the same Jev
observer without feeding those evaluation scores into PPO.

### 4. Evaluate, inspect reward hacking, and report

```bash
jev-reward evaluate \
  --checkpoint runs/comparison/jev_terminal-seed0/checkpoints/update-00100 \
  --split long --episodes 100 --out runs/jev-long-check

jev-reward score-traces --provider jev \
  --input runs/comparison/jev_terminal-seed0/train-trajectories.jsonl \
  --cases 100 --out runs/on-policy-judge-audit

jev-reward report --runs runs/comparison --out runs/report --plots
# Add actual rental assumptions to enable complete estimated-cost curves:
# --gpu-hourly-usd <actor-rate> --llm-server-hourly-usd <separate-judge-rate>
```

Reports contain per-seed curves, seed-bootstrap intervals, paired seed differences,
independent-success versus updates/tokens/wall-time, and judge-score versus actual
success. Missing costs are never silently treated as zero. Cost curves require
complete supplied assumptions; see [the protocol](docs/protocol.md).

### Resume after a completed checkpoint

```bash
jev-reward train --config configs/default.yaml --arm jev_terminal --seed 0 \
  --resume runs/comparison/jev_terminal-seed0/checkpoints/update-00010 \
  --out runs/jev-seed0-continuation
```

Resume requires the same configuration/arm/seed and writes a **new empty directory**.
It restores adapter, value head, optimizer, sampling/shuffle/Torch RNGs, completed
update index, and counters. It does not resume an unfinished rollout/update or copy
prior raw traces/caches. Retain the ancestor run; report only one completed run per
arm/seed. Backend nondeterminism or a moving hosted judge can still prevent exact replay.

## Repository map

- `src/jev_reward/env.py`: generator, strict tool interface, observed-state memory, private verifier.
- `judges.py`, `rewards.py`: TypeSafe/LLM adapters, frozen rubric, cache/accounting, reward arms.
- `policy.py`, `ppo.py`: native Qwen loading, LoRA/value head, exact sampling, token PPO/GAE.
- `experiment.py`, `cli.py`, `analysis.py`: runs, checkpoints, audits, paired reporting.
- `configs/`: complete-default and small GPU integration configurations.
- `tests/`: ten inexpensive invariant tests; no model downloads or paid requests.
- `docs/protocol.md`: what comparisons mean and what they cannot establish.
- `docs/implementation.md`: design choices and known efficiency limits.
- `docs/review.md`: review scope, issues addressed, and unverified integration surfaces.

## References

API integration follows [TypeSafe's HTTP reference](https://docs.typesafe.ai/api),
not an assumed SDK or OpenAI-compatible Jev endpoint. See also
[Score](https://docs.typesafe.ai/primitives/score),
[Qwen3.5-4B model card](https://huggingface.co/Qwen/Qwen3.5-4B),
[Transformers Qwen3.5](https://huggingface.co/docs/transformers/model_doc/qwen3_5),
[PEFT LoRA](https://huggingface.co/docs/peft/package_reference/lora),
[PPO](https://arxiv.org/abs/1707.06347), and
[GAE](https://arxiv.org/abs/1506.02438).
