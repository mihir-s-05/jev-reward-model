# Jev as a reward model

An inspectable PPO experiment with **Qwen/Qwen3.5-4B** as the actor. The question is whether a frozen, task-conditioned evaluator provides rewards that improve **independently measured task success**, including on unseen dependency structures and longer trajectories.

**Implementation status:** reviewed code with CPU unit tests covering the actor device path, PPO, simulator, and mocked judges. No full Qwen 4B training results, live Jev/vLLM requests, or hardware throughput numbers are claimed. Run actor preflight and judge audits on the target machine before committing a training budget. [Review and remaining runtime checks](docs/review.md).

## Comparisons

| Config | Training reward | Purpose |
|---|---|---|
| `grounded` | Exact terminal success | Oracle-reward PPO baseline |
| `jev_terminal` | Jev's terminal success probability only | Can Jev replace the oracle reward? |
| `jev_shaping` | Exact terminal success + Jev potential differences | Does frequent evaluation improve learning? |
| `qwen_judge` | Frozen Qwen's terminal success estimate only | Alternative general evaluator |
| `qwen_shaping` | Exact terminal success + frozen Qwen potential differences | Matched shaping control |

The fifth arm extends the original four-way comparison so evaluator identity and shaping can be separated. All use the same actor initialization, LoRA, policy-specific critic, PPO settings, task stream, and update budget for each seed. Qwen judging runs on a **separate frozen server**, never the changing actor. Its generated probabilities are not assumed calibrated.

The default study is a controlled diagnostic, not evidence of a universal reward model. A fixed update budget is not a fixed token or dollar budget; the report aligns completed evaluations at common measured budgets.

## Task: dependency workflows

An agent receives a shuffled catalog of operations, prerequisites, permanent prohibitions, and an exact completion command. It emits one command per turn:

```text
DO verify_invoice_<opaque-id>
FINISH <episode-id>
```

The simulator applies, rejects, or ignores the action and returns a factual receipt. Finishing early, executing a forbidden command, or exhausting the budget without a correct finish fails. A rejected prerequisite attempt can be retried. Forbidden actions remain disqualifying even after subsequent valid work. No generated code is executed and no external systems are modified.

Operation IDs are random, not topological indices. The actor must use the dependency graph rather than sort names. Stored topological order and computed success/progress/violation labels never enter evaluator requests. Actor and judge get the same public rules and authoritative event evidence, not the actor's self-reported accomplishments.

| Split | Dependency families | Required operations |
|---|---|---|
| Train | Chain, fork/join, two chains | 4, 8, 12 |
| Validation / in-distribution test | Same families, new instances | 4, 8, 12 |
| Structural test | Barrier, overlapping dependencies | 4, 8, 12 |
| Longer-horizon test | Training families, new instances | 20, 28 |

This tests temporal constraints and contextual evaluation without sandbox/verifier ambiguity. It does **not** test coding, open-world verification, multimodal reasoning, or hundreds of agent interactions. Check untrained performance first: a near-perfect baseline leaves no useful learning signal, while zero success may need a shared curriculum or warm-start in a subsequent, explicitly separate experiment.

## Setup

Use Python 3.11+. CUDA is the default actor device (`device: cuda:0`, `dtype: bfloat16`). CPU is a first-class alternative for the **full** train/eval path (`device: cpu`, `dtype: float32`) — not a smoke-only mode. Install an appropriate PyTorch build for the machine (CUDA or CPU) before the training extras. The reviewed model integration pins Transformers 5.17.0 and PEFT 0.21.0.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[train,dev,plots]'
pytest -q
python -m jev_reward_model.data --out-dir data --seed 7

# Resolve ONCE and reuse this same commit for every actor and the frozen judge.
export QWEN_REVISION="$(python -c 'from huggingface_hub import HfApi; print(HfApi().model_info("Qwen/Qwen3.5-4B").sha)')"
python -m jev_reward_model.preflight --config configs/grounded.yaml \
  --model-revision "$QWEN_REVISION"
```

The preflight loads the real model on the configured device, compares batched cached generation probabilities with the uncached PPO scoring path, and checks finite, nonzero actor/critic gradients. It makes no optimizer step or API call. Investigate a failure; do not simply relax its likelihood tolerance.

For dataset creation and judge-only auditing, `pip install -e .` suffices: no training libraries or actor GPU are required. `.env.example` documents variables; files are **not automatically loaded**. Export credentials in your shell, never in YAML or committed code.

### CPU actor (full pipeline)

CPU runs use the same modules as CUDA: preflight, rollouts, command-level PPO, evaluation, checkpointing, and resume. They are much slower (4B float32 generation and backward on CPU). Do not stub out PPO or skip rollouts; reduce `updates` / `eval_tasks` only if you are debugging, not for a matched study.

```bash
python -m jev_reward_model.preflight --config configs/grounded_cpu.yaml \
  --model-revision "$QWEN_REVISION"
python -m jev_reward_model.evaluate --config configs/grounded_cpu.yaml \
  --model-revision "$QWEN_REVISION" --tasks data/validation.jsonl \
  --output-dir runs/base-validation-cpu
python -m jev_reward_model.train --config configs/grounded_cpu.yaml \
  --seed 0 --model-revision "$QWEN_REVISION" \
  --output-dir runs/grounded-cpu/seed-0
```

`configs/grounded_cpu.yaml` is the example full-run CPU config (`device: cpu`, `dtype: float32`). Any other arm can be run on CPU by adding the same two keys (or omitting `dtype`, which then defaults to `float32` when `device` is `cpu`). Explicit `dtype: bfloat16` with `device: cpu` is rejected. CUDA configs are unchanged.

**Reward-arm limits on CPU:** `grounded` needs only the actor device. `jev_terminal` / `jev_shaping` work with a CPU actor plus the remote Jev HTTP API (`TYPESAFE_API_KEY`). `qwen_judge` / `qwen_shaping` still need a **separate frozen Qwen judge server** (vLLM or equivalent) — the actor being on CPU does not serve that judge, and this harness does not load a second in-process Qwen as a judge.

### Jev

```bash
export TYPESAFE_API_KEY='your-key'
python -m jev_reward_model.audit --backend jev \
  --tasks data/validation.jsonl --limit 8 --output-dir audits/jev-preflight
```

This explicitly makes paid/API requests: six counterexample traces per task, with terminal and progress questions together. Inspect `scores.jsonl`, `summary.json`, and raw `judge_requests.jsonl` before training. The client follows the [TypeSafe HTTP API](https://docs.typesafe.ai/api), not an inferred SDK interface.

### Frozen Qwen alternative

Serve the **unmodified** same pinned checkpoint with a compatible vLLM installation in a separate environment/device or another host. Do not install a second serving stack into the pinned training environment without checking dependencies. Example server command after installing vLLM according to its documentation:

```bash
# Run on the judge machine/GPU; use the SAME exported checkpoint revision.
vllm serve Qwen/Qwen3.5-4B --revision "$QWEN_REVISION" \
  --served-model-name Qwen/Qwen3.5-4B --host 127.0.0.1 --port 8001 \
  --dtype bfloat16 --max-model-len 8192
```

The client defaults to `http://localhost:8001/v1/chat/completions`. Change `qwen_judge_endpoint` for a remote deployment; use authentication/TLS rather than exposing an unauthenticated server publicly. `QWEN_JUDGE_API_KEY` is optional for an authenticated endpoint. Record the server package version and launch command with your experiment. The configured revision is provenance supplied by you, not a cryptographic check of server weights.

```bash
python -m jev_reward_model.audit --backend qwen \
  --qwen-judge-revision "$QWEN_REVISION" \
  --tasks data/validation.jsonl --limit 8 --output-dir audits/qwen-preflight
```

Both evaluators receive the same rubric. Qwen uses JSON-schema structured output with thinking disabled; malformed/truncated responses fail the experiment rather than produce invented rewards. A live server smoke test remains necessary. Do not assume actor and judge fit simultaneously on one GPU. Jev arms need only the actor device; the Qwen arms additionally need serving capacity.

## Run the experiment

First measure the untrained actor on validation, not the reserved test sets:

```bash
python -m jev_reward_model.evaluate --config configs/grounded.yaml \
  --model-revision "$QWEN_REVISION" --tasks data/validation.jsonl \
  --output-dir runs/base-validation
```

Train one arm and seed:

```bash
python -m jev_reward_model.train --config configs/jev_terminal.yaml \
  --seed 0 --model-revision "$QWEN_REVISION" \
  --qwen-judge-revision "$QWEN_REVISION" --output-dir runs/jev_terminal/seed-0
```

Or, after both evaluator preflights, launch all five arms across three seeds:

```bash
bash scripts/sweep.sh
```

That command starts **15 training runs** and makes evaluator requests; it is not a dry run. It rotates arm order across seeds to reduce simple time-order confounding. Stop/adjust the study based on validation diagnostics before spending the complete budget. Use the same changes across matched arms. Defaults are a starting protocol, not tuned or validated hyperparameters.

Every run rejects a nonempty output directory. Configs are strict; typos and old scaffold keys fail. Data splits are independent files, explicitly tagged, and checked for overlap. Checkpoints save actor adapters, critic, optimizer, RNG states, dataset hashes, resolved config, and cumulative accounting. Evaluate a saved run with:

```bash
for split in test_id test_ood test_long; do
  python -m jev_reward_model.evaluate --run-dir runs/jev_terminal/seed-0 \
    --tasks "data/${split}.jsonl" \
    --output-dir "runs/jev_terminal/seed-0/eval-${split}"
done
```

Repeat for every arm/seed, using final or **validation-selected** checkpoints consistently. Do not choose checkpoints from test results. `--checkpoint PATH` overrides the latest checkpoint within a saved run. Independent evaluation uses only simulator truth and makes no judge calls.

### Resume

Resume a **trusted local** checkpoint into a **new** output directory. Use its saved resolved config, increasing `updates` to the desired total if necessary:

```bash
python -m jev_reward_model.train \
  --config runs/jev_terminal/seed-0/resolved_config.yaml \
  --resume runs/jev_terminal/seed-0/checkpoint-000050 \
  --output-dir runs/jev_terminal/seed-0-resumed
```

Dataset hashes and all substantive settings must match. The cache is copied, RNG/optimizer state restored, and earlier costs carried forward. Old logs remain untouched; elapsed cost includes repeated validation after resuming. Use one run segment per seed in the report, not the original and its continuation as separate seeds. Loading optimizer checkpoints uses PyTorch's trusted serialization path: never load an untrusted checkpoint.

## Audits and reporting

Audit the **last** actual rollouts, not just synthetic cases or the easy initial policy:

```bash
python -m jev_reward_model.audit --backend jev \
  --traces runs/jev_terminal/seed-0/train_traces.jsonl --last --limit 128 \
  --views full recent ledger --output-dir audits/jev-late-policy

python -m jev_reward_model.report runs/{grounded,jev_terminal,jev_shaping,qwen_judge,qwen_shaping}/seed-* \
  --out-dir reports/comparison
```

The report writes checkpoint CSVs, reward-vs-ground-truth alignment, and success curves versus environment steps, action tokens, elapsed time, and (when supplied) estimated total dollars. It rejects mismatched nuisance configs/data. Budgets use only evaluations completed at or below that budget, never interpolated future results. Seed standard errors are descriptive, not significance tests.

Set `actor_gpu_usd_per_hour` and `qwen_judge_gpu_usd_per_hour` in **all matched configs** to your actual reserved-device rates. Unknown prices remain unknown; local inference is not labeled free. Jev's default `$0.042/M input tokens` is a configurable launch-price assumption, **not a verified invoice**. Provider usage distinguishes billed requests from replay/cache usage; retries/errors may have unreported billing. Actor and local-judge device charges cover elapsed reserved time, including waits/evaluation, but not unrelated pre-study deployment time.

`judge_view: full` is the primary experiment. `recent` intentionally removes older evidence; `ledger` retains every syntactic DO/FINISH attempt but drops inert narration. Actor context remains full in every condition. Offline view audits do not establish that training with that view improves performance; train separate matched studies to test that claim.

## Implementation map

| Module | Responsibility |
|---|---|
| `env.py`, `data.py` | Deterministic simulator, hidden labels, disjoint datasets |
| `policy.py`, `preflight.py` | Actual Qwen loading, language-only LoRA, generation/likelihood/critic (CUDA or CPU) |
| `ppo.py`, `rollout.py` | Complete episodes, GAE, command-level clipped PPO, gradient accumulation |
| `judge.py`, `jev.py`, `rewards.py` | Shared rubric, validated HTTP/cache/audit, five reward conditions |
| `train.py`, `evaluate.py` | Training/checkpoints and independent evaluation |
| `audit.py`, `report.py` | Counterexamples, calibration, context ablations, common-budget comparison |
| `config.py`, `utils.py` | Strict settings, JSON, fingerprints, provenance |

See [design and interpretation](docs/design.md) for the objective, invariants, artifacts, and limitations. Version 2 replaces the earlier scaffold; its old datasets/configs/checkpoints are not compatible. This is intentionally a small single-actor-device research harness, not distributed production RL infrastructure.
