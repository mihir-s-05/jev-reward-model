# Review and validation record

## Performed

- Read the TypeSafe HTTP reference and Qwen3.5/PEFT documentation.
- Reviewed generated-token indexing, policy/value losses, reference probabilities,
  action-boundary discounting, complete-episode GAE, and zero-terminal-potential algebra.
- Reviewed the boundary between private simulator truth and public evidence. Terminal
  Jev rewards deliberately remain wrong when a mocked judge is wrong; the oracle does
  not silently repair them.
- Reviewed mechanical acceptance versus semantic success, protected mutations that
  are later undone, strict action fields/types, finite action budgets, and observable
  history construction for the full/recent/ledger evaluator ablations.
- Compiled all source/test modules and ran **10 CPU-only invariant tests**, all passing.
  The tensor tests perform only tiny algebra/backward checks, not language-model training.
- Executed the CPU-only scripted demo with a wrong-address shipment and confirmed it
  is mechanically accepted but fails the independent verifier.

## Corrections made during review

Malformed provider JSON objects fail cleanly before usage accounting; partial LoRA
checkpoint loads are rejected; the selected CUDA device is established before the
bfloat16 capability check; frozen embeddings are explicitly excluded from adapter
checkpoints; endpoint/model environment overrides are resolved into the saved config;
paired differences use matching training seeds; prior training API spend is carried
separately across resumed runs rather than silently reset.

## Not performed

No Qwen checkpoint download or inference, GPU training, live TypeSafe request, live
comparator request, throughput/memory benchmark, kernel numerical comparison, or
end-to-end learning experiment was run. Transformers and PEFT were not installed in
the review runtime. The native-loader and training integration are code-reviewed,
not empirically validated. Do not interpret passing CPU tests as those validations.

The most important next gates are `doctor`, a single-order oracle smoke run, the
scripted judge audit, a Jev terminal smoke run, and then an oracle learning pilot.
Check the runtime's sampled-versus-recomputed probability guard and inspect loading
warnings before trusting a PPO learning curve. Repeat those checks after changing
Transformers, PEFT, precision, attention kernels, or serving infrastructure.

## Intentional scope limits

Single-GPU bfloat16 LoRA, sequential actor rollouts, no SFT warm-start, no dynamic
rubric learning, no asynchronous actor/learner staleness, and no automatic hardware
or hyperparameter search. The five reward arms, private outcome evaluation, context
ablations, audit/reporting commands, and checkpoint continuation are implemented.
No benchmark results, universal calibration, or arbitrary long-horizon generality
are claimed. Hosted-model immutability cannot be guaranteed by this client.
