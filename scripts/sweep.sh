#!/usr/bin/env bash
# Run from the repository root. No server, account, or paid job is launched automatically.
set -euo pipefail
: "${TYPESAFE_API_KEY:?Set TYPESAFE_API_KEY}"
if [[ -z "${QWEN_REVISION:-}" ]]; then
  QWEN_REVISION="$(python -c 'from huggingface_hub import HfApi; print(HfApi().model_info("Qwen/Qwen3.5-4B").sha)')"
fi
printf 'Pinned actor revision: %s\n' "$QWEN_REVISION"
# Rotate arm order to reduce systematic time-of-day / mutable-service confounding.
for seed in 0 1 2; do
  arms=(grounded jev_terminal jev_shaping qwen_judge qwen_shaping)
  for ((j=0; j<${#arms[@]}; j++)); do
    arm="${arms[$(( (j + seed) % ${#arms[@]} ))]}"
    python -m jev_reward_model.train --config "configs/${arm}.yaml" \
      --seed "$seed" --model-revision "$QWEN_REVISION" \
      --qwen-judge-revision "$QWEN_REVISION" --output-dir "runs/${arm}/seed-${seed}"
  done
done
