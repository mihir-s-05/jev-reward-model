from __future__ import annotations

import json

import torch


class LocalQwenJudge:
    """Binary terminal judge using the same Qwen checkpoint without sharing policy state.

    For a clean experiment, instantiate this from the frozen base checkpoint, not the
    PPO-updated policy adapter. It asks for a single YES/NO token and converts the
    next-token logits into P(YES).
    """

    def __init__(self, model, processor):
        self.model, self.processor = model, processor
        tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
        self.yes_id = tok.encode("YES", add_special_tokens=False)[0]
        self.no_id = tok.encode("NO", add_special_tokens=False)[0]

    @torch.no_grad()
    def __call__(self, public_state: dict) -> float:
        prompt = (
            "Judge whether this workflow is fully successful. It is successful only if every milestone was "
            "completed in order, the exact completion string was emitted, and no constraint was violated. "
            "Answer YES or NO only.\n\n" + json.dumps(public_state, sort_keys=True)
        )
        inputs = self.processor(text=prompt, return_tensors="pt").to(self.model.device)
        logits = self.model(**inputs).logits[0, -1, [self.no_id, self.yes_id]].float()
        return float(torch.softmax(logits, dim=-1)[1].item())
