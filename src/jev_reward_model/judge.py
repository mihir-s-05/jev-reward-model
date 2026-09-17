from __future__ import annotations

import json

import torch


class LocalQwenJudge:
    """Frozen binary terminal judge using the same Qwen3.5-4B base checkpoint."""

    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
        tokenizer = processor.tokenizer
        yes = tokenizer.encode("YES", add_special_tokens=False)
        no = tokenizer.encode("NO", add_special_tokens=False)
        if len(yes) != 1 or len(no) != 1:
            raise RuntimeError("YES/NO must each be one token for the logit judge baseline")
        self.yes_id, self.no_id = yes[0], no[0]

    @torch.no_grad()
    def __call__(self, public_state: dict) -> float:
        messages = [
            {
                "role": "user",
                "content": (
                    "Judge whether this workflow is fully successful. It is successful only if every milestone "
                    "was completed in order, the exact completion string was emitted, and no constraint was "
                    "violated. Answer YES or NO only.\n\n" + json.dumps(public_state, sort_keys=True)
                ),
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        ).to(self.model.device)
        logits = self.model(**inputs).logits[0, -1, [self.no_id, self.yes_id]].float()
        return float(torch.softmax(logits, dim=-1)[1].item())
