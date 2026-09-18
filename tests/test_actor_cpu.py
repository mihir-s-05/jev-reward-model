"""CPU actor wiring: full generate/score/PPO/preflight path without Hub weights."""
from __future__ import annotations

import os
import random
import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch import nn

from jev_reward_model.config import ExperimentConfig
from jev_reward_model.data import make_task
from jev_reward_model.policy import Actor, prompt_token_ids
from jev_reward_model.ppo import add_gae, update
from jev_reward_model.preflight import inspect_actor
from jev_reward_model.rewards import assign_rewards
from jev_reward_model.rollout import collect


HIDDEN, VOCAB = 8, 16


class LanguageBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(HIDDEN, HIDDEN)


class LanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([LanguageBlock()])


class Inner(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, HIDDEN)
        self.language_model = LanguageModel()

    def forward(self, input_ids, attention_mask=None, use_cache=False, return_dict=True):
        hidden = self.language_model.layers[0].q_proj(self.embed(input_ids))
        return SimpleNamespace(last_hidden_state=hidden)


class GenerateOutput:
    def __init__(self, sequences, scores):
        self.sequences = sequences
        self.scores = scores


class FakeQwen(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Inner()
        self.lm_head = nn.Linear(HIDDEN, VOCAB, bias=False)
        self.config = SimpleNamespace(text_config=SimpleNamespace(hidden_size=HIDDEN),
                                      _commit_hash="fake-revision")
        self.generation_config = SimpleNamespace(eos_token_id=1)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        model = cls()
        device_map = kwargs.get("device_map") or {"": "cpu"}
        target = next(iter(device_map.values()))
        dtype = kwargs.get("dtype", torch.float32)
        return model.to(device=torch.device(target), dtype=dtype)

    def generate(self, input_ids, attention_mask=None, generation_config=None):
        hidden = self.model(input_ids).last_hidden_state
        logits = self.lm_head(hidden[:, -1].float())
        token = torch.ones(input_ids.size(0), 1, dtype=torch.long, device=input_ids.device)
        sequences = torch.cat([input_ids, token], dim=1)
        if getattr(generation_config, "return_dict_in_generate", False):
            return GenerateOutput(sequences, [logits])
        return sequences

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        return None

    def enable_input_require_grads(self):
        return None

    def get_base_model(self):
        return self

    def disable_adapter(self):
        return nullcontext()

    def save_pretrained(self, directory):
        Path(directory).mkdir(parents=True, exist_ok=True)
        (Path(directory) / "adapter_marker.txt").write_text("ok")


class TokenizersEncoding:
    """tokenizers.Encoding stand-in: integer indexing works, numel does not."""

    def __init__(self, ids):
        self.ids = list(ids)

    def __len__(self):
        return len(self.ids)


class FakeBatchEncoding(dict):
    """transformers 5 BatchEncoding stand-in for apply_chat_template(..., return_tensors='pt')."""

    def __init__(self, input_ids):
        super().__init__(input_ids=input_ids)
        self.input_ids = input_ids
        row = input_ids[0] if getattr(input_ids, "ndim", 1) == 2 else input_ids
        self._encodings = [TokenizersEncoding(row.tolist())]

    def __getitem__(self, item):
        if isinstance(item, int):
            return self._encodings[item]
        return super().__getitem__(item)


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    pad_token = None
    eos_token = "<eos>"
    padding_side = "right"

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def apply_chat_template(self, messages, **kwargs):
        return FakeBatchEncoding(torch.tensor([[2, 3, 4]]))

    def decode(self, ids, skip_special_tokens=True):
        return "WAIT"

    def save_pretrained(self, directory):
        Path(directory).mkdir(parents=True, exist_ok=True)


class GenerationConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def install_fake_hf(monkeypatch):
    transformers = ModuleType("transformers")
    transformers.AutoTokenizer = FakeTokenizer
    transformers.Qwen3_5ForConditionalGeneration = FakeQwen
    transformers.GenerationConfig = GenerationConfig
    peft = ModuleType("peft")

    def get_peft_model(base, config):
        return base

    class PeftModel:
        @staticmethod
        def from_pretrained(base, path, is_trainable=True):
            return base

    peft.LoraConfig = lambda **kwargs: SimpleNamespace(**kwargs)
    peft.get_peft_model = get_peft_model
    peft.PeftModel = PeftModel
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "peft", peft)


def cpu_cfg(**overrides):
    cfg = ExperimentConfig(device="cpu", dtype="float32", gradient_checkpointing=True,
                           max_new_tokens=4, generation_batch_size=1, rollouts_per_update=1,
                           minibatch_size=1, ppo_epochs=1, **overrides)
    cfg.validate()
    return cfg


def test_actor_accepts_cpu_float32_and_stays_on_cpu(monkeypatch, tmp_path):
    install_fake_hf(monkeypatch)
    actor = Actor(cpu_cfg())
    assert actor.device.type == "cpu"
    assert actor.value_head.weight.device.type == "cpu"
    assert all(p.device.type == "cpu" for p in actor.model.parameters())
    assert actor.value_head.weight.dtype == torch.float32
    prompt = actor.prompt({"x": 1})
    assert prompt.device.type == "cpu"
    assert prompt.ndim == 1
    assert torch.equal(prompt, torch.tensor([2, 3, 4]))
    actions = actor.generate([prompt])
    assert actions and actions[0].device.type == "cpu"
    lp, value = actor.score(prompt, actions[0])
    assert lp.device.type == "cpu" and value.device.type == "cpu"
    (-lp.mean() + value.square().mean()).backward()
    opt = actor.optimizer()
    torch.nn.utils.clip_grad_norm_(
        [p for group in opt.param_groups for p in group["params"]], 1.0, error_if_nonfinite=True)
    opt.step()
    actor.save(tmp_path / "ckpt")
    assert (tmp_path / "ckpt" / "value_head.pt").is_file()
    resumed = Actor(cpu_cfg(), tmp_path / "ckpt")
    assert resumed.value_head.weight.device.type == "cpu"


def test_cuda_config_still_requires_cuda(monkeypatch):
    install_fake_hf(monkeypatch)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA is required by this config"):
        Actor(ExperimentConfig(device="cuda:0", dtype="bfloat16"))


def test_cpu_collect_ppo_eval_and_preflight(monkeypatch):
    install_fake_hf(monkeypatch)
    cfg = cpu_cfg()
    actor = Actor(cfg)
    task = make_task(random.Random(0), "train", 0, "chain", 1)
    episodes = collect(actor, [task], cfg, training=True)
    assert episodes[0].turns
    assign_rewards(episodes, ExperimentConfig(reward="grounded", device="cpu", dtype="float32"), None)
    turns = []
    for episode in episodes:
        for turn, reward in zip(episode.turns, episode.rewards):
            turn.reward = reward
        add_gae(episode.turns, cfg.gamma, cfg.gae_lambda)
        turns.extend(episode.turns)
    losses = update(actor, turns, actor.optimizer(), cfg, random.Random(0))
    assert losses["optimizer_steps"] >= 0
    evaluated = collect(actor, [task], cfg, training=False)
    assert evaluated[0].turns
    result = inspect_actor(actor, [task.public_spec(), evaluated[0].states[0]], cfg, max_logprob_error=1e6)
    assert result["device"] == "cpu"
    assert result["dtype"] == "float32"
    assert result["status"].startswith("Actor preflight passed")


def test_prompt_token_ids_rejects_batch_encoding_integer_index():
    encoded = FakeBatchEncoding(torch.tensor([[9, 8, 7]]))
    with pytest.raises(AttributeError):
        encoded[0].numel()
    ids = prompt_token_ids(encoded)
    assert ids.ndim == 1
    assert torch.equal(ids, torch.tensor([9, 8, 7]))


@pytest.mark.parametrize("encoded", [
    torch.tensor([[2, 3, 4]]),
    torch.tensor([2, 3, 4]),
    FakeBatchEncoding(torch.tensor([[2, 3, 4]])),
    {"input_ids": torch.tensor([[2, 3, 4]])},
    {"input_ids": [[2, 3, 4]]},
])
def test_prompt_token_ids_normalizes_tensor_and_batch_encoding(encoded):
    ids = prompt_token_ids(encoded)
    assert ids.ndim == 1
    assert torch.equal(ids, torch.tensor([2, 3, 4]))


@pytest.mark.parametrize("factory", [
    lambda: torch.tensor([[2, 3, 4]]),
    lambda: torch.tensor([2, 3, 4]),
    lambda: FakeBatchEncoding(torch.tensor([[2, 3, 4]])),
    lambda: {"input_ids": torch.tensor([[2, 3, 4]])},
])
def test_actor_prompt_accepts_chat_template_return_shapes(monkeypatch, factory):
    install_fake_hf(monkeypatch)
    monkeypatch.setattr(FakeTokenizer, "apply_chat_template",
                        lambda self, messages, **kwargs: factory())
    prompt = Actor(cpu_cfg()).prompt({"x": 1})
    assert prompt.ndim == 1
    assert prompt.device.type == "cpu"
    assert torch.equal(prompt, torch.tensor([2, 3, 4]))


@pytest.mark.skipif(not os.environ.get("JEV_CPU_INTEGRATION"),
                    reason="Set JEV_CPU_INTEGRATION=1 to load real Qwen weights on CPU")
def test_real_qwen_actor_loads_on_cpu():
    actor = Actor(ExperimentConfig.load("configs/grounded_cpu.yaml"))
    assert actor.device.type == "cpu"
    assert next(actor.model.parameters()).dtype == torch.float32
