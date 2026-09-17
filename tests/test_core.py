"""A small set of invariants; no actor weights, network requests, or paid APIs."""
import asyncio
import json
import numpy as np
import httpx
import pytest

from jev_reward.config import TaskConfig, JudgeConfig, canonical, load_config
from jev_reward.env import SPLITS, FulfillmentEnv, audit_trace, expert_actions, judge_view, make_task
from jev_reward.judges import Judge, parse_jev
from jev_reward.rewards import potential_shaping, score_episode


def jev_response():
    return {"model": "jev-test-snapshot", "answers": {
        "success": {"type": "noul", "noul": 0.9},
        "compliance": {"type": "noul", "noul": 0.95},
        "progress": {"type": "score", "score": 3.5,
                     "probabilities": {"0": 0, "1": 0, "2": 0, "3": 0.5, "4": 0.5}}},
        "usage": {"input_tokens": 100, "output_tokens": 0}}


def test_expert_and_fault_controls():
    for split in SPLITS:
        for index in range(3):
            task = make_task(split, index, 2, TaskConfig())
            state, oracle = audit_trace(task, None)
            assert oracle["success"] == 1
            assert state["closed"]
            for fault in ("address", "incomplete", "protected", "claim"):
                assert audit_trace(task, fault)[1]["success"] == 0


def test_private_verifier_and_persistent_memory():
    task = make_task("validation", 0, 0, TaskConfig())
    env = FulfillmentEnv(task)
    order = task.protected[0]
    spec = task.orders[order]
    env.step(canonical({"tool": "reserve", "order": order, "sku": spec["sku"], "quantity": 1}))
    env.step(canonical({"tool": "release", "order": order}))
    for action in expert_actions(task):
        if not env.closed:
            env.step(action)
    assert env.verify()["protected_mutation"]
    view = judge_view(env.snapshot(), "ledger", 1)
    assert view["ledger"]["mutations"][0]["order"] == order
    assert "success" not in view and "oracle" not in view
    assert "correct_fraction" not in canonical(view)


def test_shaping_telescopes_and_terminal_zero():
    for gamma in (1.0, 0.93):
        phi = [0.2, 0.5, 0.4, 0.8, 0]
        rewards = potential_shaping(phi, gamma, 0.7)
        assert np.dot(gamma ** np.arange(len(rewards)), rewards) == pytest.approx(-0.7 * phi[0])
    with pytest.raises(ValueError):
        potential_shaping([0.1, 0.5], 1.0, 0.5)


def test_jev_wire_schema_and_rejection():
    response = jev_response()
    result = parse_jev(response)
    assert result.progress == 0.875 and result.success == 0.9
    response["answers"]["progress"]["score"] = 1
    with pytest.raises(ValueError):
        parse_jev(response)


def test_http_cache_and_explicit_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-not-a-real-key")
    calls = []
    def handler(request):
        calls.append(json.loads(request.content))
        assert request.url.path == "/v1/systemone"
        return httpx.Response(200, json=jev_response())
    async def exercise():
        judge = Judge("jev", JudgeConfig(), tmp_path, httpx.MockTransport(handler))
        try:
            state = {"goal": {}, "events": [], "closed": True}
            first = await judge.evaluate(state)
            second = await judge.evaluate(state)
            assert first.success == second.success == 0.9
            assert second.cache_hit and len(calls) == 1
            assert judge.stats["issued_requests"] == 1
            assert judge.stats["known_cost_usd"] == pytest.approx(100 * 0.042 / 1e6)
        finally:
            await judge.close()
    asyncio.run(exercise())


def test_terminal_reward_does_not_leak_oracle():
    class ConstantJudge:
        async def evaluate(self, state):
            assert "oracle" not in state
            return parse_jev(jev_response())
    async def exercise():
        snapshots = [{"events": [], "closed": False}, {"events": [], "closed": True}]
        reward, _ = await score_episode("jev_terminal", snapshots, 0, ConstantJudge(), 1.0, 0.5)
        assert reward == [0.9]  # Deliberately wrong judge is NOT corrected by oracle.
        reward, _ = await score_episode("oracle_terminal", snapshots, 0, None, 1.0, 0.5)
        assert reward == [0.0]
    asyncio.run(exercise())


def test_config_rejects_typos(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("ppo:\n  learning_rat: 0.1\n")
    with pytest.raises(ValueError):
        load_config(path)
