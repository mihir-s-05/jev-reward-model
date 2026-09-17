import json

import httpx
import pytest

from jev_reward_model.config import ExperimentConfig
from jev_reward_model.jev import JevClient
from jev_reward_model.judge import parse_typesafe, questions


def response():
    return {"model": "jev-test", "answers": {
        "success": {"type": "noul", "noul": 0.8},
        "progress": {"type": "score", "score": 2.0,
                     "legend": {str(i): text for i, text in enumerate(questions("progress")["progress"]["criteria"])},
                     "probabilities": {"0": 0, "1": 0, "2": 1, "3": 0, "4": 0}, "confidence": 1}},
        "usage": {"input_tokens": 100, "output_tokens": 0}}


def test_documented_schema_not_confidence_as_reward():
    parsed = parse_typesafe(response(), "both")
    assert parsed["success"] == 0.8 and parsed["progress"] == 0.5
    bad = response()
    bad["answers"]["success"]["noul"] = float("nan")
    with pytest.raises(ValueError):
        parse_typesafe(bad, "terminal")


def test_http_contract_cache_and_billing(monkeypatch, tmp_path):
    monkeypatch.setenv("TYPESAFE_API_KEY", "fake-test-key")
    calls = []
    def handler(request):
        payload = json.loads(request.content)
        assert request.url.path == "/v1/systemone"
        assert set(payload) == {"state", "model", "questions"}
        assert request.headers["authorization"] == "Bearer fake-test-key"
        calls.append(payload)
        return httpx.Response(200, json=response())
    client = JevClient(ExperimentConfig(), tmp_path)
    client.http.close()
    client.http = httpx.Client(transport=httpx.MockTransport(handler),
                              headers={"Authorization": "Bearer fake-test-key"})
    try:
        client.evaluate_many([{"evidence": 1}], "both")
        client.evaluate_many([{"evidence": 1}], "both")
        assert len(calls) == 1
        assert client.metrics()["input_tokens"] == 100
        assert client.metrics()["replay_input_tokens"] == 200
        assert client.metrics()["cache_hits"] == 1
        assert "fake-test-key" not in (tmp_path / "judge_requests.jsonl").read_text()
    finally:
        client.close()
