"""Fast offline tests for AgriSense Gemini visitor explanation layer."""
import io
import json
from urllib.error import HTTPError

from fastapi.testclient import TestClient

import app.assistant as assistant
from app.assistant import AssistantRequest, explain, plain_language_answer
from app.main import app

client = TestClient(app)


def sample(**overrides):
    data = dict(disease="apple black rot", confidence=0.48, uncertain=True,
                mask_coverage=0.19, question="Can you tell how serious this is?")
    data.update(overrides)
    return data


def test_built_in_explanation_does_not_confuse_coverage_with_severity(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    answer = explain(AssistantRequest(**sample()))
    assert answer.mode == "guide"
    assert "not a measure" in answer.answer
    assert "Apple Black Rot" in answer.answer
    assert "GEMINI_API_KEY" in answer.note


def test_confidence_not_treated_as_calibrated_probability():
    answer = plain_language_answer(AssistantRequest(**sample(question="How confident are you?")))
    assert "not the chance" in answer


def test_gemini_request_metadata_only_and_server_side_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-only-secret")
    monkeypatch.setenv("AGRISENSE_AI_MODEL", "gemini-3.8-flash")
    monkeypatch.setattr(assistant, "_allow_ai_call", lambda: True)
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def read(self, _limit):
            return json.dumps({"candidates": [{"content": {"parts": [
                {"text": "This is a tentative result. Check the plant."}
            ]}}]}).encode()

    def fake_open(request, timeout):
        captured["url"] = request.full_url
        captured["headers"] = dict(request.header_items())
        captured["payload"] = json.loads(request.data)
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(assistant, "urlopen", fake_open)
    answer = explain(AssistantRequest(**sample()))
    assert answer.mode == "ai"
    assert "tentative" in answer.answer
    assert "Gemini" in answer.note
    assert "/v1beta/models/gemini-3.8-flash:generateContent" in captured["url"]
    assert "test-only-secret" not in captured["url"]
    assert captured["headers"]["X-goog-api-key"] == "test-only-secret"
    body = captured["payload"]
    data_text = body["contents"][0]["parts"][0]["text"]
    assert json.loads(data_text) == {
        "predicted_condition": "apple black rot",
        "model_confidence_score": 0.48,
        "uncertain": True,
        "segmentation_image_coverage_fraction": 0.19,
        "visitor_question": "Can you tell how serious this is?",
    }
    assert "inlineData" not in str(body) and "inline_data" not in str(body)
    assert "test-only-secret" not in json.dumps(body)
    assert "maxOutputTokens" in body["generationConfig"]


def test_older_flash_model_uses_thinking_budget_zero(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-only-secret")
    monkeypatch.setenv("AGRISENSE_AI_MODEL", "gemini-2.5-flash")

    class FakeResponse:
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def read(self, _limit):
            return b'{"candidates":[{"content":{"parts":[{"text":"Clear response."}]}}]}'

    def fake_open(req, timeout):
        body = json.loads(req.data)
        assert body["generationConfig"]["thinkingConfig"]["thinkingBudget"] == 0
        return FakeResponse()

    monkeypatch.setattr(assistant, "urlopen", fake_open)
    assert assistant._call_gemini(AssistantRequest(**sample())) == "Clear response."


def test_network_failure_returns_built_in_explanation(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-only")
    monkeypatch.setattr(assistant, "_allow_ai_call", lambda: True)
    def failure(_):
        raise TimeoutError("network unavailable")
    monkeypatch.setattr(assistant, "_call_gemini", failure)
    assert explain(AssistantRequest(**sample())).mode == "guide"


def test_gemini_quota_failure_is_graceful(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-only")
    monkeypatch.setattr(assistant, "_allow_ai_call", lambda: True)
    def failure(_):
        raise HTTPError("https://generativelanguage.googleapis.com", 429,
                        "RESOURCE_EXHAUSTED", hdrs=None, fp=io.BytesIO(b''))
    monkeypatch.setattr(assistant, "_call_gemini", failure)
    result = explain(AssistantRequest(**sample()))
    assert result.mode == "guide"
    assert "quota" in result.note.lower()


def test_exhibition_request_cap_falls_back(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-only")
    monkeypatch.setattr(assistant, "_allow_ai_call", lambda: False)
    result = explain(AssistantRequest(**sample()))
    assert result.mode == "guide"
    assert "hourly" in result.note.lower()


def test_endpoint_responds_and_validates_label(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    result = client.post("/api/assistant", json=sample())
    assert result.status_code == 200
    assert result.json()["mode"] == "guide"
    assert "severity" in result.json()["answer"]
    assert client.post("/api/assistant", json=sample(disease="not-a-real-class")).status_code == 422


def test_bad_score_and_oversized_question_rejected():
    assert client.post("/api/assistant", json=sample(confidence=8)).status_code == 422
    assert client.post("/api/assistant", json=sample(question="x" * 401)).status_code == 422
