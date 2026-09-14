"""RED tests for the model-agnostic LLM engine (OpenAI-compatible wire format)."""
import json

import httpx
import pytest

from nexus.engine import LLMEngine, NexusError


def make_engine(handler) -> LLMEngine:
    transport = httpx.MockTransport(handler)
    return LLMEngine(
        base_url="https://fake.test/v1",
        api_key="sk-test",
        model="glm-5.3",
        transport=transport,
    )


def ok(payload):
    return httpx.Response(200, json=payload)


def test_sends_model_and_auth_and_parses_content():
    captured = {}

    def handler(request):
        captured["auth"] = request.headers.get("authorization")
        captured["body"] = json.loads(request.content)
        return ok({
            "choices": [{"message": {"role": "assistant", "content": "hello"}}],
            "usage": {"total_tokens": 7},
        })

    resp = make_engine(handler).chat([{"role": "user", "content": "hi"}])
    assert resp.content == "hello"
    assert resp.tool_calls == []
    assert captured["auth"] == "Bearer sk-test"
    assert captured["body"]["model"] == "glm-5.3"
    assert captured["body"]["messages"] == [{"role": "user", "content": "hi"}]


def test_parses_tool_calls_with_json_arguments():
    def handler(request):
        return ok({
            "choices": [{"message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "c1", "type": "function", "function": {
                    "name": "write_file", "arguments": "{\"path\": \"a.txt\", \"text\": \"x\"}"}}],
            }}],
        })

    resp = make_engine(handler).chat([{"role": "user", "content": "go"}], tools=[])
    assert len(resp.tool_calls) == 1
    tc = resp.tool_calls[0]
    assert tc.name == "write_file"
    assert tc.arguments == {"path": "a.txt", "text": "x"}
    assert tc.id == "c1"


def test_tolerates_malformed_tool_arguments():
    def handler(request):
        return ok({"choices": [{"message": {"tool_calls": [{"id": "c2", "type": "function", "function": {
            "name": "sh", "arguments": "not-json"}}]}}]})

    resp = make_engine(handler).chat([{"role": "user", "content": "go"}])
    assert resp.tool_calls[0].arguments == {"_raw": "not-json"}


def test_http_error_raises_nexus_error_with_status():
    def handler(request):
        return httpx.Response(429, json={"error": {"message": "rate"}})

    with pytest.raises(NexusError) as ei:
        make_engine(handler).chat([{"role": "user", "content": "x"}])
    assert "429" in str(ei.value)


def test_network_error_raises_nexus_error():
    def handler(request):
        raise httpx.ConnectError("boom")

    with pytest.raises(NexusError):
        make_engine(handler).chat([{"role": "user", "content": "x"}])


def test_null_content_is_empty_string():
    def handler(request):
        return ok({"choices": [{"message": {"content": None}}]})

    resp = make_engine(handler).chat([{"role": "user", "content": "x"}])
    assert resp.content == ""
