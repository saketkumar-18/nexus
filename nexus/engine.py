"""Model-agnostic LLM engine.

Speaks the OpenAI-compatible chat-completions wire format, which GLM, Qwen,
Ollama, vLLM, OpenRouter and OpenAI itself all expose — so the model can be
swapped by config alone. Supports tool calling.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import httpx


class NexusError(Exception):
    """Any NEXUS-level failure: engine, tool or loop."""


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class ChatResponse:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = ""
    usage: dict = field(default_factory=dict)

    def to_message(self) -> dict[str, Any]:
        msg: dict[str, Any] = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            msg["tool_calls"] = [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}}
                for tc in self.tool_calls
            ]
        return msg


class LLMEngine:
    """Thin, testable client over an OpenAI-compatible endpoint."""

    def __init__(self, base_url: str, api_key: str, model: str,
                 timeout: float = 120.0, transport: httpx.BaseTransport | None = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self._client = httpx.Client(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
            transport=transport,  # injected in tests
        )

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> ChatResponse:
        payload: dict[str, Any] = {"model": self.model, "messages": messages}
        if tools:
            payload["tools"] = tools
        try:
            resp = self._client.post("/chat/completions", json=payload)
        except httpx.HTTPError as e:
            raise NexusError(f"engine network error ({type(e).__name__}): {e}") from e
        if resp.status_code >= 400:
            raise NexusError(f"engine HTTP {resp.status_code}: {resp.text[:300]}")
        return self._parse(resp.json())

    @staticmethod
    def _parse(data: dict) -> ChatResponse:
        try:
            choice = data["choices"][0]
            message = choice["message"]
        except (KeyError, IndexError, TypeError) as e:
            raise NexusError(f"malformed engine response: {data!r:.300}") from e
        tool_calls = []
        for raw in message.get("tool_calls") or []:
            fn = raw.get("function", {})
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args) if args.strip() else {}
                except json.JSONDecodeError:
                    args = {"_raw": args}
            args = _normalize_tool_args(args) if isinstance(args, dict) else (args or {})
            tool_calls.append(ToolCall(id=raw.get("id", ""), name=fn.get("name", ""), arguments=args))
        return ChatResponse(
            content=message.get("content") or "",
            tool_calls=tool_calls,
            finish_reason=choice.get("finish_reason", ""),
            usage=data.get("usage", {}),
        )

    def close(self) -> None:
        self._client.close()


def _normalize_tool_args(args: dict) -> dict:
    """Unwrap spec-style arguments seen from small local models:
    {"type": "function", "function": "name", "parameters": {real args}}."""
    if (set(args) == {"type", "function", "parameters"}
            and args.get("type") == "function" and isinstance(args.get("parameters"), dict)):
        return args["parameters"]
    return args


def engine_from_env() -> LLMEngine:
    """Build an engine from NEXUS_* env vars (12-factor, no keys in code)."""
    import os

    base_url = os.environ.get("NEXUS_LLM_BASE_URL", "https://api.z.ai/api/paas/v4")
    api_key = os.environ.get("NEXUS_LLM_API_KEY", "")
    model = os.environ.get("NEXUS_LLM_MODEL", "glm-5.3")
    if not api_key:
        raise NexusError(
            "NEXUS_LLM_API_KEY not set. For a local Ollama backend set "
            "NEXUS_LLM_BASE_URL=http://localhost:11434/v1 and any dummy key."
        )
    return LLMEngine(base_url=base_url, api_key=api_key, model=model)
