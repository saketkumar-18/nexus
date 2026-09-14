"""Tool registry: typed call dispatch + OpenAI tool schema export."""
from __future__ import annotations

import json
from typing import Any, Callable

from nexus.engine import NexusError


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, dict[str, Any]] = {}

    def register(self, name: str, description: str, schema: dict,
                 fn: Callable[..., Any]) -> None:
        self._tools[name] = {"description": description, "schema": schema, "fn": fn}

    def call(self, name: str, **kwargs: Any) -> Any:
        tool = self._tools.get(name)
        if tool is None:
            raise NexusError(f"unknown tool: {name}")
        try:
            return tool["fn"](**kwargs)
        except NexusError:
            raise
        except Exception as e:  # tool crash -> agent-visible error, never loop death
            raise NexusError(f"tool {name} crashed: {e}") from e

    def openai_tools(self) -> list[dict]:
        return [
            {"type": "function",
             "function": {"name": name, "description": t["description"],
                          "parameters": t["schema"]}}
            for name, t in sorted(self._tools.items())
        ]

    def __contains__(self, name: str) -> bool:
        return name in self._tools
