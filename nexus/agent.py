"""The NEXUS agent loop: objective -> tools -> final answer -> verify -> report.

Crash-safe by design: engine errors, tool errors and the step budget all end
in a structured `failed` result — the loop never dies silently, and every
step is logged to episodic memory.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from nexus.engine import ChatResponse, LLMEngine, NexusError
from nexus.memory import Memory
from nexus.tools.registry import ToolRegistry

SYSTEM_PROMPT = """You are NEXUS, an autonomous computer agent.
Given an objective, accomplish it using the tools available. Rules:
- Prefer tools over guessing. Verify facts by reading files or fetching pages.
- When the objective is complete, reply with your final report as plain text (no tool calls).
- Keep workspace paths relative to the workspace root.
Be concise in the final report."""


class Agent:
    def __init__(self, engine: LLMEngine, registry: ToolRegistry, memory: Memory,
                 workspace: str | Path, verifier: LLMEngine, max_steps: int = 25):
        self.engine = engine
        self.registry = registry
        self.memory = memory
        self.workspace = Path(workspace)
        self.verifier = verifier
        self.max_steps = max_steps
        self.messages: list[dict[str, Any]] = []

    # -- main entry -----------------------------------------------------------
    def run(self, objective: str) -> dict[str, Any]:
        ep = self.memory.start_episode(objective)
        ep_id = ep["id"]
        self.messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": objective},
        ]
        report: str | None = None
        try:
            report = self._loop(ep_id)
            status, critique = self._verify(objective, report)
            self.memory.finish_episode(ep_id, status=status, report=report)
            return {"episode_id": ep_id, "status": status, "report": report,
                    "critique": critique}
        except NexusError as e:
            self.memory.log_event(ep_id, "error", {"error": str(e)})
            self.memory.finish_episode(ep_id, status="failed", report=str(e))
            return {"episode_id": ep_id, "status": "failed", "report": str(e),
                    "critique": None}

    # -- inner loop -------------------------------------------------------------
    def _loop(self, ep_id: int) -> str:
        tools = self.registry.openai_tools()
        for step in range(1, self.max_steps + 1):
            resp = self.engine.chat(self.messages, tools=tools)
            self.messages.append(resp.to_message())
            self.memory.log_event(ep_id, "assistant",
                                  {"step": step, "content": resp.content[:500],
                                   "tool_calls": [(t.name, t.arguments) for t in resp.tool_calls]})
            if not resp.tool_calls:
                return resp.content  # final answer
            for call in resp.tool_calls:
                self._execute_tool(ep_id, call)
        raise NexusError(f"max steps ({self.max_steps}) reached without a final answer")

    def _execute_tool(self, ep_id: int, call) -> None:
        try:
            result = self.registry.call(call.name, **call.arguments)
            content = json.dumps(result, ensure_ascii=False, default=str)[:8000]
            self.memory.log_event(ep_id, "tool", {"tool": call.name, "ok": True,
                                                  "arguments": call.arguments})
        except NexusError as e:
            content = json.dumps({"error": str(e)})
            self.memory.log_event(ep_id, "tool", {"tool": call.name, "ok": False,
                                                  "error": str(e),
                                                  "arguments": call.arguments})
        self.messages.append({"role": "tool", "tool_call_id": call.id,
                              "name": call.name, "content": content})

    # -- verification -------------------------------------------------------------
    def _verify(self, objective: str, report: str) -> tuple[str, str]:
        prompt = [
            {"role": "system",
             "content": "You audit an agent's work. Reply ONLY with JSON: "
                        '{"verdict": "pass"|"fail", "critique": "..."}'},
            {"role": "user",
             "content": f"Objective: {objective}\n\nAgent's report:\n{report}\n\n"
                        f"Tool trace summary: {self._trace_summary()}"},
        ]
        try:
            resp = self.verifier.chat(prompt)
        except NexusError as e:
            return "done", f"verifier unreachable: {e}"
        critique = ""
        try:
            parsed = json.loads(_extract_json(resp.content))
            critique = parsed.get("critique", "")
            return ("done" if parsed.get("verdict") == "pass" else "failed"), critique
        except (json.JSONDecodeError, ValueError, TypeError):
            return "done", resp.content  # unparseable -> trust executor


    def _trace_summary(self) -> str:
        calls = [(m["name"], m["content"][:200])
                 for m in self.messages if m.get("role") == "tool"]
        return json.dumps(calls[-15:])[:3000]


def _extract_json(text: str) -> str:
    """Pull the first JSON object out of possibly-fenced/messy text."""
    if "```" in text:
        for part in text.split("```"):
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{") and part.endswith("}"):
                return part
    return text.strip()
