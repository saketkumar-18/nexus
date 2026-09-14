"""The NEXUS agent loop: objective -> tools -> final answer -> verify -> report.

Crash-safe by design: engine errors, tool errors and the step budget all end
in a structured `failed` result — the loop never dies silently, and every
step is logged to episodic memory.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from nexus.engine import ChatResponse, LLMEngine, NexusError, ToolCall
from nexus.memory import Memory
from nexus.tools.registry import ToolRegistry

SYSTEM_PROMPT = """You are NEXUS, an autonomous computer agent.
Given an objective, accomplish it using the tools available. Rules:
- Use tools; never claim work is done without a successful tool result proving it.
- fs tools take paths RELATIVE to the workspace root, e.g. "notes.txt". Never use absolute paths.
- When the objective names a specific filename, use EXACTLY that filename — do not invent others.
- After writing a file, read it back to verify.
- When the objective is complete, reply with your final report as plain text (no tool calls).
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
                # fallback: some models emit tool calls as plain content JSON
                fb = self._content_tool_calls(resp.content)
                if fb:
                    # re-declare natively so tool-result messages stay well-formed
                    self.messages[-1] = _fb_message(resp.content, fb)
                    for call in fb:
                        self._execute_tool(ep_id, call)
                    continue
                return resp.content  # final answer
            for call in resp.tool_calls:
                self._execute_tool(ep_id, call)
        raise NexusError(f"max steps ({self.max_steps}) reached without a final answer")

    def _content_tool_calls(self, content: str) -> list[ToolCall]:
        """Parse tool calls the model wrote as JSON in plain content."""
        calls: list[ToolCall] = []
        for obj in _json_objects(content or ""):
            if not isinstance(obj, dict):
                continue
            name, params = None, {}
            if isinstance(obj.get("name"), str):
                name = obj["name"]
                params = obj.get("parameters") or obj.get("param") or obj.get("arguments") or obj.get("args") or {}
            elif obj.get("type") == "function" and isinstance(obj.get("function"), dict):
                inner = obj["function"]
                if isinstance(inner.get("name"), str):
                    name = inner["name"]
                params = inner.get("parameters") or inner.get("param") or {}
            elif isinstance(obj.get("function"), str) and isinstance(obj.get("parameters"), dict):
                name = obj["function"]
                params = obj["parameters"]
            if name and name in self.registry and isinstance(params, dict):
                calls.append(ToolCall(id=f"content-{len(calls)}", name=name,
                                     arguments=dict(params)))
        return calls

    def _execute_tool(self, ep_id: int, call) -> None:
        call.arguments = _normalize_args(call.arguments)
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


def _fb_message(content: str, calls: list[ToolCall]) -> dict[str, Any]:
    """Assistant message carrying content-parsed calls in native tool_calls form."""
    return {
        "role": "assistant", "content": content,
        "tool_calls": [
            {"id": c.id, "type": "function",
             "function": {"name": c.name, "arguments": json.dumps(c.arguments)}}
            for c in calls
        ],
    }


def _normalize_args(args: Any) -> dict:
    """Normalize spec-style tool arguments ('parameters'/'param' wrapping)."""
    if not isinstance(args, dict):
        return {}
    spec_keys = ("parameters", "param", "arguments", "args")
    if (isinstance(args.get("type"), str) and args["type"] == "function"
            and isinstance(args.get("function"), (str, dict))
            and any(k in args for k in spec_keys)):
        for k in spec_keys:
            if isinstance(args.get(k), dict):
                return args[k]
    if isinstance(args.get("function"), str):
        for k in spec_keys:
            if isinstance(args.get(k), dict):
                return args[k]
    return args


def _json_objects(text: str) -> list[Any]:
    """Extract top-level JSON objects from text via balanced-brace scanning.

    Tolerant of trailing garbage after the closing brace (small-model habit),
    fences, and multiple consecutive objects.
    """
    out: list[Any] = []
    depth, start = 0, None
    in_str, esc = False, False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth:
                depth -= 1
                if depth == 0 and start is not None:
                    try:
                        out.append(json.loads(text[start:i + 1]))
                    except json.JSONDecodeError:
                        pass
                    start = None
    return out
