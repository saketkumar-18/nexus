"""RED tests for the Agent loop: plan->execute->verify->report, crash-safe."""
import json

import pytest

from nexus.agent import Agent
from nexus.engine import ChatResponse, NexusError, ToolCall


class FakeEngine:
    """Scripted engine: pops queued responses; can raise NexusError."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    def chat(self, messages, tools=None):
        self.calls += 1
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return step


def tc(name, **args):
    return ChatResponse(content="", tool_calls=[ToolCall(id=f"c_{name}", name=name, arguments=args)])


@pytest.fixture()
def setup(tmp_path):
    from nexus.memory import Memory
    from nexus.tools.builtin import build_tools

    return {
        "workspace": tmp_path / "ws",
        "mem": Memory(db_path=tmp_path / "mem.db"),
        "tools": build_tools(workspace=tmp_path / "ws"),
    }


def make_agent(setup, script, max_steps=8):
    return Agent(
        engine=FakeEngine(script),
        registry=setup["tools"],
        memory=setup["mem"],
        workspace=setup["workspace"],
        verifier=FakeEngine([]),
        max_steps=max_steps,
    )


def test_executes_tools_until_final_answer_and_logs(setup):
    agent = make_agent(setup, [
        tc("fs_write", path="report.md", text="# cleaned"),
        ChatResponse(content="Downloads organized into 4 folders."),
    ])
    agent.verifier.script = [ChatResponse(content=json.dumps({"verdict": "pass", "critique": "solid"}))]
    result = agent.run("clean my downloads")
    assert result["status"] == "done"
    assert "4 folders" in result["report"]
    assert (setup["workspace"] / "report.md").read_text() == "# cleaned"
    events = setup["mem"].episode_events(result["episode_id"])
    kinds = [e["kind"] for e in events]
    assert "assistant" in kinds and "tool" in kinds
    assert setup["mem"].recent_episodes()[0]["status"] == "done"


def test_verifier_fail_marks_failed_with_critique(setup):
    agent = make_agent(setup, [
        tc("fs_write", path="r.md", text="x"),
        ChatResponse(content="I organized everything."),
    ])
    agent.verifier.script = [ChatResponse(content=json.dumps({"verdict": "fail", "critique": "no evidence files were moved"}))]
    result = agent.run("clean downloads")
    assert result["status"] == "failed"
    assert "no evidence" in result["critique"]
    assert setup["mem"].recent_episodes()[0]["status"] == "failed"


def test_unknown_tool_error_fed_back_loop_survives(setup):
    agent = make_agent(setup, [
        tc("does_not_exist", x=1),
        ChatResponse(content="recovered and finished."),
    ])
    agent.verifier.script = [ChatResponse(content=json.dumps({"verdict": "pass"}))]
    result = agent.run("objective")
    assert result["status"] == "done"
    # the error must have been appended as a tool result message
    msgs = agent.messages
    tool_results = [m for m in msgs if m.get("role") == "tool"]
    assert any("unknown tool" in m["content"] for m in tool_results)


def test_max_steps_guard_fails_cleanly(setup):
    agent = make_agent(setup, [tc("fs_list") for _ in range(50)], max_steps=3)
    result = agent.run("loop forever")
    assert result["status"] == "failed"
    assert "max steps" in result["report"].lower()
    assert setup["mem"].recent_episodes()[0]["status"] == "failed"


def test_engine_crash_mid_loop_marks_failed(setup):
    agent = make_agent(setup, [
        tc("fs_list"),
        NexusError("engine HTTP 429: rate limited"),
    ])
    result = agent.run("objective")
    assert result["status"] == "failed"
    assert "429" in result["report"]
    assert setup["mem"].recent_episodes()[0]["status"] == "failed"


def test_crashed_tool_call_error_fed_back(setup):
    agent = make_agent(setup, [
        tc("fs_read", path="ghost.txt"),  # raises NexusError inside tool
        ChatResponse(content="ok after error"),
    ])
    agent.verifier.script = [ChatResponse(content=json.dumps({"verdict": "pass"}))]
    result = agent.run("objective")
    assert result["status"] == "done"
    tool_results = [m for m in agent.messages if m.get("role") == "tool"]
    assert any("not found" in m["content"] for m in tool_results)


def test_verifier_unparseable_counts_as_done_with_raw_critique(setup):
    agent = make_agent(setup, [ChatResponse(content="done it")])
    agent.verifier.script = [ChatResponse(content="seems fine to me")]
    result = agent.run("objective")
    assert result["status"] == "done"
    assert "seems fine" in result["critique"]
