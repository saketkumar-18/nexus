"""RED: content-fallback tool parsing + spec-args normalization at execution."""
import json

import pytest

from nexus.agent import Agent
from nexus.engine import ChatResponse, ToolCall
from nexus.memory import Memory
from nexus.tools.builtin import build_tools


class FakeEngine:
    def __init__(self, script):
        self.script = list(script)

    def chat(self, messages, tools=None):
        return self.script.pop(0)


def make(tmp_path, script):
    ws = tmp_path / "ws"
    return Agent(
        engine=FakeEngine(script),
        registry=build_tools(workspace=ws),
        memory=Memory(db_path=tmp_path / "m.db"),
        workspace=ws,
        verifier=FakeEngine([ChatResponse(content=json.dumps({"verdict": "pass"}))]),
    )


def test_content_json_tool_call_is_executed(tmp_path):
    agent = make(tmp_path, [
        ChatResponse(content='{"name": "fs_write", "parameters": {"path": "x.txt", "text": "hi"}}'),
        ChatResponse(content="wrote x.txt"),
    ])
    result = agent.run("write x.txt")
    assert result["status"] == "done"
    assert (tmp_path / "ws" / "x.txt").read_text() == "hi"


def test_sloppy_llama_shape_with_param_key(tmp_path):
    agent = make(tmp_path, [
        ChatResponse(content='{"type":"function","function":{"name":"fs_write","param":{"path":"y.txt","text":"yo"}}}'),
        ChatResponse(content="done writing y.txt"),
    ])
    result = agent.run("write y.txt")
    assert result["status"] == "done"
    assert (tmp_path / "ws" / "y.txt").read_text() == "yo"


def test_sloppy_shape_with_trailing_garbage(tmp_path):
    agent = make(tmp_path, [
        ChatResponse(content='{"type":"function","function":{"name":"fs_write","param":{"path":"z.txt","text":"z"}}}"'),
        ChatResponse(content="done"),
    ])
    result = agent.run("write z.txt")
    assert result["status"] == "done"
    assert (tmp_path / "ws" / "z.txt").read_text() == "z"


def test_ordinary_content_is_final_answer_not_tool_call(tmp_path):
    agent = make(tmp_path, [ChatResponse(content="I organized your downloads into 4 folders.")])
    result = agent.run("clean downloads")
    assert result["status"] == "done"
    assert "4 folders" in result["report"]


def test_json_with_unknown_tool_name_ignored(tmp_path):
    agent = make(tmp_path, [
        ChatResponse(content='{"name": "hack_the_planet", "parameters": {"p": 1}}'),
        ChatResponse(content="nothing to do"),
    ])
    result = agent.run("objective")
    assert result["status"] == "done"
    assert not any((tmp_path / "ws").iterdir())


def test_spec_style_arguments_normalized_at_execution(tmp_path):
    """Direct ToolCall with spec-style args (bypassing engine parse) must still work."""
    agent = make(tmp_path, [
        ChatResponse(content="", tool_calls=[ToolCall(
            id="c1", name="fs_write",
            arguments={"type": "function", "function": "fs_write",
                       "parameters": {"path": "s.txt", "text": "spec"}})]),
        ChatResponse(content="done"),
    ])
    result = agent.run("write s.txt")
    assert result["status"] == "done"
    assert (tmp_path / "ws" / "s.txt").read_text() == "spec"


def test_repeat_content_calls_not_infinite(tmp_path):
    agent = Agent(
        engine=FakeEngine([ChatResponse(content='{"name": "fs_list", "param": {}}') for _ in range(50)]),
        registry=build_tools(workspace=tmp_path / "ws"),
        memory=Memory(db_path=tmp_path / "m.db"),
        workspace=tmp_path / "ws",
        verifier=FakeEngine([]),
        max_steps=4,
    )
    result = agent.run("loop")
    assert result["status"] == "failed"
    assert "max steps" in result["report"].lower()
