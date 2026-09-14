"""RED: real-world malformed tool calls must be normalized, not crash the loop.

Small local models (llama3.2:1b on Ollama) sometimes emit the full function
SPEC as the arguments object. The engine must unwrap it so the agent loop
still executes the intended call.
"""


def test_unwraps_nested_function_spec_arguments():
    from nexus.engine import LLMEngine
    import httpx, json

    def handler(request):
        return httpx.Response(200, json={"choices": [{"message": {"tool_calls": [{
            "id": "c9", "type": "function",
            "function": {"name": "fs_write",
                         "arguments": json.dumps({"type": "function", "function": "fs_write",
                                                 "parameters": {"path": "notes.txt", "text": "hi"}})}}]}}]})

    eng = LLMEngine(base_url="https://x.test/v1", api_key="k", model="m",
                    transport=httpx.MockTransport(handler))
    resp = eng.chat([{"role": "user", "content": "go"}])
    assert resp.tool_calls[0].arguments == {"path": "notes.txt", "text": "hi"}


def test_agent_survives_spec_style_arguments_and_completes(tmp_path):
    import json
    from nexus.agent import Agent
    from nexus.engine import ChatResponse, ToolCall
    from nexus.memory import Memory
    from nexus.tools.builtin import build_tools

    class FakeEngine:
        def __init__(self, script):
            self.script = list(script)

        def chat(self, messages, tools=None):
            return self.script.pop(0)

    spec_args = {"type": "function", "function": "fs_write",
                 "parameters": {"path": "notes.txt", "text": "nexus was here"}}
    agent = Agent(
        engine=FakeEngine([
            ChatResponse(content="", tool_calls=[ToolCall(id="c1", name="fs_write", arguments=spec_args)]),
            ChatResponse(content="wrote and verified notes.txt"),
        ]),
        registry=build_tools(workspace=tmp_path / "ws"),
        memory=Memory(db_path=tmp_path / "m.db"),
        workspace=tmp_path / "ws",
        verifier=FakeEngine([ChatResponse(content=json.dumps({"verdict": "pass"}))]),
    )
    result = agent.run("write notes.txt")
    assert result["status"] == "done"
    assert (tmp_path / "ws" / "notes.txt").read_text() == "nexus was here"
