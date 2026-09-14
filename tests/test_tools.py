"""RED tests for the tool registry + built-in tools (fs, sh, web)."""
import pytest

from nexus.engine import NexusError
from nexus.tools.registry import ToolRegistry
from nexus.tools.builtin import build_tools


@pytest.fixture()
def reg(tmp_path):
    return build_tools(workspace=tmp_path)


def test_register_and_call_roundtrip(reg):
    reg.register("echo", "echoes", {"type": "object", "properties": {}}, lambda **kw: kw)
    assert reg.call("echo", x=1) == {"x": 1}


def test_unknown_tool_raises(reg):
    with pytest.raises(NexusError, match="unknown tool"):
        reg.call("nope")


def test_openai_schema_shape(reg):
    tools = reg.openai_tools()
    names = {t["function"]["name"] for t in tools}
    assert {"fs_read", "fs_write", "fs_list", "sh", "web_fetch"} <= names
    assert all(t["type"] == "function" for t in tools)


def test_fs_write_read_list_roundtrip(reg, tmp_path):
    reg.call("fs_write", path="sub/a.txt", text="hello")
    assert reg.call("fs_read", path="sub/a.txt")["content"] == "hello"
    listing = reg.call("fs_list", path="sub")
    assert any(e["name"] == "a.txt" and e["type"] == "file" for e in listing["entries"])


def test_fs_read_missing_file_raises(reg):
    with pytest.raises(NexusError, match="not found"):
        reg.call("fs_read", path="ghost.txt")


def test_fs_paths_cannot_escape_workspace(reg, tmp_path):
    with pytest.raises(NexusError, match="outside"):
        reg.call("fs_read", path=str(tmp_path.parent / "secret.txt"))
    with pytest.raises(NexusError, match="outside"):
        reg.call("fs_write", path="../../escape.txt", text="x")


def test_sh_runs_command_in_workspace_and_captures_output(reg, tmp_path):
    out = reg.call("sh", command="echo nexus-runs")
    assert out["rc"] == 0
    assert "nexus-runs" in out["stdout"]
    assert out["cwd"] == str(tmp_path)


def test_sh_blocked_dangerous_command(reg):
    with pytest.raises(NexusError, match="blocked"):
        reg.call("sh", command="rm -rf /")


def test_sh_nonzero_rc_is_returned_not_raised(reg):
    out = reg.call("sh", command="exit 3")
    assert out["rc"] == 3


def test_web_fetch_returns_text(reg):
    import httpx

    def handler(request):
        return httpx.Response(200, text="<h1>hosted page</h1>")

    reg.register_transport(httpx.MockTransport(handler))
    out = reg.call("web_fetch", url="https://example.com")
    assert "hosted page" in out["text"]
    assert out["status"] == 200


def test_web_fetch_bad_url_raises(reg):
    with pytest.raises(NexusError):
        reg.call("web_fetch", url="https://invalid.invalid.example")
