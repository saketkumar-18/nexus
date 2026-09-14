"""RED: tool results speak workspace-relative paths (model round-trip safety)."""
import httpx

from nexus.tools.builtin import build_tools


def test_fs_write_returns_relative_path(tmp_path):
    reg = build_tools(workspace=tmp_path)
    out = reg.call("fs_write", path="sub/a.txt", text="x")
    assert out["path"] == "sub/a.txt"
    assert "\\" not in out["path"] and "/" not in out["path"].replace("sub/", "")


def test_fs_read_returns_relative_path_as_given(tmp_path):
    reg = build_tools(workspace=tmp_path)
    reg.call("fs_write", path="b.txt", text="y")
    out = reg.call("fs_read", path="b.txt")
    assert out["path"] == "b.txt"
    assert "content" in out


def test_fs_list_returns_relative_path(tmp_path):
    reg = build_tools(workspace=tmp_path)
    out = reg.call("fs_list", path=".")
    assert out["path"] == "."
    assert isinstance(out["entries"], list)


def test_escape_error_mentions_hint_and_root(tmp_path):
    reg = build_tools(workspace=tmp_path)
    try:
        reg.call("fs_read", path="C:/Windows/system.ini")
        raise AssertionError("should have raised")
    except Exception as e:
        assert "relative" in str(e).lower()
        assert str(tmp_path) in str(e) or "workspace" in str(e)
