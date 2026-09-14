"""Built-in tools: sandboxed filesystem, shell, and web fetch.

Every fs path is resolved inside a workspace directory — traversal outside
is refused. Shell commands run with cwd=workspace and a dangerous-pattern
blocklist. Web fetch pulls readable text via httpx.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any

import httpx

from nexus.engine import NexusError
from nexus.tools.registry import ToolRegistry

# commands the agent may never issue via the sh tool
_BLOCKED = [
    r"\brm\s+-rf\s+[/~]",       # rm -rf on root/home
    r"\bmkfs\b",
    r"\bshutdown\b",
    r"\bformat\s+[a-z]:",       # windows format
    r"del\s+/[sq]\s+\\",        # windows recursive delete of drive root
    r":\(\)\{.*\};:",           # fork bomb
    r"\bcurl\b[^|]*\|\s*\bsh\b",  # curl | sh
]


def _resolve(workspace: Path, path: str) -> Path:
    """Resolve `path` inside workspace; refuse escape (test FIRST impl)."""
    p = Path(path)
    if not p.is_absolute():
        p = workspace / p
    p = p.resolve()
    if not p.is_relative_to(workspace.resolve()):
        raise NexusError(
            f"path outside workspace: {path!r}. All fs tools take paths RELATIVE to "
            f"the workspace root ({workspace}). e.g. 'notes.txt' or 'sub/dir/file.txt'.")
    return p


def _rel(workspace: Path, p: Path) -> str:
    """Workspace-relative POSIX path for tool results (model round-trip safety)."""
    return p.resolve().relative_to(workspace.resolve()).as_posix()


def _sh(command: str, cwd: Path) -> dict[str, Any]:
    for pattern in _BLOCKED:
        if re.search(pattern, command):
            raise NexusError(f"blocked dangerous command: {command!r}")
    try:
        proc = subprocess.run(
            command, shell=True, cwd=str(cwd), capture_output=True,
            text=True, timeout=60,
        )
    except subprocess.TimeoutExpired:
        raise NexusError(f"command timed out after 60s: {command!r}")
    return {"rc": proc.returncode, "stdout": proc.stdout[-4000:], "stderr": proc.stderr[-2000:],
            "cwd": str(cwd)}


def build_tools(workspace: str | Path, http_transport: httpx.BaseTransport | None = None,
                ) -> ToolRegistry:
    workspace = Path(workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    reg = ToolRegistry()
    # holder so tests can swap the HTTP transport cleanly
    state: dict = {"client": httpx.Client(timeout=30, transport=http_transport)}

    def _set_http_transport(transport: httpx.BaseTransport) -> None:
        state["client"] = httpx.Client(timeout=30, transport=transport)


    reg.register(
        "fs_read", "Read a UTF-8 text file from the workspace.",
        {"type": "object", "properties": {
            "path": {"type": "string"}}, "required": ["path"]},
        lambda path: _fs_read(workspace, path),
    )
    reg.register(
        "fs_write", "Write (create or overwrite) a UTF-8 text file in the workspace.",
        {"type": "object", "properties": {
            "path": {"type": "string"}, "text": {"type": "string"}},
         "required": ["path", "text"]},
        lambda path, text: _fs_write(workspace, path, text),
    )
    reg.register(
        "fs_list", "List a workspace directory.",
        {"type": "object", "properties": {
            "path": {"type": "string", "default": "."}}, "required": []},
        lambda path=".": _fs_list(workspace, path),
    )
    reg.register(
        "sh", "Run a shell command inside the workspace (60s timeout, output truncated).",
        {"type": "object", "properties": {
            "command": {"type": "string"}}, "required": ["command"]},
        lambda command: _sh(command, workspace),
    )
    reg.register(
        "web_fetch", "Fetch a URL and return readable text content.",
        {"type": "object", "properties": {
            "url": {"type": "string"}}, "required": ["url"]},
        lambda url: _web_fetch(state["client"], url),
    )
    reg.register_transport = _set_http_transport
    return reg


def _fs_read(workspace: Path, path: str) -> dict:
    p = _resolve(workspace, path)
    if not p.is_file():
        raise NexusError(f"not found: {path}")
    return {"path": _rel(workspace, p),
            "content": p.read_text(encoding="utf-8", errors="replace")[:20000]}


def _fs_write(workspace: Path, path: str, text: str) -> dict:
    p = _resolve(workspace, path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return {"path": _rel(workspace, p), "bytes": len(text.encode("utf-8"))}


def _fs_list(workspace: Path, path: str = ".") -> dict:
    p = _resolve(workspace, path)
    if not p.is_dir():
        raise NexusError(f"not a directory: {path}")
    entries = []
    for child in sorted(p.iterdir()):
        entries.append({
            "name": child.name,
            "type": "dir" if child.is_dir() else "file",
            "bytes": child.stat().st_size if child.is_file() else None,
        })
    return {"path": _rel(workspace, p) if p != workspace else ".", "entries": entries}


def _web_fetch(http: httpx.Client, url: str) -> dict:
    if not re.match(r"^https?://", url):
        raise NexusError(f"invalid url: {url}")
    try:
        resp = http.get(url)
    except httpx.HTTPError as e:
        raise NexusError(f"fetch failed: {e}") from e
    return {"url": url, "status": resp.status_code, "text": resp.text[:30000]}
