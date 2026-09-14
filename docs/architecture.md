# NEXUS Architecture

## Design goals

1. **Model-agnostic** — one wire protocol (OpenAI-compatible chat completions).
2. **Local-first** — default deployment is a localhost Ollama; zero cloud dependency.
3. **Crash-safe** — any failure ends in a structured `failed` result, never a hang.
4. **Auditable** — every step persisted in SQLite before the next one starts.

## Components

### LLMEngine (`nexus/engine.py`)

Stateless client over `/chat/completions`. Parses native `tool_calls`; tolerates
malformed string arguments (`{"_raw": ...}`). The companion normalization lives
in `agent._normalize_args` (execution time), so even engines that bypass parsing
are covered.

### Agent loop (`nexus/agent.py`)

```
run(objective):
  start episode → messages = [system, user]
  loop ≤ max_steps:
      resp = engine.chat(messages, tools)
      log assistant step
      if native tool_calls: execute each
      elif content parses as tool call JSON: execute each   ← fallback path
      else: return content as final report
  (budget exhausted) → NexusError → episode failed
  verify(objective, report) → (status, critique)
  finish episode
```

The fallback parser `_content_tool_calls` + `_json_objects` exists because real
local models (observed: `llama3.2:1b` via Ollama) sometimes emit tool calls as
plain-text JSON instead of the native field. The scanner is a balanced-brace
walk (string-aware), so trailing garbage after the closing brace doesn't kill
the parse.

### Tool layer (`nexus/tools/`)

Registry maps name → (description, JSON-schema, callable) and exports the
OpenAI tools array. `build_tools(workspace)` wires the built-ins:

| tool   | contract |
|--------|----------|
| `fs_read` / `fs_write` / `fs_list` | workspace-relative POSIX paths in AND out; escape attempts raise a corrective error |
| `sh`   | cwd=workspace, 60s timeout, blocklist, non-zero rc returned (not raised) |
| `web_fetch` | http(s) only, text truncated at 30k chars |

Sandbox rule: `_resolve()` rejects any path that resolves outside the workspace
root. This is the single choke point all fs tools go through.

### Memory (`nexus/memory.py`)

SQLite: `episodes` (objective/status/report), `events` (per-step assistant/tool
log), `facts` (upserted key-value for durable preferences). WAL not needed at
this scale; single-writer CLI process.

### Verifier

Second pass with the same engine: objective + report + last 15 tool results →
must reply `{"verdict": "pass"|"fail", "critique": "..."}`. Unparseable verifier
output degrades to `done` with raw critique as the critique. The verifier gates
`status` but the **ground truth lives in the workspace and DB** — the live E2E
asserts both.

## Hardening ledger (each entry = a live failure, then a test)

| # | Live failure observed | Fix | Test |
|---|------------------------|-----|------|
| 1 | 1b model emitted function *spec* as arguments → tool crash | `_normalize_args` unwrap | `test_malformed_tools.py` |
| 2 | Tool calls emitted as plain-content JSON, no native field | content fallback + balanced-brace scanner | `test_content_fallback.py` |
| 3 | Trailing garbage after JSON object (extra `"}`) | scanner ignores post-brace text | `test_content_fallback.py` |
| 4 | `fs_write` echoed absolute Windows path → model invented `/workspace/...` | tools return workspace-relative POSIX paths | `test_relative_paths.py` |
| 5 | Model claimed success with nothing done; verifier rubber-stamped | ground-truth assertion in E2E; "never claim done without tool proof" prompt rule | `scripts/e2e_live.py` |

## Non-goals (MVP)

- No browser automation yet (CDP tool is next; design leaves room in the registry).
- Single agent, no planner/researcher/coder fan-out yet — the loop is the seam
  where multi-agent slots in.
- No streaming; batch turns per step are fine for objectives.
