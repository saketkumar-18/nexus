# NEXUS

**Natural-language Execution & eXpert Utility System** — a local-first, model-agnostic
autonomous computer agent. You give it an objective; it plans, uses tools, verifies
its own work, and reports back.

```
objective ─► planner/executor loop ─► tools ─► ground-truth check ─► verified report
                │                                                        │
                └────────────── SQLite episodic memory ────────────────┘
```

## Why this exists

Most "AI agent" demos are a chatbot with a tool button. NEXUS is built on three
unpopular opinions, each backed by a live failure it survived:

1. **Never trust the model's claim of success.** In the first live run, a
   small model wrote nothing, *claimed* success, and a naive verifier agreed.
   NEXUS therefore checks ground truth (does the file exist? did the command's
   exit code say what the report says?) and logs every step.
2. **Real models are messy; the harness must be tolerant.** Local models emit
   tool calls as raw JSON in plain text, wrap arguments in spec-shaped objects,
   append trailing garbage, and invent absolute paths. NEXUS's parsing layer
   handles all four — each discovered by a live run, each locked in by a test.
3. **Model-agnostic by design.** One engine speaks the OpenAI-compatible wire
   format: GLM-5.3, Qwen, any Ollama model, vLLM, OpenRouter — swap by config.

## Quick start

```bash
git clone https://github.com/saketkumar-18/nexus.git
cd nexus
python -m venv .venv && .venv/Scripts/pip install -e ".[dev]"   # or .venv/bin/pip
cp .env.example .env    # configure backend
nexus run "write hello.txt containing 'hi from nexus' and read it back"
```

`.env` (12-factor; no keys in code):

```ini
# Cloud (GLM):               # Local (Ollama):
NEXUS_LLM_BASE_URL=https://api.z.ai/api/paas/v4
NEXUS_LLM_API_KEY=sk-...
NEXUS_LLM_MODEL=glm-5.3
                             # NEXUS_LLM_BASE_URL=http://localhost:11434/v1
                             # NEXUS_LLM_API_KEY=ollama
                             # NEXUS_LLM_MODEL=llama3.2:1b
NEXUS_WORKSPACE=./my-workspace
NEXUS_DB=nexus.db
```

`pip install -e .` puts `nexus` on PATH; `nexus history` lists past episodes.

## What a run looks like

```
$ nexus run "In the workspace: write notes.txt containing 'nexus was here', read it back to verify."
episode #1 — done
I wrote notes.txt (15 bytes) and read back the exact content.
verifier: pass
```

Every episode is logged in SQLite (objectives, per-step assistant actions, tool
results, verifier verdicts) — `nexus history` shows the audit trail.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        nexus run "<objective>"                │
└──────────────┬───────────────────────────────────────────────┘
               ▼
        ┌─────────────┐   OpenAI-compatible    ┌──────────────────┐
        │  LLMEngine  │◄────── wire ──────────►│ GLM-5.3 / Qwen / │
        └──────┬──────┘                        │ Ollama / vLLM    │
               ▼                               └──────────────────┘
        ┌─────────────────────────────────────────────┐
        │ Agent loop (≤ max_steps)                    │
        │  • native tool_calls OR content-JSON        │
        │    fallback parsing (balanced-brace scan)   │
        │  • argument normalization (spec-style wrap) │
        │  • errors fed back as tool results         │
        └──────┬───────────────────────┬────────────┘
               ▼                       ▼
      ┌─────────────────┐     ┌──────────────────┐
      │ ToolRegistry    │     │ Memory (SQLite)  │
      │ • fs_read/write │     │ episodes, events,│
      │ • fs_list       │     │ durable facts    │
      │ • sh (sandboxed)│     └──────────────────┘
      │ • web_fetch     │
      └─────────────────┘
               ▼
      ┌─────────────────┐
      │ Verifier pass   │  auditors result JSON; unparseable ⇒ done
      └─────────────────┘
               ▼
        report + status (done/failed) + critique
```

**Tool sandbox guarantees** (unit-tested):
- fs tools resolve inside the workspace root only; traversal (`../../`, absolute
  paths) raises a corrective error, never writes outside
- `sh` runs with cwd=workspace, 60s timeout, blocked-command list
  (`rm -rf /`, `mkfs`, `format`, fork bombs, `curl | sh`)
- tool crashes become agent-visible errors, never loop death

**Live E2E** (`scripts/e2e_live.py`): real objective against a real Ollama
model, with a ground-truth assertion — passed against `llama3.2:1b`
(3 tool calls, exact file content verified). The run also demonstrates honest
reporting: the tiny model's *report text* was gibberish while the workspace
contained exactly the requested file — which is exactly why NEXUS verifies
against reality instead of trusting prose.

## Repository layout

```
nexus/
  engine.py            model-agnostic LLM client + malformed-call hardening
  agent.py             loop, content-fallback parsing, verifier
  memory.py            SQLite episodic memory + facts
  cli.py               `nexus run` / `nexus history`
  tools/
    registry.py        typed dispatch + OpenAI schema export
    builtin.py         sandboxed fs/sh/web tools
tests/                 48 tests — engine, memory, tools, loop, CLI, malformed
scripts/e2e_live.py    live end-to-end smoke vs a real model
docs/                  architecture.md, ethics.md, demo UI + e2e_demo.json
```

## Status

MVP complete and green: 48 unit tests, live E2E passed. Roadmap: browser tool
(CDP), multi-agent fan-out (planner/researcher/coder/reviewer), web UI.

## License

MIT
