"""Live E2E smoke test: NEXUS vs a real local model (Ollama, OpenAI-compatible).

Runs a real objective through the full loop (plan -> tools -> verify -> report),
asserts the workspace was actually changed, and dumps the episode to
docs/e2e_demo.json for the demo replay UI.

Usage:
    python scripts/e2e_live.py [model]

Requires Ollama on :11434. Exits non-zero on any failure.
"""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nexus.agent import Agent  # noqa: E402
from nexus.engine import LLMEngine  # noqa: E402
from nexus.memory import Memory  # noqa: E402
from nexus.tools.builtin import build_tools  # noqa: E402

MODEL = sys.argv[1] if len(sys.argv) > 1 else "llama3.2:1b"
WORKSPACE = ROOT / "workspace-e2e"
DB = ROOT / "workspace-e2e" / "nexus.db"

OBJECTIVE = (
    "In the workspace: (1) write a file named notes.txt containing exactly the text "
    "'nexus was here', (2) read it back to verify the content, then report."
)


def main() -> int:
    if WORKSPACE.exists():
        shutil.rmtree(WORKSPACE)
    WORKSPACE.mkdir(parents=True)

    engine = LLMEngine(base_url="http://localhost:11434/v1", api_key="ollama", model=MODEL)
    mem = Memory(db_path=DB)
    reg = build_tools(workspace=WORKSPACE)
    agent = Agent(engine=engine, registry=reg, memory=mem,
                  workspace=WORKSPACE, verifier=engine, max_steps=30)

    print(f"=== NEXUS live E2E — model: {MODEL} ===")
    result = agent.run(OBJECTIVE)
    print(f"status: {result['status']}  episode: {result['episode_id']}")
    print(f"report: {result['report']}")
    if result.get("critique"):
        print(f"verifier: {result['critique']}")

    # ground truth: did the agent actually do the work?
    notes = WORKSPACE / "notes.txt"
    file_ok = notes.is_file() and notes.read_text(encoding="utf-8").strip() == "nexus was here"
    print(f"workspace ground truth: notes.txt correct={file_ok}")
    tool_events = [e for e in mem.episode_events(result["episode_id"]) if e["kind"] == "tool"]
    print(f"tool calls executed: {len(tool_events)}")
    for e in tool_events:
        d = e["data"]
        print(f"  - {d.get('tool')} ok={d.get('ok')} args={str(d.get('arguments'))[:80]}")

    # dump episode for the demo replay UI
    dump = {
        "model": MODEL,
        "objective": OBJECTIVE,
        "result": {k: result[k] for k in ("episode_id", "status", "report", "critique")},
        "events": [
            {"ts": e["ts"], "kind": e["kind"], "data": e["data"]}
            for e in mem.episode_events(result["episode_id"])
        ],
        "ground_truth": {"notes_txt_correct": file_ok},
    }
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "e2e_demo.json").write_text(json.dumps(dump, indent=2), encoding="utf-8")

    mem.close()
    passed = result["status"] == "done" and file_ok and len(tool_events) >= 1
    print(f"=== E2E {'PASSED' if passed else 'FAILED'} ===")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
