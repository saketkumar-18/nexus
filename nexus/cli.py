"""nexus command-line entry: `python -m nexus run "objective"`."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from nexus import __version__


def run_agent(objective: str, workspace: str | None = None, db: str | None = None,
              max_steps: int = 25) -> dict:
    """Build a full agent from env/config and run one objective. Returns result dict."""
    from nexus.agent import Agent
    from nexus.engine import engine_from_env
    from nexus.memory import Memory
    from nexus.tools.builtin import build_tools

    engine = engine_from_env()
    workspace = workspace or os.environ.get("NEXUS_WORKSPACE", "./nexus-workspace")
    db = db or os.environ.get("NEXUS_DB", "nexus.db")
    mem = Memory(db_path=db)
    try:
        reg = build_tools(workspace=workspace)
        agent = Agent(engine=engine, registry=reg, memory=mem,
                      workspace=Path(workspace), verifier=engine, max_steps=max_steps)
        return agent.run(objective)
    finally:
        mem.close()


def _cmd_run(args) -> int:
    result = run_agent(
        args.objective,
        workspace=args.workspace or os.environ.get("NEXUS_WORKSPACE"),
        db=args.db or os.environ.get("NEXUS_DB"),
        max_steps=args.max_steps,
    )
    if args.as_json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"episode #{result['episode_id']} — {result['status']}")
        print()
        print(result["report"])
        if result.get("critique"):
            print()
            print(f"verifier: {result['critique']}")
    return 0 if result["status"] == "done" else 1


def _cmd_history(args) -> int:
    from nexus.memory import Memory

    mem = Memory(db_path=args.db or os.environ.get("NEXUS_DB", "nexus.db"))
    try:
        for ep in mem.recent_episodes(limit=args.limit):
            print(f"[{ep['id']:>4}] {ep['status']:<8} {ep['objective']}")
    finally:
        mem.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(
        prog="nexus", description="NEXUS — local-first autonomous computer agent")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="cmd")

    run_p = sub.add_parser("run", help="run an objective")
    run_p.add_argument("--json", action="store_true", dest="as_json",
                       help="emit machine-readable JSON")
    run_p.add_argument("--db", help="memory DB path (default NEXUS_DB / nexus.db)")
    run_p.add_argument("--workspace", help="workspace dir (default NEXUS_WORKSPACE)")
    run_p.add_argument("--max-steps", type=int, default=25)
    run_p.add_argument("objective")
    run_p.set_defaults(fn=_cmd_run)

    hist_p = sub.add_parser("history", help="list recent episodes")
    hist_p.add_argument("--db")
    hist_p.add_argument("--limit", type=int, default=20)
    hist_p.set_defaults(fn=_cmd_history)

    args = parser.parse_args(argv)
    if not getattr(args, "fn", None):
        parser.print_usage(sys.stderr)
        return 2
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
