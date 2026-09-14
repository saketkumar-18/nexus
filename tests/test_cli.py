"""RED tests for the CLI entry point."""
import io
import json
import sys

import pytest

from nexus import cli


class FakeRunner:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def __call__(self, objective, **kw):
        self.calls.append((objective, kw))
        return self.result


def test_run_command_prints_report_and_exits_zero(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("NEXUS_DB", str(tmp_path / "db.sqlite"))
    monkeypatch.setenv("NEXUS_WORKSPACE", str(tmp_path / "ws"))
    runner = FakeRunner({"episode_id": 1, "status": "done",
                         "report": "organized downloads", "critique": ""})
    monkeypatch.setattr(cli, "run_agent", runner, raising=False)
    rc = cli.main(["run", "clean downloads"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "organized downloads" in out
    assert runner.calls[0][0] == "clean downloads"
    assert runner.calls[0][1]["workspace"] == str(tmp_path / "ws")


def test_failed_episode_exits_nonzero(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("NEXUS_DB", str(tmp_path / "db.sqlite"))
    monkeypatch.setenv("NEXUS_WORKSPACE", str(tmp_path / "ws"))
    monkeypatch.setattr(
        cli, "run_agent",
        FakeRunner({"episode_id": 2, "status": "failed",
                    "report": "max steps reached", "critique": None}),
        raising=False)
    assert cli.main(["run", "impossible"]) == 1


def test_json_flag_emits_parseable_json(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("NEXUS_DB", str(tmp_path / "db.sqlite"))
    monkeypatch.setenv("NEXUS_WORKSPACE", str(tmp_path / "ws"))
    monkeypatch.setattr(
        cli, "run_agent",
        FakeRunner({"episode_id": 3, "status": "done", "report": "r", "critique": ""}),
        raising=False)
    rc = cli.main(["run", "--json", "do something"])
    payload = json.loads(capsys.readouterr().out)
    assert rc == 0 and payload["status"] == "done" and payload["episode_id"] == 3


def test_history_lists_episodes(tmp_path, capsys):
    from nexus.memory import Memory
    monkeypatch_db = tmp_path / "h.sqlite"
    mem = Memory(db_path=monkeypatch_db)
    mem.start_episode("obj one")
    ep = mem.start_episode("obj two")
    mem.finish_episode(ep["id"], status="done", report="fin")
    mem.close()
    rc = cli.main(["history", "--db", str(monkeypatch_db)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "obj one" in out and "running" in out
    assert "obj two" in out and "done" in out


def test_no_args_shows_usage(capsys):
    assert cli.main([]) == 2
    assert "usage" in capsys.readouterr().err.lower()
