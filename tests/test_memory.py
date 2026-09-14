"""RED tests for SQLite episodic memory."""
import pytest

from nexus.memory import Memory


@pytest.fixture()
def mem(tmp_path):
    return Memory(db_path=tmp_path / "mem.db")


def test_start_episode_returns_id_and_running_status(mem):
    ep = mem.start_episode("clean downloads folder")
    assert ep["id"] >= 1
    assert ep["status"] == "running"
    assert mem.recent_episodes()[0]["objective"] == "clean downloads folder"


def test_log_and_fetch_events(mem):
    ep = mem.start_episode("obj")
    mem.log_event(ep["id"], kind="plan", data={"tasks": ["a", "b"]})
    mem.log_event(ep["id"], kind="tool", data={"tool": "sh", "rc": 0})
    events = mem.episode_events(ep["id"])
    assert [e["kind"] for e in events] == ["plan", "tool"]
    assert events[1]["data"] == {"tool": "sh", "rc": 0}
    assert events[0]["ts"] <= events[1]["ts"]


def test_finish_episode_sets_status_and_report(mem):
    ep = mem.start_episode("obj")
    mem.finish_episode(ep["id"], status="done", report="all good")
    got = mem.recent_episodes()[0]
    assert got["status"] == "done"
    assert got["report"] == "all good"


def test_fact_upsert_and_recall(mem):
    mem.remember_fact("preferred_stack", "python")
    mem.remember_fact("preferred_stack", "python+tsx")
    assert mem.recall_facts("preferred_stack") == "python+tsx"
    assert mem.recall_facts("never_set") is None


def test_persists_across_connections(tmp_path):
    db = tmp_path / "mem.db"
    m1 = Memory(db_path=db)
    m1.remember_fact("k", "v")
    m2 = Memory(db_path=db)
    assert m2.recall_facts("k") == "v"


def test_failed_episode_visible_in_recent(mem):
    ep = mem.start_episode("x")
    mem.finish_episode(ep["id"], status="failed", report="boom")
    assert mem.recent_episodes()[0]["status"] == "failed"
