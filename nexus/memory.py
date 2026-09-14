"""SQLite episodic memory: episodes, event log, durable facts."""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    objective TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'running',
    report TEXT,
    created REAL NOT NULL,
    finished REAL
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id INTEGER NOT NULL REFERENCES episodes(id),
    ts REAL NOT NULL,
    kind TEXT NOT NULL,
    data TEXT
);
CREATE TABLE IF NOT EXISTS facts (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated REAL NOT NULL
);
"""


class Memory:
    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        fresh = not Path(self.db_path).exists()
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        if fresh:
            self.conn.commit()

    # -- episodes -----------------------------------------------------------
    def start_episode(self, objective: str) -> dict:
        cur = self.conn.execute(
            "INSERT INTO episodes (objective, created) VALUES (?, ?)",
            (objective, time.time()),
        )
        self.conn.commit()
        return self._episode(cur.lastrowid)

    def finish_episode(self, episode_id: int, status: str, report: str | None = None) -> None:
        self.conn.execute(
            "UPDATE episodes SET status=?, report=?, finished=? WHERE id=?",
            (status, report, time.time(), episode_id),
        )
        self.conn.commit()

    def _episode(self, row_id: int) -> dict:
        row = self.conn.execute("SELECT * FROM episodes WHERE id=?", (row_id,)).fetchone()
        return dict(row)

    def recent_episodes(self, limit: int = 20) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM episodes ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    # -- events -------------------------------------------------------------
    def log_event(self, episode_id: int, kind: str, data: dict | None = None) -> None:
        self.conn.execute(
            "INSERT INTO events (episode_id, ts, kind, data) VALUES (?, ?, ?, ?)",
            (episode_id, time.time(), kind, json.dumps(data) if data is not None else None),
        )
        self.conn.commit()

    def episode_events(self, episode_id: int) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM events WHERE episode_id=? ORDER BY id", (episode_id,)
        ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            if d["data"] is not None:
                d["data"] = json.loads(d["data"])
            out.append(d)
        return out

    # -- durable facts --------------------------------------------------------
    def remember_fact(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO facts (key, value, updated) VALUES (?, ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated=excluded.updated",
            (key, value, time.time()),
        )
        self.conn.commit()

    def recall_facts(self, key: str) -> str | None:
        row = self.conn.execute("SELECT value FROM facts WHERE key=?", (key,)).fetchone()
        return row["value"] if row else None

    def all_facts(self) -> dict[str, str]:
        rows = self.conn.execute("SELECT key, value FROM facts ORDER BY key").fetchall()
        return {r["key"]: r["value"] for r in rows}

    def close(self) -> None:
        self.conn.close()
