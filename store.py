"""SQLite-lagring av økter og transkriberte segmenter."""
from __future__ import annotations

import os
import sqlite3
import threading
import time
from typing import Optional

from config import DATA_DIR

DB_PATH = os.path.join(DATA_DIR, "lytt.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    title       TEXT NOT NULL,
    started_at  REAL NOT NULL,
    ended_at    REAL,
    status      TEXT NOT NULL DEFAULT 'recording'   -- recording | paused | stopped
);
CREATE TABLE IF NOT EXISTS segments (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    t_start     REAL NOT NULL,    -- sekunder fra øktstart
    t_end       REAL NOT NULL,
    source      TEXT NOT NULL,    -- mic | system
    speaker     TEXT NOT NULL,
    language    TEXT,
    text        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_segments_session ON segments(session_id, t_start);
"""


class Store:
    def __init__(self, path: str = DB_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._lock = threading.RLock()
        self._con = sqlite3.connect(path, check_same_thread=False)
        self._con.row_factory = sqlite3.Row
        self._con.execute("PRAGMA journal_mode=WAL")
        self._con.execute("PRAGMA foreign_keys=ON")
        self._con.executescript(SCHEMA)
        # Økter som lå åpne da programmet sist ble avsluttet, markeres som stoppet.
        self._con.execute(
            "UPDATE sessions SET status='stopped', ended_at=COALESCE(ended_at, started_at) "
            "WHERE status != 'stopped'"
        )
        self._con.commit()

    # ---------- økter ----------
    def create_session(self, title: Optional[str] = None) -> dict:
        now = time.time()
        title = title or time.strftime("Meeting %Y-%m-%d %H:%M", time.localtime(now))
        with self._lock:
            cur = self._con.execute(
                "INSERT INTO sessions(title, started_at, status) VALUES (?,?, 'recording')",
                (title, now),
            )
            self._con.commit()
            return self.get_session(cur.lastrowid)

    def set_status(self, session_id: int, status: str) -> None:
        with self._lock:
            if status == "stopped":
                self._con.execute(
                    "UPDATE sessions SET status=?, ended_at=? WHERE id=?",
                    (status, time.time(), session_id),
                )
            else:
                self._con.execute("UPDATE sessions SET status=? WHERE id=?", (status, session_id))
            self._con.commit()

    def rename_session(self, session_id: int, title: str) -> None:
        with self._lock:
            self._con.execute("UPDATE sessions SET title=? WHERE id=?", (title.strip() or "Untitled", session_id))
            self._con.commit()

    def delete_session(self, session_id: int) -> None:
        with self._lock:
            self._con.execute("DELETE FROM segments WHERE session_id=?", (session_id,))
            self._con.execute("DELETE FROM sessions WHERE id=?", (session_id,))
            self._con.commit()

    def get_session(self, session_id: int) -> Optional[dict]:
        with self._lock:
            row = self._con.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
            return dict(row) if row else None

    def list_sessions(self) -> list[dict]:
        with self._lock:
            rows = self._con.execute(
                """
                SELECT s.*, COUNT(g.id) AS n_segments,
                       COALESCE(MAX(g.t_end), 0) AS duration,
                       COALESCE(SUM(LENGTH(g.text)), 0) AS n_chars
                FROM sessions s LEFT JOIN segments g ON g.session_id = s.id
                GROUP BY s.id ORDER BY s.started_at DESC
                """
            ).fetchall()
            return [dict(r) for r in rows]

    # ---------- segmenter ----------
    def add_segment(self, seg: dict) -> dict:
        with self._lock:
            cur = self._con.execute(
                "INSERT INTO segments(session_id, t_start, t_end, source, speaker, language, text) "
                "VALUES (?,?,?,?,?,?,?)",
                (seg["session_id"], seg["t_start"], seg["t_end"], seg["source"],
                 seg["speaker"], seg.get("language"), seg["text"]),
            )
            self._con.commit()
            seg = dict(seg)
            seg["id"] = cur.lastrowid
            return seg

    def get_segments(self, session_id: int) -> list[dict]:
        with self._lock:
            rows = self._con.execute(
                "SELECT * FROM segments WHERE session_id=? ORDER BY t_start, id", (session_id,)
            ).fetchall()
            return [dict(r) for r in rows]

    def rename_speaker(self, session_id: int, old: str, new: str) -> None:
        new = new.strip()
        if not new:
            return
        with self._lock:
            self._con.execute(
                "UPDATE segments SET speaker=? WHERE session_id=? AND speaker=?", (new, session_id, old)
            )
            self._con.commit()

    def delete_segment(self, segment_id: int) -> None:
        with self._lock:
            self._con.execute("DELETE FROM segments WHERE id=?", (segment_id,))
            self._con.commit()

    def update_segment_text(self, segment_id: int, text: str) -> None:
        with self._lock:
            self._con.execute("UPDATE segments SET text=? WHERE id=?", (text.strip(), segment_id))
            self._con.commit()

    # ---------- eksport ----------
    def transcript_text(self, session_id: int, with_speakers: bool = True, with_time: bool = False) -> str:
        """Slår sammen påfølgende segmenter fra samme taler til avsnitt."""
        segs = self.get_segments(session_id)
        lines: list[str] = []
        cur_speaker = None
        cur_parts: list[str] = []
        cur_start = 0.0

        def flush():
            if not cur_parts:
                return
            body = " ".join(cur_parts).strip()
            if with_speakers:
                stamp = f"[{_fmt(cur_start)}] " if with_time else ""
                lines.append(f"{stamp}{cur_speaker}: {body}")
            else:
                lines.append(body)

        for s in segs:
            if s["speaker"] != cur_speaker:
                flush()
                cur_speaker = s["speaker"]
                cur_parts = []
                cur_start = s["t_start"]
            cur_parts.append(s["text"])
        flush()
        return "\n\n".join(lines)


def _fmt(sec: float) -> str:
    sec = int(sec)
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"
