"""
database.py
------------
Thin SQLite wrapper. Two tables:

  traffic_log   - one row per (road, signal-cycle) snapshot: counts,
                  density, green time actually granted, timestamp.
  vehicle_events- one row per individual vehicle crossing the line
                  (for fine-grained analytics / CSV export).

All timestamps are stored as ISO date + time strings so SQLite's plain
TEXT columns can be queried/sorted lexicographically without extensions.
"""

import csv
import sqlite3
from contextlib import contextmanager
from typing import Iterable, Optional

import config
from utils import get_logger, now_str, today_str

logger = get_logger(__name__)


SCHEMA = """
CREATE TABLE IF NOT EXISTS traffic_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    date            TEXT NOT NULL,
    time            TEXT NOT NULL,
    road            TEXT NOT NULL,
    vehicle_count   INTEGER NOT NULL,
    density         TEXT NOT NULL,
    signal_time     INTEGER NOT NULL,
    emergency       INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS vehicle_events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    date            TEXT NOT NULL,
    time            TEXT NOT NULL,
    road            TEXT NOT NULL,
    track_id        INTEGER NOT NULL,
    vehicle_class   TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_traffic_log_date ON traffic_log(date);
CREATE INDEX IF NOT EXISTS idx_vehicle_events_date ON vehicle_events(date);
"""


class Database:
    def __init__(self, db_path: str = None):
        self.db_path = db_path or config.DATABASE_PATH
        self._init_schema()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self):
        with self._connect() as conn:
            conn.executescript(SCHEMA)
        logger.info(f"Database ready at {self.db_path}")

    # ------------------------------------------------------------------
    def log_signal_cycle(self, road: str, vehicle_count: int, density: str,
                          signal_time: int, emergency: bool = False):
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO traffic_log (date, time, road, vehicle_count, density, signal_time, emergency)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (today_str(), now_str().split(" ")[1], road, vehicle_count, density,
                 signal_time, int(emergency)),
            )

    def log_vehicle_event(self, road: str, track_id: int, vehicle_class: str):
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO vehicle_events (date, time, road, track_id, vehicle_class)
                   VALUES (?, ?, ?, ?, ?)""",
                (today_str(), now_str().split(" ")[1], road, track_id, vehicle_class),
            )

    # ------------------------------------------------------------------
    def fetch_traffic_log(self, date: Optional[str] = None) -> list:
        with self._connect() as conn:
            if date:
                rows = conn.execute(
                    "SELECT * FROM traffic_log WHERE date = ? ORDER BY id", (date,)
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM traffic_log ORDER BY id").fetchall()
            return [dict(r) for r in rows]

    def fetch_vehicle_events(self, date: Optional[str] = None) -> list:
        with self._connect() as conn:
            if date:
                rows = conn.execute(
                    "SELECT * FROM vehicle_events WHERE date = ? ORDER BY id", (date,)
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM vehicle_events ORDER BY id").fetchall()
            return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    def export_csv(self, table: str, out_path: str):
        """table must be 'traffic_log' or 'vehicle_events'."""
        if table not in ("traffic_log", "vehicle_events"):
            raise ValueError("table must be 'traffic_log' or 'vehicle_events'")

        rows = self.fetch_traffic_log() if table == "traffic_log" else self.fetch_vehicle_events()
        if not rows:
            logger.warning(f"No rows to export for {table}")
            return None

        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        logger.info(f"Exported {len(rows)} rows from {table} to {out_path}")
        return out_path
