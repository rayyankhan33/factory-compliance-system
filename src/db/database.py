"""
Persistent storage layer used by Module 3 (Escalation), Module 4 (Reports),
and Module 5 (Dashboard). SQLite is sufficient at this data volume (per-clip
violation events, not high-frequency telemetry) and requires no separate
server process, which matters for a take-home reviewer who just wants to
clone and run.
"""
import sqlite3
import json
from pathlib import Path
from contextlib import contextmanager

DEFAULT_DB_PATH = Path(__file__).resolve().parents[2] / "outputs" / "compliance.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS compliance_events (
    event_id            TEXT PRIMARY KEY,
    timestamp           TEXT NOT NULL,        -- ISO 8601, wall-clock detection time
    clip_id              TEXT NOT NULL,
    clip_timestamp_sec   REAL,                  -- offset within the source clip
    zone                 TEXT,
    class_id             INTEGER NOT NULL,
    behavior_class       TEXT NOT NULL,
    policy_rule_ref       TEXT NOT NULL,
    event_description    TEXT NOT NULL,
    severity              TEXT NOT NULL,
    escalation_action     TEXT NOT NULL,
    context_json          TEXT                  -- raw detection context, for debugging/audit
);

CREATE TABLE IF NOT EXISTS alerts (
    alert_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id      TEXT NOT NULL REFERENCES compliance_events(event_id),
    created_at    TEXT NOT NULL,
    severity      TEXT NOT NULL,
    clip_id       TEXT NOT NULL,
    message       TEXT NOT NULL,
    acknowledged  INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_events_timestamp ON compliance_events(timestamp);
CREATE INDEX IF NOT EXISTS idx_events_severity ON compliance_events(severity);
CREATE INDEX IF NOT EXISTS idx_events_class ON compliance_events(class_id);
CREATE INDEX IF NOT EXISTS idx_alerts_ack ON alerts(acknowledged);
"""


@contextmanager
def get_connection(db_path=DEFAULT_DB_PATH):
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_db(db_path=DEFAULT_DB_PATH):
    with get_connection(db_path) as conn:
        conn.executescript(SCHEMA)
        conn.commit()


def insert_event(event, db_path=DEFAULT_DB_PATH):
    """event: dict matching the Module 4 report schema (+ optional context dict)."""
    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO compliance_events
                (event_id, timestamp, clip_id, clip_timestamp_sec, zone, class_id,
                 behavior_class, policy_rule_ref, event_description, severity,
                 escalation_action, context_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event["event_id"],
                event["timestamp"],
                event["clip_id"],
                event.get("clip_timestamp_sec"),
                event.get("zone"),
                event["class_id"],
                event["behavior_class"],
                event["policy_rule_ref"],
                event["event_description"],
                event["severity"],
                event["escalation_action"],
                json.dumps(event.get("context", {})),
            ),
        )
        conn.commit()


def insert_alert(event, message, db_path=DEFAULT_DB_PATH):
    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO alerts (event_id, created_at, severity, clip_id, message, acknowledged)
            VALUES (?, ?, ?, ?, ?, 0)
            """,
            (event["event_id"], event["timestamp"], event["severity"], event["clip_id"], message),
        )
        conn.commit()


def get_active_alerts(db_path=DEFAULT_DB_PATH):
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM alerts WHERE acknowledged = 0 ORDER BY created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]


def acknowledge_alert(alert_id, db_path=DEFAULT_DB_PATH):
    with get_connection(db_path) as conn:
        conn.execute("UPDATE alerts SET acknowledged = 1 WHERE alert_id = ?", (alert_id,))
        conn.commit()


def query_events(db_path=DEFAULT_DB_PATH, start_date=None, end_date=None,
                  severities=None, class_ids=None, clip_id=None, limit=1000):
    query = "SELECT * FROM compliance_events WHERE 1=1"
    params = []
    if start_date:
        query += " AND timestamp >= ?"
        params.append(start_date)
    if end_date:
        query += " AND timestamp <= ?"
        params.append(end_date)
    if severities:
        query += f" AND severity IN ({','.join('?' * len(severities))})"
        params.extend(severities)
    if class_ids:
        query += f" AND class_id IN ({','.join('?' * len(class_ids))})"
        params.extend(class_ids)
    if clip_id:
        query += " AND clip_id = ?"
        params.append(clip_id)
    query += " ORDER BY timestamp DESC LIMIT ?"
    params.append(limit)

    with get_connection(db_path) as conn:
        rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]


def get_latest_status_per_clip(db_path=DEFAULT_DB_PATH):
    """Used by the dashboard's Live Feed Monitor view."""
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT clip_id, behavior_class, severity, timestamp
            FROM compliance_events
            WHERE (clip_id, timestamp) IN (
                SELECT clip_id, MAX(timestamp) FROM compliance_events GROUP BY clip_id
            )
            """
        ).fetchall()
        return {r["clip_id"]: dict(r) for r in rows}
