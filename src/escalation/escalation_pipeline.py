"""
Escalation Pipeline — Module 3.

Implements the mandatory routing rules:
    LOW / MEDIUM  -> persistent DB log only
    HIGH / CRITICAL -> persistent DB log AND a real-time alert push

Real-time alert mechanism
--------------------------
The alert "queue" is the `alerts` table in SQLite (see db/database.py):
inserting a row = pushing to the queue, acknowledging a row = consuming it.
This is a deliberate scope decision for a 3-day take-home: Streamlit reruns
its script from scratch on every interaction, so there is no long-lived
in-process pub/sub channel to push into across the dashboard's own process
boundary. A polling read against a persistent queue table is the simplest
mechanism that is still genuinely real-time (sub-second latency at this
event volume) and trivially swappable for a real WebSocket/SSE channel if
this were ever upgraded to a non-Streamlit frontend (see README "Production
upgrade path").

Multiple simultaneous violations in the same clip
---------------------------------------------------
Each detection from Module 1 is escalated independently and gets its own
event_id / report / (if applicable) alert. If a single clip contains both a
LOW and a CRITICAL event, both are logged, and only the CRITICAL one also
raises an alert -- the pipeline does not collapse multiple violations into a
single severity for the clip, since the policy's routing rule is defined
per-violation, not per-clip.
"""
from severity.severity_matrix import assign_severity, Severity
from reports.report_generator import build_report, append_to_csv
from db import database


ESCALATION_ACTIONS = {
    Severity.LOW: "Logged to DB",
    Severity.MEDIUM: "Logged to DB",
    Severity.HIGH: "Real-time alert triggered + DB log",
    Severity.CRITICAL: "Real-time alert triggered + DB log",
}


def process_detection(detection, policy_rules_by_class, db_path=database.DEFAULT_DB_PATH):
    """
    Runs one raw detection through severity classification, report
    generation, persistence, and (if applicable) real-time alerting.
    Returns the finished report dict.
    """
    severity = assign_severity(detection, policy_rules_by_class)
    escalation_action = ESCALATION_ACTIONS[severity]

    report = build_report(detection, severity, escalation_action, policy_rules_by_class)

    database.insert_event(report, db_path=db_path)
    append_to_csv(report)

    if severity in (Severity.HIGH, Severity.CRITICAL):
        message = (
            f"{severity.value} — {report['behavior_class']} detected "
            f"in {report['zone']} (clip {report['clip_id']})"
        )
        database.insert_alert(report, message, db_path=db_path)

    return report
