"""
Severity Categorization Matrix — Module 2.

Assigns one of LOW / MEDIUM / HIGH / CRITICAL to each raw detection produced
by Module 1 (the Detection Engine). The base tier per behavior class is
derived from policy signals in policy_rules.json (callout_type and
high_frequency_flagged), and is then adjusted up or down by detection-time
context (personnel proximity, recurrence within the same clip) following the
LOW/MED/HIGH/CRIT criteria defined in the assignment brief.

Rationale per class (documented in README, summarized here):

  class 0 — Safe Walkway Violation (WARNING, flagged high-frequency)
      Policy explicitly frames this as a *behavioral* deviation, not a state
      condition, and the WARNING text ties it to "proximity to forklift and
      machinery hazards" rather than to certain injury. -> default MEDIUM.
      Escalates to HIGH when the detector's context confirms the person is
      actually near a forklift/machine bounding box (i.e. the hazard named
      in the WARNING is concretely present), matching the brief's HIGH
      definition: "active unsafe behavior with concurrent personnel exposure".

  class 1 — Unauthorized Intervention (CRITICAL SAFETY NOTICE)
      By definition this event only exists while a person is actively
      touching/adjusting equipment, so personnel exposure is concurrent and
      guaranteed at the moment of detection. -> default HIGH.
      Escalates to CRITICAL if the same person/zone shows repeated
      unauthorized interventions within one clip, matching the brief's CRIT
      criterion of "high-frequency recurrence".

  class 2 — Opened Panel Cover (WARNING)
      Policy explicitly states this is an unsafe state "regardless of...
      whether personnel are in the immediate vicinity" -- i.e. the textbook
      example the brief gives for LOW ("a state-based finding... with no
      concurrent personnel exposure"). -> default LOW.
      Escalates to MEDIUM only if a person is detected near the open panel
      in-frame (personnel now present, but not in the act of an unsafe
      behavior themselves, matching MEDIUM).

  class 3 — Carrying Overload with Forklift (CRITICAL SAFETY NOTICE)
      This is the only rule in the entire document with explicit "will
      trigger an immediate alert" language tied to an unambiguous, purely
      quantifiable threshold (3+ blocks). There is no policy text suggesting
      this should ever be downgraded. -> always CRITICAL.
"""
from enum import Enum


class Severity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


def load_policy_rules_by_class(policy_rules):
    return {r["class_id"]: r for r in policy_rules}


def assign_severity(detection, policy_rules_by_class):
    """
    detection: dict with at least:
        - class_id (int 0-3)
        - context: dict of detection-time signals, may include:
            near_hazard (bool)            - class 0: person bbox close to a forklift/machine bbox
            recurrence_count_in_clip (int) - class 1: count of unauthorized-intervention
                                              events for the same person/zone in this clip
            personnel_nearby (bool)        - class 2: a person detected near the open panel

    Returns a Severity value. Falls back to MEDIUM with a flag if class_id is
    unrecognized (should not happen if policy parsing succeeded), so the
    pipeline never silently drops an event.
    """
    rule = policy_rules_by_class.get(detection["class_id"])
    context = detection.get("context", {})

    if rule is None:
        return Severity.MEDIUM

    class_id = detection["class_id"]

    if class_id == 0:  # Safe Walkway Violation
        if context.get("near_hazard"):
            return Severity.HIGH
        return Severity.MEDIUM

    if class_id == 1:  # Unauthorized Intervention
        if context.get("recurrence_count_in_clip", 1) >= 2:
            return Severity.CRITICAL
        return Severity.HIGH

    if class_id == 2:  # Opened Panel Cover
        if context.get("personnel_nearby"):
            return Severity.MEDIUM
        return Severity.LOW

    if class_id == 3:  # Carrying Overload with Forklift
        return Severity.CRITICAL

    # Generic policy-driven fallback for any class not explicitly handled
    # above (keeps the module extensible if the policy is amended with a
    # 5th domain in future): use the callout type as the base signal.
    if rule["callout_type"] == "CRITICAL_SAFETY_NOTICE":
        return Severity.HIGH
    if rule["callout_type"] == "WARNING":
        return Severity.MEDIUM
    return Severity.LOW
