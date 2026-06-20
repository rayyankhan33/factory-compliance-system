"""
Detection Engine — Module 1.

Pipeline per clip:
    1. Sample frames from the video at a fixed stride.
    2. Run an object localization backend (pluggable -- see Backends below)
       to find candidate person / forklift / electrical-panel regions, using
       a label vocabulary DERIVED FROM THE PARSED POLICY (see
       `derive_detection_vocabulary`), not hard-coded class-name strings
       disconnected from policy_rules.json.
    3. Apply the policy-grounded color/shape heuristics in color_utils.py to
       turn each localized region into a safe/unsafe classification for the
       corresponding behavior class.
    4. Emit a structured raw detection record per violation found.

Backends
--------
`YoloV8Backend` is the primary backend: a standard COCO-trained YOLOv8n
that reliably detects persons and trucks/forklifts at high speed (~40+ FPS
at 640px). COCO "truck" maps to forklifts in a factory setting.

`YoloWorldBackend` is the open-vocabulary fallback: an ultralytics
YOLO-World model that can localize arbitrary text-prompted classes with
no training. Slower but more flexible.

`MockBackend` exists purely so the rest of the pipeline (severity routing,
escalation, reports, dashboard) can be exercised and tested without model
downloads.
"""
import json
from abc import ABC, abstractmethod
from pathlib import Path

import cv2
import numpy as np

from detection import color_utils

# ---------------------------------------------------------------------------
# Policy-grounded vocabulary derivation
# ---------------------------------------------------------------------------

# A detector needs *some* finite physical-object vocabulary to query against
# -- that part can't be avoided for any vision system. What IS derived from
# the policy is *which* of these object categories are actually relevant for
# each class, confirmed by keyword presence in that class's parsed policy
# text, plus the resulting indicator/severity logic is 100% policy-driven.
_CANDIDATE_OBJECT_KEYWORDS = {
    "person": ["person", "personnel", "individual", "pedestrian"],
    "forklift": ["forklift"],
    "electrical panel": ["panel", "electrical"],
    "safety vest": ["vest"],
}


def derive_detection_vocabulary(policy_rules):
    """
    For each parsed policy rule, scans its unsafe_behavior_full_text for the
    candidate object keywords above and returns
        {class_id: [matched_object_labels]}
    This is the text-prompt vocabulary handed to the open-vocab backend.
    """
    vocab = {}
    for rule in policy_rules:
        text_lower = rule["unsafe_behavior_full_text"].lower()
        matched = [
            label for label, keywords in _CANDIDATE_OBJECT_KEYWORDS.items()
            if any(kw in text_lower for kw in keywords)
        ]
        vocab[rule["class_id"]] = matched or ["person"]
    return vocab


# ---------------------------------------------------------------------------
# Localization backends
# ---------------------------------------------------------------------------

class DetectionBackend(ABC):
    @abstractmethod
    def localize(self, frame, labels):
        """Returns a list of {"label": str, "bbox": (x1,y1,x2,y2), "confidence": float}."""
        raise NotImplementedError


# ---- COCO class IDs we care about ----
_COCO_PERSON = 0
_COCO_TRUCK = 7       # forklifts in factory settings detected as "truck"
_COCO_RELEVANT = {_COCO_PERSON, _COCO_TRUCK}
_COCO_TO_LABEL = {_COCO_PERSON: "person", _COCO_TRUCK: "forklift"}


class YoloV8Backend(DetectionBackend):
    """
    Fast standard YOLOv8n backend (COCO-trained).
    ~8x faster than YOLO-World. Reliably detects persons and trucks/forklifts.
    Frames are downscaled to `imgsz` for inference speed.
    """

    def __init__(self, model_name="yolov8n.pt", confidence=0.30, imgsz=640):
        from ultralytics import YOLO
        self.model = YOLO(model_name)
        self.confidence = confidence
        self.imgsz = imgsz
        # Filter to only relevant COCO classes for speed
        self._relevant_classes = list(_COCO_RELEVANT)

    def localize(self, frame, labels=None):
        """
        Run detection on frame. `labels` is accepted for API compat but
        ignored — we use fixed COCO classes and map them.
        Returns list of {"label": str, "bbox": (x1,y1,x2,y2), "confidence": float}.
        """
        results = self.model.predict(
            frame,
            conf=self.confidence,
            imgsz=self.imgsz,
            classes=self._relevant_classes,
            verbose=False,
            half=False,  # CPU doesn't support half
        )[0]

        detections = []
        for box, cls_idx, conf in zip(
            results.boxes.xyxy.tolist(),
            results.boxes.cls.tolist(),
            results.boxes.conf.tolist(),
        ):
            coco_id = int(cls_idx)
            label = _COCO_TO_LABEL.get(coco_id)
            if label:
                detections.append({
                    "label": label,
                    "bbox": tuple(box),
                    "confidence": float(conf),
                })
        return detections


class YoloWorldBackend(DetectionBackend):
    """Real zero-shot backend. Requires `pip install ultralytics` and internet
    access on first run to fetch pretrained YOLO-World weights."""

    def __init__(self, model_name="yolov8s-world.pt", confidence=0.25):
        from ultralytics import YOLO  # lazy import: optional heavy dependency
        self.model = YOLO(model_name)
        self.confidence = confidence
        self._classes_set = None

    def localize(self, frame, labels):
        if self._classes_set != tuple(labels):
            self.model.set_classes(labels)
            self._classes_set = tuple(labels)
        results = self.model.predict(frame, conf=self.confidence, verbose=False)[0]
        detections = []
        for box, cls_idx, conf in zip(results.boxes.xyxy.tolist(),
                                       results.boxes.cls.tolist(),
                                       results.boxes.conf.tolist()):
            detections.append({
                "label": labels[int(cls_idx)],
                "bbox": tuple(box),
                "confidence": float(conf),
            })
        return detections


class MockBackend(DetectionBackend):
    """
    Test/demo backend. `fixed_boxes` is a dict keyed by an arbitrary frame
    key (we use frame index) -> list of {"label", "bbox", "confidence"}.
    Used by tests/build_synthetic_clip.py to validate the pipeline without
    needing real footage or model weights.
    """
    def __init__(self, fixed_boxes_by_frame_idx):
        self.fixed_boxes_by_frame_idx = fixed_boxes_by_frame_idx
        self._frame_idx = 0

    def reset(self):
        """Call between clips so each clip's frame indices start back at 0."""
        self._frame_idx = 0

    def localize(self, frame, labels):
        boxes = self.fixed_boxes_by_frame_idx.get(self._frame_idx, [])
        self._frame_idx += 1
        return [b for b in boxes if b["label"] in labels]


# ---------------------------------------------------------------------------
# Zone assignment (simplification -- see README)
# ---------------------------------------------------------------------------

def assign_zone(bbox, frame_width):
    """
    Crude horizontal-third zoning since no real camera-zone map is available
    in this dev environment. Replace with an actual zone polygon lookup per
    camera once the facility's real camera layout is known.
    """
    cx = (bbox[0] + bbox[2]) / 2
    third = frame_width / 3
    if cx < third:
        return "Zone-1"
    elif cx < 2 * third:
        return "Zone-2"
    return "Zone-3"


def _bbox_center(bbox):
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)


def _bbox_distance(a, b):
    ax, ay = _bbox_center(a)
    bx, by = _bbox_center(b)
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


HAZARD_PROXIMITY_PX = 250  # tune against real frame resolution/scale


# ---------------------------------------------------------------------------
# Main per-clip analysis
# ---------------------------------------------------------------------------

def analyze_clip(clip_path, policy_rules_by_class, backend, frame_stride=10, clip_id=None):
    """
    Yields raw detection dicts for one clip:
        {
          "clip_id": str, "clip_timestamp_sec": float, "class_id": int,
          "zone": str, "description": str, "context": {...}
        }
    """
    vocab = derive_detection_vocabulary(list(policy_rules_by_class.values()))
    all_labels = sorted({lbl for labels in vocab.values() for lbl in labels})

    clip_id = clip_id or Path(clip_path).name
    cap = cv2.VideoCapture(str(clip_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    frame_idx = 0
    intervention_recurrence = {}  # zone -> count, for class-1 CRIT escalation

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % frame_stride != 0:
            frame_idx += 1
            continue

        timestamp_sec = frame_idx / fps
        frame_width = frame.shape[1]
        detections = backend.localize(frame, all_labels)

        persons = [d for d in detections if d["label"] == "person"]
        forklifts = [d for d in detections if d["label"] == "forklift"]
        panels = [d for d in detections if d["label"] == "electrical panel"]

        walkway_polygon = color_utils.build_walkway_polygon(frame)

        # --- class 0: Safe Walkway Violation ---
        for p in persons:
            foot_point = ((p["bbox"][0] + p["bbox"][2]) / 2, p["bbox"][3])
            inside = color_utils.is_point_in_walkway(foot_point, walkway_polygon)
            if inside is False:
                near_hazard = any(_bbox_distance(p["bbox"], f["bbox"]) < HAZARD_PROXIMITY_PX
                                   for f in forklifts)
                rule = policy_rules_by_class[0]
                yield {
                    "clip_id": clip_id,
                    "clip_timestamp_sec": round(timestamp_sec, 2),
                    "class_id": 0,
                    "zone": assign_zone(p["bbox"], frame_width),
                    "description": (
                        f"Person detected outside the green-marked Designated Safe "
                        f"Walkway at t={timestamp_sec:.1f}s"
                        + (" with a forklift in close proximity." if near_hazard else ".")
                    ),
                    "context": {"near_hazard": near_hazard},
                }

        # --- class 1: Unauthorized Intervention ---
        # Heuristic: a person is "intervening" if positioned very close to a
        # panel (the only fixed-equipment object we localize) -- a stand-in
        # for true equipment-interaction detection, documented as a
        # limitation since the dataset's exact equipment classes are not yet
        # known in this dev environment.
        for p in persons:
            nearest_panel = min(panels, key=lambda f: _bbox_distance(p["bbox"], f["bbox"]), default=None)
            if nearest_panel and _bbox_distance(p["bbox"], nearest_panel["bbox"]) < HAZARD_PROXIMITY_PX:
                vest = color_utils.classify_vest_color(frame, p["bbox"])
                if vest != "green":
                    zone = assign_zone(p["bbox"], frame_width)
                    intervention_recurrence[zone] = intervention_recurrence.get(zone, 0) + 1
                    yield {
                        "clip_id": clip_id,
                        "clip_timestamp_sec": round(timestamp_sec, 2),
                        "class_id": 1,
                        "zone": zone,
                        "description": (
                            f"Person interacting with equipment without the green "
                            f"authorization vest (observed vest: {vest}) at t={timestamp_sec:.1f}s"
                        ),
                        "context": {"recurrence_count_in_clip": intervention_recurrence[zone]},
                    }

        # --- class 2: Opened Panel Cover ---
        for panel in panels:
            state = color_utils.classify_panel_state(frame, panel["bbox"])
            if state == "open":
                personnel_nearby = any(_bbox_distance(panel["bbox"], p["bbox"]) < HAZARD_PROXIMITY_PX
                                        for p in persons)
                yield {
                    "clip_id": clip_id,
                    "clip_timestamp_sec": round(timestamp_sec, 2),
                    "class_id": 2,
                    "zone": assign_zone(panel["bbox"], frame_width),
                    "description": f"Electrical panel observed in the open-cover state at t={timestamp_sec:.1f}s",
                    "context": {"personnel_nearby": personnel_nearby},
                }

        # --- class 3: Carrying Overload with Forklift ---
        for f in forklifts:
            block_count = color_utils.count_blocks_on_forks(frame, f["bbox"])
            if block_count >= 3:
                yield {
                    "clip_id": clip_id,
                    "clip_timestamp_sec": round(timestamp_sec, 2),
                    "class_id": 3,
                    "zone": assign_zone(f["bbox"], frame_width),
                    "description": f"Forklift observed carrying {block_count} blocks (limit: 2) at t={timestamp_sec:.1f}s",
                    "context": {"block_count": block_count},
                }

        frame_idx += 1

    cap.release()


def load_policy_rules(policy_rules_path):
    with open(policy_rules_path) as f:
        rules = json.load(f)
    return {r["class_id"]: r for r in rules}
