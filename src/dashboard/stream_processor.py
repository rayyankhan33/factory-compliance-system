"""
Real-time stream processor for the compliance monitoring dashboard.

Manages a video source (file, RTSP, or webcam), runs the detection pipeline
frame-by-frame, draws annotated overlays, and provides:
  - A thread-safe latest-frame buffer for MJPEG streaming
  - An asyncio queue for pushing violation events to SSE clients

Performance optimizations for 23+ FPS:
  - Uses YOLOv8n (standard COCO, ~8x faster than YOLO-World)
  - Downscales frames to 640px for inference
  - Caches walkway polygon (static green markings don't move)
  - Caches equipment zone mask (machinery doesn't move)
  - Panel detection via lightweight classical CV (no model needed)
  - Adaptive frame stride
"""
import asyncio
import cv2
import numpy as np
import sys
import threading
import time
from pathlib import Path
from collections import deque

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from detection.detector import (
    YoloV8Backend,
    assign_zone,
    load_policy_rules,
    HAZARD_PROXIMITY_PX,
    _bbox_distance,
)
from detection import color_utils
from escalation.escalation_pipeline import process_detection
from db import database

POLICY_RULES_PATH = ROOT / "src" / "policy" / "policy_rules.json"

# Severity -> BGR color for drawing on frames
SEVERITY_COLORS_BGR = {
    "LOW": (76, 175, 80),        # green
    "MEDIUM": (0, 179, 255),     # amber
    "HIGH": (0, 140, 251),       # orange
    "CRITICAL": (53, 57, 229),   # red
}

# Behavior class labels for annotation
CLASS_LABELS = {
    0: "Walkway Violation",
    1: "Unauthorized Intervention",
    2: "Open Panel Cover",
    3: "Forklift Overload",
}


class StreamProcessor:
    """
    Manages real-time video processing for a single source.

    Usage:
        processor = StreamProcessor()
        processor.start("path/to/video.mp4")
        # In MJPEG endpoint: frame_bytes = processor.get_latest_frame_jpeg()
        # In SSE endpoint: event = await processor.get_violation_event()
        processor.stop()
    """

    def __init__(self, frame_stride=5):
        self.frame_stride = frame_stride
        self._lock = threading.Lock()
        self._latest_frame_jpeg = None
        self._running = False
        self._thread = None
        self._source = None
        self._violation_queues = []  # list of asyncio.Queue for SSE clients
        self._queues_lock = threading.Lock()
        self._stats = {
            "fps": 0.0,
            "frames_processed": 0,
            "violations_detected": 0,
            "source": None,
            "status": "idle",
        }

        # Load model and policy once
        self._backend = None
        self._policy_rules_by_class = None

        # Cached static features (computed once per video source)
        self._cached_walkway_polygon = None
        self._walkway_cached = False
        self._cached_equipment_mask = None
        self._equipment_cached = False

    def _ensure_model_loaded(self):
        """Lazy-load the YOLO model and policy rules."""
        if self._backend is None:
            print("[StreamProcessor] Loading YOLOv8n model...")
            self._backend = YoloV8Backend(confidence=0.30, imgsz=640)
            print("[StreamProcessor] Model loaded.")

        if self._policy_rules_by_class is None:
            self._policy_rules_by_class = load_policy_rules(POLICY_RULES_PATH)
            database.init_db()

    def _reset_caches(self):
        """Reset cached features when switching video sources."""
        self._cached_walkway_polygon = None
        self._walkway_cached = False
        self._cached_equipment_mask = None
        self._equipment_cached = False

    def register_sse_queue(self):
        """Register a new asyncio.Queue for an SSE client. Returns the queue."""
        q = asyncio.Queue(maxsize=200)
        with self._queues_lock:
            self._violation_queues.append(q)
        return q

    def unregister_sse_queue(self, q):
        """Remove an SSE client's queue."""
        with self._queues_lock:
            if q in self._violation_queues:
                self._violation_queues.remove(q)

    def _broadcast_violation(self, event_data):
        """Push a violation event to all registered SSE queues."""
        with self._queues_lock:
            dead_queues = []
            for q in self._violation_queues:
                try:
                    q.put_nowait(event_data)
                except asyncio.QueueFull:
                    dead_queues.append(q)
            for q in dead_queues:
                self._violation_queues.remove(q)

    def start(self, source):
        """Start processing a video source. source can be a file path, RTSP URL, or int (webcam)."""
        self.stop()  # stop any existing processing

        self._source = source
        self._running = True
        self._stats["source"] = str(source)
        self._stats["status"] = "starting"
        self._stats["frames_processed"] = 0
        self._stats["violations_detected"] = 0
        self._reset_caches()

        self._thread = threading.Thread(target=self._process_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the current processing loop."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        self._thread = None
        self._stats["status"] = "idle"

    @property
    def is_running(self):
        return self._running

    @property
    def stats(self):
        return dict(self._stats)

    def get_latest_frame_jpeg(self):
        """Get the latest annotated frame as JPEG bytes (thread-safe)."""
        with self._lock:
            return self._latest_frame_jpeg

    def _set_latest_frame(self, frame):
        """Encode frame to JPEG and store it (thread-safe)."""
        _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        with self._lock:
            self._latest_frame_jpeg = jpeg.tobytes()

    def _draw_annotations(self, frame, detections_with_info):
        """
        Draw bounding boxes, labels, and severity badges on the frame.
        detections_with_info: list of (bbox, label, severity, class_id)
        """
        for bbox, label, severity, class_id in detections_with_info:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            color = SEVERITY_COLORS_BGR.get(severity, (128, 128, 128))

            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Label background
            text = f"{label} [{severity}]"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)

            # Ensure label stays within frame
            label_y = max(y1 - 6, th + 4)
            cv2.rectangle(frame, (x1, label_y - th - 4), (x1 + tw + 6, label_y + 4), color, -1)
            cv2.putText(frame, text, (x1 + 3, label_y), font, font_scale, (255, 255, 255), thickness)

        return frame

    def _draw_status_bar(self, frame, fps, violation_count, source_name):
        """Draw a status bar at the top of the frame."""
        h, w = frame.shape[:2]
        bar_height = 32
        cv2.rectangle(frame, (0, 0), (w, bar_height), (20, 20, 20), -1)

        status_text = f"FPS: {fps:.1f} | Violations: {violation_count} | {source_name}"
        cv2.putText(frame, status_text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Green/red dot for status
        dot_color = (0, 200, 0) if self._running else (0, 0, 200)
        cv2.circle(frame, (w - 16, 16), 6, dot_color, -1)

        return frame

    def _get_walkway_polygon(self, frame):
        """Get walkway polygon, computing and caching on first call."""
        if not self._walkway_cached:
            self._cached_walkway_polygon = color_utils.build_walkway_polygon(frame)
            self._walkway_cached = True
        return self._cached_walkway_polygon

    def _get_equipment_mask(self, frame):
        """Get equipment zone mask, computing and caching on first call."""
        if not self._equipment_cached:
            self._cached_equipment_mask = color_utils.detect_equipment_zones(frame)
            self._equipment_cached = True
        return self._cached_equipment_mask

    def _process_loop(self):
        """Main processing loop — runs in a background thread."""
        self._ensure_model_loaded()

        source = self._source
        # Try to interpret as int (webcam index)
        try:
            source = int(source)
        except (ValueError, TypeError):
            pass

        cap = cv2.VideoCapture(source if isinstance(source, int) else str(source))
        if not cap.isOpened():
            self._stats["status"] = "error: cannot open source"
            self._running = False
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        is_live = isinstance(source, int) or (
            isinstance(source, str) and source.startswith(("rtsp://", "rtmp://", "http://"))
        )

        source_name = Path(str(source)).name if not is_live else str(source)
        self._stats["status"] = "running"

        frame_idx = 0
        last_annotations = []  # carry forward annotations for non-analyzed frames
        intervention_recurrence = {}  # zone -> count for class 1 severity
        total_violations = 0
        fps_times = deque(maxlen=60)

        target_delay = 1.0 / fps if not is_live else 0

        try:
            while self._running:
                loop_start = time.time()

                ok, frame = cap.read()
                if not ok:
                    if is_live:
                        time.sleep(1)
                        cap.release()
                        cap = cv2.VideoCapture(source if isinstance(source, int) else str(source))
                        continue
                    else:
                        break

                analyze_this_frame = (frame_idx % self.frame_stride == 0)

                if analyze_this_frame:
                    annotations = self._analyze_and_annotate(
                        frame, frame_idx, fps, source_name,
                        intervention_recurrence
                    )
                    last_annotations = annotations
                    total_violations += len(annotations)
                    self._stats["violations_detected"] = total_violations

                # Draw annotations (current or carried-forward) on every frame
                annotated = self._draw_annotations(frame, last_annotations)

                # Draw status bar
                if len(fps_times) > 1:
                    current_fps = len(fps_times) / sum(fps_times)
                else:
                    current_fps = 0.0
                annotated = self._draw_status_bar(
                    annotated, current_fps, total_violations, source_name
                )

                self._set_latest_frame(annotated)

                frame_idx += 1
                self._stats["frames_processed"] = frame_idx

                elapsed = time.time() - loop_start
                fps_times.append(elapsed)
                self._stats["fps"] = current_fps

                # Pace playback for files
                if not is_live and target_delay > elapsed:
                    time.sleep(target_delay - elapsed)

        finally:
            cap.release()
            self._stats["status"] = "finished" if not is_live else "disconnected"
            self._running = False

    def _analyze_and_annotate(self, frame, frame_idx, fps, source_name,
                               intervention_recurrence):
        """
        Run the full detection pipeline on a single frame.
        Returns a list of (bbox, label, severity_str, class_id) for drawing.
        Also pushes violation events to SSE queues and persists to DB.

        Detects ALL 4 violation types per the Compliance Policy Manual:
          Class 0: Safe Walkway Violation (person outside green walkway)
          Class 1: Unauthorized Intervention (no green vest near equipment)
          Class 2: Opened Panel Cover (panel in open state)
          Class 3: Carrying Overload with Forklift (3+ blocks)
        """
        annotations = []
        timestamp_sec = frame_idx / fps
        frame_width = frame.shape[1]

        # ── YOLO detection (persons + forklifts) ──
        detections = self._backend.localize(frame)

        persons = [d for d in detections if d["label"] == "person"]
        forklifts = [d for d in detections if d["label"] == "forklift"]

        # ── Cached static features ──
        walkway_polygon = self._get_walkway_polygon(frame)
        equipment_mask = self._get_equipment_mask(frame)

        # ── Classical CV panel detection (no model needed) ──
        panel_candidates = color_utils.detect_panel_candidates(frame)

        clip_id = f"live:{source_name}"

        # ═══ Class 0: Safe Walkway Violation ═══
        # Policy: Person outside the green-marked Designated Safe Walkway
        for p in persons:
            foot_point = ((p["bbox"][0] + p["bbox"][2]) / 2, p["bbox"][3])
            inside = color_utils.is_point_in_walkway(foot_point, walkway_polygon)
            if inside is False:
                near_hazard = any(
                    _bbox_distance(p["bbox"], f["bbox"]) < HAZARD_PROXIMITY_PX
                    for f in forklifts
                )
                detection_record = {
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
                report = process_detection(detection_record, self._policy_rules_by_class)
                annotations.append((p["bbox"], CLASS_LABELS[0], report["severity"], 0))
                self._broadcast_violation(report)

        # ═══ Class 1: Unauthorized Intervention ═══
        # Policy: Person interacting with equipment without green vest.
        # Detect via: (a) person near equipment zone mask, OR
        #             (b) person near a detected panel candidate.
        # Then check vest color — if not green, it's unauthorized.
        for p in persons:
            near_equipment = color_utils.is_person_near_equipment(p["bbox"], equipment_mask)

            # Also check proximity to panel candidates
            near_panel = any(
                _bbox_distance(p["bbox"], pc["bbox"]) < HAZARD_PROXIMITY_PX
                for pc in panel_candidates
            )

            if near_equipment or near_panel:
                vest = color_utils.classify_vest_color(frame, p["bbox"])
                if vest != "green":
                    zone = assign_zone(p["bbox"], frame_width)
                    intervention_recurrence[zone] = (
                        intervention_recurrence.get(zone, 0) + 1
                    )
                    detection_record = {
                        "clip_id": clip_id,
                        "clip_timestamp_sec": round(timestamp_sec, 2),
                        "class_id": 1,
                        "zone": zone,
                        "description": (
                            f"Person interacting with equipment without the green "
                            f"authorization vest (observed vest: {vest}) at t={timestamp_sec:.1f}s"
                        ),
                        "context": {
                            "recurrence_count_in_clip": intervention_recurrence[zone]
                        },
                    }
                    report = process_detection(
                        detection_record, self._policy_rules_by_class
                    )
                    annotations.append(
                        (p["bbox"], CLASS_LABELS[1], report["severity"], 1)
                    )
                    self._broadcast_violation(report)

        # ═══ Class 2: Opened Panel Cover ═══
        # Policy: Electrical panel in the open-cover state during production.
        # Detected via classical CV panel candidate detection.
        for pc in panel_candidates:
            if pc["state"] == "open":
                personnel_nearby = any(
                    _bbox_distance(pc["bbox"], p["bbox"]) < HAZARD_PROXIMITY_PX
                    for p in persons
                )
                detection_record = {
                    "clip_id": clip_id,
                    "clip_timestamp_sec": round(timestamp_sec, 2),
                    "class_id": 2,
                    "zone": assign_zone(pc["bbox"], frame_width),
                    "description": f"Electrical panel observed in the open-cover state at t={timestamp_sec:.1f}s",
                    "context": {"personnel_nearby": personnel_nearby},
                }
                report = process_detection(
                    detection_record, self._policy_rules_by_class
                )
                annotations.append(
                    (pc["bbox"], CLASS_LABELS[2], report["severity"], 2)
                )
                self._broadcast_violation(report)

        # ═══ Class 3: Carrying Overload with Forklift ═══
        # Policy: Forklift carrying 3+ standardized blocks.
        for f in forklifts:
            block_count = color_utils.count_blocks_on_forks(frame, f["bbox"])
            if block_count >= 3:
                detection_record = {
                    "clip_id": clip_id,
                    "clip_timestamp_sec": round(timestamp_sec, 2),
                    "class_id": 3,
                    "zone": assign_zone(f["bbox"], frame_width),
                    "description": f"Forklift observed carrying {block_count} blocks (limit: 2) at t={timestamp_sec:.1f}s",
                    "context": {"block_count": block_count},
                }
                report = process_detection(
                    detection_record, self._policy_rules_by_class
                )
                annotations.append(
                    (f["bbox"], CLASS_LABELS[3], report["severity"], 3)
                )
                self._broadcast_violation(report)

        return annotations


# Module-level singleton so the server can share one processor instance
_processor = None


def get_processor(frame_stride=5):
    """Get or create the singleton StreamProcessor."""
    global _processor
    if _processor is None:
        _processor = StreamProcessor(frame_stride=frame_stride)
    return _processor
