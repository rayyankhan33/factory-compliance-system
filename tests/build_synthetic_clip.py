"""
Builds a synthetic test clip (and a matching MockBackend boxes config) with
KNOWN ground truth, used to validate that severity/escalation/report/
dashboard wiring is correct end-to-end -- independent of real-world
detection accuracy, which cannot be verified until the real Kaggle dataset
is available (see README "Known Limitations").

The MockBackend supplies object LOCATION (bounding boxes) directly -- the
one part that genuinely requires a trained/zero-shot vision model on real
footage. Everything downstream of localization (vest color classification,
walkway-boundary check, panel open/closed heuristic, block counting,
severity assignment, escalation routing, report writing) runs the exact
same real code as production and is genuinely exercised by this test.

Run:
    python tests/build_synthetic_clip.py
    python src/pipeline.py --backend mock --mock-config tests/synthetic_boxes.json --frame-stride 5 --data-dir tests/synthetic_data
"""
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "tests" / "synthetic_data"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CLIP_PATH = OUT_DIR / "synthetic_test_clip.mp4"
BOXES_PATH = ROOT / "tests" / "synthetic_boxes.json"

WIDTH, HEIGHT = 900, 700
WALKWAY_Y_TOP = 480
FPS = 10
TOTAL_FRAMES = 40  # with frame_stride=5 -> 8 sampled frames, scenario indices 0..7

GREEN = (0, 200, 0)      # BGR -- matches color_utils green_hsv range
RED = (0, 0, 200)        # BGR -- matches color_utils red_hsv range
GRAY = (190, 190, 190)
DARK = (40, 40, 40)


def base_frame():
    frame = np.full((HEIGHT, WIDTH, 3), GRAY, dtype=np.uint8)
    frame[WALKWAY_Y_TOP:HEIGHT, :] = GREEN  # Designated Safe Walkway floor marking
    return frame


def draw_filled(frame, bbox, color):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)


def draw_noisy_panel(frame, bbox, rng):
    """Simulates an OPEN panel: high edge density / color variance inside the bbox."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    noise = rng.integers(0, 255, size=(y2 - y1, x2 - x1, 3), dtype=np.uint8)
    frame[y1:y2, x1:x2] = noise


def draw_blocks(frame, fork_bbox, n_blocks):
    """Draws n_blocks small bordered squares inside the fork region (CLOSED panel
    style flat background first, then distinct-edged squares for contour counting)."""
    x1, y1, x2, y2 = [int(v) for v in fork_bbox]
    frame[y1:y2, x1:x2] = (210, 210, 210)
    block_size = 42  # area ~1764px, close to color_utils CONFIG["single_block_area_estimate"]=1800
    gap = 15
    cx = x1 + 10
    for i in range(n_blocks):
        bx1 = cx + i * (block_size + gap)
        bx2 = bx1 + block_size
        by1 = y1 + 10
        by2 = by1 + block_size
        if bx2 > x2:
            break
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), DARK, -1)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (255, 255, 255), 2)


def build():
    rng = np.random.default_rng(42)
    writer = cv2.VideoWriter(str(CLIP_PATH), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))

    boxes_by_call_index = {}

    # idx0 (frame 0): person outside walkway, no forklift nearby -> class0 MEDIUM
    person_bbox_0 = (350, 250, 450, 420)
    boxes_by_call_index[0] = [
        {"label": "person", "bbox": person_bbox_0, "confidence": 0.9},
    ]

    # idx1 (frame 5): same person + forklift nearby -> class0 HIGH (near_hazard)
    person_bbox_1 = (350, 250, 450, 420)
    forklift_bbox_1 = (350, 200, 500, 420)
    boxes_by_call_index[1] = [
        {"label": "person", "bbox": person_bbox_1, "confidence": 0.9},
        {"label": "forklift", "bbox": forklift_bbox_1, "confidence": 0.85},
    ]

    # idx2 (frame 10): person (red-black vest) near panel, Zone-1 -> class1 HIGH (1st occurrence)
    person_bbox_2 = (50, 300, 150, 500)
    panel_bbox_2 = (200, 300, 320, 500)  # near person (proximity check) but non-overlapping pixels
    boxes_by_call_index[2] = [
        {"label": "person", "bbox": person_bbox_2, "confidence": 0.9},
        {"label": "electrical panel", "bbox": panel_bbox_2, "confidence": 0.8},
    ]

    # idx3 (frame 15): same Zone-1 unauthorized intervention again -> class1 CRITICAL (recurrence=2)
    boxes_by_call_index[3] = [
        {"label": "person", "bbox": person_bbox_2, "confidence": 0.9},
        {"label": "electrical panel", "bbox": panel_bbox_2, "confidence": 0.8},
    ]

    # idx4 (frame 20): open panel, no one nearby -> class2 LOW
    panel_bbox_4 = (600, 100, 750, 300)
    boxes_by_call_index[4] = [
        {"label": "electrical panel", "bbox": panel_bbox_4, "confidence": 0.8},
    ]

    # idx5 (frame 25): open panel + authorized (green-vest) person nearby -> class2 MEDIUM
    # (person has green vest so this frame does NOT also fire class1)
    person_bbox_5 = (760, 150, 830, 320)
    boxes_by_call_index[5] = [
        {"label": "electrical panel", "bbox": panel_bbox_4, "confidence": 0.8},
        {"label": "person", "bbox": person_bbox_5, "confidence": 0.9},
    ]

    # idx6 (frame 30): forklift carrying 4 blocks -> class3 CRITICAL
    forklift_bbox_6 = (100, 500, 420, 620)
    boxes_by_call_index[6] = [
        {"label": "forklift", "bbox": forklift_bbox_6, "confidence": 0.85},
    ]

    # idx7 (frame 35): forklift carrying 2 blocks (safe) -> no violation expected
    forklift_bbox_7 = (100, 500, 420, 620)
    boxes_by_call_index[7] = [
        {"label": "forklift", "bbox": forklift_bbox_7, "confidence": 0.85},
    ]

    for frame_idx in range(TOTAL_FRAMES):
        frame = base_frame()

        if frame_idx == 0:
            draw_filled(frame, person_bbox_0, (150, 100, 50))  # arbitrary person color, vest irrelevant for class0
        elif frame_idx == 5:
            draw_filled(frame, person_bbox_1, (150, 100, 50))
            draw_filled(frame, forklift_bbox_1, (90, 60, 30))
        elif frame_idx in (10, 15):
            draw_filled(frame, panel_bbox_2, (160, 160, 160))   # neutral panel (closed-looking), irrelevant here
            draw_filled(frame, person_bbox_2, RED)               # red-black vest -> classified red_black
        elif frame_idx == 20:
            draw_noisy_panel(frame, panel_bbox_4, rng)
        elif frame_idx == 25:
            draw_noisy_panel(frame, panel_bbox_4, rng)
            draw_filled(frame, person_bbox_5, GREEN)             # green vest -> authorized, no class1 noise
        elif frame_idx == 30:
            draw_blocks(frame, forklift_bbox_6, n_blocks=4)
        elif frame_idx == 35:
            draw_blocks(frame, forklift_bbox_7, n_blocks=2)

        writer.write(frame)

    writer.release()

    with open(BOXES_PATH, "w") as f:
        json.dump({str(k): v for k, v in boxes_by_call_index.items()}, f, indent=2)

    print(f"Wrote {CLIP_PATH}")
    print(f"Wrote {BOXES_PATH}")
    print("\nExpected results when run through the pipeline:")
    print("  frame  0 -> class0 Safe Walkway Violation     -> MEDIUM")
    print("  frame  5 -> class0 Safe Walkway Violation     -> HIGH   (near forklift)")
    print("  frame 10 -> class1 Unauthorized Intervention  -> HIGH   (1st occurrence, Zone-1)")
    print("  frame 15 -> class1 Unauthorized Intervention  -> CRITICAL (2nd occurrence, Zone-1)")
    print("  frame 20 -> class2 Opened Panel Cover         -> LOW    (no personnel nearby)")
    print("  frame 25 -> class2 Opened Panel Cover         -> MEDIUM (personnel nearby)")
    print("  frame 30 -> class3 Carrying Overload          -> CRITICAL (4 blocks)")
    print("  frame 35 -> (no event -- 2 blocks is Safe Carrying, correctly NOT flagged)")


if __name__ == "__main__":
    build()
