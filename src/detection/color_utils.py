"""
Color & shape heuristics for the Detection Engine (Module 1).

These implement the *observable indicators* the policy itself names as the
basis for classification (Section 8): vest color, walkway floor-marking
color, panel open/closed state, and block count on a forklift's forks.
None of this requires a trained model — it is classical CV grounded
directly in the policy's stated visual indicators.

CALIBRATION NOTE (read this before relying on real footage):
All thresholds below are reasonable *starting points*, not measured from the
real dataset (this development environment could not download the Kaggle
clips — see README "Known Limitations"). Before trusting outputs on real
clips, sample a handful of frames per behavior class and re-tune the
constants in `CONFIG` against the actual pixel values you observe (a tiny
script for this -- `tools/calibrate_colors.py` -- is included).
"""
import cv2
import numpy as np

CONFIG = {
    # Green used for both the safety vest and the walkway floor markings.
    # OpenCV HSV ranges: H in [0,179], S,V in [0,255].
    "green_hsv_lower": (40, 60, 40),
    "green_hsv_upper": (85, 255, 255),

    # Red component of the red-black general vest. (Black alone is too
    # ambiguous against shadows, so we key on the red patches.)
    "red_hsv_lower_1": (0, 70, 50),
    "red_hsv_upper_1": (10, 255, 255),
    "red_hsv_lower_2": (170, 70, 50),
    "red_hsv_upper_2": (179, 255, 255),

    "min_color_fraction": 0.06,   # min fraction of bbox pixels to call a color "present"

    # Panel open/closed heuristic thresholds (Canny edge density + grayscale stddev).
    "panel_edge_density_open_threshold": 0.06,
    "panel_color_std_open_threshold": 35,

    # Forklift fork-region block counting.
    "block_min_contour_area": 400,
    "block_max_contour_area": 8000,
    "single_block_area_estimate": 1800,
}


def _hsv(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


def _crop(frame, bbox):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def classify_vest_color(frame, bbox, config=CONFIG):
    """Returns 'green' (authorized), 'red_black' (general/unauthorized), or 'unknown'."""
    crop = _crop(frame, bbox)
    if crop is None or crop.size == 0:
        return "unknown"
    hsv = _hsv(crop)
    total = hsv.shape[0] * hsv.shape[1]

    green_mask = cv2.inRange(hsv, np.array(config["green_hsv_lower"]), np.array(config["green_hsv_upper"]))
    green_frac = cv2.countNonZero(green_mask) / total

    red_mask_1 = cv2.inRange(hsv, np.array(config["red_hsv_lower_1"]), np.array(config["red_hsv_upper_1"]))
    red_mask_2 = cv2.inRange(hsv, np.array(config["red_hsv_lower_2"]), np.array(config["red_hsv_upper_2"]))
    red_frac = (cv2.countNonZero(red_mask_1) + cv2.countNonZero(red_mask_2)) / total

    if green_frac >= config["min_color_fraction"] and green_frac > red_frac:
        return "green"
    if red_frac >= config["min_color_fraction"] and red_frac > green_frac:
        return "red_black"
    return "unknown"


def build_walkway_polygon(frame, config=CONFIG):
    """
    Detects green floor-marking pixels and returns a convex hull polygon
    (np.ndarray of points) approximating the Designated Safe Walkway
    interior. Returns None if no green floor markings are found in frame.
    """
    hsv = _hsv(frame)
    mask = cv2.inRange(hsv, np.array(config["green_hsv_lower"]), np.array(config["green_hsv_upper"]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    points = cv2.findNonZero(mask)
    if points is None or len(points) < 10:
        return None
    return cv2.convexHull(points)


def is_point_in_walkway(point, walkway_polygon):
    """point: (x, y). Returns True if inside/on the walkway polygon."""
    if walkway_polygon is None:
        return None  # unknown -- no walkway markings detected in this frame
    result = cv2.pointPolygonTest(walkway_polygon, (float(point[0]), float(point[1])), False)
    return result >= 0


def classify_panel_state(frame, bbox, config=CONFIG):
    """Returns 'open' or 'closed' for an electrical panel bounding box."""
    crop = _crop(frame, bbox)
    if crop is None or crop.size == 0:
        return "closed"
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = cv2.countNonZero(edges) / edges.size
    color_std = float(np.std(gray))

    if (edge_density >= config["panel_edge_density_open_threshold"] or
            color_std >= config["panel_color_std_open_threshold"]):
        return "open"
    return "closed"


def count_blocks_on_forks(frame, bbox, config=CONFIG):
    """
    Estimates the number of standardized blocks within the fork region of a
    forklift bounding box via contour counting. Returns an int.
    """
    crop = _crop(frame, bbox)
    if crop is None or crop.size == 0:
        return 0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 120)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    total_area = 0
    count = 0
    for c in contours:
        area = cv2.contourArea(c)
        if config["block_min_contour_area"] <= area <= config["block_max_contour_area"]:
            count += 1
            total_area += area

    if count == 0 and total_area == 0:
        return 0
    # If blocks are stacked/touching, contours may merge into fewer, larger
    # blobs -- fall back to an area-based estimate in that case.
    area_estimate = round(total_area / config["single_block_area_estimate"])
    return max(count, area_estimate)
