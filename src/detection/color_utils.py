"""
Color & shape heuristics for the Detection Engine (Module 1).

These implement the *observable indicators* the policy itself names as the
basis for classification (Section 8): vest color, walkway floor-marking
color, panel open/closed state, and block count on a forklift's forks.
None of this requires a trained model — it is classical CV grounded
directly in the policy's stated visual indicators.

Includes equipment zone detection and panel candidate detection via
classical CV (contour analysis), enabling violations to be detected
without relying on an open-vocabulary model for these object types.
"""
import cv2
import numpy as np

CONFIG = {
    # Green used for both the safety vest and the walkway floor markings.
    # OpenCV HSV ranges: H in [0,179], S,V in [0,255].
    "green_hsv_lower": (35, 40, 35),
    "green_hsv_upper": (90, 255, 255),

    # Red component of the red-black general vest. (Black alone is too
    # ambiguous against shadows, so we key on the red patches.)
    "red_hsv_lower_1": (0, 50, 40),
    "red_hsv_upper_1": (12, 255, 255),
    "red_hsv_lower_2": (165, 50, 40),
    "red_hsv_upper_2": (179, 255, 255),

    "min_color_fraction": 0.04,   # min fraction of bbox pixels to call a color "present"

    # Panel open/closed heuristic thresholds (Canny edge density + grayscale stddev).
    # Raised from 0.04/28 to reduce false positives from textured surfaces.
    "panel_edge_density_open_threshold": 0.08,
    "panel_color_std_open_threshold": 40,

    # Orange block color range (the standardized blocks are distinctly orange).
    "orange_hsv_lower": (5, 100, 100),
    "orange_hsv_upper": (25, 255, 255),
    # Minimum orange blob area as fraction of forklift bbox to count as a block.
    "block_min_orange_area_ratio": 0.008,

    # Equipment zone detection
    "equipment_edge_density_threshold": 0.12,
    "equipment_min_area": 3000,

    # Panel candidate detection
    "panel_min_area": 3000,      # raised from 1500 to filter small labels
    "panel_max_area": 80000,
    "panel_min_aspect": 0.4,
    "panel_max_aspect": 3.0,
    "panel_max_y_ratio": 0.65,    # reject candidates below this (ground level)
    "panel_min_rectangularity": 0.55,  # contour area / bounding rect area
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


# ---------------------------------------------------------------------------
# Vest color classification (for Class 1: Unauthorized Intervention)
# ---------------------------------------------------------------------------

def classify_vest_color(frame, bbox, config=CONFIG):
    """Returns 'green' (authorized), 'red_black' (general/unauthorized), or 'unknown'."""
    # Focus on the upper body (torso region) for better vest detection
    x1, y1, x2, y2 = [int(v) for v in bbox]
    box_h = y2 - y1
    # Vest is typically in the upper 40-65% of a person bbox
    torso_y1 = y1 + int(box_h * 0.15)
    torso_y2 = y1 + int(box_h * 0.60)
    torso_bbox = (x1, torso_y1, x2, torso_y2)

    crop = _crop(frame, torso_bbox)
    if crop is None or crop.size == 0:
        # Fallback to full bbox
        crop = _crop(frame, bbox)
        if crop is None or crop.size == 0:
            return "unknown"

    hsv = _hsv(crop)
    total = hsv.shape[0] * hsv.shape[1]
    if total == 0:
        return "unknown"

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


# ---------------------------------------------------------------------------
# Walkway mask (for Class 0: Safe Walkway Violation)
# ---------------------------------------------------------------------------

def build_walkway_polygon(frame, config=CONFIG):
    """
    Detects green floor-marking pixels and returns a dilated binary MASK
    (np.ndarray, same H×W as frame) where 255 = walkway zone.

    The green walkway markings on the production floor define the permitted
    pedestrian path. We detect green pixels with tighter saturation to avoid
    picking up vests and other non-floor green objects, then heavily dilate
    to fill the walkway corridor between the marking lines.

    Returns None if no green floor markings are found.
    """
    hsv = _hsv(frame)
    h, w = frame.shape[:2]

    # Tighter green range focused on floor markings (higher saturation
    # to filter out vests and foliage; floor paint is more vivid)
    mask = cv2.inRange(
        hsv,
        np.array((40, 80, 50)),
        np.array((80, 255, 255)),
    )

    # Focus on the lower 80% of the frame (floor markings are on the ground)
    mask[:int(h * 0.2), :] = 0

    # Morphological cleanup
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    green_pixels = cv2.countNonZero(mask)
    if green_pixels < 200:
        return None

    # Heavily dilate to fill the walkway corridor between marking lines.
    # The walkway is the AREA BETWEEN the green lines, so we expand each
    # green line outward to cover the entire permissible walkway region.
    dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (80, 80))
    walkway_mask = cv2.dilate(mask, dilation_kernel, iterations=1)

    return walkway_mask


def is_point_in_walkway(point, walkway_mask):
    """
    point: (x, y). Returns True if the point falls within the walkway mask.
    walkway_mask: binary mask (H×W) where 255 = walkway zone, or None.
    """
    if walkway_mask is None:
        return None  # unknown -- no walkway markings detected in this frame
    x, y = int(round(point[0])), int(round(point[1]))
    h, w = walkway_mask.shape[:2]
    if 0 <= x < w and 0 <= y < h:
        return bool(walkway_mask[y, x] > 0)
    return False  # out of frame = outside walkway


# ---------------------------------------------------------------------------
# Equipment zone detection (for Class 1: Unauthorized Intervention)
# ---------------------------------------------------------------------------

def detect_equipment_zones(frame, config=CONFIG):
    """
    Detect static equipment/machinery zones using edge density analysis.
    Factory equipment has high edge density (complex geometry, pipes, wires).
    Returns a binary mask where True = equipment zone.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)

    # Use a block-based edge density approach
    # Divide frame into grid cells and mark high-edge-density cells as equipment
    cell_h, cell_w = 64, 64
    h, w = edges.shape
    mask = np.zeros((h, w), dtype=np.uint8)

    for y in range(0, h - cell_h, cell_h // 2):
        for x in range(0, w - cell_w, cell_w // 2):
            cell = edges[y:y + cell_h, x:x + cell_w]
            density = cv2.countNonZero(cell) / cell.size
            if density >= config["equipment_edge_density_threshold"]:
                mask[y:y + cell_h, x:x + cell_w] = 255

    # Clean up the mask
    kernel = np.ones((15, 15), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))

    return mask


def is_person_near_equipment(person_bbox, equipment_mask):
    """Check if a person's lower body overlaps with an equipment zone."""
    if equipment_mask is None:
        return False
    x1, y1, x2, y2 = [int(v) for v in person_bbox]
    h, w = equipment_mask.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return False

    # Check the region around the person (expanded slightly)
    expand = 30
    ex1 = max(0, x1 - expand)
    ey1 = max(0, y1 - expand)
    ex2 = min(w, x2 + expand)
    ey2 = min(h, y2 + expand)

    region = equipment_mask[ey1:ey2, ex1:ex2]
    if region.size == 0:
        return False

    # If >15% of the expanded person region overlaps with equipment
    overlap = cv2.countNonZero(region) / region.size
    return overlap > 0.15


# ---------------------------------------------------------------------------
# Panel candidate detection (for Class 2: Opened Panel Cover)
# ---------------------------------------------------------------------------

def detect_panel_candidates(frame, config=CONFIG):
    """
    Detect electrical panel candidates using classical CV.
    Panels are rectangular, typically metallic/gray, wall-mounted.
    When open, they show internal wiring (high edges, color variance).
    When closed, they're uniform rectangles (low edges, low variance).

    Filters applied to reduce false positives:
      - Position: reject ground-level candidates (y_ratio > 0.65)
      - Rectangularity: contour must fill >55% of its bounding rect
      - Color: reject regions dominated by orange or green (blocks/floor)
      - Size: minimum area raised to 3000 px²

    Returns list of {"bbox": (x1,y1,x2,y2), "state": "open"|"closed"}.
    """
    frame_h, frame_w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 120)

    # Dilate to connect nearby edges
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Pre-compute HSV for color checks
    hsv = _hsv(frame)

    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < config["panel_min_area"] or area > config["panel_max_area"]:
            continue

        # Check if roughly rectangular
        rect = cv2.minAreaRect(cnt)
        rect_w, rect_h = rect[1]
        if rect_w == 0 or rect_h == 0:
            continue
        aspect = max(rect_w, rect_h) / min(rect_w, rect_h)
        if aspect < config["panel_min_aspect"] or aspect > config["panel_max_aspect"]:
            continue

        # Get bounding box
        x, y, w, h = cv2.boundingRect(cnt)
        bbox = (x, y, x + w, y + h)

        # ── Filter 1: Position ──
        # Panels are wall-mounted; reject candidates in the bottom portion
        # of the frame (ground-level objects like floor markings).
        y_ratio = y / frame_h
        if y_ratio > config["panel_max_y_ratio"]:
            continue

        # ── Filter 2: Rectangularity ──
        # Real panels are rectangular. Check that the contour fills a
        # reasonable fraction of its bounding rect (filters irregular shapes).
        rect_area = w * h
        if rect_area > 0:
            rectangularity = area / rect_area
            if rectangularity < config["panel_min_rectangularity"]:
                continue

        # ── Filter 3: Color ──
        # Reject regions dominated by orange (those are forklift blocks)
        # or bright green (floor markings / vests).
        hsv_crop = hsv[max(0, y):min(frame_h, y + h), max(0, x):min(frame_w, x + w)]
        if hsv_crop.size > 0:
            total_px = hsv_crop.shape[0] * hsv_crop.shape[1]
            # Orange check
            orange_mask = cv2.inRange(
                hsv_crop,
                np.array(config["orange_hsv_lower"]),
                np.array(config["orange_hsv_upper"]),
            )
            if cv2.countNonZero(orange_mask) / total_px > 0.15:
                continue  # too much orange — likely a block, not a panel
            # Green check
            green_mask = cv2.inRange(
                hsv_crop,
                np.array(config["green_hsv_lower"]),
                np.array(config["green_hsv_upper"]),
            )
            if cv2.countNonZero(green_mask) / total_px > 0.20:
                continue  # too much green — likely a floor marking

        # Classify state
        state = classify_panel_state(frame, bbox, config)
        candidates.append({"bbox": bbox, "state": state})

    return candidates


def classify_panel_state(frame, bbox, config=CONFIG):
    """Returns 'open' or 'closed' for an electrical panel bounding box."""
    crop = _crop(frame, bbox)
    if crop is None or crop.size == 0:
        return "closed"
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = cv2.countNonZero(edges) / edges.size
    color_std = float(np.std(gray))

    # An open panel shows internal components: high edge density AND high
    # color variance. Both conditions must be met to avoid false positives
    # from textured surfaces and instruction labels.
    if (edge_density >= config["panel_edge_density_open_threshold"] and
            color_std >= config["panel_color_std_open_threshold"]):
        return "open"
    return "closed"


# ---------------------------------------------------------------------------
# Forklift block counting (for Class 3: Carrying Overload)
# ---------------------------------------------------------------------------

def count_blocks_on_forks(frame, bbox, config=CONFIG):
    """
    Count the number of standardized orange blocks on a forklift using
    orange color segmentation.

    The standardized blocks are distinctly orange (HSV H≈5-25, high S/V).
    Instead of unreliable edge-based contour counting, we:
      1. Crop the forklift bounding box
      2. Segment orange pixels via HSV thresholding
      3. Erode to separate touching/stacked blocks
      4. Find connected components and count distinct orange blobs

    Returns the number of blocks detected (int).
    """
    crop = _crop(frame, bbox)
    if crop is None or crop.size == 0:
        return 0

    crop_h, crop_w = crop.shape[:2]
    bbox_area = crop_h * crop_w
    min_blob_area = int(bbox_area * config["block_min_orange_area_ratio"])

    # ── Step 1: Segment orange pixels ──
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    orange_mask = cv2.inRange(
        hsv,
        np.array(config["orange_hsv_lower"]),
        np.array(config["orange_hsv_upper"]),
    )

    # Quick check: if very few orange pixels, no blocks
    orange_fraction = cv2.countNonZero(orange_mask) / bbox_area
    if orange_fraction < 0.02:
        return 0

    # ── Step 2: Morphological cleanup ──
    # Close small gaps within a block
    orange_mask = cv2.morphologyEx(
        orange_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)
    )
    # Erode to separate touching/stacked blocks
    erode_size = max(3, int(crop_h * 0.02))
    orange_mask = cv2.erode(
        orange_mask, np.ones((erode_size, erode_size), np.uint8), iterations=1
    )

    # ── Step 3: Count connected components ──
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        orange_mask, connectivity=8
    )

    # Count blobs that are large enough to be a block (label 0 = background)
    block_count = 0
    for i in range(1, num_labels):
        blob_area = stats[i, cv2.CC_STAT_AREA]
        if blob_area >= min_blob_area:
            block_count += 1

    # ── Step 4: Area-based fallback ──
    # If blocks are fully merged into one large orange region (e.g. 3 blocks
    # touching tightly), estimate count from total orange area.
    # Typical single block covers ~8-15% of forklift bbox area.
    if block_count <= 1 and orange_fraction > 0.10:
        # Estimate: each block ≈ 10% of bbox, but we use a conservative
        # divisor to avoid overcounting.
        area_estimate = max(1, round(orange_fraction / 0.12))
        block_count = max(block_count, area_estimate)

    return block_count
