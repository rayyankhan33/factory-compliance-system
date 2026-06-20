# Factory Compliance & Alert Escalation System

An end-to-end pipeline that watches factory-floor video, detects four
policy-defined unsafe behaviors, classifies their severity, escalates
high-risk events in real time, and surfaces everything in a dashboard.

Built against `KMP-OHS-POL-001` (the included `compliance_policy.pdf`) and
the Kaggle "Video Dataset for Safe and Unsafe Behaviours" dataset.

## Quick start

```bash
pip install -r requirements.txt

# 1. Parse the policy PDF into structured rules (only needs to be re-run if
#    the policy document changes)
python src/policy/parse_policy.py compliance_policy.pdf src/policy/policy_rules.json
```

### If you have the labeled Kaggle dataset (train/ and test/ folders of class subfolders)

Don't copy or flatten the clips into `data/` — point the evaluator directly
at the dataset's own folder structure instead. The Kaggle dataset's
subfolder names (e.g. `0_safe_walkway_violation`, `4_safe_walkway`) are
ground-truth labels, and `src/evaluate.py` uses them to measure real
detection accuracy instead of just running blind:

```bash
python src/evaluate.py --dataset-dir /path/to/kaggle/test --backend yolo_world
```

This walks every class subfolder, matches its name against
`policy_rules.json`'s safe/unsafe behavior names (so it works regardless of
the exact naming convention), runs the real detector on every clip, and
reports per-class **recall on unsafe clips** and **false-alarm rate on safe
clips**. That's the calibration feedback loop: low recall or a high
false-alarm rate for a class points you at that class's thresholds in
`src/detection/color_utils.py`'s `CONFIG` dict (see "Known limitations &
calibration checklist" below). `tools/calibrate_colors.py` helps you sample
real HSV values from a frame to retune those ranges.

Run it against `train/` too if you want more clips in the accuracy
estimate — no training actually happens (this is a zero-shot detector), so
`train/` is just more labeled data to evaluate against, not used to fit
anything.

```bash
# quick smoke test on a handful of clips per folder before a full run
python src/evaluate.py --dataset-dir /path/to/kaggle/test --backend yolo_world --max-clips-per-folder 5
```

### If you just have raw, unlabeled clips (e.g. new footage, no class folders)

Drop them in `data/` (flat `data/*.mp4`) **or** keep the Kaggle folder layout under
`data/test/` and `data/train/` — the pipeline discovers clips recursively in
either layout. Then run the production pipeline to populate the dashboard:

```bash
python src/pipeline.py --backend yolo_world --data-dir data --reset
streamlit run src/dashboard/app.py
```

Use `--data-dir data/test` to process only the test split. Clip IDs in reports
always use paths relative to `data/` (e.g. `test/0_safe_walkway_violation/0_te1.mp4`)
so the dashboard can match videos to logged events.

Pass `--reset` to clear old synthetic or stale results before a fresh run.

The first `yolo_world` run downloads pretrained YOLO-World weights
(`yolov8s-world.pt`), so it needs internet access once. After that it runs
fully offline.

If you'd rather sanity-check the wiring before pointing it at real footage:

```bash
python tests/build_synthetic_clip.py
python src/pipeline.py --backend mock --mock-config tests/synthetic_boxes.json \
    --data-dir tests/synthetic_data --frame-stride 5
```

This regenerates a tiny synthetic clip with known ground truth and replays
it through the real severity/escalation/report code (see "Testing
strategy" below).

## Architecture

```
compliance_policy.pdf
        │  (Module 1a: deterministic regex/structure parser)
        ▼
policy_rules.json  ──────────────┐  single source of truth for severity,
        │                        │  detection vocabulary, and report refs
        ▼                        │
data/*.mp4                       │
   │                             │
   ▼                             │
Detection Engine  ◄──────────────┘   (YOLO-World zero-shot localization
   │  raw detection                  + OpenCV color/shape heuristics)
   ▼
Severity Matrix  (LOW / MEDIUM / HIGH / CRITICAL)
   │
   ▼
Escalation Pipeline ──► SQLite `compliance_events` (always)
   │                └─► SQLite `alerts` queue (HIGH / CRITICAL only)
   ▼
Report Generator ──► outputs/compliance_log.csv (append-only audit log)
   │
   ▼
Streamlit Dashboard (Live Feed / Alert Timeline / Historical Log+Export)
```

## Module-by-module notes

### Module 1 — Policy parsing (`src/policy/parse_policy.py`)

Deterministic, NOT LLM-based. The policy document has a very regular
structure (numbered sections, a fixed "Required Behavior" / "Non-Compliant
Behavior" pattern per domain, uppercase `WARNING` / `CRITICAL SAFETY NOTICE`
callout tokens), so a regex/structure-based parser can extract every field
as a verbatim substring of the source PDF — there's no hallucination risk to
guard against in the first place.

**Faithfulness check**: every extracted field is round-tripped against the
cleaned source text in `verify_against_source()`. The parser refuses to
write `policy_rules.json` (exits non-zero) if anything it extracted isn't
verbatim-present in the PDF. This is the automated verification step the
assignment brief asks about. If the policy document were free-form prose
instead of this regular structure, the right tradeoff would flip: an
LLM-based extraction pass with the same round-trip verification, plus a
human spot-check, would be more practical than hand-written regex.

### Module 1 — Detection Engine (`src/detection/`)

- **Localization**: `detector.py` wraps a pluggable backend
  (`DetectionBackend`). The real backend, `YoloWorldBackend`, is
  `ultralytics` YOLO-World — an open-vocabulary, zero-shot detector. No
  training or labeling was needed, which is why this was chosen given the
  ~3-day timeline. The text-prompt vocabulary it queries against is derived
  from `policy_rules.json` (`derive_detection_vocabulary`), not hardcoded
  disconnected from the policy.
- **Classification**: once an object is localized, `color_utils.py` applies
  the *literal observable indicators the policy itself names* — vest color
  (green vs red-black), walkway floor-marking color, panel open/closed
  state (edge density / color variance), and block count on the forks
  (contour counting). None of this needs a trained model.
- **`MockBackend`** exists only for testing (see below) — it returns
  precomputed boxes instead of running real CV, so the rest of the system
  can be validated without GPU/internet/real footage.

### Module 2 — Severity Matrix (`src/severity/severity_matrix.py`)

Each class's **default** tier is derived from its policy callout type
(`WARNING` vs `CRITICAL_SAFETY_NOTICE`) and the specific hazard language in
its section text; the tier is then adjusted by detection-time context:

| Class | Default | Escalates to | Why |
|---|---|---|---|
| 0 Safe Walkway Violation | MEDIUM | HIGH if person is near a forklift/machine | WARNING-level, behavioral (not state-based); HIGH only once the named hazard ("proximity to forklift and machinery") is actually confirmed present |
| 1 Unauthorized Intervention | HIGH | CRITICAL if ≥2 occurrences in the same clip/zone | CRITICAL SAFETY NOTICE; personnel exposure is concurrent by definition; recurrence matches the brief's "high-frequency recurrence" CRIT criterion |
| 2 Opened Panel Cover | LOW | MEDIUM if a person is nearby | WARNING-level; policy explicitly says this is unsafe "regardless of... whether personnel are in the immediate vicinity" — the textbook LOW example (state-based, no personnel exposure) |
| 3 Carrying Overload with Forklift | CRITICAL always | — | The *only* rule in the document with explicit "will trigger an immediate alert" language tied to an unambiguous, purely quantifiable threshold |

### Module 3 — Escalation Pipeline (`src/escalation/escalation_pipeline.py`)

LOW/MEDIUM → DB log only. HIGH/CRITICAL → DB log **and** a row in the
`alerts` SQLite table, which acts as the real-time notification queue.

This is a deliberate scope decision: Streamlit reruns its whole script on
every interaction, so there's no long-lived in-process pub/sub channel to
push into across the dashboard's process boundary. A persistent queue table
that the dashboard polls is the simplest mechanism that's still genuinely
real-time at this event volume, and it's a one-line swap for a real
WebSocket/SSE channel if this were rebuilt on a non-Streamlit frontend.

Multiple simultaneous violations in one clip are handled independently —
each detection gets its own event/report/(if applicable) alert; severities
are never collapsed or averaged across a clip.

### Module 4 — Report Generation (`src/reports/report_generator.py`)

Every event is written to both SQLite (queryable, used by the dashboard)
and an append-only CSV (`outputs/compliance_log.csv`), with all required
fields: `event_id, timestamp, clip_id, zone, behavior_class,
policy_rule_ref, event_description, severity, escalation_action`.

### Module 5 — Dashboard (`src/dashboard/app.py`, Streamlit)

Three tabs:
- **Live Feed Monitor** — pick a clip, watch it, see its latest status and
  any active (unacknowledged) alert.
- **Alert Timeline Stream** — chronological, severity-color-coded feed of
  every event, with an optional 5s auto-refresh.
- **Historical Log & Export** — filter by date range / severity / behavior
  class, then export the filtered set as CSV or JSON.

## Testing strategy (`tests/`)

This development environment could not download the Kaggle dataset or
YOLO-World's pretrained weights (sandboxed, package-registry-only network
access), so real detection accuracy could not be verified here.

To still validate everything *downstream* of localization,
`tests/build_synthetic_clip.py` generates a short synthetic clip with known
ground truth (a painted walkway, a red-vs-green "vest" rectangle, a
noise-textured "open panel", bordered squares as "blocks") plus a matching
`MockBackend` config that supplies the bounding boxes directly. Running the
real pipeline against this produces exactly the expected severities for all
8 scripted scenarios, including a true negative (2 blocks = safe, correctly
produces no event). This proves the severity/escalation/report/dashboard
wiring is correct; it does **not** prove YOLO-World's real-world detection
accuracy, which can only be evaluated once you have real footage.

## Known limitations & calibration checklist

Run through this once you have real clips loaded in `data/`:

1. **Color thresholds** (`src/detection/color_utils.py`, `CONFIG` dict) —
   `green_hsv_lower/upper`, `red_hsv_lower/upper`: sample a few real frames
   and check the actual vest/walkway paint colors fall inside these ranges.
2. **Panel open/closed thresholds** — `panel_edge_density_open_threshold`,
   `panel_color_std_open_threshold`: these are starting guesses; verify
   against real open vs. closed panel frames.
3. **Block counting** — `block_min/max_contour_area`,
   `single_block_area_estimate`: must match the real blocks' pixel size at
   your camera's resolution/distance. (The included synthetic test had to
   have its own block size tuned to match this constant — same exercise
   you'll need to do with real frames.)
4. **Zone assignment** (`detector.py`, `assign_zone`) — currently a crude
   horizontal-thirds split. Replace with a real zone polygon lookup once the
   facility's two camera layouts (per Section 7 of the policy) are known.
5. **HAZARD_PROXIMITY_PX** (`detector.py`) — pixel-distance threshold for
   "near a forklift" / "near a panel"; tune to your frame resolution.
6. **Equipment-interaction detection** (class 1) currently approximates
   "interacting with equipment" as "person is near a localized electrical
   panel" — the only fixed-equipment class YOLO-World is prompted for. If
   the dataset's unsafe-intervention clips involve other machinery, add
   those as additional prompted labels.

## Production upgrade path

- Swap the SQLite `alerts`-table polling for a real WebSocket/SSE push if
  moving off Streamlit.
- Swap zero-shot YOLO-World for a fine-tuned detector once enough labeled
  frames exist, for better accuracy/speed.
- Replace the horizontal-thirds zoning with real per-camera zone polygons.

## Repository layout

```
factory-compliance-system/
├── README.md
├── requirements.txt
├── compliance_policy.pdf
├── data/                       # raw unlabeled clips for production/live-monitoring runs
├── outputs/
│   ├── compliance.db           # generated
│   └── compliance_log.csv      # generated
├── src/
│   ├── policy/parse_policy.py, policy_rules.json
│   ├── detection/detector.py, color_utils.py
│   ├── severity/severity_matrix.py
│   ├── escalation/escalation_pipeline.py
│   ├── reports/report_generator.py
│   ├── db/database.py
│   ├── dashboard/app.py
│   ├── pipeline.py             # orchestrator entry point (unlabeled clips -> dashboard)
│   └── evaluate.py             # accuracy check against a labeled dataset (e.g. Kaggle test/)
├── tools/
│   └── calibrate_colors.py     # sample real HSV values from a frame to tune CONFIG thresholds
└── tests/
    ├── build_synthetic_clip.py
    ├── synthetic_boxes.json
    └── synthetic_data/synthetic_test_clip.mp4
```
