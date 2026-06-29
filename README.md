# 🏭 Factory Compliance & Alert Escalation System

An end-to-end automated compliance monitoring system that ingests factory surveillance video, parses a regulatory policy document, detects unsafe behaviors in real time, classifies violations by severity, and routes alerts through a live operations dashboard.

Built as part of an AI engineering internship screening challenge.

---

## 📸 Screenshots

> **Dashboard — Live Feed Monitor**
> 
<img width="1366" height="768" alt="image" src="https://github.com/user-attachments/assets/3c47224f-5e6b-4340-92a3-cbd772dce58c" />

> **Dashboard — Alert Timeline Stream**
> <img width="1366" height="768" alt="image" src="https://github.com/user-attachments/assets/e2ff9b96-deb0-4911-8561-4f6dd3867afc" />


> **Dashboard — Historical Log & Export**
> <img width="1366" height="768" alt="image" src="https://github.com/user-attachments/assets/023d6d04-ef4d-40cf-94f6-3e662b0de2b5" />


---

## 🧠 How It Works

The system is a 5-module pipeline. Each module feeds into the next.

```
Policy PDF ──► parse_policy.py ──► policy_rules.json
                                         │
Factory Video ──► StreamProcessor ──► YOLOv8n (localize persons & forklifts)
                                         │
                                   color_utils.py (classical CV compliance checks)
                                         │
                                   severity_matrix.py (LOW / MEDIUM / HIGH / CRITICAL)
                                         │
                                   escalation_pipeline.py (route to alert or DB)
                                         │
                                   report_generator.py (UUID, timestamp, policy ref)
                                         │
                              SQLite DB + CSV audit log + Dashboard
```

### Detection Approach

**The policy parser runs once at startup**, extracting 4 compliance rules from the PDF into `policy_rules.json`. It never runs again during video processing.

**At runtime**, every 5th video frame is analyzed:

1. **YOLOv8n** detects where persons and forklifts are (pixel bounding boxes). It knows nothing about compliance rules — it only answers *where are the objects?*
2. **Classical CV** (`color_utils.py`) then examines those bounding box regions and answers the policy questions:

| Class | Unsafe Behavior | How Detected |
|---|---|---|
| 0 | Safe Walkway Violation | Person's foot point outside green floor mask (HSV) |
| 1 | Unauthorized Intervention | Person near equipment without green vest (HSV torso crop) |
| 2 | Opened Panel Cover | Rectangular wall region with high edge density (Canny) |
| 3 | Carrying Overload with Forklift | 3+ orange blobs on forklift forks (connected components) |

Non-analyzed frames carry forward the previous bounding box annotations so the displayed video remains smooth.

---

## 🗂️ Project Structure

```
factory-compliance-system/
├── README.md
├── compliance_policy.pdf
├── requirements.txt
├── data/                          # Place input video clips here
├── outputs/                       # Generated reports and logs
│   └── compliance_log.csv
├── tools/
│   └── calibrate_colors.py        # HSV threshold calibration helper
└── src/
    ├── pipeline.py                # Offline batch runner
    ├── evaluate.py                # Detection accuracy evaluation
    ├── video_discovery.py         # Clip discovery utility
    ├── policy/
    │   ├── parse_policy.py        # Module 1 — Policy PDF parser
    │   └── policy_rules.json      # Pre-extracted compliance rules
    ├── detection/
    │   ├── detector.py            # Module 1 — Detection engine (YOLOv8n)
    │   └── color_utils.py         # Classical CV heuristics
    ├── severity/
    │   └── severity_matrix.py     # Module 2 — Severity classification
    ├── escalation/
    │   └── escalation_pipeline.py # Module 3 — Alert routing
    ├── reports/
    │   └── report_generator.py    # Module 4 — Compliance report builder
    ├── db/
    │   └── database.py            # SQLite persistence layer
    └── dashboard/
        ├── server.py              # Module 5 — FastAPI server (recommended)
        ├── app.py                 # Module 5 — Streamlit alternative
        ├── stream_processor.py    # Live video + detection loop
        └── static/                # HTML / CSS / JS frontend
```

---

## ⚙️ Setup

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/factory-compliance-system.git
cd factory-compliance-system
pip install -r requirements.txt
```

### Add Video Clips

Download the dataset from Kaggle and place clips inside the `data/` directory:

**Dataset:** [Video Dataset for Safe and Unsafe Behaviours](https://www.kaggle.com/datasets/trnhhnggiang/video-dataset-for-safe-and-unsafe-behaviours)

```
data/
├── clip_001.mp4
├── clip_002.mp4
└── ...
```

---

## 🚀 Running the System

### Option A — Live Dashboard (recommended)

```bash
cd src
uvicorn dashboard.server:app --host 0.0.0.0 --port 8000 --reload
```

Then open **http://localhost:8000** in your browser.

- Select a video clip from the dropdown
- Click **Start Stream**
- The dashboard runs detection live and flashes alerts for HIGH / CRITICAL violations

### Option B — Streamlit Dashboard (simpler)

```bash
cd src
streamlit run dashboard/app.py
```

Good for reviewing batch-processed results. Does not support true real-time streaming.

### Option C — Offline Batch Pipeline

Process all clips in `data/` and populate the database without opening the dashboard:

```bash
cd src
python pipeline.py

# Reset database and reprocess from scratch
python pipeline.py --reset
```

---

## 📊 Severity Classification

Severity tiers are derived directly from the policy document's callout language, not set arbitrarily.

| Tier | Behavior Classes | Policy Signal | Escalation |
|---|---|---|---|
| LOW | Opened Panel Cover (no person nearby) | WARNING callout | DB log only |
| MEDIUM | Safe Walkway Violation | WARNING callout + high frequency flag | DB log only |
| HIGH | Safe Walkway Violation (near forklift) · Unauthorized Intervention | CRITICAL SAFETY NOTICE | Real-time alert + DB log |
| CRITICAL | Unauthorized Intervention (recurring) · Forklift Overload | CRITICAL SAFETY NOTICE | Real-time alert + DB log |

---

## 📋 Compliance Report Fields

Every detected violation automatically generates a structured record:

| Field | Description |
|---|---|
| `event_id` | UUID4 unique identifier |
| `timestamp` | ISO 8601 UTC |
| `clip_id` | Source video filename |
| `zone` | Zone-1 / Zone-2 / Zone-3 (horizontal thirds) |
| `behavior_class` | Exact name from policy document |
| `policy_rule_ref` | e.g. `Section 3.3.2` |
| `event_description` | Human-readable description of observed behavior |
| `severity` | LOW / MEDIUM / HIGH / CRITICAL |
| `escalation_action` | Action taken by the escalation pipeline |

Reports are persisted to SQLite and appended to `outputs/compliance_log.csv`.

---

## 🎛️ Calibrating Color Thresholds

The classical CV detection relies on HSV color thresholds that may need tuning for different cameras or lighting conditions. Use the calibration tool:

```bash
# Step 1 — save a frame so you can identify pixel coordinates visually
python tools/calibrate_colors.py data/your_clip.mp4 --frame 30 --save-frame /tmp/frame.png

# Step 2 — sample the HSV values of a region (e.g. a green vest)
python tools/calibrate_colors.py data/your_clip.mp4 --frame 30 --bbox 120,80,260,400
```

The tool prints the H/S/V min, mean, and max for that region plus a suggested CONFIG range. Update the relevant thresholds in `src/detection/color_utils.py`'s `CONFIG` dictionary accordingly.

---

## 📈 Evaluating Detection Accuracy

```bash
cd src
python evaluate.py
```

Runs the detector against the labelled Kaggle dataset folder structure and prints per-class recall and false alarm rate. Results indicate which behavior classes and which CONFIG thresholds need tuning.

---

## 🔧 Architecture Decisions & Limitations

### Why YOLOv8n and not a custom-trained model?
YOLOv8n is the fastest variant in the YOLOv8 family and reliably detects persons and forklifts (the two objects that need localization) from COCO pretraining. A custom model would require annotated factory footage which was not available. The compliance decisions — vest color, block count, panel state, walkway position — are made by classical CV operating on the bounding boxes YOLO produces, not by the model itself.

### Why classical CV instead of a vision-language model for compliance checks?
Each compliance check maps directly to a visual observable defined in the policy: green vest color, orange block count, edge density of a panel surface, green floor paint position. These are precise, low-level visual properties that HSV thresholding and morphological operations handle reliably and transparently. A vision-language model would be slower, harder to audit, and would introduce hallucination risk for a safety-critical application.

### Why a deterministic policy parser instead of an LLM?
The policy document has a regular, numbered section structure. A deterministic regex parser extracts rules faithfully and verifiably — it includes a verbatim substring check that confirms every extracted string physically exists in the source document. An LLM parser could hallucinate a rule that sounds plausible but isn't in the document, which is unacceptable for a compliance system.

### Known limitations
- **HSV thresholds** are sensitive to lighting changes. Recalibrate using `tools/calibrate_colors.py` if deploying in a different lighting environment.
- **Zone assignment** is a horizontal-thirds approximation. A production deployment would use a camera-specific polygon-to-zone mapping.
- **Panel detection** uses classical CV (not YOLO) since COCO has no electrical panel class. It may false-positive on any large rectangular metallic surface.
- **Frame stride of 5** means detection runs on approximately 3–5 frames per second from a 25 FPS source. Brief violations under 200ms may be missed.
- **Forklift mapping** assumes COCO class 7 ("truck") corresponds to a forklift in an indoor factory context. This breaks in loading bays with real trucks.

---

## 📦 Dependencies

Key libraries used:

| Library | Purpose |
|---|---|
| `ultralytics` | YOLOv8n inference |
| `opencv-python` | Frame reading, HSV ops, Canny, morphology |
| `fastapi` + `uvicorn` | Dashboard web server + SSE streaming |
| `streamlit` | Alternative dashboard |
| `pdfplumber` | Policy PDF text extraction |
| `numpy` | Array operations |
| `sqlite3` | Built-in — compliance event storage |

Full list in `requirements.txt`.

---

## 📄 License

For assessment and portfolio use only.
