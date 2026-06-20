"""
End-to-end orchestrator: data/**/*.mp4 -> Detection -> Severity -> Escalation
-> Reports (DB + CSV). Run this before opening the dashboard so it has data
to show.

Usage:
    python src/pipeline.py --backend yolo_world --data-dir data --reset
    python src/pipeline.py --backend mock --mock-config tests/synthetic_boxes.json
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # allow `import detection`, `import severity`, etc.

from detection.detector import analyze_clip, load_policy_rules, MockBackend
from escalation.escalation_pipeline import process_detection
from db import database
from video_discovery import discover_video_clips, clip_id_for_path, resolve_clip_id_root

ROOT = Path(__file__).resolve().parents[1]
POLICY_RULES_PATH = ROOT / "src" / "policy" / "policy_rules.json"
CSV_LOG_PATH = ROOT / "outputs" / "compliance_log.csv"
PROJECT_DATA_DIR = ROOT / "data"


def build_backend(name, mock_config_path=None):
    if name == "yolo_world":
        from detection.detector import YoloWorldBackend
        return YoloWorldBackend()
    if name == "mock":
        import json
        if mock_config_path:
            with open(mock_config_path) as f:
                fixed_boxes = json.load(f)
            # JSON keys are strings; convert back to int frame indices.
            fixed_boxes = {int(k): v for k, v in fixed_boxes.items()}
        else:
            fixed_boxes = {}
        return MockBackend(fixed_boxes)
    raise ValueError(f"Unknown backend: {name}")


def reset_outputs():
    """Clear prior pipeline results so the dashboard reflects only the latest run."""
    db_path = database.DEFAULT_DB_PATH
    if db_path.exists():
        db_path.unlink()
    if CSV_LOG_PATH.exists():
        CSV_LOG_PATH.unlink()


def run(data_dir, backend_name, mock_config_path=None, frame_stride=10, reset=False):
    if not POLICY_RULES_PATH.exists():
        print("policy_rules.json not found -- run src/policy/parse_policy.py first.", file=sys.stderr)
        sys.exit(1)

    policy_rules_by_class = load_policy_rules(POLICY_RULES_PATH)
    if reset:
        reset_outputs()
    database.init_db()

    data_root = Path(data_dir)
    clip_paths = discover_video_clips(data_root)
    clip_id_root = resolve_clip_id_root(data_root, PROJECT_DATA_DIR)

    if not clip_paths:
        print(f"No video clips found under {data_dir}. Add clips and re-run.")
        return

    print(f"Found {len(clip_paths)} clip(s) under {data_dir}")

    total_events = 0
    backend = build_backend(backend_name, mock_config_path)  # load model ONCE, reused across clips
    for clip_path in clip_paths:
        clip_id = clip_id_for_path(clip_path, data_root, clip_id_root=clip_id_root)
        print(f"Processing {clip_id} ...")
        if hasattr(backend, "reset"):
            backend.reset()
        for detection in analyze_clip(
            clip_path,
            policy_rules_by_class,
            backend,
            frame_stride=frame_stride,
            clip_id=clip_id,
        ):
            report = process_detection(detection, policy_rules_by_class)
            total_events += 1
            print(f"  -> {report['severity']:8s} {report['behavior_class']:35s} {report['escalation_action']}")

    print(f"\nDone. {total_events} compliance events written to outputs/compliance.db and outputs/compliance_log.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument("--backend", choices=["yolo_world", "mock"], default="yolo_world")
    parser.add_argument("--mock-config", default=None, help="Path to JSON fixed-boxes config (mock backend only)")
    parser.add_argument("--frame-stride", type=int, default=10)
    parser.add_argument("--reset", action="store_true", help="Clear existing DB/CSV before processing")
    args = parser.parse_args()
    run(args.data_dir, args.backend, args.mock_config, args.frame_stride, args.reset)
