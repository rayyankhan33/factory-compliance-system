"""
Evaluate detection accuracy against a LABELED dataset (e.g. the Kaggle
"videodataset-for-safe-and-unsafe-behaviours" train/ or test/ split), using
its own folder names as ground truth -- no model training involved, this is
a pure inference-time accuracy check for the zero-shot detector + heuristics.

Point this directly at the dataset's class folders (e.g.
`.../test/0_safe_walkway_violation/`, `.../test/4_safe_walkway/`, etc.) --
no need to flatten or copy clips anywhere first. Folder names are matched
to the parsed policy's safe_behavior / unsafe_behavior names by word-set
similarity, so this works regardless of the exact folder-naming convention,
as long as the words substantially match what's in policy_rules.json.

For each clip:
    - if its folder represents an UNSAFE behavior for class X, a correct run
      detects at least one class-X event somewhere in the clip (recall).
    - if its folder represents the SAFE counterpart for class X, a correct
      run detects NO class-X event anywhere in the clip (false alarm check).

This directly produces the numbers needed for the calibration checklist in
the README (which CONFIG thresholds in color_utils.py to retune, and how
far off they currently are).

Usage:
    python src/evaluate.py --dataset-dir /path/to/kaggle/test --backend yolo_world
    python src/evaluate.py --dataset-dir /path/to/kaggle/test --backend yolo_world --max-clips-per-folder 20
"""
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from detection.detector import analyze_clip, load_policy_rules
from pipeline import build_backend, POLICY_RULES_PATH

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


def _slug_words(text):
    return set(re.sub(r"[^a-z0-9]+", " ", text.lower()).strip().split())


def build_candidates(policy_rules_by_class):
    """[(class_id, 'safe'|'unsafe', word_set), ...] derived from policy_rules.json."""
    candidates = []
    for class_id, rule in policy_rules_by_class.items():
        candidates.append((class_id, "unsafe", _slug_words(rule["unsafe_behavior"])))
        candidates.append((class_id, "safe", _slug_words(rule["safe_behavior"])))
    return candidates


def match_folder_to_label(folder_name, candidates, min_score=0.4):
    """Strips a leading numeric index prefix (Kaggle convention: '0_some_behavior')
    and matches the remaining words against policy_rules.json via Jaccard
    similarity over word sets. Returns (class_id, 'safe'|'unsafe') or None."""
    name = re.sub(r"^\d+[_\-]?", "", folder_name)
    words = _slug_words(name.replace("_", " ").replace("-", " "))
    if not words:
        return None

    best, best_score = None, 0.0
    for class_id, kind, cand_words in candidates:
        if not cand_words:
            continue
        union = words | cand_words
        score = len(words & cand_words) / len(union) if union else 0
        if score > best_score:
            best_score, best = score, (class_id, kind)

    return best if best_score >= min_score else None


def evaluate(dataset_dir, backend_name="yolo_world", frame_stride=10, max_clips_per_folder=None):
    policy_rules_by_class = load_policy_rules(POLICY_RULES_PATH)
    candidates = build_candidates(policy_rules_by_class)

    folders = sorted(p for p in Path(dataset_dir).iterdir() if p.is_dir())
    if not folders:
        print(f"No subfolders found in {dataset_dir}")
        return

    backend = build_backend(backend_name)  # load the model ONCE, reuse across every clip

    results = {}  # class_id -> {"tp":0, "fn":0, "fp":0, "tn":0}
    skipped = []

    for folder in folders:
        match = match_folder_to_label(folder.name, candidates)
        if match is None:
            skipped.append(folder.name)
            continue
        class_id, kind = match

        clips = sorted(p for p in folder.glob("*") if p.suffix.lower() in VIDEO_EXTENSIONS)
        if max_clips_per_folder:
            clips = clips[:max_clips_per_folder]
        if not clips:
            continue

        print(f"Folder '{folder.name}' -> class {class_id} ({kind}), evaluating {len(clips)} clip(s)...")
        stats = results.setdefault(class_id, {"tp": 0, "fn": 0, "fp": 0, "tn": 0})

        for clip_path in clips:
            if hasattr(backend, "reset"):
                backend.reset()
            detected_classes = {d["class_id"] for d in
                                 analyze_clip(clip_path, policy_rules_by_class, backend, frame_stride=frame_stride)}
            fired = class_id in detected_classes

            if kind == "unsafe":
                stats["tp" if fired else "fn"] += 1
            else:
                stats["fp" if fired else "tn"] += 1

    if skipped:
        print(f"\n[note] could not confidently match {len(skipped)} folder(s) to a policy behavior, skipped: {skipped}")

    print("\n=== Per-class detection accuracy (against labeled dataset) ===")
    if not results:
        print("No labeled folders were evaluated.")
        return

    for class_id in sorted(results):
        rule = policy_rules_by_class[class_id]
        s = results[class_id]
        total_unsafe = s["tp"] + s["fn"]
        total_safe = s["fp"] + s["tn"]
        print(f"\nClass {class_id} -- {rule['unsafe_behavior']} ({rule['policy_rule_ref']}):")
        if total_unsafe:
            recall = s["tp"] / total_unsafe
            print(f"  Recall on unsafe clips:     {s['tp']}/{total_unsafe} = {recall:.1%}")
        if total_safe:
            false_alarm_rate = s["fp"] / total_safe
            print(f"  False alarm rate on safe clips: {s['fp']}/{total_safe} = {false_alarm_rate:.1%}")
        if not total_unsafe and not total_safe:
            print("  (no clips evaluated)")

    print(
        "\nLow recall or a high false-alarm rate for a class points at that class's CONFIG "
        "constants in src/detection/color_utils.py -- see README 'Known limitations & "
        "calibration checklist'."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True, help="Path to a folder of class subfolders (e.g. the Kaggle test/ split)")
    parser.add_argument("--backend", choices=["yolo_world", "mock"], default="yolo_world")
    parser.add_argument("--frame-stride", type=int, default=10)
    parser.add_argument("--max-clips-per-folder", type=int, default=None, help="Cap clips per folder for a quick smoke test")
    args = parser.parse_args()
    evaluate(args.dataset_dir, args.backend, args.frame_stride, args.max_clips_per_folder)
