"""Shared helpers for locating input video clips in flat or nested dataset layouts."""
from pathlib import Path

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


def discover_video_clips(data_dir, recursive=True):
    """
    Find video clips under `data_dir`.

    Supports both layouts:
      - flat:   data/*.mp4
      - nested: data/test/0_safe_walkway_violation/*.mp4  (Kaggle-style)
    """
    root = Path(data_dir)
    if not root.exists():
        return []

    if recursive:
        pattern = root.rglob("*")
    else:
        pattern = root.glob("*")

    return sorted(
        p for p in pattern
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )


def clip_id_for_path(clip_path, data_dir, clip_id_root=None):
    """Stable clip identifier used in reports and the dashboard."""
    clip_path = Path(clip_path).resolve()
    anchor = (Path(clip_id_root) if clip_id_root else Path(data_dir)).resolve()
    try:
        return str(clip_path.relative_to(anchor)).replace("\\", "/")
    except ValueError:
        return clip_path.name


def resolve_clip_id_root(data_dir, project_data_dir):
    """
    Keep clip IDs stable for the dashboard when processing a dataset split
    (e.g. data/test/) instead of the whole data/ tree.
    """
    data_dir = Path(data_dir).resolve()
    project_data_dir = Path(project_data_dir).resolve()
    try:
        data_dir.relative_to(project_data_dir)
        return project_data_dir
    except ValueError:
        return data_dir
