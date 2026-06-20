"""
FastAPI-based Operations Dashboard server — replaces the Streamlit app
for real-time video compliance monitoring.

Run with:
    python src/dashboard/server.py

Then open http://localhost:8000 in your browser.
"""
import asyncio
import json
import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import (
    StreamingResponse,
    JSONResponse,
    FileResponse,
    HTMLResponse,
)
from fastapi.staticfiles import StaticFiles
import uvicorn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from dashboard.stream_processor import get_processor
from db import database
from video_discovery import discover_video_clips, clip_id_for_path, resolve_clip_id_root

app = FastAPI(title="Factory Compliance Dashboard")

# Serve static files (HTML, CSS, JS)
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

DATA_DIR = ROOT / "data"


# ---------------------------------------------------------------------------
# Root — serve the dashboard HTML
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def index():
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        return HTMLResponse("<h1>Dashboard not found. Ensure static/index.html exists.</h1>")
    return FileResponse(str(index_file), media_type="text/html")


# ---------------------------------------------------------------------------
# Video listing
# ---------------------------------------------------------------------------
@app.get("/api/videos")
async def list_videos():
    """List all available video clips from the data/ directory."""
    clip_paths = discover_video_clips(DATA_DIR) if DATA_DIR.exists() else []
    clip_id_root = resolve_clip_id_root(DATA_DIR, DATA_DIR)

    videos = []
    for p in clip_paths:
        cid = clip_id_for_path(p, DATA_DIR, clip_id_root=clip_id_root)
        # Infer category from directory name
        category = p.parent.name if p.parent != DATA_DIR else "uncategorized"
        videos.append({
            "clip_id": cid,
            "path": str(p),
            "filename": p.name,
            "category": category,
            "size_mb": round(p.stat().st_size / (1024 * 1024), 1),
        })
    return {"videos": videos, "total": len(videos)}


# ---------------------------------------------------------------------------
# MJPEG Video Stream
# ---------------------------------------------------------------------------
@app.get("/api/stream/start")
async def start_stream(source: str = Query(..., description="Video file path, RTSP URL, or webcam index")):
    """Start processing a video source and return status."""
    processor = get_processor()
    processor.start(source)
    return {"status": "started", "source": source}


@app.get("/api/stream/stop")
async def stop_stream():
    """Stop the current video processing."""
    processor = get_processor()
    processor.stop()
    return {"status": "stopped"}


@app.get("/api/stream/status")
async def stream_status():
    """Get current stream processing stats."""
    processor = get_processor()
    return processor.stats


@app.get("/api/stream/feed")
async def video_feed():
    """MJPEG streaming endpoint. Connect via <img src='/api/stream/feed'>."""
    processor = get_processor()

    async def generate():
        while True:
            frame_bytes = processor.get_latest_frame_jpeg()
            if frame_bytes:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + frame_bytes
                    + b"\r\n"
                )
            await asyncio.sleep(0.033)  # ~30fps max output rate

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ---------------------------------------------------------------------------
# SSE — Live Violations Stream
# ---------------------------------------------------------------------------
@app.get("/api/violations/live")
async def violations_live():
    """Server-Sent Events endpoint for live violation notifications."""
    processor = get_processor()
    queue = processor.register_sse_queue()

    async def event_generator():
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=30)
                    # Serialize the event, handling non-serializable fields
                    serializable = {
                        k: v for k, v in event.items()
                        if k != "context" and not isinstance(v, bytes)
                    }
                    serializable["context"] = str(event.get("context", {}))
                    data = json.dumps(serializable, default=str)
                    yield f"data: {data}\n\n"
                except asyncio.TimeoutError:
                    # Send keepalive comment
                    yield ": keepalive\n\n"
        finally:
            processor.unregister_sse_queue(queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# Historical violations
# ---------------------------------------------------------------------------
@app.get("/api/violations/history")
async def violations_history(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    severity: Optional[str] = None,
    class_id: Optional[str] = None,
    limit: int = 500,
):
    """Query historical violation events from the database."""
    database.init_db()

    severities = severity.split(",") if severity else None
    class_ids = [int(c) for c in class_id.split(",")] if class_id else None

    events = database.query_events(
        start_date=start_date,
        end_date=end_date,
        severities=severities,
        class_ids=class_ids,
        limit=limit,
    )
    return {"events": events, "total": len(events)}


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
@app.get("/api/export/{fmt}")
async def export_data(
    fmt: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    severity: Optional[str] = None,
    class_id: Optional[str] = None,
):
    """Export violations as CSV or JSON."""
    database.init_db()

    severities = severity.split(",") if severity else None
    class_ids = [int(c) for c in class_id.split(",")] if class_id else None

    events = database.query_events(
        start_date=start_date,
        end_date=end_date,
        severities=severities,
        class_ids=class_ids,
        limit=50000,
    )

    if fmt == "json":
        return JSONResponse(
            content=events,
            headers={"Content-Disposition": "attachment; filename=compliance_export.json"},
        )
    elif fmt == "csv":
        import csv
        import io

        if not events:
            return StreamingResponse(
                iter(["No data"]),
                media_type="text/csv",
                headers={"Content-Disposition": "attachment; filename=compliance_export.csv"},
            )

        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=events[0].keys())
        writer.writeheader()
        writer.writerows(events)

        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=compliance_export.csv"},
        )
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {fmt}. Use 'json' or 'csv'.")


# ---------------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------------
@app.get("/api/alerts")
async def get_alerts():
    """Get all active (unacknowledged) alerts."""
    database.init_db()
    alerts = database.get_active_alerts()
    return {"alerts": alerts, "total": len(alerts)}


@app.post("/api/alerts/{alert_id}/ack")
async def acknowledge_alert(alert_id: int):
    """Acknowledge an alert."""
    database.init_db()
    database.acknowledge_alert(alert_id)
    return {"status": "acknowledged", "alert_id": alert_id}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Factory Compliance Dashboard")
    print("  Open http://localhost:8000 in your browser")
    print("=" * 60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
