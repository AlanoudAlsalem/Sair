"""
SairAI — FastAPI Backend
=========================
REST API + WebSocket server that processes drone video and streams
real-time traffic metrics to the React dashboard.

What this does in plain English:
    This is the "server" — the brain that sits between the AI pipeline
    and the web dashboard you see in the browser. It:
    1. Loads a drone video file
    2. Runs each frame through the detection/tracking pipeline
    3. Sends the results to the dashboard via WebSocket (a live connection
       that pushes updates instantly, like a live chat)
    4. Also provides REST endpoints for things like "give me the current
       metrics" or "list available videos"
"""

from __future__ import annotations
import asyncio
import base64
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    API_HOST, API_PORT, CORS_ORIGINS, YOLO_MODEL, YOLO_CONFIDENCE,
    YOLO_IOU_THRESHOLD, YOLO_DEVICE, YOLO_IMGSZ, VIDEO_DIR, OUTPUT_DIR,
    HOMOGRAPHY_REFERENCE_POINTS, SPEED_STATIONARY_THRESHOLD_KPH,
    FPS_OVERRIDE, LANES_DIR, USE_SAHI, SAHI_SLICE_SIZE, SAHI_OVERLAP_RATIO,
    load_lanes, save_lanes,
)

# ── Import FastAPI (with fallback message) ────────────────────────
try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("=" * 60)
    print("FastAPI is not installed. Run:")
    print("  pip install fastapi uvicorn aiofiles python-multipart")
    print("=" * 60)
    sys.exit(1)

from backend.pipeline import SairAIPipeline

# ── Logging ───────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("sairai.api")

# ── FastAPI App ───────────────────────────────────────────────────
app = FastAPI(
    title="SairAI — Traffic Intelligence API",
    description="Real-time drone-based traffic monitoring for Amman",
    version="1.0.0",
)

# CORS (allow the React frontend to connect)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the React frontend as static files
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

# ── Global state ──────────────────────────────────────────────────
pipeline: Optional[SairAIPipeline] = None
current_metrics: dict = {}
is_processing = False
connected_clients: list = []


# ── Startup ───────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    global pipeline
    logger.info("Initialising SairAI pipeline...")
    pipeline = SairAIPipeline(
        model_path=YOLO_MODEL,
        confidence=YOLO_CONFIDENCE,
        iou_threshold=YOLO_IOU_THRESHOLD,
        device=YOLO_DEVICE,
        imgsz=YOLO_IMGSZ,
        homography_points=HOMOGRAPHY_REFERENCE_POINTS,
        use_sahi=USE_SAHI,
        sahi_slice_size=SAHI_SLICE_SIZE,
        sahi_overlap_ratio=SAHI_OVERLAP_RATIO,
    )
    logger.info("Pipeline ready.")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    LANES_DIR.mkdir(parents=True, exist_ok=True)


# ── REST Endpoints ────────────────────────────────────────────────

@app.get("/")
async def root():
    """Serve the React frontend."""
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "SairAI API is running. Frontend not found — see /docs for API."}


@app.get("/api/health")
async def health():
    return {"status": "ok", "processing": is_processing}


@app.get("/api/videos")
async def list_videos():
    """List available video files in the data/videos directory."""
    videos = []
    for ext in ["*.mp4", "*.mov", "*.avi", "*.mkv"]:
        for f in VIDEO_DIR.glob(ext):
            cap = cv2.VideoCapture(str(f))
            info = {
                "name": f.name,
                "path": str(f),
                "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                "fps": cap.get(cv2.CAP_PROP_FPS),
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "duration_sec": round(
                    int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / max(cap.get(cv2.CAP_PROP_FPS), 1), 1
                ),
            }
            cap.release()
            videos.append(info)
    return {"videos": videos}


@app.get("/api/metrics")
async def get_current_metrics():
    """Get the latest computed metrics snapshot."""
    return current_metrics or {"message": "No metrics yet. Start processing a video."}


@app.get("/api/lanes/{video_name}")
async def get_lanes(video_name: str):
    """Get lane definitions for a video."""
    lanes = load_lanes(video_name)
    return {"video": video_name, "lanes": lanes}


@app.post("/api/lanes/{video_name}")
async def set_lanes(video_name: str, body: dict):
    """
    Save lane definitions for a video and hot-reload them into the pipeline.

    Body: { "lanes": [ { "id", "label", "direction", "polygon": [[x,y],...] }, ... ] }
    """
    lanes = body.get("lanes", [])
    save_lanes(video_name, lanes)
    if pipeline:
        pipeline.set_lanes(lanes)
    return {"status": "ok", "lanes_saved": len(lanes)}


@app.get("/api/preview/{video_name}")
async def preview_frame(video_name: str, frame: int = 0):
    """Return a raw (un-annotated) JPEG frame for the lane annotation tool."""
    video_path = VIDEO_DIR / video_name
    if not video_path.exists():
        raise HTTPException(404, f"Video not found: {video_name}")
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ret, img = cap.read()
    cap.release()
    if not ret:
        raise HTTPException(400, f"Cannot read frame {frame}")
    _, jpeg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return StreamingResponse(iter([jpeg.tobytes()]), media_type="image/jpeg")


@app.get("/api/frame/{video_name}/{frame_number}")
async def get_frame(video_name: str, frame_number: int):
    """Get a specific annotated frame as JPEG."""
    video_path = VIDEO_DIR / video_name
    if not video_path.exists():
        raise HTTPException(404, f"Video not found: {video_name}")

    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise HTTPException(400, f"Cannot read frame {frame_number}")

    annotated, metrics = pipeline.process_frame(frame)
    _, jpeg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])

    return StreamingResponse(
        iter([jpeg.tobytes()]),
        media_type="image/jpeg",
    )


# ── WebSocket: real-time video processing stream ──────────────────

@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """
    WebSocket endpoint for real-time video processing.

    Client sends: {"action": "start", "video": "filename.mp4", "skip_frames": 2}
    Server sends: JSON metrics + base64-encoded JPEG frame per processed frame.
    """
    await websocket.accept()
    connected_clients.append(websocket)
    logger.info(f"WebSocket client connected. Total: {len(connected_clients)}")

    global is_processing, current_metrics

    try:
        while True:
            # Wait for a command from the client
            data = await websocket.receive_text()
            msg = json.loads(data)
            action = msg.get("action", "")

            if action == "start":
                video_name = msg.get("video", "")
                skip = msg.get("skip_frames", 2)
                max_frames = msg.get("max_frames", None)

                video_path = VIDEO_DIR / video_name
                if not video_path.exists():
                    await websocket.send_json({"error": f"Video not found: {video_name}"})
                    continue

                # Load lane definitions for this video
                lanes = load_lanes(video_name)
                if lanes:
                    pipeline.set_lanes(lanes)
                    logger.info(f"Loaded {len(lanes)} lane(s) for {video_name}")

                is_processing = True
                await websocket.send_json({
                    "status": "processing_started",
                    "video": video_name,
                    "lanes": lanes,
                })

                try:
                    loop = asyncio.get_event_loop()
                    frame_gen = pipeline.process_video(
                        str(video_path),
                        skip_frames=skip,
                        max_frames=max_frames,
                    )

                    def _next_frame(gen):
                        try:
                            return next(gen)
                        except StopIteration:
                            return None

                    while is_processing:
                        result = await loop.run_in_executor(None, _next_frame, frame_gen)
                        if result is None:
                            break

                        annotated, metrics = result

                        _, jpeg = cv2.imencode(
                            ".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 70]
                        )
                        frame_b64 = base64.b64encode(jpeg.tobytes()).decode("utf-8")

                        payload = {
                            "type": "frame",
                            "frame": frame_b64,
                            "metrics": metrics.to_dict(),
                        }
                        current_metrics = metrics.to_dict()

                        await websocket.send_json(payload)
                        await asyncio.sleep(0.03)

                except Exception as e:
                    logger.error(f"Processing error: {e}", exc_info=True)
                    await websocket.send_json({"error": str(e)})

                is_processing = False
                await websocket.send_json({"status": "processing_complete"})

            elif action == "stop":
                is_processing = False
                await websocket.send_json({"status": "stopped"})

            elif action == "ping":
                await websocket.send_json({"status": "pong"})

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected.")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
    finally:
        if websocket in connected_clients:
            connected_clients.remove(websocket)


# ── Run ───────────────────────────────────────────────────────────

def start_server():
    """Start the FastAPI server with uvicorn."""
    import uvicorn
    logger.info(f"Starting SairAI server on http://{API_HOST}:{API_PORT}")
    uvicorn.run(
        "backend.main:app",
        host=API_HOST,
        port=API_PORT,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    start_server()
