# SairAI — Drone-Based Real-Time Traffic Intelligence for Amman

A tethered drone + AI vision system that gives Amman traffic authorities a bird's-eye, multi-lane, real-time view of congested intersections at roughly **one-tenth the cost** of fixed-camera instrumentation.

**Pillar:** Robotics & Drones (with AI/ML integration)
**Author:** Anoud

---

## What It Does

SairAI processes drone video footage of traffic intersections and computes real-time metrics:

- **Vehicle detection & counting** — cars, trucks, buses, motorcycles (using YOLOv8x + SAHI for small-object accuracy)
- **Per-lane congestion tracking** — define lane polygons, get per-lane vehicle counts, speed averages, and congestion ratios
- **Speed estimation** — average and 85th percentile, in km/h
- **Queue length** — number of stationary vehicles, globally and per-lane
- **Flow rate** — vehicles per minute passing through
- **Violation detection** — wrong-way driving, blocked intersections

All visualised on a live React dashboard with annotated video, per-lane congestion overlays, metric cards, and time-series charts.

## Architecture

```
Drone Camera → YOLOv8x + SAHI Detection → ByteTrack Tracking
    → Lane Assignment → Homography (px→m) → Traffic Metrics Engine
    → FastAPI + WebSocket → React Dashboard
```

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/SairAI.git
cd SairAI

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate   # Mac/Linux
# venv\Scripts\activate    # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your drone video
cp your_drone_video.mp4 data/videos/

# 5. (Optional) Define lanes for your video
# Create data/lanes/your_drone_video.mp4.json — see docs for format

# 6. Run the dashboard
python run.py
# Open http://localhost:8000 in your browser
```

### Command-line processing (no dashboard)

```bash
python scripts/process_video.py data/videos/your_video.mp4

# With options:
python scripts/process_video.py data/videos/your_video.mp4 \
    --skip 2 --max-frames 500 --model yolov8m.pt --imgsz 1280
```

## Project Structure

```
SairAI/
├── run.py                  # Main entry point — starts the server
├── config.py               # All configuration (model, SAHI, lanes, thresholds)
├── requirements.txt        # Python dependencies
├── backend/
│   ├── detection.py        # YOLOv8 + SAHI vehicle detection
│   ├── tracker.py          # ByteTrack multi-object tracking
│   ├── homography.py       # Pixel-to-meter coordinate transform
│   ├── metrics.py          # Traffic metrics + LaneManager + congestion
│   ├── pipeline.py         # Full processing pipeline with lane overlays
│   └── main.py             # FastAPI server + WebSocket + lane API
├── frontend/
│   └── index.html          # React dashboard (single file, no build step)
├── scripts/
│   ├── process_video.py    # CLI batch processing tool
│   └── calibrate.py        # Homography calibration helper
├── data/
│   ├── videos/             # Place drone videos here
│   ├── output/             # Processed results
│   └── lanes/              # Per-video lane definitions (JSON)
└── docs/
    └── DOCUMENTATION.md    # Comprehensive beginner-friendly documentation
```

## Key Features

### SAHI (Slicing Aided Hyper Inference)
Drone footage shows vehicles as tiny objects. SAHI slices each frame into overlapping 640x640 tiles, runs YOLOv8 on each tile, then merges results. This dramatically improves detection of motorcycles and distant vehicles.

### Per-Lane Congestion
Define lane polygons (via JSON or API), and the system computes per-lane metrics: vehicle count, speed, queue, and a congestion ratio (vehicle area / lane area). Lanes turn red when congested.

### Lane API
- `GET /api/lanes/{video}` — get lane definitions
- `POST /api/lanes/{video}` — save/update lanes (hot-swaps into running pipeline)
- `GET /api/preview/{video}?frame=0` — get a raw frame for lane annotation

## Why Drones Beat Fixed Cameras

| Dimension | Fixed Cameras (6-approach) | Tethered Drone |
|-----------|---------------------------|----------------|
| Cost per intersection | $50,000–$90,000 | $6,000–$10,000 |
| View angle | Shallow, per-approach | Near-vertical, full intersection |
| Occlusion | Severe (buses, trucks) | Negligible |
| Deployment time | Months | Days |
| Reusable at another site | No | Yes |

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Detection | YOLOv8x (Ultralytics) + SAHI |
| Tracking | ByteTrack (supervision) |
| Lane management | Custom LaneManager (polygon-based) |
| Backend | FastAPI + WebSocket |
| Frontend | React + Chart.js |
| Video I/O | OpenCV |

## Configuration

Key settings in `config.py` (all overridable via environment variables):

| Setting | Default | Description |
|---------|---------|-------------|
| `YOLO_MODEL` | `yolov8x.pt` | Model size (n/s/m/l/x) |
| `YOLO_CONFIDENCE` | `0.10` | Detection confidence threshold |
| `YOLO_IMGSZ` | `1280` | Inference resolution |
| `YOLO_DEVICE` | `mps` | Compute device (mps/cuda:0/cpu) |
| `USE_SAHI` | `true` | Enable SAHI sliced inference |
| `CONGESTION_THRESHOLD` | `0.35` | Lane congestion ratio threshold |
| `RESIZE_RATIO` | `1` | Frame resize factor |

## Documentation

See [`docs/DOCUMENTATION.md`](docs/DOCUMENTATION.md) for a comprehensive, beginner-friendly explanation of everything — how the AI works, what SAHI does, how lanes work, why we chose YOLOv8 over DART, and more.

## Limitations

- **Proof of Concept** — processes pre-recorded video, not live drone feeds
- **Western-trained model** — may need fine-tuning for Amman-specific vehicle types
- **Manual lane/calibration setup** — requires hand-drawn lane polygons and manual pixel-to-meter calibration
- **SAHI speed trade-off** — sliced inference is 3-5x slower than standard inference
- **Weather dependent** — rain, dust, and wind can affect drone operations

## License

This project was created for educational and competition purposes.

---

*Built for the Robotics & Drones pillar by Anoud, April 2026*
