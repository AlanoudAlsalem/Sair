# SairAI — Complete Project Documentation

**Written for anyone.** You don't need to know programming to follow this. Every technical thing is explained like you're hearing it for the first time.

---

## Table of Contents

1. [What Is This Project?](#1-what-is-this-project)
2. [The Big Picture — How It All Fits Together](#2-the-big-picture)
3. [Why We Chose YOLOv8 (and Why Not DART)](#3-why-yolov8-and-not-dart)
4. [Every File Explained](#4-every-file-explained)
5. [The AI Pipeline — Step by Step](#5-the-ai-pipeline)
6. [SAHI — How We Detect Tiny Vehicles](#6-sahi)
7. [Lanes — How the System Knows Which Road Is Which](#7-lanes)
8. [The Dashboard — What You See in the Browser](#8-the-dashboard)
9. [How to Set Up and Run Everything](#9-how-to-set-up-and-run)
10. [Calibration — Teaching the System About Real-World Distances](#10-calibration)
11. [What Each Metric Means](#11-what-each-metric-means)
12. [Limitations and What Could Be Better](#12-limitations)
13. [Technical Architecture (For the Nerds)](#13-technical-architecture)

---

## 1. What Is This Project?

SairAI is a system that uses a **drone flying above an intersection** combined with **artificial intelligence** to monitor traffic in real time.

Think of it like this: imagine you could hover above the worst roundabout in Amman and instantly know:

- How many cars are there right now?
- How fast are they going?
- How many are stuck in a queue?
- Is anyone driving the wrong way?
- How many vehicles pass through per minute?
- Which specific lane is congested and which is flowing?

That's what SairAI does. It takes the video feed from a drone, runs it through an AI that can see and identify every vehicle, tracks each one across time, assigns it to a lane, and computes all those numbers. Then it displays everything on a nice dashboard that a traffic engineer can look at on their laptop.

**Why does this matter?** Amman is the 5th most congested city in the Arab world. Setting up traditional camera systems at intersections costs $50,000–$90,000 per intersection. A tethered drone costs about $6,000–$10,000 and gives you a BETTER view because it looks straight down instead of from the side.

---

## 2. The Big Picture

Here's what happens from start to finish:

```
DRONE WITH CAMERA
       |
       | (records video of intersection from above)
       v
VIDEO FILE (or live feed)
       |
       | (gets fed into the AI system frame by frame)
       v
STEP 1: DETECTION (YOLOv8 + SAHI)
       | "I see 14 vehicles in this frame — here are their positions"
       | SAHI slices the image into tiles so we don't miss tiny vehicles
       v
STEP 2: TRACKING (ByteTrack)
       | "Car #47 was here last frame, now it's here — same car"
       v
STEP 3: LANE ASSIGNMENT
       | "Car #47 is in Inbound Lane 2"
       v
STEP 4: COORDINATE CONVERSION (Homography)
       | "Car #47 moved 3.2 meters in the last 0.1 seconds"
       v
STEP 5: METRICS COMPUTATION (per-lane + global)
       | "Lane 2: congested (ratio 0.42), avg speed 8 km/h, 6 queued"
       | "Global: avg speed 23 km/h, total flow 42 veh/min"
       v
STEP 6: DASHBOARD
       | Shows everything on a web page with live video, per-lane
       | congestion overlays, numbers, charts, and violation alerts
       v
TRAFFIC ENGINEER MAKES BETTER DECISIONS
```

---

## 3. Why We Chose YOLOv8 (and Why Not DART)

We evaluated two AI models for this project:

### DART (Detect Anything in Real Time)

DART is a brand-new model (2025) built on Meta's SAM3 (Segment Anything Model 3). It's impressive because:

- It's "training-free" — you don't need to teach it what cars look like, it already knows
- It can detect literally ANY object you describe in words
- It does really accurate segmentation (outlines the exact shape of objects, not just rectangles)

**But here's the problem for our use case:**

- **Speed: DART runs at ~16 frames per second** on an expensive RTX 4080 GPU. That sounds fast, but traffic video is typically 30 FPS. So DART can't keep up with real-time video without dropping frames.
- **We don't need open-vocabulary detection.** We know exactly what we're looking for: cars, trucks, buses, and motorcycles. We don't need a model that can detect "anything" — we need one that detects vehicles FAST.
- **Hardware requirements:** DART needs a powerful GPU with lots of memory. A drone's edge computer typically has a much smaller GPU.

### YOLOv8 (You Only Look Once, version 8)

YOLOv8 is the industry standard for real-time object detection. Here's why we chose it:

- **Speed: 100+ FPS** on the same hardware. That's 6x faster than DART.
- **It already knows vehicles perfectly.** It was trained on the COCO dataset, which includes cars, trucks, buses, and motorcycles as standard classes.
- **Battle-tested.** YOLOv8 is used by thousands of companies for traffic monitoring. It's reliable, well-documented, and has a huge community.
- **Lightweight options.** YOLOv8n (nano) can even run on a Raspberry Pi or a small drone computer. DART cannot.
- **Easy to fine-tune.** If we need it to recognise Amman-specific vehicles (yellow taxis, service minibuses), we can retrain it on local data easily.
- **Works brilliantly with SAHI.** We pair YOLOv8 with SAHI (Slicing Aided Hyper Inference) to handle the small-object problem in drone footage. More on this in Section 6.

### The Verdict

| Feature | DART | YOLOv8 + SAHI |
|---------|------|---------------|
| Speed (FPS) | ~16 | 100+ (standard), ~20-30 (with SAHI) |
| Real-time capable? | Barely | Yes |
| Detects vehicles? | Yes (any object) | Yes (pretrained) |
| Small object recall | Good | Excellent with SAHI slicing |
| Edge deployment? | Needs big GPU | Runs on small hardware |
| Community & docs | Very new | Massive ecosystem |
| Accuracy on vehicles | High | High |

**Bottom line:** DART is a cool technology that might be great for future versions (especially if we want to detect unusual objects like stalled vehicles or road debris), but for real-time traffic monitoring with known vehicle classes, **YOLOv8 + SAHI is the right tool.** It's faster, lighter, proven, handles tiny vehicles in drone footage, and is perfectly suited to this job.

---

## 4. Every File Explained

Here's what every single file in this project does:

```
SairAI/
├── run.py                  ← THE MAIN FILE. Run this to start everything.
├── config.py               ← All settings in one place (model, thresholds, lanes, SAHI, etc.)
├── requirements.txt        ← List of Python packages needed (like a shopping list)
├── .gitignore              ← Tells Git which files NOT to upload (videos, model weights, venv, etc.)
├── README.md               ← Quick-start guide for the GitHub page
│
├── backend/                ← The "brain" — all the AI and server code
│   ├── __init__.py         ← Tells Python this folder is a package (basically empty)
│   ├── detection.py        ← YOLOv8 + SAHI wrapper — finds vehicles in each frame
│   ├── tracker.py          ← ByteTrack wrapper — follows vehicles across frames
│   ├── homography.py       ← Pixel-to-meter converter — turns pixels into real distances
│   ├── metrics.py          ← Computes speed, count, queue, flow, per-lane congestion, violations
│   │                          Also contains the LaneManager class for lane assignment
│   ├── pipeline.py         ← Ties detection→tracking→lanes→metrics into one assembly line
│   └── main.py             ← FastAPI server — serves the dashboard and streams data
│
├── frontend/               ← The "face" — what you see in the browser
│   └── index.html          ← Single-file React dashboard with live charts and video
│
├── scripts/                ← Command-line utilities
│   ├── process_video.py    ← Process a video file and save CSV/JSON metrics (no dashboard)
│   └── calibrate.py        ← Interactive tool to set up pixel-to-meter calibration
│
├── data/                   ← Where videos, outputs, and lane definitions go
│   ├── videos/             ← PUT YOUR DRONE VIDEOS HERE
│   ├── output/             ← Processed videos and CSV metrics get saved here
│   └── lanes/              ← Lane definitions per video (JSON files, auto-created)
│
├── docs/                   ← You're reading this right now
│   └── DOCUMENTATION.md    ← This file
│
└── tests/                  ← Unit tests for the pipeline components
    └── test_pipeline.py    ← Tests for homography, detection, tracking, metrics
```

---

## 5. The AI Pipeline — Step by Step

### Step 1: Detection (detection.py)

**What it does:** Looks at a single image (one frame of the video) and finds every vehicle in it.

**How it works:** We use a neural network called YOLOv8 (You Only Look Once). It's called that because unlike older methods that scan the image multiple times, YOLO looks at the entire image in a single pass and immediately outputs all the objects it found.

We use **YOLOv8x** (the "extra-large" variant) with an inference resolution of **1280 pixels** for maximum accuracy on drone footage where vehicles can appear small. The confidence threshold is set low (10%) because we'd rather catch every vehicle and let the tracker filter out false positives than miss real ones.

For each vehicle, it outputs:
- A **bounding box** — the rectangle around the vehicle (top-left corner and bottom-right corner, in pixels)
- A **confidence score** — how sure the AI is that this is actually a vehicle (0% to 100%)
- A **class** — what type of vehicle (car, truck, bus, motorcycle)

We throw away anything that isn't a vehicle (pedestrians, traffic lights, etc.).

When **SAHI is enabled** (which it is by default), detection gets an extra boost for small objects. See Section 6 for how that works.

**Analogy:** It's like having a superhuman spotter on a rooftop who can instantly point at every car and say "that's a car, I'm 92% sure" — except this spotter processes hundreds of images per second.

### Step 2: Tracking (tracker.py)

**What it does:** Connects detections across frames so we know "this car in frame 1 is the SAME car in frame 2."

**How it works:** We use ByteTrack, a tracking algorithm from the `supervision` library. It works like this:
1. In frame 1, it sees 14 vehicles and assigns each an ID: #1, #2, #3, ...
2. In frame 2, it sees 14 vehicles again. It compares the new positions to the old positions and figures out which is which.
3. If a car was at position (100, 200) in frame 1 and there's a car at (105, 202) in frame 2, it's almost certainly the same car — so it keeps the same ID.
4. If a new car appears that wasn't there before, it gets a new ID. If a car disappears (left the frame), its ID is kept in a buffer for 30 frames in case it reappears.

The tracker has been tuned with a low activation threshold (0.20) to pick up even low-confidence detections, and a matching threshold of 0.5 for robust association.

This gives us the "trail" of each vehicle — a list of all positions it's been seen at. This is CRUCIAL because we need the trail to compute speed (how far did it move between frames?).

**Analogy:** It's like a bouncer at a club who puts a stamp on everyone's hand. When someone comes back to the door, the bouncer checks their stamp instead of asking for ID again.

### Step 3: Lane Assignment (metrics.py — LaneManager)

**What it does:** Figures out which lane each vehicle is driving in.

**How it works:** You define lanes as polygons (shapes drawn on the video frame). Each polygon represents one lane. When the system tracks a vehicle, it checks which polygon the vehicle's center point falls inside — that's the lane it's assigned to.

The lane definitions are stored as JSON files in `data/lanes/`, one per video. You can create them through the API or manually. Each lane has:
- An **id** (like `"inbound_1"`)
- A **label** (like `"Inbound Lane 1"`)
- A **direction** (`"inbound"` or `"outbound"`)
- A **polygon** — a list of pixel coordinates that draw the lane's boundary

Once vehicles are assigned to lanes, the system computes **per-lane congestion**. It does this by calculating the ratio of total vehicle bounding-box area to lane polygon area. If this ratio exceeds the congestion threshold (default 35%), that lane is flagged as congested and displayed in red on the video overlay.

**Analogy:** Imagine painting each lane on the road a different colour. When a car drives over the blue paint, it's in the blue lane. Simple.

### Step 4: Coordinate Conversion — Homography (homography.py)

**What it does:** Converts pixel coordinates to real-world meters.

**The problem:** The camera sees everything in pixels. "The car is at pixel (320, 480)." But we need meters to compute speed in km/h and queue length in meters.

**How it works:** We use a mathematical technique called a **homography**. Here's the intuition:

1. You identify 4 points in the image whose real-world positions you know. For example, you know that a standard lane is 3.65 meters wide, so if you can see a lane in the image, you know the real distance between its edges.
2. You tell the computer: "Pixel (200, 150) is at 0 meters, 0 meters in the real world. Pixel (600, 150) is at 12 meters, 0 meters."
3. The computer figures out the mathematical formula to convert ANY pixel to meters.

Because the drone looks **straight down**, this works extremely well. A fixed camera on a pole looks from the side, which creates severe perspective distortion (things far away look smaller). A drone's view is like looking at a map — minimal distortion.

**Analogy:** It's like using the scale bar on a map. If you know 1 cm on the map = 100 m in real life, you can measure any distance on the map and convert it.

### Step 5: Metrics Computation (metrics.py)

**What it does:** Takes all the tracked, lane-assigned, calibrated vehicle data and computes the numbers traffic engineers care about — both globally and per-lane.

**Global metrics:**

- **Vehicle count** — how many vehicles are visible right now, broken down by type (cars, trucks, buses, motorcycles)
- **Average speed** — the mean speed of all vehicles in km/h. Computed by looking at how far each tracked vehicle moved in real-world meters divided by the time between frames.
- **85th percentile speed** — the speed that 85% of vehicles are below. This is what traffic engineers use instead of averages because one speeder doesn't skew it.
- **Queue length** — how many vehicles are moving slower than 5 km/h (basically stopped). These are your "queued" vehicles.
- **Flow rate** — vehicles per minute passing through the intersection. Computed from the rolling window of unique vehicle IDs seen.
- **Violations** — wrong-way driving (vehicle moving opposite to expected direction) or blocked intersection (vehicle stopped in the conflict zone).

**Per-lane metrics (when lanes are defined):**

- **Lane vehicle count** — vehicles in each specific lane
- **Lane average speed** and **85th percentile speed**
- **Lane queue count** — how many vehicles are stopped in that lane
- **Congestion ratio** — the ratio of total vehicle area to lane area (a direct measure of how packed the lane is)
- **Congestion flag** — whether the lane exceeds the congestion threshold (default 35%)

### Step 6: Annotation (pipeline.py)

**What it does:** Draws all the information back onto the video frame so you can see it visually.

Each vehicle gets:
- A coloured bounding box (green = moving, orange = slow, red = queued/stopped)
- A label showing its ID, class, and speed
- Violations get a red circle with a warning label

Lane overlays are drawn underneath the vehicles:
- Each defined lane gets a semi-transparent coloured polygon
- Green = flowing, red = congested
- Lane labels and vehicle counts are drawn on each lane overlay

Plus an info overlay in the corner showing aggregate numbers and per-lane stats.

---

## 6. SAHI — How We Detect Tiny Vehicles

### The Problem

Drone footage is shot from 60–120 meters in the air. At that altitude, a car might only be 30–50 pixels wide in a 1920x1080 frame. A motorcycle might be 15 pixels. YOLOv8, like most object detectors, struggles with objects that small because it was trained on images where objects are hundreds of pixels wide.

### The Solution: SAHI (Slicing Aided Hyper Inference)

SAHI is a technique that solves this without retraining the model:

1. **Slice** the full image into overlapping tiles (default: 640x640 pixels with 25% overlap)
2. **Run YOLOv8** on each tile separately — now vehicles that were 30 pixels in the full image are much larger relative to the tile size
3. **Also run YOLOv8** on the full image (to catch large vehicles that span multiple tiles)
4. **Merge** all detections, removing duplicates using Non-Maximum Suppression (NMS)

This dramatically improves detection of small objects — especially motorcycles and distant vehicles — at the cost of processing time (since we're running the model multiple times per frame).

### Configuration

In `config.py`:
- `USE_SAHI = true` — enables SAHI sliced inference (default: on)
- `SAHI_SLICE_SIZE = 640` — size of each tile in pixels
- `SAHI_OVERLAP_RATIO = 0.25` — how much tiles overlap (25%)

If you need more speed and can tolerate missing small vehicles, set `USE_SAHI = false` to use standard single-pass inference.

---

## 7. Lanes — How the System Knows Which Road Is Which

### Why Lanes Matter

Without lanes, SairAI can tell you "there are 20 vehicles and the average speed is 15 km/h." With lanes, it can tell you "Inbound Lane 1 is congested with 8 stopped vehicles, while Inbound Lane 2 is flowing at 35 km/h." That's vastly more useful for a traffic engineer deciding which signal phase to extend.

### How Lane Definitions Work

Lanes are defined as JSON files stored in `data/lanes/`. Each video gets its own file named `<video_filename>.json`. A lane definition looks like this:

```json
[
    {
        "id": "inbound_1",
        "label": "Inbound Lane 1",
        "direction": "inbound",
        "polygon": [[100, 200], [300, 200], [350, 600], [50, 600]]
    },
    {
        "id": "outbound_1",
        "label": "Outbound Lane 1",
        "direction": "outbound",
        "polygon": [[400, 200], [600, 200], [650, 600], [350, 600]]
    }
]
```

The `polygon` is a list of [x, y] pixel coordinates that trace the boundary of the lane on the video frame.

### How to Define Lanes

**Option 1: Through the API**
Send a POST request to `/api/lanes/<video_name>` with your lane definitions. The dashboard's lane drawing tool uses this endpoint.

**Option 2: Manually**
1. Open a frame of your video in any image editor
2. Note the pixel coordinates of each lane's corners
3. Create a JSON file in `data/lanes/<video_name>.json`

**Option 3: Preview endpoint**
Use `GET /api/preview/<video_name>?frame=0` to get a raw JPEG frame, measure your points, and create the JSON.

### How Congestion Is Measured

The system uses a **bbox-area ratio** to measure congestion:

```
congestion_ratio = (sum of all vehicle bounding box areas in the lane) / (lane polygon area)
```

If this ratio exceeds the `CONGESTION_THRESHOLD` (default: 0.35 or 35%), the lane is flagged as congested. The intuition: if vehicles' boxes cover more than 35% of the lane's area, traffic is dense enough to be considered congested.

### Hot-Swapping Lanes

Lanes can be changed while the system is running. When you POST new lane definitions through the API, the pipeline hot-swaps them in without stopping video processing.

---

## 8. The Dashboard — What You See in the Browser

The dashboard is a **React** web application served by the **FastAPI** backend. When you open it in your browser, you see:

### Header
- The SairAI logo and tagline
- A status indicator (LIVE when processing, READY when connected, OFFLINE when the server is down)
- A frame counter

### Metric Cards (top row)
Six cards showing live numbers:
1. **Total Vehicles** — with car/truck breakdown
2. **Avg Speed** — in km/h
3. **85th % Speed** — in km/h
4. **Queued** — number of stationary vehicles (turns red when > 5)
5. **Flow Rate** — vehicles per minute
6. **Violations** — count of detected rule violations

### Video Feed (main area)
Shows the live annotated video — you can see the bounding boxes, vehicle IDs, and speed labels drawn on top of the actual drone footage. When lanes are defined, you'll also see coloured lane overlays (green = flowing, red = congested) with per-lane vehicle counts.

### Side Panel (right)
- **Controls** — select which video to process, set frame skip rate, start/stop buttons
- **Vehicle Breakdown** — counts of cars vs trucks vs buses vs motorcycles
- **Violations** — list of any detected violations with details

### Time Series Chart (bottom)
A live-updating chart showing vehicles, speed, and queue over time. This is where you see trends — "the queue started building at 8:15 AM and peaked at 8:32 AM."

### How it communicates with the backend
The dashboard uses a **WebSocket** — think of it like an open phone line between the browser and the server. The server processes video frames in a background thread (using `run_in_executor` so the server stays responsive) and pushes results to the browser instantly. No need for the browser to keep asking "any updates?" — the server just tells it.

---

## 9. How to Set Up and Run

### Prerequisites
- Python 3.9 or newer
- A GPU is recommended but not required (CPU works, just slower). On Mac, the system defaults to MPS (Apple Silicon GPU).
- A drone video file

### Step 1: Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/SairAI.git
cd SairAI
```

### Step 2: Create a virtual environment (recommended)
```bash
python -m venv venv

# On Mac/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### Step 3: Install dependencies
```bash
pip install -r requirements.txt
```

This installs everything: YOLOv8, SAHI, ByteTrack (supervision), FastAPI, OpenCV, and all the other packages.

**First time only:** YOLOv8 will automatically download its model weights the first time you run it. The default model is `yolov8x.pt` (~130 MB for extra-large).

### Step 4: Add your drone videos
Copy your drone video files into:
```
SairAI/data/videos/
```

### Step 5: (Optional) Define lanes
Create lane definitions in `data/lanes/<video_name>.json` or use the API after starting the server.

### Step 6: Run the dashboard
```bash
python run.py
```

Open your browser to **http://localhost:8000**. You'll see the dashboard. Select your video from the dropdown and click "Start Processing."

### Alternative: Process a video from the command line
If you just want to process a video and get CSV/JSON output without the dashboard:
```bash
python scripts/process_video.py data/videos/your_video.mp4
```

Additional CLI options:
```bash
python scripts/process_video.py data/videos/your_video.mp4 \
    --skip 2 \                    # Process every 3rd frame (faster)
    --max-frames 500 \            # Stop after 500 frames
    --model yolov8m.pt \          # Use a different model
    --imgsz 1280 \                # Set inference resolution
    --device cuda:0               # Use GPU
```

This creates:
- `data/output/your_video_annotated.mp4` — the video with boxes drawn
- `data/output/your_video_metrics.csv` — per-frame metrics
- `data/output/your_video_summary.json` — overall summary

---

## 10. Calibration — Teaching the System About Real-World Distances

For accurate speed and distance measurements, you need to calibrate the system for your specific video. This means telling the system how many pixels equal how many meters.

### Quick calibration (easiest)

If you know the approximate scale:
1. Find something of known width in the video (e.g., a lane marking)
2. Measure its width in pixels (open the video frame in any image editor)
3. A standard lane in Jordan is 3.65 meters wide
4. Calculate: `pixels_per_meter = lane_width_in_pixels / 3.65`
5. Edit `config.py` and update the `HOMOGRAPHY_REFERENCE_POINTS` or just use `calibrate_from_scale()` in your code

### Full calibration (most accurate)

Use the interactive calibration tool:
```bash
python scripts/calibrate.py data/videos/your_video.mp4
```

This opens the first frame of your video. Click on 4+ points whose real-world positions you know (lane edges, crosswalk corners, etc.), then enter their real-world coordinates in meters. The tool computes the homography and gives you the values to paste into `config.py`.

### What if you don't calibrate?

The system will still work, but speeds will be in "relative" units, not accurate km/h. Vehicle counts, queue counts, lane congestion ratios, and flow rates are still accurate regardless of calibration since they don't depend on meter conversion.

---

## 11. What Each Metric Means

### Global Metrics

| Metric | What It Means | Why Traffic Engineers Care |
|--------|--------------|--------------------------|
| **Total Vehicles** | How many vehicles are visible in the current frame | Basic load indicator — "how busy is this intersection right now?" |
| **Avg Speed** | The average speed of all tracked vehicles | Low speed = congestion. If avg speed drops below 15 km/h, the intersection is struggling. |
| **85th Percentile Speed** | 85% of vehicles are going slower than this | The standard metric for setting speed limits and evaluating traffic signal timing. Less affected by outliers than average. |
| **Queue Length** | Number of vehicles moving <5 km/h | Directly measures how many people are stuck. More useful than "congestion level" because it's a real number. |
| **Flow Rate (veh/min)** | Vehicles passing through per minute | The fundamental measure of intersection capacity. If flow drops while count rises, the intersection is becoming saturated. |
| **Violations** | Wrong-way driving, blocked intersections | Safety events that need immediate attention. In a production system, these could trigger alerts to enforcement. |

### Per-Lane Metrics (when lanes are defined)

| Metric | What It Means | Why It Matters |
|--------|--------------|----------------|
| **Lane Vehicle Count** | Vehicles currently in this specific lane | Shows which lanes are carrying load and which are underutilised |
| **Lane Avg / P85 Speed** | Speed stats for this lane only | A slow lane next to a fast lane suggests a bottleneck or lane blockage |
| **Lane Queue Count** | Stopped vehicles in this lane | Pinpoints exactly where queues are forming |
| **Congestion Ratio** | Vehicle area ÷ lane area (0.0 – 1.0) | Direct physical measure of how packed the lane is. Above 0.35 = congested. |
| **Congestion Flag** | Is the lane congested? (yes/no) | Quick glance indicator — congested lanes are drawn in red on the video |

---

## 12. Limitations and What Could Be Better

### Current limitations

1. **This is a Proof of Concept.** It processes pre-recorded video, not a live drone feed. The code is structured to support live feeds, but the actual hardware integration (talking to a drone's camera over a network) isn't implemented.

2. **Model accuracy on Jordanian vehicles.** YOLOv8's default weights were trained mostly on Western vehicles. It knows "car" and "truck" and "bus," but it might confuse a yellow Amman taxi with a regular car, or miss service minibuses. A production version would need fine-tuning on Amman footage.

3. **Lane definitions are manual.** You currently need to draw lane polygons by hand (or through the API). A future version could auto-detect lane markings and define lanes automatically.

4. **Calibration is manual.** You have to tell the system the pixel-to-meter conversion. A future version could auto-detect lane markings and self-calibrate.

5. **Weather sensitivity.** Drone cameras struggle in heavy rain, dust, or strong winds. The AI model's accuracy also drops in poor lighting.

6. **Privacy.** While top-down drone footage generally doesn't capture faces, license plates ARE visible. The architecture supports plate blurring (see `BLUR_LICENSE_PLATES` in config), but it needs a plate-detection model that isn't included in this PoC.

7. **SAHI adds processing time.** Sliced inference dramatically improves small-object detection but makes each frame take 3-5x longer than standard inference. For real-time applications, you may need to choose between SAHI accuracy and raw speed.

### What a production version would add

- Live drone feed integration (RTSP/RTMP video stream)
- Fine-tuned model on Amman-specific vehicles
- Automatic lane boundary detection from road markings
- Signal timing recommendations (or a reinforcement learning agent for automatic optimization)
- Historical data storage and trend analysis
- Alert system (SMS/email when congestion ratio exceeds threshold)
- Multi-intersection monitoring (dashboard showing all monitored intersections on a map)
- Automatic lane-marking-based homography calibration

---

## 13. Technical Architecture (For the Nerds)

### Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| Object detection | YOLOv8x (Ultralytics) | Best accuracy for drone footage at usable speed |
| Small-object boost | SAHI (Slicing Aided Hyper Inference) | Tiles the image so tiny vehicles get detected |
| Tracking | ByteTrack (via supervision) | State-of-the-art multi-object tracker, handles occlusion well |
| Lane management | Custom LaneManager class | Polygon-based lane assignment + bbox-area congestion metric |
| Coordinate transform | OpenCV homography | Standard, battle-tested, fast |
| Backend API | FastAPI + WebSocket | Async Python, great WebSocket support, auto-generates API docs |
| Frontend | React (CDN) + Chart.js | No build step needed, works as a single HTML file |
| Video I/O | OpenCV VideoCapture | Universal codec support |

### Data flow

```
VideoCapture.read()                 # Get frame (numpy array, H×W×3, BGR)
        │
   cv2.resize(RESIZE_RATIO)        # Optional resize for performance
        │
VehicleDetector.detect()            # → List[Detection]
        │  Uses SAHI slicing if enabled:
        │    1. Slice into 640×640 tiles with 25% overlap
        │    2. Run YOLOv8x on each tile + full image
        │    3. Merge with NMS (IoS metric)
        │
VehicleTracker.update()             # → List[TrackedVehicle]
        │  ByteTrack assigns persistent IDs
        │  (activation_thresh=0.20, lost_buffer=30)
        │
LaneManager.assign_vehicle_to_lane()  # → lane_id per vehicle
        │  Point-in-polygon test for each vehicle center
        │
HomographyTransform.pixel_to_m()    # pixel coords → meter coords
        │
TrafficMetricsEngine.compute()      # → FrameMetrics
        │  Global: counts, speeds, queues, flow
        │  Per-lane: counts, speeds, congestion_ratio, is_congested
        │
LaneManager.compute_lane_metrics()  # → Dict[lane_id, LaneMetrics]
        │  congestion_ratio = sum(bbox_area) / lane_polygon_area
        │
pipeline._annotate_frame()          # Draw lanes (green/red) + boxes + HUD
        │
        ├──→ WebSocket.send_json()   # Metrics + base64 JPEG → browser
        │    (via run_in_executor for non-blocking async)
        │
        └──→ VideoWriter.write()     # Save annotated frame to output video
```

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Serve the React dashboard |
| GET | `/api/health` | Health check (is the server running?) |
| GET | `/api/videos` | List available video files with metadata |
| GET | `/api/metrics` | Get latest metrics snapshot |
| GET | `/api/lanes/{video_name}` | Get lane definitions for a video |
| POST | `/api/lanes/{video_name}` | Save/update lane definitions (hot-swaps into pipeline) |
| GET | `/api/preview/{video_name}?frame=N` | Get a raw un-annotated JPEG frame |
| GET | `/api/frame/{video}/{frame}` | Get a specific annotated frame as JPEG |
| WS | `/ws/stream` | WebSocket for real-time video processing |
| GET | `/docs` | Auto-generated API documentation (Swagger UI) |

### WebSocket Protocol

The client sends JSON commands:
```json
{"action": "start", "video": "filename.mp4", "skip_frames": 2}
{"action": "stop"}
{"action": "ping"}
```

On `start`, the server:
1. Loads lane definitions for the video (from `data/lanes/`)
2. Hot-swaps them into the pipeline
3. Begins processing frames in a background executor thread
4. Streams results back:

```json
{
    "type": "frame",
    "frame": "<base64-encoded JPEG>",
    "metrics": {
        "frame_number": 142,
        "total_vehicles": 14,
        "avg_speed_kph": 23.4,
        "queue_vehicles": 3,
        "flow_rate_vpm": 42.1,
        "lanes": {
            "inbound_1": {
                "lane_id": "inbound_1",
                "label": "Inbound Lane 1",
                "direction": "inbound",
                "vehicle_count": 6,
                "avg_speed_kph": 12.3,
                "congestion_ratio": 0.42,
                "is_congested": true,
                "polygon": [[100,200], [300,200], ...]
            }
        },
        "violations": [],
        "vehicles": [
            {"id": 47, "class": "car", "speed_kph": 31.2, "lane_id": "inbound_1", ...}
        ]
    }
}
```

### Configuration Reference

All config lives in `config.py`. Key settings:

| Setting | Default | What It Does |
|---------|---------|-------------|
| `YOLO_MODEL` | `yolov8x.pt` | Which YOLOv8 model to use (n/s/m/l/x) |
| `YOLO_CONFIDENCE` | `0.10` | Minimum detection confidence (low = catch more, risk false positives) |
| `YOLO_IMGSZ` | `1280` | Inference resolution in pixels |
| `YOLO_DEVICE` | `mps` | Compute device (mps = Apple GPU, cuda:0 = NVIDIA, cpu) |
| `USE_SAHI` | `true` | Enable SAHI sliced inference |
| `SAHI_SLICE_SIZE` | `640` | Tile size for SAHI slicing |
| `SAHI_OVERLAP_RATIO` | `0.25` | Overlap between SAHI tiles |
| `TRACKER_MATCH_THRESH` | `0.20` | ByteTrack activation threshold |
| `TRACKER_LOST_FRAMES` | `30` | Frames to keep a lost track before retiring |
| `CONGESTION_THRESHOLD` | `0.35` | Lane congestion ratio threshold (above = congested) |
| `RESIZE_RATIO` | `1` | Frame resize before processing (1 = no resize) |
| `SPEED_STATIONARY_THRESHOLD_KPH` | `5.0` | Below this speed = "queued" |

### Performance Considerations

- **SAHI trade-off:** SAHI sliced inference improves recall for small objects dramatically but takes 3-5x longer per frame. For maximum speed, disable SAHI (`USE_SAHI=false`) and use standard inference.
- **Model size vs speed:** `yolov8x` (extra-large) gives best accuracy but is slowest. For faster processing, try `yolov8m` or `yolov8s`. For edge deployment, use `yolov8n`.
- **Frame skipping:** Processing every frame is unnecessary for traffic metrics. Skipping every other frame (skip=1) halves the compute load with negligible accuracy loss.
- **JPEG quality:** Frames are encoded at 70% quality for WebSocket transmission — good balance between quality and bandwidth.
- **Async processing:** The WebSocket handler uses `run_in_executor` to process frames in a background thread, keeping the server responsive to new connections and API requests.
- **Rolling windows:** Metrics are computed over a 60-second rolling window, not all-time. This means the numbers reflect recent traffic, not the average since you started.

---

*This documentation was written as part of the SairAI project by Anoud. Last updated: April 2026.*
