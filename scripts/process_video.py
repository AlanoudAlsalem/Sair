#!/usr/bin/env python3
"""
SairAI — Standalone Video Processing Script
=============================================
Run this from the command line to process a drone video and produce:
    1. An annotated output video (boxes drawn on vehicles)
    2. A CSV file with per-frame traffic metrics
    3. A JSON summary of the whole video

Usage:
    python scripts/process_video.py data/videos/my_clip.mp4

    # Process only every 3rd frame (faster):
    python scripts/process_video.py data/videos/my_clip.mp4 --skip 2

    # Limit to first 500 frames:
    python scripts/process_video.py data/videos/my_clip.mp4 --max-frames 500

    # Use a beefier model:
    python scripts/process_video.py data/videos/my_clip.mp4 --model yolov8m.pt
"""

from __future__ import annotations
import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.pipeline import SairAIPipeline
from config import (
    YOLO_MODEL, YOLO_CONFIDENCE, YOLO_IOU_THRESHOLD, YOLO_DEVICE,
    YOLO_IMGSZ, HOMOGRAPHY_REFERENCE_POINTS, OUTPUT_DIR,
    USE_SAHI, SAHI_SLICE_SIZE, SAHI_OVERLAP_RATIO, load_lanes,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("sairai.cli")


def main():
    parser = argparse.ArgumentParser(
        description="SairAI — Process a drone traffic video and output metrics."
    )
    parser.add_argument("video", help="Path to input video file")
    parser.add_argument("--output", "-o", help="Output directory (default: data/output/)")
    parser.add_argument("--model", default=YOLO_MODEL, help="YOLOv8 model weights")
    parser.add_argument("--confidence", type=float, default=YOLO_CONFIDENCE)
    parser.add_argument("--skip", type=int, default=0, help="Skip N frames between processed frames")
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after N frames")
    parser.add_argument("--no-video-output", action="store_true", help="Skip writing annotated video")
    parser.add_argument("--device", default=YOLO_DEVICE, help="Device: cpu, cuda:0, mps")
    parser.add_argument("--imgsz", type=int, default=YOLO_IMGSZ, help="YOLO inference resolution")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        logger.error(f"Video file not found: {video_path}")
        sys.exit(1)

    output_dir = Path(args.output) if args.output else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = video_path.stem
    output_video = output_dir / f"{stem}_annotated.mp4"
    output_csv = output_dir / f"{stem}_metrics.csv"
    output_json = output_dir / f"{stem}_summary.json"

    # Load lane definitions if they exist for this video
    lanes = load_lanes(video_path.name)

    logger.info("Initialising SairAI pipeline...")
    pipeline = SairAIPipeline(
        model_path=args.model,
        confidence=args.confidence,
        device=args.device,
        imgsz=args.imgsz,
        homography_points=HOMOGRAPHY_REFERENCE_POINTS,
        lanes=lanes or None,
        use_sahi=USE_SAHI,
        sahi_slice_size=SAHI_SLICE_SIZE,
        sahi_overlap_ratio=SAHI_OVERLAP_RATIO,
    )

    # Open CSV writer
    csv_file = open(output_csv, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "frame", "timestamp_sec", "total_vehicles", "cars", "trucks",
        "buses", "motorcycles", "avg_speed_kph", "p85_speed_kph",
        "queue_vehicles", "flow_rate_vpm", "violations",
    ])

    # Process video
    logger.info(f"Processing: {video_path}")
    start_time = time.time()
    all_metrics = []

    video_output_path = None if args.no_video_output else str(output_video)

    for annotated, metrics in pipeline.process_video(
        str(video_path),
        output_path=video_output_path,
        skip_frames=args.skip,
        max_frames=args.max_frames,
    ):
        m = metrics.to_dict()
        all_metrics.append(m)

        # Write CSV row
        csv_writer.writerow([
            m["frame_number"],
            m["timestamp"],
            m["total_vehicles"],
            m["breakdown"]["cars"],
            m["breakdown"]["trucks"],
            m["breakdown"]["buses"],
            m["breakdown"]["motorcycles"],
            m["avg_speed_kph"],
            m["p85_speed_kph"],
            m["queue_vehicles"],
            m["flow_rate_vpm"],
            len(m["violations"]),
        ])

        # Progress logging every 100 frames
        if len(all_metrics) % 100 == 0:
            elapsed = time.time() - start_time
            fps = len(all_metrics) / elapsed
            logger.info(
                f"  Processed {len(all_metrics)} frames "
                f"({fps:.1f} fps) — {m['total_vehicles']} vehicles"
            )

    csv_file.close()
    elapsed = time.time() - start_time

    # Compute summary
    if all_metrics:
        avg_vehicles = sum(m["total_vehicles"] for m in all_metrics) / len(all_metrics)
        avg_speed = sum(m["avg_speed_kph"] for m in all_metrics) / len(all_metrics)
        max_queue = max(m["queue_vehicles"] for m in all_metrics)
        total_violations = sum(len(m["violations"]) for m in all_metrics)
    else:
        avg_vehicles = avg_speed = max_queue = total_violations = 0

    summary = {
        "video": str(video_path),
        "total_frames_processed": len(all_metrics),
        "processing_time_sec": round(elapsed, 2),
        "processing_fps": round(len(all_metrics) / max(elapsed, 0.001), 2),
        "avg_vehicles_per_frame": round(avg_vehicles, 1),
        "avg_speed_kph": round(avg_speed, 1),
        "max_queue_vehicles": max_queue,
        "total_violation_events": total_violations,
        "model": args.model,
    }

    with open(output_json, "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary
    logger.info("=" * 50)
    logger.info("PROCESSING COMPLETE")
    logger.info("=" * 50)
    logger.info(f"  Frames processed: {len(all_metrics)}")
    logger.info(f"  Time: {elapsed:.1f}s ({summary['processing_fps']:.1f} fps)")
    logger.info(f"  Avg vehicles/frame: {avg_vehicles:.1f}")
    logger.info(f"  Avg speed: {avg_speed:.1f} km/h")
    logger.info(f"  Max queue: {max_queue} vehicles")
    logger.info(f"  Violations: {total_violations}")
    logger.info(f"  Output video: {output_video}")
    logger.info(f"  Metrics CSV:  {output_csv}")
    logger.info(f"  Summary JSON: {output_json}")


if __name__ == "__main__":
    main()
