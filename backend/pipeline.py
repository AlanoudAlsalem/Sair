"""
SairAI — Video Processing Pipeline
====================================
Ties together detection, tracking, homography, metrics, and lane
management into a single class that processes video frames.

    Frame → Detect vehicles → Track them → Assign to lanes →
    Compute speed/count/queue per lane → Check congestion → Annotate
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Generator, List, Optional, Tuple

import cv2
import numpy as np

from backend.detection import VehicleDetector
from backend.tracker import VehicleTracker
from backend.homography import HomographyTransform
from backend.metrics import (
    TrafficMetricsEngine, FrameMetrics, LaneManager, ViolationDetector,
)

from config import RESIZE_RATIO, CONGESTION_THRESHOLD

logger = logging.getLogger(__name__)


class SairAIPipeline:
    """
    Main processing pipeline. Feed it frames, get back annotated
    images and traffic metrics (with per-lane stats).
    """

    def __init__(
        self,
        model_path: str = "yolov8x.pt",
        confidence: float = 0.15,
        iou_threshold: float = 0.45,
        device: str = "",
        imgsz: int = 1280,
        homography_points=None,
        pixels_per_meter: float = 10.0,
        fps: float = 30.0,
        speed_threshold_kph: float = 5.0,
        lanes: Optional[List[dict]] = None,
        use_sahi: bool = True,
        sahi_slice_size: int = 640,
        sahi_overlap_ratio: float = 0.25,
    ):
        from config import TRACKER_MATCH_THRESH, TRACKER_LOST_FRAMES

        self.detector = VehicleDetector(
            model_path=model_path,
            confidence=confidence,
            iou_threshold=iou_threshold,
            device=device,
            imgsz=imgsz,
            use_sahi=use_sahi,
            sahi_slice_size=sahi_slice_size,
            sahi_overlap_ratio=sahi_overlap_ratio,
        )

        self.tracker = VehicleTracker(
            track_activation_threshold=TRACKER_MATCH_THRESH,
            lost_track_buffer=TRACKER_LOST_FRAMES,
        )

        self.homography = HomographyTransform()
        if homography_points and len(homography_points) >= 4:
            self.homography.calibrate(homography_points)
        else:
            self.homography.calibrate_from_scale(pixels_per_meter)

        self.lane_manager = None
        if lanes:
            self.lane_manager = LaneManager(
                lanes=lanes,
                congestion_threshold=CONGESTION_THRESHOLD,
            )
            logger.info(f"Lane manager loaded with {len(lanes)} lane(s)")

        self.metrics_engine = TrafficMetricsEngine(
            homography=self.homography,
            fps=fps,
            speed_threshold_kph=speed_threshold_kph,
            lane_manager=self.lane_manager,
        )

        self.violation_detector = ViolationDetector()
        self.fps = fps
        self.frame_count = 0

    def set_lanes(self, lanes: List[dict]):
        """Hot-swap lane definitions (e.g. from the API)."""
        self.lane_manager = LaneManager(
            lanes=lanes,
            congestion_threshold=CONGESTION_THRESHOLD,
        )
        self.metrics_engine.lane_manager = self.lane_manager
        logger.info(f"Lanes updated: {len(lanes)} lane(s)")

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, FrameMetrics]:
        self.frame_count += 1

        frame = cv2.resize(frame, None, fx=RESIZE_RATIO, fy=RESIZE_RATIO)
        detections = self.detector.detect(frame)
        tracked = self.tracker.update(detections, self.frame_count)
        metrics = self.metrics_engine.compute(tracked, self.frame_count)

        violations = self.violation_detector.check(tracked, self.fps)
        metrics.violations = violations

        annotated = self._annotate_frame(frame.copy(), tracked, metrics)
        return annotated, metrics

    def process_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        skip_frames: int = 0,
        max_frames: Optional[int] = None,
    ) -> Generator[Tuple[np.ndarray, FrameMetrics], None, None]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.fps = video_fps
        self.metrics_engine.fps = video_fps

        logger.info(
            f"Processing video: {video_path}\n"
            f"  Resolution: {width}x{height}, FPS: {video_fps:.1f}, "
            f"Frames: {total_frames}"
        )

        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, video_fps, (width, height))

        processed = 0
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if skip_frames > 0 and frame_idx % (skip_frames + 1) != 0:
                continue

            annotated, metrics = self.process_frame(frame)
            processed += 1

            if writer:
                writer.write(annotated)

            yield annotated, metrics

            if max_frames and processed >= max_frames:
                break

        cap.release()
        if writer:
            writer.release()
            logger.info(f"Output video saved to: {output_path}")

        logger.info(f"Processed {processed} frames out of {total_frames} total.")

    # ── Annotation ─────────────────────────────────────────────────

    def _annotate_frame(
        self,
        frame: np.ndarray,
        tracked_vehicles,
        metrics: FrameMetrics,
    ) -> np.ndarray:
        """Draw lane overlays, bounding boxes, IDs, speeds, and info HUD."""

        # Draw lane polygons first (underneath everything)
        self._draw_lanes(frame, metrics)

        for tv in tracked_vehicles:
            x1, y1, x2, y2 = tv.bbox.astype(int)

            speed = 0.0
            for vp in metrics.vehicle_positions:
                if vp["id"] == tv.track_id:
                    speed = vp["speed_kph"]
                    break

            if speed < 5.0:
                color = (0, 0, 255)
            elif speed < 20.0:
                color = (0, 165, 255)
            else:
                color = (0, 255, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = f"#{tv.track_id} {tv.class_name} {speed:.0f}km/h"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(
                frame,
                (x1, y1 - label_size[1] - 6),
                (x1 + label_size[0], y1),
                color, -1,
            )
            cv2.putText(
                frame, label, (x1, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
            )

        # Global HUD (top-left)
        overlay_lines = [
            f"Vehicles: {metrics.total_vehicles}",
            f"Avg Speed: {metrics.avg_speed_kph:.1f} km/h",
            f"Queue: {metrics.total_queue_vehicles}",
            f"Flow: {metrics.flow_rate_vpm:.1f} veh/min",
            f"Frame: {metrics.frame_number}",
        ]

        y_offset = 30
        for line in overlay_lines:
            cv2.putText(frame, line, (15, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, line, (15, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
            y_offset += 30

        # Per-lane HUD (below global stats)
        for lid, lm in metrics.lane_metrics.items():
            status = "CONGESTED" if lm.is_congested else "CLEAR"
            color = (0, 0, 255) if lm.is_congested else (0, 200, 0)
            line = f"{lm.label}: {lm.vehicle_count}v  {lm.avg_speed_kph:.0f}km/h  [{status}]"
            cv2.putText(frame, line, (15, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_offset += 28

        # Violations
        for v in metrics.violations:
            vx, vy = int(v["position"][0]), int(v["position"][1])
            cv2.circle(frame, (vx, vy), 30, (0, 0, 255), 3)
            cv2.putText(frame, v["type"].upper(), (vx - 40, vy - 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        return frame

    def _draw_lanes(self, frame: np.ndarray, metrics: FrameMetrics):
        """Draw semi-transparent lane overlays — green (clear) or red (congested)."""
        if not metrics.lane_metrics:
            return

        overlay = frame.copy()
        for lid, lm in metrics.lane_metrics.items():
            if not lm.polygon:
                continue
            pts = np.array(lm.polygon, dtype=np.int32)

            if lm.is_congested:
                fill_color = (0, 0, 200)
                border_color = (0, 0, 255)
            else:
                fill_color = (0, 160, 0)
                border_color = (0, 220, 0)

            cv2.fillPoly(overlay, [pts], fill_color)
            cv2.polylines(frame, [pts], True, border_color, 2)

            # Lane label at centroid
            cx = int(np.mean(pts[:, 0]))
            cy = int(np.mean(pts[:, 1]))
            cv2.putText(frame, lm.label, (cx - 40, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
