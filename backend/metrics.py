"""
SairAI — Traffic Metrics Engine
=================================
Computes real-time traffic metrics from tracked vehicle data,
including per-lane congestion and speed averages.
"""

from __future__ import annotations
import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from backend.tracker import TrackedVehicle
from backend.homography import HomographyTransform

logger = logging.getLogger(__name__)


@dataclass
class LaneMetrics:
    """Metrics for a single lane / region."""
    lane_id: str
    label: str = ""
    direction: str = ""
    vehicle_count: int = 0
    avg_speed_kph: float = 0.0
    p85_speed_kph: float = 0.0
    queue_length_vehicles: int = 0
    queue_length_meters: float = 0.0
    flow_rate_vpm: float = 0.0
    congestion_ratio: float = 0.0
    is_congested: bool = False
    polygon: List[List[float]] = field(default_factory=list)


@dataclass
class FrameMetrics:
    """All metrics for a single processed frame."""
    frame_number: int
    timestamp: float
    total_vehicles: int = 0
    total_cars: int = 0
    total_trucks: int = 0
    total_buses: int = 0
    total_motorcycles: int = 0
    avg_speed_kph: float = 0.0
    p85_speed_kph: float = 0.0
    total_queue_vehicles: int = 0
    flow_rate_vpm: float = 0.0
    lane_metrics: Dict[str, LaneMetrics] = field(default_factory=dict)
    violations: List[dict] = field(default_factory=list)
    vehicle_positions: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialise to a JSON-friendly dictionary for the API."""
        return {
            "frame_number": self.frame_number,
            "timestamp": round(self.timestamp, 3),
            "total_vehicles": self.total_vehicles,
            "breakdown": {
                "cars": self.total_cars,
                "trucks": self.total_trucks,
                "buses": self.total_buses,
                "motorcycles": self.total_motorcycles,
            },
            "avg_speed_kph": round(self.avg_speed_kph, 1),
            "p85_speed_kph": round(self.p85_speed_kph, 1),
            "queue_vehicles": self.total_queue_vehicles,
            "flow_rate_vpm": round(self.flow_rate_vpm, 1),
            "lanes": {
                k: {
                    "lane_id": v.lane_id,
                    "label": v.label,
                    "direction": v.direction,
                    "vehicle_count": v.vehicle_count,
                    "avg_speed_kph": round(v.avg_speed_kph, 1),
                    "p85_speed_kph": round(v.p85_speed_kph, 1),
                    "queue_vehicles": v.queue_length_vehicles,
                    "flow_rate_vpm": round(v.flow_rate_vpm, 1),
                    "congestion_ratio": round(v.congestion_ratio, 3),
                    "is_congested": v.is_congested,
                    "polygon": v.polygon,
                }
                for k, v in self.lane_metrics.items()
            },
            "violations": self.violations,
            "vehicles": self.vehicle_positions,
        }


def _polygon_area(polygon: np.ndarray) -> float:
    """Shoelace formula for polygon area in pixels^2."""
    n = len(polygon)
    if n < 3:
        return 0.0
    x = polygon[:, 0]
    y = polygon[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def _bbox_area(bbox: np.ndarray) -> float:
    return float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))


def _point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    """Check if a 2D point is inside a polygon using OpenCV."""
    return cv2.pointPolygonTest(
        polygon.astype(np.float32), (float(point[0]), float(point[1])), False
    ) >= 0


class LaneManager:
    """
    Manages lane definitions and assigns vehicles to lanes.
    Computes per-lane congestion using bbox-area / lane-area ratio.
    """

    def __init__(self, lanes: List[dict], congestion_threshold: float = 0.35):
        self.lanes = lanes
        self.congestion_threshold = congestion_threshold
        self._lane_polygons: Dict[str, np.ndarray] = {}
        self._lane_areas: Dict[str, float] = {}

        for lane in lanes:
            poly = np.array(lane["polygon"], dtype=np.float32)
            self._lane_polygons[lane["id"]] = poly
            self._lane_areas[lane["id"]] = _polygon_area(poly)

    @property
    def has_lanes(self) -> bool:
        return len(self.lanes) > 0

    def assign_vehicle_to_lane(self, center: np.ndarray) -> Optional[str]:
        """Return the lane_id the vehicle center falls in, or None."""
        for lane_id, poly in self._lane_polygons.items():
            if _point_in_polygon(center, poly):
                return lane_id
        return None

    def compute_lane_metrics(
        self,
        vehicle_data: List[dict],
        speed_threshold_kph: float,
    ) -> Dict[str, LaneMetrics]:
        """
        Given per-vehicle data (with lane_id, speed, bbox),
        compute congestion and speed stats per lane.
        """
        lane_vehicles: Dict[str, List[dict]] = defaultdict(list)
        for v in vehicle_data:
            lid = v.get("lane_id")
            if lid:
                lane_vehicles[lid].append(v)

        result: Dict[str, LaneMetrics] = {}
        for lane_def in self.lanes:
            lid = lane_def["id"]
            vehicles = lane_vehicles.get(lid, [])
            speeds = [v["speed_kph"] for v in vehicles]

            total_bbox_area = sum(_bbox_area(np.array(v["bbox"])) for v in vehicles)
            lane_area = self._lane_areas.get(lid, 1.0)
            congestion_ratio = total_bbox_area / lane_area if lane_area > 0 else 0.0

            speed_arr = np.array(speeds) if speeds else np.array([0.0])

            lm = LaneMetrics(
                lane_id=lid,
                label=lane_def.get("label", lid),
                direction=lane_def.get("direction", ""),
                vehicle_count=len(vehicles),
                avg_speed_kph=float(np.mean(speed_arr)) if speeds else 0.0,
                p85_speed_kph=float(np.percentile(speed_arr, 85)) if speeds else 0.0,
                queue_length_vehicles=sum(1 for s in speeds if s < speed_threshold_kph),
                congestion_ratio=congestion_ratio,
                is_congested=congestion_ratio >= self.congestion_threshold,
                polygon=lane_def["polygon"],
            )
            result[lid] = lm

        return result


class TrafficMetricsEngine:
    """
    Computes per-frame and rolling-window traffic metrics,
    including per-lane congestion and speed averages.
    """

    def __init__(
        self,
        homography: HomographyTransform,
        fps: float = 30.0,
        speed_threshold_kph: float = 5.0,
        rolling_window_sec: float = 60.0,
        lane_manager: Optional[LaneManager] = None,
    ):
        self.homography = homography
        self.fps = fps
        self.speed_threshold_kph = speed_threshold_kph
        self.rolling_window_sec = rolling_window_sec
        self.lane_manager = lane_manager

        max_frames = int(fps * rolling_window_sec)
        self.speed_buffer: deque = deque(maxlen=max_frames)
        self.count_buffer: deque = deque(maxlen=max_frames)
        self.timestamp_buffer: deque = deque(maxlen=max_frames)
        self.seen_ids: deque = deque(maxlen=max_frames)

    def compute(
        self,
        tracked_vehicles: List[TrackedVehicle],
        frame_number: int,
    ) -> FrameMetrics:
        timestamp = frame_number / self.fps
        metrics = FrameMetrics(
            frame_number=frame_number,
            timestamp=timestamp,
            total_vehicles=len(tracked_vehicles),
        )

        speeds = []
        queue_count = 0
        class_counts = defaultdict(int)

        for tv in tracked_vehicles:
            class_counts[tv.class_name] += 1
            speed = self._compute_speed(tv)
            speeds.append(speed)

            if speed < self.speed_threshold_kph:
                queue_count += 1

            lane_id = None
            if self.lane_manager and self.lane_manager.has_lanes:
                lane_id = self.lane_manager.assign_vehicle_to_lane(tv.center)

            metrics.vehicle_positions.append({
                "id": tv.track_id,
                "class": tv.class_name,
                "x": float(tv.center[0]),
                "y": float(tv.center[1]),
                "speed_kph": round(speed, 1),
                "bbox": tv.bbox.tolist(),
                "queued": speed < self.speed_threshold_kph,
                "lane_id": lane_id,
            })

        if speeds:
            speed_arr = np.array(speeds)
            metrics.avg_speed_kph = float(np.mean(speed_arr))
            metrics.p85_speed_kph = float(np.percentile(speed_arr, 85))

        metrics.total_queue_vehicles = queue_count
        metrics.total_cars = class_counts.get("car", 0)
        metrics.total_trucks = class_counts.get("truck", 0)
        metrics.total_buses = class_counts.get("bus", 0)
        metrics.total_motorcycles = class_counts.get("motorcycle", 0)

        # Per-lane metrics
        if self.lane_manager and self.lane_manager.has_lanes:
            metrics.lane_metrics = self.lane_manager.compute_lane_metrics(
                metrics.vehicle_positions,
                self.speed_threshold_kph,
            )

        # Rolling buffers
        self.speed_buffer.append(speeds)
        self.count_buffer.append(len(tracked_vehicles))
        self.timestamp_buffer.append(timestamp)
        current_ids = {tv.track_id for tv in tracked_vehicles}
        self.seen_ids.append(current_ids)

        if len(self.timestamp_buffer) >= 2:
            window_duration = self.timestamp_buffer[-1] - self.timestamp_buffer[0]
            if window_duration > 0:
                all_ids = set()
                for ids in self.seen_ids:
                    all_ids.update(ids)
                metrics.flow_rate_vpm = len(all_ids) / (window_duration / 60.0)

        return metrics

    def _compute_speed(self, tv: TrackedVehicle) -> float:
        """
        Compute instantaneous speed in km/h from the last few positions.
        Uses a small window (last 5 positions) for smoothing.
        """
        if len(tv.positions) < 2:
            return 0.0

        window = min(5, len(tv.positions))
        recent_pos = tv.positions[-window:]
        recent_frames = tv.frame_numbers[-window:]

        pos_meters = np.array([
            self.homography.pixel_to_meters(p[0], p[1])
            for p in recent_pos
        ])

        diffs = np.diff(pos_meters, axis=0)
        distances = np.linalg.norm(diffs, axis=1)
        total_distance_m = float(np.sum(distances))

        frame_delta = recent_frames[-1] - recent_frames[0]
        if frame_delta == 0:
            return 0.0
        total_time_s = frame_delta / self.fps

        speed_ms = total_distance_m / total_time_s
        speed_kph = speed_ms * 3.6

        return min(speed_kph, 200.0)


class ViolationDetector:
    """
    Detects simple traffic rule violations based on vehicle tracks.

    Currently supports:
        - Wrong-way driving (vehicle moving against expected direction)
        - Stopped in intersection (vehicle stationary in conflict zone)
    """

    def __init__(
        self,
        expected_direction: Optional[np.ndarray] = None,
        conflict_zone: Optional[np.ndarray] = None,
    ):
        """
        Parameters
        ----------
        expected_direction : np.ndarray, shape (2,)
            Unit vector of the expected traffic flow direction.
        conflict_zone : np.ndarray, shape (N, 2)
            Polygon (in pixels) defining the intersection conflict zone.
        """
        self.expected_direction = expected_direction
        self.conflict_zone = conflict_zone

    def check(self, tracked_vehicles: List[TrackedVehicle], fps: float) -> List[dict]:
        """Check all tracked vehicles for violations."""
        violations = []

        for tv in tracked_vehicles:
            if len(tv.positions) < 3:
                continue

            # Wrong-way detection
            if self.expected_direction is not None:
                movement = np.array(tv.positions[-1]) - np.array(tv.positions[-3])
                if np.linalg.norm(movement) > 5:  # only if actually moving
                    movement_norm = movement / np.linalg.norm(movement)
                    dot = np.dot(movement_norm, self.expected_direction)
                    if dot < -0.5:  # moving roughly opposite to expected
                        violations.append({
                            "type": "wrong_way",
                            "vehicle_id": tv.track_id,
                            "class": tv.class_name,
                            "position": tv.center.tolist(),
                            "severity": "high",
                        })

            # Stopped in conflict zone
            if self.conflict_zone is not None and len(tv.positions) >= 10:
                recent_movement = np.linalg.norm(
                    np.array(tv.positions[-1]) - np.array(tv.positions[-10])
                )
                is_stationary = recent_movement < 5  # pixels
                in_zone = cv2.pointPolygonTest(
                    self.conflict_zone.astype(np.float32),
                    tuple(tv.center.astype(float)),
                    False,
                ) >= 0
                if is_stationary and in_zone:
                    violations.append({
                        "type": "blocked_intersection",
                        "vehicle_id": tv.track_id,
                        "class": tv.class_name,
                        "position": tv.center.tolist(),
                        "severity": "medium",
                    })

        return violations
