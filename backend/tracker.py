"""
SairAI — Tracking Module
=========================
Wraps ByteTrack (via the `supervision` library) to assign persistent IDs
to vehicles across frames.

Detection tells us "there are 12 cars in this frame." But it doesn't
know WHICH car is which from one frame to the next. The tracker
solves this: it gives each car a unique ID (like #47) and follows
that same car across many frames. This is what lets us compute
speed (how far did car #47 move?) and queue length (how many cars
haven't moved?).
"""

from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import supervision as sv

from backend.detection import Detection

logger = logging.getLogger(__name__)


@dataclass
class TrackedVehicle:
    """A vehicle with a persistent ID and its history of positions."""
    track_id: int
    class_name: str
    class_id: int
    bbox: np.ndarray          # latest bounding box [x1, y1, x2, y2]
    center: np.ndarray        # latest center point [cx, cy] in pixels
    confidence: float
    positions: List[np.ndarray]  # history of center positions (pixels)
    frame_numbers: List[int]     # corresponding frame numbers


class VehicleTracker:
    """
    Assigns persistent IDs to detections across frames using
    ByteTrack via the supervision library.
    """

    def __init__(
        self,
        track_activation_threshold: float = 0.20,
        lost_track_buffer: int = 30,
        minimum_matching_threshold: float = 0.5,
        frame_rate: int = 30,
    ):
        self.tracks: Dict[int, TrackedVehicle] = {}
        self.frame_count = 0

        self.byte_tracker = sv.ByteTrack(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
            frame_rate=frame_rate,
        )
        logger.info(
            f"ByteTrack tracker initialised "
            f"(activation={track_activation_threshold}, match={minimum_matching_threshold})"
        )

    def update(self, detections: List[Detection], frame_number: int) -> List[TrackedVehicle]:
        """
        Feed a new frame's detections into the tracker.

        Returns a list of TrackedVehicle objects with persistent IDs.
        """
        self.frame_count = frame_number

        if not detections:
            return []

        bboxes = np.array([d.bbox for d in detections])
        confidences = np.array([d.confidence for d in detections])
        class_ids = np.array([d.class_id for d in detections])

        sv_detections = sv.Detections(
            xyxy=bboxes,
            confidence=confidences,
            class_id=class_ids,
        )

        tracked = self.byte_tracker.update_with_detections(sv_detections)

        results = []
        if tracked.tracker_id is not None:
            from config import VEHICLE_CLASS_IDS

            for i, track_id in enumerate(tracked.tracker_id):
                track_id = int(track_id)
                bbox = tracked.xyxy[i]
                center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
                cls_id = int(tracked.class_id[i]) if tracked.class_id is not None else 2
                conf = float(tracked.confidence[i]) if tracked.confidence is not None else 0.5
                class_name = VEHICLE_CLASS_IDS.get(cls_id, "vehicle")

                if track_id in self.tracks:
                    tv = self.tracks[track_id]
                    tv.bbox = bbox
                    tv.center = center
                    tv.confidence = conf
                    tv.positions.append(center.copy())
                    tv.frame_numbers.append(frame_number)
                else:
                    tv = TrackedVehicle(
                        track_id=track_id,
                        class_name=class_name,
                        class_id=cls_id,
                        bbox=bbox,
                        center=center,
                        confidence=conf,
                        positions=[center.copy()],
                        frame_numbers=[frame_number],
                    )
                    self.tracks[track_id] = tv

                results.append(tv)

        return results
