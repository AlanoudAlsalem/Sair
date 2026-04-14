"""
SairAI — Detection Module
==========================
Wraps YOLOv8 inference and filters for vehicle classes only.

Uses SAHI (Slicing Aided Hyper Inference) for drone footage where
vehicles — especially motorcycles — can be very small. SAHI slices
the frame into overlapping tiles, runs YOLO on each tile, then merges
the results. This dramatically improves recall for small objects.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """One detected vehicle in a single frame."""
    bbox: np.ndarray          # [x1, y1, x2, y2] in pixels
    confidence: float         # 0.0 – 1.0
    class_id: int             # COCO class ID
    class_name: str           # human-readable label
    center: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))

    def __post_init__(self):
        x1, y1, x2, y2 = self.bbox
        self.center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])


class VehicleDetector:
    """
    Runs YOLOv8 on a frame with optional SAHI sliced inference and
    returns a list of Detection objects for vehicles only.

    SAHI slices large frames into overlapping tiles so small vehicles
    (motorcycles, distant cars) are detected at a usable resolution.
    """

    def __init__(
        self,
        model_path: str = "yolov8x.pt",
        confidence: float = 0.15,
        iou_threshold: float = 0.45,
        device: str = "",
        imgsz: int = 1280,
        vehicle_class_ids: Optional[dict] = None,
        use_sahi: bool = True,
        sahi_slice_size: int = 640,
        sahi_overlap_ratio: float = 0.25,
    ):
        from config import VEHICLE_CLASS_IDS
        self.vehicle_class_ids = vehicle_class_ids or VEHICLE_CLASS_IDS
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.device = device
        self.imgsz = imgsz
        self.use_sahi = use_sahi
        self.sahi_slice_size = sahi_slice_size
        self.sahi_overlap_ratio = sahi_overlap_ratio

        logger.info(f"Loading YOLO model: {model_path} (imgsz={imgsz}, conf={confidence}, sahi={use_sahi})")
        self.model = YOLO(model_path)

        self._sahi_model = None
        if self.use_sahi:
            try:
                from sahi import AutoDetectionModel
                self._sahi_model = AutoDetectionModel.from_pretrained(
                    model_type="yolov8",
                    model_path=model_path,
                    confidence_threshold=confidence,
                    device=device or "cpu",
                )
                logger.info(f"SAHI model loaded (slice={sahi_slice_size}, overlap={sahi_overlap_ratio})")
            except Exception as e:
                logger.warning(f"SAHI init failed, falling back to standard inference: {e}")
                self.use_sahi = False

        logger.info("Detection model loaded successfully.")

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run detection on a single BGR frame.
        Uses SAHI sliced inference when enabled, otherwise standard YOLO.
        """
        if self.use_sahi and self._sahi_model is not None:
            return self._detect_sahi(frame)
        return self._detect_standard(frame)

    def _detect_sahi(self, frame: np.ndarray) -> List[Detection]:
        """SAHI sliced inference — much better recall for small objects."""
        from sahi.predict import get_sliced_prediction

        result = get_sliced_prediction(
            image=frame,
            detection_model=self._sahi_model,
            slice_height=self.sahi_slice_size,
            slice_width=self.sahi_slice_size,
            overlap_height_ratio=self.sahi_overlap_ratio,
            overlap_width_ratio=self.sahi_overlap_ratio,
            perform_standard_pred=True,
            postprocess_type="NMS",
            postprocess_match_metric="IOS",
            postprocess_match_threshold=self.iou_threshold,
            verbose=0,
        )

        detections = []
        for pred in result.object_prediction_list:
            cls_id = pred.category.id
            if cls_id not in self.vehicle_class_ids:
                continue

            bb = pred.bbox
            bbox = np.array([bb.minx, bb.miny, bb.maxx, bb.maxy], dtype=np.float32)
            conf = pred.score.value
            label = self.vehicle_class_ids[cls_id]

            detections.append(Detection(
                bbox=bbox,
                confidence=conf,
                class_id=cls_id,
                class_name=label,
            ))

        return detections

    def _detect_standard(self, frame: np.ndarray) -> List[Detection]:
        """Standard single-pass YOLO inference."""
        results = self.model(
            frame,
            conf=self.confidence,
            iou=self.iou_threshold,
            device=self.device,
            imgsz=self.imgsz,
            verbose=False,
        )[0]

        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in self.vehicle_class_ids:
                continue

            bbox = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            label = self.vehicle_class_ids[cls_id]

            detections.append(Detection(
                bbox=bbox,
                confidence=conf,
                class_id=cls_id,
                class_name=label,
            ))

        return detections
