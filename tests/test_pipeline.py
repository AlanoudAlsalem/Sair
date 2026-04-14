"""
SairAI — Basic Tests
=====================
Run with: python -m pytest tests/ -v
"""

import sys
from pathlib import Path

import numpy as np

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_homography_calibrate_from_scale():
    """Test that scale-based calibration works."""
    from backend.homography import HomographyTransform

    h = HomographyTransform()
    h.calibrate_from_scale(pixels_per_meter=10.0)

    assert h.is_calibrated
    x, y = h.pixel_to_meters(100, 200)
    assert abs(x - 10.0) < 0.01
    assert abs(y - 20.0) < 0.01


def test_homography_full_calibration():
    """Test full 4-point homography calibration."""
    from backend.homography import HomographyTransform

    points = [
        (0, 0, 0.0, 0.0),
        (100, 0, 10.0, 0.0),
        (100, 100, 10.0, 10.0),
        (0, 100, 0.0, 10.0),
    ]
    h = HomographyTransform(points)

    assert h.is_calibrated
    x, y = h.pixel_to_meters(50, 50)
    assert abs(x - 5.0) < 0.1
    assert abs(y - 5.0) < 0.1


def test_detection_mock():
    """Test that mock detection produces valid results."""
    from backend.detection import VehicleDetector

    detector = VehicleDetector()  # will use mock since ultralytics likely not installed
    fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = detector.detect(fake_frame)

    assert isinstance(detections, list)
    if detections:  # mock always produces some
        d = detections[0]
        assert hasattr(d, "bbox")
        assert hasattr(d, "confidence")
        assert hasattr(d, "class_name")
        assert len(d.bbox) == 4


def test_tracker_simple_iou():
    """Test the simple IoU fallback tracker."""
    from backend.detection import Detection
    from backend.tracker import VehicleTracker

    tracker = VehicleTracker()

    # Frame 1: one car
    dets1 = [Detection(
        bbox=np.array([100, 100, 200, 200], dtype=float),
        confidence=0.9,
        class_id=2,
        class_name="car",
    )]
    results1 = tracker.update(dets1, frame_number=1)
    assert len(results1) == 1
    first_id = results1[0].track_id

    # Frame 2: same car, slightly moved
    dets2 = [Detection(
        bbox=np.array([105, 102, 205, 202], dtype=float),
        confidence=0.9,
        class_id=2,
        class_name="car",
    )]
    results2 = tracker.update(dets2, frame_number=2)
    assert len(results2) == 1
    # Should keep the same ID
    assert results2[0].track_id == first_id


def test_metrics_computation():
    """Test that metrics engine produces valid output."""
    from backend.homography import HomographyTransform
    from backend.metrics import TrafficMetricsEngine
    from backend.tracker import TrackedVehicle

    h = HomographyTransform()
    h.calibrate_from_scale(10.0)

    engine = TrafficMetricsEngine(homography=h, fps=30.0)

    vehicles = [
        TrackedVehicle(
            track_id=1,
            class_name="car",
            class_id=2,
            bbox=np.array([100, 100, 200, 200]),
            center=np.array([150, 150]),
            confidence=0.9,
            positions=[np.array([140, 140]), np.array([150, 150])],
            frame_numbers=[1, 2],
        ),
    ]

    metrics = engine.compute(vehicles, frame_number=2)

    assert metrics.total_vehicles == 1
    assert metrics.total_cars == 1
    assert metrics.frame_number == 2
    assert isinstance(metrics.to_dict(), dict)


def test_config_imports():
    """Test that config.py loads without errors."""
    import config
    assert hasattr(config, "YOLO_MODEL")
    assert hasattr(config, "VEHICLE_CLASS_IDS")
    assert isinstance(config.VEHICLE_CLASS_IDS, dict)
