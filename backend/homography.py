"""
SairAI — Homography Module
============================
Converts pixel coordinates (from the camera image) into real-world
meter coordinates on the ground.

What this does in plain English:
    The drone camera gives us everything in PIXELS — "the car is at
    pixel (320, 240)." But we need METERS — "the car is 15.2 m from
    the intersection centre." A homography is a math trick that converts
    between the two. You give it 4 known reference points (e.g., "this
    lane marking is at pixel (200,150) and it's at 0m, 0m in real life")
    and it figures out the transformation for every other pixel.

    Because the drone is looking nearly straight down, this transformation
    is extremely accurate — much better than a side-mounted fixed camera
    where perspective distortion is severe.
"""

from __future__ import annotations
import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class HomographyTransform:
    """
    Computes and applies a pixel → real-world-meters homography.

    Usage:
        1. Provide 4+ reference point pairs (pixel coords → meter coords).
        2. Call transform() on any pixel coordinate to get meters.
    """

    def __init__(
        self,
        reference_points: Optional[List[Tuple[float, float, float, float]]] = None,
    ):
        """
        Parameters
        ----------
        reference_points : list of (px_x, px_y, real_x, real_y)
            At least 4 known point correspondences.
        """
        self.H: Optional[np.ndarray] = None  # 3x3 homography matrix
        self.H_inv: Optional[np.ndarray] = None  # inverse (meters → pixels)
        self.is_calibrated = False
        self.pixels_per_meter: float = 1.0  # fallback scale factor

        if reference_points and len(reference_points) >= 4:
            self.calibrate(reference_points)

    def calibrate(self, reference_points: List[Tuple[float, float, float, float]]):
        """
        Compute the homography matrix from reference points.

        Each point is (pixel_x, pixel_y, real_x_meters, real_y_meters).
        """
        pixel_pts = np.array(
            [[p[0], p[1]] for p in reference_points], dtype=np.float32
        )
        real_pts = np.array(
            [[p[2], p[3]] for p in reference_points], dtype=np.float32
        )

        self.H, status = cv2.findHomography(pixel_pts, real_pts, cv2.RANSAC, 5.0)

        if self.H is not None:
            self.H_inv = np.linalg.inv(self.H)
            self.is_calibrated = True

            # Compute approximate pixels-per-meter for the centre of the image
            dx_px = np.linalg.norm(pixel_pts[1] - pixel_pts[0])
            dx_m = np.linalg.norm(real_pts[1] - real_pts[0])
            self.pixels_per_meter = dx_px / dx_m if dx_m > 0 else 1.0

            logger.info(
                f"Homography calibrated. ~{self.pixels_per_meter:.1f} px/m. "
                f"Status: {status.ravel().tolist()}"
            )
        else:
            logger.error("Homography computation FAILED. Check your reference points.")

    def calibrate_from_scale(self, pixels_per_meter: float):
        """
        Quick calibration when you don't have 4 reference points but
        you know the approximate scale (e.g., from measuring a lane
        width in pixels and knowing it's 3.65 m).
        """
        self.pixels_per_meter = pixels_per_meter
        # Build a simple scaling matrix (no rotation/perspective)
        s = 1.0 / pixels_per_meter
        self.H = np.array([
            [s, 0, 0],
            [0, s, 0],
            [0, 0, 1],
        ], dtype=np.float64)
        self.H_inv = np.linalg.inv(self.H)
        self.is_calibrated = True
        logger.info(f"Scale-based calibration: {pixels_per_meter:.1f} px/m")

    def pixel_to_meters(self, px_x: float, px_y: float) -> Tuple[float, float]:
        """Convert a single pixel coordinate to real-world meters."""
        if not self.is_calibrated:
            # Fallback: simple scale
            return (px_x / self.pixels_per_meter, px_y / self.pixels_per_meter)

        pt = np.array([px_x, px_y, 1.0])
        transformed = self.H @ pt
        transformed /= transformed[2]  # normalise homogeneous coords
        return (float(transformed[0]), float(transformed[1]))

    def meters_to_pixel(self, real_x: float, real_y: float) -> Tuple[float, float]:
        """Convert real-world meters back to pixel coordinates."""
        if self.H_inv is None:
            return (real_x * self.pixels_per_meter, real_y * self.pixels_per_meter)

        pt = np.array([real_x, real_y, 1.0])
        transformed = self.H_inv @ pt
        transformed /= transformed[2]
        return (float(transformed[0]), float(transformed[1]))

    def pixel_distance_to_meters(self, px_dist: float) -> float:
        """
        Approximate conversion of a pixel distance to meters.
        Only accurate near the image centre for perspective transforms.
        """
        return px_dist / self.pixels_per_meter

    def batch_pixel_to_meters(self, points: np.ndarray) -> np.ndarray:
        """
        Convert an Nx2 array of pixel coordinates to meters.
        Returns Nx2 array in meters.
        """
        if not self.is_calibrated or self.H is None:
            return points / self.pixels_per_meter

        n = len(points)
        homogeneous = np.hstack([points, np.ones((n, 1))])
        transformed = (self.H @ homogeneous.T).T
        transformed[:, 0] /= transformed[:, 2]
        transformed[:, 1] /= transformed[:, 2]
        return transformed[:, :2]
