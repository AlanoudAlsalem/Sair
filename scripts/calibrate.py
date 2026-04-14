#!/usr/bin/env python3
"""
SairAI — Homography Calibration Helper
========================================
Interactive tool to select reference points on a video frame and
compute the pixel-to-meter calibration.

Usage:
    python scripts/calibrate.py data/videos/my_clip.mp4

What it does:
    1. Opens the first frame of your drone video
    2. You click on 4+ points whose real-world positions you know
       (e.g., lane markings, crosswalk edges, known distances)
    3. You type in the real-world meter coordinates for each point
    4. It computes the homography and prints the values to paste
       into config.py

NOTE: This requires a display (won't work in headless/SSH mode).
      In headless mode, measure points manually and edit config.py.
"""

from __future__ import annotations
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.homography import HomographyTransform


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/calibrate.py <video_path>")
        sys.exit(1)

    video_path = sys.argv[1]
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"Error: Cannot read video: {video_path}")
        sys.exit(1)

    print("=" * 60)
    print("SairAI Homography Calibration")
    print("=" * 60)
    print()
    print("You'll need at least 4 reference points where you know both:")
    print("  - The pixel coordinates (from the image)")
    print("  - The real-world coordinates in meters")
    print()
    print("Good reference points to use:")
    print("  - Lane markings (standard lane width = 3.65 m in Jordan)")
    print("  - Crosswalk stripes (standard width = 3.0 m)")
    print("  - Known building distances from satellite images")
    print()

    # Try to open a window for clicking
    try:
        points_px = []

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                points_px.append((x, y))
                cv2.circle(display, (x, y), 5, (0, 255, 0), -1)
                cv2.putText(
                    display, f"P{len(points_px)}: ({x},{y})",
                    (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                )
                cv2.imshow("Calibration - Click 4+ points, then press 'q'", display)
                print(f"  Point {len(points_px)}: pixel ({x}, {y})")

        display = frame.copy()
        cv2.namedWindow("Calibration - Click 4+ points, then press 'q'", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Calibration - Click 4+ points, then press 'q'", mouse_callback)
        cv2.imshow("Calibration - Click 4+ points, then press 'q'", display)

        print("Click on reference points in the image. Press 'q' when done.")
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") and len(points_px) >= 4:
                break
        cv2.destroyAllWindows()

    except cv2.error:
        # Headless mode — manual entry
        print("No display available. Entering manual mode.")
        print("Enter pixel coordinates for each point (x y):")
        points_px = []
        for i in range(4):
            while True:
                try:
                    coords = input(f"  Point {i+1} pixel (x y): ").split()
                    points_px.append((int(coords[0]), int(coords[1])))
                    break
                except (ValueError, IndexError):
                    print("  Please enter two integers: x y")

    # Now get real-world coordinates
    print()
    print("Now enter real-world coordinates (in meters) for each point:")
    reference_points = []
    for i, (px, py) in enumerate(points_px):
        while True:
            try:
                coords = input(f"  Point {i+1} at pixel ({px},{py}) — real-world (x_m y_m): ").split()
                rx, ry = float(coords[0]), float(coords[1])
                reference_points.append((px, py, rx, ry))
                break
            except (ValueError, IndexError):
                print("  Please enter two numbers: x_meters y_meters")

    # Compute homography
    h = HomographyTransform(reference_points)

    if h.is_calibrated:
        print()
        print("=" * 60)
        print("CALIBRATION SUCCESSFUL!")
        print("=" * 60)
        print()
        print("Paste this into config.py:")
        print()
        print("HOMOGRAPHY_REFERENCE_POINTS = [")
        for p in reference_points:
            print(f"    ({p[0]}, {p[1]}, {p[2]}, {p[3]}),")
        print("]")
        print()
        print(f"Approximate scale: {h.pixels_per_meter:.1f} pixels per meter")

        # Sanity check: measure a known distance
        if len(reference_points) >= 2:
            p1 = reference_points[0]
            p2 = reference_points[1]
            m1 = h.pixel_to_meters(p1[0], p1[1])
            m2 = h.pixel_to_meters(p2[0], p2[1])
            dist = np.sqrt((m2[0]-m1[0])**2 + (m2[1]-m1[1])**2)
            expected = np.sqrt((p2[2]-p1[2])**2 + (p2[3]-p1[3])**2)
            print(f"Sanity check: P1→P2 distance = {dist:.2f} m (expected: {expected:.2f} m)")
    else:
        print("CALIBRATION FAILED. Check your reference points.")
        sys.exit(1)


if __name__ == "__main__":
    main()
