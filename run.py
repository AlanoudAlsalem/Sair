#!/usr/bin/env python3
"""
SairAI — Main Entry Point
===========================
Starts the FastAPI server + React dashboard.

Usage:
    python run.py                    # Start on default port 8000
    python run.py --port 3000        # Custom port
    python run.py --host 127.0.0.1   # Localhost only
"""

import argparse
import sys
import os
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    parser = argparse.ArgumentParser(description="Start SairAI dashboard server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    parser.add_argument("--reload", action="store_true", help="Auto-reload on code changes")
    args = parser.parse_args()

    # Check dependencies
    missing = []
    for pkg in ["fastapi", "uvicorn", "cv2", "numpy"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        print("=" * 60)
        print("MISSING DEPENDENCIES")
        print("=" * 60)
        print()
        print("The following packages need to be installed:")
        for m in missing:
            print(f"  - {m}")
        print()
        print("Quick fix:")
        print("  pip install -r requirements.txt")
        print()
        print("Or install individually:")
        print("  pip install fastapi uvicorn ultralytics supervision opencv-python numpy pandas aiofiles")
        print("=" * 60)
        sys.exit(1)

    # Create data directories if they don't exist
    (Path(__file__).parent / "data" / "videos").mkdir(parents=True, exist_ok=True)
    (Path(__file__).parent / "data" / "output").mkdir(parents=True, exist_ok=True)

    print()
    print("  ╔══════════════════════════════════════════╗")
    print("  ║         SairAI Traffic Intelligence       ║")
    print("  ║   Real-Time Drone Monitoring for Amman    ║")
    print("  ╚══════════════════════════════════════════╝")
    print()
    print(f"  Dashboard:  http://localhost:{args.port}")
    print(f"  API docs:   http://localhost:{args.port}/docs")
    print()
    print("  Put your drone videos in: data/videos/")
    print("  Press Ctrl+C to stop the server.")
    print()

    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
