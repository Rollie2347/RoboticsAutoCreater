"""
main.py

Live tracker entry point. Opens every camera in config.CAMERA_IDS, detects
the robot, its heading markers and each game-element class, and shows:
    - each camera's feed with detection boxes and a heading arrow
    - a top-down field map with positions, heading, per-class counts and
      robot-to-element distances

Prerequisites (run once, and again whenever a camera moves or lighting
changes):
    python list_cameras.py         # figure out CAMERA_IDS
    python calibrate_intrinsics.py # build intrinsics.json (recommended)
    python calibrate_cameras.py    # build calibration.json
    python tune_colors.py --label robot
    python tune_colors.py --label robot_front     # optional, enables heading
    python tune_colors.py --label robot_back      # optional, enables heading
    python tune_colors.py --label pollen
    python tune_colors.py --label nectar_red
    python tune_colors.py --label nectar_blue

The top-down map's positions are smoothed frame-to-frame with a Kalman
filter (see ObjectSmoother in tracking.py) to cut jitter; set
config.ENABLE_SMOOTHING = False to see raw fused positions instead. A
point still coasting on its motion model (no camera saw it that frame) is
drawn as a hollow circle labeled "(predicted)".

Controls:
    q / ESC = quit

Logging (for the frame-rate check and for comparing against Pinpoint):
    python main.py --log run1.csv

writes one row per processed frame -- wall-clock and monotonic
timestamps, frame interval, raw and smoothed robot pose, and element
counts -- and prints a frame-rate summary when you quit. The live FPS is
also shown on the field map.
"""
import argparse
import csv
import sys
import time
from collections import deque

import cv2
import numpy as np

import config
import tracking


def open_cameras():
    backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else 0
    caps = {}
    for cam_id in config.CAMERA_IDS:
        cap = cv2.VideoCapture(cam_id, backend)
        # Ask for compressed MJPG *before* setting the size. Uncompressed
        # YUY2 at 1280x720 needs ~55 MB/s at 30 fps -- more than USB 2 carries
        # -- so most webcams quietly drop to ~10 fps in that mode.
        if config.CAPTURE_FOURCC:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*config.CAPTURE_FOURCC))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, config.FRAME_FPS)
        if not cap.isOpened():
            for c in caps.values():
                c.release()
            raise RuntimeError(f"Could not open camera {cam_id}. Check config.CAMERA_IDS "
                                f"(run list_cameras.py) and that no other app is using it.")
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        fourcc_str = "".join(chr((fourcc >> (8 * i)) & 0xFF) for i in range(4)) if fourcc else "?"
        print(f"Camera {cam_id}: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
              f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} @ {cap.get(cv2.CAP_PROP_FPS):.0f} fps "
              f"requested, format {fourcc_str!r} (the log shows what you actually get)")
        caps[cam_id] = cap
    return caps


def _pose_fields(point):
    if point is None:
        return ["", "", ""]
    h = "" if point.heading_deg is None else f"{point.heading_deg:.2f}"
    return [f"{point.position_in[0]:.2f}", f"{point.position_in[1]:.2f}", h]


def _summarize(intervals_s):
    if len(intervals_s) < 2:
        print("Not enough frames for a frame-rate summary.")
        return
    dt = np.array(intervals_s)
    below = 1.0 / config.MIN_TARGET_FPS
    print("\nFrame-rate summary")
    print(f"  frames: {len(dt) + 1}   mean: {1 / dt.mean():.1f} fps   median: {1 / np.median(dt):.1f} fps")
    print(f"  slowest interval: {dt.max() * 1000:.0f} ms   95th percentile: {np.percentile(dt, 95) * 1000:.0f} ms")
    print(f"  intervals slower than {config.MIN_TARGET_FPS:.0f} fps ({below * 1000:.0f} ms): "
          f"{int((dt > below).sum())} of {len(dt)} ({100 * (dt > below).mean():.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Live robot / game element tracker")
    parser.add_argument("--log", metavar="CSV", help="write one row per frame to this CSV file")
    args = parser.parse_args()

    try:
        tracker = tracking.MultiCameraTracker()
    except RuntimeError as e:
        print(f"Setup incomplete: {e}")
        return

    caps = open_cameras()
    smoother = tracking.ObjectSmoother()

    log_file = writer = None
    if args.log:
        log_file = open(args.log, "w", newline="")
        writer = csv.writer(log_file)
        writer.writerow(["frame", "wall_time_s", "mono_time_s", "dt_ms",
                         "raw_x_in", "raw_y_in", "raw_heading_deg",
                         "x_in", "y_in", "heading_deg", "coasting"]
                        + [f"count_{c}" for c in config.BALL_CLASSES])
        print(f"Logging every frame to {args.log}")

    intervals = []           # seconds between processed frames, for the summary
    recent = deque(maxlen=30)
    last_mono = None
    frame_no = 0
    print("Tracking live. Press q to quit.")

    try:
        while True:
            frames = []
            for cam_id in config.CAMERA_IDS:
                ok, frame = caps[cam_id].read()
                if not ok:
                    print(f"Warning: camera {cam_id} dropped a frame")
                    frames.append(None)
                    continue
                result = tracker.process_frame(cam_id, frame)
                frames.append(result)
                cv2.imshow(f"camera {cam_id}", result.annotated)

            # Timestamp right after capture + detection, so dt is the real
            # interval between frames the smoother sees.
            wall = time.time()
            mono = time.perf_counter()
            dt = (mono - last_mono) if last_mono is not None else 1.0 / config.FRAME_FPS
            if last_mono is not None:
                intervals.append(dt)
                recent.append(dt)
            last_mono = mono

            fused_robot = tracking.fuse_robot(frames)
            fused_balls = tracking.fuse_balls(frames)
            raw_fields = _pose_fields(fused_robot)  # before smoothing touches it

            if config.ENABLE_SMOOTHING:
                fused_robot, fused_balls = smoother.update(fused_robot, fused_balls, dt)

            field_map = tracking.draw_field_map(fused_robot, fused_balls)
            if recent:
                fps = 1.0 / (sum(recent) / len(recent))
                color = (255, 255, 255) if fps >= config.MIN_TARGET_FPS else (0, 0, 255)
                cv2.putText(field_map, f"{fps:.1f} fps", (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, color, 1, cv2.LINE_AA)
            cv2.imshow("field map (top-down)", field_map)

            if writer is not None:
                counts = tracking.count_by_class(fused_balls)
                coasting = "" if fused_robot is None else int(not fused_robot.camera_ids)
                writer.writerow([frame_no, f"{wall:.4f}", f"{mono:.4f}", f"{dt * 1000:.1f}"]
                                + raw_fields + _pose_fields(fused_robot) + [coasting]
                                + [counts.get(c, 0) for c in config.BALL_CLASSES])
            frame_no += 1

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
    finally:
        for cap in caps.values():
            cap.release()
        cv2.destroyAllWindows()
        if log_file is not None:
            log_file.close()
        _summarize(intervals)


if __name__ == "__main__":
    main()
