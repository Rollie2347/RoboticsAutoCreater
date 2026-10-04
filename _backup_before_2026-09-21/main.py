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
"""
import sys
import time

import cv2

import config
import tracking


def open_cameras():
    backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else 0
    caps = {}
    for cam_id in config.CAMERA_IDS:
        cap = cv2.VideoCapture(cam_id, backend)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, config.FRAME_FPS)
        if not cap.isOpened():
            for c in caps.values():
                c.release()
            raise RuntimeError(f"Could not open camera {cam_id}. Check config.CAMERA_IDS "
                                f"(run list_cameras.py) and that no other app is using it.")
        caps[cam_id] = cap
    return caps


def main():
    try:
        tracker = tracking.MultiCameraTracker()
    except RuntimeError as e:
        print(f"Setup incomplete: {e}")
        return

    caps = open_cameras()
    smoother = tracking.ObjectSmoother()
    last_time = time.time()
    print("Tracking live. Press q to quit.")

    try:
        while True:
            now = time.time()
            dt = now - last_time
            last_time = now

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

            fused_robot = tracking.fuse_robot(frames)
            fused_balls = tracking.fuse_balls(frames)

            if config.ENABLE_SMOOTHING:
                fused_robot, fused_balls = smoother.update(fused_robot, fused_balls, dt)

            field_map = tracking.draw_field_map(fused_robot, fused_balls)
            cv2.imshow("field map (top-down)", field_map)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
    finally:
        for cap in caps.values():
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
