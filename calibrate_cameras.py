"""
calibrate_cameras.py

Re-run this any time a camera moves. It builds a pixel-to-field homography
for each camera in config.CAMERA_IDS by having you click known points on
the field (e.g. the black-line intersections between FTC foam tiles, or the
field perimeter corners) and typing in their real-world coordinates.

Field coordinate system: origin (0, 0) at one corner of the field, X and Y
in inches, increasing across the field. Pick one corner and stick with it
for every camera -- all three homographies must share the same world frame
or the multi-camera fusion in tracking.py won't line up.

Usage:
    python calibrate_cameras.py

For each camera:
    1. A live preview opens. Press SPACE when the field is clearly visible
       and the camera is in its final mounted position.
    2. Click each reference point on the frozen frame, in any order. After
       each click you'll be prompted in the terminal for that point's real
       world (x, y) in inches.
    3. Click at least 4 points (more is better -- 6-8 spread across the
       frame gives a more accurate homography and lets us report
       reprojection error).
    4. Press 'f' to finish that camera, 'u' to undo the last point, 'r' to
       start over, 'q' to abort.

Results are saved to calibration.json, keyed by camera id, and are picked
up automatically by tracking.py / main.py.

If intrinsics.json has an entry for a camera (from calibrate_intrinsics.py),
frames are undistorted before you click on them, so the homography is built
from the same corrected pixels tracking.py will use at runtime. If a camera
has no intrinsics yet, you'll see a warning and calibration proceeds on raw
(distorted) frames -- run calibrate_intrinsics.py first for best accuracy.
"""
import json
import sys

import cv2
import numpy as np

import config
import camera_utils


class ClickCollector:
    def __init__(self):
        self.pixel_points = []

    def on_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.pixel_points.append((float(x), float(y)))


def freeze_frame(cap, cam_id, undistorter):
    print(f"\n=== Camera {cam_id} === press SPACE to freeze the frame, q to abort")
    while True:
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Camera {cam_id}: failed to read a frame")
        if undistorter is not None:
            frame = undistorter.undistort(frame)
        preview = frame.copy()
        cv2.putText(preview, f"Camera {cam_id} -- SPACE to freeze, q to abort",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("calibrate", preview)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            return frame
        if key == ord("q"):
            return None


def collect_points(frame, cam_id):
    collector = ClickCollector()
    cv2.setMouseCallback("calibrate", collector.on_click)
    world_points = []

    prev_click_count = 0
    while True:
        display = frame.copy()
        for i, (px, py) in enumerate(collector.pixel_points):
            cv2.drawMarker(display, (int(px), int(py)), (0, 0, 255),
                            cv2.MARKER_CROSS, 16, 2)
            cv2.putText(display, str(i), (int(px) + 8, int(py) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        msg = (f"Camera {cam_id}: {len(collector.pixel_points)} point(s) clicked "
               f"(need >=4). f=finish  u=undo  r=reset  q=abort")
        cv2.putText(display, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("calibrate", display)
        key = cv2.waitKey(20) & 0xFF

        # A new click landed since the last loop iteration -- ask for its
        # real-world coordinate in the terminal right away.
        if len(collector.pixel_points) > prev_click_count:
            px, py = collector.pixel_points[-1]
            wx, wy = prompt_world_point(len(collector.pixel_points) - 1, px, py)
            world_points.append((wx, wy))
            prev_click_count = len(collector.pixel_points)

        if key == ord("f"):
            if len(collector.pixel_points) >= 4:
                return collector.pixel_points, world_points
            print("Need at least 4 points before finishing.")
        elif key == ord("u"):
            if collector.pixel_points:
                collector.pixel_points.pop()
                world_points.pop()
                prev_click_count = len(collector.pixel_points)
        elif key == ord("r"):
            collector.pixel_points.clear()
            world_points.clear()
            prev_click_count = 0
        elif key == ord("q"):
            return None, None


def prompt_world_point(index, px, py):
    print(f"  Point {index} clicked at pixel ({px:.0f}, {py:.0f}).")
    while True:
        raw = input(f"    Enter real-world field coords for point {index} as 'x,y' (inches): ").strip()
        try:
            wx_str, wy_str = raw.split(",")
            return float(wx_str.strip()), float(wy_str.strip())
        except ValueError:
            print("    Couldn't parse that -- expected format: 24,48")


def compute_homography(pixel_points, world_points):
    src = np.array(pixel_points, dtype=np.float64)
    dst = np.array(world_points, dtype=np.float64)

    if len(src) == 4:
        H, _ = cv2.findHomography(src, dst, method=0)
    else:
        H, _ = cv2.findHomography(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)

    # Reprojection error, in inches, so you know how trustworthy this
    # calibration is. A few tenths of an inch is good; several inches means
    # re-click more carefully or add more points spread across the frame.
    projected = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
    errors = np.linalg.norm(projected - dst, axis=1)
    print(f"  Reprojection error: mean={errors.mean():.3f}in  max={errors.max():.3f}in")

    return H


def load_calibration():
    try:
        with open(config.CALIBRATION_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def save_calibration(data):
    with open(config.CALIBRATION_FILE, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nSaved calibration to {config.CALIBRATION_FILE}")


def calibrate_one_camera(cam_id):
    backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else 0
    cap = cv2.VideoCapture(cam_id, backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    if not cap.isOpened():
        print(f"Could not open camera {cam_id}, skipping.")
        cap.release()
        return None

    resolution = (config.FRAME_WIDTH, config.FRAME_HEIGHT)
    undistorter = camera_utils.build_undistorters({cam_id: resolution})[cam_id]
    if undistorter is None:
        print(f"  No intrinsic calibration found for camera {cam_id} -- proceeding without "
              f"lens undistortion (run calibrate_intrinsics.py first for best accuracy).")

    result = None
    frame = freeze_frame(cap, cam_id, undistorter)
    if frame is not None:
        pixel_points, world_points = collect_points(frame, cam_id)
        if pixel_points:
            H = compute_homography(pixel_points, world_points)
            h, w = frame.shape[:2]
            result = {
                "resolution": [w, h],
                "homography": H.tolist(),
                "points_px": pixel_points,
                "points_world": world_points,
            }

    cap.release()
    return result


def main():
    calibration = load_calibration()
    for cam_id in config.CAMERA_IDS:
        entry = calibrate_one_camera(cam_id)
        if entry is not None:
            calibration[str(cam_id)] = entry
        else:
            print(f"Camera {cam_id} not calibrated this run (kept previous value if any).")
    cv2.destroyAllWindows()
    save_calibration(calibration)


if __name__ == "__main__":
    main()
