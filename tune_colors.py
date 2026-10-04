"""
tune_colors.py

Interactive HSV threshold tuner. Point one of your webcams at the field
with the robot (and a ball) actually in frame, then drag the trackbars
until the mask window cleanly isolates just that object. Save when happy.

The thresholds you save here are shared across all cameras (one lighting
setup, one field) -- re-run this if the room lighting changes significantly
or a robot marker/game piece color changes.

Run it once per label. Valid labels come from config.COLOR_LABELS:

    python tune_colors.py --label robot          # required (robot position)
    python tune_colors.py --label robot_front    # optional (heading)
    python tune_colors.py --label robot_back     # optional (heading)
    python tune_colors.py --label pollen         # yellow, 2.8 in
    python tune_colors.py --label nectar_red     # red, 3.6 in
    python tune_colors.py --label nectar_blue    # blue, 3.6 in

Heading needs BOTH robot_front and robot_back. Ball classes are independent
-- any you haven't tuned yet are just skipped by the tracker.

Add --camera N to preview a camera other than the first in CAMERA_IDS.

Controls:
    s = save current thresholds for this label to hsv_config.json
    q = quit without saving further changes
"""
import argparse
import json
import sys

import cv2
import numpy as np

import config

TRACKBAR_WINDOW = "HSV tuner"


def nothing(_):
    pass

"""
Red: near 0 or near 179 (it wraps around both ends)
Orange: ~10–20
Yellow: ~25–35
Green: ~40–80
Cyan: ~85–100
Blue: ~100–130
Purple/Magenta: ~140–160
"""
##run python tune_colors.py --label ball --camera 0/1
##python tune_colors.py --label robot --camera 0/1
def build_trackbars(initial):
    cv2.namedWindow(TRACKBAR_WINDOW)
    cv2.createTrackbar("H min", TRACKBAR_WINDOW, initial["h_min"], 179, nothing)
    cv2.createTrackbar("H max", TRACKBAR_WINDOW, initial["h_max"], 179, nothing)
    cv2.createTrackbar("S min", TRACKBAR_WINDOW, initial["s_min"], 255, nothing)
    cv2.createTrackbar("S max", TRACKBAR_WINDOW, initial["s_max"], 255, nothing)
    cv2.createTrackbar("V min", TRACKBAR_WINDOW, initial["v_min"], 255, nothing)
    cv2.createTrackbar("V max", TRACKBAR_WINDOW, initial["v_max"], 255, nothing)


def read_trackbars():
    return {
        "h_min": cv2.getTrackbarPos("H min", TRACKBAR_WINDOW),
        "h_max": cv2.getTrackbarPos("H max", TRACKBAR_WINDOW),
        "s_min": cv2.getTrackbarPos("S min", TRACKBAR_WINDOW),
        "s_max": cv2.getTrackbarPos("S max", TRACKBAR_WINDOW),
        "v_min": cv2.getTrackbarPos("V min", TRACKBAR_WINDOW),
        "v_max": cv2.getTrackbarPos("V max", TRACKBAR_WINDOW),
    }


def load_hsv_config():
    try:
        with open(config.HSV_CONFIG_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def save_hsv_config(data):
    with open(config.HSV_CONFIG_FILE, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved to {config.HSV_CONFIG_FILE}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True, choices=config.COLOR_LABELS,
                         help="which object you're tuning a color for")
    parser.add_argument("--camera", type=int, default=config.CAMERA_IDS[0],
                         help="camera index to preview while tuning")
    args = parser.parse_args()

    all_hsv = load_hsv_config()
    initial = all_hsv.get(args.label, {
        "h_min": 0, "h_max": 179, "s_min": 80, "s_max": 255, "v_min": 80, "v_max": 255,
    })

    backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else 0
    cap = cv2.VideoCapture(args.camera, backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    if not cap.isOpened():
        print(f"Could not open camera {args.camera}")
        return

    build_trackbars(initial)
    print(f"Tuning label '{args.label}' on camera {args.camera}. Press 's' to save, 'q' to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame")
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        vals = read_trackbars()
        lower = np.array([vals["h_min"], vals["s_min"], vals["v_min"]])
        upper = np.array([vals["h_max"], vals["s_max"], vals["v_max"]])
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        result = cv2.bitwise_and(frame, frame, mask=mask)

        cv2.imshow("frame", frame)
        cv2.imshow("mask", mask)
        cv2.imshow("result", result)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            all_hsv[args.label] = vals
            save_hsv_config(all_hsv)
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
