"""
calibrate_intrinsics.py

Lens (intrinsic) calibration -- run this BEFORE calibrate_cameras.py, and
again if you ever swap a camera or change its zoom/focus. It measures each
camera's actual lens distortion using a printed checkerboard, so every
downstream frame (both during homography calibration and live tracking)
can be undistorted before any field-position math happens.

You need a printed checkerboard pattern -- any standard chessboard
calibration target works (search "opencv checkerboard calibration pattern"
for a printable PDF). After printing:
    1. Measure one square's actual edge length with a ruler.
    2. Set config.CHECKERBOARD_SQUARE_SIZE_IN to that measured value.
    3. Set config.CHECKERBOARD_INNER_CORNERS to (columns, rows) of INTERNAL
       corners (where 4 squares meet) -- not the number of squares.

Usage:
    python calibrate_intrinsics.py

For each camera:
    1. A live preview opens. Detected checkerboard corners are drawn in
       real time when the board is visible and fully in frame.
    2. Press 'c' to capture the current frame as a calibration sample
       (only works when corners are currently detected). Move/tilt/rotate
       the board between captures -- vary distance and angle, and cover
       different regions of the frame (especially corners/edges), so the
       distortion model is well constrained everywhere, not just where you
       held the board first.
    3. Capture at least config.MIN_INTRINSIC_CAPTURES good samples (15 by
       default). Press 'f' to finish once you have enough -- it'll refuse
       and tell you how many more it needs if you try too early.
    4. Press 'q' to abort a camera without saving.

Results (camera matrix + distortion coefficients) are saved to
intrinsics.json, keyed by camera id, and picked up automatically by
calibrate_cameras.py and tracking.py.

IMPORTANT: because undistortion shifts pixel coordinates, re-run
calibrate_cameras.py (the homography step) after calibrating or changing
a camera's intrinsics here -- an old homography built from un-undistorted
pixels will be wrong once undistortion is applied on top of it.
"""
import sys

import cv2
import numpy as np

import config
import camera_utils


def build_object_points() -> np.ndarray:
    cols, rows = config.CHECKERBOARD_INNER_CORNERS
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= config.CHECKERBOARD_SQUARE_SIZE_IN
    return objp


def calibrate_one_camera(cam_id):
    backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else 0
    cap = cv2.VideoCapture(cam_id, backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    if not cap.isOpened():
        print(f"Could not open camera {cam_id}, skipping.")
        cap.release()
        return None

    pattern_size = config.CHECKERBOARD_INNER_CORNERS
    objp = build_object_points()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objpoints = []  # 3D real-world points (same board, reused each capture)
    imgpoints = []  # 2D detected corner pixels, one set per capture
    image_size = None

    print(f"\n=== Camera {cam_id} === show the checkerboard, 'c' to capture, "
          f"'f' to finish (needs >={config.MIN_INTRINSIC_CAPTURES}), 'q' to abort")

    result = None
    while True:
        ok, frame = cap.read()
        if not ok:
            print(f"Camera {cam_id}: failed to read a frame")
            break
        image_size = (frame.shape[1], frame.shape[0])

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(
            gray, pattern_size,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)

        display = frame.copy()
        if found:
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            cv2.drawChessboardCorners(display, pattern_size, corners_refined, found)
        else:
            corners_refined = None

        status = "BOARD DETECTED - press c to capture" if found else "no board detected"
        cv2.putText(display, f"Camera {cam_id}: {status}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if found else (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(display, f"captures: {len(objpoints)} / {config.MIN_INTRINSIC_CAPTURES} min "
                              f"  c=capture  f=finish  q=abort",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("calibrate_intrinsics", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("c") and found:
            objpoints.append(objp)
            imgpoints.append(corners_refined)
            print(f"  captured sample {len(objpoints)}")
        elif key == ord("f"):
            if len(objpoints) >= config.MIN_INTRINSIC_CAPTURES:
                break
            print(f"  Need at least {config.MIN_INTRINSIC_CAPTURES} captures "
                  f"(have {len(objpoints)}).")
        elif key == ord("q"):
            objpoints = []
            break

    cap.release()

    if len(objpoints) >= config.MIN_INTRINSIC_CAPTURES:
        rms_error, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(
            objpoints, imgpoints, image_size, None, None)
        print(f"  Calibrated camera {cam_id}: RMS reprojection error = {rms_error:.4f} px "
              f"(lower is better; under ~0.5px is good)")
        result = {
            "resolution": list(image_size),
            "camera_matrix": camera_matrix.tolist(),
            "dist_coeffs": dist_coeffs.tolist(),
            "rms_error_px": rms_error,
        }

    return result


def main():
    intrinsics = camera_utils.load_intrinsics()
    for cam_id in config.CAMERA_IDS:
        entry = calibrate_one_camera(cam_id)
        if entry is not None:
            intrinsics[str(cam_id)] = entry
        else:
            print(f"Camera {cam_id} not calibrated this run (kept previous value if any).")
    cv2.destroyAllWindows()
    camera_utils.save_intrinsics(intrinsics)
    print(f"\nSaved intrinsics to {config.INTRINSICS_FILE}")
    print("Now re-run calibrate_cameras.py so the homography is built from undistorted frames.")


if __name__ == "__main__":
    main()
