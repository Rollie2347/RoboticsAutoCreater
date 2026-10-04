"""
camera_utils.py

Shared helpers for loading per-camera lens (intrinsic) calibration and
undistorting frames. Used by calibrate_cameras.py (so the homography is
built from already-undistorted images) and tracking.py (so every tracked
frame is undistorted before detection) -- both must undistort the same way,
or pixel coordinates won't be consistent between calibration time and
runtime.

If no intrinsics have been saved for a camera (calibrate_intrinsics.py
hasn't been run for it yet), callers fall back to using the raw frame
un-undistorted -- the pipeline still works, just with whatever lens
distortion error that camera has.
"""
import json
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

import config


def load_intrinsics() -> Dict[str, dict]:
    try:
        with open(config.INTRINSICS_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def save_intrinsics(data: dict):
    with open(config.INTRINSICS_FILE, "w") as f:
        json.dump(data, f, indent=2)


class Undistorter:
    """Precomputes undistortion remap tables once per camera so undistorting
    each live frame is a cheap cv2.remap() instead of recomputing the
    correction from scratch every call.

    alpha=1 keeps the full original field of view (you may see black
    curved borders where there's no valid pixel data). Use alpha=0 instead
    if you'd rather crop out those borders and don't mind a slightly
    narrower field of view.
    """

    def __init__(self, camera_matrix: np.ndarray, dist_coeffs: np.ndarray,
                 resolution: Tuple[int, int], alpha: float = 1.0):
        w, h = resolution
        self.new_camera_matrix, self.roi = cv2.getOptimalNewCameraMatrix(
            camera_matrix, dist_coeffs, (w, h), alpha, (w, h))
        self.map1, self.map2 = cv2.initUndistortRectifyMap(
            camera_matrix, dist_coeffs, None, self.new_camera_matrix, (w, h), cv2.CV_16SC2)

    def undistort(self, frame: np.ndarray) -> np.ndarray:
        return cv2.remap(frame, self.map1, self.map2, interpolation=cv2.INTER_LINEAR)


def build_undistorters(resolution_by_cam: Dict[int, Tuple[int, int]]) -> Dict[int, Optional[Undistorter]]:
    """Returns {cam_id: Undistorter or None}. None means no saved intrinsics
    for that camera yet -- caller should use the raw frame as-is."""
    intrinsics = load_intrinsics()
    result: Dict[int, Optional[Undistorter]] = {}
    for cam_id, resolution in resolution_by_cam.items():
        entry = intrinsics.get(str(cam_id))
        if entry is None:
            result[cam_id] = None
            continue
        camera_matrix = np.array(entry["camera_matrix"])
        dist_coeffs = np.array(entry["dist_coeffs"])
        result[cam_id] = Undistorter(camera_matrix, dist_coeffs, tuple(resolution))
    return result
