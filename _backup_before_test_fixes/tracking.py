"""
tracking.py

Turns per-camera pixel detections (from detection.py) into fused,
real-world field coordinates, and renders both the camera overlay and a
top-down field map.

Pipeline per frame:
    1. undistort the frame (if that camera has intrinsics)
    2. detect the robot patch, the two heading markers, and each ball class
    3. project every detection's ground-contact pixel to field (x, y) inches
       via that camera's homography
    4. fuse detections of the same object across cameras (a no-op with one
       camera, a weighted average with more)
    5. smooth positions with a Kalman filter and heading with a unit-vector
       average, then compute robot-to-ball distances
"""
import json
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

import config
import detection
import camera_utils

# Draw colors, BGR.
CLASS_COLORS = {
    "robot": (255, 128, 0),
    "robot_front": (0, 255, 0),
    "robot_back": (255, 0, 255),
    "pollen": (0, 215, 255),
    "nectar_red": (60, 60, 255),
    "nectar_blue": (255, 140, 0),
}
DEFAULT_COLOR = (200, 200, 200)


def color_for(label: str):
    return CLASS_COLORS.get(label, DEFAULT_COLOR)


@dataclass
class FieldPoint:
    label: str
    position_in: Tuple[float, float]
    camera_ids: List[int] = field(default_factory=list)
    weight: float = 1.0
    count: int = 1
    heading_deg: Optional[float] = None


@dataclass
class FrameResult:
    """What one camera saw this frame, already in field coordinates."""
    cam_id: int
    annotated: Optional[np.ndarray] = None
    robot: Optional[Tuple[Tuple[float, float], float]] = None  # (position, weight)
    heading_deg: Optional[float] = None
    balls: Dict[str, List[Tuple[Tuple[float, float], float, int]]] = field(default_factory=dict)


def load_calibration() -> Dict[str, dict]:
    with open(config.CALIBRATION_FILE, "r") as f:
        return json.load(f)


def load_hsv_config() -> Dict[str, dict]:
    with open(config.HSV_CONFIG_FILE, "r") as f:
        return json.load(f)


def pixel_to_field(homography: np.ndarray, px: Tuple[float, float]) -> Tuple[float, float]:
    pt = np.array([[px]], dtype=np.float64)  # shape (1, 1, 2)
    out = cv2.perspectiveTransform(pt, homography)
    return float(out[0, 0, 0]), float(out[0, 0, 1])


def field_to_pixel(homography_inv: np.ndarray, field_pt: Tuple[float, float]) -> Tuple[float, float]:
    pt = np.array([[field_pt]], dtype=np.float64)
    out = cv2.perspectiveTransform(pt, homography_inv)
    return float(out[0, 0, 0]), float(out[0, 0, 1])


def edge_weight(px: Tuple[float, float], resolution: Tuple[int, int]) -> float:
    x, y = px
    w, h = resolution
    m = config.EDGE_MARGIN_PX
    near_edge = x < m or y < m or x > (w - m) or y > (h - m)
    return config.EDGE_WEIGHT if near_edge else 1.0


def distance_in(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


class MultiCameraTracker:
    """Owns the per-camera calibration/HSV config and turns raw frames into
    field-space detections."""

    def __init__(self):
        self.calibration = load_calibration()
        self.hsv_config = load_hsv_config()

        missing = [str(c) for c in config.CAMERA_IDS if str(c) not in self.calibration]
        if missing:
            raise RuntimeError(
                f"No calibration found for camera(s) {missing}. Run calibrate_cameras.py first."
            )
        if config.ROBOT_LABEL not in self.hsv_config:
            raise RuntimeError(
                f"No HSV thresholds for '{config.ROBOT_LABEL}'. Run: "
                f"python tune_colors.py --label {config.ROBOT_LABEL}"
            )

        # Ball classes are optional individually -- tune them as you go.
        self.active_ball_classes = [c for c in config.BALL_CLASSES if c in self.hsv_config]
        for cls in config.BALL_CLASSES:
            if cls not in self.active_ball_classes:
                print(f"Note: '{cls}' has no HSV thresholds yet and will be skipped. "
                      f"Tune it with: python tune_colors.py --label {cls}")

        # Heading needs BOTH markers; without them position still works.
        self.heading_available = (config.ROBOT_FRONT_LABEL in self.hsv_config
                                   and config.ROBOT_BACK_LABEL in self.hsv_config)
        if not self.heading_available:
            print(f"Note: heading disabled -- tune both "
                  f"'{config.ROBOT_FRONT_LABEL}' and '{config.ROBOT_BACK_LABEL}' to enable it.")

        self.homographies = {
            cam_id: np.array(self.calibration[str(cam_id)]["homography"])
            for cam_id in config.CAMERA_IDS
        }
        self.h_inverses = {
            cam_id: np.linalg.inv(self.homographies[cam_id])
            for cam_id in config.CAMERA_IDS
        }
        self.resolutions = {
            cam_id: tuple(self.calibration[str(cam_id)]["resolution"])
            for cam_id in config.CAMERA_IDS
        }

        self.undistorters = camera_utils.build_undistorters(self.resolutions)
        for cam_id, undistorter in self.undistorters.items():
            if undistorter is None:
                print(f"Note: camera {cam_id} has no intrinsic calibration "
                      f"(run calibrate_intrinsics.py) -- tracking without lens undistortion.")

    def _px_diameter_at(self, cam_id: int) -> Callable[[tuple, float], float]:
        """Returns a function predicting how many pixels across an object of
        a given real diameter appears at a given ground pixel. Used for size
        gating and for deciding when a blob is a cluster."""
        H = self.homographies[cam_id]
        H_inv = self.h_inverses[cam_id]

        def predict(ground_px: tuple, diameter_in: float) -> float:
            try:
                fx, fy = pixel_to_field(H, ground_px)
                origin = field_to_pixel(H_inv, (fx, fy))
                along_x = field_to_pixel(H_inv, (fx + diameter_in, fy))
                along_y = field_to_pixel(H_inv, (fx, fy + diameter_in))
                dx = math.hypot(along_x[0] - origin[0], along_x[1] - origin[1])
                dy = math.hypot(along_y[0] - origin[0], along_y[1] - origin[1])
                return (dx + dy) / 2.0
            except (cv2.error, ValueError):
                return 0.0

        return predict

    def process_frame(self, cam_id: int, frame: np.ndarray) -> FrameResult:
        undistorter = self.undistorters[cam_id]
        if undistorter is not None:
            frame = undistorter.undistort(frame)

        H = self.homographies[cam_id]
        resolution = self.resolutions[cam_id]
        predict_px_diameter = self._px_diameter_at(cam_id)
        annotated = frame.copy()
        result = FrameResult(cam_id=cam_id, annotated=annotated)

        # --- robot body patch -------------------------------------------------
        robot_det = detection.detect_marker(
            frame, self.hsv_config[config.ROBOT_LABEL],
            config.MIN_ROBOT_AREA_PX, config.ROBOT_LABEL)
        robot_pos = None
        if robot_det is not None:
            robot_pos = pixel_to_field(H, robot_det.ground_px)
            weight = edge_weight(robot_det.ground_px, resolution)
            result.robot = (robot_pos, weight)
            _draw_detection(annotated, robot_det,
                             f"robot {robot_pos[0]:.1f},{robot_pos[1]:.1f}in")

        # --- heading markers --------------------------------------------------
        if self.heading_available:
            front = detection.detect_marker(
                frame, self.hsv_config[config.ROBOT_FRONT_LABEL],
                config.MIN_ROBOT_AREA_PX, config.ROBOT_FRONT_LABEL)
            back = detection.detect_marker(
                frame, self.hsv_config[config.ROBOT_BACK_LABEL],
                config.MIN_ROBOT_AREA_PX, config.ROBOT_BACK_LABEL)

            if front is not None and back is not None:
                front_pos = pixel_to_field(H, front.ground_px)
                back_pos = pixel_to_field(H, back.ground_px)
                result.heading_deg = math.degrees(math.atan2(
                    front_pos[1] - back_pos[1], front_pos[0] - back_pos[0]))

                # Markers sit above the field plane, so their projected
                # positions are less trustworthy than the body patch's --
                # use them for direction, and only fall back to their
                # midpoint for position if the body patch wasn't found.
                if result.robot is None:
                    midpoint = ((front_pos[0] + back_pos[0]) / 2.0,
                                (front_pos[1] + back_pos[1]) / 2.0)
                    weight = min(edge_weight(front.ground_px, resolution),
                                  edge_weight(back.ground_px, resolution))
                    result.robot = (midpoint, weight)

                cv2.arrowedLine(annotated,
                                 (int(back.centroid_px[0]), int(back.centroid_px[1])),
                                 (int(front.centroid_px[0]), int(front.centroid_px[1])),
                                 (255, 255, 255), 2, tipLength=0.3)
            if front is not None:
                _draw_detection(annotated, front, "front")
            if back is not None:
                _draw_detection(annotated, back, "back")

        # --- game elements ----------------------------------------------------
        for cls in self.active_ball_classes:
            diameter_in = config.BALL_CLASSES[cls]
            dets = detection.detect_balls(
                frame, self.hsv_config[cls], cls, diameter_in,
                config.MIN_BALL_AREA_PX, config.BALL_MIN_CIRCULARITY,
                px_diameter_at=predict_px_diameter)

            entries = []
            for d in dets:
                pos = pixel_to_field(H, d.ground_px)
                weight = edge_weight(d.ground_px, resolution)
                entries.append((pos, weight, d.count))
                suffix = f" x{d.count}" if d.count > 1 else ""
                _draw_detection(annotated, d, f"{cls}{suffix}")
            result.balls[cls] = entries

        header = f"cam {cam_id}"
        if result.heading_deg is not None:
            header += f"   heading {result.heading_deg:+.1f}deg"
        cv2.putText(annotated, header, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2, cv2.LINE_AA)

        return result


def _draw_detection(frame, det: detection.Detection, label: str):
    color = color_for(det.label)
    x, y, w, h = det.bbox
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    gx, gy = det.ground_px
    cv2.drawMarker(frame, (int(gx), int(gy)), color, cv2.MARKER_TILTED_CROSS, 12, 2)
    cv2.putText(frame, label, (x, max(12, y - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, 2, cv2.LINE_AA)


def fuse_robot(frames: List[Optional[FrameResult]]) -> Optional[FieldPoint]:
    """Weighted-average the robot position across cameras, and average the
    heading as a unit vector (averaging raw angles breaks across 0/360)."""
    positions, weights, cams = [], [], []
    heading_vectors = []

    for result in frames:
        if result is None:
            continue
        if result.robot is not None:
            pos, w = result.robot
            positions.append(pos)
            weights.append(w)
            cams.append(result.cam_id)
        if result.heading_deg is not None:
            radians = math.radians(result.heading_deg)
            heading_vectors.append((math.cos(radians), math.sin(radians)))

    if not positions:
        return None

    fused = np.average(np.array(positions), axis=0, weights=np.array(weights))

    heading_deg = None
    if heading_vectors:
        mean_vector = np.mean(np.array(heading_vectors), axis=0)
        if np.linalg.norm(mean_vector) > 1e-6:
            heading_deg = math.degrees(math.atan2(mean_vector[1], mean_vector[0]))

    return FieldPoint(label="robot",
                       position_in=(float(fused[0]), float(fused[1])),
                       camera_ids=cams,
                       weight=float(np.sum(weights)),
                       heading_deg=heading_deg)


def fuse_balls(frames: List[Optional[FrameResult]]) -> Dict[str, List[FieldPoint]]:
    """Greedy clustering across cameras, per class: detections within
    config.FUSION_MATCH_RADIUS_IN of an existing cluster's running average
    are merged into it, otherwise they start a new one. With a single
    camera this just passes detections straight through."""
    fused_by_class: Dict[str, List[FieldPoint]] = {}

    for cls in config.BALL_CLASSES:
        clusters: List[dict] = []

        for result in frames:
            if result is None:
                continue
            for pos, weight, count in result.balls.get(cls, []):
                pos_arr = np.array(pos)
                best_cluster, best_dist = None, None
                for cluster in clusters:
                    center = np.average(np.array(cluster["positions"]), axis=0,
                                         weights=cluster["weights"])
                    dist = float(np.linalg.norm(pos_arr - center))
                    if dist <= config.FUSION_MATCH_RADIUS_IN and (best_dist is None or dist < best_dist):
                        best_cluster, best_dist = cluster, dist

                if best_cluster is not None:
                    best_cluster["positions"].append(pos)
                    best_cluster["weights"].append(weight)
                    best_cluster["cams"].append(result.cam_id)
                    best_cluster["count"] = max(best_cluster["count"], count)
                else:
                    clusters.append({"positions": [pos], "weights": [weight],
                                      "cams": [result.cam_id], "count": count})

        points = []
        for cluster in clusters:
            fused = np.average(np.array(cluster["positions"]), axis=0,
                                weights=np.array(cluster["weights"]))
            points.append(FieldPoint(label=cls,
                                      position_in=(float(fused[0]), float(fused[1])),
                                      camera_ids=cluster["cams"],
                                      weight=float(np.sum(cluster["weights"])),
                                      count=cluster["count"]))
        fused_by_class[cls] = points

    return fused_by_class


class KalmanTrack:
    """Constant-velocity Kalman filter for one object's fused 2D field
    position. State: [x, y, vx, vy] (inches, inches/sec). Measurement: [x, y].
    """

    def __init__(self, initial_pos: Tuple[float, float]):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                               [0, 1, 0, 0]], dtype=np.float32)
        self.kf.transitionMatrix = np.eye(4, dtype=np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * config.SMOOTHING_PROCESS_NOISE
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * config.SMOOTHING_MEASUREMENT_NOISE
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        self.kf.statePost = np.array([[initial_pos[0]], [initial_pos[1]], [0], [0]],
                                      dtype=np.float32)
        self.missed_frames = 0

    def predict(self, dt: float) -> Tuple[float, float]:
        self.kf.transitionMatrix = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float32)
        pred = self.kf.predict()
        return float(pred[0, 0]), float(pred[1, 0])

    def correct(self, measured_pos: Tuple[float, float]) -> Tuple[float, float]:
        meas = np.array([[np.float32(measured_pos[0])], [np.float32(measured_pos[1])]])
        corrected = self.kf.correct(meas)
        self.missed_frames = 0
        return float(corrected[0, 0]), float(corrected[1, 0])


class ObjectSmoother:
    """Persistent Kalman tracks for the robot and for each ball class, plus
    unit-vector smoothing for heading. Call update() once per frame."""

    def __init__(self):
        self._robot_track: Optional[KalmanTrack] = None
        self._ball_tracks: Dict[str, List[KalmanTrack]] = {}
        self._heading_vector: Optional[Tuple[float, float]] = None

    def update(self, fused_robot: Optional[FieldPoint],
               fused_balls: Dict[str, List[FieldPoint]],
               dt: float) -> Tuple[Optional[FieldPoint], Dict[str, List[FieldPoint]]]:
        dt = max(dt, 1e-3)
        robot = self._update_robot(fused_robot, dt)
        balls = {cls: self._update_ball_class(cls, points, dt)
                 for cls, points in fused_balls.items()}
        return robot, balls

    def _smooth_heading(self, heading_deg: Optional[float]) -> Optional[float]:
        if heading_deg is None:
            # Keep reporting the last known heading rather than dropping it;
            # it stops being reported once the robot track itself is dropped.
            if self._heading_vector is None:
                return None
            return math.degrees(math.atan2(self._heading_vector[1],
                                            self._heading_vector[0]))

        radians = math.radians(heading_deg)
        measurement = (math.cos(radians), math.sin(radians))
        if self._heading_vector is None:
            self._heading_vector = measurement
        else:
            a = config.HEADING_SMOOTHING
            blended = (a * self._heading_vector[0] + (1 - a) * measurement[0],
                       a * self._heading_vector[1] + (1 - a) * measurement[1])
            norm = math.hypot(blended[0], blended[1])
            self._heading_vector = (blended[0] / norm, blended[1] / norm) if norm > 1e-6 else measurement

        return math.degrees(math.atan2(self._heading_vector[1], self._heading_vector[0]))

    def _update_robot(self, fused_robot: Optional[FieldPoint], dt: float) -> Optional[FieldPoint]:
        if self._robot_track is None:
            if fused_robot is None:
                return None
            self._robot_track = KalmanTrack(fused_robot.position_in)
            fused_robot.heading_deg = self._smooth_heading(fused_robot.heading_deg)
            return fused_robot

        predicted = self._robot_track.predict(dt)

        if fused_robot is not None:
            corrected = self._robot_track.correct(fused_robot.position_in)
            return FieldPoint(label="robot", position_in=corrected,
                               camera_ids=fused_robot.camera_ids,
                               weight=fused_robot.weight,
                               heading_deg=self._smooth_heading(fused_robot.heading_deg))

        self._robot_track.missed_frames += 1
        if self._robot_track.missed_frames > config.SMOOTHING_MAX_MISSED_FRAMES:
            self._robot_track = None
            self._heading_vector = None
            return None
        return FieldPoint(label="robot", position_in=predicted, camera_ids=[],
                           weight=0.0, heading_deg=self._smooth_heading(None))

    def _update_ball_class(self, cls: str, points: List[FieldPoint],
                            dt: float) -> List[FieldPoint]:
        tracks = self._ball_tracks.setdefault(cls, [])
        predictions = [(track, track.predict(dt)) for track in tracks]
        unmatched = list(points)
        results: List[FieldPoint] = []

        for track, predicted in predictions:
            best, best_dist = None, None
            for candidate in unmatched:
                dist = distance_in(predicted, candidate.position_in)
                if dist <= config.FUSION_MATCH_RADIUS_IN and (best_dist is None or dist < best_dist):
                    best, best_dist = candidate, dist

            if best is not None:
                corrected = track.correct(best.position_in)
                results.append(FieldPoint(label=cls, position_in=corrected,
                                           camera_ids=best.camera_ids,
                                           weight=best.weight, count=best.count))
                unmatched.remove(best)
            else:
                track.missed_frames += 1
                if track.missed_frames <= config.SMOOTHING_MAX_MISSED_FRAMES:
                    results.append(FieldPoint(label=cls, position_in=predicted,
                                               camera_ids=[], weight=0.0))

        self._ball_tracks[cls] = [t for t in tracks
                                   if t.missed_frames <= config.SMOOTHING_MAX_MISSED_FRAMES]

        for leftover in unmatched:
            self._ball_tracks[cls].append(KalmanTrack(leftover.position_in))
            results.append(leftover)

        return results


def count_by_class(balls: Dict[str, List[FieldPoint]]) -> Dict[str, int]:
    """Total number of each element class currently seen, counting an
    unsplit cluster as however many balls it was estimated to contain."""
    return {cls: sum(p.count for p in points) for cls, points in balls.items()}


def draw_field_map(robot: Optional[FieldPoint],
                    balls: Dict[str, List[FieldPoint]],
                    scale_px_per_in: float = 4.0) -> np.ndarray:
    w = int(config.FIELD_WIDTH_IN * scale_px_per_in)
    h = int(config.FIELD_HEIGHT_IN * scale_px_per_in)
    img = np.full((h, w, 3), 60, dtype=np.uint8)  # gray field background

    tile_px = max(1, int(config.TILE_SIZE_IN * scale_px_per_in))
    for x in range(0, w + 1, tile_px):
        cv2.line(img, (x, 0), (x, h), (30, 30, 30), 1)
    for y in range(0, h + 1, tile_px):
        cv2.line(img, (0, y), (w, y), (30, 30, 30), 1)

    def to_px(pos):
        return int(pos[0] * scale_px_per_in), int(pos[1] * scale_px_per_in)

    for cls, points in balls.items():
        color = color_for(cls)
        for point in points:
            position = to_px(point.position_in)
            coasting = not point.camera_ids
            radius = max(4, int(config.BALL_CLASSES.get(cls, 3.0) / 2.0 * scale_px_per_in))
            cv2.circle(img, position, radius, color, 1 if coasting else -1)

            label = f"{cls}"
            if point.count > 1:
                label += f" x{point.count}"
            if coasting:
                label += " (predicted)"
            if robot is not None:
                label += f"  {distance_in(robot.position_in, point.position_in):.1f}in"
                cv2.line(img, to_px(robot.position_in), position, color, 1)
            cv2.putText(img, label, (position[0] + radius + 3, position[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    if robot is not None:
        position = to_px(robot.position_in)
        coasting = not robot.camera_ids
        color = color_for("robot")
        cv2.circle(img, position, 10, color, 1 if coasting else -1)

        label = f"robot ({robot.position_in[0]:.1f}, {robot.position_in[1]:.1f}) in"
        if robot.heading_deg is not None:
            label += f"  {robot.heading_deg:+.0f}deg"
            radians = math.radians(robot.heading_deg)
            arrow_len = 14.0 * scale_px_per_in
            tip = (int(position[0] + arrow_len * math.cos(radians)),
                   int(position[1] + arrow_len * math.sin(radians)))
            cv2.arrowedLine(img, position, tip, (255, 255, 255), 2, tipLength=0.3)
        if coasting:
            label += " (predicted)"
        cv2.putText(img, label, (position[0] + 12, position[1] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    totals = count_by_class(balls)
    summary = "  ".join(f"{cls}:{n}" for cls, n in totals.items() if n)
    if summary:
        cv2.putText(img, summary, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return img
