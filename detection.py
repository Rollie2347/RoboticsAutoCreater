"""
detection.py

Color-threshold detection of the robot, its heading markers, and the game
elements in a single camera frame. Everything here works on one frame at a
time; tracking.py turns these pixel detections into real-world field
coordinates.

Two things worth understanding before editing this file:

Ground-contact point vs centroid
---------------------------------
A homography maps points that lie ON the field plane (z=0). A ball's pixel
centroid corresponds to its physical center, which sits one radius above
the field -- feeding the centroid into the homography introduces a parallax
error that grows with how oblique the camera angle is. Instead we use the
bottom-most pixel of the detected blob (its ground-contact point) as the
input to the homography, which is much closer to a true z=0 point.

Size gating
-----------
Different game elements have different real diameters. Because the
homography tells us how many pixels an inch covers at any point on the
field, we can predict how large a given element *should* look where it was
found, and reject blobs that don't match. That check is independent of
lighting, so it stays reliable when color thresholds drift. tracking.py
supplies the prediction through the `px_diameter_at` callback.
"""
import math
from dataclasses import dataclass
from typing import Callable, List, Optional

import cv2
import numpy as np

import config


@dataclass
class Detection:
    label: str
    ground_px: tuple  # (x, y) pixel used for the homography (ground-contact point)
    centroid_px: tuple  # (x, y) pixel centroid, for drawing
    radius_px: float  # only meaningful for circular (ball) detections
    bbox: tuple  # (x, y, w, h)
    area_px: float
    count: int = 1  # >1 when a cluster could not be fully split but was counted
    from_cluster: bool = False


def _mask_for_range(hsv_frame: np.ndarray, thresh: dict) -> np.ndarray:
    h_min, h_max = thresh["h_min"], thresh["h_max"]
    s_range = (thresh["s_min"], thresh["s_max"])
    v_range = (thresh["v_min"], thresh["v_max"])

    if h_min <= h_max:
        lower = np.array([h_min, s_range[0], v_range[0]])
        upper = np.array([h_max, s_range[1], v_range[1]])
        mask = cv2.inRange(hsv_frame, lower, upper)
    else:
        # Hue wraps around (e.g. red spans both ends of the 0-179 range).
        lower1 = np.array([h_min, s_range[0], v_range[0]])
        upper1 = np.array([179, s_range[1], v_range[1]])
        lower2 = np.array([0, s_range[0], v_range[0]])
        upper2 = np.array([h_max, s_range[1], v_range[1]])
        mask = cv2.bitwise_or(cv2.inRange(hsv_frame, lower1, upper1),
                               cv2.inRange(hsv_frame, lower2, upper2))

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    return mask


def _ground_point_from_bbox(bbox) -> tuple:
    x, y, w, h = bbox
    return (x + w / 2.0, y + h)


def _detection_from_contour(contour, label: str, from_cluster: bool = False,
                             count: int = 1) -> Optional[Detection]:
    area = cv2.contourArea(contour)
    if area <= 0:
        return None
    bbox = cv2.boundingRect(contour)
    (cx, cy), radius = cv2.minEnclosingCircle(contour)
    return Detection(
        label=label,
        ground_px=_ground_point_from_bbox(bbox),
        centroid_px=(cx, cy),
        radius_px=radius,
        bbox=bbox,
        area_px=area,
        count=count,
        from_cluster=from_cluster,
    )


def _circularity(contour) -> float:
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 0:
        return 0.0
    return 4 * np.pi * area / (perimeter * perimeter)


def _split_cluster(contour, expected_px_diameter: float,
                   px_to_field: Optional[Callable[[tuple], tuple]] = None,
                   diameter_in: float = 0.0) -> List[np.ndarray]:
    """Split a blob of touching balls into separate contours. Returns [] if
    it can't confidently split, in which case the caller falls back to
    counting by area.

    The distance transform peaks near the center of each ball in the blob.
    Seeds are its *local* maxima, not one global threshold: between touching
    balls the blob's waist is still ~2/3 as deep as a ball center (more
    after the mask's morphological close), so a global threshold merges
    neighbouring seeds. Each blob pixel then goes to its nearest seed.
    """
    if expected_px_diameter <= 0:
        return []

    try:
        x, y, w, h = cv2.boundingRect(contour)
        if w <= 2 or h <= 2:
            return []

        pad = 2  # keep the blob off the ROI border
        roi = np.zeros((h + 2 * pad, w + 2 * pad), dtype=np.uint8)
        cv2.drawContours(roi, [contour], -1, 255, thickness=cv2.FILLED,
                          offset=(-x + pad, -y + pad))

        dist = cv2.distanceTransform(roi, cv2.DIST_L2, 5)
        radius = expected_px_diameter / 2.0

        # A pixel is a peak if it's the max within ~one ball radius and deep
        # enough inside the blob to be a ball center rather than an edge bump.
        k = max(3, int(round(radius * config.CLUSTER_PEAK_WINDOW)) | 1)
        local_max = cv2.dilate(dist, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        peaks = ((dist >= local_max - 1e-3)
                 & (dist >= config.CLUSTER_SEED_FRACTION * radius)).astype(np.uint8)

        # Collapse plateaus/near-duplicates: peaks closer than
        # CLUSTER_MIN_SEED_SEP ball diameters belong to the same ball.
        #
        # Separation is measured in FIELD inches when a homography is
        # available. In pixels, ball spacing along the camera's viewing
        # direction is compressed by ~sin(elevation), so no pixel threshold
        # works at every angle. All seeds are ball centers at the same
        # height, so projecting them through the ground homography distorts
        # them uniformly (a few % scale) and their spacing stays reliable.
        n, _, _, centers = cv2.connectedComponentsWithStats(peaks)
        ordered = sorted(centers[1:], key=lambda c: -dist[int(c[1]), int(c[0])])
        if px_to_field is not None and diameter_in > 0:
            min_sep = config.CLUSTER_MIN_SEED_SEP * diameter_in
            to_metric = lambda cx, cy: px_to_field((cx + x - pad, cy + y - pad))
        else:
            min_sep = config.CLUSTER_MIN_SEED_SEP_PX * expected_px_diameter
            to_metric = lambda cx, cy: (cx, cy)
        seeds_xy, seeds_metric = [], []
        for cx, cy in ordered:
            m = to_metric(cx, cy)
            if all(math.hypot(m[0] - sm[0], m[1] - sm[1]) >= min_sep for sm in seeds_metric):
                seeds_xy.append((cx, cy))
                seeds_metric.append(m)

        if len(seeds_xy) < 2 or len(seeds_xy) > config.CLUSTER_MAX_SPLIT:
            return []

        # Assign every blob pixel to its nearest seed. Balls in a cluster are
        # the same size, so the true cut between two touching balls is the
        # perpendicular bisector of their centers -- exactly what
        # nearest-seed assignment produces. (cv2.watershed floods by color
        # gradient, which is flat inside a binary mask, so it doesn't help.)
        ys, xs = np.nonzero(roi)
        seeds = np.array(seeds_xy, dtype=np.float32)
        d2 = ((xs[:, None] - seeds[None, :, 0]) ** 2
              + (ys[:, None] - seeds[None, :, 1]) ** 2)
        owner = np.argmin(d2, axis=1)

        pieces = []
        for i in range(len(seeds_xy)):
            piece_mask = np.zeros(roi.shape, dtype=np.uint8)
            sel = owner == i
            piece_mask[ys[sel], xs[sel]] = 255
            piece_contours, _ = cv2.findContours(
                piece_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not piece_contours:
                continue
            largest = max(piece_contours, key=cv2.contourArea)
            # Shift back from padded-ROI coordinates into full-frame coordinates.
            pieces.append(largest + np.array([[x - pad, y - pad]], dtype=largest.dtype))

        return pieces if len(pieces) >= 2 else []
    except cv2.error:
        # Watershed is fiddly; never let a split failure kill the frame.
        return []


def detect_balls(frame: np.ndarray, thresholds: dict, label: str,
                  diameter_in: float, min_area: float, min_circularity: float,
                  px_diameter_at: Optional[Callable[[tuple, float], float]] = None,
                  px_to_field: Optional[Callable[[tuple], tuple]] = None
                  ) -> List[Detection]:
    """Find every ball of one class in the frame.

    px_diameter_at(ground_px, diameter_in) should return how many pixels
    across an object of that real diameter would appear at that point on
    the field. When it's None (no homography available), size gating and
    declustering are skipped and only the circularity filter applies.

    px_to_field(px) maps a pixel to field inches; it lets declustering
    measure seed spacing in real units (see _split_cluster).
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = _mask_for_range(hsv, thresholds)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections: List[Detection] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        bbox = cv2.boundingRect(contour)
        ground_px = _ground_point_from_bbox(bbox)

        expected_px_d = None
        if px_diameter_at is not None:
            expected_px_d = px_diameter_at(ground_px, diameter_in)

        # --- no homography: fall back to circularity + a raw pixel cap ------
        if not expected_px_d or expected_px_d <= 0:
            if config.MAX_BALL_AREA_PX and area > config.MAX_BALL_AREA_PX:
                continue
            if _circularity(contour) < min_circularity:
                continue
            det = _detection_from_contour(contour, label)
            if det:
                detections.append(det)
            continue

        expected_area = np.pi / 4.0 * expected_px_d * expected_px_d

        # --- too big to be balls at all: reject outright ---------------------
        # Checked before the cluster branch, because that branch would
        # otherwise clamp an arbitrarily large blob to CLUSTER_MAX_SPLIT and
        # report it as a pile rather than discarding it.
        if config.MAX_CLUSTER_BALLS and area > config.MAX_CLUSTER_BALLS * expected_area:
            continue

        # --- cluster: try to split it, else count it by area ----------------
        if config.ENABLE_DECLUSTER and area >= config.CLUSTER_AREA_FACTOR * expected_area:
            pieces = _split_cluster(contour, expected_px_d, px_to_field, diameter_in)
            if pieces:
                # A piece can still hold more than one ball (one mostly hidden
                # behind another gives no seed of its own). So the total comes
                # from the whole blob's area, every piece counts at least 1,
                # and any remainder goes to the pieces with the most area not
                # yet accounted for. Pieces are never dropped for being small:
                # they're parts of a blob that already passed the checks.
                total = max(len(pieces), min(int(round(area / expected_area)),
                                             config.CLUSTER_MAX_SPLIT))
                piece_areas = [cv2.contourArea(p) for p in pieces]
                counts = [1] * len(pieces)
                for _ in range(total - len(pieces)):
                    residual = [pa / expected_area - c for pa, c in zip(piece_areas, counts)]
                    counts[int(np.argmax(residual))] += 1
                for piece, piece_count in zip(pieces, counts):
                    det = _detection_from_contour(piece, label, from_cluster=True,
                                                   count=piece_count)
                    if det:
                        detections.append(det)
                continue

            # Couldn't split it cleanly -- still report how many are in there,
            # positioned at the blob as a whole.
            estimated = int(round(area / expected_area))
            estimated = max(1, min(estimated, config.CLUSTER_MAX_SPLIT))
            det = _detection_from_contour(contour, label, from_cluster=True,
                                           count=estimated)
            if det:
                detections.append(det)
            continue

        # --- single ball: check shape and size ------------------------------
        if _circularity(contour) < min_circularity:
            continue

        measured_px_d = max(bbox[2], bbox[3])
        ratio = measured_px_d / expected_px_d
        if abs(ratio - 1.0) > config.BALL_SIZE_TOLERANCE:
            continue

        det = _detection_from_contour(contour, label)
        if det:
            detections.append(det)

    return detections


def detect_marker(frame: np.ndarray, thresholds: dict, min_area: float,
                   label: str) -> Optional[Detection]:
    """Find the single largest blob of a color -- used for the robot body
    patch and for the front/back heading markers. Smaller false positives
    elsewhere in the frame are discarded."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = _mask_for_range(hsv, thresholds)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = [c for c in contours if cv2.contourArea(c) >= min_area]
    if not candidates:
        return None

    largest = max(candidates, key=cv2.contourArea)
    moments = cv2.moments(largest)
    if moments["m00"] == 0:
        return None

    bbox = cv2.boundingRect(largest)
    return Detection(
        label=label,
        ground_px=_ground_point_from_bbox(bbox),
        centroid_px=(moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]),
        radius_px=0.0,
        bbox=bbox,
        area_px=cv2.contourArea(largest),
    )


def detect_robot(frame: np.ndarray, thresholds: dict, min_area: float) -> List[Detection]:
    """Backwards-compatible wrapper: returns a list of 0 or 1 detections."""
    det = detect_marker(frame, thresholds, min_area, config.ROBOT_LABEL)
    return [det] if det else []
