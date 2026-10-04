"""
check_log.py

Grades a run recorded with `python main.py --log FILE.csv` against the field
test targets in the README ("Field test checklist"). Prints PASS/FAIL per
check so you don't have to judge by eye at the field.

Usage:
    python check_log.py run.csv                         # frame rate only
    python check_log.py still.csv --truth 24,48,90      # robot parked at a known spot/heading
    python check_log.py drive.csv --moving              # smoothness, jumps, dropouts
    python check_log.py pile.csv --count pollen=5       # element count
    python check_log.py run.csv --skip 2 --until 30     # only use 2 s .. 30 s of the run

Options can be combined. Coordinates are field inches in the same axes you
used in calibrate_cameras.py; heading is degrees, 0 = +X, 90 = +Y.
"""
import argparse
import csv
import math
import sys

import numpy as np

import config

# Targets from the field test checklist.
POS_TOL_IN = 1.0
HEADING_TOL_DEG = 2.0
JUMP_TOL_IN = 1.0          # a frame's step may differ from its neighbours' by at most this
HEADING_JUMP_TOL_DEG = 15.0
MAX_GAP_S = 0.25           # longest stretch with no fresh detection while moving


def _f(v):
    return float(v) if v not in ("", None) else None


def load(path, skip_s, until_s):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit(f"{path} has no rows.")
    t0 = float(rows[0]["mono_time_s"])
    out = []
    for r in rows:
        t = float(r["mono_time_s"]) - t0
        if t < skip_s or (until_s is not None and t > until_s):
            continue
        out.append({
            "t": t,
            "dt_ms": float(r["dt_ms"]),
            "raw": (_f(r["raw_x_in"]), _f(r["raw_y_in"]), _f(r["raw_heading_deg"])),
            "pose": (_f(r["x_in"]), _f(r["y_in"]), _f(r["heading_deg"])),
            "coasting": r["coasting"] == "1",
            "counts": {k[len("count_"):]: int(v) for k, v in r.items() if k.startswith("count_") and v != ""},
        })
    if len(out) < 2:
        sys.exit("Fewer than 2 rows in the selected time range.")
    return out


def verdict(ok):
    return "PASS" if ok else "FAIL"


def circ_mean_deg(angles):
    a = np.radians(angles)
    return math.degrees(math.atan2(np.sin(a).mean(), np.cos(a).mean()))


def angdiff(a, b):
    return (a - b + 180.0) % 360.0 - 180.0


def check_fps(rows):
    dt = np.array([r["dt_ms"] for r in rows[1:]]) / 1000.0
    need = config.MIN_TARGET_FPS
    slow = (dt > 1.0 / need).mean() * 100
    mean_fps = 1.0 / dt.mean()
    ok = mean_fps >= need and slow <= 5.0
    print(f"[{verdict(ok)}] Frame rate: mean {mean_fps:.1f} fps, median {1 / np.median(dt):.1f} fps, "
          f"slowest gap {dt.max() * 1000:.0f} ms, {slow:.1f}% of gaps slower than {need:.0f} fps "
          f"(target: mean >= {need:.0f} fps and <= 5% slow gaps)")
    if not ok:
        print("       -> Check the startup line says 'MJPG'. If it does, processing is the limit: "
              "close other programs, or lower FRAME_WIDTH/FRAME_HEIGHT and recalibrate.")


def check_still(rows, truth):
    tx, ty, th = truth
    seen = [r for r in rows if r["pose"][0] is not None and not r["coasting"]]
    if not seen:
        print("[FAIL] Standing still: robot never seen. Is it in the HIVE's shadow or out of frame?")
        return
    xs = np.array([r["pose"][0] for r in seen]); ys = np.array([r["pose"][1] for r in seen])
    mx, my = float(np.median(xs)), float(np.median(ys))
    pos_err = math.hypot(mx - tx, my - ty)
    print(f"[{verdict(pos_err <= POS_TOL_IN)}] Position: map says ({mx:.2f}, {my:.2f}), truth ({tx:g}, {ty:g}) "
          f"-> {pos_err:.2f} in off (target <= {POS_TOL_IN:g} in). Seen in {len(seen)}/{len(rows)} frames.")
    if pos_err > POS_TOL_IN:
        dx, dy = mx - tx, my - ty
        print(f"       -> Error vector ({dx:+.2f}, {dy:+.2f}) in. If it points AWAY from the camera and grows "
              "toward the far side, ROBOT_TAG_HEIGHT_IN is too low (or intrinsics.json is missing). "
              "If it's the same direction everywhere, re-run calibrate_cameras.py.")

    hs = [r["pose"][2] for r in seen if r["pose"][2] is not None]
    if th is not None:
        if not hs:
            print("[FAIL] Heading: never reported. Both markers tuned, visible, and not touching the frame edge?")
        else:
            mh = circ_mean_deg(hs)
            herr = abs(angdiff(mh, th))
            print(f"[{verdict(herr <= HEADING_TOL_DEG)}] Heading: map says {mh:.2f} deg, truth {th:g} deg "
                  f"-> {herr:.2f} deg off (target <= {HEADING_TOL_DEG:g} deg).")
            if herr > HEADING_TOL_DEG and abs(abs(angdiff(mh, th)) - 180) < 20:
                print("       -> Off by ~180 deg: robot_front and robot_back are swapped.")

    raw = [(r["raw"][0], r["raw"][1]) for r in seen if r["raw"][0] is not None]
    if len(raw) >= 30:
        raw = np.array(raw)
        noise = float(np.sqrt(raw.var(axis=0).sum() / 2))
        suggest = max(noise, 0.2)  # below this the filter stops smoothing at all
        print(f"       Raw position noise: {noise:.2f} in (1 std dev per axis). "
              f"Set SMOOTHING_MEASUREMENT_NOISE = {suggest:.2f} (now {config.SMOOTHING_MEASUREMENT_NOISE:g})"
              + ("; floored at 0.2 so smoothing still does something." if noise < 0.2 else "."))


def check_moving(rows):
    pts = [(r["t"], r["pose"]) for r in rows]
    missing = [r["pose"][0] is None for r in rows]
    stale = [r["pose"][0] is None or r["coasting"] for r in rows]

    # longest stretch without a fresh detection
    longest = 0.0; start = None
    for r, s in zip(rows, stale):
        if s and start is None:
            start = r["t"]
        elif not s and start is not None:
            longest = max(longest, r["t"] - start); start = None
    if start is not None:
        longest = max(longest, rows[-1]["t"] - start)
    ok_gap = longest <= MAX_GAP_S
    print(f"[{verdict(ok_gap)}] Dropouts: {sum(stale)} of {len(rows)} frames without a fresh detection "
          f"({sum(missing)} with nothing on the map); longest gap {longest * 1000:.0f} ms "
          f"(target <= {MAX_GAP_S * 1000:.0f} ms).")
    if not ok_gap:
        print("       -> Look at where it happened (see the times below). Near the HIVE = its shadow; "
              "at the image edge = tag touching the frame border.")
        t_prev = None
        for r, s in zip(rows, stale):
            if s and (t_prev is None or r["t"] - t_prev > 0.2):
                p = next((q["pose"] for q in reversed(rows)
                          if q["t"] < r["t"] and q["pose"][0] is not None and not q["coasting"]), None)
                where = f" last seen near ({p[0]:.0f}, {p[1]:.0f}) in" if p else ""
                print(f"          gap starts at {r['t']:.2f} s{where}")
            if s:
                t_prev = r["t"]

    # jumps: a step that differs from the average of its neighbours' steps
    worst = 0.0; worst_t = None
    for i in range(1, len(pts) - 1):
        (_, a), (_, b), (t, c) = pts[i - 1], pts[i], pts[i + 1]
        if None in (a[0], b[0], c[0]):
            continue
        s1 = np.subtract(b[:2], a[:2]); s2 = np.subtract(c[:2], b[:2])
        d = float(np.linalg.norm(s2 - s1))
        if d > worst:
            worst, worst_t = d, t
    print(f"[{verdict(worst <= JUMP_TOL_IN)}] Position jumps: worst change in step size {worst:.2f} in"
          f"{f' at {worst_t:.2f} s' if worst_t is not None else ''} (target <= {JUMP_TOL_IN:g} in).")

    hworst = 0.0; hworst_t = None
    for i in range(1, len(pts)):
        a, b = pts[i - 1][1][2], pts[i][1][2]
        if a is None or b is None:
            continue
        d = abs(angdiff(b, a))
        if d > hworst:
            hworst, hworst_t = d, pts[i][0]
    print(f"[{verdict(hworst <= HEADING_JUMP_TOL_DEG)}] Heading jumps: worst frame-to-frame change {hworst:.1f} deg"
          f"{f' at {hworst_t:.2f} s' if hworst_t is not None else ''} (target <= {HEADING_JUMP_TOL_DEG:g} deg; "
          "a 180 deg/s turn at 30 fps is 6 deg/frame).")

    # Net distance between where it was parked at the start and at the end
    # (median of the first/last 0.5 s of real detections). Unlike summed
    # path length, parked jitter doesn't inflate this.
    fresh = [r for r in rows if r["pose"][0] is not None and not r["coasting"]]
    if len(fresh) >= 10:
        t_first, t_last = fresh[0]["t"], fresh[-1]["t"]
        a = np.median([r["pose"][:2] for r in fresh if r["t"] <= t_first + 0.5], axis=0)
        b = np.median([r["pose"][:2] for r in fresh if r["t"] >= t_last - 0.5], axis=0)
        print(f"       Start ({a[0]:.1f}, {a[1]:.1f}) -> end ({b[0]:.1f}, {b[1]:.1f}): net distance "
              f"{float(np.linalg.norm(b - a)):.2f} in (compare with the real distance, e.g. 72 in; "
              "target within 1.5 in).")


def check_count(rows, spec):
    cls, _, n = spec.partition("=")
    if cls not in config.BALL_CLASSES or not n.isdigit():
        sys.exit(f"--count needs CLASS=N with CLASS one of {list(config.BALL_CLASSES)}")
    n = int(n)
    counts = np.array([r["counts"].get(cls, 0) for r in rows])
    values, freq = np.unique(counts, return_counts=True)
    mode = int(values[np.argmax(freq)])
    within = float((np.abs(counts - n) <= 1).mean() * 100)
    exact = float((counts == n).mean() * 100)
    ok = abs(mode - n) <= 1 and within >= 95
    dist = ", ".join(f"{int(v)}: {int(c)}" for v, c in zip(values, freq))
    print(f"[{verdict(ok)}] Count {cls}: truth {n}, most frames say {mode}; exact in {exact:.0f}% of frames, "
          f"within +/-1 in {within:.0f}% (target: within +/-1 in >= 95%). Frames per count: {dist}")
    if mode == 0:
        print("       -> Nothing counted: pile hidden (HIVE shadow / near wall), or HSV not tuned for this light.")
    elif mode > n + 1:
        print("       -> Overcount: check the mask in tune_colors.py for stray blobs of the same color nearby.")


def main():
    p = argparse.ArgumentParser(description="Grade a main.py --log CSV against the field test targets.")
    p.add_argument("csv")
    p.add_argument("--truth", help="X,Y[,HEADING] where the robot really is (inches, degrees)")
    p.add_argument("--moving", action="store_true", help="check smoothness / jumps / dropouts")
    p.add_argument("--count", action="append", default=[], help="CLASS=N, e.g. pollen=5 (repeatable)")
    p.add_argument("--skip", type=float, default=1.0, help="ignore the first N seconds (default 1)")
    p.add_argument("--until", type=float, default=None, help="ignore everything after N seconds")
    a = p.parse_args()

    rows = load(a.csv, a.skip, a.until)
    print(f"{a.csv}: {len(rows)} frames, {rows[-1]['t'] - rows[0]['t']:.1f} s analysed\n")
    check_fps(rows)
    if a.truth:
        parts = [float(v) for v in a.truth.split(",")]
        if len(parts) not in (2, 3):
            sys.exit("--truth needs X,Y or X,Y,HEADING")
        check_still(rows, (parts[0], parts[1], parts[2] if len(parts) == 3 else None))
    if a.moving:
        check_moving(rows)
    for spec in a.count:
        check_count(rows, spec)


if __name__ == "__main__":
    main()
