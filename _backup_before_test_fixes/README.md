# RoboticsDistanceTracker

Tracks a robot and game elements on a gray FTC field with a webcam and
reports real-world positions, heading, and distances in inches. Detection
is color-threshold based; geometry is a pixel->field homography (not stereo
triangulation), so accuracy depends on good calibration and clean color
thresholds.

## Camera setup

This runs on a **single camera**, used for both the practice field and
competition. `CAMERA_IDS` in `config.py` is a list and nothing is hardcoded
to a particular count — `main.py`, `calibrate_cameras.py`,
`calibrate_intrinsics.py` and `tracking.py` all loop over whatever is in
it, and the fusion functions handle "however many cameras actually saw this
object," including exactly one. Adding a second camera later is purely a
config change.

What single-camera operation costs you:

- **No fusion redundancy.** Fusion still runs but is effectively
  pass-through — a weighted average of one measurement. Nothing corrects a
  bad detection.
- **Occlusion is unrecoverable.** Anything hidden behind field structures
  or the robot itself is gone for those frames. The Kalman smoother coasts
  a track through short gaps but can't invent a position for something
  that stays hidden.
- **Calibrate per venue.** At competition the camera sits on a tripod at an
  oblique angle in a new position each event, so `calibrate_cameras.py`
  must be re-run on site. Check that enough tile seams or perimeter corners
  are visible from your seat to click 4+ well-spread points — if they
  aren't, no pose is possible from that vantage point.

## Tracked objects

Every object is a **label** with its own HSV thresholds in
`hsv_config.json`, tuned via `tune_colors.py`. Labels are defined in
`config.py`:

| Label | Purpose | Required? |
|---|---|---|
| `robot` | Color patch on the robot — gives position | Yes |
| `robot_front` | Heading marker at the front | Optional |
| `robot_back` | Heading marker at the back | Optional |
| `pollen` | Yellow element, 2.8 in diameter | Optional |
| `nectar_red` | Red element, 3.6 in diameter | Optional |
| `nectar_blue` | Blue element, 3.6 in diameter | Optional |

Only `robot` is required. Any class you haven't tuned yet is simply skipped
with a console note, so you can bring them online one at a time.

**Element classes carry a real-world diameter** (`BALL_CLASSES` in
`config.py`), and that gets used as a second, independent filter — see
"Size gating" below.

### Size gating

Color thresholds drift between venues and lighting setups; physical size
doesn't. Since the homography tells us how many pixels an inch covers
anywhere on the field, the tracker predicts how big an element *should*
look where it was found and rejects blobs that don't match within
`BALL_SIZE_TOLERANCE` (default ±35%).

This is what keeps 2.8 in and 3.6 in elements apart even if their color
masks bleed into each other, and it's why the tolerance needs real slack —
elements aren't perfect spheres and the homography has its own error.
Tighten it only if classes are being confused with each other.

### Declustering

Balls that touch each other merge into a single contour. When a blob's area
reaches `CLUSTER_AREA_FACTOR` × the expected area of one ball at that
position, it's treated as a cluster and split:

1. A distance transform of the blob peaks at the center of each ball in it.
2. Thresholding those peaks at `CLUSTER_SEED_FRACTION` × the expected ball
   radius yields one seed per ball.
3. Watershed grows the seeds until they meet, cutting the blob along the
   necks where the balls touch.

If the split fails or finds fewer than two pieces, the blob is reported as
a **single detection with a count** estimated from `area / expected_area`,
shown as `pollen x3` on the overlay and map. That keeps counts right even
when individual positions in a dense pile can't be resolved.

Knobs: `ENABLE_DECLUSTER`, `CLUSTER_AREA_FACTOR`, `CLUSTER_SEED_FRACTION`,
`CLUSTER_MAX_SPLIT`. Lower `CLUSTER_SEED_FRACTION` splits more
aggressively (risks cutting one ball in two); raise it to be conservative
(risks leaving two merged).

## Robot heading (pose)

Position alone isn't pose — you need heading too. Heading comes from **two
colored markers on the robot**, one at the front and one at the back. The
vector from back marker to front marker gives an unambiguous heading
(unlike an oriented bounding box, which only resolves direction mod 180°
and can't tell front from back).

Setup:

1. Put two distinctly colored patches on top of the robot, one at the
   front and one at the back, as far apart as practical — a longer baseline
   means less angular error.
2. **Mind the color budget.** Yellow, red, and blue are used by game
   elements, so markers need hues that can't collide. Green and
   magenta/purple are the open slots (see the hue table below).
3. Tune both: `python tune_colors.py --label robot_front` and
   `--label robot_back`.

Heading needs *both* markers tuned and visible. If either is missing, the
tracker prints a note at startup and reports position without heading —
nothing breaks.

Notes on accuracy: the markers sit above the field plane, so their
projected positions carry more error than the robot's ground-contact patch.
The tracker therefore uses the markers **only for direction**, and takes
position from the `robot` patch — falling back to the marker midpoint only
if the patch isn't found that frame. Heading is smoothed by averaging the
heading *unit vector* (`HEADING_SMOOTHING`), which avoids the wraparound
bug you get from averaging raw angles across 0/360°.

Heading is drawn as a white arrow on both the camera overlay (between the
markers) and the top-down map (from the robot position).

## Setup

```bash
pip install -r requirements.txt
```

```bash
python list_cameras.py         # find out which index is which physical camera
python calibrate_intrinsics.py # checkerboard capture -> intrinsics.json
python calibrate_cameras.py    # click field points -> calibration.json
python tune_colors.py --label robot
python tune_colors.py --label robot_front     # optional, enables heading
python tune_colors.py --label robot_back      # optional, enables heading
python tune_colors.py --label pollen
python tune_colors.py --label nectar_red
python tune_colors.py --label nectar_blue
```

`calibrate_intrinsics.py` corrects lens distortion and must run *before*
`calibrate_cameras.py` — it changes how pixels map to the frame, so the
homography has to be built on already-undistorted frames.

Then:

```bash
python main.py
```

Press `q` in any window to quit.

## Lens (intrinsic) calibration

`calibrate_intrinsics.py` measures the camera's actual lens distortion
using a printed checkerboard, so every frame can be straightened out
*before* the homography math runs. This is the biggest single lever on
accuracy, because the homography assumes straight lines in the real world
stay straight in the image — real lenses bend that, worst near the frame
edges.

What you need: a printed checkerboard calibration pattern (search "opencv
checkerboard calibration pattern" for a printable PDF). After printing:

1. Measure one square's actual edge length with a ruler and set
   `config.CHECKERBOARD_SQUARE_SIZE_IN` to that measured value — printer
   scaling isn't always exact, so don't trust the PDF's stated size.
2. Count the printed board's **internal** corners (where 4 squares meet,
   not the number of squares) and set `config.CHECKERBOARD_INNER_CORNERS =
   (columns, rows)` to match.

Running it, per camera:

1. A live preview opens. Detected checkerboard corners are drawn in real
   time whenever the board is fully visible in frame.
2. Press **`c`** to capture the current frame as a sample (only works when
   corners are detected — you'll see "BOARD DETECTED"). Between captures,
   move the board: vary distance, angle and tilt, and cover different
   regions of the frame, especially corners and edges. That variety is
   what constrains the distortion model — holding it still and pressing
   `c` fifteen times won't help.
3. Capture at least `MIN_INTRINSIC_CAPTURES` (15 by default), then press
   **`f`** to finish.
4. It prints an RMS reprojection error in pixels — under ~0.5px is good.
5. Results save to `intrinsics.json`. Never hand-edit it; re-run the script
   if you swap a camera or change its zoom/focus.

**After running this, re-run `calibrate_cameras.py`.** Undistortion shifts
where things land in pixel space, so a homography built before intrinsics
existed no longer matches.

If a camera has no entry in `intrinsics.json` the pipeline still works —
you just get a console warning and no distortion correction.

## Calibration walkthrough (field homography)

`calibrate_cameras.py` is a program you **run**, not a file you hand-edit.
Here's what happens:

1. A window opens showing the live feed, with `SPACE to freeze, q to abort`.
2. Once the camera is in its final position, press **spacebar** to freeze.
3. Click points on the frozen image at locations you can precisely
   identify in the real world — tile-seam intersections are ideal, since
   they're sharp and evenly spaced (24 in apart).
4. Each click prompts you **in the terminal** for that point's real-world
   coordinate:
   ```
   Point 0 clicked at pixel (312, 480).
     Enter real-world field coords for point 0 as 'x,y' (inches): 24,24
   ```
   Decide your coordinate system before you start — e.g. "the corner
   nearest the camera is (0,0), X increases right, Y increases away" — and
   use it consistently.
5. Click at least 4 points, ideally 6–8, **spread across the whole frame**
   (corners and edges, not clustered in the middle). This is the single
   biggest lever on accuracy.
6. Press **`f`** to finish. It prints a reprojection error (e.g.
   `mean=0.18in max=0.41in`). `u` undoes the last point, `r` resets.
7. Results save to `calibration.json` — never hand-edited, just re-run.

**Hard rule:** whatever corner you call `(0,0)` and whatever direction you
call `+X`/`+Y`, use the same convention every time. If you ever add a
second camera, both must share the origin and axes or fused positions will
be wrong even when each calibration looks fine on its own.

## Tuning workflow

1. **`python list_cameras.py`** — identify the camera index, set
   `CAMERA_IDS`.
2. **`python calibrate_intrinsics.py`** — once per camera, or after a
   zoom/focus change. Must precede the homography step.
3. **`python calibrate_cameras.py`** — click field points. Check the
   printed reprojection error; if it's more than ~1 inch the points were
   probably too clustered. **Re-run at every venue**, since the camera
   position changes.
4. **`python tune_colors.py --label <name>`** — once per label you want
   active. Save each with `s`.
5. **`python main.py`** — judge whether it all worked together:
   - Detection boxes should track objects tightly and stay locked on.
   - The map's dot should match reality — put the robot on a known tile
     corner and check the map agrees.
   - The heading arrow should point where the robot's front actually is.
   - **Flickering detection** → HSV range too tight, or `MIN_BALL_AREA_PX`
     / `MIN_ROBOT_AREA_PX` too high.
   - **False positives** → HSV range too wide, area threshold too low, or
     `BALL_MIN_CIRCULARITY` too low.
   - **Real balls not detected** despite a clean mask → `BALL_MIN_CIRCULARITY`
     is currently `0.85`, which is strict (near-perfect circle). Balls get
     squashed by perspective at oblique angles and aren't perfect spheres
     to begin with, so this is the first thing to loosen — try 0.7. Also
     check `BALL_SIZE_TOLERANCE` isn't rejecting them on size.
   - **Classes confused with each other** → tighten `BALL_SIZE_TOLERANCE`,
     or separate the HSV ranges further.
   - **Piles counted wrong** → adjust `CLUSTER_SEED_FRACTION` and
     `CLUSTER_AREA_FACTOR`.
   - **Positions off by inches** → back to step 3.

## Color tuning (HSV)

OpenCV's HSV: **Hue** is the color itself (0–179 in OpenCV — half the usual
0–360 wheel, so it wraps at 179), **Saturation** is how pure/vivid vs.
washed-out-gray a pixel is, **Value** is brightness. The gray field is
low-saturation by nature, which is what makes color thresholding work here.

| Color | Approx. H range |
|---|---|
| Red | near 0, or near 179 (wraps both ends) |
| Orange | ~10–20 |
| Yellow | ~25–35 |
| Green | ~40–80 |
| Cyan | ~85–100 |
| Blue | ~100–130 |
| Purple/Magenta | ~140–160 |

**Running it:** `python tune_colors.py --label pollen` opens three windows —
`frame` (plain feed), `mask` (white = matches the current range), `result`
(frame with everything outside the mask blacked out) — plus an `HSV tuner`
window with six sliders.

**Slider order that works well:**

1. Raise **S min** first (try 100–150) — this alone usually removes most of
   the gray background.
2. Raise **V min** if shadows/dark background still show up white.
3. Narrow **H min/H max** around the object's hue while watching `mask`
   live, until white appears only where the object is.
4. Fine-tune **S max / V max** only if needed — near 255 is usually fine.

**What "good" looks like:** a single solid white blob shaped like the
object, black elsewhere. Heavy speckle means tighten S/V min; black holes
inside the blob mean the V range is too narrow for glare/shadow — widen it.

**Red:** since red wraps around both ends of the hue range, set **H min
higher than H max** (e.g. min=170, max=10). Both the tuner and the detector
handle wraparound automatically — this matters for `nectar_red`.

**Saving:** press **`s`** once the mask looks clean. Each run saves one
label into `hsv_config.json`, which accumulates all of them.

## Smoothing

Raw positions jitter frame to frame even with good calibration, since every
measurement carries pixel-level noise. `ObjectSmoother` in `tracking.py`
runs a constant-velocity Kalman filter per tracked object (the robot, and
each ball of each class independently) before positions reach the map.

It also lets a track *coast* on its predicted position for up to
`SMOOTHING_MAX_MISSED_FRAMES` frames when nothing detects that object
(brief occlusion, dropped frame), instead of the reading vanishing or
jumping. A coasting point is drawn as a **hollow circle labeled
"(predicted)"**. After that many misses the track is dropped.

Knobs in `config.py`:

- `ENABLE_SMOOTHING` — `False` bypasses smoothing entirely, useful when
  debugging detection and you don't want filter lag in the way.
- `SMOOTHING_PROCESS_NOISE` — how much the filter trusts its motion model
  vs. new measurements. Raise if smoothed positions lag behind fast
  movement; lower for more jitter reduction on steady movement.
- `SMOOTHING_MEASUREMENT_NOISE` — assumed noise per measurement. Raise for
  more smoothing (and more lag); lower if it feels sluggish.
- `SMOOTHING_MAX_MISSED_FRAMES` — coast length, also reused as the
  association radius between frames.
- `HEADING_SMOOTHING` — separate unit-vector smoothing for heading. 0 = raw,
  0.9 = very heavy and laggy.

Camera overlay windows are **not** smoothed — they show raw instantaneous
detections on purpose, which is what you want while tuning. Only the
top-down map and its readouts are smoothed.

## Files

- `config.py` — camera indices/resolution, field size, object classes and
  diameters, detection thresholds, decluster and smoothing tuning. **The
  only file meant for routine hand-editing.**
- `intrinsics.json` — generated by `calibrate_intrinsics.py`.
- `calibration.json` — generated by `calibrate_cameras.py`.
- `hsv_config.json` — generated by `tune_colors.py`.

Don't hand-edit the three JSON files; re-run the tool that makes them so
the values stay tied to real measurements.

- `detection.py` — per-frame color threshold, contour extraction, size
  gating, cluster splitting.
- `tracking.py` — projection to field coordinates, multi-camera fusion,
  heading, Kalman smoothing, map rendering.
- `camera_utils.py` — lens undistortion helpers shared by the tools.
- `main.py` — the live loop.

## Accuracy notes

- **Reprojection error** is printed after calibration. More than ~1 inch
  means re-click with points spread further apart and closer to the field
  edges rather than clustered in the middle.
- **Single camera** means fusion has no redundancy and occluded objects are
  lost for those frames. The math treats the field as a flat plane, so one
  camera works fine — it just has no second opinion when a detection is bad.
- **Elevated objects break the ground-plane assumption.** The homography is
  only valid at z=0, so anything not resting on the tiles will project to
  an incorrect field position. Objects sitting in raised structures need to
  be counted within fixed known regions rather than located geometrically.
- **Ball height parallax**: `BALL_RADIUS_IN` isn't used for a direct
  correction — the ground-contact-pixel approach absorbs most of it. If
  you see a systematic offset on balls from a high, steeply-angled camera,
  modeling this explicitly is the next step.
- **Lens distortion** is corrected for any camera with an `intrinsics.json`
  entry; without one it falls back to raw frames with a warning. Even with
  correction, edge detections are down-weighted during fusion
  (`EDGE_WEIGHT`), since correction is never perfect at the frame border.
