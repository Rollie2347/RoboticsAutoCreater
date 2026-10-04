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

**How the expected size is computed.** A sphere looks equally wide in every
image direction, but a length drawn *on the field* shrinks along the
camera's viewing direction (by roughly sin(elevation)). So the prediction
uses the local Jacobian of the field→pixel mapping and takes its **larger
singular value** — the sideways, un-shrunk scale — times the ball diameter.
An earlier version averaged the field-X and field-Y lengths, which
under-predicted ball size by 10–70% and made single balls read as `x2`/`x3`
at shallow angles. With the current method a real ball measures 1.02–1.18×
its prediction at every angle tested.

### Declustering

Balls that touch each other merge into a single contour. When a blob's area
reaches `CLUSTER_AREA_FACTOR` × the expected area of one ball at that
position, it's treated as a cluster and split:

1. A distance transform of the blob peaks near the center of each ball.
2. Seeds are the **local maxima** of that distance map (max within a window
   of `CLUSTER_PEAK_WINDOW` × ball radius, and at least
   `CLUSTER_SEED_FRACTION` × ball radius deep).
3. Seeds closer together than `CLUSTER_MIN_SEED_SEP` ball diameters are
   merged. This distance is measured in **field inches** (seeds projected
   through the homography), not pixels — see below.
4. Every blob pixel is assigned to its nearest seed. Balls in a cluster are
   the same size, so the correct cut between two touching balls is the
   perpendicular bisector of their centers, which is exactly what
   nearest-seed assignment produces.
5. The cluster's total count comes from `area / expected_area` for the
   whole blob. Each piece counts at least 1, and any remainder goes to the
   pieces with the most unaccounted area — so a piece hiding a second ball
   behind it reports `x2`.

If fewer than two seeds are found, the blob is reported as a **single
detection with a count** (e.g. `pollen x3`). Counts stay right even when
individual positions in a dense pile can't be resolved.

Why it's built this way (each was a measured failure of the first version):

- **Local maxima, not one global threshold.** Between two touching balls
  the blob's waist is still ~2/3 as deep as a ball center (~0.85 after the
  mask's 7×7 morphological close), because of perspective and
  anti-aliasing. A single threshold on the distance map merged neighbouring
  seeds almost every time — the original split succeeded in 12 of 120 test
  scenes.
- **Field-inch separation.** Along the viewing direction, ball spacing in
  pixels shrinks by ~sin(elevation), so no pixel threshold works at every
  camera angle. All seeds are ball centers at the same height, so the
  ground homography distorts them uniformly (a few %) and their spacing in
  inches stays reliable. Touching balls are 1.0 diameter apart; the filled
  gap in the middle of a 2×2 square is 0.71 — `CLUSTER_MIN_SEED_SEP = 0.85`
  sits between.
- **Nearest-seed instead of `cv2.watershed`.** OpenCV's watershed floods by
  color gradient, which is flat inside a binary mask, so it added nothing.

Knobs: `ENABLE_DECLUSTER`, `CLUSTER_AREA_FACTOR`, `CLUSTER_SEED_FRACTION`,
`CLUSTER_PEAK_WINDOW`, `CLUSTER_MIN_SEED_SEP` (and
`CLUSTER_MIN_SEED_SEP_PX`, the pixel fallback used only without a
homography), `CLUSTER_MAX_SPLIT`. The current values sit in the middle of a
stable plateau from a parameter sweep; change them only if real piles are
miscounted.

**Where splitting works.** Counts are reliable almost everywhere, but
*individual positions* need balls roughly ≥17 px across and a camera
≥~45° above the field. Below ~30° elevation, balls behind other balls are
physically hidden; past ~170 in from the camera, balls are ~13 px and too
small to split. Put the camera high and close for the best results.

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
if the patch isn't found that frame.

Direction survives marker height because both markers are at the same
height: projecting them through the ground homography scales both toward
the camera by the same factor, which preserves the direction between them.
Tested heading error is mean 0.1–0.2°, max under 1° (markers from 0.5 in to
12 in high, 20–75° camera elevation).

Other details that matter:

- **Markers have their own minimum area**, `MIN_MARKER_AREA_PX` (60). A 3 in
  marker 12 ft away is only ~100–300 px, so the robot patch's 500 px
  minimum silently filtered markers out and heading never appeared.
- **Flat patches use their centroid**, not the bottom of their bounding
  box. The bottom-edge rule is for balls; on a flat patch it shifted the
  robot ~half a patch width toward the camera (~4.5 in on an 8 in patch).
- **Tags cut off by the frame edge are ignored.** If the `robot` patch or a
  marker touches the frame border, part of it is missing, so its centroid
  shifts inward (measured: ~1.7 in off with just 4 px cut). The tracker
  treats it as not seen that frame and the smoother coasts instead.

Heading is smoothed by averaging the heading *unit vector*
(`HEADING_SMOOTHING`), which avoids the wraparound bug you get from
averaging raw angles across 0/360°. Keep it light (default **0.2**): it's
applied per frame, so it adds lag while turning — 0.6 trailed a 180°/s
spin by ~9°, 0.2 by under 2°. The raw two-marker heading is already
accurate to under 1°, so heavy smoothing only costs lag.

### Tag height correction

The homography only knows the floor. A patch `z` inches above the tiles is
seen along a ray that hits the floor *past* it, so it lands too far from the
camera by about `z / tan(elevation)` — e.g. 4 in up at 30° ≈ 7 in off, and
worse toward the far side and frame edges where the camera looks flatter.

The tracker removes this exactly once it knows where the camera is:

1. At startup it runs `cv2.solvePnP` on the points you clicked in
   `calibrate_cameras.py` (saved in `calibration.json`) with the lens matrix
   from `intrinsics.json`, and prints the camera's solved position and
   height. **Check that height with a tape measure** — it's a free test of
   your whole calibration.
2. Each tag/marker position is pulled back along its ray toward the point
   under the camera by `(h_cam − z) / h_cam`. That's exact for a pinhole
   camera.

To turn it on, measure tile-to-patch-surface with a ruler and set
`ROBOT_TAG_HEIGHT_IN` (the `robot` patch) and `ROBOT_MARKER_HEIGHT_IN`
(the heading markers) in `config.py`. Both default to 0 = no correction.
It needs `intrinsics.json`; without it you get a startup note and no
correction.

Simulated (both camera placements, tags 0.5–12 in high, every tile corner
the camera can see, 4 headings each): camera solved within 0.2 in, position
error ≤ 0.27 in and heading ≤ 0.75° everywhere, versus 1.2–13 in of error
without the correction. Measuring the tag height 0.25 in wrong adds
~0.3 in of error; 0.5 in wrong still stays under 1 in.

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

Then measure and set `ROBOT_TAG_HEIGHT_IN` / `ROBOT_MARKER_HEIGHT_IN` (see
"Tag height correction"), and:

```bash
python main.py                  # live tracking
python main.py --log run1.csv   # same, plus one CSV row per frame
```

Press `q` in any window to quit.

## Frame rate and logging

**Camera format.** `main.py` asks the webcam for compressed **MJPG**
(`CAPTURE_FOURCC` in `config.py`) before setting the resolution.
Uncompressed YUY2 at 1280×720 needs ~55 MB/s at 30 fps — more than USB 2
carries — so most webcams quietly drop to ~10 fps in that mode. At startup
it prints the size, fps and format it asked for; the log shows what the
camera actually delivers.

**Live readout.** The field map shows the current FPS (average of the last
30 frames), in red when it's under `MIN_TARGET_FPS` (default 20).

**CSV log.** `python main.py --log run1.csv` writes one row per frame:

| Column | Meaning |
|---|---|
| `frame` | frame number |
| `wall_time_s` | Unix time — for lining up with other logs (e.g. Pinpoint over UDP) |
| `mono_time_s` | monotonic clock — for intervals; never jumps |
| `dt_ms` | time since the previous frame |
| `raw_x_in`, `raw_y_in`, `raw_heading_deg` | fused pose before smoothing |
| `x_in`, `y_in`, `heading_deg` | smoothed pose (what the map shows) |
| `coasting` | 1 if nothing saw the robot this frame and the smoother predicted it |
| `count_<class>` | element counts, as on the map |

Blank pose cells mean no robot that frame. When you quit it prints a
summary: mean/median fps, slowest interval, 95th percentile, and how many
intervals were slower than `MIN_TARGET_FPS`.

**Processing cost** measured on a 2-core cloud VM with a busy frame (robot +
20 elements, all six labels, lens undistortion, map): median 26 ms per frame
(~38 fps ceiling) — detection 20 ms, fuse + smooth 4 ms, map 2 ms. Your
laptop's number is in the log; if it's under 20 fps with MJPG confirmed,
processing is the bottleneck.

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

`calibrate_cameras.py` run it
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

    **`python tune_colors.py --label robot`**          # required (robot position)
    **`python tune_colors.py --label robot_front`**       # optional (heading)
    **`python tune_colors.py --label robot_back`**        # optional (heading)
    **`python tune_colors.py --label pollen`**            # yellow
    **`python tune_colors.py --label nectar_red`**        # red
    **`python tune_colors.py --label nectar_blue`**       #blue
5. **`python main.py`** — judge whether it all worked together:
   - Detection boxes should track objects tightly and stay locked on.
   - The map's dot should match reality — put the robot on a known tile
     corner and check the map agrees.
   - The heading arrow should point where the robot's front actually is.
   - **Flickering detection** → HSV range too tight, or `MIN_BALL_AREA_PX`
     / `MIN_ROBOT_AREA_PX` / `MIN_MARKER_AREA_PX` too high.
   - **False positives** → HSV range too wide, area threshold too low, or
     `BALL_MIN_CIRCULARITY` too low.
   - **Real balls not detected** despite a clean mask → check
     `BALL_MIN_CIRCULARITY` (now `0.70`). A ~20 px circle only scores
     0.80–0.91 because its stair-stepped pixel edge inflates the perimeter;
     the old `0.85` dropped about a third of perfect balls. Also check
     `BALL_SIZE_TOLERANCE` isn't rejecting them on size, and that distant
     balls clear `MIN_BALL_AREA_PX` (now 60; a 2.8 in ball ~17 ft away is
     ~110 px).
   - **Heading never appears** → both markers must be tuned *and* clear
     `MIN_MARKER_AREA_PX`.
   - **Position off by several inches, more at the far side** → set
     `ROBOT_TAG_HEIGHT_IN` / `ROBOT_MARKER_HEIGHT_IN` and check the camera
     height printed at startup against a tape measure.
   - **Heading arrow lags in turns** → lower `HEADING_SMOOTHING`.
   - **FPS shown in red** → check the startup line says `'MJPG'`; if it does,
     processing is the limit (see "Frame rate and logging").
   - **Robot vanishes near the HIVE** → it's in the HIVE's shadow from the
     camera; see "Accuracy notes".
   - **Classes confused with each other** → tighten `BALL_SIZE_TOLERANCE`,
     or separate the HSV ranges further.
   - **Piles counted wrong** → first check the camera placement limits under
     "Declustering"; then adjust `CLUSTER_MIN_SEED_SEP`,
     `CLUSTER_PEAK_WINDOW`, `CLUSTER_AREA_FACTOR`.
   - **Positions off by inches** → back to step 3.

## Field test checklist

Run these at the field after setup, and again at each venue. Every test is
recorded with `main.py --log` and graded with `check_log.py`, which prints
PASS/FAIL against the targets below plus a hint when something fails.
Record each test in its own CSV and name it after what you did, e.g.
`still_24_48_90.csv`, so you can re-grade later.

### Before testing (once per camera setup)

Do these in order. Each one catches a problem that would otherwise show up
as a confusing failure later.

1. **Lock the camera down.** Tripod or pole in its final spot, zoom/focus
   fixed, cable taped. Put tape marks under the tripod feet. If the camera
   moves even slightly after step 3, every position is off; redo step 3.
2. **Lens calibration** — `python calibrate_intrinsics.py` (15+ captures
   covering the corners of the frame; RMS under ~0.5 px).
3. **Field calibration** — `python calibrate_cameras.py`. Click 6–8 tile
   corners spread over the whole image, not bunched in the middle, and away
   from the HIVE. Mean reprojection error should be under ~0.5 in.
   Write down which corner is (0,0) and which way +X and +Y go; every
   `--truth` value below uses those axes, and heading 0° = pointing along
   +X, 90° = along +Y.
4. **Measure the tag heights** with a ruler, tile to the top surface of each
   patch, and set `ROBOT_TAG_HEIGHT_IN` and `ROBOT_MARKER_HEIGHT_IN` in
   `config.py`. Measure to the nearest ¼ in; ½ in wrong still passes, 1 in
   wrong usually won't.
5. **Tune colors under the field's real lighting** — all labels you use
   (`tune_colors.py --label ...`). Tuning at home and testing at the field
   is the most common cause of flicker and missed balls.
6. **Check the camera height.** Start `python main.py` and read the line
   `Camera N: solved position (...), H in above the tiles`. Tape-measure from
   the tile to the lens. **They should agree within ~1 in.** If they don't,
   the calibration is off: redo step 3 (and step 2 if it's still wrong).
   This one check validates the whole calibration chain.
7. **Map the HIVE blind zone.** Log while you push the robot slowly in a
   loop around the HIVE, about a foot away from it:
   `python main.py --log blindzone.csv`. Watch the map: where the robot dot
   turns hollow ("predicted") and then disappears, the camera can't see it.
   Put tape on the tiles at the edges of that zone. Then
   `python check_log.py blindzone.csv --moving` lists where each gap
   started. Expect roughly a quarter of the field (see "Accuracy notes").
   Don't run the tests below inside the taped zone — they'll fail for a
   reason code can't fix.

**Placing the robot precisely.** Mark the center of the `robot` patch on
the robot, and use a tile corner as the target: stand over it and line the
mark up with the corner (a small square or a plumb line helps). For heading,
push one side of the robot flush against a straightedge laid along a tile
seam. A robot placed ½ in off will read ½ in off — placement is usually the
biggest error in this test.

### Test 1 — Frame rate

- **Do:** `python main.py --log fps.csv`, leave it running 30 s with the
  robot and some balls in view, quit with `q`.
- **Check:** `python check_log.py fps.csv`
- **Pass:** mean ≥ 20 fps and ≤ 5% of frame gaps slower than 20 fps. Every
  other log you record is also graded for frame rate, so watch this line
  throughout — especially with lots of balls on the field.
- **If it fails:** the startup line must say `'MJPG'`. If it does, close
  other programs (browser, Zoom, OBS) and retest. Still failing → lower
  `FRAME_WIDTH`/`FRAME_HEIGHT` (e.g. 960×540) and redo calibration steps
  2–3, since both depend on resolution.

### Test 2 — Standing still (position ≤ 1 in, heading ≤ 2°)

- **Where:** 5 tile corners outside the blind zone: one near the camera, one
  on the far side, one near the **left edge of the image**, one near the
  **right edge**, one next to the HIVE on the camera side. Edges matter most:
  that's where lens and height errors grow.
- **Do:** at each corner, for each heading 0°, 90°, 180°, 270°: place the
  robot, start `python main.py --log still_X_Y_H.csv`, wait 5 s, quit.
  That's 20 short logs.
- **Check:** `python check_log.py still_24_48_90.csv --truth 24,48,90`
- **Pass:** position ≤ 1 in and heading ≤ 2° on all 20.
- **Also:** each run prints `Raw position noise` and a suggested
  `SMOOTHING_MEASUREMENT_NOISE`. Use the typical value across your 20 runs
  (it replaces the 1.0 guess; it's floored at 0.2).
- **If it fails:**
  - Error points *away from the camera* and grows toward the far side →
    tag height too low in config, or `intrinsics.json` missing.
  - Error points *toward* the camera → tag height set too high.
  - Same error direction and size everywhere → calibration clicks; redo
    step 3.
  - Only fails near the image edges → redo lens calibration with more
    captures at the frame corners.
  - Heading off by ~180° → `robot_front` and `robot_back` are swapped.
  - Heading missing → a marker is out of view, touching the frame edge,
    or needs retuning.

### Test 3 — Moving (smooth, no jumps or dropouts)

- **Do:** start `python main.py --log drive.csv`, then:
  1. Park 3 s.
  2. Drive a straight 72 in (3 tiles) along a seam at normal speed.
  3. Stop, park 2 s.
  4. Turn 90° in place (about the tag's center, so it doesn't shift).
  5. Park 3 s, quit.

  Repeat once driving *toward/away from* the camera and once *across* its
  view.
- **Check:**
  - `python check_log.py drive.csv --moving` — smoothness for the whole
    run.
  - `python check_log.py drive.csv --truth X,Y,H --skip T` — how accurate
    the final parked spot is, where `T` is a time after you stopped (the
    `--moving` output shows run times).
- **Pass:**
  - no gap over 250 ms;
  - no position jump over 1 in;
  - no heading jump over 15°;
  - the printed start → end net distance within 1.5 in of the real
    distance (72 in for the line; an in-place turn adds nothing);
  - final parked position ≤ 1 in and heading ≤ 2°.
- **Watch the map live:** the dot should glide, not step; the arrow should
  swing smoothly through the turn and not trail behind it.
- **If it fails:**
  - Gaps → the check prints where each started. Near the HIVE → blind zone;
    at the image edge → tag leaving the frame.
  - Jumps while fully in view → a marker or tag flickering; retune its
    colors.
  - The arrow lags the turn → lower `HEADING_SMOOTHING`.
  - Position trails behind on fast moves → raise `SMOOTHING_PROCESS_NOISE`.

### Test 4 — Around the HIVE

- **Do:** log while driving a full loop around the HIVE about 6 in from it:
  `python main.py --log hive.csv`, then
  `python check_log.py hive.csv --moving`.
- **Pass:** the gaps listed only start where you taped the blind zone in
  setup step 7. On the camera side and along the flanks up to the tape:
  no gaps, no jumps.
- **Also check:** when the robot comes back out of the shadow, the dot
  reappears in the right place within a frame or two (no leftover
  "predicted" dot stuck behind the HIVE).
- **If gaps appear outside the tape:** the HIVE's arms or cells are hiding
  the tag at some angles; widen the taped zone, or raise the camera.

### Test 5 — Clumps (count right, or off by at most 1)

Set up each layout, log 10 s, and grade it. Do them once near the camera and
once on the far side (outside the blind zone):

| Layout | How | Check |
|---|---|---|
| Pile of 5 | 5 pollen pushed together into a tight pile | `--count pollen=5` |
| Line of 5 | 5 pollen touching in a straight line | `--count pollen=5` |
| 3 against a wall | 3 touching pollen against a side wall; then the far wall; then the wall nearest the camera | `--count pollen=3` |
| Stacked | 4 in a square with 1 on top | `--count pollen=5` (expect 4) |
| Other colors | a pile of 3 `nectar_red`, then 3 `nectar_blue` | `--count nectar_red=3` etc. |

- **Check:** `python check_log.py pile5.csv --count pollen=5`
- **Pass:** most frames within ±1 and ≥ 95% of frames within ±1.
- **Known limits:**
  - Stacked piles read one short.
  - Balls against the wall nearest the camera only count if that panel is
    see-through on camera.
  - Anything in the blind zone reads 0.
- **If it fails:**
  - Count 0 → the balls are hidden, or the colors need retuning in this
    light.
  - Too high → stray same-color blobs; check the mask in `tune_colors.py`.
  - Too low in piles only → note the layout and send the log.

### Test 6 — Venue / lighting check

At competition (or after changing the practice lights), after steps 1–6
above: repeat **one** Test 2 pose near the image edge and the Test 5 pile of
5. If both pass, the rest of the setup carried over.

### Results sheet

Copy this into your notes and fill it in each session:

| Test | Log file | Result | Numbers (err / fps / count) | Notes |
|---|---|---|---|---|
| Camera height check | — | | solved __ in vs taped __ in | |
| Blind zone taped | blindzone.csv | | | |
| 1 Frame rate | fps.csv | | | |
| 2 Still ×20 | still_*.csv | | worst __ in / __° | noise __ in |
| 3 Line + turn | drive*.csv | | | |
| 4 HIVE loop | hive.csv | | | |
| 5 Clumps | pile5, line5, wall3_* | | | |

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
- `SMOOTHING_PROCESS_NOISE` — typical unmodeled acceleration, in
  **in/s²** (default 200; FTC drivetrains commonly hit 100–250). Raise if
  smoothed positions lag behind hard stops and turns; lower for more
  jitter reduction.
- `SMOOTHING_MEASUREMENT_NOISE` — per-frame position noise in **inches**
  (1 std dev). The default 1.0 is a placeholder: **measure it** by parking
  the robot, setting `ENABLE_SMOOTHING = False`, and taking the standard
  deviation of its position over ~10 s.
- `SMOOTHING_MAX_MISSED_FRAMES` — coast length. (Matching new detections to
  existing ball tracks uses `FUSION_MATCH_RADIUS_IN`.)

The process noise is the standard constant-velocity (white-noise
acceleration) model, rebuilt from the real frame `dt` every frame, so
smoothing behaves the same at 15 fps or 60 fps. A new track starts with
position known to measurement accuracy and velocity unknown (±100 in/s),
so a robot already moving when first seen locks on quickly. Coasting ball
tracks keep their cluster count (`x3` stays `x3`).

The first version added a fixed `5·I` process noise directly to position
every frame, which made the filter trust each raw measurement ~90% — it
only cut jitter by ~14%. Measured with 1 in measurement noise:

| Scenario | Raw | Old filter | Current |
|---|---|---|---|
| Stationary | 1.40 in | 1.21 in | 0.86 in |
| Straight line, 40 in/s | 1.47 in | 1.27 in | 0.94 in |
| Circle, 60 in/s | 1.42 in | 1.29 in | 1.03 in |
| 5-frame dropout (coast error) | — | 3.66 in | 0.31 in |
| Hard stop from 50 in/s (overshoot) | — | 1.15 in | 1.29 in |
- `HEADING_SMOOTHING` — separate unit-vector smoothing for heading. 0 = raw,
  0.9 = very heavy and laggy. Default 0.2 (see "Robot heading" for why it's
  light).

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
- `main.py` — the live loop; `--log FILE.csv` writes per-frame timestamps,
  pose and counts, and prints a frame-rate summary on exit.
- `check_log.py` — grades a `--log` CSV against the field test targets
  (frame rate, `--truth X,Y,H`, `--moving`, `--count CLASS=N`).

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
  correction — the ground-contact-pixel approach absorbs most of it.
  Measured ball position error is ~0.9 in median, worst ~1.5 in (one ball
  radius) from a steep (75°) camera, where the bottom of the silhouette is
  no longer the contact point. Split cluster balls are about as accurate
  (median ~1 in). Modeling this explicitly needs camera pose.
- **Elevated robot patch**: corrected once `ROBOT_TAG_HEIGHT_IN` is set and
  `intrinsics.json` exists — see "Tag height correction". Without that,
  position error is about `patch height / tan(camera elevation)`.
- **The HIVE blocks about a quarter of the field.** It's 49.5 × 39 in at the
  base and 44 in tall at field center, so from one camera everything in its
  shadow is invisible: in simulation ~23 of 85 grid spots from an 8 ft
  tripod at the side wall, ~25 of 95 from a 10 ft pole at a corner. A robot
  driving into the shadow coasts on its prediction for
  `SMOOTHING_MAX_MISSED_FRAMES` frames (hollow "predicted" dot), then drops
  off the map until it comes out; clumps in the shadow read 0. Check your
  real blind zone before trusting a count or position behind the HIVE; only
  a second camera removes it.
- **Walls.** Balls against a wall between the camera and the field are
  visible only if the panel is see-through on camera. With a clear panel
  3-against-the-wall read exactly 3; treated as opaque they read 0.
- **Lens distortion** is corrected for any camera with an `intrinsics.json`
  entry; without one it falls back to raw frames with a warning. Even with
  correction, edge detections are down-weighted during fusion
  (`EDGE_WEIGHT`), since correction is never perfect at the frame border.
- **Camera placement**: high and close is best. Below ~30° elevation,
  balls hidden behind other balls can't be counted; beyond ~170 in, clusters
  can't be split into individual positions (counts still mostly hold).

## Verification log

### 2026-09-18 — first executed test pass

Everything from the Kalman work onward had only been reviewed by
inspection. It was run for the first time against a simulated pinhole
camera (1280×720, f≈900 px) that renders true sphere silhouettes and flat
marker patches, at 15–75° elevation and 100–220 in range. The test harness
lives outside this project; nothing in the project uses simulated data.
All files pass `py_compile`. Originals from before these fixes are in
`_backup_before_test_fixes/`.

**Bugs found (root cause → fix):**

| # | Symptom | Root cause | Fix |
|---|---|---|---|
| 1 | Heading never appeared | Markers used `MIN_ROBOT_AREA_PX` (500); real markers are ~100–300 px | New `MIN_MARKER_AREA_PX = 60` |
| 2 | Clusters almost never split (12/120 scenes); every count came from the area fallback, with one position per pile | Global distance-map threshold can't separate balls whose waist is 2/3–0.85 as deep as their centers | Local-maximum seeds, field-inch separation, nearest-seed split, blob-level count allocation |
| 3 | Single balls read as `x2`/`x3` below ~35° | Expected size averaged field-X/Y lengths, one of which shrinks along the viewing direction | Larger singular value of the local field→pixel Jacobian |
| 4 | ~⅓ of perfect balls dropped | `BALL_MIN_CIRCULARITY = 0.85`; a ~20 px circle scores 0.80–0.91 | 0.70 |
| 5 | Smoothing cut jitter only ~14% | Fixed `5·I` process noise on position every frame, independent of dt | dt-scaled constant-velocity process noise, physical units |
| 6 | Robot ~4.5 in toward camera | Flat patches used bbox bottom (ball rule) | Centroid for robot patch and markers |
| 7 | Distant pollen missed | `MIN_BALL_AREA_PX = 150` above a 2.8 in ball's area at ~17 ft | 60 (size gating handles noise) |
| 8 | `x3` became `x1` while coasting | `KalmanTrack` didn't store count | Track carries count |

**Results, original → fixed:**

| Test | Original | Fixed |
|---|---|---|
| Cluster scenes with correct count (216) | 63 | 164 |
| Clusters fully split into individual balls | 0 | 73 |
| Single balls, correct (280 cases) | — | 280/280, none split in two |
| Heading error | no heading | mean 0.1–0.2°, max < 1° |
| Kalman stationary / moving / 5-frame coast | 1.21 / 1.27 / 3.66 in | 0.86 / 0.94 / 0.31 in |

**Still open (known, not changed):**

- A large same-colored non-ball object skips shape/size checks on the
  cluster path and can appear as e.g. `pollen x12`.
- A ball that's picked up and moved leaves a phantom at the old spot for
  `SMOOTHING_MAX_MISSED_FRAMES` frames (coasting points count toward
  totals).
- `HEADING_SMOOTHING` is per frame, so its lag depends on frame rate
  (~4–7° behind at 90°/s spin at 30 fps).
- `SMOOTHING_MEASUREMENT_NOISE` still needs measuring on the real setup.
- Nothing has run on real camera footage yet. Next: measure the
  checkerboard → `calibrate_intrinsics.py` → re-run
  `calibrate_cameras.py` → tune labels → test real piles and the heading
  arrow.

### 2026-09-21 — acceptance checks (simulated)

Your four field checks, run against a full-field simulator: two camera
placements (8 ft tripod 2 ft outside the middle of a side wall; 10 ft pole
3 ft outside a corner), 70° lens, ±1 px calibration clicks on every visible
tile corner, per-frame sensor noise, the HIVE as a solid 49.5 × 39 × 44 in
block at center (worst case — the real frame is partly open), 12 in walls.
Tag heights swept 0.5–12 in. Numbers below are after the fixes listed
further down. These say the *code* can meet each target; your field run
with the real camera is still what confirms it.

| Check | Target | Result (where the robot/balls are visible) |
|---|---|---|
| Standing still, every visible tile corner × 0/90/180/270° | ≤ 1 in, ≤ 2° | ≤ 0.27 in, ≤ 0.75° (tag height set correctly); edges of frame included |
| Straight 72 in line, 40 in/s | smooth, no jumps/dropouts | RMS 0.24–0.31 in, no dropouts; largest step error ~0.7 in, at an instant 0→40 in/s start the sim uses |
| 90° spin (180°/s) and 90° arc | smooth | heading within 1.5–1.75° during the spin, ≤ 1.3° on the arc; position RMS ≤ 0.3 in |
| Next to the HIVE (camera side / flanks) | no dropouts | none on the camera side; on a flank the robot drops out exactly where it enters the HIVE's shadow |
| Behind the HIVE | — | not visible from one camera: coasts 10 frames, then off the map until it reappears |
| Pile of 5 / line of 5 / 3 on a wall | within ±1 | 40/40 visible cases within ±1, 21/40 exact; 4+1 stacked pyramid always reads 4; hidden clumps read 0 |
| Frame rate | ≥ 20 fps | processing ~38 fps ceiling on a 2-core VM; camera fps unverified until you log a real run |

**Bugs found and fixed:**

| # | Symptom | Root cause | Fix |
|---|---|---|---|
| 1 | Robot 1.2–13 in off unless the tag is flat on the tiles | Raised tag projected through the floor homography with no height correction | Camera pose from `solvePnP` on the saved calibration clicks + `intrinsics.json`; tag/marker positions scaled by `(h − z)/h`. New `ROBOT_TAG_HEIGHT_IN`, `ROBOT_MARKER_HEIGHT_IN` |
| 2 | Map counted a pile of 5 as 3 (and any split pile as one ball) | `fuse_balls` merged detections from the *same* camera within 8 in and kept `max(count)` — undoing declustering before the map | Never merge two detections from one camera |
| 3 | Heading ~9° behind during a 180°/s turn | `HEADING_SMOOTHING = 0.6` is per-frame exponential smoothing | 0.2 (under 2° lag) |
| 4 | Robot ~1.7 in off when the tag is partly outside the frame | Clipped blob → centroid shifted inward | Tags/markers touching the frame border are treated as not seen |
| 5 | Probably ~10 fps with many webcams | Capture never requested MJPG, so DirectShow may pick uncompressed YUY2 | `CAPTURE_FOURCC = "MJPG"` set before the resolution |
| 6 | No way to check frame rate | No timestamp logging | `main.py --log`, live FPS on the map, exit summary |

**Tried and rejected:** learning one ball's blob area from isolated balls
to fix 5 → 6 overcounts. It made tight piles undercount (balls overlap on
screen, so each adds less area than an isolated one) and put a stacked pile
off by 2. Within ±1 went from 40/40 to 38/40, so it was reverted.

**Still open:**

- The HIVE blind zone (about a quarter of the field from one camera).
- Clumps of 5 are exact only about half the time (always within ±1).
- A 4+1 stacked pile reads 4; the top ball hides mostly behind the others.
- Nothing has run on real footage yet.

