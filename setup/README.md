# setup/

**The physical rig, described in four files.** These are the only files you edit by hand when something on the table changes. They are in git, so the whole team (and every program) sees the same rig.

| File | Describes | Read by |
|---|---|---|
| `camera.yml` | which camera is used, its resolution, whether it's mirrored | Python tools, KinectReader |
| `plate.yml` | the ArUco plate: dictionary, marker size, where each marker is stuck | Python tools, KinectReader |
| `projector.json` | the projector lens (fixed) and where `cam1` was calibrated to | TouchDesigner |
| `model.json` | which model in `models/` is on the plate | TouchDesigner, simulation |

The `.yml` files are in OpenCV's YAML flavour because the C++ tracker reads them with `cv::FileStorage`. The `.json` files are read by TouchDesigner, which reads JSON without extra packages.

What gets **generated** from these (lens numbers, world origin) lives in `calibration/<camera name>/`, not here.

## What to do when something changes

Run everything from `tools/`. `python -m checks.status` tells you what is missing or out of date at any point.

### The camera moved (same camera)
1. `python -m calibration.set_origin` with the plate on its home spot.

Nothing to edit. The lens numbers survive a move.

### A different camera (e.g. the 4K webcam)
1. In `camera.yml`: a new `name` (e.g. `webcam_4k`), `source: "webcam"`, its `index`, `width`, `height`, `mirrored: 0`, `autofocus: 0`.
2. `python -m checks.live_view` to see that it opens at the right resolution.
3. `python -m calibration.camera_lens` → `calibration/webcam_4k/intrinsics.yml`
4. `python -m calibration.set_origin` → `calibration/webcam_4k/world_origin.yml`

The Kinect's files stay in `calibration/kinect_v2/`, so switching back is just changing `name` and `source` back.
Note: `KinectReader.exe` only drives a Kinect. A webcam works with the Python tools; live tracking from a webcam needs a Python (or C++ OpenCV) tracking loop.

### Same camera, new resolution
Treat it as a new lens: `python -m calibration.camera_lens`, then `set_origin`. The tools warn when the frames don't match the calibration's resolution.

### New marker size (or dictionary)
1. In `plate.yml`: `markerSizeMm` (and `dictionary`) to what you want.
2. `python -m calibration.make_markers` → print from `tools/output/` at 100% scale.
3. Measure the printed black square, put the **measured** size in `markerSizeMm`.
4. Stick them on, measure each centre from the plate centre, update `markers:`.
5. `python -m checks.live_view`: fit error under ~1 px means the numbers are right.
6. `python -m calibration.set_origin`

### A marker moved or was re-stuck
Steps 4–6 above.

### The projector moved
Recalibrate `cam1` in TouchDesigner, then save it into `projector.json` (see `touchdesigner/README.md`). The FOV never changes unless the projector itself does.

### A different 3D model
See `models/README.md`. The plate and markers don't change, so nothing needs recalibrating.
