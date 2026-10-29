# setup/

The physical rig, described in four files. These are the only files you edit by hand when something on the table changes.

| File | Describes | Read by |
|---|---|---|
| `camera.yml` | Which camera, its resolution and focus | Python tools |
| `plate.yml` | Marker dictionary, size and where each marker is stuck | Python tools |
| `projector.json` | The projector lens and the calibrated position of `cam1` | TouchDesigner |
| `model.json` | Which model in `models/` is on the plate | TouchDesigner, simulation |

Calibration generated from these lives in `calibration/<camera name>/`.

## What to do when something changes

Run commands from `tools/` as `python -m <command>`. `checks.status` shows what is out of date at any point.

| Change | Steps |
|---|---|
| Camera moved | `calibration.set_origin` (plate on its home spot, all markers visible) |
| Focus or resolution changed | [Calibrate the lens](#calibrating-the-lens), then `calibration.set_origin` |
| Different camera | New `name`, `index`, `width`, `height` in `camera.yml`; check it with `checks.live_view`; [calibrate the lens](#calibrating-the-lens); `calibration.set_origin` |
| New marker size or dictionary | See [New markers](#new-markers) |
| A marker was re-stuck | `checks.plate_fit --write`, then `calibration.set_origin` |
| Projector moved | Recalibrate `cam1` in TouchDesigner and save it (see `touchdesigner/README.md`) |
| Different projector | See [Different projector](#different-projector) |
| Different 3D model | See `models/README.md`. No recalibration needed. |

Each camera keeps its own folder in `calibration/`, so switching back to one is just changing `name` back.

## Calibrating the lens

1. Print the checkerboard (`calibration.make_checkerboard`) and measure one square.
2. Run `calibration.camera_lens`. Press `s` to open the camera settings and focus on the table.
3. Hold the board in view and press SPACE when the corners light up. Take 20–30 shots: near and far, tilted up to ~45°, covering every part of the image (especially corners and edges). Press `q` when done.
4. Copy the focus value it prints into `focus:` in `camera.yml` and keep `autofocus: 0`.

A reprojection error under 0.5 px is good; over 1.0, retake.

## New markers

1. Set `markerSizeMm` (and `dictionary`) in `plate.yml`.
2. `calibration.make_markers`, then print from `tools/output/` at 100% scale.
3. Measure the printed black square and put the **measured** size in `markerSizeMm`.
4. Stick the markers outside the model's footprint, roughly measure each centre from the plate centre, and update `markers:`.
5. `checks.plate_fit --write` refines the positions with the camera. Aim for under ~1 mm per marker.
6. `calibration.set_origin`

## Different projector

Update `name`, `throwRatio` and `horizontalFovDeg` in `projector.json` from the projector's specs. For a fixed lens, horizontal FOV = 2·atan(1 / (2 · throw ratio)). Set `cam1` to `null`, then recalibrate and save as for a moved projector.
