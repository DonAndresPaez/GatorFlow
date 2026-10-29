# GatorFlow

Real-time aerodynamic projection mapping. A webcam tracks a marker plate with a 3D-printed model on it, a simulation computes the pressure over the model, and TouchDesigner projects that heatmap back onto the physical object, following it as it moves.

```
webcam ──► tools/tracking/tracker.py ──OSC /car/transform──► TouchDesigner (port 9000) ──► projector
                                                    └──────► simulation    (port 9001)
```

## Repository layout

| Folder | Contents |
|---|---|
| `setup/` | The physical rig: camera, marker plate, projector, active model. Edit these when the hardware changes. |
| `calibration/` | Lens and origin files, generated per camera (not in git). |
| `models/` | One folder per 3D model: STL + `model.json`. |
| `tools/` | Python: the live tracker, calibration, checks and tests. |
| `touchdesigner/` | The TouchDesigner loader script. |

## Quick start

Needs Python 3.10+ and a USB webcam (we use a Logitech MX Brio on a USB 3 port).

```powershell
cd tools
pip install -r requirements.txt
pytest                      # runs the tests and checks that setup/ is valid
python -m checks.status     # shows the rig and anything that still needs calibrating
```

## Running a session

From `tools/`:

```powershell
python -m checks.live_view    # 1. confirm the plate is detected (q to quit)
python -m tracking.tracker    # 2. track the plate and send its pose
```

Then open the TouchDesigner project in `touchdesigner/`. Only one program can use the camera at a time, so close `live_view` before starting the tracker.

Tracker options: `--no-view` hides the preview, `--detect-width 1280` raises a low frame rate, `--alpha 0.3` steadies a jittery pose (lower is steadier but laggier; default 0.5).

No camera? `python -m checks.fake_tracker` sends test poses so TouchDesigner and the simulation can be developed without one.

## The pose message

This is the contract between the tracker, TouchDesigner and the simulation.

| | |
|---|---|
| Transport | OSC over UDP to `127.0.0.1`: port `9000` (TouchDesigner) and `9001` (simulation) |
| Address | `/car/transform` |
| Values | `tx, ty, tz` in millimetres, `rx, ry, rz` in degrees (applied X, then Y, then Z) |
| Origin | The plate's home spot on the table, recorded by `calibration.set_origin` |
| Rate | Every frame the plate is seen; nothing while it's lost, so receivers hold the last pose |

Ports and address are set in `tools/common/config.py`. In TouchDesigner the pose drives `geo1` only, never the projector camera `cam1`.

## Tools

Run from `tools/` as `python -m <command>`.

| Command | Purpose |
|---|---|
| `tracking.tracker` | The live tracker: camera → plate pose → OSC |
| `calibration.make_checkerboard` | Checkerboard to print for lens calibration |
| `calibration.camera_lens` | Calibrate the camera lens |
| `calibration.make_markers` | ArUco markers to print, at the size in `setup/plate.yml` |
| `calibration.set_origin` | Record the plate's home position |
| `checks.status` | What's calibrated and what needs redoing (no camera needed) |
| `checks.live_view` | Live detection preview; sends nothing |
| `checks.plate_fit` | Fit error per marker; `--write` measures the real marker positions |
| `checks.check_marker` | Diagnose why a marker isn't detected |
| `checks.monitor` | Print the poses arriving on port 9001 |
| `checks.fake_tracker` | Send fake poses without a camera |

Shared code is in `tools/common/`. Generated images go to `tools/output/` (not in git).

## When something changes

Moved the camera, new markers, different projector or model: [`setup/README.md`](setup/README.md) has the steps for each case. `python -m checks.status` always shows what is out of date.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Webcam won't open, or the window is black | Close other apps using the camera (Teams, Zoom, Logi Tune), allow desktop apps in Windows camera privacy settings, or try `index: 1` in `setup/camera.yml` |
| Camera gives a lower resolution than asked | Use a USB 3 port, set `mjpg: 1` in `camera.yml`, close Logitech's apps |
| Image goes soft or the pose drifts | Autofocus turned back on: keep `autofocus: 0` in `camera.yml` and Logi Tune closed |
| High fit error | `setup/plate.yml` doesn't match the plate; run `checks.plate_fit` |
| Markers in view but not detected | Glare or projector light on the markers; run `checks.check_marker` |
| TouchDesigner doesn't move | Run `checks.monitor` alongside the tracker; if poses arrive there, the problem is on the TD side |
