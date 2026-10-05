# GatorFlow

Real-time aerodynamic projection mapping. A camera tracks an ArUco marker plate with a 3D-printed model on it, a simulation computes the pressure on the model, and TouchDesigner projects the result back onto the object.

```
camera ──► KinectReader (C++) ──OSC /car/transform──► TouchDesigner ──► projector
               ▲      reads                              ▲ reads
               │                                         │
        setup/camera.yml, setup/plate.yml         setup/projector.json, setup/model.json
        calibration/<camera>/*.yml  ◄── written by tools/calibration (Python)
```

## Where things are

| Folder | What | Edit by hand? |
|---|---|---|
| `setup/` | **The rig**: camera, marker plate, projector, active 3D-model. If anything physical changes, this needs to be modified. | yes |
| `calibration/` | Generated lens + origin files, one folder per camera | no, rerun the tools |
| `models/` | One folder per 3D model: STL + `model.json` | when adding a model |
| `tools/` | Python: calibration, printables, checks, tests | code |
| `KinectReader/` | C++: the live tracker (Kinect SDK → ArUco → pose → OSC) | code |
| `touchdesigner/` | The `.toe` and `load_setup.py` | in TD |
| `docs/` | Message/file formats, lab procedure and log | yes |

## If you change something:

All commands run from `tools/`. `python -m checks.status` shows the current rig and a TODO list of anything out of date.

| I changed… | Edit | Then run |
|---|---|---|
| the camera's position | nothing | `calibration.set_origin` |
| to a different camera | `setup/camera.yml` (new `name`) | `calibration.camera_lens`, `calibration.set_origin` |
| the camera's resolution | `setup/camera.yml` | `calibration.camera_lens`, `calibration.set_origin` |
| the marker size or dictionary | `setup/plate.yml` | `calibration.make_markers` → print → measure → update `plate.yml` → `calibration.set_origin` |
| where a marker is stuck | `setup/plate.yml` | `checks.live_view` (fit error < 1 px), `calibration.set_origin` |
| the projector's position | nothing by hand | recalibrate `cam1` in TD → `save_projector()` (see `touchdesigner/README.md`) |
| the 3D model | add `models/<name>/`, set `setup/model.json` | `load()` in TD. No recalibration. |

Step-by-step versions of each are in `setup/README.md`.

## Running a session

```powershell
cd tools
python -m checks.status             # what's set up, what needs redoing
python -m checks.live_view          # plate detected? fit error low?

cd ..\KinectReader\build\Debug
.\KinectReader.exe                  # live tracking, sends OSC to TD (9000) and the simulation (9001)
```

Then open `touchdesigner/ProjectMappingSetUp.toe`; it loads `setup/` on start.

## Tools (`tools/`, Python)

| Command | What it does |
|---|---|
| **calibration** | |
| `python -m calibration.make_checkerboard` | `output/checkerboard.png` to print |
| `python -m calibration.camera_lens` | measures the lens → `calibration/<camera>/intrinsics.yml` |
| `python -m calibration.make_markers` | `output/marker_<id>.png` at the size in `setup/plate.yml` |
| `python -m calibration.set_origin` | captures the plate's home spot → `calibration/<camera>/world_origin.yml` |
| **checks** | |
| `python -m checks.status` | current rig + what needs redoing (no camera needed) |
| `python -m checks.live_view` | live view: which markers are seen, fit error, pose |
| `python -m checks.check_marker` | why isn't the marker detected: brightness, sharpness, decode |
| `python -m checks.monitor` | prints the poses KinectReader is sending (port 9001) |
| `python -m checks.fake_tracker` | fake poses, so TD and the simulation can be tested with no camera |
| `pytest` | tests, including a check that the files in `setup/` are valid |

`tools/common/` is the shared library (`paths.py` knows every file location, `camera.py`, `plate.py`, `config.py` for ports). Generated images go to `tools/output/` (not in git).

Setup: `pip install -r tools/requirements.txt`. For the Kinect from Python also `pip install pykinect2 comtypes`, and if it crashes on import, `python -m common.patch_pykinect2`.

## Building KinectReader

Needs the Kinect for Windows SDK 2.0 (sets `KINECTSDK20_DIR`), OpenCV with contrib (path set in `CMakeLists.txt`), and Visual Studio 2022.

```powershell
cd KinectReader
cmake -B build
cmake --build build --config Debug
```

It finds `setup/` by walking up from the folder it runs in. From somewhere else, pass the repo folder: `KinectReader.exe "C:\path\to\GatorFlow"`.

Why two languages: the Kinect v2 has no working Python binding on current Python versions, so the live path is C++ against the Kinect SDK. The setup tools don't need that API and are quicker to work on in Python. A plain webcam works from Python directly (`source: "webcam"` in `setup/camera.yml`).
