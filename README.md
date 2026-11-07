# GatorFlow

Real-time aerodynamic projection mapping. A camera tracks an ArUco marker plate with a 3D-printed model on it, a simulation computes the pressure on the model, and TouchDesigner projects the result back onto the object.

```
camera ──► KinectReader (C++) ──OSC /car/transform──► TouchDesigner ──► projector
               ▲      reads                              ▲ reads   │ ▲
               │                                         │         ▼ │ colour per point
        setup/camera.yml, setup/plate.yml         setup/projector.json, setup/model.json
        calibration/<camera>/*.yml  ◄── written by tools/calibration (Python)
                                                  tools/aero (Python, run inside TouchDesigner)
```

## Where things are

| Folder | What | Edit by hand? |
|---|---|---|
| `setup/` | **The rig**: camera, marker plate, projector, active 3D-model. If anything physical changes, this needs to be modified. | yes |
| `calibration/` | Generated lens + origin files, one folder per camera | no, rerun the tools |
| `models/` | One folder per 3D model: STL + `model.json` | when adding a model |
| `tools/` | Python: calibration, printables, checks, tests, and `aero/` (the solvers TouchDesigner runs) | code |
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
| **aero** | |
| `python -m aero.decimate SOURCE.stl OUT.stl` | simplifies a detailed STL (e.g. 130k triangles) to ~6000 for the solver, shell by shell, dropping internal cavities (needs `fast-simplification`) |
| `python -m aero.precompute` | builds the panel method's cache for the active model (TD also does this by itself) |
| `python -m aero.train` | pretrains the ML surrogate → `models/<name>/aero/surrogate.npz` (needs PyTorch: `aero/requirements-train.txt`) |
| `pytest` | tests, including a check that the files in `setup/` are valid |

`tools/common/` is the shared library (`paths.py` knows every file location, `camera.py`, `plate.py`, `config.py` for ports). Generated images go to `tools/output/` (not in git).

Setup: `pip install -r tools/requirements.txt`. For the Kinect from Python also `pip install pykinect2 comtypes`, and if it crashes on import, `python -m common.patch_pykinect2`.

## Aerodynamics (`tools/aero/`)

Runs inside TouchDesigner (NumPy only), from one Script POP, `geo1/aero/aero_pop`, which `op('aero_td').module.install()` sets up (see `touchdesigner/aero_td.py`). Each frame it takes the plate pose, turns the fixed wind (`WIND_DIRECTION_WORLD` in `aero/flow.py`) into the plate's frame, and draws a **skin**: the solver's own panels, a fraction of a millimetre off the model, coloured by pressure. The model's mesh itself draws black underneath. Two pipelines, picked on the POP's Aero page:

| Solver | What | Speed |
|---|---|---|
| Panel method | 3D source panels on the STL, the plate as a mirror-image ground. Potential flow: right at the front and over the top, too much suction behind sharp edges (no separation). | one-off build per model (~15 s for PMB-1A, plus ~20 s for the flow-line field; background, cached), then ~1 ms |
| ML surrogate | MLP + PCA modes, pretrained by `python -m aero.train` on the panel method or on your own CFD results (`--data`). Wind speed is an input, so it can change live. | ~1 ms, no PyTorch in TD |

In the same mesh, **flow lines** (`aero/flowlines.py`): ribbons that follow the air across the model's surface and over the plate around it, coloured by air speed (dim violet slow, pale yellow fast) and animated as dashes that move with the local speed. Their dash gaps are see-through, so the skin shows between them and the bare plate stays dark. They come from the panel method's velocity field whichever solver colours the skin, and are re-traced in the background when the model turns. Being potential flow, the lines close up again behind the model: no wake.

**Wings** (`aero/lifting.py`, opt-in with `"kutta": true` in a model's `model.json`): sources alone give wings no lift, so the lifting version adds doublets, the Kutta condition at each wing's trailing edge and a wake behind it, with the plate's mirror giving ground effect. It is validated on clean wings (lift slope within 4% of lifting-line theory, more downforce near the ground, sharp and blunt trailing edges), and still runs live. It is off by default because on printed models it can go wrong: rounded trailing edges and coarse decimation make the trailing-edge condition unreliable, and on the F1 car it did (see `models/README.md`). A **Wing loading** display shows, on thin parts, the pressure difference across them: red on the side pushed, blue on the side pulled.

Which model is on the plate still comes from `setup/model.json`, through `aero/catalog.py`'s `ModelSelector`, so an automatic selector (depth-image fit, a marker per model) can replace it without touching the solvers.

## Building KinectReader

Needs the Kinect for Windows SDK 2.0 (sets `KINECTSDK20_DIR`), OpenCV with contrib (path set in `CMakeLists.txt`), and Visual Studio 2022.

```powershell
cd KinectReader
cmake -B build
cmake --build build --config Debug
```

It finds `setup/` by walking up from the folder it runs in. From somewhere else, pass the repo folder: `KinectReader.exe "C:\path\to\GatorFlow"`.

Why two languages: the Kinect v2 has no working Python binding on current Python versions, so the live path is C++ against the Kinect SDK. The setup tools don't need that API and are quicker to work on in Python. A plain webcam works from Python directly (`source: "webcam"` in `setup/camera.yml`).
