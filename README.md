# GatorFlow

Real-time aerodynamic projection mapping. A Kinect tracks a 3D-printed model, a simulation computes the pressure on it, and TouchDesigner projects the result back onto the object.

**Only the tracking stage exists so far.**

```
Kinect ──► KinectReader (C++) ──OSC /car/transform──► TouchDesigner ──► projector
                  ▲                                   (later: simulation on 9001)
                  │ reads at startup
            shared/calibration.yml
            shared/world_origin.yml
                  ▲
                  │ written by
            Python calibration tools
```

## Layout

| Folder | What | Owner |
|---|---|---|
| `KinectReader/` | The tracker that runs during a session: Kinect SDK → ArUco → pose → OSC | Jules |
| `Python/` | Calibration and setup tooling, plus test helpers that need no Kinect | Andres |
| `shared/` | `calibration.yml` and `world_origin.yml`, written by Python and read by C++ |  |
| `docs/` | The message format, the lab procedure, the lab log |  |
| `models/` | `PMB-1A.stl`, the printed practice piece (300 × 300 mm, 58 mm tall, Z up) |  |

Why two languages: the Kinect v2 has no working Python binding on current Python versions, so the live path is C++ against the Kinect SDK. The setup tools don't touch that sensor API and are quicker to work on in Python.

## Order of a session

```powershell
# once per printed marker / board
cd Python
python -m tracking.make_marker
python -m tracking.make_checkerboard

# once per camera
python -m tracking.calibrate        # writes shared/calibration.yml

# every time the Kinect moves
python -m tracking.set_origin       # writes shared/world_origin.yml

# then the tracker itself
cd ..\KinectReader\build\Debug
.\KinectReader.exe
```

Watch what it sends without TouchDesigner:

```powershell
cd Python
python -m tracking.monitor          # prints poses arriving on 9001
```

The live tracking loop lives only in `KinectReader`. Python keeps the marker
description and the origin transform in `tracking/marker.py`, because
`set_origin` needs the same math to produce the matrix the C++ reads.

## Building KinectReader

Needs the Kinect for Windows SDK 2.0 (sets `KINECTSDK20_DIR`), OpenCV with contrib (the path is set in `CMakeLists.txt`), and Visual Studio 2022.

```powershell
cd KinectReader
cmake -B build
cmake --build build --config Debug
```

## Python tools

| Command (from `Python/`) | What it does |
|---|---|
| `python -m tracking.make_marker` | `marker.png` to print |
| `python -m tracking.make_checkerboard` | `checkerboard.png` to print |
| `python -m tracking.calibrate` | measures the lens → `shared/calibration.yml` |
| `python -m tracking.set_origin` | captures the home spot → `shared/world_origin.yml` |
| `python -m tracking.check_marker` | why isn't the marker detected: brightness, sharpness, decode |
| `python -m tracking.monitor` | prints the poses KinectReader is sending |
| `python -m tracking.fake_tracker` | fake poses, so TouchDesigner can be tested with no camera at all |
| `pytest` | tests |
