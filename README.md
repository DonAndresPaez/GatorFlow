# GatorFlow

Real-time aerodynamic projection mapping: a Kinect tracks a 3D-printed model, a simulation computes the pressure on it, and TouchDesigner projects the result back onto the object.

**Right now the repo only holds the tracking stage.** The simulation and TouchDesigner parts get added once tracking works on real hardware.

```
[ Kinect ] → tracking → OSC pose → (later: simulation → heatmap → TouchDesigner → projector)
```

## Folders

| Path | What |
|---|---|
| `tracking/` | Finds the printed model with the camera and sends its pose over OSC |
| `models/` | `PMB-1A.stl`, the printed practice piece: 300 × 300 mm footprint, 58 mm tall, Z is up |
| `docs/` | `osc_interface.md` (the message format), `lab_day.md` (setup checklist) |
| `tests/` | `pytest` |
| `config.py` | Ports and the message address |

## Setup

```powershell
pip install -r requirements.txt
```

## Run (from the repo root)

| Command | What it does |
|---|---|
| `python -m tracking.make_marker` | Makes `marker.png` to print |
| `python -m tracking.calibrate` | Measures the camera lens into `tracking/camera.json` |
| `python -m tracking.set_origin` | Saves where the world's origin is, after the camera is mounted |
| `python -m tracking.track` | The real tracker |
| `python -m tracking.monitor` | Prints the poses arriving over OSC |
| `python -m tracking.fake_tracker` | Sends a fake pose, no camera needed |
| `pytest` | Tests |

## Order of work

1. ✅ Tracking code, tested on fake data
2. ⬜ Tracking working on the Kinect in the lab (`docs/lab_day.md`)
3. ⬜ TouchDesigner receiving the pose and moving the model on screen
4. ⬜ Projector calibration on the printed piece
5. ⬜ Simulation: pressure on the mesh → heatmap
6. ⬜ Full pipeline
