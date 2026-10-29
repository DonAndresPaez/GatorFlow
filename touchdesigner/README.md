# touchdesigner/

The TouchDesigner side: receives the pose, shows the heatmap on the model and drives the projector.

| File | What |
|---|---|
| `ProjectMappingSetUp.toe` | The network (not added yet) |
| `load_setup.py` | Loads `setup/projector.json` and `setup/model.json` into the network, and saves `cam1` back |

Save the `.toe` in this folder: `load_setup.py` finds the repo as the folder above it. Wiring steps are at the top of `load_setup.py`.

## What drives what

| Operator | Driven by | Never by |
|---|---|---|
| `cam1` (the projector) | `setup/projector.json`: FOV from the lens, position from static calibration | the tracking |
| `geo1` (the model) | OSC `/car/transform` on port 9000 | the calibration |
| `geo1/mount` | the mount offset in `models/<name>/model.json` | OSC |

## Moving the projector

1. Recalibrate `cam1` with the sliders, then lock it.
2. In the Textport: `op('load_setup').module.save_projector()`
3. Commit `setup/projector.json`.

## Swapping the model

Change `active` in `setup/model.json`, then `op('load_setup').module.load()`.
