# touchdesigner/

The TouchDesigner side: receives the pose, shows the heatmap on the model, drives the projector.

| File | What |
|---|---|
| `ProjectMappingSetUp.toe` | the network (save the current one here; not added yet) |
| `load_setup.py` | copies `setup/projector.json` and `setup/model.json` into the network, and saves cam1's calibration back |

Save the `.toe` **in this folder**: `load_setup.py` finds the repo as the folder above the `.toe`.

## What the network expects

| Operator | Driven by | Never driven by |
|---|---|---|
| `cam1` (the projector) | `setup/projector.json`: FOV from the lens, position from the static calibration | the tracking |
| `geo1` (the model) | OSC `/car/transform` on port 9000 (see `docs/osc_interface.md`) | the calibration sliders |
| `geo1/mount` (Transform SOP) | `models/<name>/model.json` mount offset | OSC |

Wiring steps are at the top of `load_setup.py`.

## Moving the projector

1. Move it, keep the same lens. The FOV in `setup/projector.json` does not change.
2. Recalibrate `cam1` with the sliders (static calibration, `cam1` locked afterwards).
3. In the Textport: `op('load_setup').module.save_projector()`.
4. Commit `setup/projector.json`. Next time the `.toe` opens, `cam1` comes back to that spot.

## Swapping the model

Change `active` in `setup/model.json` (see `models/README.md`), then `op('load_setup').module.load()`.

## Testing without the camera

From `tools/`: `python -m checks.fake_tracker` sends the same OSC messages the tracker does.
