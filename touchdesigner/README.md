# touchdesigner/

The TouchDesigner side: receives the pose, draws the pressure skin and flow lines over the model, drives the projector.

| File | What |
|---|---|
| `ProjectMappingSetUp.toe` | the network: OSC In → geo1 (converted to TouchDesigner's Y up in `oscin1_callbacks`), `filein1` → `transform1` (axes, 0.01 units per mm) → geo1 (drawn black by `constant1`), `geo1/aero/aero_pop` (skin + flow lines), cam1 calibration sliders, `render1` → `window1` (the projector). The aero operators are installed in it. It has no `load_setup` DAT or `mount` SOP, so `aero_td.py` keeps `filein1` on the active model's placed OBJ (`models/<name>/aero/<name>_placed.obj`: mount offset and scale already applied), reloading it when the model or its `model.json` changes |
| `load_setup.py` | copies `setup/projector.json` and `setup/model.json` into the network, and saves cam1's calibration back. The File In SOP can't read STL, so it loads an OBJ copy it writes to `models/<name>/aero/` |
| `aero_td.py` | runs `tools/aero` (pose → pressure heatmap and flow lines); `install()` builds the operators, details at its top |
| `aero_pop.py` | callbacks for geo1/aero's Script POP `aero_pop` (the skin and the flow lines), forwards to `aero_td.py` |

Save the `.toe` **in this folder**: `load_setup.py` finds the repo as the folder above the `.toe`.

## What the network expects

| Operator | Driven by | Never driven by |
|---|---|---|
| `cam1` (the projector) | `setup/projector.json`: FOV from the lens, position from the static calibration | the tracking |
| `geo1` (the model) | OSC `/car/transform` on port 9000 (see `docs/osc_interface.md`) | the calibration sliders |
| `geo1/mount` (Transform SOP) | `models/<name>/model.json` mount offset | OSC |
| `geo1/aero` (Geometry COMP inside geo1) + `aero_pop` (Script POP) | `aero_td.py`: one mesh, the pressure skin then the flow-line ribbons, in the plate frame; `aero_mat` blends the lines over the skin; draw priority -1, after the model | |
| geo1's own material | black (set by `install()`): the skin carries the colours | |

Wiring steps are at the top of `load_setup.py`.

## Moving the projector

1. Move it, keep the same lens. The FOV in `setup/projector.json` does not change.
2. Recalibrate `cam1` with the sliders (static calibration, `cam1` locked afterwards).
3. In the Textport: `op('load_setup').module.save_projector()`.
4. Commit `setup/projector.json`. Next time the `.toe` opens, `cam1` comes back to that spot.

## Swapping the model

Change `active` in `setup/model.json` (see `models/README.md`), then `op('load_setup').module.load()`.

## Setting up

With a Text DAT `aero_td` (File: `aero_td.py`, Sync to File on), run `op('aero_td').module.install()` in the Textport once and save the .toe. It adds `geo1/aero` with the Script POP and its material, an Execute DAT and a status DAT, and makes geo1's own material black. It also replaces the older `aero_color` Script SOP and `geo1/flow` if it finds them.

## The pressure skin and flow lines

Everything is on `geo1/aero/aero_pop`:

- **Aero page:** Solver (Panel method / ML surrogate), Wind speed (m/s, export a slider to it), Display (Cp, Pa, or Wing loading: on thin parts, the pressure difference across them, everything else dark; it shows something only on models with lift turned on), the colour ranges, Heatmap brightness (default 0.6; lower it so the lines stand out, 0 for lines only), Skin offset (mm off the model; 0 = automatic, just under the lines), Reload. Blue is suction, red is where the wind hits. The skin shows grey with a status warning while the panel method's one-off build runs, and the ML solver falls back to the panel method (with a warning) until `python -m aero.train` has been run for the model. Press Reload after retraining.
- **Flow lines page:** Surface lines and Ground lines (on/off and how many; capped so they don't run together on small models), Line width (mm; 0 = 1/180 of the model but at least 1.5 mm, about two projector pixels), Animate, Flow rate, Dash length, and the speed colour range. They follow the panel method's velocity even when the skin uses the ML surrogate. After the model turns more than a degree they are re-traced in the background (well under a second; the old lines stay up meanwhile).

The wind's direction is fixed in `tools/aero/flow.py` (`WIND_DIRECTION_WORLD`). The skin is built from the solver's panels, not from the File In SOP's mesh, so the colours land in the right place whatever the File In SOP shows.

## Testing without the camera

From `tools/`: `python -m checks.fake_tracker` sends the same OSC messages the tracker does.
