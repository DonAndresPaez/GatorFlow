# models/

One folder per physical model that can sit on the plate.

```
models/
└── PMB-1A/
    ├── PMB-1A.stl     the mesh, in mm, Z up
    ├── model.json     name, STL file, size, and where it sits on the plate
    └── aero/
        ├── surrogate.npz          the trained ML surrogate (python -m aero.train), committed
        ├── panel_<hash>_v3.npz    panel-method cache, rebuilt by itself, not in git
        ├── field_<hash>_*.npz     flow-line velocity grid over the plate, rebuilt by itself, not in git
        ├── <name>.obj             the STL as OBJ for TouchDesigner's File In SOP (load_setup.py mounts it), not in git
        └── <name>_placed.obj      the same, already mounted and scaled (for a .toe with no mount SOP), not in git
```

`setup/model.json` says which one is on the plate right now (`"active": "PMB-1A"`).

## Adding a model

1. Export the STL in **millimetres, Z up, origin at the centre of its base**. Then the model sits correctly on the plate with a zero offset.
2. Make `models/<NAME>/` with the STL and a `model.json` (copy PMB-1A's and edit it).
3. If the model doesn't sit centred on the plate, or its STL origin isn't the centre of its base, set `mountOffsetMm` / `mountRotationDeg`. They're in the plate frame: X to the plate's right edge, Y to its far edge, Z up.
4. If the print is a different size from the STL, set `"scale"` (printed size / STL size, e.g. `3` for a 102 mm STL printed 306 mm long). It scales the whole placed model about the plate centre, so leave `sizeMm` and `mountOffsetMm` as they are for the STL. The panel method rebuilds by itself (the model shows grey for a minute); retrain the network afterwards, as the old one is refused.
5. Set `"active"` in `setup/model.json` to the new name.
6. In TouchDesigner: `op('load_setup').module.load()`.
7. Aerodynamics: from `tools/`, `python -m aero.precompute` (optional, TD builds it too) and `python -m aero.train`, then commit `aero/surrogate.npz`. Until it is trained, the ML solver falls back to the panel method.

For the aerodynamics the STL should be **closed** (watertight) and stand on Z = 0; faces lying flat on the plate are dropped and the plate mirrors the rest. Separate parts are fine (the F1 car is a body and four wheels): faces of two parts that face each other across less than 1 mm (print-in-place clearances) are dropped too, since no real air flows through a slit that narrow and potential flow would race through it.

**Wings:** `"kutta": true` in `model.json` turns on lift (aero/lifting.py) for the model's wings: trailing edges are found where a thin part's upward-facing top ends at the back (wheels are recognised and left out), a wake trails back from each, and the Kutta condition sets the circulation. Turn it on for models whose wings end in a sharp or flat (blunt) back edge and are meshed finely enough to show it, then check the result. It stays off for the F1: its wings end in rounded edges on a coarse mesh, where the trailing-edge condition comes out unreliable (the rear wing produced wildly wrong circulation and was rejected by the solver's guard; the rest made lift instead of downforce). Getting the F1's wings right needs cleaner wing geometry, a thin-wing (vortex lattice) treatment, or CFD results fed to the network.

Keep it under about 8000 triangles: the panel build needs an N x N matrix (8000 → 0.5 GB, about a minute). Printable models are often far bigger, so simplify them first:

```
cd tools
pip install fast-simplification
python -m aero.decimate path/to/original.stl ../models/<NAME>/<NAME>.stl --triangles 6000
```

It simplifies each part on its own (keeping every one closed), drops internal cavities, and reports each part's volume change; raise `--triangles` if a thin part changed a lot. Commit the simplified STL, and say in `model.json`'s notes where the original came from and under what licence.

The tracker doesn't care which model is on the plate. It tracks the plate, so no recalibration is needed. For the same reason, **keep the markers outside every model's footprint** (as the camera sees it), and mount models the same way every time (pegs or a socket on the plate) so the offset stays true.
