# models/

One folder per physical model that can sit on the plate.

```
models/
└── PMB-1A/
    ├── PMB-1A.stl     the mesh, in mm, Z up
    └── model.json     name, STL file, size, and where it sits on the plate
```

`setup/model.json` says which one is on the plate right now (`"active": "PMB-1A"`).

## Adding a model

1. Export the STL in **millimetres, Z up, origin at the centre of its base**. Then the model sits correctly on the plate with a zero offset.
2. Make `models/<NAME>/` with the STL and a `model.json` (copy PMB-1A's and edit it).
3. If the model doesn't sit centred on the plate, or its STL origin isn't the centre of its base, set `mountOffsetMm` / `mountRotationDeg`. They're in the plate frame: X to the plate's right edge, Y to its far edge, Z up.
4. Set `"active"` in `setup/model.json` to the new name.
5. In TouchDesigner: `op('load_setup').module.load()`.

The tracker doesn't care which model is on the plate. It tracks the plate, so no recalibration is needed. For the same reason, **keep the markers outside every model's footprint** (as the camera sees it), and mount models the same way every time (pegs or a socket on the plate) so the offset stays true.
