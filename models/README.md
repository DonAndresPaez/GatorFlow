# models/

One folder per physical model that can sit on the plate. `setup/model.json` says which one is active.

```
models/
└── PMB-1A/
    ├── PMB-1A.stl     the mesh: millimetres, Z up
    └── model.json     name, STL file, size, and where it sits on the plate
```

## Adding a model

1. Export the STL in millimetres, Z up, with the origin at the centre of its base.
2. Create `models/<NAME>/` with the STL and a `model.json` (copy PMB-1A's).
3. If it doesn't sit centred on the plate, set `mountOffsetMm` / `mountRotationDeg` (plate frame: X right, Y far edge, Z up).
4. Set `"active"` in `setup/model.json` and run `op('load_setup').module.load()` in TouchDesigner.

The tracker follows the plate, not the model, so swapping models needs no recalibration. Keep markers outside every model's footprint and mount models the same way each time.
