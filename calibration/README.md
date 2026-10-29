# calibration/

Generated files, one folder per camera. Written by `tools/calibration/`, read by the tracker. Not in git: they belong to one camera in one mounting.

```
calibration/
└── <camera name>/          name from setup/camera.yml
    ├── intrinsics.yml      the lens            python -m calibration.camera_lens
    └── world_origin.yml    the camera's spot   python -m calibration.set_origin
```

| File | Redo when |
|---|---|
| `intrinsics.yml` | New camera, or the resolution, zoom or focus changed. Not when the camera only moves. |
| `world_origin.yml` | The camera moved, the lens was recalibrated, or the markers changed |

Don't edit these by hand; rerun the tools.
