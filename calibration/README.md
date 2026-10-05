# calibration/

**Generated files, one folder per camera.** Written by the tools in `tools/calibration/`, read by the Python tools and KinectReader. Not in git: they belong to one camera in one mounting, not to the project.

```
calibration/
└── <camera name>/            the name comes from setup/camera.yml
    ├── intrinsics.yml        the lens          python -m calibration.camera_lens
    ├── intrinsics.backup.yml the previous lens calibration (kept automatically)
    └── world_origin.yml      the camera's spot python -m calibration.set_origin
```

| File | Redo when |
|---|---|
| `intrinsics.yml` | new camera, new resolution, zoom or focus changed. **Not** when the camera moves. |
| `world_origin.yml` | the camera moved at all, the lens was recalibrated, or the plate's markers changed |

Formats are in `docs/osc_interface.md`. Don't edit these by hand; rerun the tool.
