# OSC interface

Every value here is also in `config.py`. Change both together.

## Pose: tracking → TouchDesigner

| | |
|---|---|
| Protocol | OSC over UDP |
| Ports | `9000` TouchDesigner, `9001` second copy (the simulation later; `tracking.monitor` for now) |
| Address | `/car/transform` |
| Values | `[tx, ty, tz, rx, ry, rz]`, floats |
| Units | position in **millimetres**, rotation in **degrees** |
| Axes | TouchDesigner world: **Y up**. Rotation order X, then Y, then Z |
| Origin | the model's home spot on the table, captured by `python -m tracking.set_origin` |
| Rate | every camera frame where the marker is found (~30 Hz). Nothing is sent when it's lost, so TouchDesigner holds the last pose. |

The tracker does all the unit and axis conversion, since OpenCV works in radians with Y pointing down. TouchDesigner uses the numbers as they arrive, and they move **`geo1` only, never `cam1`**.

**Axis check in the lab:** move the model by hand along +X, +Y, +Z and watch the numbers. If an axis is flipped, fix it in `tracking/world_origin.json` by rerunning `set_origin`, or in `CAMERA_TO_WORLD`, not in TouchDesigner.

## Still to define

The heatmap format (simulation → TouchDesigner) gets written here when that stage starts.
