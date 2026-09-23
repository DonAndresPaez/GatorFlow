# Message and file formats

Three interfaces hold this project together. Change one and update this file in the same commit.

## 1. Pose: KinectReader → TouchDesigner (and later the simulation)

| | |
|---|---|
| Sent by | `KinectReader.exe` (C++). `tracking.fake_tracker` sends the same message shape with made-up numbers. |
| Protocol | OSC over UDP |
| Ports | `9000` TouchDesigner, `9001` the simulation later (`tracking.monitor` uses it meanwhile) |
| Address | `/car/transform` |
| Values | 6 floats: `tx, ty, tz, rx, ry, rz` |
| Units | position in **millimetres**, rotation in **degrees** |
| Axes | TouchDesigner world: **Y up**. Rotation order X, then Y, then Z |
| Origin | the model's home spot on the table, captured by `set_origin` |
| Rate | every frame where the marker is found (~30 Hz). Nothing is sent when it's lost, so TouchDesigner holds the last pose. |
| Target in TD | `geo1` only, **never** `cam1` |

**Fallback format.** `OutputConfiguration::SendOsc` in `main.cpp` can be set to `false`, which sends a plain text line instead — `"12.500 -3.100 840.000 1.20 0.50 -0.30"`. That needs a **UDP In DAT** in TouchDesigner rather than an OSC In DAT. Pick one with whoever builds the TD side and note it here.

**Who converts what.** The tracker does all of it: OpenCV works in radians with Y pointing down, and the Kinect's colour frame arrives mirrored. By the time a pose leaves the tracker it is already un-mirrored, Y-up, in degrees, and measured from the home spot. TouchDesigner uses the numbers as they arrive.

**Axis check in the lab:** move the piece by hand along +X, +Y, +Z and watch the numbers. A flipped axis means the origin capture or the mirror flag is wrong, not TouchDesigner.

## 2. `shared/calibration.yml` — Python → C++

Written by `python -m tracking.calibrate`, read by `loadCameraCalibration`. OpenCV YAML, so `cv2.FileStorage` writes exactly what `cv::FileStorage` reads.

| Key | Type |
|---|---|
| `cameraMatrix` | 3×3 double |
| `distortionCoefficients` | 5×1 double |
| `imageWidth`, `imageHeight` | int (informational) |
| `reprojectionErrorPx` | double (informational; under 0.5 is good) |

## 3. `shared/world_origin.yml` — Python → C++

Written by `python -m tracking.set_origin`, read by `loadWorldOrigin`.

| Key | Type |
|---|---|
| `cameraToWorld` | 4×4 double, millimetres |

It's the inverse of the marker's pose at its home spot, so the chain

```
cameraToWorld · diag(1, −1, −1, 1) · (marker pose in the camera)
```

gives zero when the piece is parked at home. If the file is missing, the tracker uses the identity matrix and reports poses relative to the camera, which still works but isn't anchored to the table.

**Regenerate it whenever the Kinect moves.** It encodes where the sensor is.

## Still to define

The heatmap format (simulation → TouchDesigner) gets written here when that stage starts.
