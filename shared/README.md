# shared/

The handoff between the two languages. Python writes these, `KinectReader` (C++) reads them at startup.

| File | Written by | Holds |
|---|---|---|
| `calibration.yml` | `python -m tracking.calibrate` | the camera's lens numbers (`cameraMatrix`, `distortionCoefficients`) |
| `world_origin.yml` | `python -m tracking.set_origin` | `cameraToWorld`, the 4×4 that makes the model's home spot read as zero |

Both are OpenCV's own YAML format, so `cv::FileStorage` on the C++ side reads exactly what `cv2.FileStorage` on the Python side wrote. No conversion, no parsing by hand.

Neither file is in git: they describe one camera in one physical mounting, so they belong to a machine, not to the project. Regenerate them after the Kinect moves.

`KinectReader` looks for them in this folder by walking up from wherever the executable runs. You can also pass explicit paths:

```powershell
KinectReader.exe path\to\calibration.yml path\to\world_origin.yml
```
