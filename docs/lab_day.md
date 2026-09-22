# Lab day checklist — tracking setup

Goal of the session: the Kinect sees the marker, and the pose it sends is measured from the car's home spot on the table. Nothing about the projector image yet.

## Bring

- [ ] Printed marker on stiff card (`python -m tracking.make_marker`, print at 100%, glue flat)
- [ ] Printed checkerboard on stiff card, if calibrating today
- [ ] Ruler or tape measure, tape, the printed piece (PMB-1A, 300 x 300 mm)
- [ ] Laptop with `pip install -r requirements.txt` already done
- [ ] Kinect + its power adapter and USB 3 cable (Kinect v2 needs USB 3)

## 1. Camera talks to the laptop (15 min)

```powershell
python -m tracking.track
```
Black window or an error means the wrong camera. Try `CAMERA_INDEX = 1`, then 2. Close Zoom/Teams first, they lock the camera.

Success: live video with "lost" in the corner.

## 2. Marker is detected (10 min)

Hold the marker up. The corner label flips to LOCK and colored axes appear on it.

If not: more light, less glare, marker flatter, or fill more of the frame.

## 3. Mount the Kinect (20 min)

Next to the projector, looking at the same area. Clamp or tape it so it cannot shift — everything after this is measured relative to where it sits. Check that the car's whole working area is in frame, and that nothing blocks the view.

## 4. Camera calibration (30 min, can be postponed)

```powershell
python -m tracking.calibrate
```
15-20 shots of the checkerboard at different angles and distances. Aim for a reprojection error under 0.5 px. Skip today if short on time: tracking still works, distances are just a few percent off.

## 5. Set the world origin (10 min) ← the real deliverable

Tape a mark on the table where the printed piece will sit. Put the marker flat there (later it gets fixed to the piece itself), then:
```powershell
python -m tracking.set_origin
```
Writes `tracking/world_origin.json`. The check line at the end should print six zeros, and the noise numbers should be under ~2 mm.

**Redo this if the Kinect moves.**

## 6. Measurement test (20 min)

Two terminals:
```powershell
python -m tracking.monitor      # terminal 1
python -m tracking.track        # terminal 2
```
With the marker at home: all six numbers near zero.

The printed piece is 300 x 300 mm, so there is plenty of flat area to tape the marker onto once the loose-marker test passes. Try that too if time allows: the numbers should behave the same.

| Move the marker | Expected |
|---|---|
| 100 mm right | one position number goes to about +100 |
| 100 mm toward the ceiling camera | another goes to about +100 |
| rotate 90° flat on the table | one rotation number goes to about 90 |
| leave it still | numbers drift by only 1-2 mm |

Write down which of X/Y/Z moved for each direction. If a sign is flipped, that's fixed later in TouchDesigner's axis check, not by editing the tracker.

## 7. Real conditions (15 min)

Turn the projector on, aimed at the marker area. Watch whether the projected light washes out detection. If it does, note it: the marker may need to sit where the projection doesn't hit it, or the projected image may need a dark patch around it.

## Record before leaving

- Camera index, Kinect mounting spot, distance to the table
- Reprojection error from calibration
- Which direction maps to which axis
- Noise level while still, and the message rate from the monitor
- Whether the projector interferes

## If it goes badly

Fallback in order: no Kinect → use a laptop webcam for the whole test; no detection at all → print the marker bigger; no time → at minimum get steps 1, 2 and 5 done, since the origin is what everything else builds on.
