# Lab day plan — tracking only

**Goal:** the Kinect, mounted next to the projector, sees the marker, and `python -m tracking.track` reports the printed piece's position measured from its home spot on the table.

**Not today:** TouchDesigner, the projected image, and the simulation. None of that code exists yet. The projector only gets switched on at the end, to see whether its light disturbs the camera.

**Minimum success:** steps 1, 2 and 5 done. Everything later builds on the origin from step 5.

## Before leaving

- [ ] `python -m tracking.make_marker`, print `marker.png` at 100% scale (turn off "fit to page"), glue to stiff card
- [ ] Measure the printed black square. If it isn't 80 mm, set `MARKER_SIZE_MM` in `tracking/track.py`
- [ ] `pip install -r requirements.txt` on the laptop you're bringing
- [ ] `pytest` passes (6 tests)
- [ ] Commit and push, so the laptop and GitHub agree
- [ ] Pack: checkerboard printout, ruler or tape measure, tape, the PMB-1A piece, Kinect + power adapter + USB 3 cable

## 1. Camera opens — 15 min

```powershell
python -m tracking.track
```
Expect live video with "lost" in the corner.

| Problem | Try |
|---|---|
| Black window or an error | `CAMERA_INDEX = 1` in `tracking/track.py`, then 2 |
| Camera busy | close Zoom, Teams, the Kinect SDK viewer |
| `camera.json` error | you're not in the repo root |

## 2. Marker detected, Kinect in hand — 10 min

Hold the Kinect and walk the marker around before mounting anything. The corner label flips to LOCK and colored axes appear on the marker.

Learn two numbers here: the farthest distance that still holds LOCK, and the steepest angle that still holds it. They decide where the Kinect can go in step 3.

| Problem | Try |
|---|---|
| Never detects | more light, less glare, marker flatter, marker bigger in frame |
| Flickers in and out | move closer, or aim away from a light source |

## 3. Mount the Kinect — 20 min

Next to the projector, pointing at the same area. Check before fixing it: the whole area the piece will move in is in frame, the distance is inside what you measured in step 2, and nothing (your arm, the projector body) blocks the view.

Clamp or tape it so it cannot shift. Everything measured after this assumes it stays put.

## 4. Camera calibration — 30 min, skippable

```powershell
python -m tracking.calibrate
```
SPACE keeps a shot, q finishes. 15–20 shots, varying one thing at a time: distance (close, middle, far), tilt (up to ~45° in each direction), and position in the frame (centre, each corner). Fill about a third of the frame with the board. Hold still for each shot so it isn't blurred. Aim for reprojection error under 0.5 px, and retake if it's over 1.0.

Skip if time is short. Tracking still works with the placeholder lens numbers, distances are just a few percent off. Just note that you skipped it.

## 5. Set the world origin — 10 min ← the day's deliverable

Tape a mark on the table where the piece will sit. Put the marker flat on that mark, keep it still:
```powershell
python -m tracking.set_origin
```
It collects 60 readings and writes `tracking/world_origin.json`.

Check the two lines it prints: the final pose should be six zeros, and the noise should be under about 2 mm. Higher noise means glare, motion blur, or a bent marker.

From here on, the tracker reports "where the piece is relative to home" instead of "where it is relative to the camera". **Rerun this if the Kinect moves at all.**

## 6. Measurement test — 20 min

Two terminals from the repo root:
```powershell
python -m tracking.monitor      # terminal 1
python -m tracking.track        # terminal 2
```

| Do this | Expect |
|---|---|
| Marker on the home mark | all six numbers within a few mm / degrees of zero |
| Slide 100 mm right | one position number → about +100 |
| Slide 100 mm away from you | another → about ±100 |
| Lift 100 mm off the table | the third → about ±100 |
| Rotate 90° flat on the table | one rotation number → about 90 |
| Leave still for 30 s | drift only 1–2 mm |
| Cover the marker | numbers stop updating, no crash |

**Write down which axis moved for each direction, and its sign.** That table is what makes the TouchDesigner setup quick later. Don't edit the code to "fix" signs today.

Also note the message rate the monitor prints. Under about 15/s means the camera or the laptop is the bottleneck.

## 7. Marker on the real piece — 15 min

Tape the marker onto a flat spot on PMB-1A (300 × 300 mm, so there's room). Repeat the moves from step 6 with the piece itself.

Decide and note: where on the piece the marker sits, and whether it stays visible when the piece is turned the way it will be during a demo.

## 8. Projector interference — 15 min

Turn the projector on, aimed at the area, and watch the LOCK label while it shines on the marker. This is the one risk that can only be answered in the real room.

If detection drops, note which of these helps: a marker position the projection doesn't reach, a darker projected image, or a larger marker.

## Before leaving, write down

Fill in `docs/lab_log.md` and commit it. Numbers to capture: camera index, where the Kinect is mounted and how far from the table, calibration error (or "skipped"), the axis table from step 6, the noise level and message rate, where the marker sits on the piece, and what the projector did.

## If things go wrong

- Kinect won't work at all → use the laptop webcam and do steps 1, 2, 5, 6 anyway. The Kinect swap is then just a camera index change.
- No detection anywhere → print the marker larger, at 120 mm.
- Out of time → steps 1, 2, 5. Nothing else matters if the origin isn't set.
