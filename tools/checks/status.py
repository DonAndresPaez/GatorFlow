'''status.py: what the rig is set to right now, and what needs redoing.

Run: python -m checks.status        (from tools/; needs no camera)

Reads setup/, models/ and calibration/ and prints one block per part of the
rig, with a TODO line for anything missing or out of date. Run it after you
change anything in setup/, and at the start of every lab session.
'''

import json
from datetime import datetime

import numpy as np

from common import paths
from common.camera import PROFILE, load_intrinsics
from common.plate import LAYOUT, load_camera_to_world

GOOD_MARKER_PX = 40   # below this, detection and corner accuracy start to suffer

todo = []


def when(path):
    return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")


def rel(path):
    return path.relative_to(paths.REPO).as_posix()


def main():
    print(f"\nCAMERA   {rel(paths.CAMERA_FILE)}")
    print(f"  {PROFILE.name}  ({PROFILE.source}{', mirrored' if PROFILE.mirrored else ''})")
    if PROFILE.source == "webcam":
        print(f"  index {PROFILE.index}, asks for {PROFILE.width}x{PROFILE.height} @ {PROFILE.fps} fps,"
              f" autofocus {'ON (turn it off)' if PROFILE.autofocus else 'off'}")
        print("  note: KinectReader.exe only drives a Kinect; a webcam works with the Python tools")

    intrinsics_path, origin_path = paths.intrinsics_file(), paths.origin_file()
    lens = None
    print(f"\nLENS     {rel(intrinsics_path)}")
    if intrinsics_path.exists():
        lens = load_intrinsics()
        print(f"  {lens.image_size[0]}x{lens.image_size[1]}, error {lens.error_px:.2f} px, from {when(intrinsics_path)}")
        if lens.error_px > 1.0:
            todo.append("lens error over 1 px: python -m calibration.camera_lens (retake the shots)")
        if PROFILE.source == "webcam" and PROFILE.width and (PROFILE.width, PROFILE.height) != tuple(lens.image_size):
            todo.append(f"camera.yml asks for {PROFILE.width}x{PROFILE.height} but the lens was calibrated at "
                        f"{lens.image_size[0]}x{lens.image_size[1]}: python -m calibration.camera_lens")
    else:
        print("  missing")
        todo.append(f"no lens calibration for '{PROFILE.name}': python -m calibration.camera_lens")

    print(f"\nPLATE    {rel(paths.PLATE_FILE)}")
    span = np.ptp(LAYOUT.centers_mm, axis=0) if len(LAYOUT.ids) > 1 else np.zeros(2)
    print(f"  {len(LAYOUT.ids)} x {LAYOUT.size_mm:g} mm {LAYOUT.dictionary_name} markers, ids {LAYOUT.ids}")
    print(f"  marker centres span {span[0]:.0f} x {span[1]:.0f} mm (wider = steadier rotation)")

    print(f"\nORIGIN   {rel(origin_path)}")
    if origin_path.exists():
        camera_to_world = load_camera_to_world(origin_path)
        distance = np.linalg.norm(camera_to_world[:3, 3])
        print(f"  camera is {distance:.0f} mm from the plate's home spot, captured {when(origin_path)}")
        if lens is not None:
            px = lens.K[0, 0] * LAYOUT.size_mm / distance
            print(f"  markers appear about {px:.0f} px across at home (aim for {GOOD_MARKER_PX}+)")
            if px < GOOD_MARKER_PX:
                todo.append("markers are small in the image: bigger markers, a closer camera, or more resolution")
            if origin_path.stat().st_mtime < intrinsics_path.stat().st_mtime:
                todo.append("lens recalibrated after the origin was captured: python -m calibration.set_origin")
        if origin_path.stat().st_mtime < paths.PLATE_FILE.stat().st_mtime:
            todo.append("plate.yml changed after the origin was captured: if markers moved or changed size, "
                        "python -m calibration.set_origin")
    else:
        print("  missing (poses would be measured from the camera)")
        todo.append(f"no origin for '{PROFILE.name}': plate on its home spot, python -m calibration.set_origin")

    print(f"\nMODEL    {rel(paths.MODEL_FILE)}")
    try:
        model = paths.active_model()
        size = model.get("sizeMm")
        print(f"  {model['name']}: {rel(model['stl'])}" + (f", {size[0]} x {size[1]} x {size[2]} mm" if size else ""))
        print(f"  mounted at {model['mountOffsetMm']} mm, turned {model['mountRotationDeg']} deg on the plate")
        if not model["stl"].exists():
            todo.append(f"model STL missing: {rel(model['stl'])}")
    except (FileNotFoundError, KeyError) as error:
        print(f"  problem: {error}")
        todo.append("fix setup/model.json or the model's folder in models/")

    print(f"\nPROJECTOR {rel(paths.PROJECTOR_FILE)}")
    projector = json.loads(paths.PROJECTOR_FILE.read_text())
    print(f"  {projector['name']}, {projector['lens']['horizontalFovDeg']} deg horizontal FOV")
    if projector.get("cam1"):
        c = projector["cam1"]
        print(f"  cam1 saved: t ({c['tx']:.3g}, {c['ty']:.3g}, {c['tz']:.3g})  r ({c['rx']:.3g}, {c['ry']:.3g}, {c['rz']:.3g})")
    else:
        print("  cam1 not saved yet (TouchDesigner keeps its own values)")
        todo.append("after calibrating cam1 in TouchDesigner: op('load_setup').module.save_projector()")

    print("\nTODO" if todo else "\nNothing to redo.")
    for item in todo:
        print(f"  - {item}")
    print()


if __name__ == "__main__":
    main()
