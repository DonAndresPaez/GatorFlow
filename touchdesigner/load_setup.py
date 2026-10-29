'''load_setup.py: puts the rig files from setup/ into the TouchDesigner network.

So that nobody types calibration numbers into TouchDesigner by hand:
  setup/projector.json  -> cam1 (field of view, and its calibrated position)
  setup/model.json      -> the STL inside geo1, and where it sits on the plate

ONE-TIME WIRING (in the .toe, which lives in this folder):
  1. A Text DAT named `load_setup`. File parameter: load_setup.py, and turn on
     "Sync to File" so edits here show up in TouchDesigner.
  2. An Execute DAT with Start = On, and in onStart:
         op('load_setup').module.load()
  3. Inside geo1:  File In SOP  ->  Transform SOP named `mount`  ->  (material, out)
     geo1 itself stays driven by the OSC pose. `mount` places the model on the
     plate, so the OSC pose and the model offset never fight over geo1.
  4. cam1's View Angle Method set to Horizontal, so `fov` means horizontal FOV.

EVERYDAY USE (Textport, or buttons that call these):
  op('load_setup').module.load()             re-read setup/ (after editing it, or swapping models)
  op('load_setup').module.save_projector()   after calibrating cam1 with the sliders: store
                                             its position in setup/projector.json (then commit)

Remember the calibration rules: cam1's FOV is the projector lens and is never a
calibration knob; cam1 never follows the tracking; only geo1 does.
'''

import json
from pathlib import Path

# ---- names in the network: change these if the .toe uses different ones ----
CAMERA = '/project1/cam1'
GEO = '/project1/geo1'
MOUNT_SOP = 'mount'

# TouchDesigner units per millimetre. The STL is in mm and the OSC pose is in mm,
# so 1.0 keeps everything in mm. If the scene is built in metres, use 0.001
# AND scale the incoming OSC translation the same way.
TD_UNITS_PER_MM = 1.0

TRANSFORM_PARS = ('tx', 'ty', 'tz', 'rx', 'ry', 'rz')


def repo():
    # The .toe is saved in <repo>/touchdesigner/, so the repo is one level up.
    return Path(project.folder).parent


def _read(relative):
    return json.loads((repo() / relative).read_text())


def load():
    _load_projector()
    _load_model()


def _load_projector():
    cam = op(CAMERA)
    if cam is None:
        print(f'load_setup: no {CAMERA}, skipping the projector')
        return
    projector = _read('setup/projector.json')
    cam.par.fov = projector['lens']['horizontalFovDeg']
    saved = projector.get('cam1')
    if saved:
        for name in TRANSFORM_PARS:
            setattr(cam.par, name, saved[name])
        print(f"load_setup: cam1 <- {projector['name']} (FOV {cam.par.fov.eval()}, saved position)")
    else:
        print(f"load_setup: cam1 FOV <- {cam.par.fov.eval()}; no saved position yet, kept the .toe's")


def _load_model():
    geo = op(GEO)
    if geo is None:
        print(f'load_setup: no {GEO}, skipping the model')
        return
    active = _read('setup/model.json')['active']
    folder = repo() / 'models' / active
    model = json.loads((folder / 'model.json').read_text())
    stl = folder / model['stl']

    file_in = geo.findChildren(type=fileinSOP, depth=1)
    if not file_in:
        print(f'load_setup: put a File In SOP inside {GEO} (see the top of load_setup.py)')
        return
    file_in[0].par.file = stl.as_posix()

    mount = geo.op(MOUNT_SOP)
    if mount is None:
        print(f"load_setup: loaded {model['name']}, but there is no Transform SOP '{MOUNT_SOP}' "
              f'in {GEO}, so the mount offset was not applied')
        return
    scale = TD_UNITS_PER_MM * (1.0 if model.get('stlUnits', 'mm') == 'mm' else 1000.0)
    offset = model.get('mountOffsetMm', [0, 0, 0])
    mount.par.tx, mount.par.ty, mount.par.tz = (v * TD_UNITS_PER_MM for v in offset)
    mount.par.rx, mount.par.ry = 0, 0
    mount.par.rz = model.get('mountRotationDeg', 0)
    mount.par.scale = scale
    print(f"load_setup: geo1 <- {model['name']} ({stl.name}), offset {offset} mm")


def save_projector():
    cam = op(CAMERA)
    path = repo() / 'setup' / 'projector.json'
    projector = json.loads(path.read_text())
    projector['cam1'] = {name: round(float(getattr(cam.par, name).eval()), 4) for name in TRANSFORM_PARS}
    path.write_text(json.dumps(projector, indent=2) + '\n')
    print(f"load_setup: saved cam1 {projector['cam1']} to setup/projector.json")
