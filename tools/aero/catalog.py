'''catalog.py: the models in models/, and which one is on the plate.

Imported by: aero/pipeline.py, aero/precompute.py, aero/train.py, tests
Run directly: no

Choosing the model is behind a ModelSelector, so it can be swapped without
touching the solvers. Today it is ManualSelector: setup/model.json's `active`,
re-read when the file changes. An automatic one (e.g. fit each STL to the
Kinect depth image, or one ArUco id per model) only has to implement
select(context) and return one of available_models(); pass it to AeroPipeline.
'''

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from aero.geometry import load_panels, mount_transform, write_obj

REPO = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO / "models"
MODEL_FILE = REPO / "setup" / "model.json"


@dataclass
class ModelInfo:
    name: str
    folder: Path
    stl: Path
    units: str
    size_mm: list
    mount_offset_mm: list
    mount_rotation_deg: float
    kutta: bool = False         # "kutta": true in model.json: wings get lift (lifting.py)
    scale: float = 1.0          # "scale" in model.json: printed size / STL size

    @property
    def mount(self):
        """4x4 STL frame -> plate frame, scale included."""
        return mount_transform(self.mount_offset_mm, self.mount_rotation_deg, self.scale)

    @property
    def aero_dir(self):
        """Where this model's solver cache and trained network live."""
        return self.folder / "aero"

    def panels(self, ground=True):
        return load_panels(self.stl, self.mount, ground, self.units)

    def placed_obj(self):
        """aero/<stl name>_placed.obj: the model as it sits on the plate (units,
        mount and scale applied, mm, plate frame), for a network with no mount
        SOP of its own. Rewritten when the STL or model.json changes."""
        path = self.aero_dir / f"{self.stl.stem}_placed.obj"
        newest = max(self.stl.stat().st_mtime, (self.folder / "model.json").stat().st_mtime)
        if not path.exists() or path.stat().st_mtime < newest:
            units = np.diag([1000.0 if self.units == "m" else 1.0] * 3 + [1.0])
            write_obj(self.stl, path, self.mount @ units)
        return path


def available_models(models_dir=MODELS_DIR):
    """Names of every models/<name>/ that has a model.json."""
    return sorted(p.parent.name for p in Path(models_dir).glob("*/model.json"))


def model_info(name, models_dir=MODELS_DIR):
    folder = Path(models_dir) / name
    path = folder / "model.json"
    if not path.exists():
        raise KeyError(f"no model '{name}' in {models_dir} (have: {', '.join(available_models(models_dir))})")
    data = json.loads(path.read_text())
    scale = float(data.get("scale", 1.0))
    if not scale > 0:
        raise ValueError(f"{path}: scale must be above 0, not {data['scale']}")
    return ModelInfo(
        name=data.get("name", name),
        folder=folder,
        stl=folder / data["stl"],
        units=data.get("stlUnits", "mm"),
        size_mm=data.get("sizeMm"),
        mount_offset_mm=[float(v) for v in data.get("mountOffsetMm", [0, 0, 0])],
        mount_rotation_deg=float(data.get("mountRotationDeg", 0.0)),
        kutta=bool(data.get("kutta", False)),
        scale=scale,
    )


class ModelSelector:
    """Decides which model is on the plate. `context` is whatever the caller
    has (TouchDesigner passes a dict; a depth-based selector would want the
    Kinect depth TOP in it). Returns a name from available_models()."""

    def select(self, context=None):
        raise NotImplementedError


class ManualSelector(ModelSelector):
    """setup/model.json's `active`. Re-reads it only when the file changes."""

    def __init__(self, model_file=MODEL_FILE):
        self.model_file = Path(model_file)
        self._stamp = None
        self._name = None

    def select(self, context=None):
        stamp = self.model_file.stat().st_mtime_ns
        if stamp != self._stamp:
            self._name = json.loads(self.model_file.read_text())["active"]
            self._stamp = stamp
        return self._name


class FixedSelector(ModelSelector):
    """Always the same model. For scripts and tests."""

    def __init__(self, name):
        self.name = name

    def select(self, context=None):
        return self.name

