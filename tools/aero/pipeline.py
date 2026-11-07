'''pipeline.py: pose in, the mesh TouchDesigner draws out.

Imported by: touchdesigner/aero_td.py, tests
Run directly: no

    pipeline = AeroPipeline()
    result = pipeline.evaluate(pose, solver="panel" or "ml", speed=12.0)
    skin = pipeline.skin(result, display="cp", low=-2, high=1)
    ribbons, notes = pipeline.flow_lines(pose)
    mesh = Mesh.merge(skin, line_mesh(ribbons))      # one mesh, one Script POP

Each evaluate():
  1. asks the ModelSelector which model is on the plate (setup/model.json for
     now); on a change it loads that model's panels and network;
  2. turns the fixed wind into the plate's frame using the pose;
  3. gets Cp on every panel from the chosen solver;
  4. spreads it to the panels' corners.

The skin is the model's wetted surface (the solver's own panels), a fraction
of a millimetre off the model and coloured by pressure. Being built from the
solver's panels, it needs nothing from TouchDesigner's copy of the mesh.

The panel solver's one-off build runs on a background thread, so
TouchDesigner keeps drawing (grey, with a status line) until it is ready.
After that both solvers answer in about a millisecond. "ml" falls back to
"panel" when no trained network matches the model, and says so in the status.

Flow lines (flowlines.py) come from the panel method whichever solver
colours the surface (the network only predicts pressure):
    ribbons, notes = pipeline.flow_lines(pose)
They need a second one-off build (the velocity grid over the plate, also in
the background and cached), and are re-traced on a background thread when
the wind has turned by more than a degree; until then the last ones stay up.
'''

import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from aero import catalog, colors, flowlines, lifting, panel
from aero.geometry import thin_partners
from aero.flow import DEFAULT_SPEED, dynamic_pressure, wind_vector, wind_yaw_in_plate
from aero.surrogate import Surrogate, surrogate_file


# While a re-trace runs on its thread, Python hands control between threads
# this often (s) instead of every 5 ms. Measured in TouchDesigner with the
# model turning: 5 ms gives 60 ms frame hitches, 0.5 ms a steady ~50 fps.
TRACE_SWITCH_INTERVAL = 0.0005

# Below this share of the wind lying along the plate, the plate is close to
# face-on to the wind and the ground-mirror model no longer means much.
MIN_ALONG = 0.3


@dataclass
class Mesh:
    """Triangles to draw, in the plate frame (mm). Colours are RGBA with the
    colour already multiplied by alpha, so they blend with 'one' over
    'one minus source alpha': opaque skin, see-through dash gaps."""
    points: np.ndarray      # (V, 3)
    colors: np.ndarray      # (V, 4)
    triangles: np.ndarray   # (T, 3)

    @staticmethod
    def merge(*meshes):
        """One mesh from several, drawn in the order given."""
        meshes = [m for m in meshes if m is not None and len(m.triangles)]
        if not meshes:
            return Mesh(np.zeros((0, 3), np.float32), np.zeros((0, 4), np.float32), np.zeros((0, 3), np.uint32))
        starts = np.cumsum([0] + [len(m.points) for m in meshes[:-1]])
        return Mesh(np.concatenate([m.points for m in meshes]).astype(np.float32),
                    np.concatenate([m.colors for m in meshes]).astype(np.float32),
                    np.concatenate([m.triangles + s for m, s in zip(meshes, starts)]).astype(np.uint32))


def line_mesh(ribbons, seconds=0.0, rate=0.0, dash=1.0, low=0.0, high=2.0):
    """The flow-line ribbons as a Mesh (None stays None)."""
    if ribbons is None:
        return None
    return Mesh(ribbons.points, flowlines.ribbon_colors(ribbons, seconds, rate, dash, low, high), ribbons.triangles)


@dataclass
class Result:
    model: str
    solver: str                     # the solver that actually answered ("" if none yet)
    yaw_deg: float
    speed: float                    # m/s along the plate
    cp_panels: np.ndarray = None    # (N,) or None while nothing is ready
    cp_vertices: np.ndarray = None  # (V,)
    seconds: float = 0.0
    notes: list = field(default_factory=list)

    @property
    def ready(self):
        return self.cp_vertices is not None

    @property
    def status(self):
        head = (f"{self.model} | {self.solver or 'waiting'} | yaw {self.yaw_deg:6.1f} deg | "
                f"{self.speed:4.1f} m/s | {self.seconds * 1000:.1f} ms")
        return " | ".join([head, *self.notes])


def _stamp(name, models_dir):
    """When the model's model.json last changed (an edited scale or mount reloads it)."""
    path = Path(models_dir) / name / "model.json"
    return path.stat().st_mtime_ns if path.exists() else None


class _Loaded:
    """One model's panels, panel solution (once built) and network (if trained)."""

    def __init__(self, name, background, models_dir=catalog.MODELS_DIR):
        self.name = name
        self.stamp = _stamp(name, models_dir)
        self.info = catalog.model_info(name, models_dir)
        self.panels = self.info.panels()
        self.solution = None
        self.progress = 0.0
        self.error = None
        self.field = None
        self.field_progress = 0.0
        self.field_error = None
        self.tracer = None
        self._vertex_normals = None
        self._partners = None
        self._cancelled = False
        self.surrogate, self.surrogate_note = self._load_surrogate()
        if background:
            self.thread = threading.Thread(target=self._build, name=f"aero-panel-{name}", daemon=True)
            self.thread.start()
        else:
            self.thread = None
            self._build()

    def _load_surrogate(self):
        path = surrogate_file(self.info)
        if not path.exists():
            return None, "no trained network (python -m aero.train)"
        try:
            surrogate = Surrogate.load(path)
        except Exception as error:      # a half-written or old file should not stop the show
            return None, f"could not read {path.name}: {error}"
        if surrogate.key != self.panels.key or len(surrogate) != len(self.panels):
            return None, "network was trained on a different STL or mount: retrain (python -m aero.train)"
        if surrogate.source.startswith("panel v") and surrogate.source != f"panel v{panel.SOLVER_VERSION}":
            return None, f"network was taught by an older panel method ({surrogate.source}): retrain (python -m aero.train)"
        return surrogate, ""

    def _build(self):
        try:
            self.solution = panel.load_or_build(
                self.info, self.panels,
                progress=lambda f: setattr(self, "progress", f),
                cancel=lambda: self._cancelled)
            self.progress = 1.0
        except panel.Cancelled:
            return
        except Exception as error:
            self.error = f"panel build failed: {error}"
            return
        try:
            self.field = flowlines.load_or_build_field(
                self.info, self.panels, self.solution,
                progress=lambda f: setattr(self, "field_progress", f),
                cancel=lambda: self._cancelled)
            self.tracer = flowlines.SurfaceTracer(self.panels)
            # For the wing-loading display (~2 s on the F1): the other side of
            # each thin part, wheels left out (a tyre is a thin ring, not a wing).
            partners = thin_partners(self.panels)
            partners[lifting.wheels(self.panels)] = -1
            self._partners = partners
        except panel.Cancelled:
            pass
        except Exception as error:
            self.field_error = f"flow field build failed: {error}"

    def cancel(self):
        self._cancelled = True

    @property
    def partners(self):
        """geometry.thin_partners: the other side of each wing panel (-1: not a
        wing). Worked out by the background build; None until then."""
        return self._partners

    @property
    def vertex_normals(self):
        """Unit normal at each panel corner: the panels' normals averaged, by area."""
        if self._vertex_normals is None:
            n = np.stack([self.panels.to_vertices(self.panels.normals[:, k]) for k in range(3)], axis=1)
            self._vertex_normals = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        return self._vertex_normals


class AeroPipeline:
    def __init__(self, selector=None, background=True, models_dir=catalog.MODELS_DIR):
        self.selector = selector or catalog.ManualSelector()
        self.background = background
        self.models_dir = models_dir
        self._loaded = None
        self._lines = None          # (request, yaw, Ribbons) last traced
        self._lines_thread = None
        self.lines_done = 0         # how many have finished (TouchDesigner redraws when this changes)

    # ---- the model ----
    def model(self, context=None):
        name = self.selector.select(context)
        if (self._loaded is None or self._loaded.name != name
                or self._loaded.stamp != _stamp(name, self.models_dir)):
            if self._loaded is not None:
                self._loaded.cancel()
            self._loaded = _Loaded(name, self.background, self.models_dir)
            self._lines = None
        return self._loaded

    def reload(self):
        """Forget everything loaded (after retraining, or editing a model.json)."""
        if self._loaded is not None:
            self._loaded.cancel()
        self._loaded = None
        self._lines = None

    # ---- the flow ----
    def evaluate(self, pose, solver="panel", speed=DEFAULT_SPEED, context=None):
        started = time.perf_counter()
        loaded = self.model(context)
        yaw, along = wind_yaw_in_plate(pose)
        result = Result(loaded.name, "", yaw, float(speed) * along)
        if along < MIN_ALONG:
            result.notes.append("plate tilted nearly face-on to the wind")

        cp = None
        if solver == "ml":
            if loaded.surrogate is not None:
                cp = loaded.surrogate.predict(yaw, result.speed)
                result.solver = "ml"
                if not loaded.surrogate.in_range(result.speed):
                    low, high = loaded.surrogate.speed_range
                    result.notes.append(f"speed outside the trained {low:g}-{high:g} m/s")
            else:
                result.notes.append(loaded.surrogate_note + "; using panel")
        elif solver != "panel":
            result.notes.append(f"unknown solver '{solver}', using panel")

        if cp is None:
            if loaded.solution is not None:
                cp = loaded.solution.cp(wind_vector(yaw))
                result.solver = "panel"
            elif loaded.error:
                result.notes.append(loaded.error)
            else:
                result.notes.append(f"building panel matrices {loaded.progress * 100:.0f}%")

        if cp is not None:
            result.cp_panels = cp
            result.cp_vertices = loaded.panels.to_vertices(cp)
        result.seconds = time.perf_counter() - started
        return result

    # ---- flow lines ----
    def line_style(self, width=0.0):
        """Ribbon width, lift off the surface and point spacing (mm) for this
        model. width 0 = automatic: 1/180 of the model, but at least 1.5 mm,
        about two projector pixels at 2 m (thinner lines vanish on the model)."""
        size = flowlines.model_size(self.model().panels)
        width = width if width > 0 else max(size / 180.0, 1.5)
        return width, max(0.003 * size, 0.2), max(0.006 * size, 0.4 * width)

    def flow_lines(self, pose, surface=60, ground=28, width=0.0, turn_deg=1.0):
        """The latest Ribbons (or None) and notes. Starts a re-trace in the
        background when the wind has turned more than turn_deg or a setting
        changed; the ribbons returned meanwhile are the previous ones. The
        counts are capped so lines don't merge (flowlines.max_line_counts)."""
        loaded = self.model()
        notes = []
        if loaded.field is None:
            if loaded.field_error or loaded.error:
                notes.append(loaded.field_error or loaded.error)
            elif loaded.solution is None:
                notes.append("flow lines wait for the panel matrices")
            else:
                notes.append(f"building flow field {loaded.field_progress * 100:.0f}%")
            return None, notes
        yaw, _ = wind_yaw_in_plate(pose)
        width, lift, spacing = self.line_style(width)
        most_surface, most_ground = flowlines.max_line_counts(loaded.panels, loaded.field, width)
        surface, ground = min(int(surface), most_surface), min(int(ground), most_ground)
        request = (loaded.name, surface, ground, round(width, 4))
        current = self._lines if self._lines is not None and self._lines[0][0] == loaded.name else None
        stale = (current is None or current[0] != request
                 or abs((yaw - current[1] + 180.0) % 360.0 - 180.0) > turn_deg)
        busy = self._lines_thread is not None and self._lines_thread.is_alive()
        if stale and not busy:
            def trace():
                interval = sys.getswitchinterval()
                if self.background:
                    sys.setswitchinterval(min(interval, TRACE_SWITCH_INTERVAL))
                try:
                    if loaded.tracer is None:
                        loaded.tracer = flowlines.SurfaceTracer(loaded.panels)
                    lines = []
                    if ground > 0:
                        lines += flowlines.ground_lines(loaded.field, yaw, ground, lift=0.3 * lift, spacing=spacing)
                    if surface > 0:
                        velocity = loaded.solution.surface_velocity(wind_vector(yaw))
                        lines += loaded.tracer.lines(velocity, surface, lift=lift, spacing=spacing)
                    self._lines = (request, yaw, flowlines.ribbons(flowlines.pack_lines(lines, yaw), width))
                    self.lines_done += 1
                finally:
                    sys.setswitchinterval(interval)
            if self.background:
                self._lines_thread = threading.Thread(target=trace, name="aero-lines", daemon=True)
                self._lines_thread.start()
            else:
                trace()
                current = self._lines
        return (current[2] if current is not None else None), notes

    @property
    def lines_pending(self):
        """True while a re-trace is running (TouchDesigner keeps cooking to pick it up)."""
        return self._lines_thread is not None and self._lines_thread.is_alive()

    # ---- the skin ----
    def skin(self, result, display="cp", low=-2.0, high=1.0, gain=1.0, offset=0.0):
        """The wetted surface as a Mesh, grey until a solver is ready, coloured by
          "cp"        the pressure coefficient
          "pressure"  pressure in Pa relative to the room (q * Cp)
          "loading"   on thin parts (wings, fins, the floor), the pressure
                      difference across the part, Cp here - Cp on the other
                      side: high (red) on the side pushed, low (blue) on the
                      side pulled, so a wing's work shows from either side;
                      everything else dark.
        offset: mm off the model along the normals; 0 = automatic, just under
        the surface lines so they are drawn on top of it."""
        loaded = self.model()
        panels = loaded.panels
        if offset <= 0:
            offset = 0.3 * self.line_style()[1]
        if not result.ready:
            rgb = np.tile(colors.NO_DATA, (len(panels.vertices), 1))
        elif display == "loading":
            partner = loaded.partners
            if partner is None:                               # still being worked out: show nothing yet
                partner = np.full(len(panels), -1)
            wing = partner >= 0
            difference = np.where(wing, result.cp_panels - result.cp_panels[np.maximum(partner, 0)], 0.0)
            covered = panels.to_vertices(wing.astype(float)) > 0.5
            values = panels.to_vertices(difference, use=wing)
            rgb = np.where(covered[:, None], colors.colormap(values, low, high), colors.OFF) * gain
        else:
            values = result.cp_vertices * (dynamic_pressure(result.speed) if display == "pressure" else 1.0)
            rgb = colors.colormap(values, low, high) * gain
        return Mesh(panels.vertices + offset * loaded.vertex_normals,
                    np.column_stack([rgb, np.ones(len(rgb))]), panels.faces)
