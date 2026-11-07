'''flow.py: the wind, and what it looks like from the model's point of view.

Imported by: aero/pipeline.py, aero/train.py, tests
Run directly: no

The wind is fixed in the room (WIND_DIRECTION_WORLD). The model turns on the
plate, so the solvers need the wind in the PLATE frame. The pose is the one
KinectReader sends and geo1 carries: [tx, ty, tz, rx, ry, rz], degrees,
rotation order X then Y then Z (common/config.py EULER_ORDER).

calibration.set_origin makes the plate's home spot read as all zeros, so the
TouchDesigner world IS the plate frame at home: X to the plate's right edge,
Y to its far edge, Z up. WIND_DIRECTION_WORLD is in those axes.

The plate is the ground, and the ground mirror only works for wind along it,
so the wind is projected onto the plate. The angle in the plate (yaw) is all
the solvers see; a tilted plate only changes which way is "along".
'''

import numpy as np

# ---- the wind tunnel: change these to match where the fan is ----
# Direction the air moves, in the world at home. (1, 0, 0): blowing from the
# plate's left edge toward its right edge.
WIND_DIRECTION_WORLD = (1.0, 0.0, 0.0)
AIR_DENSITY = 1.204             # kg/m^3, air at 20 C
DEFAULT_SPEED = 10.0            # m/s, until TouchDesigner says otherwise


def rotation_from_pose(pose):
    """3x3 plate -> world from [tx, ty, tz, rx, ry, rz], X applied first
    (scipy 'xyz' extrinsic, TouchDesigner's 'Rx Ry Rz')."""
    rx, ry, rz = np.radians(np.asarray(pose, np.float64)[3:6])
    cx, sx, cy, sy, cz, sz = np.cos(rx), np.sin(rx), np.cos(ry), np.sin(ry), np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def wind_yaw_in_plate(pose, wind_world=WIND_DIRECTION_WORLD):
    """The wind's direction along the plate, as an angle in degrees from the
    plate's +X toward +Y. Returns (yaw_deg, along) where `along` is how much of
    the wind lies in the plate (1 when flat, 0 when the wind hits it face-on)."""
    wind = np.asarray(wind_world, np.float64)
    wind = wind / np.linalg.norm(wind)
    local = rotation_from_pose(pose).T @ wind
    along = float(np.hypot(local[0], local[1]))
    if along < 1e-6:
        return 0.0, 0.0
    return float(np.degrees(np.arctan2(local[1], local[0]))), along


def pose_from_y_up(tx, ty, tz, rx, ry, rz):
    """The tracker's pose back from geo1's parameters when an OSC callback has
    turned it into TouchDesigner's Y up before applying it, as
    ProjectMappingSetUp.toe's does: t -> (tx, tz, -ty) * scale and
    r -> (rx, rz, -ry) with rotate order xzy. The rotation comes back exactly;
    the translation stays in TouchDesigner units (only the rotation is used)."""
    return [tx, -tz, ty, rx, -rz, ry]


def wind_vector(yaw_deg):
    """Unit wind along the plate for a yaw."""
    a = np.radians(yaw_deg)
    return np.array([np.cos(a), np.sin(a), 0.0])


def dynamic_pressure(speed, density=AIR_DENSITY):
    """q = rho V^2 / 2, Pa. Pressure on the surface is q * Cp (relative to the room)."""
    return 0.5 * density * float(speed) ** 2
