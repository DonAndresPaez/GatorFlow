'''test_aero.py - checks the aerodynamics against answers known exactly.

Run: pytest tests/test_aero.py

A sphere in uniform potential flow has Cp = 1 - 9/4 sin^2(angle from the
wind axis). A hemisphere sitting on the ground is half of that sphere (the
ground mirror makes the other half), so it must give the same numbers: that
checks the source panels, the mirror, and dropping the base. Then the wind's
frame, the trained network, and the whole pipeline on a throwaway model.
No TouchDesigner, Kinect or OpenCV needed; the training test needs PyTorch.
'''

import json

import numpy as np
import pytest

from aero import catalog, colors, flow, panel
from aero.geometry import panels_from_triangles, read_stl
from aero.pipeline import AeroPipeline
from aero.surrogate import Surrogate, surrogate_file

RADIUS = 50.0


def _outward(triangles, outward_of):
    """Flip triangles whose winding doesn't face away from outward_of(centres)."""
    n = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    flip = np.einsum("ij,ij->i", n, outward_of(triangles.mean(axis=1))) < 0
    triangles[flip] = triangles[flip][:, [0, 2, 1]]
    return triangles


def sphere(rings=24, segments=48, hemisphere=False):
    """Latitude-longitude sphere (or hemisphere with a flat base on Z = 0)."""
    top = np.pi / 2 if hemisphere else np.pi
    theta = np.linspace(0, top, rings + 1)
    phi = np.linspace(0, 2 * np.pi, segments + 1)
    grid = RADIUS * np.stack([np.sin(theta)[:, None] * np.cos(phi), np.sin(theta)[:, None] * np.sin(phi),
                              np.cos(theta)[:, None] * np.ones_like(phi)], axis=-1)
    if hemisphere:
        grid[-1, :, 2] = 0.0
    tris = []
    for i in range(rings):
        for j in range(segments):
            a, b, c, d = grid[i, j], grid[i + 1, j], grid[i + 1, j + 1], grid[i, j + 1]
            if i > 0:
                tris.append([a, b, d])
            if i < rings - 1 or hemisphere:
                tris.append([b, c, d])
    tris = _outward(np.array(tris), lambda c: c)
    if hemisphere:
        rim = grid[-1, :-1]
        base = np.array([[np.zeros(3), rim[j], rim[(j + 1) % segments]] for j in range(segments)])
        base = _outward(base, lambda c: np.tile([0, 0, -1.0], (len(c), 1)))
        tris = np.concatenate([tris, base])
    return tris


def exact_sphere_cp(normals, wind):
    along = normals @ (np.asarray(wind) / np.linalg.norm(wind))
    return 1.0 - 2.25 * (1.0 - along ** 2)


def test_source_panel_matches_quadrature():
    tri = np.array([[0, 0, 0], [1.0, 0.2, 0.1], [0.3, 0.9, -0.1]])
    rng = np.random.default_rng(0)
    u = rng.random((300_000, 2))
    u[u.sum(1) > 1] = 1 - u[u.sum(1) > 1]
    q = tri[0] + u[:, :1] * (tri[1] - tri[0]) + u[:, 1:] * (tri[2] - tri[0])
    area = 0.5 * np.linalg.norm(np.cross(tri[1] - tri[0], tri[2] - tri[0]))
    for p in ([0.4, 0.3, 0.8], [1.5, -0.4, 0.2], [-0.5, 1.2, -0.6]):
        d = np.array(p) - q
        expected = area * np.mean(d / np.linalg.norm(d, axis=1)[:, None] ** 3, axis=0) / (4 * np.pi)
        got = panel.source_velocity(np.array([p]), tri[None])[0]
        assert np.allclose(got, expected, atol=2e-4)


def test_free_sphere():
    panels = panels_from_triangles(sphere(), ground=False)
    solution = panel.build(panels, ground=False)
    for wind in ([1, 0, 0], [0.3, -0.8, 0.5]):
        error = solution.cp(wind) - exact_sphere_cp(panels.normals, wind)
        # First-order panels on a latitude-longitude mesh (slivers at the poles).
        assert np.abs(error).mean() < 0.02
        assert np.abs(error).max() < 0.08


def test_hemisphere_on_ground_is_half_a_sphere():
    panels = panels_from_triangles(sphere(hemisphere=True), ground=True)
    assert np.all(panels.centroids[:, 2] > 0.1), "the base should have been dropped"
    solution = panel.build(panels, ground=True)
    for yaw in (0.0, 37.0, 210.0):
        wind = flow.wind_vector(yaw)
        error = solution.cp(wind) - exact_sphere_cp(panels.normals, wind)
        assert np.abs(error).mean() < 0.015
        assert np.abs(error).max() < 0.08


def test_inward_facing_stl_is_turned_round():
    tris = sphere(hemisphere=True)
    outward = panels_from_triangles(tris)
    inward = panels_from_triangles(tris[:, [0, 2, 1]])
    assert np.allclose(np.sort(outward.normals[:, 2]), np.sort(inward.normals[:, 2]))


def write_binary_stl(path, triangles):
    record = np.zeros(len(triangles), [("normal", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
    record["v"] = triangles
    path.write_bytes(b"\0" * 80 + np.uint32(len(triangles)).tobytes() + record.tobytes())


def test_stl_binary_and_ascii(tmp_path):
    tris = sphere(rings=4, segments=8)
    write_binary_stl(tmp_path / "b.stl", tris)
    lines = ["solid t"] + ["facet normal 0 0 0\nouter loop\n" + "".join(f"vertex {x} {y} {z}\n" for x, y, z in t)
                           + "endloop\nendfacet" for t in tris] + ["endsolid t"]
    (tmp_path / "a.stl").write_text("\n".join(lines))
    assert np.allclose(read_stl(tmp_path / "b.stl"), tris, atol=1e-4)
    assert np.allclose(read_stl(tmp_path / "a.stl"), tris)


def test_wind_in_plate_frame():
    # At home the plate frame is the world: wind +X is yaw 0.
    assert flow.wind_yaw_in_plate([0] * 6)[0] == pytest.approx(0.0)
    # Plate turned 90 degrees anticlockwise: the wind now comes along its -Y.
    yaw, along = flow.wind_yaw_in_plate([10, 20, 0, 0, 0, 90])
    assert yaw == pytest.approx(-90.0) and along == pytest.approx(1.0)
    # Tilted 30 degrees about the wind's own axis: still yaw 0, all along.
    assert flow.wind_yaw_in_plate([0, 0, 0, 30, 0, 0]) == pytest.approx((0.0, 1.0))
    # Tilted 60 degrees about Y: only cos 60 of the wind lies along the plate.
    assert flow.wind_yaw_in_plate([0, 0, 0, 0, 60, 0])[1] == pytest.approx(0.5)


def test_colormap_ends():
    rgb = colors.colormap([-5.0, -2.0, 1.0, 9.0], -2.0, 1.0)
    assert np.allclose(rgb[0], colors.STOPS[0]) and np.allclose(rgb[1], colors.STOPS[0])
    assert np.allclose(rgb[2], colors.STOPS[-1]) and np.allclose(rgb[3], colors.STOPS[-1])


@pytest.fixture
def dome_models(tmp_path):
    """A models/ folder with one model, a 50 mm dome."""
    folder = tmp_path / "Dome"
    folder.mkdir()
    write_binary_stl(folder / "Dome.stl", sphere(rings=12, segments=24, hemisphere=True))
    (folder / "model.json").write_text(json.dumps({"name": "Dome", "stl": "Dome.stl", "sizeMm": [100, 100, 50]}))
    return tmp_path


def test_surrogate_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    original = Surrogate([rng.normal(size=(3, 8)), rng.normal(size=(8, 4))], [rng.normal(size=8), rng.normal(size=4)],
                         rng.normal(size=20), rng.normal(size=(4, 20)), rng.random(4) + 0.5, (2.0, 20.0), "k", "test")
    original.save(tmp_path / "s.npz")
    loaded = Surrogate.load(tmp_path / "s.npz")
    assert loaded.key == "k" and loaded.speed_range == (2.0, 20.0)
    assert np.allclose(loaded.predict(33.0, 7.0), original.predict(33.0, 7.0), atol=1e-4)


def test_pipeline_panel_and_ml_fallback(dome_models):
    aero = AeroPipeline(catalog.FixedSelector("Dome"), background=False, models_dir=dome_models)
    result = aero.evaluate([0, 0, 0, 0, 0, 0], solver="ml", speed=12.0)
    assert result.solver == "panel" and "no trained network" in result.status
    loaded = aero.model()
    assert result.cp_panels.max() == pytest.approx(1.0, abs=0.05)       # stagnation point
    assert result.cp_panels.min() == pytest.approx(-1.25, abs=0.08)     # top of the dome

    assert panel.cache_file(loaded.info, loaded.panels).exists()


def test_model_scale(dome_models):
    """"scale" grows the placed model about the plate centre; Cp doesn't depend on size."""
    from aero.geometry import mount_transform
    T = mount_transform([10, -5, 3], 90.0, scale=2.0)
    assert np.allclose(T @ [1, 0, 0, 1], [20, -8, 6, 1])        # = 2 x (rotated point + offset)

    small = catalog.model_info("Dome", dome_models)
    aero = AeroPipeline(catalog.FixedSelector("Dome"), background=False, models_dir=dome_models)
    before = aero.model()
    path = dome_models / "Dome" / "model.json"
    path.write_text(json.dumps({**json.loads(path.read_text()), "scale": 2.5}))
    big = catalog.model_info("Dome", dome_models)
    assert big.scale == 2.5
    after = aero.model()                    # the edited model.json is picked up without a reload
    assert after is not before and np.allclose(after.panels.vertices, 2.5 * before.panels.vertices)
    a, b = small.panels(), big.panels()
    assert np.allclose(b.vertices, 2.5 * a.vertices) and a.key != b.key
    assert b.vertices[:, 2].min() == pytest.approx(0.0, abs=1e-9)  # still standing on the plate
    wind = flow.wind_vector(0.0)
    assert np.allclose(panel.build(b).cp(wind), panel.build(a).cp(wind), atol=1e-6)

    def obj_points(obj):
        return np.array([line.split()[1:] for line in obj.read_text().splitlines() if line.startswith("v ")], float)

    # The placed OBJ (for a .toe with no mount SOP) is the panels' mesh, and follows model.json.
    assert np.allclose(obj_points(big.placed_obj()), b.vertices, atol=1e-5)
    path.write_text(json.dumps({**json.loads(path.read_text()), "scale": 1}))
    assert np.allclose(obj_points(catalog.model_info("Dome", dome_models).placed_obj()), a.vertices, atol=1e-5)

    path.write_text(json.dumps({**json.loads(path.read_text()), "scale": 0}))
    with pytest.raises(ValueError):
        catalog.model_info("Dome", dome_models)


def test_trained_surrogate_matches_panel(dome_models):
    pytest.importorskip("torch")
    from aero import train

    info = catalog.model_info("Dome", dome_models)
    panels = info.panels()
    solution = panel.load_or_build(info, panels)
    yaw, speed, cp = train.panel_dataset(solution, 800, (2.0, 20.0))
    surrogate = train.fit(yaw, speed, cp, (2.0, 20.0), epochs=1500, log=lambda *_: None)
    surrogate.key = panels.key
    surrogate.save(surrogate_file(info))

    aero = AeroPipeline(catalog.FixedSelector("Dome"), background=False, models_dir=dome_models)
    pose = [0, 0, 0, 0, 0, 25]
    ml = aero.evaluate(pose, solver="ml", speed=8.0)
    exact = aero.evaluate(pose, solver="panel", speed=8.0)
    assert ml.solver == "ml" and not ml.notes
    assert np.sqrt(np.mean((ml.cp_panels - exact.cp_panels) ** 2)) < 0.02


def test_obj_copy_for_touchdesigner(tmp_path):
    # TouchDesigner's File In SOP can't read STL; load_setup.py loads this OBJ instead.
    from aero.geometry import MIN_PANELS, merge_corners, obj_for, refine

    tris = sphere(rings=6, segments=12, hemisphere=True)
    write_binary_stl(tmp_path / "m.stl", tris)
    obj = obj_for(tmp_path / "m.stl", tmp_path / "aero")
    lines = obj.read_text().splitlines()
    v = np.array([l.split()[1:] for l in lines if l.startswith("v ")], float)
    f = np.array([l.split()[1:] for l in lines if l.startswith("f ")], int) - 1
    # Coarse, so refined exactly as the solver's panels are.
    fine = refine(tris.astype(np.float32).astype(np.float64))
    assert len(f) == len(fine) >= MIN_PANELS
    assert np.allclose(v[f], fine, atol=1e-5)
    assert len(v) == len(merge_corners(fine)[0])


def test_refine_keeps_a_cube_a_cube():
    from aero.geometry import refine

    c = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], float) * 20
    faces = [(0, 2, 3), (0, 3, 1), (4, 5, 7), (4, 7, 6), (0, 1, 5), (0, 5, 4),
             (2, 6, 7), (2, 7, 3), (0, 4, 6), (0, 6, 2), (1, 3, 7), (1, 7, 5)]
    tris = refine(c[np.array(faces)])
    assert len(tris) == 12 * 4 ** 4
    area = 0.5 * np.linalg.norm(np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]), axis=1)
    assert area.sum() == pytest.approx(6 * 20 * 20)
    panels = panels_from_triangles(tris)                 # base dropped: 5 faces left
    assert len(panels) == 5 * 2 * 4 ** 4


# ---- flow lines ----

@pytest.fixture(scope="module")
def dome_solution():
    panels = panels_from_triangles(sphere(hemisphere=True), ground=True)
    return panels, panel.build(panels, ground=True)


def test_velocity_off_the_surface_matches_the_sphere(dome_solution):
    from aero.panel import velocity_matrices

    panels, solution = dome_solution
    # Potential flow past a sphere: phi = U x (1 + R^3 / (2 r^3)).
    points = np.array([[0, 0, 1.5 * RADIUS], [0, 1.3 * RADIUS, 2.0], [2.0 * RADIUS, 0.4 * RADIUS, 10.0], [0, 0, 20 * RADIUS]])
    G, _ = velocity_matrices(panels, solution, points)
    for p, g in zip(points, G):
        r = np.linalg.norm(p)
        x = p[0]
        exact = np.array([1 + RADIUS ** 3 / (2 * r ** 3), 0, 0]) - 1.5 * RADIUS ** 3 * x * p / r ** 5
        assert np.allclose(g @ [1, 0, 0], exact, atol=0.02)


def test_ground_lines_go_round_the_dome(dome_solution):
    from aero import flowlines

    panels, solution = dome_solution
    field = flowlines.build_ground_field(panels, solution, n=64)
    assert field.inside[32, 32] and not field.inside[0, 0]
    lines = flowlines.ground_lines(field, 0.0, count=12)
    half = field.radius / np.sqrt(2)
    upstream = [line for line in lines if line[0][0, 0] < -0.95 * half]
    behind = [line for line in lines if line[0][0, 0] > -0.95 * half]
    assert len(upstream) >= 10
    for pts, normals, speed, ground in lines:
        assert ground and np.allclose(normals, [0, 0, 1])
        assert np.hypot(pts[:, 0], pts[:, 1]).min() > 0.9 * RADIUS       # never through the dome
    for pts, normals, speed, ground in upstream:
        assert speed[0] == pytest.approx(1.0, abs=0.1)                     # wind speed far upstream
    # The line that stops at the front stagnation point leaves a gap behind
    # the dome; the downstream rake fills it with a line leaving the back.
    assert behind and all(pts[-1, 0] > 0.95 * half and pts[0, 0] > 0 for pts, *_ in behind)
    # Mirror-symmetric about the wind axis.
    first, last = upstream[0][0], upstream[-1][0]
    n = min(len(first), len(last))
    assert np.allclose(first[:n, 1], -last[:n, 1], atol=0.05 * RADIUS)


def test_surface_lines_stay_on_the_dome_and_follow_the_wind(dome_solution):
    from aero import flowlines

    panels, solution = dome_solution
    tracer = flowlines.SurfaceTracer(panels)
    lines = tracer.lines(solution.surface_velocity([1, 0, 0]), count=30, lift=0.5)
    assert len(lines) >= 20
    for pts, normals, speed, ground in lines:
        r = np.linalg.norm(pts, axis=1)
        assert not ground and np.all(np.abs(r - RADIUS) < 0.03 * RADIUS + 0.5)
        assert pts[-1, 0] > pts[0, 0]                                       # downstream is +X
        assert speed.max() < 1.6                                            # sphere peak is 1.5


def test_ribbons_and_dashes():
    from aero import flowlines

    p = np.column_stack([np.linspace(0, 10, 11), np.zeros(11), np.zeros(11)])
    lines = flowlines.pack_lines([(p, np.tile([0, 0, 1.0], (11, 1)), np.ones(11), True),
                             (p + [0, 5, 0], np.tile([0, 0, 1.0], (11, 1)), np.full(11, 0.5), True)], 0.0)
    assert lines.time[lines.starts[1] - 1] == pytest.approx(10.0)          # 10 mm at wind speed
    assert lines.time[-1] == pytest.approx(20.0)                           # half speed: twice the time
    rib = flowlines.ribbons(lines, width=2.0)
    assert rib.points.shape == (44, 3) and rib.triangles.shape == (40, 3)
    assert np.allclose(np.abs(rib.points[0::2, 1] - rib.points[1::2, 1]), 2.0)
    solid = flowlines.ribbon_colors(rib)
    moving = [flowlines.ribbon_colors(rib, t, rate=4.0, dash=5.0) for t in (0.0, 0.3)]
    assert solid.shape == (44, 4) and np.all(solid >= 0) and np.all(solid <= 1)
    assert np.all(moving[0][:, :3] <= solid[:, :3] + 1e-6) and not np.allclose(moving[0], moving[1])


def test_pipeline_flow_lines(dome_models):
    aero = AeroPipeline(catalog.FixedSelector("Dome"), background=False, models_dir=dome_models)
    ribbons, notes = aero.flow_lines([0, 0, 0, 0, 0, 30], surface=20, ground=10)
    assert ribbons is not None and not notes and len(ribbons.triangles) > 100
    again, _ = aero.flow_lines([0, 0, 0, 0, 0, 30.5], surface=20, ground=10)
    assert again is ribbons                                                # within a degree: kept
    turned, _ = aero.flow_lines([0, 0, 0, 0, 0, 40], surface=20, ground=10)
    assert turned is not ribbons


# ---- what TouchDesigner relies on ----

def _euler(order, angles_deg):
    """Rotation from TouchDesigner's rx, ry, rz (always X, Y, Z angles) and its
    rotate order: 'xzy' applies the X turn, then Z, then Y."""
    axes = {"x": 0, "y": 1, "z": 2}
    R = np.eye(3)
    for axis in order:
        angle = np.radians(angles_deg[axes[axis]])
        c, s = np.cos(angle), np.sin(angle)
        i, j = [k for k in range(3) if k != axes[axis]]
        step = np.eye(3)
        step[i, i], step[i, j], step[j, i], step[j, j] = c, -s, s, c
        if axis == "y":
            step = step.T                        # about Y, +angle turns Z toward X
        R = step @ R
    return R


def test_y_up_pose_is_undone():
    # ProjectMappingSetUp.toe's OSC callback: geo1 r = (rx, rz, -ry), rotate order xzy.
    # That has to be the same rotation as the tracker's, seen with Y up (C = -90 about X).
    C = _euler("xyz", [-90, 0, 0])
    rng = np.random.default_rng(3)
    for rx, ry, rz in rng.uniform(-60, 60, (10, 3)):
        g = (rx, rz, -ry)
        assert np.allclose(_euler("xzy", g), C @ flow.rotation_from_pose([0, 0, 0, rx, ry, rz]) @ C.T)
        back = flow.pose_from_y_up(0.1, 0.2, 0.3, *g)
        assert np.allclose(back[3:], [rx, ry, rz])
        assert flow.wind_yaw_in_plate(back) == pytest.approx(flow.wind_yaw_in_plate([0, 0, 0, rx, ry, rz]))


def test_skin_sits_on_the_model_coloured_by_pressure(dome_models):
    from aero.pipeline import Result

    aero = AeroPipeline(catalog.FixedSelector("Dome"), background=False, models_dir=dome_models)
    result = aero.evaluate([0] * 6, speed=12.0)
    panels = aero.model().panels
    skin = aero.skin(result, offset=0.5)
    used = np.unique(panels.faces)
    r = np.linalg.norm(skin.points[used], axis=1)
    assert np.allclose(r, RADIUS + 0.5, atol=0.02 * RADIUS)                # 0.5 mm off the dome
    assert np.array_equal(skin.triangles, panels.faces)                   # the wetted panels, base dropped
    assert np.allclose(skin.colors[:, 3], 1.0)                             # opaque
    # Pressure display: q * Cp, so a range in Pa that is q times the Cp range gives the same colours.
    q = flow.dynamic_pressure(12.0)
    assert np.allclose(aero.skin(result, "pressure", -2 * q, q).colors, aero.skin(result).colors)
    assert np.allclose(aero.skin(result, gain=0.5).colors[:, :3], 0.5 * aero.skin(result).colors[:, :3])
    waiting = aero.skin(Result("Dome", "", 0.0, 10.0))                     # no solver answer yet: grey
    assert np.allclose(waiting.colors[:, :3], colors.NO_DATA)


def test_skin_and_lines_merge_into_one_mesh(dome_models):
    from aero.pipeline import Mesh, line_mesh

    aero = AeroPipeline(catalog.FixedSelector("Dome"), background=False, models_dir=dome_models)
    skin = aero.skin(aero.evaluate([0] * 6))
    ribbons, _ = aero.flow_lines([0] * 6, surface=20, ground=10)
    lines = line_mesh(ribbons, seconds=0.3, rate=4.0, dash=5.0)
    mesh = Mesh.merge(skin, lines, None)
    n = len(skin.points)
    assert len(mesh.points) == n + len(lines.points)
    assert np.array_equal(mesh.triangles[:len(skin.triangles)], skin.triangles)     # skin first
    assert np.array_equal(mesh.triangles[len(skin.triangles):], lines.triangles + n)
    assert mesh.triangles.max() < len(mesh.points)
    # Lines: premultiplied, so colour never exceeds alpha; dash gaps see-through.
    a = lines.colors[:, 3]
    assert np.all(lines.colors[:, :3] <= a[:, None] + 1e-6) and a.min() < 0.2 and a.max() > 0.9
    assert Mesh.merge(None).points.shape == (0, 3)


def test_line_counts_are_capped_for_wide_lines(dome_models):
    from aero import flowlines

    aero = AeroPipeline(catalog.FixedSelector("Dome"), background=False, models_dir=dome_models)
    loaded = aero.model()
    wide = flowlines.max_line_counts(loaded.panels, loaded.field, width=10.0)
    thin = flowlines.max_line_counts(loaded.panels, loaded.field, width=1.5)
    assert wide[0] < thin[0] and wide[1] < thin[1]
    aero.flow_lines([0] * 6, surface=500, ground=500, width=10.0)
    request = aero._lines[0]
    assert request[1:3] == wide                                            # (model, surface, ground, width)


# ---- detailed and multi-part models ----

def _box(lo, hi):
    """Closed box, faces out, as (12, 3, 3) triangles."""
    c = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])], float)
    faces = [(0, 2, 3), (0, 3, 1), (4, 5, 7), (4, 7, 6), (0, 1, 5), (0, 5, 4),
             (2, 6, 7), (2, 7, 3), (0, 4, 6), (0, 6, 2), (1, 3, 7), (1, 7, 5)]
    return c[np.array(faces)][:, [0, 2, 1]]


def test_faces_across_a_print_gap_are_not_wetted():
    from aero.geometry import refine

    def wetted_x_faces(gap):
        # Two blocks side by side on the plate, `gap` mm apart along X.
        tris = refine(np.concatenate([_box((-20, -10, 0), (0, 10, 10)), _box((gap, -10, 0), (gap + 20, 10, 10))]))
        p = panels_from_triangles(tris)
        facing_gap = (np.abs(p.normals[:, 0]) > 0.9) & (np.abs(p.centroids[:, 0] - gap / 2) < gap)
        return facing_gap.sum(), len(p)

    assert wetted_x_faces(0.5)[0] == 0                                     # the two sides across 0.5 mm: gone
    across_3mm, total = wetted_x_faces(3.0)
    assert across_3mm > 0 and total == wetted_x_faces(3.0)[1]              # 3 mm is real air: kept
    # A thin plate (one shell, both sides 0.5 mm apart) is a wing, not a gap.
    plate = panels_from_triangles(refine(_box((-20, -20, 5), (20, 20, 5.5))))
    assert np.sum(np.abs(plate.normals[:, 2]) > 0.9) == len(plate) - np.sum(np.abs(plate.normals[:, 2]) < 0.1)
    assert np.any(plate.normals[:, 2] < -0.9) and np.any(plate.normals[:, 2] > 0.9)


def test_shells_closed_volume_and_stl_round_trip(tmp_path):
    from aero.geometry import is_closed, merge_corners, read_stl, shells, signed_volume, write_stl

    tris = np.concatenate([_box((0, 0, 0), (1, 2, 3)), _box((5, 0, 0), (6, 1, 1))])
    v, f = merge_corners(tris)
    parts = shells(f)
    assert sorted(len(p) for p in parts) == [12, 12]
    assert is_closed(f) and not is_closed(f[1:])                            # one triangle missing: a hole
    assert signed_volume(tris) == pytest.approx(7.0)
    assert signed_volume(tris[:, [0, 2, 1]]) == pytest.approx(-7.0)         # turned inside out
    write_stl(tmp_path / "b.stl", tris)
    assert np.allclose(read_stl(tmp_path / "b.stl"), tris, atol=1e-5)


def test_decimate_keeps_shells_closed_and_drops_cavities():
    pytest.importorskip("fast_simplification")
    from aero.decimate import decimate
    from aero.geometry import is_closed, merge_corners, shells, signed_volume

    ball = sphere(rings=40, segments=80)                                   # ~6200 triangles
    cavity = (sphere(rings=6, segments=12) * 0.2)[:, [0, 2, 1]]            # small, inside out
    out = decimate(np.concatenate([ball, cavity]), target=1000, log=lambda *_: None)
    v, f = merge_corners(out)
    assert len(shells(f)) == 1 and is_closed(f)                             # the cavity is gone
    assert 900 <= len(out) <= 1100
    assert signed_volume(out) == pytest.approx(signed_volume(ball), rel=0.02)


# ---- wings: the lifting panel method ----

def naca0012_wing(chord=100.0, span=600.0, alpha_deg=5.0, height=0.0, n_chord=20, n_span=16, cut=1.0):
    """Closed rectangular NACA 0012 wing along Y: leading edge at x=0, sharp
    trailing edge at x=chord, pitched nose-up by alpha about the quarter chord.
    cut < 1: the section stops at cut * chord with a flat back face (a blunt
    trailing edge, as printable wings have)."""
    x = cut * 0.5 * (1 - np.cos(np.linspace(0, np.pi, n_chord + 1)))
    half = 0.6 * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x ** 2 + 0.2843 * x ** 3 - 0.1036 * x ** 4)
    lower = x[1:-1] if cut == 1.0 else x[1:]                                # a cut section keeps its own back corner
    loop = chord * np.concatenate([np.column_stack([x[::-1], half[::-1]]),
                                   np.column_stack([lower, -half[1:len(lower) + 1]])])
    pts = np.array([[[px, y, pz] for px, pz in loop] for y in np.linspace(-span / 2, span / 2, n_span + 1)])
    m, tris = len(loop), []
    for j in range(n_span):
        for i in range(m):
            a, b, c, d = pts[j, i], pts[j, (i + 1) % m], pts[j + 1, (i + 1) % m], pts[j + 1, i]
            tris += [[a, b, c], [a, c, d]]
    for j, flip in ((0, False), (n_span, True)):                           # tip caps
        centre = pts[j].mean(axis=0)
        tris += [([centre, pts[j, i], pts[j, (i + 1) % m]])[::-1 if flip else 1] for i in range(m)]
    a = np.radians(alpha_deg)
    rot = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    q = np.array([0.25 * chord, 0, 0])
    return (np.array(tris) - q) @ rot.T + q + [0, 0, height]


def _lift_coefficient(alpha, height=None, cut=1.0):
    from aero.geometry import panel_rings, thin_partners
    from aero import lifting

    p = panels_from_triangles(naca0012_wing(alpha_deg=alpha, height=height or 0.0, cut=cut), ground=height is not None)
    corners, upper, lower = lifting.trailing_edges(p, thin_partners(p))
    _, M_lift, _ = lifting.build(p, corners, upper, lower, panel_rings(p, lifting.GRADIENT_RING), height is not None)
    v = M_lift @ np.array([1.0, 0, 0])
    cp = 1 - np.einsum("ij,ij->i", v, v)
    return -(cp * p.area_normals[:, 2]).sum() / (100.0 * 600.0), len(corners)


def test_lifting_kernels_match_integration_and_each_other():
    from aero.lifting import doublet_velocity, potentials

    tri = np.array([[0, 0, 0], [1.0, 0.2, 0.1], [0.3, 0.9, -0.1]])
    n = np.cross(tri[1] - tri[0], tri[2] - tri[0])
    n /= np.linalg.norm(n)
    rng = np.random.default_rng(1)
    u = rng.random((300_000, 2))
    u[u.sum(1) > 1] = 1 - u[u.sum(1) > 1]
    q = tri[0] + u[:, :1] * (tri[1] - tri[0]) + u[:, 1:] * (tri[2] - tri[0])
    area = 0.5 * np.linalg.norm(np.cross(tri[1] - tri[0], tri[2] - tri[0]))
    for p in ([0.4, 0.3, 0.8], [1.5, -0.4, 0.2], [0.4, 0.3, -0.5]):
        p = np.array(p)
        d = p - q
        r = np.linalg.norm(d, axis=1)
        s, dd = potentials(p[None], tri[None])
        assert s[0] == pytest.approx(-area * np.mean(1 / r) / (4 * np.pi), rel=2e-3)
        assert dd[0] == pytest.approx(area * np.mean(d @ n / r ** 3) / (4 * np.pi), rel=2e-3, abs=1e-5)
        h = 1e-5                                                            # doublet velocity = grad(potential)
        grad = [(potentials((p + h * e)[None], tri[None])[1][0] - potentials((p - h * e)[None], tri[None])[1][0]) / (2 * h)
                for e in np.eye(3)]
        assert np.allclose(doublet_velocity(p[None], tri[None])[0], grad, atol=1e-6)
    c = tri.mean(0)
    jump = potentials((c + 1e-7 * n)[None], tri[None])[1][0] - potentials((c - 1e-7 * n)[None], tri[None])[1][0]
    assert jump == pytest.approx(1.0) and potentials(c[None], tri[None], np.array([True]))[1][0] == -0.5


def test_doublet_formulation_without_wake_matches_the_sphere():
    from aero import lifting
    from aero.geometry import panel_rings

    p = panels_from_triangles(sphere(hemisphere=True), ground=True)
    none = np.zeros((0, 2), int), np.zeros(0, int), np.zeros(0, int)
    M, _, _ = lifting.build(p, *none, panel_rings(p, lifting.GRADIENT_RING), ground=True)
    for yaw in (0.0, 37.0):
        wind = flow.wind_vector(yaw)
        v = M @ wind
        error = 1 - np.einsum("ij,ij->i", v, v) - exact_sphere_cp(p.normals, wind)
        assert np.abs(error).mean() < 0.015 and np.abs(error).max() < 0.06


def test_wing_lift_matches_lifting_line_theory():
    cl0, edges = _lift_coefficient(0.0)
    cl5, _ = _lift_coefficient(5.0)
    assert edges == 16                                                     # one trailing edge per span strip
    assert abs(cl0) < 1e-3                                                 # symmetric section, no incidence
    theory = 2 * np.pi * np.radians(5.0) * 6 / (6 + 2)                     # aspect ratio 6
    assert cl5 == pytest.approx(theory, rel=0.1)


def test_blunt_trailing_edge_lifts_like_a_sharp_one():
    blunt, edges = _lift_coefficient(5.0, cut=0.9)                          # flat back face, 2.7 mm tall
    sharp, _ = _lift_coefficient(5.0)
    assert edges == 16                                                      # one wake per strip, off the back face
    assert blunt == pytest.approx(0.9 * sharp, rel=0.2)                     # 90% of the chord, about 90% of the lift


def test_square_backed_solids_have_no_trailing_edges():
    from aero import lifting
    from aero.geometry import refine, thin_partners

    for tris in (refine(_box((-10, -10, 0), (10, 10, 20))), sphere(hemisphere=True)):
        p = panels_from_triangles(tris)
        corners, _, _ = lifting.trailing_edges(p, thin_partners(p))
        assert len(corners) == 0


def test_inverted_wing_makes_more_downforce_near_the_ground():
    free, _ = _lift_coefficient(-5.0)
    near, _ = _lift_coefficient(-5.0, height=30.0)                         # quarter chord 0.3 chords up
    assert free < 0 and near < 1.1 * free                                  # at least 10% more downforce


def test_lift_fades_out_as_the_wind_turns_to_the_side():
    w = [float(panel.lift_weight(flow.wind_vector(a))) for a in (0, 30, 40, 57, 75, 90, 180)]
    assert w[0] == w[1] == w[2] == 1.0 and 0 < w[3] < 1 and w[4] == w[5] == w[6] == 0.0
    assert float(panel.lift_weight(flow.wind_vector(-30))) == 1.0


def test_pipeline_uses_lift_for_a_model_with_wings(tmp_path):
    folder = tmp_path / "Wing"
    folder.mkdir()
    write_binary_stl(folder / "Wing.stl", naca0012_wing(alpha_deg=-5.0, height=40.0, span=300.0))
    (folder / "model.json").write_text(json.dumps({"name": "Wing", "stl": "Wing.stl", "sizeMm": [100, 300, 50], "kutta": True}))
    aero = AeroPipeline(catalog.FixedSelector("Wing"), background=False, models_dir=tmp_path)
    loaded = aero.model()
    assert loaded.solution.M_lift is not None and loaded.field.G_lift is not None
    ahead, side = aero.evaluate([0] * 6), aero.evaluate([0, 0, 0, 0, 0, 90])
    z = loaded.panels.area_normals[:, 2]
    assert -(ahead.cp_panels * z).sum() < -(side.cp_panels * z).sum() - 1.0   # downforce from ahead, gone side-on
    ribbons, notes = aero.flow_lines([0] * 6, surface=10, ground=8)
    assert ribbons is not None and not notes


def test_wing_loading_display_shows_the_pressure_side(tmp_path):
    from aero.geometry import thin_partners

    folder = tmp_path / "Wing"
    folder.mkdir()
    write_binary_stl(folder / "Wing.stl", naca0012_wing(alpha_deg=-5.0, height=40.0, span=300.0))
    (folder / "model.json").write_text(json.dumps({"name": "Wing", "stl": "Wing.stl", "sizeMm": [100, 300, 50], "kutta": True}))
    aero = AeroPipeline(catalog.FixedSelector("Wing"), background=False, models_dir=tmp_path)
    result = aero.evaluate([0] * 6)
    panels = aero.model().panels
    partner = thin_partners(panels)
    wing = partner >= 0
    top = wing & (panels.normals[:, 2] > 0.5)
    loading = result.cp_panels - result.cp_panels[np.maximum(partner, 0)]
    assert wing.sum() > 100
    assert np.mean(loading[top]) > 0.1                     # an inverted wing: its top is the pressure side
    skin = aero.skin(result, display="loading", low=-1.5, high=1.5)
    assert skin.colors.shape == (len(panels.vertices), 4)
    off = ~(panels.to_vertices(wing.astype(float)) > 0.5)
    used = np.zeros(len(panels.vertices), bool)
    used[panels.faces.ravel()] = True
    assert np.allclose(skin.colors[off & used, :3], colors.OFF)           # not a wing: dark
