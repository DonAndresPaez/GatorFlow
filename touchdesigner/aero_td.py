'''aero_td.py: runs the aerodynamics (tools/aero/) inside TouchDesigner: a skin
over the model coloured by pressure, with moving flow lines over it and the
plate, drawn by one Script POP.

  OSC pose -> geo1's transform -> this reads it -> tools/aero
     -> geo1/aero/aero_pop (Script POP)  skin + flow-line ribbons, one mesh
  -> cam1 -> projector

SETUP (once per .toe, which lives in this folder):
  1. A Text DAT named `aero_td`, File parameter: aero_td.py, "Sync to File" on.
  2. In the Textport:   op('aero_td').module.install()
     It adds what is missing:
       geo1/aero              Geometry COMP inside geo1, so it moves with the
                              plate. When the model's File In SOP feeds a
                              Transform SOP that turns the mesh into the
                              scene's axes and units (ProjectMappingSetUp.toe's
                              filein1 -> transform1), it copies that transform
                              by expression; otherwise it scales mm to
                              load_setup.py's TD_UNITS_PER_MM
       geo1/aero/aero_pop     Script POP (callbacks: aero_pop.py): the skin and
                              the lines, Aero and Flow lines parameter pages
       geo1/aero/aero_mat     Constant MAT: point colours, blended so the dash
                              gaps show the skin and black adds no light
       aero_frames            Execute DAT calling on_frame_start() every frame
       aero_status            Text DAT with the live status line
     and makes geo1's own material black when it is a Constant MAT (or adds a
     black one): the skin carries the colours, and black projects nothing.
     It replaces the older aero_color Script SOP and geo1/flow if it finds
     them (the File In SOP is reconnected to what aero_color fed).
     geo1 itself, its File In SOP and `mount` are load_setup.py's (made here
     if missing); the File In SOP can also sit beside geo1.
  3. Save the .toe.

EVERYDAY USE (on geo1/aero/aero_pop):
  Aero page: Solver (Panel method / ML surrogate), Wind speed, Display (Cp,
    Pa, or Wing loading for models with lift on), colour ranges, Heatmap
    brightness (turn it down so the lines stand out), Skin offset, Reload.
  Flow lines page: Surface lines / Ground lines (on/off and how many), Line
    width (mm, 0 = automatic), Animate, Flow rate, Dash length, speed colours.
  Export a slider or CHOP to Wind speed to drive it live. Swapping models:
  setup/model.json as usual. After retraining or editing tools/aero: Reload.

Python: needs only NumPy, which TouchDesigner ships. tools/ is put on sys.path
so `import aero` finds tools/aero/.
'''

import importlib
import os
import sys
import time
from pathlib import Path

import numpy as np

# ---- names in the network: change these if the .toe uses different ones ----
GEO = '/project1/geo1'
AERO_GEO = GEO + '/aero'
AERO_POP = AERO_GEO + '/aero_pop'
STATUS_DAT = '/project1/aero_status'
FRAMES_DAT = '/project1/aero_frames'
LOAD_SETUP = '/project1/load_setup'
OLD_SCRIPT_SOP = 'aero_color'      # replaced by aero_pop: removed by install()
OLD_FLOW_GEO = GEO + '/flow'

TRANSFORM_PARS = ('tx', 'ty', 'tz', 'rx', 'ry', 'rz')
POSE_AXES = 'auto'         # how geo1 carries the OSC pose: 'auto', 'tracker' or 'y_up' (see _pose)

# TouchDesigner keeps hold of Python between cooks, so the background builds
# and re-traces barely run unless the main thread lets go. While one is
# pending, each frame start sleeps this long (s) to give them a turn.
# Measured: a re-trace then lands in ~0.3 s instead of several seconds.
BACKGROUND_SLICE = 0.003
# The parameter added last. A .toe saved before it has the older set: the
# parameters are rebuilt (keeping their values) when the project opens.
NEWEST_PAR = 'Loadrange1'

HERE = Path(project.folder)
TOOLS = HERE.parent / 'tools'
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import aero.pipeline as _aero  # noqa: E402  (needs the sys.path line above)

_pipeline = None
_waiting = True            # the skin is still waiting for the panel method
_lines_waiting = True      # the flow lines are still waiting for their first trace
_lines_drawn = -1          # pipeline.lines_done when the POP last cooked
_status = ''


def pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = _aero.AeroPipeline()
    return _pipeline


def reload():
    '''Re-import tools/aero and drop loaded models (after retraining or editing the code).'''
    global _aero, _pipeline, _waiting, _lines_waiting
    if _pipeline is not None:
        _pipeline.reload()
    for name in [n for n in sys.modules if n == 'aero' or n.startswith('aero.')]:
        del sys.modules[name]
    _aero = importlib.import_module('aero.pipeline')
    _pipeline, _waiting, _lines_waiting = None, True, True
    print('aero_td: reloaded tools/aero')


def _pose(geo):
    '''The tracker's pose ([tx, ty, tz, rx, ry, rz], Z up, mm) from geo1.

    Two ways a network can apply the OSC pose to geo1:
      'tracker'  as sent: Z up, mm, rotate order xyz (load_setup.py's layout)
      'y_up'     converted to TouchDesigner's Y up by the OSC callback, as in
                 ProjectMappingSetUp.toe: t -> (tx, tz, -ty) * scale,
                 r -> (rx, rz, -ry) with rotate order xzy. Undone here
                 (aero.flow.pose_from_y_up).
    POSE_AXES 'auto' picks 'y_up' when geo1's rotate order is xzy.'''
    tx, ty, tz, rx, ry, rz = (float(getattr(geo.par, name).eval()) for name in TRANSFORM_PARS)
    axes = POSE_AXES
    if axes == 'auto':
        axes = 'y_up' if geo.par.rord.eval() == 'xzy' else 'tracker'
    if axes == 'y_up':
        from aero.flow import pose_from_y_up
        return pose_from_y_up(tx, ty, tz, rx, ry, rz)
    return [tx, ty, tz, rx, ry, rz]


def _set_defaults(pairs):
    # A new parameter starts at zero, not at its default, so each is set to it too.
    for par, value in pairs:
        par.default = par.val = value


# ---- Script POP callbacks (aero_pop.py forwards to these) ----
def setup_parameters(scriptOp):
    from aero.flow import DEFAULT_SPEED
    page = scriptOp.appendCustomPage('Aero')
    p = page.appendMenu('Solver', label='Solver')[0]
    p.menuNames = ['panel', 'ml']
    p.menuLabels = ['Panel method', 'ML surrogate']
    _set_defaults([(p, 'panel')])
    p = page.appendFloat('Speed', label='Wind speed (m/s)')[0]
    p.min, p.clampMin, p.normMin, p.normMax = 0.0, True, 0.0, 30.0
    _set_defaults([(p, DEFAULT_SPEED)])
    p = page.appendMenu('Display', label='Display')[0]
    p.menuNames = ['cp', 'pressure', 'loading']
    p.menuLabels = ['Pressure coefficient (Cp)', 'Pressure (Pa)', 'Wing loading (Cp difference across wings)']
    _set_defaults([(p, 'cp')])
    _set_defaults(zip(page.appendFloat('Cprange', label='Cp range (blue, red)', size=2), (-2.0, 1.0)))
    _set_defaults(zip(page.appendFloat('Parange', label='Pressure range Pa (blue, red)', size=2), (-150.0, 75.0)))
    _set_defaults(zip(page.appendFloat('Loadrange', label='Wing loading range (blue, red)', size=2), (-1.5, 1.5)))
    p = page.appendFloat('Heatmapgain', label='Heatmap brightness')[0]
    p.min, p.clampMin, p.normMin, p.normMax = 0.0, True, 0.0, 1.0
    _set_defaults([(p, 0.6)])
    p = page.appendFloat('Skinoffset', label='Skin offset (mm, 0 = auto)')[0]
    p.min, p.clampMin, p.normMax = 0.0, True, 5.0
    _set_defaults([(p, 0.0)])
    page.appendPulse('Reload', label='Reload models and code')

    page = scriptOp.appendCustomPage('Flow lines')
    _set_defaults([(page.appendToggle('Surfacelines', label='Surface lines')[0], True)])
    p = page.appendInt('Surfacecount', label='Surface line seeds')[0]
    p.min, p.clampMin, p.normMax = 0, True, 200
    _set_defaults([(p, 60)])
    _set_defaults([(page.appendToggle('Groundlines', label='Ground lines')[0], True)])
    p = page.appendInt('Groundcount', label='Ground lines')[0]
    p.min, p.clampMin, p.normMax = 0, True, 80
    _set_defaults([(p, 28)])
    p = page.appendFloat('Linewidth', label='Line width (mm, 0 = auto)')[0]
    p.min, p.clampMin, p.normMax = 0.0, True, 10.0
    _set_defaults([(p, 0.0)])
    _set_defaults([(page.appendToggle('Animate', label='Animate')[0], True)])
    p = page.appendFloat('Flowrate', label='Flow rate (model lengths / s)')[0]
    p.min, p.clampMin, p.normMax = 0.0, True, 2.0
    _set_defaults([(p, 0.5)])
    p = page.appendFloat('Dashlength', label='Dash length (x model size)')[0]
    p.min, p.clampMin, p.normMax = 0.01, True, 0.5
    _set_defaults([(p, 0.15)])
    _set_defaults(zip(page.appendFloat('Speedrange', label='Speed colours (x wind, dim, bright)', size=2), (0.0, 2.0)))


def _upgrade_parameters(pop):
    '''What Setup Parameters does, for a .toe saved with an older set of
    parameters, keeping every value, expression and binding already set
    (a menu value that no longer exists goes back to its default).'''
    kept = {p.name: (p.mode, p.expr, p.bindExpr, p.val) for p in pop.customPars}
    pop.destroyCustomPars()
    setup_parameters(pop)
    for p in pop.customPars:
        if p.name not in kept or p.isPulse:
            continue
        mode, expr, bind, val = kept[p.name]
        if mode == ParMode.EXPRESSION:
            p.expr, p.mode = expr, mode
        elif mode == ParMode.BIND:
            p.bindExpr, p.mode = bind, mode
        elif not p.isMenu or val in p.menuNames:
            p.val = val


def on_pulse(par):
    if par.name == 'Reload':
        reload()
        par.owner.cook(force=True)


def cook(scriptOp):
    global _waiting, _lines_waiting, _lines_drawn, _status
    par = scriptOp.par
    if not hasattr(par, NEWEST_PAR):      # rebuilt from on_frame_start next frame
        scriptOp.addWarning('parameters missing or out of date: click Setup Parameters')
        return
    aero = pipeline()
    pose = _pose(op(GEO))
    display = par.Display.eval()
    low, high = {'pressure': (par.Parange1, par.Parange2),
                 'loading': (par.Loadrange1, par.Loadrange2)}.get(display, (par.Cprange1, par.Cprange2))
    low, high = low.eval(), high.eval()
    result = aero.evaluate(pose, par.Solver.eval(), float(par.Speed.eval()))
    skin = aero.skin(result, display, float(low), float(high),
                     gain=float(par.Heatmapgain.eval()), offset=float(par.Skinoffset.eval()))

    surface = int(par.Surfacecount.eval()) if par.Surfacelines.eval() else 0
    ground = int(par.Groundcount.eval()) if par.Groundlines.eval() else 0
    _lines_drawn = aero.lines_done
    ribbons, line_notes = aero.flow_lines(pose, surface, ground, float(par.Linewidth.eval()))
    size = _aero.flowlines.model_size(aero.model().panels)
    animate = par.Animate.eval()
    # Python's clock, not absTime: reading absTime makes a time dependency, and
    # on_frame_start already cooks this once a frame while animating.
    lines = _aero.line_mesh(ribbons, time.perf_counter() if animate else 0.0,
                            rate=float(par.Flowrate.eval()) * size if animate else 0.0,
                            dash=float(par.Dashlength.eval()) * size,
                            low=float(par.Speedrange1.eval()), high=float(par.Speedrange2.eval()))
    mesh = _aero.Mesh.merge(skin, lines)        # skin first: the lines blend over it

    staging = scriptOp.createGeometryStagingData()
    staging.createPointAttrib(name='P', numComps=3)
    staging.createPointAttrib(name='Color', numComps=4, qualifier=AttribQualifier.COLOR)
    staging.copyNumpyArrayToPointAttrib('P', np.ascontiguousarray(mesh.points))
    staging.copyNumpyArrayToPointAttrib('Color', np.ascontiguousarray(mesh.colors))
    staging.copyNumpyArrayToPrimTriangles(np.ascontiguousarray(mesh.triangles))
    scriptOp.buildGeometry(staging)

    _waiting = not result.ready
    _lines_waiting = ribbons is None
    notes = result.notes + line_notes
    _status = ' | '.join([result.status] + line_notes)
    if notes:
        scriptOp.addWarning('; '.join(notes))


# ---- the mesh, when there is no load_setup.py ----
_mesh_check = 0
_mesh_stamp = None         # the placed OBJ's mtime when the File In SOP last loaded it


def _sync_mesh():
    '''Keep the model's File In SOP on the active model as it sits on the
    plate (models/<name>/aero/<name>_placed.obj: File In can't read STL, and
    with no mount SOP in the network the OBJ carries the mount offset and
    scale itself), following setup/model.json when the model is swapped and
    reloading when its model.json changes. Only when there is no load_setup.py
    (which loads the STL's own OBJ into its mount SOP), and only when the File
    In SOP already points at a file in models/ or at a missing file: a mesh
    chosen by hand is left alone. Returns True when it changed the File In SOP.'''
    global _mesh_stamp
    geo = op(GEO)
    file_in = _model_file_in(geo) if geo is not None else None
    if op(LOAD_SETUP) is not None or file_in is None:
        return False
    current = file_in.par.file.eval()
    path = Path(current) if os.path.isabs(current) else Path(project.folder, current)
    models = (HERE.parent / 'models').resolve()
    if path.exists() and models not in path.resolve().parents:
        return False
    from aero.catalog import model_info
    obj = model_info(pipeline().selector.select()).placed_obj()
    wanted = Path(os.path.relpath(obj, project.folder)).as_posix()
    stamp = obj.stat().st_mtime_ns
    if current == wanted and stamp == _mesh_stamp:
        return False
    _mesh_stamp = stamp
    file_in.par.file = wanted
    for name in ('reloadpulse', 'reload'):
        if hasattr(file_in.par, name) and getattr(file_in.par, name).isPulse:
            getattr(file_in.par, name).pulse()
            break
    print(f'aero_td: {file_in.path} <- {wanted}')
    return True


# ---- Execute DAT ----
def on_frame_start():
    '''Keep re-cooking while something is still getting ready (the panel
    method, the flow field, a re-trace) and once more when a re-trace has
    finished, so new results show as soon as they exist, and show the status
    line. Pose and setting changes re-cook by themselves: TouchDesigner sees
    the cook read them. The status is written here, not while cooking:
    changing another operator during a cook makes the cooking operator cook
    again every frame.'''
    global _mesh_check
    _mesh_check += 1
    if _mesh_check % 30 == 1:             # twice a second is plenty to notice a model swap
        _sync_mesh()
    pop = op(AERO_POP)
    if pop is not None and not hasattr(pop.par, NEWEST_PAR):
        _upgrade_parameters(pop)
    if pop is not None:
        tracing = _pipeline is not None and _pipeline.lines_pending
        if _waiting or _lines_waiting or tracing:
            time.sleep(BACKGROUND_SLICE)
        # Animating: cook every frame from here. Relying on a time dependency
        # (reading absTime in the cook) froze the dashes after Animate was
        # switched off and on again (measured: 1 cook in 60 frames).
        animate = hasattr(pop.par, 'Animate') and pop.par.Animate.eval()
        if animate or _waiting or _lines_waiting or tracing or (
                _pipeline is not None and _pipeline.lines_done != _lines_drawn):
            pop.cook(force=True)
    status = op(STATUS_DAT)
    if status is not None and status.text != _status:
        status.text = _status


# ---- one-off network setup ----
def _set_menu(par, *wanted):
    '''Set a menu parameter to the first entry whose name contains one of `wanted`.'''
    names = [n.lower() for n in par.menuNames]
    for w in wanted:
        for i, n in enumerate(names):
            if n == w or w in n:
                par.val = par.menuNames[i]
                return True
    return False


def _callbacks(scriptop, filename):
    dat = scriptop.par.callbacks.eval()
    dat.par.file = filename         # relative to the .toe's folder, so it works on any machine
    dat.par.syncfile = True
    return dat


def _model_file_in(geo):
    '''The File In SOP that loads the model: inside geo1 (load_setup.py's
    layout), or the only one beside geo1 (a network that transforms the mesh
    before geo1, like ProjectMappingSetUp.toe).'''
    inside = geo.findChildren(type=fileinSOP, depth=1)
    if inside:
        return inside[0]
    beside = geo.parent().findChildren(type=fileinSOP, depth=1)
    return beside[0] if len(beside) == 1 else None


def _remove_old(geo, made):
    '''Take out the aero_color Script SOP and geo1/flow from before aero_pop,
    reconnecting the File In SOP to whatever aero_color fed.'''
    for old_sop in (geo.op(OLD_SCRIPT_SOP), geo.parent().op(OLD_SCRIPT_SOP)):
        if old_sop is None:
            continue
        source = old_sop.inputs[0] if old_sop.inputs else None
        fed = list(old_sop.outputConnectors[0].connections)
        callbacks = old_sop.par.callbacks.eval()
        for connector in fed:
            if source is not None:
                connector.connect(source)
        old_sop.destroy()
        if callbacks is not None and callbacks.valid:
            callbacks.destroy()
        made.append(f'removed {OLD_SCRIPT_SOP} (replaced by aero_pop)')
    if op(OLD_FLOW_GEO) is not None:
        op(OLD_FLOW_GEO).destroy()
        made.append(f'removed {OLD_FLOW_GEO} (replaced by aero_pop)')


# Transform parameters a Geometry COMP shares with a Transform SOP.
_COPIED_PARS = ('xord', 'rord', 'tx', 'ty', 'tz', 'rx', 'ry', 'rz', 'sx', 'sy', 'sz', 'px', 'py', 'pz', 'scale')


def install():
    '''Add the aero operators to the network (see the top of this file).'''
    root = op(Path(GEO).parent.as_posix())
    made = []
    geo = op(GEO)
    if geo is None:
        geo = root.create(geometryCOMP, Path(GEO).name)
        for child in list(geo.children):
            child.destroy()         # the default torus of a brand new Geometry COMP
        made.append(geo.path)
    file_in = _model_file_in(geo)
    if file_in is None:
        file_in = geo.create(fileinSOP, 'filein1')
        made.append(file_in.path)
    if file_in.parent() is geo and geo.op('mount') is None:
        mount = geo.create(transformSOP, 'mount')
        mount.inputConnectors[0].connect(file_in)
        mount.display = mount.render = True
        made.append(mount.path)
    _remove_old(geo, made)
    if _sync_mesh():
        made.append(f'{file_in.path}.file -> {file_in.par.file.eval()}')

    # The model itself draws black and writes no depth: the skin carries the
    # colours and hides what is behind it, and a model in the depth buffer
    # just under the skin would show through it in specks.
    model_mat = op(geo.par.material.eval()) if geo.par.material.eval() else None
    if model_mat is None:
        model_mat = geo.create(constantMAT, 'model_mat')
        geo.par.material = model_mat.path
        made.append(model_mat.path)
    if model_mat.type == 'constant':
        model_mat.par.colorr = model_mat.par.colorg = model_mat.par.colorb = 0.0
        model_mat.par.applypointcolor = False
        model_mat.par.depthwriting = False
        made.append(f'{model_mat.path} set to black, no depth')

    aero_geo = op(AERO_GEO)
    if aero_geo is None:
        aero_geo = geo.create(geometryCOMP, Path(AERO_GEO).name)
        for child in list(aero_geo.children):
            child.destroy()
        made.append(aero_geo.path)
        after = [c.owner for c in file_in.outputConnectors[0].connections]
        model_transform = next((o for o in after if o.type == 'transform' and o.name != 'mount'), None)
        if model_transform is not None:
            # The skin and lines are in the plate frame (the STL's, mm):
            # give them the same axes and units as the mesh.
            for name in _COPIED_PARS:
                if hasattr(aero_geo.par, name) and hasattr(model_transform.par, name):
                    getattr(aero_geo.par, name).expr = f"op('{model_transform.path}').par.{name}"
        else:
            setup = op(LOAD_SETUP)
            units = getattr(setup.module, 'TD_UNITS_PER_MM', 1.0) if setup is not None else 1.0
            aero_geo.par.scale = units          # mm, like the OSC pose
        aero_geo.par.drawpriority = -1          # drawn after the (black) model
    pop = op(AERO_POP)
    if pop is None:
        pop = aero_geo.create(scriptPOP, Path(AERO_POP).name)
        _callbacks(pop, 'aero_pop.py')
        made.append(pop.path)
    pop.display = pop.render = True
    if not hasattr(pop.par, 'Solver'):
        pop.par.setuppars.pulse()
    if not aero_geo.par.material.eval():
        mat = aero_geo.create(constantMAT, 'aero_mat')
        mat.par.applypointcolor = True
        mat.par.blending = True                 # colours come premultiplied by alpha
        _set_menu(mat.par.srcblend, 'one')
        _set_menu(mat.par.destblend, 'omsa')
        _set_menu(mat.par.cullface, 'neither', 'none')
        aero_geo.par.material = mat.path
        made.append(mat.path)

    for name, kind in ((Path(FRAMES_DAT).name, executeDAT), (Path(STATUS_DAT).name, textDAT)):
        if root.op(name) is None:
            dat = root.create(kind, name)
            made.append(dat.path)
            if kind is executeDAT:
                dat.text = ("def onFrameStart(frame):\n"
                            "    op('aero_td').module.on_frame_start()\n"
                            "    return\n")
                dat.par.framestart = True

    setup = op(LOAD_SETUP)
    if setup is not None:
        setup.module.load()
    print('aero_td: installed' + (': ' + ', '.join(made) if made else ', nothing was missing'))
    return made
