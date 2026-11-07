'''aero_pop.py: the callbacks DAT of geo1/aero's Script POP `aero_pop`, which
draws the pressure skin and the flow lines as one mesh.

Everything happens in aero_td.py; this only forwards to it, so the Script POP
never needs editing. op('aero_td').module.install() sets it up.
'''

AERO = '/project1/aero_td'      # the Text DAT holding aero_td.py


def onSetupParameters(scriptOp):
    op(AERO).module.setup_parameters(scriptOp)


def onPulse(par):
    op(AERO).module.on_pulse(par)


def onCook(scriptOp):
    op(AERO).module.cook(scriptOp)


def onGetCookLevel(scriptOp):
    # Cook when something it reads changes (pose, parameters). While
    # animating, aero_td.on_frame_start() also cooks it once a frame.
    return CookLevel.AUTOMATIC
