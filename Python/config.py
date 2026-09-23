"""Settings shared by every part of GatorFlow.

If you change something here, update docs/osc_interface.md too.
"""

# ---- Tracking -> TouchDesigner (OSC over UDP) ----
OSC_HOST = "127.0.0.1"
TD_PORT = 9000           # TouchDesigner will listen here (moves geo1)
SIM_PORT = 9001          # the simulation will listen here; for now tracking.monitor uses it
OSC_ADDRESS = "/car/transform"
# Message: [tx, ty, tz, rx, ry, rz]  translate in mm, rotate in degrees,
# TouchDesigner world (Y up), rotation order X then Y then Z.
EULER_ORDER = "xyz"      # scipy name for that order
