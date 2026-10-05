'''config.py: the network settings everything has to agree on.

Imported by: checks/monitor.py, checks/fake_tracker.py, common/plate.py
Mirrored in: KinectReader/src/main.cpp (OutputConfiguration namespace)
            and docs/osc_interface.md

Ports, the OSC address, and the rotation order. These describe the software,
not the rig, so they live here and not in setup/. If something here changes,
change it in main.cpp and docs/osc_interface.md too, or the tracker and
TouchDesigner stop agreeing on what a message means.
'''

# ---- Tracking -> TouchDesigner (OSC over UDP) ----
OSC_HOST = "127.0.0.1"
TD_PORT = 9000           # TouchDesigner listens here (moves geo1)
SIM_PORT = 9001          # the simulation will listen here; for now checks.monitor uses it
OSC_ADDRESS = "/car/transform"
# Message: [tx, ty, tz, rx, ry, rz]  translate in mm, rotate in degrees,
# TouchDesigner world, rotation order X then Y then Z.
EULER_ORDER = "xyz"      # scipy name for that order
