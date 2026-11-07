'''colors.py: values -> RGB for the projector.

Imported by: aero/pipeline.py (the heatmap), aero/flowlines.py (the line speed colours)
Run directly: no

Blue = low pressure (suction), red = high (stagnation, where the wind hits),
through cyan, green and yellow: the usual CFD heatmap, readable on a grey
printed model. Values outside the range are held at the end colours.
'''

import numpy as np

# Evenly spaced stops from the low end to the high end.
STOPS = np.array([
    [0.19, 0.07, 0.23],
    [0.16, 0.47, 0.91],
    [0.10, 0.80, 0.80],
    [0.40, 0.95, 0.35],
    [0.95, 0.85, 0.20],
    [0.97, 0.45, 0.10],
    [0.70, 0.05, 0.02],
])

NO_DATA = (0.5, 0.5, 0.5)      # grey while a solver is still getting ready
OFF = (0.06, 0.06, 0.06)       # parts a display leaves out (not wings, in the wing-loading display)


def colormap(values, low, high, stops=STOPS):
    """(n,) values -> (n, 3) RGB in 0..1, along evenly spaced colour stops."""
    t = (np.asarray(values, np.float64) - low) / max(high - low, 1e-12)
    t = np.clip(t, 0.0, 1.0) * (len(stops) - 1)
    i = np.minimum(t.astype(int), len(stops) - 2)
    f = (t - i)[:, None]
    return stops[i] * (1.0 - f) + stops[i + 1] * f
