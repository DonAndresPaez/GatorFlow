'''surrogate.py: pipeline 2, the pretrained network, run with NumPy only.

Imported by: aero/pipeline.py, aero/train.py, tests
Run directly: no (train it with python -m aero.train)

A small network maps the flow conditions to the pressure on every panel:

    [cos yaw, sin yaw, speed]  ->  MLP  ->  K mode weights  ->  Cp on N panels

The K modes are the main patterns of the pressure field across the training
set (PCA), so the network predicts a few dozen numbers instead of one per panel.
One evaluation is a few small matrix products: well under a millisecond, so it
runs every frame inside TouchDesigner without PyTorch.

Speed is an input, so it can change live. With the panel method as teacher
the network learns that Cp does not depend on speed (in potential flow every
velocity scales with the wind). Trained on CFD results (train.py --data) it
learns how the pattern changes with speed. Pressure in Pa scales with speed
squared either way.

The file is models/<name>/aero/surrogate.npz. It remembers the geometry it was
trained on, and the pipeline refuses it if the model's STL has changed since.
'''

from pathlib import Path

import numpy as np

FILE_NAME = "surrogate.npz"


def features(yaw_deg, speed, speed_range):
    """Network inputs. Yaw as cos/sin so 359 and 1 degree are neighbours;
    speed scaled to [-1, 1] over the trained range."""
    yaw = np.radians(np.atleast_1d(np.asarray(yaw_deg, np.float64)))
    speed = np.broadcast_to(np.asarray(speed, np.float64), yaw.shape)
    low, high = speed_range
    middle, half = 0.5 * (low + high), max(0.5 * (high - low), 1e-9)
    return np.stack([np.cos(yaw), np.sin(yaw), (speed - middle) / half], axis=1)


class Surrogate:
    def __init__(self, weights, biases, mean, basis, scale, speed_range, key="", source=""):
        self.weights = [np.asarray(w, np.float64) for w in weights]   # each (in, out)
        self.biases = [np.asarray(b, np.float64) for b in biases]
        self.mean = np.asarray(mean, np.float64)                      # (N,)
        self.basis = np.asarray(basis, np.float64)                    # (K, N)
        self.scale = np.asarray(scale, np.float64)                    # (K,)
        self.speed_range = (float(speed_range[0]), float(speed_range[1]))
        self.key = key
        self.source = source

    def __len__(self):
        return len(self.mean)

    def predict_batch(self, yaw_deg, speed):
        """(S, N) Cp for S conditions."""
        h = features(yaw_deg, speed, self.speed_range)
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            h = h @ w + b
            if i < len(self.weights) - 1:
                h = np.tanh(h)
        return self.mean + (h * self.scale) @ self.basis

    def predict(self, yaw_deg, speed):
        """(N,) Cp for one condition."""
        return self.predict_batch(yaw_deg, speed)[0]

    def in_range(self, speed):
        low, high = self.speed_range
        return low <= speed <= high

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        layers = {}
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            layers[f"w{i}"] = w.astype(np.float32)
            layers[f"b{i}"] = b.astype(np.float32)
        np.savez_compressed(path, layers=len(self.weights), mean=self.mean.astype(np.float32),
                            basis=self.basis.astype(np.float32), scale=self.scale.astype(np.float32),
                            speed_range=np.array(self.speed_range), key=self.key, source=self.source,
                            **layers)

    @classmethod
    def load(cls, path):
        with np.load(path) as d:
            count = int(d["layers"])
            return cls([d[f"w{i}"] for i in range(count)], [d[f"b{i}"] for i in range(count)],
                       d["mean"], d["basis"], d["scale"], d["speed_range"],
                       str(d["key"]), str(d["source"]))


def surrogate_file(info):
    return info.aero_dir / FILE_NAME
