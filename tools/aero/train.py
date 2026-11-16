'''train.py: pretrain the network for pipeline 2 (needs PyTorch, not TouchDesigner).

Run (from tools/):
    python -m aero.train                         the active model, taught by the panel method
    python -m aero.train --model PMB-1A          a specific model
    python -m aero.train --speed 2 30            the wind speeds TouchDesigner may ask for (m/s)
    python -m aero.train --data cfd.npz          taught by your own results instead (see below)
    python -m aero.train --export-panels p.csv   where the panels are, for whoever runs the CFD

Makes: models/<name>/aero/surrogate.npz  (commit it: TouchDesigner's machine
       then needs no PyTorch, and nobody waits for training)
Needs: pip install -r aero/requirements-train.txt  (PyTorch; uses the NVIDIA GPU if CUDA is there)

--data: an .npz with yaw_deg (S,), speed (S,) and cp (S, N): Cp at each
panel centre (the order --export-panels writes) for S runs. Yaw is the wind's
angle on the plate from +X toward +Y, as in aero/flow.py.

Retrain after changing the model's STL, mountOffsetMm, mountRotationDeg or
scale, or after an update that raises panel.SOLVER_VERSION: the pipeline
refuses a network trained on different geometry or an older solver, and
falls back to the panel method. README.md, "Using the trained network",
walks through the whole loop.
'''

import argparse
import time

import numpy as np

from aero import catalog, panel
from aero.flow import wind_vector
from aero.surrogate import Surrogate, features, surrogate_file

HIDDEN = (128, 128, 128)
MAX_MODES = 64
ENERGY_KEPT = 0.99999           # fraction of the pressure field's variance the modes keep


def panel_dataset(solution, samples, speed_range, seed=0):
    """Random winds round the plate, Cp from the panel method."""
    rng = np.random.default_rng(seed)
    yaw = rng.uniform(0.0, 360.0, samples)
    speed = rng.uniform(*speed_range, samples)
    cp = solution.cp_batch(np.stack([wind_vector(y) for y in yaw]))
    return yaw, speed, cp


def principal_modes(cp, energy=ENERGY_KEPT, max_modes=MAX_MODES):
    """Mean field and the fewest modes that keep `energy` of the variance."""
    mean = cp.mean(axis=0)
    _, s, vt = np.linalg.svd(cp - mean, full_matrices=False)
    kept = np.cumsum(s ** 2) / max(np.sum(s ** 2), 1e-300)
    k = int(np.clip(np.searchsorted(kept, energy) + 1, 4, min(max_modes, len(s))))
    return mean, vt[:k]


def fit(yaw, speed, cp, speed_range, epochs=4000, seed=0, log=print):
    """Train the MLP on (yaw, speed) -> Cp, return a Surrogate (key and source unset)."""
    import torch

    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mean, basis = principal_modes(cp)
    coeffs = (cp - mean) @ basis.T
    scale = coeffs.std(axis=0) + 1e-12

    x = torch.tensor(features(yaw, speed, speed_range), dtype=torch.float32, device=device)
    y = torch.tensor(coeffs / scale, dtype=torch.float32, device=device)
    # The loss is weighted by each mode's size, so it is the error in Cp
    # itself (the modes are orthonormal), not in the scaled weights.
    weight = torch.tensor(scale ** 2 / np.sum(scale ** 2), dtype=torch.float32, device=device)

    sizes = (x.shape[1], *HIDDEN, y.shape[1])
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(torch.nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(torch.nn.Tanh())
    net = torch.nn.Sequential(*layers).to(device)
    optimiser = torch.optim.Adam(net.parameters(), lr=3e-3)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, epochs)

    log(f"training on {device}: {len(yaw)} samples, {basis.shape[0]} modes, layers {sizes}")
    started = time.time()
    for epoch in range(epochs):
        optimiser.zero_grad()
        loss = (((net(x) - y) ** 2) * weight).sum(dim=1).mean()
        loss.backward()
        optimiser.step()
        schedule.step()
        if epoch % 500 == 0 or epoch == epochs - 1:
            log(f"  epoch {epoch:5d}  loss {loss.item():.3e}  ({time.time() - started:.0f} s)")

    linear = [m for m in net if isinstance(m, torch.nn.Linear)]
    return Surrogate([m.weight.detach().cpu().numpy().T for m in linear],
                     [m.bias.detach().cpu().numpy() for m in linear],
                     mean, basis, scale, speed_range)


def report(surrogate, yaw, speed, cp, label):
    error = surrogate.predict_batch(yaw, speed) - cp
    print(f"{label}: Cp error RMS {np.sqrt(np.mean(error ** 2)):.4f}, worst {np.abs(error).max():.4f} "
          f"(Cp spans {cp.min():.2f} to {cp.max():.2f})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="name in models/ (default: setup/model.json's active)")
    ap.add_argument("--data", help=".npz with yaw_deg, speed, cp from another solver")
    ap.add_argument("--samples", type=int, default=4000, help="panel-method samples to train on")
    ap.add_argument("--speed", type=float, nargs=2, default=(1.0, 30.0), metavar=("MIN", "MAX"))
    ap.add_argument("--epochs", type=int, default=4000)
    ap.add_argument("--export-panels", metavar="CSV", help="write panel centres + normals and stop")
    args = ap.parse_args()

    name = args.model or catalog.ManualSelector().select()
    info = catalog.model_info(name)
    panels = info.panels()
    print(f"{info.name}: {len(panels)} wetted panels")

    if args.export_panels:
        table = np.column_stack([np.arange(len(panels)), panels.centroids, panels.normals, panels.areas])
        np.savetxt(args.export_panels, table, delimiter=",", fmt=["%d"] + ["%.6f"] * 7,
                   header="panel,x_mm,y_mm,z_mm,nx,ny,nz,area_mm2", comments="")
        print(f"wrote {args.export_panels} (plate frame: X right, Y far edge, Z up, plate top at Z=0)")
        return

    if args.data:
        with np.load(args.data) as d:
            yaw, speed, cp = d["yaw_deg"], d["speed"], d["cp"]
        if cp.shape[1] != len(panels):
            raise SystemExit(f"{args.data} has Cp at {cp.shape[1]} points, the model has {len(panels)} panels "
                             "(use --export-panels for the expected order)")
        speed_range = (float(speed.min()), float(speed.max()))
        source = f"data:{args.data}"
    else:
        print("panel method: loading or building the influence matrices...")
        solution = panel.load_or_build(info, panels)
        yaw, speed, cp = panel_dataset(solution, args.samples, args.speed)
        speed_range = tuple(args.speed)
        source = f"panel v{panel.SOLVER_VERSION}"

    held = np.zeros(len(yaw), bool)
    held[np.random.default_rng(1).permutation(len(yaw))[: max(1, len(yaw) // 10)]] = True
    surrogate = fit(yaw[~held], speed[~held], cp[~held], speed_range, epochs=args.epochs)
    surrogate.key, surrogate.source = panels.key, source
    report(surrogate, yaw[~held], speed[~held], cp[~held], "training set")
    report(surrogate, yaw[held], speed[held], cp[held], "held-out set")

    path = surrogate_file(info)
    surrogate.save(path)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
