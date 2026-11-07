'''precompute.py: build the panel method's cache ahead of time.

Run (from tools/):
    python -m aero.precompute               the active model
    python -m aero.precompute --model NAME
    python -m aero.precompute --all         every model in models/

Makes: models/<name>/aero/panel_<geometry>_v<version>.npz (not in git)

TouchDesigner builds it by itself on first use (in the background, model shown
grey meanwhile), so this is only to have it ready before a session, or to see
how long a new model takes.
'''

import argparse
import time

from aero import catalog, panel
from aero.flow import wind_vector


def precompute(name, force=False):
    info = catalog.model_info(name)
    panels = info.panels()
    path = panel.cache_file(info, panels)
    if path.exists() and not force:
        print(f"{name}: up to date ({path.name})")
        return
    print(f"{name}: {len(panels)} panels, building...")
    started = time.time()
    last = [0]

    def progress(fraction):
        if fraction * 10 >= last[0] + 1:
            last[0] = int(fraction * 10)
            print(f"  {fraction * 100:3.0f}%  ({time.time() - started:.0f} s)")

    solution = panel.build(panels, progress=progress, lift=info.kutta)
    solution.save(path)
    cp = solution.cp(wind_vector(0.0))
    print(f"{name}: done in {time.time() - started:.1f} s -> {path}")
    print(f"  wind along +X: Cp from {cp.min():.2f} to {cp.max():.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--force", action="store_true", help="rebuild even if the cache is there")
    args = ap.parse_args()
    names = catalog.available_models() if args.all else [args.model or catalog.ManualSelector().select()]
    for name in names:
        precompute(name, args.force)


if __name__ == "__main__":
    main()
