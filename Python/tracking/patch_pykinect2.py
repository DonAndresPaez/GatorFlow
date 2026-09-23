'''patch_pykinect2.py: make pykinect2 importable on a modern Python.

Run: python -m tracking.patch_pykinect2
Changes: the installed pykinect2 package (keeps .backup copies)

Only needed because Python still reads the Kinect during calibration.
Yes, it was a mess to install, and yes, a mess to make it compatible with modern Python.

Rerun it after reinstalling pykinect2, or in a new virtual environment.
'''

import importlib.util
import shutil
from pathlib import Path

# Lines containing these get commented out.
PATTERNS = ("_check_version(", "assert sizeof(")

# These get swapped in place: old name -> current name.
REPLACEMENTS = {"time.clock()": "time.perf_counter()"}


def find_pykinect2():
    spec = importlib.util.find_spec("pykinect2")
    if spec is None or not spec.submodule_search_locations:
        return None
    return Path(list(spec.submodule_search_locations)[0])


def patch_file(path):
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
    changed = 0
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue
        if any(p in line for p in PATTERNS):
            indent = line[:len(line) - len(stripped)]
            lines[i] = f"{indent}# GatorFlow patched out: {stripped}"
            changed += 1
            continue
        for old, new in REPLACEMENTS.items():
            if old in line:
                lines[i] = line.replace(old, new)
                changed += 1
    if changed:
        backup = path.with_suffix(path.suffix + ".backup")
        if not backup.exists():
            shutil.copy(path, backup)
        path.write_text("".join(lines), encoding="utf-8")
    return changed


def main():
    folder = find_pykinect2()
    if folder is None:
        print("pykinect2 is not installed. Run: pip install pykinect2 comtypes")
        return

    total = 0
    for name in ("PyKinectV2.py", "PyKinectRuntime.py"):
        path = folder / name
        if path.exists():
            count = patch_file(path)
            total += count
            print(f"{name}: {count} line(s) commented out")

    if total:
        print(f"\nPatched {folder}. Backups are next to the originals.")
        print("Now try: python -m tracking.track")
    else:
        print("\nNothing left to patch. If it still fails, the error is something else - send it over.")


if __name__ == "__main__":
    main()
