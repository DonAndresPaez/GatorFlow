'''make_markers.py: the ArUco markers to print and stick on the base plate.

Run: python -m calibration.make_markers              (size from setup/plate.yml)
     python -m calibration.make_markers --size 50    (any size, e.g. to try one; plate.yml is not changed)
Makes: tools/output/markers_sheet_<size>mm.pdf   all markers on one Letter page, plus a 100 mm scale bar
       tools/output/marker_<id>_<size>mm.png    one image per marker, if you'd rather print them singly

To change the marker size: set markerSizeMm in setup/plate.yml to the size you
WANT, run this, print, measure, then put the MEASURED size back in plate.yml.

Print the PDF at 100% / "Actual size" (turn off "fit to page"). Then measure
the 100 mm scale bar: if it isn't 100 mm, the printer scaled the page, and the
markers are off by the same factor.

Each marker sits inside a dashed cut line with a white border (the "quiet
zone" the detector needs). Cut along the dashed line, not closer. The grey
label inside the border marks the marker's BOTTOM edge: stick it so the label
is nearest the plate's near edge (-Y), as setup/plate.yml describes.

Measure the black squares after printing: printers lie about scale, and that
number is the ruler for every distance the tracker reports. If they are not
the size you asked for, put the measured size in markerSizeMm.
'''

import argparse

import cv2
import numpy as np
from PIL import Image

from common import paths
from common.plate import LAYOUT

PIXELS = 800             # resolution of the black square in the single PNGs

SHEET_DPI = 600
SHEET_MM = (215.9, 279.4)   # US Letter
MARGIN_MM = 15.0
QUIET_FRACTION = 0.3     # white border on each side, as a fraction of the marker size
GAP_MM = 8.0             # between cut lines


def marker_image(marker_id, side_px):
    """The marker at exactly side_px pixels, with crisp bit edges."""
    bits = LAYOUT.dictionary.markerSize + 2          # data bits + the black border
    art = cv2.aruco.generateImageMarker(LAYOUT.dictionary, marker_id, bits * 100)
    return cv2.resize(art, (side_px, side_px), interpolation=cv2.INTER_NEAREST)


def label_for(marker_id, size_mm):
    return f"id {marker_id} - bottom edge - {size_mm:g} mm"


# ---------------------------------------------------------------- single PNGs

def make_one(marker_id, size_mm):
    marker = marker_image(marker_id, PIXELS)
    border = PIXELS // 5
    page = np.full((PIXELS + 2 * border, PIXELS + 2 * border), 255, np.uint8)
    page[border:border + PIXELS, border:border + PIXELS] = marker
    label = label_for(marker_id, size_mm)
    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)
    cv2.putText(page, label, ((page.shape[1] - w) // 2, border + PIXELS + border // 2 + h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, 150, 2, cv2.LINE_AA)
    dpi = PIXELS / (size_mm / 25.4)
    name = paths.output_file(f"marker_{marker_id}_{size_mm:g}mm.png")
    Image.fromarray(page).save(name, dpi=(dpi, dpi))
    return name


# ---------------------------------------------------------------- the sheet

def mm(value):
    return int(round(value * SHEET_DPI / 25.4))


def text(page, string, x, y, height_mm, grey=0, centre=False, thickness=None):
    scale = mm(height_mm) / 22.0                     # Hershey simplex is ~22 px tall at scale 1
    thickness = thickness or max(2, int(scale * 1.6))
    (w, h), _ = cv2.getTextSize(string, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    if centre:
        x -= w // 2
    cv2.putText(page, string, (x, y + h), cv2.FONT_HERSHEY_SIMPLEX, scale, grey, thickness, cv2.LINE_AA)
    return h


def dashed_square(page, x0, y0, x1, y1, grey=170):
    dash, width = mm(2.0), max(2, mm(0.15))
    for a, b, fixed, horizontal in ((x0, x1, y0, True), (x0, x1, y1, True), (y0, y1, x0, False), (y0, y1, x1, False)):
        for start in range(a, b, 2 * dash):
            end = min(start + dash, b)
            if horizontal:
                cv2.line(page, (start, fixed), (end, fixed), grey, width)
            else:
                cv2.line(page, (fixed, start), (fixed, end), grey, width)


def scale_bar(page, x, y):
    length, tick = mm(100), mm(3)
    width = max(3, mm(0.3))
    cv2.line(page, (x, y), (x + length, y), 0, width)
    for k in range(11):
        tx = x + mm(10 * k)
        cv2.line(page, (tx, y), (tx, y - (tick if k % 5 == 0 else tick // 2)), 0, width)
    text(page, "0", x, y + mm(1.5), 2.5, centre=True)
    text(page, "100 mm", x + length, y + mm(1.5), 2.5, centre=True)
    text(page, "<- measure this: it must be exactly 100 mm", x + length + mm(12), y - mm(2), 3.0)


def make_sheet(size_mm):
    width, height = mm(SHEET_MM[0]), mm(SHEET_MM[1])
    page = np.full((height, width), 255, np.uint8)

    y = mm(MARGIN_MM)
    y += text(page, f"GatorFlow marker plate - {LAYOUT.dictionary_name}, {size_mm:g} mm markers", mm(MARGIN_MM), y, 4.5)
    y += mm(3)
    for line in ('Print at 100% / "Actual size" (not "fit to page"). Cut on the dashed lines.',
                 "Measure a black square and put the MEASURED size in setup/plate.yml."):
        y += text(page, line, mm(MARGIN_MM), y, 2.8, grey=60) + mm(1.5)
    y += mm(12)
    scale_bar(page, mm(MARGIN_MM), y)
    y += mm(14)

    side = mm(size_mm)
    quiet = mm(size_mm * QUIET_FRACTION)
    cell = side + 2 * quiet
    gap = mm(GAP_MM)
    columns = max(1, (width - 2 * mm(MARGIN_MM) + gap) // (cell + gap))
    x_start = (width - (columns * cell + (columns - 1) * gap)) // 2

    placed = 0
    for index, marker_id in enumerate(LAYOUT.ids):
        row, column = divmod(index, columns)
        x0 = x_start + column * (cell + gap)
        y0 = y + row * (cell + gap)
        if y0 + cell > height - mm(MARGIN_MM):
            print(f"WARNING: no room on the page for id {marker_id} and later; use the single PNGs for those.")
            break
        page[y0 + quiet:y0 + quiet + side, x0 + quiet:x0 + quiet + side] = marker_image(marker_id, side)
        dashed_square(page, x0, y0, x0 + cell, y0 + cell)
        # label in the bottom quiet zone, small and grey so the detector ignores it
        label_height = min(3.0, size_mm * QUIET_FRACTION * 0.3)
        text(page, label_for(marker_id, size_mm), x0 + cell // 2, y0 + quiet + side + quiet // 2 - mm(label_height) // 2,
             label_height, grey=150, centre=True, thickness=max(2, mm(0.2)))
        placed += 1

    name = paths.output_file(f"markers_sheet_{size_mm:g}mm.pdf")
    Image.fromarray(page).save(name, resolution=SHEET_DPI)
    return name, placed


def main():
    ap = argparse.ArgumentParser(description="Printable ArUco markers for the plate")
    ap.add_argument("--size", type=float, default=LAYOUT.size_mm,
                    help=f"black square in mm (default: markerSizeMm in setup/plate.yml, {LAYOUT.size_mm:g})")
    size_mm = ap.parse_args().size

    sheet, placed = make_sheet(size_mm)
    print(f"Saved {sheet}  ({placed} markers on one Letter page)")
    for marker_id in LAYOUT.ids:
        make_one(marker_id, size_mm)
    print(f"Saved marker_<id>_{size_mm:g}mm.png for ids {LAYOUT.ids} in {paths.OUTPUT_DIR}")
    print(f"Dictionary {LAYOUT.dictionary_name}. Print at 100% scale (turn off 'fit to page').")
    print(f"Each black square should measure {size_mm:g} mm, and the scale bar 100 mm. Measure after printing.")
    if size_mm != LAYOUT.size_mm:
        print(f"setup/plate.yml still says {LAYOUT.size_mm:g} mm: if you use these, put the MEASURED size there.")

if __name__ == "__main__":
    main()
