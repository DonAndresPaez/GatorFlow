'''make_markers.py: the ArUco markers to print and stick on the base plate.

Run: python -m calibration.make_markers
Makes: tools/output/marker_0.png ... (one per marker in setup/plate.yml)

To change the marker size: set markerSizeMm in setup/plate.yml to the size you
WANT, run this, print, measure, then put the MEASURED size back in plate.yml.

Each prints at 100% scale to exactly the layout's markerSizeMm, with the white
border ("quiet zone") the detector needs. The label under each marker marks
its BOTTOM edge: stick it so the label is nearest the plate's near edge (-Y),
as setup/plate.yml describes.

Measure the black squares after printing: printers lie about scale, and that
number is the ruler for every distance the tracker reports. If they are not
the size you asked for, put the measured size in markerSizeMm.
'''

import cv2
import numpy as np
from PIL import Image

from common import paths
from common.plate import LAYOUT, MARKER_SIZE_MM

PIXELS = 800          # resolution of the black square itself


def make_one(marker_id):
    marker = cv2.aruco.generateImageMarker(LAYOUT.dictionary, marker_id, PIXELS)

    # White border around it. Without it the detector struggles.
    border = PIXELS // 5
    page = np.full((PIXELS + 2 * border, PIXELS + 2 * border), 255, np.uint8)
    page[border:border + PIXELS, border:border + PIXELS] = marker

    # Small grey label in the bottom border: which id, and which edge is the bottom.
    label = f"id {marker_id}  -  bottom edge  -  {MARKER_SIZE_MM:.0f} mm"
    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)
    cv2.putText(page, label, ((page.shape[1] - w) // 2, border + PIXELS + border // 2 + h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, 150, 2, cv2.LINE_AA)

    dpi = PIXELS / (MARKER_SIZE_MM / 25.4)
    name = paths.output_file(f"marker_{marker_id}.png")
    Image.fromarray(page).save(name, dpi=(dpi, dpi))
    return name


def main():
    for marker_id in LAYOUT.ids:
        print(f"Saved {make_one(marker_id)}")
    print(f"Dictionary {LAYOUT.dictionary_name}. Print at 100% scale (turn off 'fit to page').")
    print(f"Each black square should measure {MARKER_SIZE_MM:.0f} mm. Measure after printing.")


if __name__ == "__main__":
    main()