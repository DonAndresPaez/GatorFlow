'''make_marker.py: the ArUco marker to print and stick on the model.

Run: python -m tracking.make_marker
Makes: marker.png

Prints at 100% scale to exactly MARKER_SIZE_MM (set in marker.py), with a
white border around it that the detector needs. Measure the black square after
printing: printers lie about scale, and that number is the ruler for every
distance the tracker reports.
'''

import cv2
import numpy as np
from PIL import Image

from tracking.marker import ARUCO_DICT, MARKER_ID, MARKER_SIZE_MM

PIXELS = 800          # resolution of the black square itself
OUTPUT = "marker.png"


def main():
    marker = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(ARUCO_DICT), MARKER_ID, PIXELS)

    # White border around it ("quiet zone"). Without it the detector struggles.
    border = PIXELS // 5
    page = np.full((PIXELS + 2 * border, PIXELS + 2 * border), 255, np.uint8)
    page[border:border + PIXELS, border:border + PIXELS] = marker

    dpi = PIXELS / (MARKER_SIZE_MM / 25.4)
    Image.fromarray(page).save(OUTPUT, dpi=(dpi, dpi))
    print(f"Saved {OUTPUT}: dictionary {ARUCO_DICT}, id {MARKER_ID}.")
    print(f"Print at 100% scale. The black square should measure {MARKER_SIZE_MM:.0f} mm.")
    print("Measure it after printing. If it differs, put the real number in MARKER_SIZE_MM.")


if __name__ == "__main__":
    main()
