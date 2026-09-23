"""Make the ArUco marker to print.

    python -m tracking.make_marker

Saves marker.png sized so that printing it at 100% scale (no "fit to page")
gives a black square exactly MARKER_SIZE_MM wide. Glue it to stiff card so it
stays flat: a bent marker makes the pose wobble.
"""

import cv2
import numpy as np
from PIL import Image

from tracking.track import ARUCO_DICT, MARKER_ID, MARKER_SIZE_MM

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
