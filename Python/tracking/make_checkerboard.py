"""Make the checkerboard to print for camera calibration.

    python -m tracking.make_checkerboard

Saves checkerboard.png. Print at 100% scale on Letter or A4, then MEASURE one
square and put the real number in SQUARE_MM in tracking/calibrate.py. The
squares are the ruler the calibration uses, so a 5% printing error is a 5%
error in every distance the tracker reports.

Tape it to something rigid (cardboard, clipboard, a book cover). A sagging
sheet of paper gives a curved board, and the calibration fits the curve
instead of the lens.
"""

import numpy as np
from PIL import Image

from tracking.calibrate import COLUMNS, ROWS, SQUARE_MM

PIXELS_PER_SQUARE = 100
OUTPUT = "checkerboard.png"


def main():
    # A board with 9x6 inner corners is 10x7 squares.
    squares_x, squares_y = COLUMNS + 1, ROWS + 1
    board = np.indices((squares_y, squares_x)).sum(0) % 2  # 0/1 checker pattern
    board = np.kron(board, np.ones((PIXELS_PER_SQUARE, PIXELS_PER_SQUARE), np.uint8)) * 255

    margin = PIXELS_PER_SQUARE // 2  # white edge so the outer squares are detectable
    page = np.full((board.shape[0] + 2 * margin, board.shape[1] + 2 * margin), 255, np.uint8)
    page[margin:margin + board.shape[0], margin:margin + board.shape[1]] = board

    dpi = PIXELS_PER_SQUARE / (SQUARE_MM / 25.4)
    Image.fromarray(page).save(OUTPUT, dpi=(dpi, dpi))

    print(f"Saved {OUTPUT}: {squares_x} x {squares_y} squares, {COLUMNS} x {ROWS} inner corners.")
    print(f"Printed size at 100% scale: {squares_x * SQUARE_MM:.0f} x {squares_y * SQUARE_MM:.0f} mm "
          f"plus a {SQUARE_MM / 2:.0f} mm white border.")
    print(f"Measure one square after printing. It should be {SQUARE_MM:.0f} mm.")


if __name__ == "__main__":
    main()
