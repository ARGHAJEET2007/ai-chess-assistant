import cv2
import numpy as np
import json
import argparse


# ============================================================
# CONFIGURATION
# ============================================================

BOARD_SIZE = 8
GRID_SIZE = BOARD_SIZE + 1

CAMERA_ID = 0
WINDOW_NAME = "Chess Board Calibration"
OUTPUT_FILE = "sqdict.json"

# Chess notation
FILES = "abcdefgh"
RANKS = "87654321"

# Selected corner points
points = []


# ============================================================
# COMMAND-LINE ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser(
    description="Calibrate a chess board and generate square coordinates."
)

parser.add_argument(
    "--rotate",
    type=int,
    default=0,
    choices=[0, 90, 180, 270],
    help=(
        "Camera orientation relative to the chess board. "
        "0 = front, 90 = right, 180 = back, 270 = left."
    )
)

args = parser.parse_args()
CAMERA_ROTATION = args.rotate


# ============================================================
# MOUSE CALLBACK
# ============================================================

def handle_mouse_click(event, x, y, flags, param):
    """
    Store the coordinates of the four chessboard corners.

    Corner order:
        1. Top-left
        2. Top-right
        3. Bottom-right
        4. Bottom-left
    """

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    if len(points) < 4:
        points.append((x, y))

        print(
            f"[INFO] Point {len(points)} selected: "
            f"({x}, {y})"
        )

    else:
        print(
            "[INFO] Four points already selected. "
            "Press 'r' to reset or 's' to save."
        )


# ============================================================
# BOARD ORIENTATION
# ============================================================

def convert_display_to_standard(row, col, rotation):
    """
    Convert a square's displayed row/column position
    into standard chessboard coordinates.

    Returns:
        standard_row, standard_column
    """

    if rotation == 0:
        standard_row = row
        standard_col = col

    elif rotation == 90:
        standard_row = col
        standard_col = 7 - row

    elif rotation == 180:
        standard_row = 7 - row
        standard_col = 7 - col

    elif rotation == 270:
        standard_row = 7 - col
        standard_col = row

    else:
        standard_row = row
        standard_col = col

    return standard_row, standard_col


# ============================================================
# CREATE PERSPECTIVE TRANSFORMATION
# ============================================================

def calculate_perspective_transform(corners):
    """
    Calculate the perspective transformation matrix
    for the selected chessboard corners.
    """

    source_points = np.array(
        [
            [0, 0],
            [8, 0],
            [8, 8],
            [0, 8]
        ],
        dtype=np.float32
    )

    destination_points = np.array(
        corners,
        dtype=np.float32
    )

    transformation_matrix = cv2.getPerspectiveTransform(
        source_points,
        destination_points
    )

    return transformation_matrix


# ============================================================
# CREATE BOARD GRID
# ============================================================

def generate_board_grid(transformation_matrix):
    """
    Generate the 9x9 grid points of the chessboard
    after applying perspective transformation.
    """

    grid_points = np.array(
        [
            [[x, y] for x in range(GRID_SIZE)]
            for y in range(GRID_SIZE)
        ],
        dtype=np.float32
    )

    transformed_grid = cv2.perspectiveTransform(
        grid_points.reshape(-1, 1, 2),
        transformation_matrix
    )

    return transformed_grid.reshape(
        GRID_SIZE,
        GRID_SIZE,
        2
    )


# ============================================================
# DRAW BOARD GRID
# ============================================================

def draw_board_grid(frame, grid):
    """
    Draw the 8x8 chessboard grid on the camera frame.
    """

    for row in range(GRID_SIZE):
        points_row = grid[row].astype(int)

        cv2.polylines(
            frame,
            [points_row],
            False,
            (180, 180, 180),
            1
        )

    for col in range(GRID_SIZE):
        points_column = grid[:, col].astype(int)

        cv2.polylines(
            frame,
            [points_column],
            False,
            (180, 180, 180),
            1
        )


# ============================================================
# DRAW CHESS SQUARE LABELS
# ============================================================

def draw_square_labels(frame, grid):
    """
    Display standard chess notation such as:
        a8, b8, c8 ... h1
    """

    font = cv2.FONT_HERSHEY_SIMPLEX

    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):

            # Convert displayed position to standard chess position
            standard_row, standard_col = (
                convert_display_to_standard(
                    row,
                    col,
                    CAMERA_ROTATION
                )
            )

            file_name = FILES[standard_col]
            rank_name = RANKS[standard_row]

            square_name = f"{file_name}{rank_name}"

            # Calculate square center
            top_left = grid[row, col]
            bottom_right = grid[row + 1, col + 1]

            center = top_left + (
                bottom_right - top_left
            ) / 2

            center_x = int(center[0])
            center_y = int(center[1])

            cv2.putText(
                frame,
                square_name,
                (center_x - 12, center_y + 5),
                font,
                0.5,
                (0, 255, 255),
                1,
                cv2.LINE_AA
            )


# ============================================================
# CREATE SQUARE DICTIONARY
# ============================================================

def create_square_dictionary(grid):
    """
    Create a dictionary containing the four corner coordinates
    of every chess square.

    Example:
        a8 -> [top-left, top-right, bottom-right, bottom-left]
    """

    square_dictionary = {}

    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):

            # Four corners of the current square
            top_left = grid[row, col].tolist()
            top_right = grid[row, col + 1].tolist()
            bottom_right = grid[row + 1, col + 1].tolist()
            bottom_left = grid[row + 1, col].tolist()

            square_coordinates = [
                top_left,
                top_right,
                bottom_right,
                bottom_left
            ]

            # Convert to standard chess coordinates
            standard_row, standard_col = (
                convert_display_to_standard(
                    row,
                    col,
                    CAMERA_ROTATION
                )
            )

            file_name = FILES[standard_col]
            rank_name = RANKS[standard_row]

            square_name = f"{file_name}{rank_name}"

            square_dictionary[square_name] = square_coordinates

    return square_dictionary


# ============================================================
# SAVE CALIBRATION DATA
# ============================================================

def save_calibration(grid):
    """
    Save chessboard square coordinates to sqdict.json.
    """

    square_dictionary = create_square_dictionary(grid)

    with open(OUTPUT_FILE, "w") as file:
        json.dump(
            square_dictionary,
            file,
            indent=2
        )

    print()
    print("[SUCCESS] Calibration saved.")
    print(f"[INFO] File: {OUTPUT_FILE}")
    print(f"[INFO] Camera rotation: {CAMERA_ROTATION}°")


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    # --------------------------------------------------------
    # Open camera
    # --------------------------------------------------------

    camera = cv2.VideoCapture(
        CAMERA_ID,
        cv2.CAP_DSHOW
    )

    if not camera.isOpened():
        print("[ERROR] Camera could not be opened.")
        return

    # --------------------------------------------------------
    # Create calibration window
    # --------------------------------------------------------

    cv2.namedWindow(WINDOW_NAME)

    cv2.setMouseCallback(
        WINDOW_NAME,
        handle_mouse_click
    )

    print()
    print("=" * 60)
    print("CHESS BOARD CALIBRATION")
    print("=" * 60)

    print("\nInstructions:")
    print("1. Point the camera at the chessboard.")
    print(
        "2. Click the four corners in this order:"
    )
    print("   Top-left → Top-right → Bottom-right → Bottom-left")
    print("3. Press 'r' to reset the points.")
    print("4. Press 's' to save calibration.")
    print("5. Press 'q' to quit.")
    print()
    print(
        f"[INFO] Camera rotation: "
        f"{CAMERA_ROTATION}°"
    )
    print()

    # --------------------------------------------------------
    # Main camera loop
    # --------------------------------------------------------

    while True:

        success, frame = camera.read()

        if not success:
            print("[WARNING] Failed to read camera frame.")
            continue

        display_frame = frame.copy()

        # ----------------------------------------------------
        # Draw selected corner points
        # ----------------------------------------------------

        for index, point in enumerate(points):

            cv2.circle(
                display_frame,
                point,
                6,
                (0, 0, 255),
                -1
            )

            cv2.putText(
                display_frame,
                str(index + 1),
                (point[0] + 8, point[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        # ----------------------------------------------------
        # Process board after four points are selected
        # ----------------------------------------------------

        if len(points) == 4:

            # Draw outer board boundary
            corner_points = np.array(
                points,
                dtype=np.int32
            )

            cv2.polylines(
                display_frame,
                [corner_points],
                True,
                (255, 255, 255),
                2
            )

            # Calculate perspective transformation
            transformation_matrix = (
                calculate_perspective_transform(points)
            )

            # Generate transformed board grid
            board_grid = generate_board_grid(
                transformation_matrix
            )

            # Draw grid
            draw_board_grid(
                display_frame,
                board_grid
            )

            # Draw chess notation
            draw_square_labels(
                display_frame,
                board_grid
            )

        # ----------------------------------------------------
        # Show camera frame
        # ----------------------------------------------------

        cv2.imshow(
            WINDOW_NAME,
            display_frame
        )

        key = cv2.waitKey(1) & 0xFF

        # ----------------------------------------------------
        # Quit
        # ----------------------------------------------------

        if key == ord("q"):

            print("[INFO] Exiting without saving.")
            break



        elif key == ord("r"):

            points.clear()

            print("[INFO] Calibration points reset.")


        elif key == ord("s"):

            if len(points) != 4:

                print(
                    "[WARNING] Select all four corners "
                    "before saving."
                )

                continue

            transformation_matrix = (
                calculate_perspective_transform(points)
            )

            board_grid = generate_board_grid(
                transformation_matrix
            )

            save_calibration(
                board_grid
            )

            break

    # --------------------------------------------------------
    # Release resources
    # --------------------------------------------------------

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()