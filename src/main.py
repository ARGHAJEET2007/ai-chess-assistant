import cv2
import json
import numpy as np
import chess
import chess.engine
import chess.svg
from PIL import Image
import io
import random
import os
import sys
import cairosvg


# ============================================================
# CONFIGURATION
# ============================================================

CALIBRATION_FILE = "calibration/sqdict.json"
STOCKFISH_PATH = r"stockfish/stockfish-windows-x86-64-avx2.exe"

CAMERA_INDEX = 0

MOVE_THRESHOLD = 25
MIN_CONTOUR_AREA = 250

BOARD_ORIENTATION = "TOP"
# Available orientations:
# "TOP"
# "BOTTOM"
# "SIDE_L"
# "SIDE_R"

DEBUG_MODE = False

CHESS_FILES = "abcdefgh"
CHESS_RANKS = "12345678"


# ============================================================
# LOAD STOCKFISH
# ============================================================

def start_stockfish():
    """Start the Stockfish chess engine."""

    if not os.path.exists(STOCKFISH_PATH):
        print(
            f"[ERROR] Stockfish engine not found: "
            f"{STOCKFISH_PATH}"
        )
        sys.exit(1)

    engine = chess.engine.SimpleEngine.popen_uci(
        STOCKFISH_PATH
    )

    print(
        f"[INFO] Stockfish started from: "
        f"{STOCKFISH_PATH}"
    )

    return engine


# ============================================================
# LOAD CALIBRATION DATA
# ============================================================

def load_calibration():
    """Load chessboard square coordinates from JSON."""

    if not os.path.exists(CALIBRATION_FILE):
        print(
            f"[ERROR] Calibration file not found: "
            f"{CALIBRATION_FILE}"
        )
        sys.exit(1)

    with open(CALIBRATION_FILE, "r") as file:
        square_points = json.load(file)

    print(
        f"[INFO] Loaded {len(square_points)} squares "
        f"from {CALIBRATION_FILE}"
    )

    return square_points


# ============================================================
# BOARD ORIENTATION
# ============================================================

def remap_square(square_name):
    """
    Convert a detected square according to the selected
    camera/board orientation.
    """

    file_name = square_name[0]
    rank_name = square_name[1]

    file_index = CHESS_FILES.index(file_name)
    rank_index = CHESS_RANKS.index(rank_name)

    if BOARD_ORIENTATION == "TOP":
        return square_name

    elif BOARD_ORIENTATION == "BOTTOM":
        return (
            f"{CHESS_FILES[7 - file_index]}"
            f"{CHESS_RANKS[7 - rank_index]}"
        )

    elif BOARD_ORIENTATION == "SIDE_L":
        return (
            f"{CHESS_FILES[rank_index]}"
            f"{CHESS_RANKS[7 - file_index]}"
        )

    elif BOARD_ORIENTATION == "SIDE_R":
        return (
            f"{CHESS_FILES[7 - rank_index]}"
            f"{CHESS_RANKS[file_index]}"
        )

    return square_name


# ============================================================
# GEOMETRY HELPERS
# ============================================================

def get_polygon_center(points):
    """
    Calculate the center point of a polygon.
    """

    polygon = np.array(
        points,
        np.int32
    )

    moments = cv2.moments(polygon)

    if moments["m00"] == 0:
        return (
            int(polygon[:, 0].mean()),
            int(polygon[:, 1].mean())
        )

    center_x = int(
        moments["m10"] / moments["m00"]
    )

    center_y = int(
        moments["m01"] / moments["m00"]
    )

    return center_x, center_y


def find_square(x, y, square_points):
    """
    Find which chessboard square contains point (x, y).
    """

    point = (float(x), float(y))

    for square_name, points in square_points.items():

        polygon = np.array(
            points,
            np.int32
        )

        inside = cv2.pointPolygonTest(
            polygon,
            point,
            False
        )

        if inside >= 0:
            return square_name

    return None


# ============================================================
# VISUALIZATION HELPERS
# ============================================================

def overlay_polygon(
    frame,
    polygon_points,
    color,
    alpha=0.45
):
    """
    Overlay a transparent colored polygon on the frame.
    """

    overlay = frame.copy()

    polygon = np.array(
        polygon_points,
        np.int32
    )

    cv2.fillPoly(
        overlay,
        [polygon],
        color
    )

    return cv2.addWeighted(
        overlay,
        alpha,
        frame,
        1 - alpha,
        0
    )


def draw_board_labels(
    frame,
    square_points
):
    """
    Draw chessboard boundaries and the a1 label.
    """

    output = frame.copy()

    font = cv2.FONT_HERSHEY_SIMPLEX

    for square_name, points in square_points.items():

        polygon = np.array(
            points,
            np.int32
        )

        cv2.polylines(
            output,
            [polygon],
            True,
            (255, 255, 255),
            1
        )

        # Keep the original behavior:
        # only display the mapped a1 label.
        if square_name == "a1":

            center_x, center_y = get_polygon_center(
                points
            )

            mapped_square = remap_square(
                square_name
            )

            cv2.putText(
                output,
                mapped_square,
                (
                    center_x - 12,
                    center_y + 5
                ),
                font,
                0.45,
                (0, 255, 255),
                1,
                cv2.LINE_AA
            )

    return output


def show_debug_contours(
    frame,
    contours
):
    """
    Draw contour bounding boxes and centers
    for debugging.
    """

    debug_frame = frame.copy()

    for contour in contours:

        area = cv2.contourArea(contour)

        x, y, width, height = cv2.boundingRect(
            contour
        )

        moments = cv2.moments(contour)

        if moments["m00"] != 0:

            center_x = int(
                moments["m10"] / moments["m00"]
            )

            center_y = int(
                moments["m01"] / moments["m00"]
            )

        else:

            center_x = x + width // 2
            center_y = y + height // 2

        cv2.rectangle(
            debug_frame,
            (x, y),
            (x + width, y + height),
            (0, 255, 0),
            2
        )

        cv2.circle(
            debug_frame,
            (center_x, center_y),
            3,
            (0, 0, 255),
            -1
        )

        cv2.putText(
            debug_frame,
            f"A:{int(area)}",
            (x, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )

    return debug_frame


# ============================================================
# CHESS BOARD DISPLAY
# ============================================================

def show_chess_board(
    board,
    last_move=None
):
    """
    Display the current chess position
    using python-chess SVG.
    """

    svg = chess.svg.board(
        board=board,
        lastmove=last_move,
        coordinates=True,
        size=450
    )

    png_data = cairosvg.svg2png(
        bytestring=svg.encode("utf-8")
    )

    image = Image.open(
        io.BytesIO(png_data)
    )

    image_cv = cv2.cvtColor(
        np.array(image),
        cv2.COLOR_RGB2BGR
    )

    cv2.imshow(
        "Board State",
        image_cv
    )

    cv2.waitKey(1)


# ============================================================
# CREATE BOARD MASK
# ============================================================

def create_board_mask(
    image_shape,
    square_points
):
    """
    Create a mask covering only the chessboard.
    """

    mask = np.zeros(
        image_shape,
        dtype=np.uint8
    )

    for points in square_points.values():

        polygon = np.array(
            points,
            np.int32
        )

        cv2.fillPoly(
            mask,
            [polygon],
            255
        )

    return mask


# ============================================================
# FRAME DIFFERENCE
# ============================================================

def calculate_frame_difference(
    reference_frame,
    current_frame
):
    """
    Calculate the difference between the reference
    frame and the current frame.
    """

    # Convert both frames to weighted grayscale.
    # This is kept from the original implementation
    # to work with light and dark chess pieces.

    gray_reference = (
        0.5 * reference_frame[:, :, 2]
        + 0.4 * reference_frame[:, :, 1]
        + 0.1 * reference_frame[:, :, 0]
    )

    gray_current = (
        0.5 * current_frame[:, :, 2]
        + 0.4 * current_frame[:, :, 1]
        + 0.1 * current_frame[:, :, 0]
    )

    gray_reference = gray_reference.astype(
        np.uint8
    )

    gray_current = gray_current.astype(
        np.uint8
    )

    # Reduce camera noise
    gray_reference = cv2.GaussianBlur(
        gray_reference,
        (5, 5),
        0
    )

    gray_current = cv2.GaussianBlur(
        gray_current,
        (5, 5),
        0
    )

    # Calculate frame difference
    difference = cv2.absdiff(
        gray_reference,
        gray_current
    )

    difference = cv2.GaussianBlur(
        difference,
        (3, 3),
        0
    )

    # Increase difference contrast
    difference = cv2.convertScaleAbs(
        difference,
        alpha=1.3,
        beta=0
    )

    # Threshold
    _, threshold = cv2.threshold(
        difference,
        MOVE_THRESHOLD,
        255,
        cv2.THRESH_BINARY
    )

    # Expand detected areas
    threshold = cv2.dilate(
        threshold,
        None,
        iterations=4
    )

    # Remove excessive expansion
    threshold = cv2.erode(
        threshold,
        None,
        iterations=2
    )

    return threshold


# ============================================================
# CLEAN DIFFERENCE MASK
# ============================================================

def clean_difference_mask(
    difference_mask,
    square_points
):
    """
    Limit difference detection to the chessboard
    and clean the resulting mask.
    """

    board_mask = create_board_mask(
        difference_mask.shape,
        square_points
    )

    difference_mask = cv2.bitwise_and(
        difference_mask,
        board_mask
    )

    kernel = np.ones(
        (3, 3),
        np.uint8
    )

    difference_mask = cv2.morphologyEx(
        difference_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    difference_mask = cv2.morphologyEx(
        difference_mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    return difference_mask


# ============================================================
# CONTOUR CANDIDATES
# ============================================================

def get_square_candidates(
    contour,
    square_points
):
    """
    Generate possible chessboard squares for a contour.

    Multiple vertical positions and small horizontal offsets
    are tested because the top of a chess piece is often more
    reliable than the contour centroid.
    """

    x, y, width, height = cv2.boundingRect(
        contour
    )

    moments = cv2.moments(contour)

    if moments["m00"] != 0:

        center_x = int(
            moments["m10"] / moments["m00"]
        )

    else:

        center_x = x + width // 2

    # Test several vertical positions.
    vertical_factors = [
        0.20,
        0.30,
        0.40
    ]

    # Small horizontal adjustments.
    horizontal_offsets = [
        0,
        -6,
        6
    ]

    candidates = []

    for vertical_factor in vertical_factors:

        test_y = int(
            y + vertical_factor * height
        )

        for horizontal_offset in horizontal_offsets:

            test_x = (
                center_x
                + horizontal_offset
            )

            square = find_square(
                test_x,
                test_y,
                square_points
            )

            if square:
                candidates.append(
                    (
                        square,
                        test_x,
                        test_y
                    )
                )

    # Remove duplicate squares
    unique_candidates = []
    seen_squares = set()

    for candidate in candidates:

        square = candidate[0]

        if square not in seen_squares:

            seen_squares.add(square)
            unique_candidates.append(
                candidate
            )

    return unique_candidates


def get_contour_candidates(
    contours,
    square_points
):
    """
    Build candidate square lists for all valid contours.
    """

    valid_contours = [
        contour
        for contour in contours
        if cv2.contourArea(contour)
        > MIN_CONTOUR_AREA
    ]

    candidate_lists = []

    for contour in valid_contours:

        candidates = get_square_candidates(
            contour,
            square_points
        )

        candidate_lists.append(
            (
                contour,
                candidates
            )
        )

    return candidate_lists


# ============================================================
# FIND CHESS MOVE FROM CONTOURS
# ============================================================

def detect_changed_squares(
    contours,
    board,
    square_points
):
    """
    Determine which chessboard squares changed
    after the player moved a piece.
    """

    candidate_lists = get_contour_candidates(
        contours,
        square_points
    )

    detected_squares = set()
    chosen_mapping = []

    previous_board = board.copy()

    # --------------------------------------------------------
    # Two or more contours
    # --------------------------------------------------------

    if len(candidate_lists) >= 2:

        candidate_lists = sorted(
            candidate_lists,
            key=lambda item:
                cv2.contourArea(item[0]),
            reverse=True
        )[:2]

        contour_0, candidates_0 = candidate_lists[0]
        contour_1, candidates_1 = candidate_lists[1]

        found_legal_move = False

        # Try every combination
        for square_0, x_0, y_0 in candidates_0:

            for square_1, x_1, y_1 in candidates_1:

                possible_orders = [
                    (square_0, square_1),
                    (square_1, square_0)
                ]

                for from_square, to_square in possible_orders:

                    try:
                        move = chess.Move.from_uci(
                            from_square + to_square
                        )

                    except Exception:
                        continue

                    if move in previous_board.legal_moves:

                        detected_squares = {
                            from_square,
                            to_square
                        }

                        chosen_mapping = [
                            (
                                from_square,
                                x_0 if from_square == square_0 else x_1,
                                y_0 if from_square == square_0 else y_1
                            ),
                            (
                                to_square,
                                x_1 if to_square == square_1 else x_0,
                                y_1 if to_square == square_1 else y_0
                            )
                        ]

                        found_legal_move = True
                        break

                if found_legal_move:
                    break

            if found_legal_move:
                break

        # Fallback to first candidates
        if not found_legal_move:

            if candidates_0 and candidates_1:

                square_0 = candidates_0[0][0]
                square_1 = candidates_1[0][0]

                detected_squares = {
                    square_0,
                    square_1
                }

                chosen_mapping = [
                    (
                        square_0,
                        candidates_0[0][1],
                        candidates_0[0][2]
                    ),
                    (
                        square_1,
                        candidates_1[0][1],
                        candidates_1[0][2]
                    )
                ]

    # --------------------------------------------------------
    # One contour
    # --------------------------------------------------------

    elif len(candidate_lists) == 1:

        contour, candidates = candidate_lists[0]

        if candidates:

            destination = None

            # Prefer candidate that is a legal destination
            for square, x, y in candidates:

                legal_moves = [
                    move
                    for move in previous_board.legal_moves
                    if move.uci()[2:] == square
                ]

                if legal_moves:

                    destination = (
                        square,
                        x,
                        y
                    )

                    break

            if destination:

                detected_squares = {
                    destination[0]
                }

                chosen_mapping = [
                    destination
                ]

            else:

                first_candidate = candidates[0]

                detected_squares = {
                    first_candidate[0]
                }

                chosen_mapping = [
                    first_candidate
                ]

    return detected_squares, chosen_mapping


# ============================================================
# INTERPRET DETECTED SQUARES
# ============================================================

def interpret_move(
    detected_squares,
    board
):
    """
    Convert detected changed squares into
    from_square and to_square.
    """

    if not detected_squares:

        return None, None

    previous_board = board.copy()

    # --------------------------------------------------------
    # Two changed squares
    # --------------------------------------------------------

    if len(detected_squares) == 2:

        square_a, square_b = list(
            detected_squares
        )

        piece_a = previous_board.piece_at(
            chess.parse_square(square_a)
        )

        piece_b = previous_board.piece_at(
            chess.parse_square(square_b)
        )

        # One square has a piece, one does not.
        if piece_a and not piece_b:

            return square_a, square_b

        if piece_b and not piece_a:

            return square_b, square_a

        # Fallback based on rank direction
        def rank_number(square):
            return int(square[1])

        if board.turn == chess.WHITE:

            from_square, to_square = sorted(
                [square_a, square_b],
                key=rank_number
            )

        else:

            from_square, to_square = sorted(
                [square_a, square_b],
                key=rank_number,
                reverse=True
            )

        return from_square, to_square

    # --------------------------------------------------------
    # One changed square
    # --------------------------------------------------------

    if len(detected_squares) == 1:

        to_square = list(
            detected_squares
        )[0]

        piece_on_destination = board.piece_at(
            chess.parse_square(to_square)
        )

        # ----------------------------------------------------
        # Destination currently contains a piece
        # ----------------------------------------------------

        if piece_on_destination:

            legal_moves = [
                move
                for move in board.legal_moves
                if move.uci()[2:] == to_square
            ]

            for move in legal_moves:

                from_square = move.uci()[:2]

                if previous_board.piece_at(
                    chess.parse_square(from_square)
                ):

                    return (
                        from_square,
                        to_square
                    )

            if legal_moves:

                move = legal_moves[0]

                return (
                    move.uci()[:2],
                    move.uci()[2:]
                )

        # ----------------------------------------------------
        # Destination is empty
        # ----------------------------------------------------

        else:

            file_name = to_square[0]
            rank_number = int(to_square[1])

            file_index = CHESS_FILES.index(
                file_name
            )

            search_offsets = []

            if board.turn == chess.WHITE:

                search_offsets += [
                    (0, -1),
                    (-1, 0),
                    (1, 0),
                    (-1, -1),
                    (1, -1),
                    (0, -2)
                ]

            else:

                search_offsets += [
                    (0, 1),
                    (-1, 0),
                    (1, 0),
                    (-1, 1),
                    (1, 1),
                    (0, 2)
                ]

            search_offsets += [
                (-1, 1),
                (1, 1),
                (-1, -1),
                (1, -1)
            ]

            # ------------------------------------------------
            # Search likely source squares
            # ------------------------------------------------

            for file_offset, rank_offset in search_offsets:

                candidate_file = (
                    file_index + file_offset
                )

                candidate_rank = (
                    rank_number + rank_offset
                )

                if not (
                    0 <= candidate_file < 8
                    and 1 <= candidate_rank <= 8
                ):
                    continue

                candidate_square = (
                    f"{CHESS_FILES[candidate_file]}"
                    f"{candidate_rank}"
                )

                piece = previous_board.piece_at(
                    chess.parse_square(candidate_square)
                )

                if piece:

                    try:

                        move = chess.Move.from_uci(
                            candidate_square + to_square
                        )

                        if move in board.legal_moves:

                            return (
                                candidate_square,
                                to_square
                            )

                    except Exception:
                        pass

            # ------------------------------------------------
            # Last-resort neighbor search
            # ------------------------------------------------

            fallback_from_square = None

            for file_offset in (-1, 0, 1):

                for rank_offset in (-1, 0, 1):

                    if (
                        file_offset == 0
                        and rank_offset == 0
                    ):
                        continue

                    candidate_file = (
                        file_index + file_offset
                    )

                    candidate_rank = (
                        rank_number + rank_offset
                    )

                    if not (
                        0 <= candidate_file < 8
                        and 1 <= candidate_rank <= 8
                    ):
                        continue

                    candidate_square = (
                        f"{CHESS_FILES[candidate_file]}"
                        f"{candidate_rank}"
                    )

                    piece = previous_board.piece_at(
                        chess.parse_square(candidate_square)
                    )

                    if not piece:
                        continue

                    try:

                        move = chess.Move.from_uci(
                            candidate_square + to_square
                        )

                        if move in board.legal_moves:

                            return (
                                candidate_square,
                                to_square
                            )

                    except Exception:
                        pass

                    if fallback_from_square is None:

                        fallback_from_square = (
                            candidate_square
                        )

            if fallback_from_square:

                return (
                    fallback_from_square,
                    to_square
                )

    return None, None


# ============================================================
# PROCESS PLAYER MOVE
# ============================================================

def process_player_move(
    reference_frame,
    current_frame,
    board,
    square_points,
    debug_mode
):
    """
    Detect and interpret a player's chess move.
    """

    # --------------------------------------------------------
    # Calculate frame difference
    # --------------------------------------------------------

    difference = calculate_frame_difference(
        reference_frame,
        current_frame
    )

    # --------------------------------------------------------
    # Restrict detection to chessboard
    # --------------------------------------------------------

    difference_mask = clean_difference_mask(
        difference,
        square_points
    )

    if debug_mode:
        cv2.imshow(
            "Diff",
            difference_mask
        )

    # --------------------------------------------------------
    # Find contours
    # --------------------------------------------------------

    contours, _ = cv2.findContours(
        difference_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # --------------------------------------------------------
    # Detect changed squares
    # --------------------------------------------------------

    detected_squares, chosen_mapping = (
        detect_changed_squares(
            contours,
            board,
            square_points
        )
    )

    if debug_mode:

        print(
            "[DEBUG] Candidate list processing complete."
        )

        print(
            f"[DEBUG] Chosen mapping: "
            f"{chosen_mapping}"
        )

        print(
            f"[DEBUG] Detected squares: "
            f"{detected_squares}"
        )

    # --------------------------------------------------------
    # Debug visualization
    # --------------------------------------------------------

    if debug_mode and chosen_mapping:

        debug_frame = current_frame.copy()

        for square, center_x, center_y in chosen_mapping:

            polygon = np.array(
                square_points[square],
                np.int32
            )

            cv2.polylines(
                debug_frame,
                [polygon],
                True,
                (0, 255, 0),
                2
            )

            cv2.circle(
                debug_frame,
                (
                    int(center_x),
                    int(center_y)
                ),
                4,
                (0, 0, 255),
                -1
            )

            cv2.putText(
                debug_frame,
                square,
                (
                    int(center_x) + 6,
                    int(center_y)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1
            )

        cv2.imshow(
            "Contours",
            debug_frame
        )

    print(
        f"[DEBUG] Detected squares: "
        f"{detected_squares}"
    )

    # --------------------------------------------------------
    # Convert detected squares to chess move
    # --------------------------------------------------------

    from_square, to_square = interpret_move(
        detected_squares,
        board
    )

    if not from_square or not to_square:

        print(
            "[WARN] Could not determine a valid move."
        )

        return None, None, difference_mask

    move_string = (
        from_square + to_square
    )

    try:

        move = chess.Move.from_uci(
            move_string
        )

    except Exception as error:

        print(
            f"[ERROR] Could not create chess move: "
            f"{error}"
        )

        return None, None, difference_mask

    # --------------------------------------------------------
    # Check legality
    # --------------------------------------------------------

    if move not in board.legal_moves:

        print(
            f"[WARN] Invalid chess move: "
            f"{move_string}"
        )

        return None, None, difference_mask

    return (
        from_square,
        to_square,
        difference_mask
    )


# ============================================================
# HIGHLIGHT PLAYER MOVE
# ============================================================

def highlight_player_move(
    frame,
    from_square,
    to_square,
    square_points
):
    """
    Highlight player's FROM and TO squares.
    """

    output = overlay_polygon(
        frame,
        square_points[from_square],
        (0, 255, 0),
        0.5
    )

    output = overlay_polygon(
        output,
        square_points[to_square],
        (0, 0, 255),
        0.5
    )

    return draw_board_labels(
        output,
        square_points
    )


# ============================================================
# HIGHLIGHT COMPUTER MOVE
# ============================================================

def highlight_computer_move(
    frame,
    move,
    square_points
):
    """
    Highlight Stockfish's FROM and TO squares.
    """

    move_string = move.uci()

    from_square = move_string[:2]
    to_square = move_string[2:]

    output = overlay_polygon(
        frame,
        square_points[from_square],
        (0, 255, 255),
        0.45
    )

    output = overlay_polygon(
        output,
        square_points[to_square],
        (0, 165, 255),
        0.45
    )

    return draw_board_labels(
        output,
        square_points
    )


# ============================================================
# TOGGLE DEBUG WINDOWS
# ============================================================

def close_debug_windows():
    """
    Safely close debug windows.
    """

    for window_name in [
        "Diff",
        "Contours"
    ]:

        try:

            if (
                cv2.getWindowProperty(
                    window_name,
                    cv2.WND_PROP_VISIBLE
                ) >= 0
            ):

                cv2.destroyWindow(
                    window_name
                )

        except cv2.error:
            pass


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    global DEBUG_MODE

    # --------------------------------------------------------
    # Start Stockfish
    # --------------------------------------------------------

    engine = start_stockfish()

    # --------------------------------------------------------
    # Load calibration
    # --------------------------------------------------------

    square_points = load_calibration()

    # --------------------------------------------------------
    # Open camera
    # --------------------------------------------------------

    camera = cv2.VideoCapture(
        CAMERA_INDEX,
        cv2.CAP_DSHOW
    )

    if not camera.isOpened():

        print(
            "[ERROR] Camera could not be opened."
        )

        engine.quit()
        sys.exit(1)

    # --------------------------------------------------------
    # Initialize chess game
    # --------------------------------------------------------

    board = chess.Board()

    reference_frame = None

    last_move = None

    computer_turn = False

    move_history = []

    print()
    print("=" * 60)
    print("CHESS COMPUTER VISION TRACKER")
    print("=" * 60)
    print()
    print(
        "[INFO] Controls:"
    )
    print(
        "  r  = capture board state / detect move"
    )
    print(
        "  u  = undo 1 move"
    )
    print(
        "  U  = undo 2 moves"
    )
    print(
        "  d  = toggle debug mode"
    )
    print(
        "  q  = quit"
    )
    print()

    show_chess_board(
        board
    )

    try:

        # ====================================================
        # MAIN GAME LOOP
        # ====================================================

        while not board.is_game_over():

            success, frame = camera.read()

            if not success:
                continue

            # ------------------------------------------------
            # Display camera
            # ------------------------------------------------

            display_frame = draw_board_labels(
                frame.copy(),
                square_points
            )

            cv2.imshow(
                "Chess Tracker",
                display_frame
            )

            key = cv2.waitKey(1) & 0xFF

            # =================================================
            # TOGGLE DEBUG MODE
            # =================================================

            if key == ord("d"):

                DEBUG_MODE = not DEBUG_MODE

                state = (
                    "ON"
                    if DEBUG_MODE
                    else "OFF"
                )

                print(
                    f"[INFO] Debug mode: {state}"
                )

                if not DEBUG_MODE:
                    close_debug_windows()

            # =================================================
            # PLAYER MOVE
            # =================================================

            if key == ord("r"):

                # --------------------------------------------
                # First press:
                # Save reference frame
                # --------------------------------------------

                if reference_frame is None:

                    reference_frame = frame.copy()

                    print(
                        "[DEBUG] Reference frame captured."
                    )

                # --------------------------------------------
                # Second press:
                # Compare frames
                # --------------------------------------------

                else:

                    print(
                        "[DEBUG] Current frame captured. "
                        "Processing move..."
                    )

                    (
                        from_square,
                        to_square,
                        difference_mask
                    ) = process_player_move(
                        reference_frame,
                        frame,
                        board,
                        square_points,
                        DEBUG_MODE
                    )

                    # ----------------------------------------
                    # Execute player move
                    # ----------------------------------------

                    if from_square and to_square:

                        move_string = (
                            from_square
                            + to_square
                        )

                        try:

                            move = chess.Move.from_uci(
                                move_string
                            )

                            if move in board.legal_moves:

                                board.push(move)

                                move_history.append(
                                    move
                                )

                                last_move = move

                                print(
                                    f"[YOU] You played: "
                                    f"{move_string}"
                                )

                                show_chess_board(
                                    board,
                                    last_move
                                )

                                # Highlight move
                                try:

                                    highlighted = (
                                        highlight_player_move(
                                            frame.copy(),
                                            from_square,
                                            to_square,
                                            square_points
                                        )
                                    )

                                    cv2.imshow(
                                        "Chess Tracker",
                                        highlighted
                                    )

                                    cv2.waitKey(700)

                                except Exception as error:

                                    if DEBUG_MODE:

                                        print(
                                            "[DEBUG] "
                                            "Player highlight error: "
                                            f"{error}"
                                        )

                                computer_turn = True

                            else:

                                print(
                                    f"[WARN] Invalid move: "
                                    f"{move_string}"
                                )

                        except Exception as error:

                            print(
                                "[ERROR] Move interpretation "
                                f"failed: {error}"
                            )

                    else:

                        print(
                            "[WARN] No valid move detected."
                        )

                    # Reset reference frame
                    reference_frame = None

            # =================================================
            # UNDO ONE MOVE
            # =================================================

            if key == ord("u"):

                if move_history:

                    move = move_history.pop()

                    board.pop()

                    print(
                        f"[UNDO] Removed last move: "
                        f"{move}"
                    )

                    show_chess_board(
                        board
                    )

                else:

                    print(
                        "[INFO] No move available "
                        "for undo."
                    )

            # =================================================
            # UNDO TWO MOVES
            # =================================================

            if key == ord("U"):

                if len(move_history) >= 2:

                    second_move = (
                        move_history.pop()
                    )

                    first_move = (
                        move_history.pop()
                    )

                    board.pop()
                    board.pop()

                    print(
                        "[UNDO] Removed two moves: "
                        f"{first_move}, "
                        f"{second_move}"
                    )

                    show_chess_board(
                        board
                    )

                else:

                    print(
                        "[INFO] Not enough moves "
                        "for two-step undo."
                    )

            # =================================================
            # COMPUTER TURN
            # =================================================

            if computer_turn:

                result = engine.play(
                    board,
                    chess.engine.Limit(
                        time=random.uniform(
                            0.4,
                            0.9
                        )
                    )
                )

                computer_move = result.move

                board.push(
                    computer_move
                )

                move_history.append(
                    computer_move
                )

                last_move = computer_move

                print(
                    f"[AI] Computer played: "
                    f"{computer_move.uci()}"
                )

                show_chess_board(
                    board,
                    last_move
                )

                # --------------------------------------------
                # Highlight computer move
                # --------------------------------------------

                try:

                    highlighted = (
                        highlight_computer_move(
                            frame.copy(),
                            computer_move,
                            square_points
                        )
                    )

                    cv2.imshow(
                        "Chess Tracker",
                        highlighted
                    )

                    cv2.waitKey(900)

                except Exception as error:

                    if DEBUG_MODE:

                        print(
                            "[DEBUG] Computer highlight "
                            f"error: {error}"
                        )

                computer_turn = False

            # =================================================
            # QUIT
            # =================================================

            if key == ord("q"):

                print(
                    "[INFO] Exiting program."
                )

                break

        print(
            "[INFO] Chess game finished."
        )

    finally:

        # ----------------------------------------------------
        # Release resources
        # ----------------------------------------------------

        camera.release()

        cv2.destroyAllWindows()

        engine.quit()

        print(
            "[INFO] Camera and Stockfish closed."
        )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()