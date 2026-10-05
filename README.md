# ♟️ AI Chess Assistant

A computer-vision-based chess assistant that tracks moves on a physical chessboard using a webcam, OpenCV, Python Chess, and the Stockfish chess engine.

The system uses a calibrated camera view of the chessboard to detect changes between frames, determine the player's move, validate the move using legal chess rules, and generate a response using Stockfish.

---
 📌 Overview

This project connects a physical chessboard with a computer vision system.

Instead of manually entering moves into a chess program, the camera observes the physical board and detects when pieces are moved.

### System Pipeline

Physical Chess Board
        │
        ▼
     Webcam
        │
        ▼
Board Calibration
        │
        ▼
Perspective / Square Mapping
        │
        ▼
OpenCV Frame Difference
        │
        ▼
Changed Square Detection
        │
        ▼
Legal Move Validation
        │
        ▼
   python-chess
        │
        ▼
    Stockfish
        │
        ▼
Computer Move


## 📷 PROJECT STRUCTURE

ai-chess-assistant/
│
├── assets/
│   └── images/
│       └── demo/
│
├── calibration/
│   ├── calibrate_board.py
│   └── sqdict.json
│
├── src/
│   └── main.py
│
├── stockfish/
│   └── stockfish-windows-x86-64-avx2.exe
│
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
        │
        ▼
Board Visualization


## ♟️ Stockfish Setup

This project uses Stockfish as the chess engine responsible for generating the computer's moves.
The Stockfish executable is not included in this repository.
After obtaining a compatible Stockfish executable, place it inside:


## 🎯 Board Calibration
Before running the chess assistant, the physical chessboard needs to be calibrated.
RUN: 
    python calibration/calibrate_board.py

The calibration program opens the webcam and asks you to select the four corners of the chessboard.
Select the corners in this order:
1. Top-left
2. Top-right
3. Bottom-right
4. Bottom-left

The calibration system then generates the coordinates of all 64 chess squares and saves them to:
    calibration/sqdict.json



## ▶️ Running the Chess Assistant
From the project root, run:
  python src/main.py
  
The program will:
1. Load the calibrated board coordinates.
2. Start the webcam.
3. Initialize the Stockfish chess engine.
4. Display the camera feed.
5. Wait for a reference frame.
6. Compare subsequent frames to detect board changes.
7. Determine the player's move.
8. Validate the move against the current chess position.
9. Update the chess board.
10. Ask Stockfish for the computer's response.
11. Display the updated board state.



## 🔍 How Move Detection Works
The project does not identify individual chess pieces using an object-detection model.
Instead, it uses image changes between board frames.
The basic process is:

Reference Frame
       │
       ▼
Current Frame
       │
       ▼
Grayscale Conversion
       │
       ▼
Gaussian Blur
       │
       ▼
Absolute Difference
       │
       ▼
Thresholding
       │
       ▼
Morphological Processing
       │
       ▼
Contour Detection
       │
       ▼
Changed Chess Squares
       │
       ▼
Legal Move Analysis

The detected changes are compared against the current chess position and legal moves to determine the most likely player move.



## ⚠️ Current Limitations 

The current system relies on computer vision and frame differences rather than directly recognizing individual chess pieces.
Performance can therefore be affected by:
- Camera movement
- Changes in lighting
- Shadows on the board
- Reflections
- Board positioning
- Camera angle
- Piece movement that produces weak visual differences
The camera and chessboard should remain relatively stable after calibration.
Special chess moves such as castling, en passant, and promotion may require additional testing and refinement in the physical-board detection pipeline.


## 🔮 Future Improvements 

Possible future improvements include:
- Improved piece detection
- More robust lighting handling
- Automatic board detection
- Better move detection
- Support for multiple camera configurations
- Improved handling of special chess moves
- Real-time move visualization
- More robust camera calibration
- Hardware integration with a physical chessboard

## 👨‍💻 Author
Arghajeet Das
Computer Science & Engineering Student
Rustamji Institute of Technology
GitHub:
https://github.com/ARGHAJEET2007
