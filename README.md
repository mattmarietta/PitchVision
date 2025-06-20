# PitchVision

PitchVision is a computer vision pipeline for analyzing soccer match videos. It uses deep learning and classical computer vision techniques to track players and the ball, estimate camera movement, assign teams, calculate player speed and distance, and visualize player heatmaps.

## Features

- **Object Tracking**: Detects and tracks players and the ball using a YOLOv8-based model.
- **Camera Movement Estimation**: Estimates and compensates for camera movement to improve tracking accuracy.
- **Team Assignment**: Assigns players to teams based on color clustering.
- **Ball Possession Assignment**: Determines which player has ball possession at each frame.
- **Speed & Distance Calculation**: Computes speed and distance covered by each player.
- **View Transformation**: Maps player positions to a top-down view of the field.
- **Player Heatmaps**: Visualizes player movement density over time.
- **Output Video**: Annotates and saves the processed video with all computed information.

[![Watch the demo](https://img.youtube.com/vi/rcun0pR1yZ8/0.jpg)](https://youtu.be/rcun0pR1yZ8)

// ... existing code ...
## Project Structure

```
SoccerML/
  ├── camera_movement/         # Camera movement estimation
  ├── dev_and_analysis/        # Notebooks for development and analysis
  ├── heatmap_player/          # Player heatmap generation
  ├── input_videos/            # Input soccer videos
  ├── main.py                  # Main pipeline script
  ├── models/                  # YOLOv8 model weights
  ├── output_videos/           # Output annotated videos
  ├── player_ball_assignment/  # Ball possession assignment
  ├── speed_and_distance/      # Speed and distance calculation
  ├── stubs/                   # Precomputed stubs for reproducibility/speed
  ├── team_assigner/           # Team color assignment
  ├── trackers/                # Object tracking logic
  ├── utils/                   # Utility functions
  ├── view_transformer/        # Perspective transformation
  └── yolo_inf.py              # YOLOv8 inference example
```

## Getting Started

### Prerequisites

- Python 3.8+
- [Ultralytics YOLOv8](https://docs.ultralytics.com/)
- OpenCV
- NumPy
- scikit-learn

Install dependencies:

```bash
pip install ultralytics opencv-python numpy scikit-learn
```

### Usage

1. **Place your input video** in the `input_videos/` directory (default: `08fd33_4.mp4`).
2. **Download or train YOLOv8 weights** and place them in the `models/` directory (`best.pt` or `last.pt`).
3. **Run the main pipeline:**

   ```bash
   python main.py
   ```

4. **Output** will be saved to `output_videos/output_video.avi`.

### Example

```python
python main.py
```

This will process the default video, track all players and the ball, assign teams, calculate speed and distance, and generate an annotated output video.

## Modules Overview

- **main.py**: Orchestrates the full pipeline.
- **trackers/**: Implements YOLOv8-based tracking and annotation.
- **camera_movement/**: Estimates and compensates for camera motion.
- **team_assigner/**: Assigns players to teams using color clustering.
- **player_ball_assignment/**: Assigns ball possession to players.
- **speed_and_distance/**: Calculates player speed and distance.
- **view_transformer/**: Transforms positions to a top-down field view.
- **heatmap_player/**: Generates and overlays player heatmaps.
- **utils/**: Video I/O and geometric utilities.
- **stubs/**: Precomputed results for faster runs and reproducibility.

## Customization

- To use a different input video, change the path in `main.py` or replace the file in `input_videos/`.
- To use your own YOLOv8 weights, place them in `models/` and update the model path in `main.py`.

## Notes

- The pipeline uses precomputed stubs for tracking and camera movement by default for speed. To run full inference, set `read_from_stub=False` in `main.py`.
- The `dev_and_analysis/` folder contains Jupyter notebooks for further analysis and experimentation. Might want to run the models on a Collab if you do not have access to a CUDA-enabled GPU.
- This project was developed with the help of an online tutorial as a learning exercise.

