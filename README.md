# Multi-View Point Cloud Reconstruction

A system for capturing images from multiple viewpoints and reconstructing a 3D point cloud using computer vision techniques.

## Features

- Camera calibration to obtain intrinsic parameters
- Multi-view image capture with estimated camera poses
- Virtual camera capture for point cloud files
- 3D point cloud reconstruction using triangulation
- Visualization of reconstructed point clouds

## Requirements

- Python 3.6+
- OpenCV (cv2)
- Open3D
- NumPy
- tqdm

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Capture Multi-View Images (Physical Camera)

Use the capture script to take photos from multiple viewpoints around an object. This script will:
1. Calibrate your camera (using a chessboard pattern)
2. Guide you through capturing multiple views
3. Save images and camera parameters

```bash
python capture_multiview.py --output my_capture
```

Options:
- `--output`, `-o`: Directory to save captured data (default: auto-generated timestamp)
- `--camera`, `-c`: Camera device ID (default: 0)
- `--resolution`, `-r`: Camera resolution, format: WIDTHxHEIGHT (default: 1280x720)
- `--views`, `-v`: Number of views to capture (default: 20)
- `--no-calibration`: Skip camera calibration

### 2. Capture Multi-View Images (Virtual Camera)

If you already have a point cloud file (.ply) and want to capture it from multiple virtual viewpoints:

```bash
python virtual_camera_capture.py path/to/pointcloud.ply --output my_virtual_capture
```

The script will:
1. Load the point cloud file
2. Generate multiple synthetic views around the point cloud
3. Save the rendered images and camera parameters

Options:
- `--output`, `-o`: Directory to save captured data (default: auto-generated timestamp)
- `--resolution`, `-r`: Camera resolution, format: WIDTHxHEIGHT (default: 1280x720)
- `--views`, `-v`: Number of views to capture (default: 20)

### 3. Reconstruct 3D Point Cloud

After capturing images (using either physical or virtual camera), use the reconstruction script to generate a 3D point cloud:

```bash
python reconstruct_multiview.py --data my_capture --visualize
```

Options:
- `--data`, `-d`: Directory containing capture data (images and camera parameters)
- `--visualize`, `-v`: Visualize the reconstruction result

### 4. Visualize Existing Point Cloud

You can also visualize existing PLY files:

```bash
python main.py path/to/your/reconstruction.ply
```

## Process Overview

1. **Camera Calibration**: Determines the intrinsic parameters of your camera using a chessboard pattern (physical camera only)
2. **Multi-View Capture**: Takes photos from multiple viewpoints, estimating camera positions
3. **Feature Detection**: Detects SIFT features in each image
4. **Feature Matching**: Matches corresponding features between pairs of images
5. **Triangulation**: Calculates 3D positions of matched feature points
6. **Point Cloud Assembly**: Combines points from all view pairs into a single point cloud
7. **Outlier Removal**: Filters out noisy points
8. **Surface Estimation**: Estimates normals for better visualization

## Working with Virtual Cameras

When working with virtual cameras and existing point cloud files:
1. The `virtual_camera_capture.py` script renders the point cloud from different viewpoints
2. Camera parameters are precisely known (not estimated)
3. The reconstructed point cloud from these virtual views can be compared with the original
4. This is useful for testing reconstruction algorithms with known ground truth

## Troubleshooting

If you encounter issues:
1. Ensure good lighting conditions when capturing images (physical camera)
2. Use textured objects for better feature detection
3. Move the camera steadily between viewpoints
4. Capture images with sufficient overlap between views
5. Try different feature detection parameters if reconstruction fails 