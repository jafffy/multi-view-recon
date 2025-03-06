# Point Cloud Visualizer

A simple Python script to visualize PLY point cloud files.

## Requirements

- Python 3.6+
- Open3D library

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

```bash
python visualize_pointcloud.py path/to/your/pointcloud.ply
```

## Features

- Loads and visualizes PLY point cloud files
- Displays a coordinate frame for reference (RGB axes correspond to XYZ)
- Shows basic information about the point cloud (number of points, whether it has colors and normals)

## Controls

Once the visualization window opens:
- Left-click + drag: Rotate the view
- Right-click + drag: Pan the view
- Mouse wheel: Zoom in/out
- Press 'h' to show help with more keyboard shortcuts

## Troubleshooting

If you encounter any issues:
1. Make sure your PLY file is valid and readable
2. Check if Open3D is installed correctly
3. For large point clouds, ensure your system has sufficient memory 