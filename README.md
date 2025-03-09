# Point Cloud Registration Pipeline

This repository contains a Python implementation of a complete pipeline for registering multiple RGB-D point clouds. The pipeline uses feature-based global registration followed by fine registration and multi-view pose graph optimization.

## Features

- **Preprocessing**: Downsampling, normal estimation, and FPFH feature extraction
- **Global Registration**: Using FPFH features with RANSAC for coarse alignment
- **Fine Registration**: Point-to-Plane ICP for accurate alignment
- **Multi-View Registration**: Pose Graph Optimization to minimize global errors
- **Visualization**: Interactive visualization of the registration results
- **Result Storage**: Save the integrated point cloud and transformation matrices

## Requirements

- Python 3.6+
- Open3D 0.15.0+
- NumPy
- Other standard Python libraries (os, sys, copy, argparse, time, pathlib)

Install the dependencies:

```bash
pip install open3d numpy
```

## Usage

### 1. Prepare Your Point Cloud Data

Place your point cloud files (`.ply` or `.pcd` format) in a directory. The files should be captured from different perspectives of the same scene.

You can use our `rgbd_point_cloud_viewer.py` script to convert RGB-D images to point clouds:

```bash
python rgbd_point_cloud_viewer.py --view 0 --no-display-images --depth-only
```

Repeat for different views to create a set of point clouds.

### 2. Run the Registration Pipeline

Run the registration pipeline with your point cloud data:

```bash
python point_cloud_registration.py --input_dir /path/to/your/point_clouds --output_dir results
```

### Command Line Arguments

- `--input_dir`: Directory containing input point cloud files (required)
- `--output_dir`: Directory to save results (default: "results")
- `--voxel_size`: Voxel size for downsampling (default: 0.05)
- `--no_visualization`: Disable visualization
- `--no_save`: Disable saving results

## Pipeline Overview

1. **Loading and Preprocessing**:
   - Load point clouds from files
   - Downsample using voxel grid filtering
   - Estimate surface normals
   - Compute FPFH features for feature matching

2. **Pairwise Registration**:
   - Global registration using FPFH + RANSAC
   - Fine registration using Point-to-Plane ICP

3. **Multi-View Registration**:
   - Construct a pose graph with nodes (point clouds) and edges (transformations)
   - Include loop closures between non-consecutive frames
   - Optimize the pose graph using Levenberg-Marquardt optimization

4. **Integration and Visualization**:
   - Transform all point clouds to a common coordinate system
   - Merge the transformed point clouds
   - Visualize the integrated result

## Output

The pipeline outputs:
- An integrated point cloud saved as `integrated.ply`
- Transformation matrices saved as `transformations.npy`

## Tips for Better Results

- Use a smaller voxel size for more accurate registration (but slower processing)
- Ensure good overlap between consecutive point clouds
- For challenging scenes, consider pre-aligning the point clouds
- Adjust the registration parameters in the script for your specific data

## Example

```bash
# Register a set of point clouds with a voxel size of 0.02
python point_cloud_registration.py --input_dir ./my_point_clouds --voxel_size 0.02
```

## License

This project is open-source under the MIT License.

# Multi-View RGBD Point Cloud Visualization

This script visualizes multiple RGBD views into a combined 3D point cloud. It uses the camera parameters from a JSON file to correctly position each view in 3D space.

## Features

- Loads depth and color data from multiple views
- Creates colored point clouds for each view
- Option to combine all point clouds into a single representation
- Option to use depth-only visualization
- Supports exporting combined point clouds to PLY files

## Usage

```bash
# Process all views and visualize them separately
python multi_view_point_cloud.py

# Process specific views (comma-separated, no spaces)
python multi_view_point_cloud.py --views 0,1,2,3

# Combine all views into a single point cloud
python multi_view_point_cloud.py --combine

# Combine specific views and save the result
python multi_view_point_cloud.py --views 0,5,10,15 --combine --save combined_pointcloud.ply

# Use depth-only visualization (no color)
python multi_view_point_cloud.py --depth-only

# Specify a different camera parameters file
python multi_view_point_cloud.py --params path/to/camera_parameters.json
```

## Command Line Options

- `--params`: Path to camera parameters JSON file
- `--views`: Comma-separated list of view indices to visualize (e.g., '0,1,2') or 'all'
- `--combine`: Combine all point clouds into a single one (default: show separate)
- `--save`: Save the combined point cloud to the specified file path (e.g., 'output.ply')
- `--depth-only`: Create point cloud using only depth data (no color)

## Controls

When the visualization window opens:
- Left-click + drag: Rotate the view
- Mouse wheel: Zoom in/out
- Shift + left-click + drag: Pan
- Press 'h': Display help for more controls
- Press 'q': Exit the viewer

## Requirements

- Python 3.x
- NumPy
- Matplotlib
- Open3D
- PIL (Pillow)

## Notes

The script expects a JSON file with camera parameters in the format specified in the sample. Each view must have the following parameters:
- `intrinsic_matrix`: Camera intrinsic matrix
- `extrinsic_matrix`: Camera extrinsic matrix
- `depth_path`: Path to depth data file (.npy)
- `image_path`: Path to color image file
- `depth_min`: Minimum depth value
- `depth_max`: Maximum depth value
- `depth_scale`: Depth scale factor 