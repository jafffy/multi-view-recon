# Point Cloud Alignment and Fusion

This module aligns and fuses an RGB-D point cloud with an existing PLY model. It uses feature-based alignment (RANSAC), fine registration (ICP), and weighted fusion to create an accurate combined model.

## Features

- **Initial Alignment**: Using Fast Point Feature Histograms (FPFH) with RANSAC for robust matching
- **Fine Registration**: Point-to-Plane ICP for precise alignment
- **Weighted Fusion**: Merge point clouds with customizable weighting
- **Outlier Removal**: Clean up the fused point cloud
- **Visualization**: Interactive visualization at each step

## Requirements

- Python 3.6+
- Open3D 0.15.0+
- NumPy
- Other standard Python libraries (os, sys, copy, argparse, time, pathlib)

Install dependencies:

```bash
pip install open3d numpy
```

## Usage

Basic usage to align and fuse two point clouds:

```bash
python point_cloud_alignment.py --reference model.ply --target captured_cloud.ply
```

### Command Line Arguments

- `--reference`: Path to the reference PLY model (ground truth)
- `--target`: Path to the target point cloud to align
- `--output`: Path to save the fused model (default: results/fused_model.ply)
- `--voxel_size`: Voxel size for downsampling (default: 0.05)
- `--no_visualization`: Disable visualization
- `--output_dir`: Directory to save results (default: results)

### Example

```bash
# Align captured point cloud with reference model, with higher resolution
python point_cloud_alignment.py --reference model.ply --target captured_cloud.ply --voxel_size 0.01 --output fused.ply
```

## Integration with Existing Pipeline

This alignment module can be integrated with your existing RGB-D processing pipeline:

1. Use `virtual_camera_capture.py` to capture RGB-D images from a 3D model
2. Convert RGB-D images to point clouds using `rgbd_point_cloud_viewer.py`
3. Align and fuse these point clouds with a reference model using `point_cloud_alignment.py`

## Implementation Details

### 1. Preprocessing

Before alignment, the point clouds are preprocessed:
- Downsampling using voxel grid filtering
- Normal estimation
- FPFH feature extraction for matching

### 2. Initial Alignment (RANSAC)

The initial alignment uses feature matching with RANSAC:
- FPFH features are extracted from both point clouds
- Robust matching finds an initial transformation
- This provides a coarse alignment to seed the ICP algorithm

### 3. Fine Registration (ICP)

The initial alignment is refined using Point-to-Plane ICP:
- Uses the transformation from RANSAC as starting point
- Iteratively minimizes the point-to-plane distance
- Provides precise alignment even with noisy data

### 4. Weighted Fusion

The aligned point clouds are merged with weighted fusion:
- The target point cloud is transformed to the reference coordinate system
- Points from both clouds are combined with customizable weights
- Colors and normals are properly handled
- Voxel downsampling removes redundant points
- Statistical outlier removal cleans up the result

## Customization

You can customize the alignment process by modifying:
- `voxel_size`: Controls the resolution of the downsampling
- `weight_source` and `weight_target` in `fuse_point_clouds()`: Adjust the influence of each point cloud
- Registration parameters in the `__init__` method

## Troubleshooting

- **Poor alignment**: Try adjusting the `voxel_size` or increasing `ransac_max_iterations`
- **Slow performance**: Increase `voxel_size` for faster (but less accurate) results
- **Visualization issues**: Make sure culling is disabled in the visualization options

## License

This project is open-source under the MIT License. 