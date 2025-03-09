# Complete RGB-D to Point Cloud Alignment Pipeline

This repository contains a complete end-to-end pipeline for multi-view 3D reconstruction using RGB-D data:

1. **RGB-D Capture**: Capture RGB-D images and depth data from multiple viewpoints of a 3D model
2. **Point Cloud Conversion**: Convert the RGB-D data into 3D point clouds
3. **Point Cloud Alignment**: Register, align, and fuse the point clouds into a single coherent model

## Components

The pipeline consists of several specialized modules:

### 1. Virtual Camera Capture (`virtual_camera_capture.py`)
- Captures RGB-D images from multiple viewpoints of a 3D model
- Generates depth maps and camera parameters (extrinsic/intrinsic matrices)
- Creates a structured dataset for 3D reconstruction

### 2. RGB-D Point Cloud Viewer (`rgbd_point_cloud_viewer.py`)
- Visualizes RGB-D data as point clouds
- Provides depth-only and colored point cloud visualization
- Renders 3D data with proper camera parameters

### 3. Point Cloud Alignment (`point_cloud_alignment.py`)
- Aligns point clouds using feature-based global registration (RANSAC)
- Refines alignment using point-to-plane ICP
- Fuses aligned point clouds with weighted fusion

### 4. Complete Pipeline (`rgbd_capture_to_alignment.py`)
- Integrates all components into a seamless workflow
- Manages intermediate data and processing steps
- Provides options to customize the pipeline for different use cases

## Requirements

- Python 3.6+
- Open3D 0.15.0+
- NumPy
- PIL (Pillow)
- Matplotlib
- Other standard Python libraries

Install required dependencies:

```bash
pip install open3d numpy pillow matplotlib
```

## Quick Start

To run the complete pipeline from RGB-D capture to point cloud alignment:

```bash
python rgbd_capture_to_alignment.py --ply_model /path/to/your/model.ply
```

This will:
1. Capture RGB-D data from multiple viewpoints
2. Convert the RGB-D data to point clouds
3. Align and fuse the point clouds into a single model

## Usage Options

The integrated pipeline supports several options:

```bash
python rgbd_capture_to_alignment.py --ply_model /path/to/your/model.ply [OPTIONS]
```

### Common Options

- `--ply_model`: Path to the input PLY model for RGB-D capture (required)
- `--reference_model`: Optional reference model for alignment (default: first captured point cloud)
- `--num_views`: Number of views to capture (default: 10)
- `--resolution`: Resolution for RGB-D capture (default: 1280x720)
- `--voxel_size`: Voxel size for point cloud processing (default: 0.01)
- `--output_dir`: Output directory (default: pipeline_output)
- `--skip_capture`: Skip RGB-D capture step (use existing data)
- `--skip_conversion`: Skip RGB-D to point cloud conversion step

## Example Workflow

### 1. Full Pipeline

```bash
# Run the full pipeline with 16 views and higher resolution
python rgbd_capture_to_alignment.py --ply_model path/to/model.ply --num_views 16 --voxel_size 0.005
```

### 2. Using Pre-captured Data

```bash
# Skip the capture step if you already have RGB-D data
python rgbd_capture_to_alignment.py --ply_model path/to/model.ply --skip_capture 
```

### 3. Using Pre-generated Point Clouds

```bash
# Skip both capture and conversion steps if you already have point clouds
python rgbd_capture_to_alignment.py --ply_model path/to/model.ply --skip_capture --skip_conversion
```

## Output Structure

The pipeline creates the following directory structure:

```
pipeline_output/
├── rgb_d_data/              # RGB-D data and camera parameters
│   ├── images/              # RGB images
│   ├── depth/               # Depth maps
│   └── camera_parameters.json
├── point_clouds/            # Generated point clouds
│   ├── point_cloud_000.ply
│   ├── point_cloud_001.ply
│   └── ...
└── aligned_models/          # Aligned and fused models
    ├── aligned_000.ply
    ├── aligned_001.ply
    ├── ...
    └── final_fused_model.ply # Final result
```

## Advanced Usage

### Individual Components

You can also use each component separately:

1. **RGB-D Capture**:
   ```bash
   python virtual_camera_capture.py path/to/model.ply --views 20
   ```

2. **Point Cloud Generation**:
   ```bash
   python prepare_data_for_registration.py --input camera_parameters.json
   ```

3. **Point Cloud Alignment**:
   ```bash
   python point_cloud_alignment.py --reference model.ply --target captured_cloud.ply
   ```

### Test Data Generation

You can generate test data for alignment experiments:

```bash
python generate_test_data.py --output_dir test_data
```

## Troubleshooting

- **Visualization Issues**: If point clouds are not visible, increase point size or disable culling
- **Alignment Problems**: Try adjusting the voxel size (smaller = more precise but slower)
- **Performance Issues**: Reduce number of points with larger voxel size for faster processing

## License

This project is open-source under the MIT License. 