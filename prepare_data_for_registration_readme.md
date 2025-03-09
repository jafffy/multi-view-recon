# RGB-D to Point Cloud Conversion Utility

This utility converts RGB-D data to point clouds for use with the point cloud registration pipeline.

## Usage

```bash
python prepare_data_for_registration.py --input /path/to/camera_parameters.json --output /path/to/output_dir
```

## Command Line Arguments

- `--input`: Path to the camera parameters JSON file (required)
- `--output`: Output directory for point clouds (default: "point_clouds")
- `--views`: Comma-separated list of view indices to process, or 'all' (default: "all")
- `--voxel_size`: Voxel size for downsampling in meters (default: 0.01)
- `--visualize`: Visualize each point cloud after creation (optional)

## Examples

### Process all views

```bash
python prepare_data_for_registration.py --input virtual_capture_20250307_171627/camera_parameters.json
```

### Process specific views

```bash
python prepare_data_for_registration.py --input virtual_capture_20250307_171627/camera_parameters.json --views 0,1,2
```

### Change output directory and voxel size

```bash
python prepare_data_for_registration.py --input virtual_capture_20250307_171627/camera_parameters.json --output my_point_clouds --voxel_size 0.005
```

### Enable visualization

```bash
python prepare_data_for_registration.py --input virtual_capture_20250307_171627/camera_parameters.json --visualize
```

## Integration with the Pipeline

This utility is automatically called by the `rgbd_capture_to_alignment.py` script as part of the complete pipeline. You typically don't need to run it manually unless you want to convert specific views or use custom parameters. 