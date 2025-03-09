#!/usr/bin/env python3
"""
RGB-D Capture to Point Cloud Alignment Pipeline

This script integrates the entire pipeline from RGB-D capture to point cloud alignment:
1. Capture RGB-D data from multiple viewpoints using virtual_camera_capture.py
2. Convert RGB-D data to point clouds using rgbd_point_cloud_viewer.py
3. Align and fuse the point clouds using point_cloud_alignment.py

Author: AI Assistant
Date: 2025-03-09
"""

import os
import sys
import argparse
import subprocess
import time
import json
from pathlib import Path

def run_command(cmd, description):
    """Run a command and print the output"""
    print(f"\n=== {description} ===")
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error running command: {result.stderr}")
        return False
    
    print(result.stdout)
    return True

def capture_rgb_d_data(ply_path, output_dir, num_views=10, resolution="1280x720"):
    """Capture RGB-D data using virtual_camera_capture.py"""
    cmd = [
        "python", "virtual_camera_capture.py",
        ply_path,
        "--output", output_dir,
        "--views", str(num_views),
        "--resolution", resolution
    ]
    
    return run_command(cmd, "Capturing RGB-D Data")

def convert_rgbd_to_point_clouds(camera_params_json, output_dir, voxel_size=0.01):
    """Convert RGB-D data to point clouds using a modified version of rgbd_point_cloud_viewer.py"""
    # First, let's check if the camera parameters file exists
    if not os.path.exists(camera_params_json):
        print(f"Error: Camera parameters file {camera_params_json} not found")
        return False
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load camera parameters
    with open(camera_params_json, 'r') as f:
        params = json.load(f)
    
    # Process each view
    success = True
    point_cloud_files = []
    
    for view in params["views"]:
        view_id = view["view_id"]
        output_path = os.path.join(output_dir, f"point_cloud_{view_id:03d}.ply")
        point_cloud_files.append(output_path)
        
        cmd = [
            "python", "prepare_data_for_registration.py",
            "--input", camera_params_json,
            "--output", output_dir,
            "--views", str(view_id),
            "--voxel_size", str(voxel_size)
        ]
        
        if not run_command(cmd, f"Converting View {view_id} to Point Cloud"):
            success = False
    
    return success, point_cloud_files

def register_and_align_point_clouds(point_cloud_files, reference_model, output_dir, voxel_size=0.05):
    """Register and align point clouds using point_cloud_alignment.py"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # If reference_model is not provided, use the first point cloud as reference
    if reference_model is None or not os.path.exists(reference_model):
        if len(point_cloud_files) > 0:
            reference_model = point_cloud_files[0]
            point_cloud_files = point_cloud_files[1:]
        else:
            print("Error: No point clouds available for alignment")
            return False
    
    # Align each point cloud with the reference model
    aligned_models = []
    
    for i, point_cloud_file in enumerate(point_cloud_files):
        output_path = os.path.join(output_dir, f"aligned_{i:03d}.ply")
        aligned_models.append(output_path)
        
        cmd = [
            "python", "point_cloud_alignment.py",
            "--reference", reference_model,
            "--target", point_cloud_file,
            "--output", output_path,
            "--voxel_size", str(voxel_size),
            "--no_visualization"
        ]
        
        if not run_command(cmd, f"Aligning Point Cloud {i}"):
            print(f"Warning: Failed to align point cloud {point_cloud_file}")
    
    # If we have multiple aligned models, fuse them together
    if len(aligned_models) > 1:
        # Use the first aligned model as reference and iteratively fuse with others
        reference = aligned_models[0]
        final_model = os.path.join(output_dir, "final_fused_model.ply")
        
        for i, aligned_model in enumerate(aligned_models[1:], 1):
            temp_output = os.path.join(output_dir, f"temp_fused_{i}.ply")
            
            cmd = [
                "python", "point_cloud_alignment.py",
                "--reference", reference,
                "--target", aligned_model,
                "--output", temp_output,
                "--voxel_size", str(voxel_size),
                "--no_visualization"
            ]
            
            if run_command(cmd, f"Fusing Aligned Model {i}"):
                reference = temp_output
            
        # Rename the final model
        os.rename(reference, final_model)
        print(f"\nFinal fused model saved to: {final_model}")
    
    return True

def main():
    parser = argparse.ArgumentParser(
        description="Complete pipeline from RGB-D capture to point cloud alignment")
    parser.add_argument("--ply_model", type=str, required=True,
                      help="Path to input PLY model for RGB-D capture")
    parser.add_argument("--reference_model", type=str, default=None,
                      help="Optional reference model for alignment (default: first captured point cloud)")
    parser.add_argument("--num_views", type=int, default=10,
                      help="Number of views to capture (default: 10)")
    parser.add_argument("--resolution", type=str, default="1280x720",
                      help="Resolution for RGB-D capture (default: 1280x720)")
    parser.add_argument("--voxel_size", type=float, default=0.01,
                      help="Voxel size for point cloud processing (default: 0.01)")
    parser.add_argument("--output_dir", type=str, default="pipeline_output",
                      help="Output directory (default: pipeline_output)")
    parser.add_argument("--skip_capture", action="store_true",
                      help="Skip RGB-D capture step (use existing data)")
    parser.add_argument("--skip_conversion", action="store_true",
                      help="Skip RGB-D to point cloud conversion step")
    
    args = parser.parse_args()
    
    # Create main output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Define subdirectories
    capture_dir = os.path.join(args.output_dir, "rgb_d_data")
    point_cloud_dir = os.path.join(args.output_dir, "point_clouds")
    alignment_dir = os.path.join(args.output_dir, "aligned_models")
    
    # Start timing
    start_time = time.time()
    
    # Step 1: Capture RGB-D data
    if not args.skip_capture:
        print("\n=== STEP 1: Capturing RGB-D Data ===")
        if not capture_rgb_d_data(args.ply_model, capture_dir, args.num_views, args.resolution):
            print("Error: Failed to capture RGB-D data")
            return
    else:
        print("\n=== STEP 1: Skipping RGB-D capture ===")
    
    # Camera parameters path
    camera_params_json = os.path.join(capture_dir, "camera_parameters.json")
    
    # Step 2: Convert RGB-D to point clouds
    if not args.skip_conversion:
        print("\n=== STEP 2: Converting RGB-D Data to Point Clouds ===")
        success, point_cloud_files = convert_rgbd_to_point_clouds(
            camera_params_json, point_cloud_dir, args.voxel_size)
        
        if not success:
            print("Warning: Some RGB-D data could not be converted to point clouds")
    else:
        print("\n=== STEP 2: Skipping RGB-D to point cloud conversion ===")
        # Find existing point cloud files
        point_cloud_files = sorted(list(Path(point_cloud_dir).glob("*.ply")))
        point_cloud_files = [str(p) for p in point_cloud_files]
        
        if not point_cloud_files:
            print("Error: No point cloud files found in", point_cloud_dir)
            return
    
    # Step 3: Register and align point clouds
    print("\n=== STEP 3: Registering and Aligning Point Clouds ===")
    if not register_and_align_point_clouds(
        point_cloud_files, args.reference_model, alignment_dir, args.voxel_size):
        print("Error: Failed to register and align point clouds")
        return
    
    # End timing
    end_time = time.time()
    print(f"\nComplete pipeline finished in {end_time - start_time:.2f} seconds")
    print(f"Output directory: {args.output_dir}")
    print("\nTo visualize the final model:")
    print(f"python visualize_point_cloud.py --file {os.path.join(alignment_dir, 'final_fused_model.ply')}")

if __name__ == "__main__":
    main() 