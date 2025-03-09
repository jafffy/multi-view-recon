#!/usr/bin/env python3
"""
Prepare RGB-D Data for Registration

This script converts RGB-D data to point clouds for alignment:
1. Loads camera parameters and RGB-D data
2. Converts each RGB-D view to a point cloud
3. Saves point clouds for use with the registration pipeline

Author: AI Assistant
Date: 2025-03-09
"""

import os
import sys
import numpy as np
import json
import open3d as o3d
from PIL import Image
import argparse
import copy

def load_camera_params(json_path):
    """Load camera parameters from JSON file"""
    with open(json_path, 'r') as f:
        params = json.load(f)
    return params

def create_point_cloud_from_rgbd(depth_data, color_data, intrinsic_matrix, extrinsic_matrix, 
                               depth_min, depth_max, depth_scale):
    """Convert depth and color image to a 3D point cloud"""
    # Create Open3D intrinsic object
    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    height, width = depth_data.shape
    intrinsic.set_intrinsics(
        width, height,
        intrinsic_matrix[0][0],  # fx
        intrinsic_matrix[1][1],  # fy
        intrinsic_matrix[0][2],  # cx
        intrinsic_matrix[1][2]   # cy
    )
    
    # Make sure color image has the same dimensions as depth
    if color_data.shape[0] != height or color_data.shape[1] != width:
        print(f"Resizing color image from {color_data.shape[:2]} to {(height, width)}")
        color_image = Image.fromarray(color_data)
        color_image = color_image.resize((width, height), Image.LANCZOS)
        color_data = np.array(color_image)
    
    # Create Open3D depth image
    depth_o3d = o3d.geometry.Image(depth_data.astype(np.float32))
    
    # Create Open3D color image
    color_o3d = o3d.geometry.Image(color_data.astype(np.uint8))
    
    # Create RGBD image
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d, depth_o3d,
        depth_scale=1000.0,  # Convert to meters
        depth_trunc=depth_max/1000.0,  # Max depth in meters
        convert_rgb_to_intensity=False
    )
    
    # Create point cloud from RGBD
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image, intrinsic, 
        extrinsic=np.array(extrinsic_matrix).astype(np.float64)
    )
    
    # Remove points with zero or invalid depth
    points = np.asarray(pcd.points)
    valid_points = ~np.isnan(points).any(axis=1)
    pcd = pcd.select_by_index(np.where(valid_points)[0])
    
    # Estimate normals if they don't exist
    if len(np.asarray(pcd.normals)) == 0:
        pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        pcd.orient_normals_towards_camera_location([0, 0, 0])
    
    print(f"Created point cloud with {len(pcd.points)} points")
    
    return pcd

def process_view(camera_params, view_idx, output_dir, voxel_size=0.01, visualize=False):
    """Process a single RGB-D view and convert to point cloud"""
    # Get view information
    view = camera_params["views"][view_idx]
    
    # Get paths to depth and image files
    depth_path = view["depth_path"]
    image_path = view["image_path"]
    
    # Load color image
    print(f"Loading color image from {image_path}")
    color_image = Image.open(image_path)
    color_data = np.array(color_image)
    
    # Load depth data
    print(f"Loading depth data from {depth_path}")
    depth_data = np.load(depth_path)
    
    # Get camera parameters
    intrinsic_matrix = camera_params["intrinsic_matrix"]
    extrinsic_matrix = view["extrinsic_matrix"]
    depth_min = view["depth_min"]
    depth_max = view["depth_max"]
    depth_scale = view["depth_scale"]
    
    # Create point cloud
    pcd = create_point_cloud_from_rgbd(
        depth_data, color_data, intrinsic_matrix, extrinsic_matrix, 
        depth_min, depth_max, depth_scale
    )
    
    # Downsample point cloud
    if voxel_size > 0:
        print(f"Downsampling point cloud with voxel size {voxel_size}")
        pcd = pcd.voxel_down_sample(voxel_size)
        print(f"Downsampled to {len(pcd.points)} points")
    
    # Visualize point cloud
    if visualize:
        print("Visualizing point cloud...")
        o3d.visualization.draw_geometries([pcd], 
                                        window_name=f"Point Cloud View {view_idx}",
                                        width=1280, height=720)
    
    # Save point cloud
    output_path = os.path.join(output_dir, f"point_cloud_{view_idx:03d}.ply")
    print(f"Saving point cloud to {output_path}")
    o3d.io.write_point_cloud(output_path, pcd)
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Prepare RGB-D data for point cloud registration")
    parser.add_argument("--input", type=str, required=True,
                      help="Path to the camera_parameters.json file")
    parser.add_argument("--output", type=str, default="point_clouds",
                      help="Output directory for point clouds")
    parser.add_argument("--views", type=str, default="all",
                      help="Comma-separated list of view indices to process, or 'all'")
    parser.add_argument("--voxel_size", type=float, default=0.01,
                      help="Voxel size for downsampling (in meters)")
    parser.add_argument("--visualize", action="store_true",
                      help="Visualize each point cloud after creation")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Load camera parameters
    print(f"Loading camera parameters from {args.input}")
    camera_params = load_camera_params(args.input)
    
    # Determine which views to process
    all_views = range(len(camera_params["views"]))
    if args.views.lower() == "all":
        view_indices = all_views
    else:
        try:
            view_indices = [int(idx) for idx in args.views.split(",")]
            # Validate view indices
            invalid_indices = [idx for idx in view_indices if idx not in all_views]
            if invalid_indices:
                print(f"Warning: View indices {invalid_indices} are out of range and will be skipped")
                view_indices = [idx for idx in view_indices if idx in all_views]
        except ValueError:
            print(f"Error: Invalid view indices format. Expected 'all' or comma-separated integers.")
            return
    
    # Process selected views
    point_cloud_files = []
    for view_idx in view_indices:
        print(f"\nProcessing view {view_idx}")
        output_path = process_view(
            camera_params, view_idx, args.output, args.voxel_size, args.visualize)
        point_cloud_files.append(output_path)
    
    print(f"\nSuccessfully prepared {len(point_cloud_files)} point clouds for registration")
    print(f"To register these point clouds, run:")
    print(f"python point_cloud_registration.py --input_dir {args.output}")

if __name__ == "__main__":
    main() 