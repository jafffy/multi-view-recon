#!/usr/bin/env python3
"""
Generate Test Data for Point Cloud Alignment

This script generates test data for the point cloud alignment pipeline by:
1. Creating a synthetic point cloud (or using an existing one)
2. Creating a transformed version of the same point cloud with added noise
3. Saving both point clouds for testing

Author: AI Assistant
Date: 2025-03-09
"""

import os
import numpy as np
import open3d as o3d
import argparse
import copy

def create_synthetic_point_cloud(num_points=1000):
    """
    Create a synthetic point cloud for testing.
    
    Args:
        num_points (int): Number of points in the synthetic cloud
        
    Returns:
        o3d.geometry.PointCloud: Synthetic point cloud
    """
    print(f"Generating synthetic point cloud with {num_points} points")
    
    # Create a synthetic point cloud (a simple sphere)
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=20)
    pcd = sphere.sample_points_uniformly(number_of_points=num_points)
    
    # Assign random colors
    colors = np.random.uniform(0, 1, size=(num_points, 3))
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Make sure normals are computed
    pcd.estimate_normals()
    
    return pcd

def create_transformed_point_cloud(pcd, translation=[0.2, 0.1, 0.05], rotation_degrees=15, noise_level=0.02):
    """
    Create a transformed version of a point cloud with added noise.
    
    Args:
        pcd (o3d.geometry.PointCloud): Input point cloud
        translation (list): Translation vector [x, y, z]
        rotation_degrees (float): Rotation angle in degrees
        noise_level (float): Level of Gaussian noise to add
        
    Returns:
        o3d.geometry.PointCloud: Transformed point cloud with noise
    """
    print(f"Creating transformed point cloud with:")
    print(f"  - Translation: {translation}")
    print(f"  - Rotation: {rotation_degrees} degrees")
    print(f"  - Noise level: {noise_level}")
    
    # Create a copy of the point cloud
    transformed_pcd = copy.deepcopy(pcd)
    
    # Create a rotation matrix (around y-axis)
    rotation_radians = np.radians(rotation_degrees)
    cos_angle = np.cos(rotation_radians)
    sin_angle = np.sin(rotation_radians)
    rotation_matrix = np.array([
        [cos_angle, 0, sin_angle, 0],
        [0, 1, 0, 0],
        [-sin_angle, 0, cos_angle, 0],
        [0, 0, 0, 1]
    ])
    
    # Create a translation matrix
    translation_matrix = np.eye(4)
    translation_matrix[:3, 3] = translation
    
    # Combine rotation and translation
    transformation = np.dot(translation_matrix, rotation_matrix)
    
    # Apply transformation
    transformed_pcd.transform(transformation)
    
    # Add Gaussian noise to points
    points = np.asarray(transformed_pcd.points)
    noise = np.random.normal(0, noise_level, size=points.shape)
    points_noisy = points + noise
    transformed_pcd.points = o3d.utility.Vector3dVector(points_noisy)
    
    # Recompute normals
    transformed_pcd.estimate_normals()
    
    return transformed_pcd, transformation

def visualize_point_clouds(reference_pcd, target_pcd):
    """
    Visualize reference and target point clouds side by side.
    
    Args:
        reference_pcd (o3d.geometry.PointCloud): Reference point cloud
        target_pcd (o3d.geometry.PointCloud): Target point cloud
    """
    print("Visualizing reference (blue) and target (orange) point clouds")
    
    # Create copies for visualization
    reference_vis = copy.deepcopy(reference_pcd)
    target_vis = copy.deepcopy(target_pcd)
    
    # Assign colors
    reference_vis.paint_uniform_color([0, 0.651, 0.929])  # Blue
    target_vis.paint_uniform_color([1, 0.706, 0])  # Orange
    
    # Visualize point clouds
    o3d.visualization.draw_geometries([reference_vis, target_vis],
                                     window_name="Test Data Point Clouds",
                                     width=1280, height=720)

def main():
    parser = argparse.ArgumentParser(description="Generate test data for point cloud alignment")
    parser.add_argument("--input", type=str, default=None,
                      help="Optional input PLY model to use as reference (default: generate synthetic)")
    parser.add_argument("--output_dir", type=str, default="test_data",
                      help="Output directory (default: test_data)")
    parser.add_argument("--num_points", type=int, default=5000,
                      help="Number of points in synthetic point cloud (default: 5000)")
    parser.add_argument("--translation", type=float, nargs=3, default=[0.2, 0.1, 0.05],
                      help="Translation vector [x, y, z] (default: [0.2, 0.1, 0.05])")
    parser.add_argument("--rotation", type=float, default=15.0,
                      help="Rotation angle in degrees (default: 15.0)")
    parser.add_argument("--noise", type=float, default=0.02,
                      help="Noise level (default: 0.02)")
    parser.add_argument("--no_visualization", action="store_true",
                      help="Disable visualization")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get or create reference point cloud
    if args.input is not None and os.path.exists(args.input):
        print(f"Loading reference point cloud from {args.input}")
        reference_pcd = o3d.io.read_point_cloud(args.input)
        if len(reference_pcd.points) == 0:
            print(f"Error: Input point cloud has no points. Generating synthetic point cloud instead.")
            reference_pcd = create_synthetic_point_cloud(args.num_points)
    else:
        reference_pcd = create_synthetic_point_cloud(args.num_points)
    
    # Create transformed point cloud with noise
    target_pcd, transformation = create_transformed_point_cloud(
        reference_pcd, args.translation, args.rotation, args.noise)
    
    # Save point clouds
    reference_path = os.path.join(args.output_dir, "model.ply")
    target_path = os.path.join(args.output_dir, "captured_cloud.ply")
    
    print(f"Saving reference point cloud to {reference_path}")
    o3d.io.write_point_cloud(reference_path, reference_pcd)
    
    print(f"Saving target point cloud to {target_path}")
    o3d.io.write_point_cloud(target_path, target_pcd)
    
    # Save transformation matrix for reference
    transformation_path = os.path.join(args.output_dir, "ground_truth_transformation.txt")
    print(f"Saving ground truth transformation to {transformation_path}")
    np.savetxt(transformation_path, transformation, fmt='%.6f')
    
    # Visualize point clouds
    if not args.no_visualization:
        visualize_point_clouds(reference_pcd, target_pcd)
    
    print("\nTest data generation complete!")
    print(f"Reference model: {reference_path}")
    print(f"Target point cloud: {target_path}")
    print(f"\nTo test alignment, run:")
    print(f"python point_cloud_alignment.py --reference {reference_path} --target {target_path}")

if __name__ == "__main__":
    main() 