#!/usr/bin/env python3
"""
Standalone Point Cloud Visualizer

This script loads and visualizes a point cloud file with enhanced visualization settings.
It's useful for checking point clouds when the registration pipeline visualization fails.

Usage:
    python visualize_point_cloud.py --file path/to/point_cloud.ply

Author: AI Assistant
Date: 2025-03-09
"""

import os
import numpy as np
import open3d as o3d
import argparse
import copy

def visualize_point_cloud(pcd_path):
    """
    Load and visualize a point cloud with enhanced settings.
    
    Args:
        pcd_path (str): Path to the point cloud file
    """
    print(f"Loading point cloud from {pcd_path}")
    pcd = o3d.io.read_point_cloud(pcd_path)
    
    # Print point cloud info
    print(f"Point cloud has {len(pcd.points)} points")
    if len(pcd.points) == 0:
        print("Error: Point cloud is empty")
        return
    
    # Get bounds
    points = np.asarray(pcd.points)
    print(f"Point cloud bounds: min {points.min(axis=0)}, max {points.max(axis=0)}")
    
    # Add random colors if not present
    if len(np.asarray(pcd.colors)) == 0:
        print("Point cloud has no colors, adding random colors")
        # Color by XYZ coordinates (normalized)
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        range_coords = max_coords - min_coords
        normalized = (points - min_coords) / range_coords
        pcd.colors = o3d.utility.Vector3dVector(normalized)
    
    # Create visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Point Cloud Viewer", width=1280, height=720)
    
    # Add point cloud
    vis.add_geometry(pcd)
    
    # Add coordinate frame
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    vis.add_geometry(coord_frame)
    
    # Configure rendering options
    opt = vis.get_render_option()
    opt.point_size = 10.0  # Make points very large for visibility
    opt.background_color = np.array([0.1, 0.1, 0.1])  # Dark background
    opt.show_coordinate_frame = True
    
    # Disable culling
    opt.light_on = True
    opt.mesh_show_back_face = True
    opt.mesh_show_wireframe = True
    opt.point_show_normal = False
    # Explicitly turn off culling
    if hasattr(opt, 'point_culling'):
        opt.point_culling = False
    if hasattr(opt, 'mesh_culling'):
        opt.mesh_culling = False
    
    # Get camera control
    ctrl = vis.get_view_control()
    
    # Create a view that shows all points
    vis.reset_view_point(True)
    
    # Set multiple views for the user to try
    print("\nMultiple view options:")
    print("Front view: Press '1'")
    print("Top view: Press '2'")
    print("Side view: Press '3'")
    print("Rotate view: Left-click + drag")
    print("Pan view: Shift + left-click + drag")
    print("Zoom: Mouse wheel or Ctrl + left-click + drag")
    print("Press 'h' for more options")
    print("Press 'q' to exit")
    
    # Run the visualization
    vis.run()
    vis.destroy_window()

def main():
    parser = argparse.ArgumentParser(description="Visualize a point cloud file")
    parser.add_argument("--file", type=str, required=True, help="Path to point cloud file (.ply or .pcd)")
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.file):
        print(f"Error: File {args.file} does not exist")
        return
    
    # Visualize the point cloud
    visualize_point_cloud(args.file)

if __name__ == "__main__":
    main() 