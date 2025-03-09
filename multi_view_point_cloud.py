#!/usr/bin/env python3
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from PIL import Image
import argparse

def load_camera_params(json_path):
    """Load camera parameters from JSON file"""
    with open(json_path, 'r') as f:
        params = json.load(f)
    return params

def load_depth_image(depth_path):
    """Load depth data from .npy file"""
    print(f"Loading depth data from: {depth_path}")
    
    if not os.path.exists(depth_path):
        print(f"Error: Depth file {depth_path} not found")
        return None
    
    # Load depth data from .npy file
    depth_data = np.load(depth_path)
    return depth_data

def load_color_image(image_path):
    """Load and return the RGB image"""
    print(f"Loading color image from: {image_path}")
    
    # Check if the file exists
    if not os.path.exists(image_path):
        print(f"Error: Image file {image_path} not found")
        # Return a blank image as fallback
        return np.ones((720, 1280, 3), dtype=np.uint8) * 128
    
    color_image = Image.open(image_path)
    color_data = np.array(color_image)
    return color_data

def create_colored_point_cloud(depth_data, color_data, intrinsic_matrix, extrinsic_matrix, 
                             depth_min, depth_max, depth_scale, view_id=None):
    """Convert depth and color image to colored 3D point cloud using camera parameters"""
    print(f"Creating colored point cloud for view {view_id}...")
    
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
    
    # Print shapes for debugging
    print(f"Depth data shape: {depth_data.shape}")
    print(f"Color data shape: {color_data.shape}")
    print(f"Depth range: {np.min(depth_data)} to {np.max(depth_data)}")
    
    # Make sure color image has the same dimensions as depth
    if color_data.shape[0] != height or color_data.shape[1] != width:
        print(f"Resizing color image from {color_data.shape[:2]} to {(height, width)}")
        color_image = Image.fromarray(color_data)
        color_image = color_image.resize((width, height), Image.LANCZOS)
        color_data = np.array(color_image)
    
    # Method 1: Create a point cloud first, then add colors
    # Create Open3D depth image
    depth_o3d = o3d.geometry.Image(depth_data.astype(np.float32))
    
    # Create point cloud from depth
    pcd = o3d.geometry.PointCloud.create_from_depth_image(
        depth_o3d, intrinsic, 
        extrinsic=np.array(extrinsic_matrix).astype(np.float64),
        depth_scale=1000.0,  # Convert to meters
        depth_trunc=depth_max/1000.0,  # Max depth in meters
        stride=1
    )
    
    # Remove points with zero or invalid depth
    points = np.asarray(pcd.points)
    valid_points = ~np.isnan(points).any(axis=1)
    print(f"Removing {np.sum(~valid_points)} invalid points")
    pcd = pcd.select_by_index(np.where(valid_points)[0])
    
    # Now add color to the point cloud
    # Convert the depth points back to image coordinates
    # to find corresponding colors
    
    # Convert back to Open3D for easy manipulation
    pts = np.asarray(pcd.points)
    
    # Create colors array
    colors = []
    
    # Get camera matrix 
    fx = intrinsic_matrix[0][0]
    fy = intrinsic_matrix[1][1]
    cx = intrinsic_matrix[0][2]
    cy = intrinsic_matrix[1][2]
    
    # Convert extrinsic to camera pose
    extrinsic_np = np.array(extrinsic_matrix).astype(np.float64)
    cam_to_world = np.linalg.inv(extrinsic_np)
    
    print(f"Extrinsic (world to camera):")
    print(extrinsic_np)
    print(f"Camera to world:")
    print(cam_to_world)
    
    # For each point, project back to get color
    for i, pt in enumerate(pts):
        # Transform point to camera coordinates
        pt_cam = extrinsic_np @ np.append(pt, 1.0)
        if pt_cam[2] <= 0:  # Point is behind camera
            colors.append([0.5, 0.5, 0.5])  # Gray for invalid points
            continue
            
        # Project to image coordinates
        x = int((fx * pt_cam[0] / pt_cam[2]) + cx)
        y = int((fy * pt_cam[1] / pt_cam[2]) + cy)
        
        # Check if point is in image bounds
        if 0 <= x < width and 0 <= y < height:
            colors.append(color_data[y, x] / 255.0)  # Normalize color to [0, 1]
        else:
            colors.append([0.5, 0.5, 0.5])  # Gray for out-of-bounds points
    
    # Add colors to the point cloud
    pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
    
    # Verify point cloud bounds
    if len(pcd.points) > 0:
        points = np.asarray(pcd.points)
        print(f"Point cloud bounds: min {points.min(axis=0)}, max {points.max(axis=0)}")
    
    print(f"Point cloud has {len(pcd.points)} points with color")
    
    return pcd

def create_depth_only_point_cloud(depth_data, intrinsic_matrix, extrinsic_matrix, 
                               depth_min, depth_max, depth_scale, view_id=None):
    """Create a point cloud using only depth data (for comparison)"""
    print(f"Creating depth-only point cloud for view {view_id}...")
    
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
    
    # Print depth info
    print(f"Depth data shape: {depth_data.shape}")
    print(f"Depth range: {np.min(depth_data)} to {np.max(depth_data)}")
    
    # Create Open3D depth image
    depth_o3d = o3d.geometry.Image(depth_data.astype(np.float32))
    
    # Create point cloud from depth
    pcd = o3d.geometry.PointCloud.create_from_depth_image(
        depth_o3d, intrinsic, 
        extrinsic=np.array(extrinsic_matrix).astype(np.float64),
        depth_scale=1000.0,  # Convert to meters
        depth_trunc=depth_max/1000.0,  # Max depth in meters
        stride=1
    )
    
    # Remove points with zero or invalid depth
    points = np.asarray(pcd.points)
    valid_points = ~np.isnan(points).any(axis=1)
    print(f"Removing {np.sum(~valid_points)} invalid points")
    pcd = pcd.select_by_index(np.where(valid_points)[0])
    
    # Use uniform color for better visibility
    pcd.paint_uniform_color([0.7, 0.7, 0.7])  # Light gray
    
    # Verify point cloud bounds
    if len(pcd.points) > 0:
        points = np.asarray(pcd.points)
        print(f"Point cloud bounds: min {points.min(axis=0)}, max {points.max(axis=0)}")
    
    print(f"Point cloud has {len(pcd.points)} points")
    
    return pcd

def visualize_point_clouds(point_clouds, view_indices=None):
    """Visualize multiple 3D point clouds"""
    # Add coordinate frame
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.5, origin=[0, 0, 0])
    
    # Create visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Multi-View RGBD Point Cloud", width=1280, height=720)
    
    # Add geometries
    for i, pcd in enumerate(point_clouds):
        # Optionally apply a different color to each point cloud for better visibility
        if view_indices:
            print(f"Adding point cloud for view {view_indices[i]} with {len(pcd.points)} points")
        else:
            print(f"Adding point cloud {i} with {len(pcd.points)} points")
        vis.add_geometry(pcd)
    
    vis.add_geometry(coord_frame)
    
    # Configure rendering options
    opt = vis.get_render_option()
    opt.point_size = 2.0
    opt.background_color = np.array([0.1, 0.1, 0.1])
    opt.show_coordinate_frame = True
    
    # Disable face culling
    opt.light_on = True
    opt.mesh_show_back_face = True  # Show back faces
    
    # Additional rendering options
    opt.point_show_normal = False
    opt.mesh_shade_option = o3d.visualization.MeshShadeOption.Color
    
    # Set view
    ctrl = vis.get_view_control()
    ctrl.set_front([0, 0, -1])
    ctrl.set_up([0, -1, 0])
    ctrl.set_zoom(0.8)
    
    print("Visualizing multi-view point cloud. Press 'q' to exit.")
    print("Rendering controls: left-click + drag to rotate, mouse wheel to zoom")
    print("Press 'h' for help on more controls")
    
    # Run visualization
    vis.run()
    vis.destroy_window()

def combine_point_clouds(point_clouds):
    """Combine multiple point clouds into a single point cloud"""
    print(f"Combining {len(point_clouds)} point clouds...")
    
    # Create a new point cloud
    combined_pcd = o3d.geometry.PointCloud()
    
    # Add points from all point clouds
    for pcd in point_clouds:
        combined_pcd += pcd
    
    # Optional: downsample the combined point cloud to reduce noise and size
    print(f"Combined point cloud has {len(combined_pcd.points)} points")
    print("Downsampling point cloud...")
    combined_pcd = combined_pcd.voxel_down_sample(voxel_size=0.005)  # 5mm voxel size
    print(f"After downsampling: {len(combined_pcd.points)} points")
    
    # Optional: remove statistical outliers
    print("Removing outliers...")
    combined_pcd, _ = combined_pcd.remove_statistical_outlier(
        nb_neighbors=20, std_ratio=2.0)
    print(f"After outlier removal: {len(combined_pcd.points)} points")
    
    return combined_pcd

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Visualize multiple RGB-D views as a combined point cloud")
    parser.add_argument('--params', type=str, default="virtual_capture_20250307_161550/camera_parameters.json", 
                        help="Path to camera parameters JSON file")
    parser.add_argument('--views', type=str, default='all', 
                        help="Comma-separated list of view indices to visualize (e.g., '0,1,2') or 'all'")
    parser.add_argument('--combine', action='store_true', 
                        help="Combine all point clouds into a single one (default: show separate)")
    parser.add_argument('--save', type=str, default='', 
                        help="Save the combined point cloud to the specified file path (e.g., 'output.ply')")
    parser.add_argument('--depth-only', action='store_true',
                        help="Create point cloud using only depth data (no color)")
    args = parser.parse_args()
    
    # Load camera parameters
    print(f"Loading camera parameters from: {args.params}")
    camera_params = load_camera_params(args.params)
    
    # Determine which views to process
    if args.views.lower() == 'all':
        view_indices = list(range(len(camera_params["views"])))
    else:
        try:
            view_indices = [int(idx) for idx in args.views.split(',')]
        except ValueError:
            print(f"Error: Invalid view indices format. Expected comma-separated integers.")
            return
    
    # Validate view indices
    max_view_idx = len(camera_params["views"]) - 1
    valid_indices = []
    for idx in view_indices:
        if 0 <= idx <= max_view_idx:
            valid_indices.append(idx)
        else:
            print(f"Warning: View index {idx} out of range. Valid range: 0-{max_view_idx}")
    
    if not valid_indices:
        print("Error: No valid view indices specified.")
        return
    
    print(f"Processing {len(valid_indices)} views: {valid_indices}")
    
    # Generate point clouds for selected views
    point_clouds = []
    
    for view_idx in valid_indices:
        view = camera_params["views"][view_idx]
        print(f"\nProcessing view {view_idx}")
        
        # Get paths to depth and image files
        depth_path = view["depth_path"]
        image_path = view["image_path"]
        
        # Load depth and color data
        depth_data = load_depth_image(depth_path)
        if depth_data is None:
            print(f"Skipping view {view_idx} due to missing depth data")
            continue
        
        # Only load color if we're not using depth-only mode
        if not args.depth_only:
            color_data = load_color_image(image_path)
        else:
            # Dummy color data for depth-only mode
            color_data = None
        
        # Get camera parameters
        intrinsic_matrix = camera_params["intrinsic_matrix"]
        extrinsic_matrix = view["extrinsic_matrix"]
        depth_min = view["depth_min"]
        depth_max = view["depth_max"]
        depth_scale = view["depth_scale"]
        
        # Create point cloud (depth-only or colored)
        if args.depth_only:
            # Depth-only point cloud
            pcd = create_depth_only_point_cloud(
                depth_data, intrinsic_matrix, extrinsic_matrix, 
                depth_min, depth_max, depth_scale, view_id=view_idx
            )
        else:
            # Colored point cloud
            pcd = create_colored_point_cloud(
                depth_data, color_data, intrinsic_matrix, extrinsic_matrix, 
                depth_min, depth_max, depth_scale, view_id=view_idx
            )
        
        point_clouds.append(pcd)
    
    # Process the point clouds
    if not point_clouds:
        print("Error: No valid point clouds created.")
        return
    
    if args.combine:
        # Combine all point clouds into one
        combined_pcd = combine_point_clouds(point_clouds)
        
        # Save the combined point cloud if requested
        if args.save:
            print(f"Saving combined point cloud to {args.save}")
            o3d.io.write_point_cloud(args.save, combined_pcd)
            print("Point cloud saved successfully.")
        
        # Visualize the combined point cloud
        visualize_point_clouds([combined_pcd])
    else:
        # Visualize all point clouds separately
        visualize_point_clouds(point_clouds, valid_indices)

if __name__ == "__main__":
    main() 