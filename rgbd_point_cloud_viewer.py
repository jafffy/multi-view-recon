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

def display_depth_image(depth_path):
    """Display the depth image using matplotlib"""
    # Check if the path is a .npy file or an image
    if depth_path.endswith('.npy'):
        # Load depth data from .npy file
        depth_data = np.load(depth_path)
    else:
        # Load depth image
        depth_img = Image.open(depth_path)
        depth_data = np.array(depth_img)
    
    # Normalize depth for visualization
    depth_norm = (depth_data - np.min(depth_data)) / (np.max(depth_data) - np.min(depth_data))
    
    # Display the depth image
    plt.figure(figsize=(10, 6))
    plt.imshow(depth_norm, cmap='viridis')
    plt.colorbar(label='Normalized Depth')
    plt.title(f'Depth Image: {os.path.basename(depth_path)}')
    plt.show()
    
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
    
    # Display the color image
    plt.figure(figsize=(10, 6))
    plt.imshow(color_data)
    plt.title(f'RGB Image: {os.path.basename(image_path)}')
    plt.show()
    
    return color_data

def create_colored_point_cloud(depth_data, color_data, intrinsic_matrix, extrinsic_matrix, 
                             depth_min, depth_max, depth_scale):
    """Convert depth and color image to colored 3D point cloud using camera parameters"""
    print("Creating colored point cloud...")
    
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
                               depth_min, depth_max, depth_scale):
    """Create a point cloud using only depth data (for comparison)"""
    print("Creating depth-only point cloud...")
    
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

def visualize_point_cloud(pcd):
    """Visualize the 3D point cloud"""
    # Add coordinate frame
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.5, origin=[0, 0, 0])
    
    # Increase point size for better visibility
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="RGBD Point Cloud", width=1280, height=720)
    
    # Add geometries
    vis.add_geometry(pcd)
    vis.add_geometry(coord_frame)
    
    # Configure rendering options
    opt = vis.get_render_option()
    opt.point_size = 3.0  # Increase point size further
    opt.background_color = np.array([0.1, 0.1, 0.1])  # Dark background
    opt.show_coordinate_frame = True
    
    # Disable face culling
    opt.light_on = True
    opt.mesh_show_back_face = True  # Show back faces
    
    # Additional rendering options
    opt.point_show_normal = False
    opt.mesh_shade_option = o3d.visualization.MeshShadeOption.Color
    
    # Get camera control
    ctrl = vis.get_view_control()
    
    # Try multiple view perspectives
    print("Starting with front view. You can rotate to find the best perspective.")
    ctrl.set_front([0, 0, -1])  # Looking toward negative z-axis
    ctrl.set_up([0, -1, 0])     # Up is negative y-axis
    ctrl.set_zoom(0.7)          # Zoom out a bit more
    
    print("Visualizing point cloud. Press 'q' to exit the viewer.")
    print("Rendering controls: left-click + drag to rotate, mouse wheel to zoom")
    print("Press 'h' for help on more controls")
    print("Press '1', '2', '3', '4' to switch between preset views")
    
    # Run the visualization
    vis.run()
    vis.destroy_window()

def main():
    # Path to camera parameters
    camera_params_path = "virtual_capture_20250307_171627/camera_parameters.json"
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Visualize RGB-D data as point cloud")
    parser.add_argument('--view', type=int, default=0, help="View index to render (default: 0)")
    parser.add_argument('--no-display-images', action='store_true', help="Skip displaying 2D images")
    parser.add_argument('--multi-view', action='store_true', help="Visualize multiple point clouds together")
    parser.add_argument('--view-range', type=str, default='0-0', help="Range of views to visualize (e.g., '0-5')")
    parser.add_argument('--depth-only', action='store_true', help="Create point cloud using only depth data (no color)")
    args = parser.parse_args()
    
    # Load camera parameters
    print(f"Loading camera parameters from: {camera_params_path}")
    camera_params = load_camera_params(camera_params_path)
    
    # Parse view range if multi-view is enabled
    if args.multi_view:
        try:
            start_view, end_view = map(int, args.view_range.split('-'))
            view_indices = range(start_view, end_view + 1)
        except ValueError:
            print(f"Error: Invalid view range format. Expected 'start-end', got '{args.view_range}'")
            return
        
        # Validate view range
        max_view_idx = len(camera_params["views"]) - 1
        if start_view < 0 or end_view > max_view_idx:
            print(f"Error: View range {args.view_range} out of bounds. Valid range: 0-{max_view_idx}")
            return
        
        print(f"Visualizing views {start_view} to {end_view}")
    else:
        # Single view mode
        view_idx = args.view
        if view_idx < 0 or view_idx >= len(camera_params["views"]):
            print(f"Error: View index {view_idx} out of range. Valid range: 0-{len(camera_params['views'])-1}")
            return
        view_indices = [view_idx]
    
    # Generate point clouds for all views
    all_point_clouds = []
    for view_idx in view_indices:
        view = camera_params["views"][view_idx]
        print(f"\nProcessing view {view_idx}")
        
        # Get paths to depth and image files
        depth_path = view["depth_path"]
        depth_vis_path = view["depth_vis_path"]
        image_path = view["image_path"]
        
        # Display depth image (using visualization PNG for display)
        if not args.no_display_images and not args.multi_view:
            print(f"Displaying depth visualization for view {view_idx}")
            display_depth_image(depth_vis_path)
            
            # Display color image
            print(f"Displaying color image for view {view_idx}")
            color_data = load_color_image(image_path)
        else:
            # Just load the color image without displaying
            color_data = load_color_image(image_path)
        
        # For 3D point cloud, we need to use the actual depth data
        print(f"Loading depth data from {depth_path}")
        depth_data_3d = np.load(depth_path)
        
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
                depth_data_3d, intrinsic_matrix, extrinsic_matrix, 
                depth_min, depth_max, depth_scale
            )
        else:
            # Colored point cloud
            pcd = create_colored_point_cloud(
                depth_data_3d, color_data, intrinsic_matrix, extrinsic_matrix, 
                depth_min, depth_max, depth_scale
            )
        
        all_point_clouds.append(pcd)
    
    # Visualize point cloud(s)
    if args.multi_view:
        # Visualize multiple point clouds together
        print(f"Visualizing {len(all_point_clouds)} point clouds together")
        
        # Add coordinate frame
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
        
        # Create visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Multi-View RGBD Point Cloud", width=1280, height=720)
        
        # Add geometries
        for pcd in all_point_clouds:
            vis.add_geometry(pcd)
        vis.add_geometry(coord_frame)
        
        # Configure rendering options
        opt = vis.get_render_option()
        opt.point_size = 2.0
        opt.background_color = np.array([0.1, 0.1, 0.1])
        opt.show_coordinate_frame = True
        
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
    else:
        # Visualize single point cloud
        visualize_point_cloud(all_point_clouds[0])

if __name__ == "__main__":
    main()
