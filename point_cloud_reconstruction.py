#!/usr/bin/env python
import os
import json
import time
import numpy as np
import open3d as o3d
from tqdm import tqdm
import matplotlib.pyplot as plt
from point_cloud_utils import PointCloudProcessor

class PointCloudReconstructor:
    def __init__(self, capture_dir, output_dir=None):
        """
        Initialize the point cloud reconstructor
        
        Args:
            capture_dir (str): Directory containing the captured data (RGB-D images and camera parameters)
            output_dir (str, optional): Directory to save the reconstructed point cloud. If None, uses capture_dir.
        """
        self.capture_dir = os.path.normpath(capture_dir)
        self.output_dir = output_dir if output_dir else os.path.join(self.capture_dir, "reconstruction")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create subdirectories
        self.pointcloud_dir = os.path.join(self.output_dir, "pointclouds")
        os.makedirs(self.pointcloud_dir, exist_ok=True)
        
        # Load camera parameters from the capture directory
        self.params_path = os.path.join(self.capture_dir, "camera_parameters.json")
        if not os.path.exists(self.params_path):
            raise FileNotFoundError(f"Camera parameters file not found at {self.params_path}")
        
        self.load_camera_parameters()
        
        # Set file paths for images and depth maps
        self.image_dir = os.path.join(self.capture_dir, "images")
        self.depth_dir = os.path.join(self.capture_dir, "depth")
        
        # Initialize processor for point cloud operations
        self.processor = PointCloudProcessor(cache_dir=os.path.join(self.output_dir, "cache"))
    
    def load_camera_parameters(self):
        """
        Load camera parameters from the JSON file
        """
        print(f"Loading camera parameters from {self.params_path}")
        with open(self.params_path, 'r') as f:
            self.params = json.load(f)
        
        # Extract key parameters
        self.intrinsic_matrix = np.array(self.params['intrinsic_matrix'])
        self.distortion_coeffs = np.array(self.params['distortion_coeffs'])
        self.resolution = tuple(self.params['resolution'])
        self.views = self.params['views']
        
        # Debug: Print structure of views to understand format
        print("===== Debug: Structure of camera parameters =====")
        print(f"Keys in params: {self.params.keys()}")
        print(f"Type of views: {type(self.views)}")
        if isinstance(self.views, list) and len(self.views) > 0:
            print(f"Keys in first view: {self.views[0].keys() if isinstance(self.views[0], dict) else 'Not a dict'}")
            print(f"Sample view data: {self.views[0]}")
        print("===============================================")
        
        # Create Open3D camera intrinsic object
        width, height = self.resolution
        fx = self.intrinsic_matrix[0, 0]
        fy = self.intrinsic_matrix[1, 1]
        cx = self.intrinsic_matrix[0, 2]
        cy = self.intrinsic_matrix[1, 2]
        self.intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
        
        print(f"Loaded parameters for {len(self.views)} views")
        print(f"Intrinsic matrix:\n{self.intrinsic_matrix}")
    
    def depth_to_pointcloud(self, rgb_image, depth_map, extrinsic_matrix):
        """
        Convert a depth map to a point cloud using camera parameters
        
        Args:
            rgb_image (numpy.ndarray): RGB image (H, W, 3)
            depth_map (numpy.ndarray): Depth map (H, W)
            extrinsic_matrix (numpy.ndarray): 4x4 extrinsic matrix (camera pose)
            
        Returns:
            open3d.geometry.PointCloud: Point cloud with colors
        """
        # Add debug info for depth map
        depth_valid_pixels = np.count_nonzero(depth_map > 0)
        depth_min = np.min(depth_map[depth_map > 0]) if depth_valid_pixels > 0 else 0
        depth_max = np.max(depth_map) if depth_valid_pixels > 0 else 0
        depth_mean = np.mean(depth_map[depth_map > 0]) if depth_valid_pixels > 0 else 0
        
        print(f"Depth map stats: {depth_valid_pixels} valid pixels, range: [{depth_min:.4f}, {depth_max:.4f}], mean: {depth_mean:.4f}")
        
        # Check if we have sufficient valid pixels in the depth map
        if depth_valid_pixels < 100:
            print("Warning: Too few valid depth pixels")
            return o3d.geometry.PointCloud()
        
        # Convert depth map to meters if it's not already
        # Most depth maps are in millimeters or other units, but Open3D expects meters
        # Heuristic: if max depth > 1000, it's probably in mm, so convert to meters
        depth_scale = 1.0
        if depth_max > 1000:
            depth_scale = 0.001  # mm to m
            depth_map = depth_map * depth_scale
            print(f"Scaled depth by {depth_scale} (assuming millimeters to meters)")
        
        # Create RGBD image from color and depth
        rgb_o3d = o3d.geometry.Image(rgb_image.astype(np.uint8))
        depth_o3d = o3d.geometry.Image(depth_map.astype(np.float32))
        
        # Debug the shape of images
        print(f"RGB shape: {rgb_image.shape}, Depth shape: {depth_map.shape}")
        
        # Convert the extrinsic matrix from world-to-camera to camera-to-world
        # Open3D expects camera-to-world for creating point clouds
        camera_to_world = np.linalg.inv(extrinsic_matrix)
        
        # Print extrinsic matrix for debugging
        print(f"Original extrinsic matrix (world-to-camera):\n{extrinsic_matrix}")
        print(f"Inverted extrinsic matrix (camera-to-world):\n{camera_to_world}")
        
        # Use a more conservative depth_trunc value
        depth_trunc = depth_max * 1.2  # Allow values up to 120% of the maximum
        
        # Create the RGBD image
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_o3d, depth_o3d, 
            depth_scale=1.0,  # We have already scaled the depth if needed
            depth_trunc=depth_trunc,
            convert_rgb_to_intensity=False
        )
        
        # Convert RGBD image to point cloud using camera intrinsics and extrinsics
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd, self.intrinsic, camera_to_world
        )
        
        # Report point cloud stats
        print(f"Created point cloud with {len(pcd.points)} points")
        
        return pcd
    
    def load_view_data(self, view_id):
        """
        Load RGB image and depth map for a specific view
        
        Args:
            view_id (int): View ID
            
        Returns:
            tuple: (rgb_image, depth_map, extrinsic_matrix) or None if files not found
        """
        # Check if view_id is within range
        if view_id < 0 or view_id >= len(self.views):
            print(f"Warning: View {view_id} is out of range (0-{len(self.views)-1})")
            return None
            
        view_info = self.views[view_id]
        
        # Simple direct approach - construct paths directly from image dir and view ID
        rgb_path = os.path.join(self.capture_dir, "images", f"view_{view_id:03d}.jpg")
        depth_path = os.path.join(self.capture_dir, "depth", f"depth_{view_id:03d}.npy")
        
        # Check if the direct approach paths exist
        rgb_exists = os.path.exists(rgb_path)
        depth_exists = os.path.exists(depth_path)
        
        # If direct paths don't exist, try the paths from the JSON
        if not rgb_exists or not depth_exists:
            # Load RGB image - extract the path correctly
            if not rgb_exists:
                img_rel_path = view_info['image_path']
                # Handle paths that may already contain the capture directory
                capture_dir_name = os.path.basename(self.capture_dir)
                
                # Windows paths in JSON may use backslashes, normalize for splitting
                img_rel_path = img_rel_path.replace('\\', '/')
                
                if capture_dir_name in img_rel_path:
                    parts = img_rel_path.split('/')
                    try:
                        capture_index = parts.index(capture_dir_name)
                        img_rel_path = '/'.join(parts[capture_index+1:])
                    except ValueError:
                        # If capture_dir_name is not found as an exact match, just use the path as is
                        pass
                    
                rgb_path = os.path.join(self.capture_dir, img_rel_path)
            
            # Load depth map with the same path handling
            if not depth_exists:
                depth_rel_path = view_info['depth_path']
                
                # Windows paths in JSON may use backslashes, normalize for splitting
                depth_rel_path = depth_rel_path.replace('\\', '/')
                
                if capture_dir_name in depth_rel_path:
                    parts = depth_rel_path.split('/')
                    try:
                        capture_index = parts.index(capture_dir_name)
                        depth_rel_path = '/'.join(parts[capture_index+1:])
                    except ValueError:
                        # If capture_dir_name is not found as an exact match, just use the path as is
                        pass
                    
                depth_path = os.path.join(self.capture_dir, depth_rel_path)
        
        # Final check if files exist
        if not os.path.exists(rgb_path):
            print(f"Warning: RGB image not found at {rgb_path}")
            return None
            
        if not os.path.exists(depth_path):
            print(f"Warning: Depth map not found at {depth_path}")
            return None
        
        # Load the files
        rgb_image = plt.imread(rgb_path)
        
        # Convert RGB from 0-1 to 0-255 if needed
        if rgb_image.dtype == np.float32 or rgb_image.dtype == np.float64:
            rgb_image = (rgb_image * 255).astype(np.uint8)
        
        # Load depth map and convert to float32 if needed
        depth_map = np.load(depth_path)
        
        # Print debug info for depth map
        print(f"View {view_id} - Depth map shape: {depth_map.shape}, type: {depth_map.dtype}")
        print(f"View {view_id} - Depth range: [{np.min(depth_map)}, {np.max(depth_map)}], with {np.count_nonzero(depth_map > 0)} non-zero values")
        
        # Get extrinsic matrix
        extrinsic_matrix = np.array(view_info['extrinsic_matrix'])
        
        return rgb_image, depth_map, extrinsic_matrix
    
    def reconstruct_from_single_view(self, view_id):
        """
        Reconstruct a point cloud from a single view
        
        Args:
            view_id (int): View ID
            
        Returns:
            open3d.geometry.PointCloud: Point cloud from this view
        """
        view_data = self.load_view_data(view_id)
        if view_data is None:
            print(f"Skipping view {view_id} due to missing data")
            return None
        
        rgb_image, depth_map, extrinsic_matrix = view_data
        pcd = self.depth_to_pointcloud(rgb_image, depth_map, extrinsic_matrix)
        
        # Skip empty point clouds
        if len(pcd.points) == 0:
            print(f"Warning: View {view_id} produced an empty point cloud")
            return None
        
        # Save individual point cloud
        view_pcd_path = os.path.join(self.pointcloud_dir, f"view_{view_id:03d}.ply")
        o3d.io.write_point_cloud(view_pcd_path, pcd)
        print(f"Saved point cloud for view {view_id} with {len(pcd.points)} points")
        
        return pcd
    
    def reconstruct_from_all_views(self, voxel_size=0.01, vis_result=True):
        """
        Reconstruct a complete point cloud from all views
        
        Args:
            voxel_size (float): Voxel size for downsampling (to reduce redundancy)
            vis_result (bool): Whether to visualize the result
            
        Returns:
            open3d.geometry.PointCloud: Reconstructed point cloud
        """
        print(f"Reconstructing point cloud from {len(self.views)} views")
        start_time = time.time()
        
        # Initialize a combined point cloud
        combined_pcd = o3d.geometry.PointCloud()
        total_points = 0
        successful_views = 0
        
        # Process each view
        for view_id in tqdm(range(len(self.views))):
            pcd = self.reconstruct_from_single_view(view_id)
            
            # Skip if the view didn't produce a valid point cloud
            if pcd is None or len(pcd.points) == 0:
                continue
                
            successful_views += 1
            view_points = len(pcd.points)
            total_points += view_points
            
            # Downsample before combining to reduce redundancy and memory usage
            if voxel_size > 0 and view_points > 10000:
                pcd = pcd.voxel_down_sample(voxel_size)
                print(f"Downsampled view {view_id} from {view_points} to {len(pcd.points)} points")
            
            # Check if we have enough points after downsampling
            if len(pcd.points) < 100:
                print(f"Warning: Too few points remaining after downsampling view {view_id}")
                continue
                
            # Remove outliers from this view's point cloud
            pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            
            # Add to combined point cloud
            combined_pcd += pcd
            
            # Print progress information
            if view_id % 5 == 0 or view_id == len(self.views) - 1:
                print(f"Progress: Combined {successful_views} views, total points: {len(combined_pcd.points)}")
        
        # Check if we have any points
        if len(combined_pcd.points) == 0:
            print("Error: No points were reconstructed from any view")
            return combined_pcd
            
        # Final cleaning and downsampling of the combined point cloud
        print(f"Performing final processing on combined point cloud with {len(combined_pcd.points)} points")
        
        if voxel_size > 0:
            before_points = len(combined_pcd.points)
            combined_pcd = combined_pcd.voxel_down_sample(voxel_size)
            print(f"Downsampled combined point cloud from {before_points} to {len(combined_pcd.points)} points")
        
        # Remove outliers from the combined point cloud
        before_outlier_removal = len(combined_pcd.points)
        combined_pcd, _ = combined_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        print(f"Removed outliers: {before_outlier_removal - len(combined_pcd.points)} points removed")
        
        # Estimate normals if we have enough points
        if len(combined_pcd.points) > 100:
            print("Estimating normals...")
            combined_pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30)
            )
            combined_pcd.orient_normals_consistent_tangent_plane(k=15)
        
        # Save the reconstructed point cloud
        output_path = os.path.join(self.output_dir, "reconstructed_pointcloud.ply")
        o3d.io.write_point_cloud(output_path, combined_pcd)
        
        elapsed_time = time.time() - start_time
        print(f"Reconstruction completed in {elapsed_time:.2f} seconds")
        print(f"Reconstructed point cloud has {len(combined_pcd.points)} points from {successful_views} views")
        print(f"Saved to {output_path}")
        
        # Visualize if requested
        if vis_result and len(combined_pcd.points) > 0:
            self.visualize_point_cloud(combined_pcd)
        
        return combined_pcd
    
    def visualize_point_cloud(self, pcd):
        """
        Visualize a point cloud
        
        Args:
            pcd (open3d.geometry.PointCloud): Point cloud to visualize
        """
        print("Visualizing reconstructed point cloud")
        
        # Create coordinate frame for reference
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        
        # Visualize
        o3d.visualization.draw_geometries([pcd, coordinate_frame])

    def create_unified_point_cloud(self, voxel_size=0.01, min_depth_value=0.0001, outlier_std_ratio=3.0, min_valid_pixels=10):
        """
        Create a unified point cloud from all views directly, rather than 
        creating individual point clouds and then merging them.
        This approach projects all points into a common world coordinate system.
        
        Args:
            voxel_size (float): Voxel size for downsampling (set to 0 for no downsampling)
            min_depth_value (float): Minimum depth value to consider valid (in meters)
            outlier_std_ratio (float): Standard deviation ratio for outlier removal (higher = keep more points)
            min_valid_pixels (int): Minimum number of valid pixels needed in a view
            
        Returns:
            open3d.geometry.PointCloud: Unified point cloud
        """
        print(f"Creating unified point cloud from {len(self.views)} views...")
        print(f"Settings: voxel_size={voxel_size}, min_depth={min_depth_value}, outlier_std_ratio={outlier_std_ratio}")
        start_time = time.time()
        
        # Create an empty unified point cloud
        unified_pcd = o3d.geometry.PointCloud()
        points = []
        colors = []
        
        # Process each view
        for view_id in tqdm(range(len(self.views))):
            # Load RGB and depth data for this view
            view_data = self.load_view_data(view_id)
            if view_data is None:
                continue
                
            rgb_image, depth_map, extrinsic_matrix = view_data
            
            # Use a more permissive threshold for valid depth values to include more points
            valid_depth_mask = depth_map > min_depth_value
            depth_valid_pixels = np.count_nonzero(valid_depth_mask)
            
            if depth_valid_pixels < min_valid_pixels:
                print(f"View {view_id}: Too few valid depth pixels ({depth_valid_pixels})")
                continue
            
            # Convert depth map to meters if needed
            depth_max = np.max(depth_map)
            if depth_max > 1000:
                depth_scale = 0.001  # mm to m
                depth_map = depth_map * depth_scale
                print(f"View {view_id}: Scaled depth by {depth_scale} (mm to m)")
            
            # Get camera intrinsic parameters
            fx = self.intrinsic_matrix[0, 0]
            fy = self.intrinsic_matrix[1, 1]
            cx = self.intrinsic_matrix[0, 2]
            cy = self.intrinsic_matrix[1, 2]
            
            # Inverse of the extrinsic matrix (camera-to-world transform)
            camera_to_world = np.linalg.inv(extrinsic_matrix)
            
            # Generate 3D points for all valid depth pixels
            height, width = depth_map.shape
            y_indices, x_indices = np.where(valid_depth_mask)
            
            # Stack coordinates and depth values
            z_values = depth_map[valid_depth_mask]
            
            # Calculate 3D points in camera coordinate system
            x_values = (x_indices - cx) * z_values / fx
            y_values = (y_indices - cy) * z_values / fy
            
            # Combine into Nx3 array of points in camera coordinates
            camera_points = np.stack([x_values, y_values, z_values], axis=1)
            
            # Convert to homogeneous coordinates (Nx4)
            camera_points_homogeneous = np.hstack([camera_points, np.ones((camera_points.shape[0], 1))])
            
            # Transform to world coordinates using camera-to-world transform
            world_points_homogeneous = np.dot(camera_points_homogeneous, camera_to_world.T)
            
            # Convert back from homogeneous to 3D coordinates
            world_points = world_points_homogeneous[:, :3] / world_points_homogeneous[:, 3:4]
            
            # Get colors for these points
            point_colors = rgb_image[y_indices, x_indices] / 255.0  # Normalize to [0,1]
            
            # Add to our lists
            points.append(world_points)
            colors.append(point_colors)
            
            print(f"View {view_id}: Added {len(world_points)} points")
        
        # Combine all points and colors
        if len(points) == 0:
            print("Error: No points could be created from any view")
            return unified_pcd
            
        combined_points = np.vstack(points)
        combined_colors = np.vstack(colors)
        
        print(f"Combined point cloud has {len(combined_points)} points")
        
        # Create Open3D point cloud from the combined points
        unified_pcd.points = o3d.utility.Vector3dVector(combined_points)
        unified_pcd.colors = o3d.utility.Vector3dVector(combined_colors)
        
        # Only downsample if voxel_size is positive and we have a lot of points
        if voxel_size > 0 and len(unified_pcd.points) > 100000:
            print(f"Downsampling with voxel size {voxel_size}...")
            before_points = len(unified_pcd.points)
            unified_pcd = unified_pcd.voxel_down_sample(voxel_size)
            print(f"Downsampled from {before_points} to {len(unified_pcd.points)} points")
        else:
            print("Skipping downsampling to maintain point density")
        
        # Use a more permissive outlier removal to keep more points
        if len(unified_pcd.points) > 100:
            print(f"Removing outliers with std_ratio={outlier_std_ratio}...")
            before_outlier_removal = len(unified_pcd.points)
            # Use more neighbors and a higher std ratio to keep more points
            unified_pcd, _ = unified_pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=outlier_std_ratio)
            print(f"Removed {before_outlier_removal - len(unified_pcd.points)} outlier points")
        
        # Estimate normals with a larger search radius to work with sparser data
        if len(unified_pcd.points) > 100:
            print("Estimating normals...")
            search_radius = voxel_size * 3 if voxel_size > 0 else 0.05
            unified_pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=search_radius, max_nn=50)
            )
            # Use more points for consistent orientation
            unified_pcd.orient_normals_consistent_tangent_plane(k=30)
        
        elapsed_time = time.time() - start_time
        print(f"Unified point cloud creation completed in {elapsed_time:.2f} seconds")
        
        return unified_pcd

    def reconstruct_unified(self, voxel_size=0.01, vis_result=True):
        """
        Reconstruct a point cloud by unifying all views directly
        
        Args:
            voxel_size (float): Voxel size for downsampling (set to 0 for no downsampling)
            vis_result (bool): Whether to visualize the result
            
        Returns:
            open3d.geometry.PointCloud: Reconstructed point cloud
        """
        # Create unified point cloud with more permissive settings for a denser result
        unified_pcd = self.create_unified_point_cloud(
            voxel_size=voxel_size,
            min_depth_value=0.0001,  # More permissive minimum depth
            outlier_std_ratio=3.0,   # More permissive outlier removal
            min_valid_pixels=10      # Even views with few points can contribute
        )
        
        # Save the reconstructed point cloud
        output_path = os.path.join(self.output_dir, "unified_pointcloud.ply")
        o3d.io.write_point_cloud(output_path, unified_pcd)
        
        print(f"Unified point cloud has {len(unified_pcd.points)} points")
        print(f"Saved to {output_path}")
        
        # Visualize if requested
        if vis_result and len(unified_pcd.points) > 0:
            self.visualize_point_cloud(unified_pcd)
        
        return unified_pcd

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Reconstruct point cloud from RGB-D captures")
    parser.add_argument("capture_dir", help="Directory containing captured data")
    parser.add_argument("--output-dir", help="Directory to save the reconstructed point cloud")
    parser.add_argument("--voxel-size", type=float, default=0.01, help="Voxel size for downsampling (0 for no downsampling)")
    parser.add_argument("--no-vis", action="store_true", help="Disable visualization")
    parser.add_argument("--debug-view", type=int, default=-1, help="Process only this view ID for debugging")
    parser.add_argument("--unified", action="store_true", help="Use unified reconstruction approach")
    parser.add_argument("--high-density", action="store_true", help="Produce higher density point cloud (no downsampling)")
    
    args = parser.parse_args()
    
    # If high-density is requested, set voxel_size to 0 (no downsampling)
    if args.high_density:
        args.voxel_size = 0
        print("High density mode enabled - no downsampling will be performed")
    
    reconstructor = PointCloudReconstructor(args.capture_dir, args.output_dir)
    
    # Debug a single view if requested
    if args.debug_view >= 0:
        print(f"\n===== Debugging single view {args.debug_view} =====")
        view_data = reconstructor.load_view_data(args.debug_view)
        if view_data:
            rgb_image, depth_map, extrinsic_matrix = view_data
            pcd = reconstructor.depth_to_pointcloud(rgb_image, depth_map, extrinsic_matrix)
            output_path = os.path.join(reconstructor.output_dir, f"debug_view_{args.debug_view:03d}.ply")
            o3d.io.write_point_cloud(output_path, pcd)
            print(f"Debug point cloud saved to {output_path}")
            
            if not args.no_vis:
                reconstructor.visualize_point_cloud(pcd)
    elif args.unified:
        # Use unified reconstruction approach
        reconstructor.reconstruct_unified(voxel_size=args.voxel_size, vis_result=not args.no_vis)
    else:
        # Process all views with the original approach
        reconstructor.reconstruct_from_all_views(voxel_size=args.voxel_size, vis_result=not args.no_vis)

if __name__ == "__main__":
    main() 