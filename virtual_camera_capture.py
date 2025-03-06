#!/usr/bin/env python
import os
import cv2
import numpy as np
import json
import argparse
import open3d as o3d
from datetime import datetime
import math
import time

class VirtualCameraCapture:
    def __init__(self, ply_path, output_dir=None, resolution=(1280, 720)):
        """
        Initialize the virtual camera capture system
        
        Args:
            ply_path (str): Path to the input PLY file
            output_dir (str, optional): Output directory for rendered images
            resolution (tuple): Image resolution as (width, height)
        """
        self.resolution = resolution
        self.ply_path = ply_path  # Store the PLY file path
        
        # Set output directory
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"virtual_capture_{timestamp}"
        
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
        self.images_dir = os.path.join(output_dir, "images")  # Store the images directory path
        
        # Cache directory for normals
        self.cache_dir = os.path.join(output_dir, "cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Create a cached filename based on the input ply path
        ply_basename = os.path.basename(ply_path)
        self.normals_cache_path = os.path.join(self.cache_dir, f"{os.path.splitext(ply_basename)[0]}_with_normals.ply")
        
        # Load the point cloud
        try:
            # Check if we have a cached version with normals
            if os.path.exists(self.normals_cache_path):
                print(f"Loading point cloud with cached normals from: {self.normals_cache_path}")
                self.point_cloud = o3d.io.read_point_cloud(self.normals_cache_path)
                if not self.point_cloud.has_points():
                    raise ValueError(f"No points found in cached file {self.normals_cache_path}")
                
                # Verify the cached point cloud has normals
                if not self.point_cloud.has_normals():
                    print("Cached point cloud doesn't have normals. Loading original and recomputing...")
                    self.load_and_process_point_cloud(ply_path)
            else:
                # Load and process the original point cloud
                self.load_and_process_point_cloud(ply_path)
        except Exception as e:
            raise ValueError(f"Failed to load point cloud: {e}")
        
        # Calculate bounding box and center
        self.bbox = self.point_cloud.get_axis_aligned_bounding_box()
        self.center = self.bbox.get_center()
        self.extent = self.bbox.get_extent()
        self.radius = np.linalg.norm(self.extent) * 0.8  # Camera distance from center
        
        # Display the point cloud to confirm it's loaded correctly
        self.preview_point_cloud()
        
        # Initialize camera parameters from a default visualizer setup
        self.initialize_camera_parameters()
        
        # Initialize captured data
        self.captured_frames = []
    
    def load_and_process_point_cloud(self, ply_path):
        """
        Load and process the point cloud, computing normals if needed
        
        Args:
            ply_path (str): Path to the input PLY file
        """
        print(f"Loading point cloud: {ply_path}")
        self.point_cloud = o3d.io.read_point_cloud(ply_path)
        if not self.point_cloud.has_points():
            raise ValueError(f"No points found in {ply_path}")
        
        # Print basic information about the point cloud
        print(f"Point cloud loaded: {ply_path}")
        print(f"Number of points: {len(self.point_cloud.points)}")
        print(f"Point cloud has colors: {self.point_cloud.has_colors()}")
        print(f"Point cloud has normals: {self.point_cloud.has_normals()}")
        
        # If the point cloud doesn't have normals, estimate them for better visualization
        if not self.point_cloud.has_normals():
            print("Estimating normals for better visualization...")
            self.point_cloud.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
            )
            self.point_cloud.orient_normals_consistent_tangent_plane(100)
            
            # Save the point cloud with normals for future use
            print(f"Saving point cloud with normals to cache: {self.normals_cache_path}")
            o3d.io.write_point_cloud(self.normals_cache_path, self.point_cloud)
    
    def preview_point_cloud(self):
        """
        Show the point cloud to verify it's loaded correctly
        """
        print("Showing point cloud preview (close window to continue)...")
        o3d.visualization.draw_geometries([self.point_cloud], 
                                        window_name="Point Cloud Preview",
                                        width=800, 
                                        height=600,
                                        point_show_normal=False)
    
    def initialize_camera_parameters(self):
        """
        Initialize camera parameters for rendering
        """
        width, height = self.resolution
        print(f"Initializing camera parameters for resolution {width}x{height}")
        
        # Initialize improved intrinsic parameters
        # Use a much wider field of view for better coverage
        # This corresponds to approximately 75-80 degree FOV (similar to smartphone cameras)
        focal_length = 0.65 * max(width, height)  # Reduced from 0.8 for even wider FOV
        cx, cy = width / 2, height / 2  # Principal point
        
        # Create intrinsic matrix
        self.intrinsic_matrix = np.array([
            [focal_length, 0, cx],
            [0, focal_length, cy],
            [0, 0, 1]
        ])
        
        # Create Open3D camera intrinsic
        self.intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width, height, focal_length, focal_length, cx, cy
        )
        
        # No distortion in virtual camera
        self.distortion_coeffs = np.zeros(5)
        
        vertical_fov = 2 * np.arctan(height / (2 * focal_length)) * 180 / np.pi
        horizontal_fov = 2 * np.arctan(width / (2 * focal_length)) * 180 / np.pi
        
        print("Camera parameters initialized:")
        print(f"Intrinsic matrix:\n{self.intrinsic_matrix}")
        print(f"Field of view: {vertical_fov:.1f} degrees vertical")
        print(f"Field of view: {horizontal_fov:.1f} degrees horizontal")
    
    def get_camera_positions(self, num_views):
        """
        Generate comprehensive camera positions to cover the entire point cloud
        
        Args:
            num_views (int): Total number of views to generate
        
        Returns:
            List of camera positions (eye points)
        """
        positions = []
        
        # Instead of a full sphere, use a hemisphere with more consistent baselines
        # This will ensure cameras are not too far apart and have better overlap
        
        # Calculate a better radius based on the point cloud size
        # Get much closer to the object for better feature extraction and matching
        self.radius = np.linalg.norm(self.extent) * 0.8  # Reduced from 1.2 to get much closer
        print(f"Camera distance from center: {self.radius}")
        
        # Create a hemisphere of positions with more consistent spacing
        # Use a spiral pattern on a hemisphere for more uniform coverage
        indices = np.arange(0, num_views)
        phi = np.arccos(1.0 - indices / float(num_views - 1))  # 0 to pi/2 (hemisphere)
        theta = np.pi * (1 + 5**0.5) * indices  # Golden angle in radians
        
        # Convert spherical to Cartesian coordinates
        x = self.radius * np.sin(phi) * np.cos(theta)
        y = self.radius * np.sin(phi) * np.sin(theta)
        z = self.radius * np.cos(phi)  # Will be positive for hemisphere
        
        # Add some randomization to avoid perfectly regular patterns
        # which can cause issues with feature matching
        np.random.seed(42)  # For reproducibility
        jitter = self.radius * 0.05  # 5% jitter
        x += np.random.uniform(-jitter, jitter, num_views)
        y += np.random.uniform(-jitter, jitter, num_views)
        z += np.random.uniform(-jitter, jitter, num_views)
        
        # Ensure minimum distance between consecutive cameras (for good baseline)
        min_baseline = self.radius * 0.08  # Minimum baseline - reduced for closer views
        max_baseline = self.radius * 0.25  # Maximum baseline - also reduced
        
        # Center the positions around the point cloud center
        for i in range(num_views):
            pos = np.array([
                x[i] + self.center[0],
                y[i] + self.center[1],
                z[i] + self.center[2]
            ])
            
            # Add position if it's the first one or has appropriate baseline from previous
            if i == 0:
                positions.append(pos)
            else:
                baseline = np.linalg.norm(pos - positions[-1])
                if min_baseline <= baseline <= max_baseline:
                    positions.append(pos)
                else:
                    # Adjust position to get better baseline
                    direction = pos - positions[-1]
                    direction = direction / np.linalg.norm(direction)
                    adjusted_pos = positions[-1] + direction * (min_baseline + max_baseline) / 2
                    positions.append(adjusted_pos)
        
        # If we ended up with fewer positions due to adjustments, add more
        while len(positions) < num_views:
            # Add intermediate positions between existing ones
            new_positions = []
            for i in range(len(positions) - 1):
                midpoint = (positions[i] + positions[i+1]) / 2
                # Add some jitter to avoid collinearity
                midpoint += np.random.uniform(-jitter, jitter, 3)
                new_positions.append(midpoint)
                
            # Add new positions, but don't exceed num_views
            for pos in new_positions:
                if len(positions) < num_views:
                    positions.append(pos)
                else:
                    break
        
        # Ensure we have exactly num_views positions
        positions = positions[:num_views]
        
        # Print the range of baselines for debugging
        baselines = []
        for i in range(len(positions) - 1):
            baseline = np.linalg.norm(positions[i] - positions[i+1])
            baselines.append(baseline)
        print(f"Camera baseline range: {min(baselines):.2f} to {max(baselines):.2f}")
        
        return positions
    
    def calculate_extrinsic_matrix(self, eye_pos):
        """
        Calculate extrinsic matrix for a camera position looking at center
        
        Args:
            eye_pos (np.ndarray): Camera position
            
        Returns:
            extrinsic (4x4 matrix): Camera extrinsic matrix
        """
        # Look at the center of the point cloud
        target = self.center
        
        # Try to vary the up vector slightly for each camera to avoid
        # having all cameras with the same orientation
        # This helps with feature matching between views
        
        # Generate a slightly randomized up vector
        # Start with standard up vector
        up = np.array([0, 1, 0])
        
        # Add small random perturbation to up vector (within 15 degrees)
        np.random.seed(int(np.sum(eye_pos) * 1000) % 10000)  # Deterministic but different for each position
        angle = np.random.uniform(-0.25, 0.25)  # ~15 degrees in radians
        axis = np.random.uniform(-1, 1, 3)
        axis = axis / np.linalg.norm(axis)
        
        # Create rotation matrix for the perturbation
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        ux, uy, uz = axis
        rotation_matrix = np.array([
            [cos_angle + ux*ux*(1-cos_angle), ux*uy*(1-cos_angle)-uz*sin_angle, ux*uz*(1-cos_angle)+uy*sin_angle],
            [uy*ux*(1-cos_angle)+uz*sin_angle, cos_angle+uy*uy*(1-cos_angle), uy*uz*(1-cos_angle)-ux*sin_angle],
            [uz*ux*(1-cos_angle)-uy*sin_angle, uz*uy*(1-cos_angle)+ux*sin_angle, cos_angle+uz*uz*(1-cos_angle)]
        ])
        
        # Apply rotation to up vector
        up = rotation_matrix @ up
            
        # Calculate camera basis vectors
        forward = target - eye_pos
        forward = forward / np.linalg.norm(forward)
        
        right = np.cross(forward, up)
        if np.linalg.norm(right) < 1e-6:
            # If forward and up are nearly parallel, choose a different up
            up = np.array([1, 0, 0])
            right = np.cross(forward, up)
        
        right = right / np.linalg.norm(right)
        
        new_up = np.cross(right, forward)
        new_up = new_up / np.linalg.norm(new_up)
        
        # Construct the rotation matrix
        rotation = np.eye(3)
        rotation[0, :] = right
        rotation[1, :] = new_up
        rotation[2, :] = -forward
        
        # Construct the extrinsic matrix [R|t]
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = rotation
        extrinsic[:3, 3] = -rotation @ eye_pos
        
        return extrinsic
    
    def capture_view(self, view_id, eye_pos):
        """
        Capture a single view of the point cloud from a specific camera position
        
        Args:
            view_id (int): Current view ID
            eye_pos (np.ndarray): Camera position
            
        Returns:
            bool: True if successful
        """
        print(f"Capturing view {view_id} from position {eye_pos}")
        width, height = self.resolution
        
        # Create a visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=width, height=height, visible=False)
        
        # Add the point cloud to the visualizer
        vis.add_geometry(self.point_cloud)
        
        # Apply render settings for better visibility
        render_opt = vis.get_render_option()
        render_opt.point_size = 3.0  # Larger points for better visibility
        render_opt.background_color = np.array([0, 0, 0])  # Black background
        render_opt.point_color_option = o3d.visualization.PointColorOption.Default
        render_opt.light_on = True
        
        # Set view to look at the center of the point cloud
        view_control = vis.get_view_control()
        view_control.set_lookat(self.center)
        view_control.set_front(self.center - eye_pos)
        view_control.set_up([0, 1, 0])
        view_control.set_zoom(0.7)  # Ensure the object is fully visible
        
        # Ensure the point cloud is in view
        vis.poll_events()
        vis.update_renderer()
        
        # Render and capture the image
        img = None
        for _ in range(3):  # Try a few times to ensure proper rendering
            vis.poll_events()
            vis.update_renderer()
            image = vis.capture_screen_float_buffer(do_render=True)
            if image is not None:
                img = (np.asarray(image) * 255).astype(np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                break
            time.sleep(0.1)
        
        if img is None:
            print(f"Warning: Failed to capture image for view {view_id}")
            vis.destroy_window()
            return False
        
        # Save the image
        image_path = os.path.join(self.images_dir, f"view_{view_id:03d}.jpg")
        cv2.imwrite(image_path, img)
        
        # Get the actual camera parameters
        params = view_control.convert_to_pinhole_camera_parameters()
        
        # Store frame data
        self.captured_frames.append({
            'view_id': view_id,
            'image_path': image_path,
            'extrinsic_matrix': params.extrinsic.tolist(),
            'camera_position': eye_pos.tolist()
        })
        
        # Clean up
        vis.destroy_window()
        
        # Check if the image actually contains the point cloud (not empty/black)
        non_black_pixels = np.sum(img > 10)
        if non_black_pixels < (width * height * 0.01):  # Less than 1% non-black pixels
            print(f"Warning: View {view_id} may be empty or not showing the point cloud")
            return False
            
        print(f"Saved view {view_id}: {image_path}")
        return True
    
    def capture_multiview(self, num_views=20):
        """
        Capture multiple views of the point cloud from different viewpoints
        
        Args:
            num_views (int): Number of views to capture
        """
        print(f"Starting multi-view capture ({num_views} views)")
        
        # Generate camera positions
        camera_positions = self.get_camera_positions(num_views)
        
        # Count successful captures
        success_count = 0
        
        for view_id, eye_pos in enumerate(camera_positions):
            # Capture this view
            success = self.capture_view(view_id, eye_pos)
            if success:
                success_count += 1
                print(f"Progress: {success_count}/{num_views} views captured successfully")
            else:
                print(f"Warning: Failed to capture view {view_id}")
            
            # Short delay between captures
            time.sleep(0.2)
        
        if success_count == 0:
            print("Error: No views could be captured successfully")
            return False
            
        print(f"Completed with {success_count}/{num_views} successful captures")
        return True
    
    def save_camera_parameters(self):
        """
        Save camera parameters to JSON file
        """
        params = {
            'intrinsic_matrix': self.intrinsic_matrix.tolist(),
            'distortion_coeffs': self.distortion_coeffs.tolist(),
            'resolution': self.resolution,
            'views': self.captured_frames,
            'point_cloud_path': self.ply_path,
            'center': self.center.tolist(),
            'extent': self.extent.tolist(),
            'radius': float(self.radius)
        }
        
        params_path = os.path.join(self.output_dir, 'camera_parameters.json')
        with open(params_path, 'w') as f:
            json.dump(params, f, indent=4)
            
        print(f"Camera parameters saved to: {params_path}")

def main():
    parser = argparse.ArgumentParser(description='Virtual camera multi-view capture tool')
    parser.add_argument('ply_file', type=str,
                        help='Path to the PLY file to capture')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output directory for captured data')
    parser.add_argument('--resolution', '-r', type=str, default='1280x720',
                        help='Camera resolution, format: WIDTHxHEIGHT (default: 1280x720)')
    parser.add_argument('--views', '-v', type=int, default=20,
                        help='Number of views to capture (default: 20)')
    parser.add_argument('--no-preview', action='store_true',
                        help='Skip the point cloud preview')
    
    args = parser.parse_args()
    
    # Parse resolution
    width, height = map(int, args.resolution.split('x'))
    resolution = (width, height)
    
    try:
        # Initialize virtual camera capture
        vcc = VirtualCameraCapture(
            ply_path=args.ply_file,
            output_dir=args.output,
            resolution=resolution
        )
        
        # Capture multiple views
        if vcc.capture_multiview(num_views=args.views):
            # Save camera parameters
            vcc.save_camera_parameters()
            print(f"Multi-view capture completed. Data saved to: {vcc.output_dir}")
        else:
            print("Multi-view capture failed or produced insufficient views")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 