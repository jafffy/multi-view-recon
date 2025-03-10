#!/usr/bin/env python
import os
import cv2
import numpy as np
import json
import argparse
import open3d as o3d
from datetime import datetime
import time

# Import the point cloud processor
from point_cloud_utils import PointCloudProcessor

class VirtualCameraCapture:
    def __init__(self, ply_path, output_dir=None, resolution=(1280, 720), capture_depth=True, depth_scale_factor=1.0):
        """
        Initialize the virtual camera capture system
        
        Args:
            ply_path (str): Path to the input PLY file
            output_dir (str, optional): Output directory for rendered images
            resolution (tuple): Image resolution as (width, height)
            capture_depth (bool): Whether to capture depth images
            depth_scale_factor (float): Factor to scale depth values (default: 1.0, use 0.001 to convert mm to meters)
        """
        self.resolution = resolution
        self.ply_path = ply_path  # Store the PLY file path
        self.capture_depth = capture_depth
        self.depth_scale_factor = depth_scale_factor  # Store the depth scale factor
        
        # Set output directory
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"virtual_capture_{timestamp}"
        
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
        self.images_dir = os.path.join(output_dir, "images")  # Store the images directory path
        
        # Create a directory for depth images
        if self.capture_depth:
            os.makedirs(os.path.join(output_dir, "depth"), exist_ok=True)
            self.depth_dir = os.path.join(output_dir, "depth")  # Store the depth images directory path
        
        # Cache directory for normals
        self.cache_dir = os.path.join(output_dir, "cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize the point cloud processor
        self.processor = PointCloudProcessor(ply_path, self.cache_dir)
        
        # Get point cloud properties
        self.point_cloud = self.processor.point_cloud
        self.center = self.processor.center
        self.extent = self.processor.extent
        self.radius = self.processor.radius
        self.bbox = self.processor.bbox
        
        # Display the point cloud to confirm it's loaded correctly
        self.preview_point_cloud()
        
        # Initialize camera parameters from a default visualizer setup
        self.initialize_camera_parameters()
        
        # Initialize captured data
        self.captured_frames = []
    
    def preview_point_cloud(self):
        """
        Show the point cloud to verify it's loaded correctly
        """
        self.processor.preview_point_cloud("Point Cloud Preview", 800, 600)
    
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
        return self.processor.get_camera_positions(num_views)
    
    def calculate_extrinsic_matrix(self, eye_pos):
        """
        Calculate extrinsic matrix for a camera position looking at center
        
        Args:
            eye_pos (np.ndarray): Camera position
            
        Returns:
            extrinsic (4x4 matrix): Camera extrinsic matrix
        """
        return self.processor.calculate_extrinsic_matrix(eye_pos)
    
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
        depth = None
        for _ in range(3):  # Try a few times to ensure proper rendering
            vis.poll_events()
            vis.update_renderer()
            image = vis.capture_screen_float_buffer(do_render=True)
            
            if self.capture_depth:
                depth_image = vis.capture_depth_float_buffer(do_render=True)
                if image is not None and depth_image is not None:
                    img = (np.asarray(image) * 255).astype(np.uint8)
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    depth = np.asarray(depth_image)
                    break
            else:
                if image is not None:
                    img = (np.asarray(image) * 255).astype(np.uint8)
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    break
                    
            time.sleep(0.1)
        
        if img is None:
            print(f"Warning: Failed to capture image for view {view_id}")
            vis.destroy_window()
            return False
            
        if self.capture_depth and depth is None:
            print(f"Warning: Failed to capture depth for view {view_id}")
            vis.destroy_window()
            return False
        
        # Save the RGB image
        image_path = os.path.join(self.images_dir, f"view_{view_id:03d}.jpg")
        cv2.imwrite(image_path, img)
        
        # Prepare frame data with camera extrinsics and position
        frame_data = {
            'view_id': view_id,
            'image_path': image_path,
            'extrinsic_matrix': view_control.convert_to_pinhole_camera_parameters().extrinsic.tolist(),
            'camera_position': (eye_pos * self.depth_scale_factor).tolist()
        }
        
        # Process depth data if needed
        if self.capture_depth:
            # Normalize depth for visualization
            depth_vis = np.zeros_like(depth)
            depth_mask = depth > 0
            if np.any(depth_mask):
                valid_depth = depth[depth_mask]
                depth_min = np.min(valid_depth)
                depth_max = np.max(valid_depth)
                depth_scale = depth_max - depth_min  # Calculate depth scale
                depth_vis[depth_mask] = (depth[depth_mask] - depth_min) / (depth_scale if depth_scale != 0 else 1)
                depth_vis = (depth_vis * 255).astype(np.uint8)
                
                # Apply depth scale factor for reporting and storage
                scaled_depth_min = depth_min * self.depth_scale_factor
                scaled_depth_max = depth_max * self.depth_scale_factor
                scaled_depth_scale = depth_scale * self.depth_scale_factor
                
                # Print depth scale information
                print(f"Depth information for view {view_id}:")
                print(f"  - Minimum depth: {scaled_depth_min:.4f}")
                print(f"  - Maximum depth: {scaled_depth_max:.4f}")
                print(f"  - Depth scale: {scaled_depth_scale:.4f}")
            else:
                depth_min = depth_max = depth_scale = 0
                scaled_depth_min = scaled_depth_max = scaled_depth_scale = 0
                print(f"No valid depth points for view {view_id}")
            
            # Save raw depth data (original values for processing flexibility)
            depth_path = os.path.join(self.depth_dir, f"depth_{view_id:03d}.npy")
            np.save(depth_path, depth)
            
            # Save depth visualization for inspection
            depth_vis_path = os.path.join(self.depth_dir, f"depth_{view_id:03d}.png")
            cv2.imwrite(depth_vis_path, depth_vis)
            
            # Add depth related info to frame data (using scaled values)
            frame_data['depth_path'] = depth_path
            frame_data['depth_vis_path'] = depth_vis_path
            frame_data['depth_min'] = float(scaled_depth_min)
            frame_data['depth_max'] = float(scaled_depth_max)
            frame_data['depth_scale'] = float(scaled_depth_scale)
            
            print(f"Saved view {view_id}: {image_path} and depth: {depth_path}")
        else:
            print(f"Saved view {view_id}: {image_path}")
        
        # Store frame data
        self.captured_frames.append(frame_data)
        
        # Clean up
        vis.destroy_window()
        
        # Check if the image actually contains the point cloud (not empty/black)
        non_black_pixels = np.sum(img > 10)
        if non_black_pixels < (width * height * 0.01):  # Less than 1% non-black pixels
            print(f"Warning: View {view_id} may be empty or not showing the point cloud")
            return False
            
        return True
    
    def capture_multiview(self, num_views=20):
        """
        Capture multiple views of the point cloud from different viewpoints.
        This captures both RGB images and depth maps (unless capture_depth=False).
        
        The RGB images are saved as JPEG files in the 'images' directory.
        The depth maps are saved as NumPy (.npy) files in the 'depth' directory.
        A visualization of the depth maps is also saved as PNG files.
        
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
        Save camera parameters to JSON file.
        
        This includes intrinsic and extrinsic camera parameters,
        paths to RGB images and depth maps (if captured),
        and information about the point cloud.
        """
        params = {
            'intrinsic_matrix': self.intrinsic_matrix.tolist(),
            'distortion_coeffs': self.distortion_coeffs.tolist(),
            'resolution': self.resolution,
            'views': self.captured_frames,
            'point_cloud_path': self.ply_path,
            'center': (self.center * self.depth_scale_factor).tolist(),
            'extent': (self.extent * self.depth_scale_factor).tolist(),
            'radius': float(self.radius * self.depth_scale_factor),
            'depth_scale_factor': float(self.depth_scale_factor),
            'original_units': 'millimeters',
            'scaled_units': 'millimeters' if self.depth_scale_factor == 1.0 else 'meters' if self.depth_scale_factor == 0.001 else 'custom'
        }
        
        params_path = os.path.join(self.output_dir, 'camera_parameters.json')
        with open(params_path, 'w') as f:
            json.dump(params, f, indent=4)
            
        print(f"Camera parameters saved to: {params_path}")

def main():
    parser = argparse.ArgumentParser(description='Virtual camera multi-view capture tool for RGB and depth images')
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
    parser.add_argument('--no-depth', action='store_true',
                        help='Skip capturing depth images (RGB images only)')
    parser.add_argument('--depth-scale', '-ds', type=float, default=1.0,
                        help='Scale factor for depth values (default: 1.0, use 0.001 for mm to meters)')
    
    args = parser.parse_args()
    
    # Parse resolution
    width, height = map(int, args.resolution.split('x'))
    resolution = (width, height)
    
    try:
        # Initialize virtual camera capture
        vcc = VirtualCameraCapture(
            ply_path=args.ply_file,
            output_dir=args.output,
            resolution=resolution,
            capture_depth=not args.no_depth,
            depth_scale_factor=args.depth_scale
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