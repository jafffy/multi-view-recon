#!/usr/bin/env python
import os
import sys
import numpy as np
import json
import argparse
import open3d as o3d
import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import time

def load_camera_parameters(params_path):
    """
    Load camera parameters from a JSON file
    
    Args:
        params_path (str): Path to camera parameters JSON file
    
    Returns:
        dict: Camera parameters
    """
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"Camera parameters file not found: {params_path}")
    
    with open(params_path, 'r') as f:
        params = json.load(f)
    
    return params

def create_point_cloud_from_rgbd(color_img, depth_img, intrinsic, extrinsic, depth_scale, depth_min, depth_max):
    """
    Create a point cloud from RGB and depth images
    
    Args:
        color_img (numpy.ndarray): RGB image
        depth_img (numpy.ndarray): Depth image
        intrinsic (numpy.ndarray): Camera intrinsic matrix
        extrinsic (numpy.ndarray): Camera extrinsic matrix
        depth_scale (float): Scale factor for depth values
        depth_min (float): Minimum depth value
        depth_max (float): Maximum depth value
    
    Returns:
        open3d.geometry.PointCloud: Point cloud
    """
    # Normalize depth to the range expected by Open3D
    depth_normalized = depth_img.astype(np.float32)
    
    # Clip depth values to valid range
    depth_normalized = np.clip(depth_normalized, depth_min, depth_max)
    
    # Create Open3D images
    color_o3d = o3d.geometry.Image(color_img)
    depth_o3d = o3d.geometry.Image(depth_normalized)
    
    # Create RGBD image
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d, depth_o3d, 
        depth_scale=depth_scale,
        depth_trunc=depth_max,
        convert_rgb_to_intensity=False
    )
    
    # Create intrinsic parameters
    intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(
        width=color_img.shape[1],
        height=color_img.shape[0],
        fx=intrinsic[0][0],
        fy=intrinsic[1][1],
        cx=intrinsic[0][2],
        cy=intrinsic[1][2]
    )
    
    # Create point cloud
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd, intrinsic_o3d, extrinsic=np.linalg.inv(extrinsic)
    )
    
    # Apply statistical outlier removal - with try/except in case this fails
    try:
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    except Exception as e:
        print(f"Warning: Could not apply statistical outlier removal: {e}")
    
    # Estimate normals for better rendering - with try/except in case this fails
    try:
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        pcd.orient_normals_towards_camera_location(camera_location=np.array([0, 0, 0]))
    except Exception as e:
        print(f"Warning: Could not estimate normals: {e}")
    
    # Apply voxel downsampling if point cloud is very dense - with try/except in case this fails
    try:
        if len(pcd.points) > 500000:
            pcd = pcd.voxel_down_sample(voxel_size=0.01)
    except Exception as e:
        print(f"Warning: Could not apply voxel downsampling: {e}")
    
    return pcd

class RGBDPointCloudViewer:
    def __init__(self, root, params_dir):
        """
        Initialize the RGB-D point cloud viewer
        
        Args:
            root (tk.Tk): Tkinter root window
            params_dir (str): Directory containing camera parameters JSON file
        """
        self.root = root
        self.root.title("RGB-D Point Cloud Viewer")
        self.root.geometry("1200x800")
        
        # Load camera parameters
        self.params_path = os.path.join(params_dir, "camera_parameters.json")
        self.params = load_camera_parameters(self.params_path)
        
        # Visualization settings
        self.point_size = 2.0
        self.show_coordinate_frame = True
        self.show_camera_frustum = False
        self.show_normals = False
        self.color_mode = "RGB"  # RGB, Height, Normal
        
        # Create GUI
        self.create_gui()
        
        # Current view
        self.current_view_id = 0
        self.current_point_cloud = None
        self.coord_frame = None
        self.camera_frustums = []
        self.vis = None
        
        # Initialize visualization
        self.initialize_visualization()
        
    def create_gui(self):
        """Create the GUI elements"""
        # Main panels
        left_panel = ttk.Frame(self.root, padding="10")
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        right_panel = ttk.Frame(self.root, padding="10")
        right_panel.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Frame for visualization controls
        control_frame = ttk.LabelFrame(right_panel, text="View Controls", padding="10")
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        # View selection dropdown
        ttk.Label(control_frame, text="Select View:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        
        # Get number of views
        num_views = len(self.params["views"])
        view_ids = [f"View {view['view_id']}" for view in self.params["views"]]
        
        self.view_var = tk.StringVar()
        self.view_dropdown = ttk.Combobox(control_frame, textvariable=self.view_var, values=view_ids, width=15)
        self.view_dropdown.current(0)
        self.view_dropdown.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        self.view_dropdown.bind("<<ComboboxSelected>>", self.on_view_selected)
        
        # Visualization options
        viz_frame = ttk.LabelFrame(right_panel, text="Visualization Options", padding="10")
        viz_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        # Point size slider
        ttk.Label(viz_frame, text="Point Size:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.point_size_var = tk.DoubleVar(value=self.point_size)
        point_size_slider = ttk.Scale(viz_frame, from_=1.0, to=5.0, orient=tk.HORIZONTAL, 
                                     variable=self.point_size_var, length=150)
        point_size_slider.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        point_size_slider.bind("<ButtonRelease-1>", self.update_render_options)
        
        # Show coordinate frame
        self.coord_frame_var = tk.BooleanVar(value=self.show_coordinate_frame)
        coord_frame_cb = ttk.Checkbutton(viz_frame, text="Show Coordinate Frame", 
                                       variable=self.coord_frame_var, command=self.toggle_coordinate_frame)
        coord_frame_cb.grid(row=1, column=0, columnspan=2, sticky=tk.W, padx=5, pady=5)
        
        # Show camera frustum
        self.camera_frustum_var = tk.BooleanVar(value=self.show_camera_frustum)
        camera_frustum_cb = ttk.Checkbutton(viz_frame, text="Show Camera Frustums", 
                                          variable=self.camera_frustum_var, command=self.toggle_camera_frustums)
        camera_frustum_cb.grid(row=2, column=0, columnspan=2, sticky=tk.W, padx=5, pady=5)
        
        # Show normals
        self.normals_var = tk.BooleanVar(value=self.show_normals)
        normals_cb = ttk.Checkbutton(viz_frame, text="Show Normals", 
                                    variable=self.normals_var, command=self.toggle_normals)
        normals_cb.grid(row=3, column=0, columnspan=2, sticky=tk.W, padx=5, pady=5)
        
        # Color mode
        ttk.Label(viz_frame, text="Color Mode:").grid(row=4, column=0, sticky=tk.W, padx=5, pady=5)
        
        color_modes = ["RGB", "Height", "Normal"]
        self.color_mode_var = tk.StringVar(value=self.color_mode)
        color_mode_dropdown = ttk.Combobox(viz_frame, textvariable=self.color_mode_var, 
                                         values=color_modes, width=10, state="readonly")
        color_mode_dropdown.grid(row=4, column=1, sticky=tk.W, padx=5, pady=5)
        color_mode_dropdown.bind("<<ComboboxSelected>>", self.update_color_mode)
        
        # Camera information frame
        camera_info_frame = ttk.LabelFrame(right_panel, text="Camera Information", padding="10")
        camera_info_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        # Camera position
        ttk.Label(camera_info_frame, text="Position:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.camera_pos_label = ttk.Label(camera_info_frame, text="[0, 0, 0]")
        self.camera_pos_label.grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Camera resolution
        ttk.Label(camera_info_frame, text="Resolution:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.camera_res_label = ttk.Label(camera_info_frame, text="0 x 0")
        self.camera_res_label.grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Control buttons frame
        control_buttons_frame = ttk.Frame(right_panel, padding="10")
        control_buttons_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        # Reset view button
        reset_view_btn = ttk.Button(control_buttons_frame, text="Reset View", command=self.reset_view)
        reset_view_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Capture screenshot button
        screenshot_btn = ttk.Button(control_buttons_frame, text="Take Screenshot", command=self.take_screenshot)
        screenshot_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Image preview frame
        self.preview_frame = ttk.LabelFrame(right_panel, text="Image Preview", padding="10")
        self.preview_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        # Image label
        self.image_label = ttk.Label(self.preview_frame)
        self.image_label.pack(side=tk.TOP, padx=5, pady=5)
        
        # Depth image label
        self.depth_label = ttk.Label(self.preview_frame)
        self.depth_label.pack(side=tk.TOP, padx=5, pady=5)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Add key bindings for view navigation
        self.root.bind("<Left>", lambda event: self.change_view(-1))
        self.root.bind("<Right>", lambda event: self.change_view(1))
        
    def initialize_visualization(self):
        """Initialize the Open3D visualization window"""
        # Create visualizer
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(width=800, height=600, window_name="RGB-D Point Cloud Viewer")
        
        # Set initial render options
        render_option = self.vis.get_render_option()
        render_option.point_size = self.point_size
        render_option.background_color = np.array([0.1, 0.1, 0.1])
        render_option.show_coordinate_frame = self.show_coordinate_frame
        
        # Enable depth test and lighting for better visualization
        render_option.light_on = True
        render_option.point_show_normal = self.show_normals
        
        # Configure the camera view
        view_control = self.vis.get_view_control()
        view_control.set_zoom(0.8)
        
        # Add coordinate frame
        self.coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        if self.show_coordinate_frame:
            self.vis.add_geometry(self.coord_frame)
        
        # Update visualization with initial view
        self.update_visualization()
        
    def update_visualization(self):
        """Update the visualization with the current view"""
        try:
            # Get view data
            view = self.params["views"][self.current_view_id]
            
            # Get base directory where camera_parameters.json is located
            base_dir = os.path.dirname(self.params_path)
            
            # Extract the relative paths from the view data
            img_relative_path = view["image_path"]
            depth_relative_path = view["depth_path"]
            
            # If the paths contain the directory name, remove it to avoid duplication
            dir_name = os.path.basename(base_dir)
            if img_relative_path.startswith(dir_name + "/"):
                img_relative_path = img_relative_path[len(dir_name) + 1:]
            if depth_relative_path.startswith(dir_name + "/"):
                depth_relative_path = depth_relative_path[len(dir_name) + 1:]
            
            # Construct absolute paths
            img_path = os.path.join(base_dir, img_relative_path)
            depth_path = os.path.join(base_dir, depth_relative_path)
            
            self.status_var.set(f"Loading image from: {img_path}")
            print(f"Loading image from: {img_path}")
            color_img = cv2.imread(img_path)
            if color_img is None:
                raise FileNotFoundError(f"Could not load image from {img_path}")
            color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
            
            self.status_var.set(f"Loading depth from: {depth_path}")
            print(f"Loading depth from: {depth_path}")
            depth_img = np.load(depth_path)
            
            # Normalize depth for visualization
            depth_min = view["depth_min"]
            depth_max = view["depth_max"]
            depth_scale = view["depth_scale"]
            
            # Get camera parameters
            intrinsic = np.array(self.params["intrinsic_matrix"])
            extrinsic = np.array(view["extrinsic_matrix"])
            
            # Create point cloud
            self.status_var.set("Creating point cloud...")
            pcd = create_point_cloud_from_rgbd(
                color_img, depth_img, intrinsic, extrinsic, depth_scale, depth_min, depth_max
            )
            
            # Apply color mode
            try:
                self.apply_color_mode(pcd)
            except Exception as e:
                print(f"Warning: Could not apply color mode: {e}")
            
            # Update the visualization
            try:
                if self.current_point_cloud is not None:
                    self.vis.remove_geometry(self.current_point_cloud, reset_bounding_box=False)
                
                self.current_point_cloud = pcd
                self.vis.add_geometry(self.current_point_cloud, reset_bounding_box=False)
            except Exception as e:
                print(f"Warning: Error updating geometry: {e}")
                self.status_var.set(f"Error: Could not update visualization - {str(e)}")
                return
            
            # Update camera information display
            camera_pos = view["camera_position"]
            self.camera_pos_label.config(text=f"[{camera_pos[0]:.2f}, {camera_pos[1]:.2f}, {camera_pos[2]:.2f}]")
            resolution = self.params["resolution"]
            self.camera_res_label.config(text=f"{resolution[0]} x {resolution[1]}")
            
            # Update camera frustums if enabled
            if self.show_camera_frustum:
                try:
                    self.update_camera_frustums()
                except Exception as e:
                    print(f"Warning: Could not update camera frustums: {e}")
            
            # Update visualization settings
            try:
                render_option = self.vis.get_render_option()
                render_option.point_size = self.point_size
                render_option.point_show_normal = self.show_normals
            except Exception as e:
                print(f"Warning: Could not update render options: {e}")
            
            # Reset view if it's the first view
            if self.current_view_id == 0:
                try:
                    self.vis.reset_view_point(True)
                except Exception as e:
                    print(f"Warning: Could not reset view: {e}")
            
            try:
                self.vis.poll_events()
                self.vis.update_renderer()
            except Exception as e:
                print(f"Warning: Could not update renderer: {e}")
            
            # Update image previews
            try:
                self.update_image_previews(color_img, depth_img, depth_min, depth_max)
            except Exception as e:
                print(f"Warning: Could not update image previews: {e}")
            
            self.status_var.set(f"Showing View {self.current_view_id} | {len(np.asarray(pcd.points))} points")
        except Exception as e:
            print(f"Error in update_visualization: {e}")
            self.status_var.set(f"Error: {str(e)}")
    
    def apply_color_mode(self, pcd):
        """Apply the selected color mode to the point cloud"""
        points = np.asarray(pcd.points)
        
        if self.color_mode == "RGB":
            # RGB colors are already set from the image
            pass
        elif self.color_mode == "Height":
            # Color by height (Y-coordinate)
            colors = np.zeros_like(points)
            min_y = np.min(points[:, 1])
            max_y = np.max(points[:, 1])
            norm_y = (points[:, 1] - min_y) / (max_y - min_y + 1e-8)
            
            # Create a colormap
            jet_colors = plt.cm.jet(norm_y)[:, :3]  # Remove alpha channel
            pcd.colors = o3d.utility.Vector3dVector(jet_colors)
        elif self.color_mode == "Normal":
            # Color by normal direction
            normals = np.asarray(pcd.normals)
            # Normalize normals to range [0, 1]
            colors = (normals + 1) / 2
            pcd.colors = o3d.utility.Vector3dVector(colors)
    
    def update_render_options(self, event=None):
        """Update rendering options"""
        self.point_size = self.point_size_var.get()
        render_option = self.vis.get_render_option()
        render_option.point_size = self.point_size
        self.vis.poll_events()
        self.vis.update_renderer()
    
    def toggle_coordinate_frame(self):
        """Toggle the coordinate frame visibility"""
        self.show_coordinate_frame = self.coord_frame_var.get()
        if self.show_coordinate_frame:
            # Add coordinate frame if not already added
            self.vis.add_geometry(self.coord_frame, reset_bounding_box=False)
        else:
            # Remove coordinate frame if it exists
            self.vis.remove_geometry(self.coord_frame, reset_bounding_box=False)
        
        render_option = self.vis.get_render_option()
        render_option.show_coordinate_frame = self.show_coordinate_frame
        self.vis.poll_events()
        self.vis.update_renderer()
    
    def toggle_camera_frustums(self):
        """Toggle camera frustum visualization"""
        self.show_camera_frustum = self.camera_frustum_var.get()
        
        # Remove existing frustums
        for frustum in self.camera_frustums:
            self.vis.remove_geometry(frustum, reset_bounding_box=False)
        
        self.camera_frustums = []
        
        if self.show_camera_frustum:
            self.update_camera_frustums()
        
        self.vis.poll_events()
        self.vis.update_renderer()
    
    def update_camera_frustums(self):
        """Update camera frustums visualization"""
        # Remove existing frustums
        for frustum in self.camera_frustums:
            self.vis.remove_geometry(frustum, reset_bounding_box=False)
        
        self.camera_frustums = []
        
        if not self.show_camera_frustum:
            return
        
        # Get intrinsic parameters
        intrinsic = np.array(self.params["intrinsic_matrix"])
        resolution = self.params["resolution"]
        
        # Create a camera frustum for each view
        for view in self.params["views"]:
            extrinsic = np.array(view["extrinsic_matrix"])
            
            # Create a simple frustum visualization
            frustum = o3d.geometry.LineSet()
            
            # Define frustum corners in camera space
            frustum_points = np.array([
                [0, 0, 0],  # Camera center
                # Near plane corners
                [-0.5, -0.3, 1],
                [0.5, -0.3, 1],
                [0.5, 0.3, 1],
                [-0.5, 0.3, 1]
            ])
            
            # Define lines
            frustum_lines = np.array([
                [0, 1], [0, 2], [0, 3], [0, 4],  # Lines from camera to corners
                [1, 2], [2, 3], [3, 4], [4, 1]   # Lines connecting corners
            ])
            
            # Transform points to world space
            transform = np.linalg.inv(extrinsic)
            homogeneous_points = np.hstack((frustum_points, np.ones((frustum_points.shape[0], 1))))
            transformed_points = np.dot(homogeneous_points, transform.T)[:, :3]
            
            # Create line set with colors
            frustum.points = o3d.utility.Vector3dVector(transformed_points)
            frustum.lines = o3d.utility.Vector2iVector(frustum_lines)
            
            # Set colors - make the current view's frustum cyan, others white
            if view["view_id"] == self.current_view_id:
                colors = np.ones((len(frustum_lines), 3)) * np.array([0, 1, 1])  # Cyan
            else:
                colors = np.ones((len(frustum_lines), 3)) * np.array([1, 1, 1])  # White
            
            frustum.colors = o3d.utility.Vector3dVector(colors)
            
            # Add to the list and visualizer
            self.camera_frustums.append(frustum)
            self.vis.add_geometry(frustum, reset_bounding_box=False)
    
    def toggle_normals(self):
        """Toggle normal vector visualization"""
        self.show_normals = self.normals_var.get()
        render_option = self.vis.get_render_option()
        render_option.point_show_normal = self.show_normals
        self.vis.poll_events()
        self.vis.update_renderer()
    
    def update_color_mode(self, event=None):
        """Update the color mode of the point cloud"""
        self.color_mode = self.color_mode_var.get()
        if self.current_point_cloud is not None:
            self.apply_color_mode(self.current_point_cloud)
            self.vis.update_geometry(self.current_point_cloud)
            self.vis.poll_events()
            self.vis.update_renderer()
    
    def update_image_previews(self, color_img, depth_img, depth_min, depth_max):
        """Update the image preview panels"""
        # Resize images for display
        display_width = 200
        aspect_ratio = color_img.shape[1] / color_img.shape[0]
        display_height = int(display_width / aspect_ratio)
        
        # Resize color image
        color_display = cv2.resize(color_img, (display_width, display_height))
        
        # Normalize and resize depth image for display
        depth_normalized = (depth_img - depth_min) / (depth_max - depth_min)
        depth_display = (depth_normalized * 255).astype(np.uint8)
        depth_display = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)
        depth_display = cv2.resize(depth_display, (display_width, display_height))
        
        # Convert to PIL images for Tkinter
        color_pil = Image.fromarray(color_display)
        depth_pil = Image.fromarray(depth_display)
        
        # Convert to Tkinter compatible images
        self.color_tk = ImageTk.PhotoImage(image=color_pil)
        self.depth_tk = ImageTk.PhotoImage(image=depth_pil)
        
        # Update labels
        self.image_label.configure(image=self.color_tk)
        self.depth_label.configure(image=self.depth_tk)
    
    def on_view_selected(self, event):
        """Handle view selection from dropdown"""
        selected_view = self.view_dropdown.current()
        if selected_view != self.current_view_id:
            self.current_view_id = selected_view
            self.status_var.set(f"Loading View {self.current_view_id}...")
            self.update_visualization()
    
    def change_view(self, direction):
        """Change to next/previous view"""
        num_views = len(self.params["views"])
        new_view_id = (self.current_view_id + direction) % num_views
        self.current_view_id = new_view_id
        self.view_dropdown.current(new_view_id)
        self.status_var.set(f"Loading View {self.current_view_id}...")
        self.update_visualization()
    
    def reset_view(self):
        """Reset the view to the default perspective"""
        self.vis.reset_view_point(True)
        self.vis.poll_events()
        self.vis.update_renderer()
        self.status_var.set("View reset")
    
    def take_screenshot(self):
        """Capture a screenshot of the current view"""
        base_dir = os.path.dirname(self.params_path)
        screenshot_dir = os.path.join(base_dir, "screenshots")
        os.makedirs(screenshot_dir, exist_ok=True)
        
        # Create a timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        screenshot_path = os.path.join(screenshot_dir, f"view_{self.current_view_id}_{timestamp}.png")
        
        # Capture image
        self.vis.capture_screen_image(screenshot_path, True)
        self.status_var.set(f"Screenshot saved to {screenshot_path}")
    
    def run(self):
        """Run the viewer application"""
        # Create an update function for refreshing Open3D visualization
        def update():
            try:
                self.vis.poll_events()
                self.vis.update_renderer()
                self.root.after(50, update)  # Schedule next update in 50 ms
            except Exception as e:
                print(f"Error during visualization update: {e}")
                # Still try to continue the update loop
                self.root.after(50, update)
        
        # Start the update loop
        self.root.after(50, update)
        
        # Set up a handler for window closing
        def on_closing():
            print("Closing application...")
            try:
                self.vis.destroy_window()
            except:
                pass
            self.root.destroy()
        
        self.root.protocol("WM_DELETE_WINDOW", on_closing)
        
        # Start Tkinter main loop
        self.root.mainloop()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="RGB-D Point Cloud Viewer")
    parser.add_argument("dir_path", help="Directory containing camera_parameters.json")
    args = parser.parse_args()
    
    # Check if the directory exists
    if not os.path.isdir(args.dir_path):
        print(f"Error: Directory not found: {args.dir_path}")
        sys.exit(1)
    
    # Check if camera_parameters.json exists
    params_path = os.path.join(args.dir_path, "camera_parameters.json")
    if not os.path.isfile(params_path):
        print(f"Error: camera_parameters.json not found in {args.dir_path}")
        sys.exit(1)
    
    try:
        # Create Tkinter root
        root = tk.Tk()
        
        # Create and run the viewer
        viewer = RGBDPointCloudViewer(root, args.dir_path)
        viewer.run()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 