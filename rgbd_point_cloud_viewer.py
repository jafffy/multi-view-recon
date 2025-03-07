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
    print(f"Depth image shape: {depth_img.shape}, min: {np.min(depth_img)}, max: {np.max(depth_img)}")
    print(f"Color image shape: {color_img.shape}")
    
    # Normalize depth to the range 0-1
    depth_normalized = (depth_img - depth_min) / (depth_max - depth_min)
    
    # Convert to uint16 (0-65535) for Open3D
    depth_o3d_ready = (depth_normalized * 1000).astype(np.uint16)
    
    # Log min/max values to verify scaling
    print(f"Normalized depth min: {np.min(depth_normalized)}, max: {np.max(depth_normalized)}")
    print(f"Scaled depth min: {np.min(depth_o3d_ready)}, max: {np.max(depth_o3d_ready)}")
    
    # Create Open3D images
    color_o3d = o3d.geometry.Image(color_img)
    depth_o3d = o3d.geometry.Image(depth_o3d_ready)
    
    # Create RGBD image
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d, depth_o3d, 
        depth_scale=1000.0,  # Match the scaling factor we used above
        depth_trunc=1.0,     # Since depth is normalized to 0-1
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
    
    # Create point cloud (using identity matrix first, then transform)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd, intrinsic_o3d
    )
    
    # Apply the extrinsic transformation - we take inverse since we're going from camera to world
    transformation = np.linalg.inv(extrinsic)
    pcd.transform(transformation)
    
    # Check if point cloud is empty
    if len(pcd.points) == 0:
        print("Warning: Generated point cloud is empty. Skipping processing.")
        return pcd
    
    print(f"Created point cloud with {len(pcd.points)} points")
        
    # Apply statistical outlier removal - with try/except in case this fails
    try:
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        print(f"After outlier removal: {len(pcd.points)} points")
    except Exception as e:
        print(f"Warning: Could not apply statistical outlier removal: {e}")
    
    # Estimate normals for better rendering - with try/except in case this fails
    try:
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        # Only try to orient normals if the estimation was successful
        if hasattr(pcd, 'normals') and len(pcd.normals) > 0:
            pcd.orient_normals_towards_camera_location(camera_location=np.array([0, 0, 0]))
    except Exception as e:
        print(f"Warning: Could not estimate normals: {e}")
    
    # Apply voxel downsampling if point cloud is very dense - with try/except in case this fails
    try:
        if len(pcd.points) > 500000:
            pcd = pcd.voxel_down_sample(voxel_size=0.01)
            print(f"After downsampling: {len(pcd.points)} points")
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
        self.point_size = 5.0  # Increase default point size further
        self.show_coordinate_frame = True
        self.show_camera_frustum = False
        self.show_normals = False
        self.color_mode = "RGB"  # RGB, Height, Normal
        self.depth_scale_factor = 0.01  # Use a much smaller scale factor for better visualization
        self.show_all_point_clouds = False  # Whether to show all point clouds simultaneously
        self.debug_mode = True  # Enable debug mode
        
        # Current status
        self.status_message = "Starting up..."
        
        # Create GUI
        self.create_gui()
        
        # Current view
        self.current_view_id = 0
        self.current_point_cloud = None
        self.all_point_clouds = {}  # Store point clouds for each view
        self.visible_geometries = set()  # Track which geometries are currently visible
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
        
        # Show all point clouds checkbox
        self.show_all_clouds_var = tk.BooleanVar(value=self.show_all_point_clouds)
        show_all_cb = ttk.Checkbutton(control_frame, text="Show All Views", 
                                    variable=self.show_all_clouds_var, command=self.toggle_all_point_clouds)
        show_all_cb.grid(row=1, column=0, columnspan=2, sticky=tk.W, padx=5, pady=5)
        
        # Debug mode checkbox
        self.debug_mode_var = tk.BooleanVar(value=self.debug_mode)
        debug_cb = ttk.Checkbutton(control_frame, text="Debug Mode", 
                                 variable=self.debug_mode_var, command=self.toggle_debug_mode)
        debug_cb.grid(row=2, column=0, columnspan=2, sticky=tk.W, padx=5, pady=5)
        
        # Visualization options
        viz_frame = ttk.LabelFrame(right_panel, text="Visualization Options", padding="10")
        viz_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        # Point size slider
        ttk.Label(viz_frame, text="Point Size:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.point_size_var = tk.DoubleVar(value=self.point_size)
        point_size_slider = ttk.Scale(viz_frame, from_=1.0, to=10.0, orient=tk.HORIZONTAL, 
                                    variable=self.point_size_var, length=150)
        point_size_slider.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        point_size_slider.bind("<ButtonRelease-1>", self.update_render_options)
        
        # Depth scale factor slider
        ttk.Label(viz_frame, text="Depth Scale:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.depth_scale_var = tk.DoubleVar(value=self.depth_scale_factor)
        depth_scale_slider = ttk.Scale(viz_frame, from_=0.001, to=0.1, orient=tk.HORIZONTAL, 
                                     variable=self.depth_scale_var, length=150)
        depth_scale_slider.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        depth_scale_slider.bind("<ButtonRelease-1>", self.update_depth_scale)
        
        # Show coordinate frame
        self.coord_frame_var = tk.BooleanVar(value=self.show_coordinate_frame)
        coord_frame_cb = ttk.Checkbutton(viz_frame, text="Show Coordinate Frame", 
                                      variable=self.coord_frame_var, command=self.toggle_coordinate_frame)
        coord_frame_cb.grid(row=2, column=0, columnspan=2, sticky=tk.W, padx=5, pady=5)
        
        # Show camera frustum
        self.camera_frustum_var = tk.BooleanVar(value=self.show_camera_frustum)
        camera_frustum_cb = ttk.Checkbutton(viz_frame, text="Show Camera Frustums", 
                                          variable=self.camera_frustum_var, command=self.toggle_camera_frustums)
        camera_frustum_cb.grid(row=3, column=0, columnspan=2, sticky=tk.W, padx=5, pady=5)
        
        # Show normals
        self.normals_var = tk.BooleanVar(value=self.show_normals)
        normals_cb = ttk.Checkbutton(viz_frame, text="Show Normals", 
                                    variable=self.normals_var, command=self.toggle_normals)
        normals_cb.grid(row=4, column=0, columnspan=2, sticky=tk.W, padx=5, pady=5)
        
        # Color mode
        ttk.Label(viz_frame, text="Color Mode:").grid(row=5, column=0, sticky=tk.W, padx=5, pady=5)
        
        color_modes = ["RGB", "Height", "Normal"]
        self.color_mode_var = tk.StringVar(value=self.color_mode)
        color_mode_dropdown = ttk.Combobox(viz_frame, textvariable=self.color_mode_var, 
                                         values=color_modes, width=10, state="readonly")
        color_mode_dropdown.grid(row=5, column=1, sticky=tk.W, padx=5, pady=5)
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
        
        # Depth range
        ttk.Label(camera_info_frame, text="Depth Range:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.depth_range_label = ttk.Label(camera_info_frame, text="0 - 0")
        self.depth_range_label.grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Control buttons frame
        control_buttons_frame = ttk.Frame(right_panel, padding="10")
        control_buttons_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        # Reset view button
        reset_view_btn = ttk.Button(control_buttons_frame, text="Reset View", command=self.reset_view)
        reset_view_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Reload button
        reload_btn = ttk.Button(control_buttons_frame, text="Reload", command=self.reload_current_view)
        reload_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Capture screenshot button
        screenshot_btn = ttk.Button(control_buttons_frame, text="Screenshot", command=self.take_screenshot)
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
        
        # Debug information frame - only visible in debug mode
        self.debug_frame = ttk.LabelFrame(right_panel, text="Debug Information", padding="10")
        if self.debug_mode:
            self.debug_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        # Debug text box
        self.debug_text = tk.Text(self.debug_frame, height=10, width=40, wrap=tk.WORD)
        self.debug_text.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.debug_text.insert(tk.END, "Debug information will appear here...")
        
        # Status bar
        self.status_var = tk.StringVar(value=self.status_message)
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
        
        # Add more lights to improve visibility
        try:
            # Set lighting to better show 3D structure
            render_option.light_on = True
            
            # Add stronger ambient light
            if hasattr(render_option, 'ambient_light'):
                render_option.ambient_light = np.array([0.4, 0.4, 0.4])
            
            # Increase reflection parameters if available
            if hasattr(render_option, 'shininess'):
                render_option.shininess = 100.0
        except Exception as e:
            print(f"Warning: Could not set advanced lighting: {e}")
        
        # Configure the camera view for better initial visualization
        view_control = self.vis.get_view_control()
        view_control.set_zoom(0.5)  # Reduce zoom to see more of the point cloud
        
        # Add coordinate frame
        self.coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=20.0)  # Make coordinate frame larger
        if self.show_coordinate_frame:
            self.vis.add_geometry(self.coord_frame)
            self.visible_geometries.add(self.coord_frame)
        
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
            
            # Clear debug text if in debug mode
            if self.debug_mode:
                self.debug_text.delete(1.0, tk.END)
                self.log_debug(f"Processing view {self.current_view_id}")
                self.log_debug(f"Image path: {img_path}")
                self.log_debug(f"Depth path: {depth_path}")
                self.log_debug(f"Current depth scale factor: {self.depth_scale_factor}")
            
            # Check if we already have this point cloud cached
            if self.current_view_id in self.all_point_clouds and not self.show_all_point_clouds:
                # Use cached point cloud
                self.status_var.set(f"Using cached point cloud for View {self.current_view_id}")
                pcd = self.all_point_clouds[self.current_view_id]
            else:
                # Load and process point cloud
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
                
                # Update depth range label
                self.depth_range_label.config(text=f"{depth_min:.2f} - {depth_max:.2f}")
                
                if self.debug_mode:
                    self.log_debug(f"Depth min: {depth_min}, max: {depth_max}, scale: {depth_scale}")
                    self.log_debug(f"Depth img min: {np.min(depth_img)}, max: {np.max(depth_img)}")
                
                # Get camera parameters
                intrinsic = np.array(self.params["intrinsic_matrix"])
                extrinsic = np.array(view["extrinsic_matrix"])
                
                if self.debug_mode:
                    self.log_debug(f"Intrinsic matrix: {intrinsic}")
                    self.log_debug(f"Extrinsic matrix: {extrinsic}")
                
                # Create point cloud
                self.status_var.set("Creating point cloud...")
                
                # Create point cloud with current settings
                pcd = create_point_cloud_from_rgbd(
                    color_img, depth_img, intrinsic, extrinsic, 
                    depth_scale, depth_min, depth_max
                )
                
                if self.debug_mode and pcd and len(pcd.points) > 0:
                    # Get point cloud statistics
                    points = np.asarray(pcd.points)
                    self.log_debug(f"Point cloud points: {len(pcd.points)}")
                    self.log_debug(f"Point cloud bounds X: {np.min(points[:, 0]):.2f} to {np.max(points[:, 0]):.2f}")
                    self.log_debug(f"Point cloud bounds Y: {np.min(points[:, 1]):.2f} to {np.max(points[:, 1]):.2f}")
                    self.log_debug(f"Point cloud bounds Z: {np.min(points[:, 2]):.2f} to {np.max(points[:, 2]):.2f}")
                
                # Cache the point cloud
                self.all_point_clouds[self.current_view_id] = pcd
                
                # Update image previews
                try:
                    self.update_image_previews(color_img, depth_img, depth_min, depth_max)
                except Exception as e:
                    print(f"Warning: Could not update image previews: {e}")
                    if self.debug_mode:
                        self.log_debug(f"Error updating previews: {e}")
            
            # Apply color mode
            try:
                self.apply_color_mode(pcd)
            except Exception as e:
                print(f"Warning: Could not apply color mode: {e}")
            
            # Update the visualization
            try:
                # Remove all geometries except the coordinate frame
                for geometry in list(self.visible_geometries):
                    if geometry != self.coord_frame:
                        self.vis.remove_geometry(geometry, reset_bounding_box=False)
                self.visible_geometries = set()
                if self.coord_frame and self.show_coordinate_frame:
                    self.visible_geometries.add(self.coord_frame)
                
                # Add current point cloud
                if not self.show_all_point_clouds:
                    self.vis.add_geometry(pcd, reset_bounding_box=False)
                    self.visible_geometries.add(pcd)
                    self.current_point_cloud = pcd
                else:
                    # Add all point clouds when in "show all" mode
                    for view_id, point_cloud in self.all_point_clouds.items():
                        self.vis.add_geometry(point_cloud, reset_bounding_box=False)
                        self.visible_geometries.add(point_cloud)
                    self.current_point_cloud = pcd
                
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
            
            # Reset view if it's the first view and not showing all
            if self.current_view_id == 0 and not self.show_all_point_clouds:
                try:
                    self.vis.reset_view_point(True)
                except Exception as e:
                    print(f"Warning: Could not reset view: {e}")
            
            try:
                self.vis.poll_events()
                self.vis.update_renderer()
            except Exception as e:
                print(f"Warning: Could not update renderer: {e}")
            
            num_points = len(np.asarray(pcd.points)) if pcd else 0
            if self.show_all_point_clouds:
                total_points = sum([len(np.asarray(pc.points)) for pc in self.all_point_clouds.values()])
                self.status_var.set(f"Showing All Views | Total points: {total_points}")
            else:
                self.status_var.set(f"Showing View {self.current_view_id} | {num_points} points")
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
            if self.coord_frame not in self.visible_geometries:
                self.vis.add_geometry(self.coord_frame, reset_bounding_box=False)
                self.visible_geometries.add(self.coord_frame)
        else:
            # Remove coordinate frame if it exists
            if self.coord_frame in self.visible_geometries:
                self.vis.remove_geometry(self.coord_frame, reset_bounding_box=False)
                self.visible_geometries.remove(self.coord_frame)
        
        render_option = self.vis.get_render_option()
        render_option.show_coordinate_frame = self.show_coordinate_frame
        self.vis.poll_events()
        self.vis.update_renderer()
    
    def toggle_camera_frustums(self):
        """Toggle camera frustum visualization"""
        self.show_camera_frustum = self.camera_frustum_var.get()
        
        # Remove existing frustums
        for frustum in self.camera_frustums:
            if frustum in self.visible_geometries:
                self.vis.remove_geometry(frustum, reset_bounding_box=False)
                self.visible_geometries.remove(frustum)
        
        self.camera_frustums = []
        
        if self.show_camera_frustum:
            self.update_camera_frustums()
        
        self.vis.poll_events()
        self.vis.update_renderer()
    
    def update_camera_frustums(self):
        """Update camera frustums visualization"""
        # Remove existing frustums
        for frustum in self.camera_frustums:
            if frustum in self.visible_geometries:
                self.vis.remove_geometry(frustum, reset_bounding_box=False)
                self.visible_geometries.remove(frustum)
        
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
            
            # Define frustum corners in camera space - make larger for better visibility
            scale = 30.0  # Scale up the frustum size significantly
            frustum_points = np.array([
                [0, 0, 0],  # Camera center
                # Near plane corners
                [-0.5 * scale, -0.3 * scale, 1.0 * scale],
                [0.5 * scale, -0.3 * scale, 1.0 * scale],
                [0.5 * scale, 0.3 * scale, 1.0 * scale],
                [-0.5 * scale, 0.3 * scale, 1.0 * scale]
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
            self.visible_geometries.add(frustum)
    
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
    
    def toggle_all_point_clouds(self):
        """Toggle showing all point clouds simultaneously"""
        self.show_all_point_clouds = self.show_all_clouds_var.get()
        self.update_visualization()
    
    def update_depth_scale(self, event=None):
        """Update depth scale factor and regenerate point clouds"""
        # Get the new depth scale factor
        new_scale = self.depth_scale_var.get()
        if new_scale != self.depth_scale_factor:
            self.depth_scale_factor = new_scale
            self.status_var.set(f"Updating depth scale to {self.depth_scale_factor:.2f}...")
            
            # Clear cached point clouds to force regeneration
            # First, remove them from the visualization
            try:
                for point_cloud in self.all_point_clouds.values():
                    if point_cloud in self.visible_geometries:
                        self.vis.remove_geometry(point_cloud, reset_bounding_box=False)
                        self.visible_geometries.remove(point_cloud)
            except Exception as e:
                print(f"Warning: Error removing point clouds: {e}")
            
            # Clear the cache
            self.all_point_clouds = {}
            
            # Update the visualization
            self.update_visualization()
    
    def toggle_debug_mode(self):
        """Toggle debug mode on/off"""
        self.debug_mode = self.debug_mode_var.get()
        
        # Show or hide the debug frame
        if self.debug_mode:
            self.debug_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5, after=self.preview_frame)
        else:
            self.debug_frame.pack_forget()
    
    def log_debug(self, message):
        """Log a debug message to the debug text box"""
        if self.debug_mode:
            self.debug_text.insert(tk.END, f"{message}\n")
            self.debug_text.see(tk.END)  # Scroll to the bottom
    
    def reload_current_view(self):
        """Reload the current view by clearing the cache and updating"""
        if self.current_view_id in self.all_point_clouds:
            # Remove the current point cloud from visualization
            current_pcd = self.all_point_clouds[self.current_view_id]
            if current_pcd in self.visible_geometries:
                self.vis.remove_geometry(current_pcd, reset_bounding_box=False)
                self.visible_geometries.remove(current_pcd)
            
            # Remove from cache
            del self.all_point_clouds[self.current_view_id]
        
        # Update visualization
        self.status_var.set(f"Reloading View {self.current_view_id}...")
        self.update_visualization()
    
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