#!/usr/bin/env python
import os
import sys
import numpy as np
import json
import argparse
import open3d as o3d
import matplotlib.pyplot as plt
from PIL import Image

class CameraViewVisualizer:
    def __init__(self, camera_params_path):
        """
        Initialize the camera view visualizer
        
        Args:
            camera_params_path (str): Path to camera parameters JSON file
        """
        self.camera_params_path = camera_params_path
        self.camera_params = None
        self.point_cloud = None
        self.current_view_idx = 0
        self.load_camera_parameters()
        
    def load_camera_parameters(self):
        """
        Load camera parameters from a JSON file
        """
        try:
            with open(self.camera_params_path, 'r') as f:
                self.camera_params = json.load(f)
                
            print(f"Loaded camera parameters from: {self.camera_params_path}")
            print(f"Number of views: {len(self.camera_params['views'])}")
            
            # Load the point cloud
            ply_path = self.camera_params.get('point_cloud_path')
            if ply_path:
                if not os.path.exists(ply_path):
                    # Try a relative path based on the JSON location
                    json_dir = os.path.dirname(self.camera_params_path)
                    ply_path_rel = os.path.join(json_dir, os.path.basename(ply_path))
                    if os.path.exists(ply_path_rel):
                        ply_path = ply_path_rel
                    else:
                        print(f"Warning: Point cloud file not found at {ply_path}")
                        print(f"Please enter the correct path to the PLY file:")
                        ply_path = input("> ")
                
                self.load_point_cloud(ply_path)
            else:
                print("No point cloud path found in camera parameters.")
                ply_path = input("Please enter the path to the PLY file: ")
                self.load_point_cloud(ply_path)
                
            return True
        except Exception as e:
            print(f"Error loading camera parameters: {e}")
            return False
            
    def load_point_cloud(self, ply_path):
        """
        Load the point cloud from a PLY file
        
        Args:
            ply_path (str): Path to the PLY file
        """
        try:
            print(f"Loading point cloud: {ply_path}")
            self.point_cloud = o3d.io.read_point_cloud(ply_path)
            
            if not self.point_cloud.has_points():
                print(f"Error: No points found in {ply_path}")
                return False
                
            print(f"Point cloud loaded: {ply_path}")
            print(f"Number of points: {len(self.point_cloud.points)}")
            print(f"Point cloud has colors: {self.point_cloud.has_colors()}")
            print(f"Point cloud has normals: {self.point_cloud.has_normals()}")
            
            # If the point cloud doesn't have normals, estimate them for better visualization
            if not self.point_cloud.has_normals():
                print("Estimating normals for better visualization...")
                self.point_cloud.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
                
            return True
        except Exception as e:
            print(f"Error loading point cloud: {e}")
            return False
    
    def visualize_from_camera_view(self, view_idx=None):
        """
        Visualize the point cloud from a specific camera view
        
        Args:
            view_idx (int, optional): Index of the camera view. If None, uses current view.
        """
        if view_idx is not None:
            self.current_view_idx = view_idx
        
        if not self.point_cloud or not self.camera_params:
            print("Error: Point cloud or camera parameters not loaded")
            return
            
        if self.current_view_idx < 0 or self.current_view_idx >= len(self.camera_params['views']):
            print(f"Error: Invalid view index {self.current_view_idx}")
            return
            
        # Get the current view
        view = self.camera_params['views'][self.current_view_idx]
        print(f"Visualizing from camera view {self.current_view_idx} (ID: {view['view_id']})")
        
        # Create a visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window(
            window_name=f"Camera View {view['view_id']}",
            width=self.camera_params['resolution'][0],
            height=self.camera_params['resolution'][1]
        )
        
        # Add the point cloud
        vis.add_geometry(self.point_cloud)
        
        # Set the view parameters based on the camera
        self._set_view_parameters(vis, view)
        
        # Run the visualizer
        vis.run()
        vis.destroy_window()
        
    def _set_view_parameters(self, vis, view):
        """
        Set the view parameters based on the camera parameters
        
        Args:
            vis (o3d.visualization.Visualizer): Open3D visualizer
            view (dict): Camera view parameters
        """
        # Get the view control
        view_control = vis.get_view_control()
        
        # Get intrinsic parameters
        intrinsic = self.camera_params['intrinsic_matrix']
        fx = intrinsic[0][0]
        fy = intrinsic[1][1]
        cx = intrinsic[0][2]
        cy = intrinsic[1][2]
        
        # Create pinhole camera parameters
        params = o3d.camera.PinholeCameraParameters()
        
        # Set the intrinsic parameters
        params.intrinsic.set_intrinsics(
            width=self.camera_params['resolution'][0],
            height=self.camera_params['resolution'][1],
            fx=fx, fy=fy, cx=cx, cy=cy
        )
        
        # Set the extrinsic parameters
        extrinsic = np.array(view['extrinsic_matrix'])
        params.extrinsic = extrinsic
        
        # Apply to view control
        view_control.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)
        
    def _show_image_reference(self, view_idx):
        """
        Show the reference image for comparison
        
        Args:
            view_idx (int): Index of the camera view
        """
        view = self.camera_params['views'][view_idx]
        image_path = view.get('image_path')
        
        if image_path and os.path.exists(image_path):
            img = Image.open(image_path)
            plt.figure(figsize=(10, 6))
            plt.imshow(img)
            plt.title(f"Reference Image - View {view['view_id']}")
            plt.axis('off')
            plt.show(block=False)
        else:
            print(f"Reference image not found: {image_path}")
    
    def visualize_all_views_interactively(self):
        """
        Interactively visualize all camera views, with controls to switch between views
        """
        if not self.point_cloud or not self.camera_params:
            print("Error: Point cloud or camera parameters not loaded")
            return
            
        view_count = len(self.camera_params['views'])
        print(f"Interactive view mode. Total views: {view_count}")
        print("Controls:")
        print("  - Press 'N' for next view")
        print("  - Press 'P' for previous view")
        print("  - Press 'I' to show corresponding camera image")
        print("  - Press 'Q' to quit")
        
        # Create a visualizer
        vis = o3d.visualization.VisualizerWithKeyCallback()
        running = True
        
        # Register key callback functions
        def next_view(vis):
            self.current_view_idx = (self.current_view_idx + 1) % view_count
            print(f"Switching to view {self.current_view_idx}")
            self._set_view_parameters(vis, self.camera_params['views'][self.current_view_idx])
            return True
            
        def prev_view(vis):
            self.current_view_idx = (self.current_view_idx - 1) % view_count
            print(f"Switching to view {self.current_view_idx}")
            self._set_view_parameters(vis, self.camera_params['views'][self.current_view_idx])
            return True
            
        def show_image(vis):
            self._show_image_reference(self.current_view_idx)
            return True
            
        def quit_vis(vis):
            nonlocal running
            running = False
            return False
        
        # Register the key callbacks
        vis.register_key_callback(ord('N'), next_view)
        vis.register_key_callback(ord('n'), next_view)
        vis.register_key_callback(ord('P'), prev_view)
        vis.register_key_callback(ord('p'), prev_view)
        vis.register_key_callback(ord('I'), show_image)
        vis.register_key_callback(ord('i'), show_image)
        vis.register_key_callback(ord('Q'), quit_vis)
        vis.register_key_callback(ord('q'), quit_vis)
        
        # Create window and add geometry
        vis.create_window(
            window_name="Camera View Visualization",
            width=self.camera_params['resolution'][0],
            height=self.camera_params['resolution'][1]
        )
        vis.add_geometry(self.point_cloud)
        
        # Set initial view
        self._set_view_parameters(vis, self.camera_params['views'][self.current_view_idx])
        
        # Run the visualizer
        vis.run()
        vis.destroy_window()

    def visualize_with_dropdown(self):
        """
        Visualize point cloud with a dropdown menu to select different camera views
        using PyQt5 and Open3D
        """
        try:
            from PyQt5 import QtCore, QtGui, QtWidgets
        except ImportError:
            print("PyQt5 is required but not installed. Please install it with:")
            print("pip install PyQt5")
            return

        if not self.point_cloud or not self.camera_params:
            print("Error: Point cloud or camera parameters not loaded")
            return

        # Create a new Qt Application if one doesn't already exist
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication(sys.argv)

        class CameraViewerWindow(QtWidgets.QMainWindow):
            def __init__(self, parent=None, camera_visualizer=None):
                super(CameraViewerWindow, self).__init__(parent)
                self.camera_visualizer = camera_visualizer
                self.current_view_idx = 0
                self.vis = None
                self.setup_ui()
                self.setup_visualization()

            def setup_ui(self):
                # Set window properties
                self.setWindowTitle("Camera View Selector")
                self.resize(
                    self.camera_visualizer.camera_params['resolution'][0],
                    self.camera_visualizer.camera_params['resolution'][1] + 50
                )

                # Create central widget and layout
                self.central_widget = QtWidgets.QWidget()
                self.setCentralWidget(self.central_widget)
                self.main_layout = QtWidgets.QVBoxLayout(self.central_widget)

                # Create controls layout
                self.controls_layout = QtWidgets.QHBoxLayout()

                # Add camera selector dropdown
                self.camera_label = QtWidgets.QLabel("Select Camera View:")
                self.controls_layout.addWidget(self.camera_label)

                self.camera_combo = QtWidgets.QComboBox()
                for view in self.camera_visualizer.camera_params['views']:
                    self.camera_combo.addItem(f"View {view['view_id']}")
                self.camera_combo.currentIndexChanged.connect(self.on_camera_changed)
                self.controls_layout.addWidget(self.camera_combo)

                # Add spacer
                self.controls_layout.addItem(QtWidgets.QSpacerItem(
                    40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum))

                # Add Show Image button
                self.show_image_button = QtWidgets.QPushButton("Show Reference Image")
                self.show_image_button.clicked.connect(self.on_show_image)
                self.controls_layout.addWidget(self.show_image_button)

                # Add controls to main layout
                self.main_layout.addLayout(self.controls_layout)

                # Add Open3D visualization placeholder
                self.vis_widget = QtWidgets.QWidget()
                self.vis_widget.setSizePolicy(
                    QtWidgets.QSizePolicy.Expanding,
                    QtWidgets.QSizePolicy.Expanding
                )
                self.main_layout.addWidget(self.vis_widget)

            def setup_visualization(self):
                # Create an Open3D visualization widget
                vis = o3d.visualization.Visualizer()
                vis.create_window(
                    width=self.camera_visualizer.camera_params['resolution'][0],
                    height=self.camera_visualizer.camera_params['resolution'][1],
                    visible=False
                )
                
                # Add point cloud
                vis.add_geometry(self.camera_visualizer.point_cloud)
                
                # Set initial view
                self.vis = vis
                self.update_camera_view(0)
                
                # Get the window handle and embed it
                hwnd = self.vis.get_render_window_handle()
                
                # Workaround to render the scene first
                self.vis.poll_events()
                self.vis.update_renderer()
                
                # Schedule rendering
                self.timer = QtCore.QTimer(self)
                self.timer.timeout.connect(self.update_visualization)
                self.timer.start(10)  # Update every 10ms

            def update_visualization(self):
                if self.vis:
                    self.vis.poll_events()
                    self.vis.update_renderer()

            def update_camera_view(self, view_idx):
                if not self.vis:
                    return
                    
                self.current_view_idx = view_idx
                view = self.camera_visualizer.camera_params['views'][view_idx]
                print(f"Switching to view {view_idx} (ID: {view['view_id']})")
                
                # Set the view parameters
                self.camera_visualizer._set_view_parameters(self.vis, view)
                self.vis.poll_events()
                self.vis.update_renderer()

            def on_camera_changed(self, index):
                self.update_camera_view(index)

            def on_show_image(self):
                self.camera_visualizer._show_image_reference(self.current_view_idx)

            def closeEvent(self, event):
                if self.vis:
                    self.vis.destroy_window()
                event.accept()

        # Create and show the window
        try:
            window = CameraViewerWindow(camera_visualizer=self)
            window.show()
            return app.exec_()
        except Exception as e:
            print(f"Error creating visualization window: {e}")
            # Fallback to interactive mode if GUI fails
            print("Falling back to keyboard interactive mode...")
            self.visualize_all_views_interactively()

def main():
    parser = argparse.ArgumentParser(
        description="Visualize a point cloud from camera views defined in a parameters file"
    )
    parser.add_argument(
        "--camera-params", "-c",
        type=str, required=True,
        help="Path to the camera parameters JSON file"
    )
    parser.add_argument(
        "--view-id", "-v",
        type=int, default=0,
        help="ID of the specific view to visualize (default: 0)"
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Enable interactive mode to cycle through all views"
    )
    parser.add_argument(
        "--dropdown", "-d",
        action="store_true",
        help="Enable visualization with dropdown menu for view selection"
    )
    
    args = parser.parse_args()
    
    # Create visualizer
    visualizer = CameraViewVisualizer(args.camera_params)
    
    # Choose visualization mode
    if args.dropdown:
        visualizer.visualize_with_dropdown()
    elif args.interactive:
        visualizer.visualize_all_views_interactively()
    else:
        visualizer.visualize_from_camera_view(args.view_id)

if __name__ == "__main__":
    main() 