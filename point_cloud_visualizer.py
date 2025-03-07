#!/usr/bin/env python
import os
import numpy as np
import json
import argparse
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Import the point cloud processing functionality 
from point_cloud_utils import PointCloudProcessor

class PointCloudVisualizer:
    def __init__(self, ply_path=None, camera_params_path=None):
        """
        Initialize the point cloud visualizer
        
        Args:
            ply_path (str, optional): Path to the PLY file
            camera_params_path (str, optional): Path to camera parameters JSON file
        """
        self.ply_path = ply_path
        self.camera_params_path = camera_params_path
        self.camera_params = None
        self.processor = None
        
        if ply_path:
            self.processor = PointCloudProcessor(ply_path)
            
        if camera_params_path:
            self.load_camera_parameters(camera_params_path)
    
    def load_camera_parameters(self, camera_params_path):
        """
        Load camera parameters from a JSON file
        
        Args:
            camera_params_path (str): Path to camera parameters JSON file
        """
        if not os.path.exists(camera_params_path):
            print(f"Error: Camera parameters file not found: {camera_params_path}")
            return False
            
        try:
            with open(camera_params_path, 'r') as f:
                self.camera_params = json.load(f)
                
            # If the PLY path is in the parameters but not provided to constructor,
            # use it to load the point cloud
            if not self.ply_path and 'point_cloud_path' in self.camera_params:
                self.ply_path = self.camera_params['point_cloud_path']
                if os.path.exists(self.ply_path):
                    self.processor = PointCloudProcessor(self.ply_path)
                else:
                    print(f"Warning: Point cloud file not found: {self.ply_path}")
                    
            print(f"Loaded camera parameters from: {camera_params_path}")
            print(f"Number of views: {len(self.camera_params['views'])}")
            return True
        except Exception as e:
            print(f"Error loading camera parameters: {e}")
            return False
    
    def visualize_point_cloud_standalone(self):
        """
        Visualize the point cloud using Open3D's built-in visualizer
        """
        if not self.processor or not self.processor.point_cloud:
            print("Error: No point cloud loaded")
            return
            
        print("Visualizing point cloud (close window to continue)...")
        o3d.visualization.draw_geometries([self.processor.point_cloud], 
                                        window_name="Point Cloud Visualization",
                                        width=1024, 
                                        height=768,
                                        point_show_normal=False)
    
    def visualize_cameras_and_point_cloud(self):
        """
        Create a comprehensive visualization of cameras and point cloud
        """
        if not self.processor or not self.processor.point_cloud:
            print("Error: No point cloud loaded")
            return
            
        if not self.camera_params:
            print("Error: No camera parameters loaded")
            return
        
        # Create a visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=1024, height=768, window_name="Point Cloud and Cameras")
        
        # Add the point cloud
        vis.add_geometry(self.processor.point_cloud)
        
        # Create camera frustums
        camera_geometries = self._create_camera_frustums()
        for geom in camera_geometries:
            vis.add_geometry(geom)
        
        # Add connecting lines between adjacent cameras
        path_lines = self._create_camera_path()
        vis.add_geometry(path_lines)
        
        # Add orientation axes for each camera
        camera_axes = self._create_camera_orientation_axes()
        for axis in camera_axes:
            vis.add_geometry(axis)
        
        # Add bounding box for the point cloud
        bounding_box = self._create_point_cloud_bounding_box()
        vis.add_geometry(bounding_box)
        
        # Set view to show everything
        vis.get_render_option().point_size = 2.0
        vis.get_render_option().background_color = np.array([0.1, 0.1, 0.1])
        vis.get_render_option().line_width = 2.0
        
        # Run the visualizer
        vis.run()
        vis.destroy_window()
    
    def _create_camera_frustums(self):
        """
        Create visual representations of camera frustums
        
        Returns:
            list: List of camera frustum geometries
        """
        camera_geometries = []
        
        resolution = self.camera_params['resolution']
        intrinsic = np.array(self.camera_params['intrinsic_matrix'])
        
        # Camera frustum dimensions
        frustum_scale = self.camera_params['radius'] * 0.15  # Size of frustum visualization
        
        for i, view in enumerate(self.camera_params['views']):
            # Extract camera position and extrinsic matrix
            camera_pos = np.array(view['camera_position'])
            extrinsic = np.array(view['extrinsic_matrix'])
            
            # Create a camera frustum mesh
            frustum = o3d.geometry.TriangleMesh.create_cone(radius=frustum_scale*0.3, height=frustum_scale)
            
            # Rotate and position the frustum to match camera orientation
            # The cone's default orientation is along the Z axis, so we need to rotate it
            R = extrinsic[:3, :3].T  # Transpose because Open3D uses column-major
            t = camera_pos
            
            # Rotate 180 degrees about X to point in the right direction
            Rx = np.array([
                [1, 0, 0],
                [0, -1, 0],
                [0, 0, -1]
            ])
            R = R @ Rx
            
            # Apply the transformation
            frustum.rotate(R, center=(0, 0, 0))
            frustum.translate(t)
            
            # Color the frustum based on view index (for visual distinction)
            # Use a color gradient to show camera order
            ratio = i / max(1, len(self.camera_params['views']) - 1)
            color = plt.cm.viridis(ratio)[:3]  # Get RGB from matplotlib colormap
            frustum.paint_uniform_color(color)
            
            camera_geometries.append(frustum)
            
            # Add a small sphere at camera position
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=frustum_scale*0.05)
            sphere.translate(t)
            sphere.paint_uniform_color([1, 0.7, 0])  # Yellowish
            camera_geometries.append(sphere)
            
            # Add a clear direction indicator (arrow from camera to center)
            # Get the center of the point cloud
            center = np.array(self.camera_params['center'])
            
            # Calculate the direction vector from camera to center
            direction = center - camera_pos
            direction = direction / np.linalg.norm(direction) * (frustum_scale * 0.7)
            
            # Create a cylinder to represent the direction arrow
            cylinder = o3d.geometry.TriangleMesh.create_cylinder(
                radius=frustum_scale*0.02, 
                height=np.linalg.norm(direction)
            )
            
            # Rotate cylinder to point from camera to center
            # Calculate rotation to align cylinder with direction
            z_axis = np.array([0, 0, 1])  # Default cylinder orientation
            if np.allclose(z_axis, direction / np.linalg.norm(direction), rtol=1e-5):
                rotation_axis = np.array([1, 0, 0])
                rotation_angle = 0
            else:
                rotation_axis = np.cross(z_axis, direction / np.linalg.norm(direction))
                rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
                cos_angle = np.dot(z_axis, direction / np.linalg.norm(direction))
                rotation_angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
            
            # Create rotation matrix for cylinder alignment
            if rotation_angle != 0:
                cylinder_R = cylinder.get_rotation_matrix_from_axis_angle(rotation_axis * rotation_angle)
                cylinder.rotate(cylinder_R, center=(0, 0, 0))
            
            # Position cylinder at camera position
            cylinder.translate(camera_pos)
            
            # Make the cylinder red to clearly indicate looking direction
            cylinder.paint_uniform_color([1, 0, 0])
            
            camera_geometries.append(cylinder)
            
            # Add cone at the end of the arrow
            arrowhead = o3d.geometry.TriangleMesh.create_cone(
                radius=frustum_scale*0.05, 
                height=frustum_scale*0.1
            )
            
            # Position arrowhead at the end of cylinder
            arrowhead_pos = camera_pos + direction
            
            # Rotate arrowhead to point in the direction
            if rotation_angle != 0:
                arrowhead.rotate(cylinder_R, center=(0, 0, 0))
            
            # Translate to position
            arrowhead.translate(arrowhead_pos)
            
            # Make arrowhead red
            arrowhead.paint_uniform_color([1, 0, 0])
            
            camera_geometries.append(arrowhead)
        
        return camera_geometries
    
    def _create_camera_orientation_axes(self):
        """
        Create coordinate axes to show camera orientation for each camera
        
        Returns:
            list: List of line set geometries showing orientation axes
        """
        orientation_axes = []
        
        # Axis length as a proportion of the scene radius
        axis_length = self.camera_params['radius'] * 0.1
        
        for view in self.camera_params['views']:
            # Extract camera position and extrinsic matrix
            camera_pos = np.array(view['camera_position'])
            extrinsic = np.array(view['extrinsic_matrix'])
            
            # Extract rotation matrix (transpose because Open3D uses column-major)
            R = extrinsic[:3, :3].T
            
            # Create axes endpoints
            origin = camera_pos
            x_end = origin + R[:, 0] * axis_length  # X axis - red
            y_end = origin + R[:, 1] * axis_length  # Y axis - green
            z_end = origin + R[:, 2] * axis_length  # Z axis - blue
            
            # Create line set for X axis
            x_axis = o3d.geometry.LineSet()
            x_axis.points = o3d.utility.Vector3dVector([origin, x_end])
            x_axis.lines = o3d.utility.Vector2iVector([[0, 1]])
            x_axis.colors = o3d.utility.Vector3dVector([[1, 0, 0]])  # Red
            orientation_axes.append(x_axis)
            
            # Create line set for Y axis
            y_axis = o3d.geometry.LineSet()
            y_axis.points = o3d.utility.Vector3dVector([origin, y_end])
            y_axis.lines = o3d.utility.Vector2iVector([[0, 1]])
            y_axis.colors = o3d.utility.Vector3dVector([[0, 1, 0]])  # Green
            orientation_axes.append(y_axis)
            
            # Create line set for Z axis
            z_axis = o3d.geometry.LineSet()
            z_axis.points = o3d.utility.Vector3dVector([origin, z_end])
            z_axis.lines = o3d.utility.Vector2iVector([[0, 1]])
            z_axis.colors = o3d.utility.Vector3dVector([[0, 0, 1]])  # Blue
            orientation_axes.append(z_axis)
        
        return orientation_axes
    
    def _create_point_cloud_bounding_box(self):
        """
        Create a bounding box for the point cloud
        
        Returns:
            o3d.geometry.LineSet: Line set representing the bounding box
        """
        if not self.processor or not self.processor.point_cloud:
            print("Error: No point cloud loaded")
            return None
        
        # Get the axis-aligned bounding box
        aabb = self.processor.point_cloud.get_axis_aligned_bounding_box()
        
        # Create line set from the bounding box
        bounding_box = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(aabb)
        
        # Set color to white
        bounding_box.paint_uniform_color([1, 1, 1])
        
        return bounding_box
    
    def _create_camera_path(self):
        """
        Create a line set connecting adjacent cameras to show the camera path
        
        Returns:
            o3d.geometry.LineSet: Line set representing camera path
        """
        points = []
        lines = []
        
        # Extract camera positions
        for view in self.camera_params['views']:
            points.append(view['camera_position'])
        
        # Create lines connecting adjacent cameras
        for i in range(len(points) - 1):
            lines.append([i, i + 1])
        
        # Create line set
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        
        # Set line color as a gradient
        colors = []
        for i in range(len(lines)):
            ratio = i / max(1, len(lines) - 1)
            color = plt.cm.cool(ratio)[:3]  # Get RGB from matplotlib colormap
            colors.append(color)
        
        line_set.colors = o3d.utility.Vector3dVector(colors)
        
        return line_set
    
    def plot_camera_positions_matplotlib(self):
        """
        Plot camera positions using matplotlib (2D representation)
        """
        if not self.camera_params:
            print("Error: No camera parameters loaded")
            return
            
        # Extract camera positions
        camera_positions = []
        for view in self.camera_params['views']:
            camera_positions.append(view['camera_position'])
        
        camera_positions = np.array(camera_positions)
        
        # Create two subplots: top view and side view
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        
        # TOP VIEW (XZ plane)
        # Plot camera positions
        ax1.scatter(camera_positions[:, 0], camera_positions[:, 2], c=range(len(camera_positions)), 
                  cmap='viridis', s=100, alpha=0.8)
        
        # Connect cameras with lines
        ax1.plot(camera_positions[:, 0], camera_positions[:, 2], 'b-', alpha=0.5)
        
        # Add arrows to show looking direction
        for i, view in enumerate(self.camera_params['views']):
            extrinsic = np.array(view['extrinsic_matrix'])
            camera_pos = np.array(view['camera_position'])
            
            # The third row of R gives us the negative looking direction
            R = extrinsic[:3, :3].T
            look_dir = -R[:, 2]  # Negative Z is forward in camera coordinates
            
            # Scale the arrow
            arrow_scale = self.camera_params['radius'] * 0.15
            arrow_end = camera_pos + look_dir * arrow_scale
            
            # Draw arrow
            ax1.arrow(camera_pos[0], camera_pos[2], 
                     look_dir[0] * arrow_scale, look_dir[2] * arrow_scale,
                     head_width=arrow_scale*0.1, head_length=arrow_scale*0.2, 
                     fc='red', ec='red', alpha=0.7)
            
            # Add camera index
            ax1.text(camera_pos[0], camera_pos[2], f' {i}', fontsize=8)
        
        # Add center point
        center = self.camera_params['center']
        ax1.scatter([center[0]], [center[2]], c='r', s=200, marker='*', label='Point Cloud Center')
        
        # Set labels and title
        ax1.set_xlabel('X')
        ax1.set_ylabel('Z')
        ax1.set_title('Camera Positions (Top View - XZ plane)')
        ax1.grid(True)
        ax1.axis('equal')
        ax1.legend()
        
        # SIDE VIEW (YZ plane)
        # Plot camera positions
        ax2.scatter(camera_positions[:, 1], camera_positions[:, 2], c=range(len(camera_positions)), 
                  cmap='viridis', s=100, alpha=0.8)
        
        # Connect cameras with lines
        ax2.plot(camera_positions[:, 1], camera_positions[:, 2], 'g-', alpha=0.5)
        
        # Add arrows to show looking direction
        for i, view in enumerate(self.camera_params['views']):
            extrinsic = np.array(view['extrinsic_matrix'])
            camera_pos = np.array(view['camera_position'])
            
            # The third row of R gives us the negative looking direction
            R = extrinsic[:3, :3].T
            look_dir = -R[:, 2]  # Negative Z is forward in camera coordinates
            
            # Scale the arrow
            arrow_scale = self.camera_params['radius'] * 0.15
            arrow_end = camera_pos + look_dir * arrow_scale
            
            # Draw arrow
            ax2.arrow(camera_pos[1], camera_pos[2], 
                     look_dir[1] * arrow_scale, look_dir[2] * arrow_scale,
                     head_width=arrow_scale*0.1, head_length=arrow_scale*0.2, 
                     fc='red', ec='red', alpha=0.7)
            
            # Add camera index
            ax2.text(camera_pos[1], camera_pos[2], f' {i}', fontsize=8)
        
        # Add center point
        center = self.camera_params['center']
        ax2.scatter([center[1]], [center[2]], c='r', s=200, marker='*', label='Point Cloud Center')
        
        # Set labels and title
        ax2.set_xlabel('Y')
        ax2.set_ylabel('Z')
        ax2.set_title('Camera Positions (Side View - YZ plane)')
        ax2.grid(True)
        ax2.axis('equal')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
    
    def plot_camera_positions_3d(self):
        """
        Plot camera positions in 3D using matplotlib
        """
        if not self.camera_params:
            print("Error: No camera parameters loaded")
            return
            
        # Extract camera positions
        camera_positions = []
        for view in self.camera_params['views']:
            camera_positions.append(view['camera_position'])
        
        camera_positions = np.array(camera_positions)
        
        # Create figure
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot camera positions
        sc = ax.scatter(camera_positions[:, 0], camera_positions[:, 1], camera_positions[:, 2], 
                       c=range(len(camera_positions)), cmap='viridis', s=100, alpha=0.8)
        
        # Connect cameras with lines
        ax.plot(camera_positions[:, 0], camera_positions[:, 1], camera_positions[:, 2], 'b-', alpha=0.5)
        
        # Add camera indices
        for i, pos in enumerate(camera_positions):
            ax.text(pos[0], pos[1], pos[2], f' {i}', fontsize=8)
        
        # Add center point
        center = self.camera_params['center']
        ax.scatter([center[0]], [center[1]], [center[2]], c='r', s=200, marker='*', label='Point Cloud Center')
        
        # Add colorbar to show camera order
        cbar = plt.colorbar(sc, ax=ax, pad=0.1)
        cbar.set_label('Camera Order')
        
        # Show camera orientations
        for i, view in enumerate(self.camera_params['views']):
            extrinsic = np.array(view['extrinsic_matrix'])
            camera_pos = np.array(view['camera_position'])
            
            # Get rotation matrix
            R = extrinsic[:3, :3].T
            
            # Scale for the orientation arrows
            arrow_scale = self.camera_params['radius'] * 0.1
            
            # Plot x-axis (red)
            ax.quiver(camera_pos[0], camera_pos[1], camera_pos[2],
                     R[0, 0], R[1, 0], R[2, 0],
                     color='r', length=arrow_scale, normalize=True)
                     
            # Plot y-axis (green)
            ax.quiver(camera_pos[0], camera_pos[1], camera_pos[2],
                     R[0, 1], R[1, 1], R[2, 1],
                     color='g', length=arrow_scale, normalize=True)
                     
            # Plot z-axis (blue)
            ax.quiver(camera_pos[0], camera_pos[1], camera_pos[2],
                     R[0, 2], R[1, 2], R[2, 2],
                     color='b', length=arrow_scale, normalize=True)
        
        # If point cloud processor is available, add bounding box
        if self.processor and self.processor.point_cloud:
            # Get point cloud points
            points = np.asarray(self.processor.point_cloud.points)
            
            # Compute min/max bounds
            min_bound = np.min(points, axis=0)
            max_bound = np.max(points, axis=0)
            
            # Create vertices of the bounding box
            x_min, y_min, z_min = min_bound
            x_max, y_max, z_max = max_bound
            
            # List of vertices
            vertices = [
                [x_min, y_min, z_min], [x_max, y_min, z_min], [x_max, y_max, z_min], [x_min, y_max, z_min],
                [x_min, y_min, z_max], [x_max, y_min, z_max], [x_max, y_max, z_max], [x_min, y_max, z_max]
            ]
            
            # List of edges: pairs of vertex indices
            edges = [
                [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
                [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
                [0, 4], [1, 5], [2, 6], [3, 7]   # Connecting edges
            ]
            
            # Plot the edges
            for edge in edges:
                v1, v2 = edge
                x = [vertices[v1][0], vertices[v2][0]]
                y = [vertices[v1][1], vertices[v2][1]]
                z = [vertices[v1][2], vertices[v2][2]]
                ax.plot(x, y, z, 'w-', alpha=0.7, linewidth=1)
        
        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Camera Positions (3D View)')
        
        # Set equal aspect ratio
        max_range = np.array([
            camera_positions[:, 0].max() - camera_positions[:, 0].min(),
            camera_positions[:, 1].max() - camera_positions[:, 1].min(),
            camera_positions[:, 2].max() - camera_positions[:, 2].min()
        ]).max() / 2.0
        
        mid_x = (camera_positions[:, 0].max() + camera_positions[:, 0].min()) * 0.5
        mid_y = (camera_positions[:, 1].max() + camera_positions[:, 1].min()) * 0.5
        mid_z = (camera_positions[:, 2].max() + camera_positions[:, 2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        ax.legend()
        plt.tight_layout()
        plt.show()
        
def main():
    parser = argparse.ArgumentParser(description='Point cloud and camera visualizer')
    parser.add_argument('--ply', type=str, default=None,
                        help='Path to the PLY file to visualize')
    parser.add_argument('--cameras', type=str, default=None,
                        help='Path to camera parameters JSON file')
    parser.add_argument('--mode', type=str, default='3d',
                        choices=['3d', '2d', 'open3d'],
                        help='Visualization mode: 3d (matplotlib 3D), 2d (top view), or open3d')
    
    args = parser.parse_args()
    
    if not args.ply and not args.cameras:
        print("Error: Either PLY file or camera parameters must be provided")
        parser.print_help()
        return
    
    visualizer = PointCloudVisualizer(ply_path=args.ply, camera_params_path=args.cameras)
    
    if args.mode == '3d':
        visualizer.plot_camera_positions_3d()
    elif args.mode == '2d':
        visualizer.plot_camera_positions_matplotlib()
    elif args.mode == 'open3d':
        visualizer.visualize_cameras_and_point_cloud()
    
if __name__ == "__main__":
    main() 