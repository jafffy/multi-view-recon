#!/usr/bin/env python
import os
import numpy as np
import open3d as o3d
import time

class PointCloudProcessor:
    def __init__(self, ply_path=None, cache_dir=None):
        """
        Initialize the point cloud processor
        
        Args:
            ply_path (str, optional): Path to the PLY file
            cache_dir (str, optional): Directory to cache processed point clouds
        """
        self.ply_path = ply_path
        self.point_cloud = None
        self.bbox = None
        self.center = None
        self.extent = None
        self.radius = None
        
        # Set up cache directory
        if cache_dir is None and ply_path is not None:
            base_dir = os.path.dirname(ply_path)
            cache_dir = os.path.join(base_dir, "cache")
        
        self.cache_dir = cache_dir
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            
        # If ply path is provided, load it
        if ply_path:
            self.load_point_cloud(ply_path)
    
    def load_point_cloud(self, ply_path):
        """
        Load and process a point cloud from a PLY file
        
        Args:
            ply_path (str): Path to the PLY file
            
        Returns:
            bool: True if successful
        """
        self.ply_path = ply_path
        
        # Create a cached filename based on the input ply path
        if self.cache_dir:
            ply_basename = os.path.basename(ply_path)
            self.normals_cache_path = os.path.join(self.cache_dir, f"{os.path.splitext(ply_basename)[0]}_with_normals.ply")
        else:
            self.normals_cache_path = None
        
        try:
            # Check if we have a cached version with normals
            if self.normals_cache_path and os.path.exists(self.normals_cache_path):
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
                
            # Calculate bounding box and center
            self.compute_bounding_info()
            
            return True
        except Exception as e:
            print(f"Error loading point cloud: {e}")
            return False
    
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
            if self.normals_cache_path:
                print(f"Saving point cloud with normals to cache: {self.normals_cache_path}")
                o3d.io.write_point_cloud(self.normals_cache_path, self.point_cloud)
    
    def compute_bounding_info(self):
        """
        Compute bounding box, center, and radius information
        """
        if not self.point_cloud or not self.point_cloud.has_points():
            raise ValueError("No point cloud loaded")
            
        self.bbox = self.point_cloud.get_axis_aligned_bounding_box()
        self.center = self.bbox.get_center()
        self.extent = self.bbox.get_extent()
        self.radius = np.linalg.norm(self.extent) * 0.8  # Standard camera distance
        
        print(f"Point cloud center: {self.center}")
        print(f"Point cloud extent: {self.extent}")
        print(f"Point cloud radius: {self.radius}")
    
    def preview_point_cloud(self, window_name="Point Cloud Preview", width=800, height=600):
        """
        Show a preview of the point cloud
        
        Args:
            window_name (str): Title for the preview window
            width (int): Window width
            height (int): Window height
        """
        if not self.point_cloud or not self.point_cloud.has_points():
            print("Error: No point cloud loaded")
            return
            
        print("Showing point cloud preview (close window to continue)...")
        o3d.visualization.draw_geometries([self.point_cloud], 
                                        window_name=window_name,
                                        width=width, 
                                        height=height,
                                        point_show_normal=False)
    
    def get_camera_positions(self, num_views, custom_radius=None):
        """
        Generate camera positions around the point cloud
        
        Args:
            num_views (int): Number of views to generate
            custom_radius (float, optional): Custom radius for camera positions
            
        Returns:
            list: List of camera positions (numpy arrays)
        """
        if not self.point_cloud or not self.center.any():
            raise ValueError("No point cloud loaded")
            
        positions = []
        
        # Use custom radius if provided, otherwise use computed radius
        radius = custom_radius if custom_radius is not None else self.radius
        
        # Create a hemisphere of positions with spiral pattern for uniform coverage
        indices = np.arange(0, num_views)
        phi = np.arccos(1.0 - indices / float(num_views - 1))  # 0 to pi/2 (hemisphere)
        theta = np.pi * (1 + 5**0.5) * indices  # Golden angle in radians
        
        # Convert spherical to Cartesian coordinates
        x = radius * np.sin(phi) * np.cos(theta)
        y = radius * np.sin(phi) * np.sin(theta)
        z = radius * np.cos(phi)  # Will be positive for hemisphere
        
        # Add some randomization to avoid perfectly regular patterns
        np.random.seed(42)  # For reproducibility
        jitter = radius * 0.05  # 5% jitter
        x += np.random.uniform(-jitter, jitter, num_views)
        y += np.random.uniform(-jitter, jitter, num_views)
        z += np.random.uniform(-jitter, jitter, num_views)
        
        # Ensure minimum distance between consecutive cameras (for good baseline)
        min_baseline = radius * 0.08  # Minimum baseline
        max_baseline = radius * 0.25  # Maximum baseline
        
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
        
        if baselines:
            print(f"Camera baseline range: {min(baselines):.2f} to {max(baselines):.2f}")
        
        return positions
    
    def calculate_extrinsic_matrix(self, eye_pos):
        """
        Calculate camera extrinsic matrix for a given position
        
        Args:
            eye_pos (np.ndarray): Camera position
            
        Returns:
            np.ndarray: 4x4 extrinsic matrix
        """
        if not self.center.any():
            raise ValueError("Point cloud center not calculated")
            
        # Look at the center of the point cloud
        target = self.center
        
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