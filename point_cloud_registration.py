#!/usr/bin/env python3
"""
Point Cloud Registration Pipeline

This script implements a complete pipeline to register multiple point clouds:
1. Preprocessing (downsampling, normal estimation, FPFH feature extraction)
2. Global registration using FPFH + RANSAC
3. Fine registration using Point-to-Plane ICP
4. Multi-view registration with pose graph optimization
5. Visualization and saving of results

Author: AI Assistant
Date: 2025-03-09
"""

import os
import sys
import numpy as np
import copy
import argparse
import time
import open3d as o3d
from pathlib import Path

class PointCloudRegistration:
    def __init__(self, voxel_size=0.05, visualize=True, save_results=True, output_dir="results", point_size=5.0):
        """
        Initialize the registration pipeline.
        
        Args:
            voxel_size (float): The voxel size for downsampling
            visualize (bool): Whether to visualize results at each step
            save_results (bool): Whether to save the final point cloud and transformations
            output_dir (str): Directory to save results
            point_size (float): Size of points for visualization
        """
        self.voxel_size = voxel_size
        self.visualize = visualize
        self.save_results = save_results
        self.output_dir = output_dir
        self.point_size = point_size
        
        # Create output directory if it doesn't exist
        if self.save_results:
            os.makedirs(self.output_dir, exist_ok=True)
        
        # Registration parameters
        self.normal_radius = 2 * self.voxel_size
        self.fpfh_radius = 5 * self.voxel_size
        self.distance_threshold = 1.5 * self.voxel_size
        self.ransac_n = 3
        self.ransac_max_iterations = 4000000
        self.ransac_confidence = 0.999
        
        # ICP parameters
        self.icp_distance_threshold = 1.0 * self.voxel_size
        self.icp_max_iterations = 100
        
        print(f"Initialized PointCloudRegistration with voxel_size={voxel_size}, point_size={point_size}")
    
    def preprocess_point_cloud(self, pcd):
        """
        Preprocess point cloud: downsample, estimate normals, and compute FPFH features.
        
        Args:
            pcd (o3d.geometry.PointCloud): Input point cloud
            
        Returns:
            tuple: (downsampled_pcd, fpfh_features)
        """
        print(":: Preprocessing point cloud...")
        
        # Make a copy of the point cloud to avoid modifying the original
        pcd_copy = copy.deepcopy(pcd)
        
        # Downsample with voxel grid filter
        pcd_down = pcd_copy.voxel_down_sample(self.voxel_size)
        
        # Estimate normals
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=self.normal_radius, max_nn=30))
        
        # Compute FPFH features
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=self.fpfh_radius, max_nn=100))
        
        print(f":: Preprocessed point cloud: {len(pcd_down.points)} points")
        
        return pcd_down, pcd_fpfh
    
    def execute_global_registration(self, source_down, target_down, source_fpfh, target_fpfh):
        """
        Perform global registration using FPFH features and RANSAC.
        
        Args:
            source_down (o3d.geometry.PointCloud): Downsampled source point cloud
            target_down (o3d.geometry.PointCloud): Downsampled target point cloud
            source_fpfh (o3d.pipelines.registration.Feature): FPFH features of source
            target_fpfh (o3d.pipelines.registration.Feature): FPFH features of target
            
        Returns:
            o3d.pipelines.registration.RegistrationResult: Registration result
        """
        print(":: Performing global registration with FPFH + RANSAC...")
        
        # RANSAC registration based on feature matching and RANSAC
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh,
            True,  # Mutual filter
            self.distance_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            self.ransac_n,
            [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
             o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(self.distance_threshold)],
            o3d.pipelines.registration.RANSACConvergenceCriteria(
                self.ransac_max_iterations, self.ransac_confidence))
        
        print(f":: Global registration finished. Fitness: {result.fitness}, RMSE: {result.inlier_rmse}")
        
        return result
    
    def refine_registration(self, source, target, result_ransac):
        """
        Refine registration using Point-to-Plane ICP.
        
        Args:
            source (o3d.geometry.PointCloud): Source point cloud
            target (o3d.geometry.PointCloud): Target point cloud
            result_ransac (o3d.pipelines.registration.RegistrationResult): Initial registration result
            
        Returns:
            o3d.pipelines.registration.RegistrationResult: Refined registration result
        """
        print(":: Refining registration with Point-to-Plane ICP...")
        
        # Check for empty point clouds
        if len(source.points) < 3 or len(target.points) < 3:
            print(":: Warning: Point clouds have too few points for ICP, using RANSAC result instead")
            return result_ransac
        
        # Create a copy of point clouds to ensure normals are computed
        source_copy = copy.deepcopy(source)
        target_copy = copy.deepcopy(target)
        
        # Ensure normals are computed
        if len(np.asarray(source_copy.normals)) == 0:
            source_copy.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=self.normal_radius, max_nn=30))
        if len(np.asarray(target_copy.normals)) == 0:
            target_copy.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=self.normal_radius, max_nn=30))
        
        # Use point-to-plane ICP
        try:
            result = o3d.pipelines.registration.registration_icp(
                source_copy, target_copy, self.icp_distance_threshold, result_ransac.transformation,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    max_iteration=self.icp_max_iterations))
            
            # Check if ICP converged properly
            if result.fitness <= 0.001:
                print(":: Warning: ICP did not converge well, fitness is very low. Using RANSAC result instead.")
                return result_ransac
                
            print(f":: ICP refinement finished. Fitness: {result.fitness}, RMSE: {result.inlier_rmse}")
            return result
            
        except Exception as e:
            print(f":: Error in ICP refinement: {e}")
            print(":: Falling back to RANSAC result")
            return result_ransac
    
    def construct_pose_graph(self, pcds, pcd_pairs):
        """
        Construct pose graph for pose graph optimization.
        
        Args:
            pcds (list): List of point clouds
            pcd_pairs (list): List of (i, j) indices of point cloud pairs for registration
            
        Returns:
            tuple: (pose_graph, odometry_transformation)
        """
        print(":: Constructing pose graph...")
        
        n_pcds = len(pcds)
        pose_graph = o3d.pipelines.registration.PoseGraph()
        odometry = np.identity(4)
        pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
        
        # For odometry, we only need to deal with consecutive pairs
        odometry_transformations = []
        for i in range(n_pcds - 1):
            # Preprocess point clouds
            source_down, source_fpfh = self.preprocess_point_cloud(pcds[i])
            target_down, target_fpfh = self.preprocess_point_cloud(pcds[i+1])
            
            # Global registration
            result_ransac = self.execute_global_registration(
                source_down, target_down, source_fpfh, target_fpfh)
            
            # Refine registration
            result_icp = self.refine_registration(pcds[i], pcds[i+1], result_ransac)
            
            # Update odometry
            odometry = np.dot(result_icp.transformation, odometry)
            pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(odometry)))
            
            # Create information matrix - identity scaled by fitness (higher fitness = more confidence)
            # The fitness value is between 0 and 1, where 1 means perfect alignment
            # We scale it to give higher confidence values
            fitness_scaling = 100.0  # Scale factor to get reasonable values
            information = np.identity(6) * max(0.001, result_icp.fitness) * fitness_scaling
            
            # Add odometry edge
            pose_graph.edges.append(
                o3d.pipelines.registration.PoseGraphEdge(
                    i, i+1,
                    result_icp.transformation,
                    information,
                    uncertain=False))
            
            odometry_transformations.append(result_icp.transformation)
        
        # For loop closure, we deal with non-consecutive pairs
        for i, j in pcd_pairs:
            if abs(i - j) == 1:
                # Skip consecutive pairs as they're already added as odometry edges
                continue
                
            print(f":: Processing loop closure between frame {i} and {j}")
            
            # Preprocess point clouds
            source_down, source_fpfh = self.preprocess_point_cloud(pcds[i])
            target_down, target_fpfh = self.preprocess_point_cloud(pcds[j])
            
            # Global registration
            result_ransac = self.execute_global_registration(
                source_down, target_down, source_fpfh, target_fpfh)
            
            # Refine registration
            result_icp = self.refine_registration(pcds[i], pcds[j], result_ransac)
            
            # Add loop closure edge
            if result_icp.fitness > 0.3:  # Only add edge if ICP fitness is good
                # Create information matrix for loop closure (with less certainty than odometry)
                information = np.identity(6) * max(0.001, result_icp.fitness) * 100.0 * 0.3  # Less confident
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(
                        i, j,
                        result_icp.transformation,
                        information,
                        uncertain=True))
        
        return pose_graph, odometry_transformations
    
    def optimize_pose_graph(self, pose_graph, max_iterations=100):
        """
        Optimize pose graph.
        
        Args:
            pose_graph (o3d.pipelines.registration.PoseGraph): Pose graph
            max_iterations (int): Maximum number of iterations for optimization
            
        Returns:
            o3d.pipelines.registration.PoseGraph: Optimized pose graph
        """
        print(":: Optimizing pose graph...")
        
        option = o3d.pipelines.registration.GlobalOptimizationOption(
            max_correspondence_distance=self.distance_threshold,
            edge_prune_threshold=0.25,
            reference_node=0)
        
        o3d.pipelines.registration.global_optimization(
            pose_graph,
            o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
            o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
            option)
        
        return pose_graph
    
    def integrate_point_clouds(self, pcds, pose_graph):
        """
        Integrate point clouds using optimized transformations.
        
        Args:
            pcds (list): List of point clouds
            pose_graph (o3d.pipelines.registration.PoseGraph): Optimized pose graph
            
        Returns:
            o3d.geometry.PointCloud: Integrated point cloud
        """
        print(":: Integrating point clouds...")
        
        # Transform and combine point clouds
        pcds_transformed = []
        for i, pcd in enumerate(pcds):
            try:
                pcd_temp = copy.deepcopy(pcd)
                pcd_temp.transform(pose_graph.nodes[i].pose)
                pcds_transformed.append(pcd_temp)
            except Exception as e:
                print(f":: Error transforming point cloud {i}: {e}")
        
        if len(pcds_transformed) == 0:
            print(":: Error: No valid transformed point clouds to integrate")
            return o3d.geometry.PointCloud()
        
        # Combine all transformed point clouds
        combined_pcd = o3d.geometry.PointCloud()
        for pcd in pcds_transformed:
            combined_pcd += pcd
        
        # Optional: downsample the combined point cloud to remove redundant points
        print(":: Downsampling combined point cloud")
        downsampled_pcd = combined_pcd.voxel_down_sample(self.voxel_size)
        
        # Clean the point cloud by removing outliers
        print(":: Cleaning integrated point cloud")
        
        # Check if we have enough points for statistical outlier removal
        if len(downsampled_pcd.points) > 10:
            try:
                # Statistical outlier removal
                cleaned_pcd, _ = downsampled_pcd.remove_statistical_outlier(
                    nb_neighbors=20, std_ratio=2.0)
                
                # If statistical removal removed too many points, revert to original
                if len(cleaned_pcd.points) < len(downsampled_pcd.points) * 0.5:
                    print(":: Warning: Statistical outlier removal removed too many points. Using downsampled cloud instead.")
                    cleaned_pcd = downsampled_pcd
            except Exception as e:
                print(f":: Error in statistical outlier removal: {e}")
                cleaned_pcd = downsampled_pcd
        else:
            print(":: Warning: Not enough points for statistical outlier removal")
            cleaned_pcd = downsampled_pcd
        
        # Ensure normals are computed for the final point cloud
        if len(np.asarray(cleaned_pcd.normals)) == 0:
            print(":: Estimating normals for integrated point cloud")
            cleaned_pcd.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=self.normal_radius*2, max_nn=30))
        
        points = np.asarray(cleaned_pcd.points)
        if len(points) > 0:
            print(f":: Point cloud bounds: min {points.min(axis=0)}, max {points.max(axis=0)}")
        
        print(f":: Integrated point cloud has {len(cleaned_pcd.points)} points")
        
        return cleaned_pcd
    
    def visualize_registration(self, source, target, transformation=None):
        """
        Visualize registration result.
        
        Args:
            source (o3d.geometry.PointCloud): Source point cloud
            target (o3d.geometry.PointCloud): Target point cloud
            transformation (numpy.ndarray): Transformation matrix
        """
        if not self.visualize:
            return
        
        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)
        
        # Color the point clouds
        source_temp.paint_uniform_color([1, 0.706, 0])  # Source in orange
        target_temp.paint_uniform_color([0, 0.651, 0.929])  # Target in blue
        
        # Apply transformation if provided
        if transformation is not None:
            source_temp.transform(transformation)
        
        # Visualize
        o3d.visualization.draw_geometries([source_temp, target_temp],
                                         window_name="Registration Result",
                                         width=1280, height=720)
    
    def visualize_integrated_point_cloud(self, pcd):
        """
        Visualize integrated point cloud.
        
        Args:
            pcd (o3d.geometry.PointCloud): Integrated point cloud
        """
        if not self.visualize:
            return
        
        print(f"Visualizing integrated point cloud with {len(pcd.points)} points...")
        
        # Check if point cloud has points
        if len(pcd.points) == 0:
            print("Warning: Integrated point cloud has no points to visualize")
            return
        
        # Create a copy of the point cloud for visualization
        pcd_vis = copy.deepcopy(pcd)
        
        # If the point cloud doesn't have colors, assign colors based on coordinates
        if len(np.asarray(pcd_vis.colors)) == 0:
            print("Point cloud has no colors, adding colors based on coordinates")
            points = np.asarray(pcd_vis.points)
            min_coords = np.min(points, axis=0)
            max_coords = np.max(points, axis=0)
            range_coords = max_coords - min_coords
            
            # Normalize coordinates to [0,1] range for coloring
            normalized = (points - min_coords) / range_coords
            pcd_vis.colors = o3d.utility.Vector3dVector(normalized)
        
        # For very small point clouds, let's create a sphere visualization
        # at each point to make them more visible
        if len(pcd_vis.points) < 500:
            print(f"Point cloud is small ({len(pcd_vis.points)} points), creating sphere representation")
            # Create a combined geometry with spheres at each point
            geometries = []
            
            # Add original point cloud with increased size
            pcd_vis_large = copy.deepcopy(pcd_vis)
            
            # Also create a sphere for each point
            sphere_radius = self.voxel_size * 2.0  # Adjust based on your data scale
            points = np.asarray(pcd_vis.points)
            colors = np.asarray(pcd_vis.colors)
            
            # For very large clouds, limit the number of spheres
            max_spheres = 100
            if len(points) > max_spheres:
                # Randomly sample points for spheres
                indices = np.random.choice(len(points), max_spheres, replace=False)
                points = points[indices]
                colors = colors[indices]
            
            # Create a sphere mesh for each point
            for i in range(len(points)):
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
                sphere.paint_uniform_color(colors[i])
                sphere.translate(points[i])
                geometries.append(sphere)
            
            # Add original point cloud to geometries
            geometries.append(pcd_vis_large)
            
            # Create visualizer
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name="Integrated Point Cloud", width=1280, height=720)
            
            # Add all geometries
            for geom in geometries:
                vis.add_geometry(geom)
            
            # Add coordinate frame
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
            vis.add_geometry(coord_frame)
        else:
            # Create visualizer
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name="Integrated Point Cloud", width=1280, height=720)
            
            # Add point cloud
            vis.add_geometry(pcd_vis)
            
            # Add coordinate frame
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
            vis.add_geometry(coord_frame)
        
        # Configure rendering options
        opt = vis.get_render_option()
        opt.point_size = self.point_size  # Use the configured point size
        opt.background_color = np.array([0.1, 0.1, 0.1])  # Dark background
        opt.show_coordinate_frame = True
        
        # Disable culling
        opt.light_on = True
        opt.mesh_show_back_face = True  # Show back faces
        
        # Additional rendering options
        opt.point_show_normal = False
        opt.mesh_shade_option = o3d.visualization.MeshShadeOption.Color
        
        # Get camera control
        ctrl = vis.get_view_control()
        
        # Reset view to show all points
        vis.reset_view_point(True)
        
        # Also provide different viewing angles
        print("\nMultiple view options:")
        print("Front view: Press '1'")
        print("Top view: Press '2'")
        print("Side view: Press '3'")
        print("Rotate view: Left-click + drag")
        print("Pan view: Shift + left-click + drag")
        print("Zoom: Mouse wheel or Ctrl + left-click + drag")
        print("Press 'h' for help on more controls")
        print("Press 'q' to exit the viewer")
        
        # Run the visualization
        vis.run()
        vis.destroy_window()
    
    def save_registration_results(self, integrated_pcd, transformations):
        """
        Save registration results.
        
        Args:
            integrated_pcd (o3d.geometry.PointCloud): Integrated point cloud
            transformations (list): List of transformation matrices
        """
        if not self.save_results:
            return
        
        # Save integrated point cloud
        o3d.io.write_point_cloud(os.path.join(self.output_dir, "integrated.ply"), integrated_pcd)
        
        # Save transformations
        np.save(os.path.join(self.output_dir, "transformations.npy"), np.array(transformations))
        
        print(f":: Results saved to {self.output_dir}")
    
    def determine_point_cloud_pairs(self, n_pcds, max_distance=3):
        """
        Determine point cloud pairs for registration.
        
        Args:
            n_pcds (int): Number of point clouds
            max_distance (int): Maximum distance between frames to consider for loop closures
            
        Returns:
            list: List of (i, j) indices of point cloud pairs
        """
        pairs = []
        
        # Add odometry pairs (consecutive frames)
        for i in range(n_pcds - 1):
            pairs.append((i, i+1))
        
        # Add loop closure pairs (non-consecutive frames within max_distance)
        for i in range(n_pcds):
            for j in range(i + 2, min(i + max_distance + 1, n_pcds)):
                pairs.append((i, j))
        
        return pairs
    
    def process_point_clouds(self, point_cloud_files):
        """
        Process point clouds for registration.
        
        Args:
            point_cloud_files (list): List of point cloud file paths
            
        Returns:
            o3d.geometry.PointCloud: Integrated point cloud
        """
        print(f":: Processing {len(point_cloud_files)} point clouds...")
        
        # Load point clouds
        pcds = []
        for file in point_cloud_files:
            try:
                print(f":: Loading {file}")
                pcd = o3d.io.read_point_cloud(file)
                
                # Check if point cloud is empty
                if len(pcd.points) == 0:
                    print(f":: Warning: Point cloud {file} is empty, skipping")
                    continue
                
                # Check if the point cloud has colors, if not, assign random colors
                if len(np.asarray(pcd.colors)) == 0:
                    print(f":: Point cloud {file} has no colors, assigning random colors")
                    pcd.paint_uniform_color([np.random.random(), np.random.random(), np.random.random()])
                
                # Make sure normals are computed
                if len(np.asarray(pcd.normals)) == 0:
                    print(f":: Point cloud {file} has no normals, estimating normals")
                    pcd.estimate_normals(
                        o3d.geometry.KDTreeSearchParamHybrid(radius=self.normal_radius, max_nn=30))
                
                pcds.append(pcd)
            except Exception as e:
                print(f":: Error loading {file}: {e}, skipping this file")
        
        n_pcds = len(pcds)
        
        if n_pcds < 2:
            print(":: Error: At least two valid point clouds are needed for registration")
            return None
        
        # Determine point cloud pairs for registration
        pcd_pairs = self.determine_point_cloud_pairs(n_pcds)
        
        try:
            # Construct pose graph
            pose_graph, odometry_transformations = self.construct_pose_graph(pcds, pcd_pairs)
            
            # Optimize pose graph
            optimized_pose_graph = self.optimize_pose_graph(pose_graph)
            
            # Integrate point clouds
            integrated_pcd = self.integrate_point_clouds(pcds, optimized_pose_graph)
            
            # Visualize integrated point cloud
            self.visualize_integrated_point_cloud(integrated_pcd)
            
            # Save results
            self.save_registration_results(integrated_pcd, odometry_transformations)
            
            return integrated_pcd
        except Exception as e:
            print(f":: Error during registration: {e}")
            import traceback
            traceback.print_exc()
            return None

def main():
    parser = argparse.ArgumentParser(description="Point Cloud Registration Pipeline")
    parser.add_argument("--input_dir", type=str, required=True,
                      help="Directory containing input point cloud files (.ply or .pcd)")
    parser.add_argument("--output_dir", type=str, default="results",
                      help="Directory to save results")
    parser.add_argument("--voxel_size", type=float, default=0.05,
                      help="Voxel size for downsampling")
    parser.add_argument("--point_size", type=float, default=5.0,
                      help="Size of points for visualization")
    parser.add_argument("--no_visualization", action="store_true",
                      help="Disable visualization")
    parser.add_argument("--no_save", action="store_true",
                      help="Disable saving results")
    
    args = parser.parse_args()
    
    # Check if input directory exists
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory {args.input_dir} does not exist")
        return
    
    # Find all point cloud files in the input directory
    point_cloud_files = []
    for ext in [".ply", ".pcd"]:
        point_cloud_files.extend(
            [str(p) for p in Path(args.input_dir).glob(f"*{ext}")])
    
    # Sort files to ensure consistent ordering
    point_cloud_files.sort()
    
    if len(point_cloud_files) < 2:
        print(f"Error: At least two point cloud files are needed in {args.input_dir}")
        return
    
    print(f"Found {len(point_cloud_files)} point cloud files in {args.input_dir}")
    
    # Create registration pipeline
    registration = PointCloudRegistration(
        voxel_size=args.voxel_size,
        visualize=not args.no_visualization,
        save_results=not args.no_save,
        output_dir=args.output_dir,
        point_size=args.point_size)
    
    # Process point clouds
    start_time = time.time()
    integrated_pcd = registration.process_point_clouds(point_cloud_files)
    end_time = time.time()
    
    print(f"Registration completed in {end_time - start_time:.2f} seconds")
    
    if integrated_pcd is None:
        print("Registration failed")
        return
    
    print("Registration pipeline completed successfully")

if __name__ == "__main__":
    main() 