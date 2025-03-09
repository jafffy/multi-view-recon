#!/usr/bin/env python3
"""
Point Cloud Alignment and Fusion Module

This script aligns and fuses an RGB-D point cloud with an existing PLY model:
1. Initial alignment using feature matching and RANSAC
2. Fine alignment using Point-to-Plane ICP
3. Weighted fusion of aligned point clouds
4. Voxel downsampling for optimization
5. Visualization and saving of the fused model

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

class PointCloudAlignment:
    def __init__(self, voxel_size=0.05, visualize=True, output_dir="results"):
        """
        Initialize the point cloud alignment and fusion module.
        
        Args:
            voxel_size (float): Voxel size for downsampling
            visualize (bool): Whether to visualize results
            output_dir (str): Directory to save results
        """
        self.voxel_size = voxel_size
        self.visualize = visualize
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
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
        
        print(f"Initialized PointCloudAlignment with voxel_size={voxel_size}")
    
    def load_point_clouds(self, reference_path, target_path):
        """
        Load reference model and target point cloud.
        
        Args:
            reference_path (str): Path to reference PLY model
            target_path (str): Path to target point cloud
            
        Returns:
            tuple: (reference_pcd, target_pcd)
        """
        print(f"Loading reference model from {reference_path}")
        reference_pcd = o3d.io.read_point_cloud(reference_path)
        
        print(f"Loading target point cloud from {target_path}")
        target_pcd = o3d.io.read_point_cloud(target_path)
        
        # Check if point clouds are valid
        if len(reference_pcd.points) == 0:
            raise ValueError(f"Reference point cloud has no points: {reference_path}")
        if len(target_pcd.points) == 0:
            raise ValueError(f"Target point cloud has no points: {target_path}")
        
        print(f"Reference model has {len(reference_pcd.points)} points")
        print(f"Target point cloud has {len(target_pcd.points)} points")
        
        return reference_pcd, target_pcd
    
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
        
        # Ensure point cloud has normals
        if len(np.asarray(pcd_down.normals)) == 0:
            print(":: Estimating normals")
            pcd_down.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=self.normal_radius, max_nn=30))
            pcd_down.orient_normals_to_align_with_direction([0, 0, -1])
        
        # Compute FPFH features
        print(":: Computing FPFH features")
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
        
        # Visualize initial alignment if needed
        if self.visualize:
            self.visualize_registration(source_down, target_down, result.transformation)
        
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
            
            print(f":: ICP refinement finished. Fitness: {result.fitness}, RMSE: {result.inlier_rmse}")
            
            # Visualize refined alignment if needed
            if self.visualize:
                self.visualize_registration(source_copy, target_copy, result.transformation)
            
            return result
            
        except Exception as e:
            print(f":: Error in ICP refinement: {e}")
            print(":: Falling back to RANSAC result")
            return result_ransac
    
    def fuse_point_clouds(self, source, target, transformation, weight_source=1.0, weight_target=1.0):
        """
        Fuse source and target point clouds with weighted fusion.
        
        Args:
            source (o3d.geometry.PointCloud): Source point cloud
            target (o3d.geometry.PointCloud): Target point cloud
            transformation (numpy.ndarray): Transformation matrix from source to target
            weight_source (float): Weight for source points (0.0-1.0)
            weight_target (float): Weight for target points (0.0-1.0)
            
        Returns:
            o3d.geometry.PointCloud: Fused point cloud
        """
        print(":: Fusing point clouds...")
        
        # Transform source point cloud to target coordinate system
        source_transformed = copy.deepcopy(source)
        source_transformed.transform(transformation)
        
        # Prepare for fusion
        source_points = np.asarray(source_transformed.points)
        source_colors = np.asarray(source_transformed.colors) if len(source_transformed.colors) > 0 else None
        source_normals = np.asarray(source_transformed.normals) if len(source_transformed.normals) > 0 else None
        
        target_points = np.asarray(target.points)
        target_colors = np.asarray(target.colors) if len(target.colors) > 0 else None
        target_normals = np.asarray(target.normals) if len(target.normals) > 0 else None
        
        # Perform simple concatenation for points
        fused_points = np.vstack((source_points, target_points))
        
        # Create the fused point cloud
        fused_pcd = o3d.geometry.PointCloud()
        fused_pcd.points = o3d.utility.Vector3dVector(fused_points)
        
        # Handle colors
        if source_colors is not None and target_colors is not None:
            fused_colors = np.vstack((source_colors * weight_source, target_colors * weight_target))
            fused_pcd.colors = o3d.utility.Vector3dVector(fused_colors)
        elif source_colors is not None:
            fused_pcd.colors = o3d.utility.Vector3dVector(np.vstack((
                source_colors * weight_source, 
                np.ones_like(target_points) * np.array([0.5, 0.5, 0.5])
            )))
        elif target_colors is not None:
            fused_pcd.colors = o3d.utility.Vector3dVector(np.vstack((
                np.ones_like(source_points) * np.array([0.5, 0.5, 0.5]),
                target_colors * weight_target
            )))
        
        # Handle normals
        if source_normals is not None and target_normals is not None:
            fused_normals = np.vstack((source_normals, target_normals))
            fused_pcd.normals = o3d.utility.Vector3dVector(fused_normals)
        elif source_normals is not None:
            # Estimate normals for the whole cloud instead of just concatenating
            fused_pcd.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=self.normal_radius, max_nn=30))
        elif target_normals is not None:
            # Estimate normals for the whole cloud instead of just concatenating
            fused_pcd.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=self.normal_radius, max_nn=30))
        
        # Voxel downsampling to remove redundant points
        print(":: Performing voxel downsampling to clean up the fused point cloud")
        fused_pcd_down = fused_pcd.voxel_down_sample(self.voxel_size)
        
        # Remove statistical outliers
        print(":: Removing outliers")
        fused_pcd_clean, _ = fused_pcd_down.remove_statistical_outlier(
            nb_neighbors=20, std_ratio=2.0)
        
        print(f":: Fused point cloud has {len(fused_pcd_clean.points)} points")
        
        return fused_pcd_clean
    
    def visualize_registration(self, source, target, transformation=None):
        """
        Visualize registration result.
        
        Args:
            source (o3d.geometry.PointCloud): Source point cloud
            target (o3d.geometry.PointCloud): Target point cloud
            transformation (numpy.ndarray): Transformation matrix
        """
        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)
        
        # Color the point clouds
        source_temp.paint_uniform_color([1, 0.706, 0])  # Source in orange
        target_temp.paint_uniform_color([0, 0.651, 0.929])  # Target in blue
        
        # Apply transformation if provided
        if transformation is not None:
            source_temp.transform(transformation)
        
        # Visualize
        print(":: Visualizing registration result (orange: source, blue: target)")
        o3d.visualization.draw_geometries([source_temp, target_temp],
                                         window_name="Registration Result",
                                         width=1280, height=720)
    
    def visualize_point_cloud(self, pcd, window_name="Point Cloud Viewer"):
        """
        Visualize a point cloud.
        
        Args:
            pcd (o3d.geometry.PointCloud): Point cloud to visualize
            window_name (str): Window title
        """
        # Check if point cloud has points
        if len(pcd.points) == 0:
            print(":: Warning: Point cloud has no points to visualize")
            return
        
        # Create visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=window_name, width=1280, height=720)
        
        # Add point cloud
        vis.add_geometry(pcd)
        
        # Add coordinate frame
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=self.voxel_size * 10, origin=[0, 0, 0])
        vis.add_geometry(coord_frame)
        
        # Configure rendering options
        opt = vis.get_render_option()
        opt.point_size = 3.0  # Increase point size for better visibility
        opt.background_color = np.array([0.1, 0.1, 0.1])  # Dark background
        opt.show_coordinate_frame = True
        
        # Disable culling
        opt.light_on = True
        opt.mesh_show_back_face = True  # Show back faces
        
        # Reset view to show all points
        vis.reset_view_point(True)
        
        print(":: Press 'h' for help on visualization controls")
        print(":: Press 'q' to exit the viewer")
        
        # Run visualization
        vis.run()
        vis.destroy_window()
    
    def save_point_cloud(self, pcd, output_path):
        """
        Save point cloud to file.
        
        Args:
            pcd (o3d.geometry.PointCloud): Point cloud to save
            output_path (str): Output file path
        """
        print(f":: Saving point cloud to {output_path}")
        o3d.io.write_point_cloud(output_path, pcd)
    
    def align_and_fuse(self, reference_path, target_path, output_path=None):
        """
        Align target point cloud to reference model and fuse them.
        
        Args:
            reference_path (str): Path to reference PLY model
            target_path (str): Path to target point cloud
            output_path (str): Path to save the fused model
            
        Returns:
            o3d.geometry.PointCloud: Fused point cloud
        """
        # Default output path if not specified
        if output_path is None:
            output_path = os.path.join(self.output_dir, "fused_model.ply")
        
        # Load point clouds
        reference_pcd, target_pcd = self.load_point_clouds(reference_path, target_path)
        
        # Preprocess point clouds
        print("\n:: Preprocessing reference model")
        reference_down, reference_fpfh = self.preprocess_point_cloud(reference_pcd)
        
        print("\n:: Preprocessing target point cloud")
        target_down, target_fpfh = self.preprocess_point_cloud(target_pcd)
        
        # Global registration (RANSAC)
        print("\n:: Performing initial alignment (RANSAC)")
        ransac_result = self.execute_global_registration(
            target_down, reference_down, target_fpfh, reference_fpfh)
        
        # Check if RANSAC was successful
        if ransac_result.fitness < 0.1:
            print(":: Warning: RANSAC alignment has low fitness. Results may be poor.")
        
        # Fine registration (ICP)
        print("\n:: Performing fine alignment (ICP)")
        icp_result = self.refine_registration(target_pcd, reference_pcd, ransac_result)
        
        # Fuse point clouds
        print("\n:: Fusing point clouds")
        fused_pcd = self.fuse_point_clouds(
            target_pcd, reference_pcd, icp_result.transformation,
            weight_source=0.8, weight_target=1.0
        )
        
        # Save fused point cloud
        self.save_point_cloud(fused_pcd, output_path)
        
        # Visualize final result
        if self.visualize:
            print("\n:: Visualizing fused point cloud")
            self.visualize_point_cloud(fused_pcd, "Fused Point Cloud")
        
        return fused_pcd

def main():
    parser = argparse.ArgumentParser(
        description="Align and fuse an RGB-D point cloud with a reference PLY model")
    parser.add_argument("--reference", type=str, required=True,
                      help="Path to reference PLY model")
    parser.add_argument("--target", type=str, required=True,
                      help="Path to target point cloud")
    parser.add_argument("--output", type=str, default=None,
                      help="Path to save the fused model (default: results/fused_model.ply)")
    parser.add_argument("--voxel_size", type=float, default=0.05,
                      help="Voxel size for downsampling (default: 0.05)")
    parser.add_argument("--no_visualization", action="store_true",
                      help="Disable visualization")
    parser.add_argument("--output_dir", type=str, default="results",
                      help="Directory to save results (default: results)")
    
    args = parser.parse_args()
    
    # Check if input files exist
    for path in [args.reference, args.target]:
        if not os.path.exists(path):
            print(f"Error: File {path} does not exist")
            return
    
    # Create alignment module
    alignment = PointCloudAlignment(
        voxel_size=args.voxel_size,
        visualize=not args.no_visualization,
        output_dir=args.output_dir
    )
    
    # Align and fuse point clouds
    start_time = time.time()
    fused_pcd = alignment.align_and_fuse(args.reference, args.target, args.output)
    end_time = time.time()
    
    print(f"\nPoint cloud alignment and fusion completed in {end_time - start_time:.2f} seconds")
    print(f"Fused model saved to: {args.output if args.output else os.path.join(args.output_dir, 'fused_model.ply')}")

if __name__ == "__main__":
    main() 