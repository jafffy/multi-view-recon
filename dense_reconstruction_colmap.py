#!/usr/bin/env python
import os
import subprocess
import argparse
import json
import numpy as np
import cv2
import open3d as o3d
from pathlib import Path
import shutil
from tqdm import tqdm
import time

class DenseMultiViewStereo:
    """
    Dense Multi-View Stereo reconstruction using COLMAP
    """
    def __init__(self, data_dir, colmap_path=None, workspace_dir=None, quality='high'):
        """
        Initialize the dense multi-view stereo reconstruction system
        
        Args:
            data_dir (str): Directory containing the capture data (images and camera parameters)
            colmap_path (str, optional): Path to COLMAP executable. If None, assumes COLMAP is in PATH
            workspace_dir (str, optional): Directory for COLMAP workspace. If None, creates 'colmap_workspace' in data_dir
            quality (str, optional): Quality level for dense reconstruction ('low', 'medium', 'high', or 'extreme')
        """
        self.data_dir = data_dir
        self.images_dir = os.path.join(data_dir, "images")
        
        # COLMAP path
        self.colmap_path = "colmap" if colmap_path is None else colmap_path
        
        # Set up COLMAP workspace
        if workspace_dir is None:
            self.workspace_dir = os.path.join(data_dir, "colmap_workspace")
        else:
            self.workspace_dir = workspace_dir
            
        # Create directories
        self.sparse_dir = os.path.join(self.workspace_dir, "sparse")
        self.dense_dir = os.path.join(self.workspace_dir, "dense")
        self.database_path = os.path.join(self.workspace_dir, "database.db")
        self.output_dir = os.path.join(data_dir, "reconstruction")
        
        # Create necessary directories
        os.makedirs(self.workspace_dir, exist_ok=True)
        os.makedirs(self.sparse_dir, exist_ok=True)
        os.makedirs(self.dense_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set quality parameters
        self.quality = quality
        self.quality_params = {
            'low': {
                'patch_match_resolution': 1,  # Downsampling factor for patch matching
                'patch_match_num_iterations': 3,
                'patch_match_window_radius': 5,
                'depth_fusion_num_samples': 3
            },
            'medium': {
                'patch_match_resolution': 1,
                'patch_match_num_iterations': 5,
                'patch_match_window_radius': 5,
                'depth_fusion_num_samples': 5
            },
            'high': {
                'patch_match_resolution': 0,  # Full resolution
                'patch_match_num_iterations': 7,
                'patch_match_window_radius': 7,
                'depth_fusion_num_samples': 7
            },
            'extreme': {
                'patch_match_resolution': 0,
                'patch_match_num_iterations': 9,
                'patch_match_window_radius': 9,
                'depth_fusion_num_samples': 9
            }
        }
        
        # Load camera parameters if available
        self.camera_params_loaded = False
        try:
            self.load_camera_parameters()
            self.camera_params_loaded = True
        except:
            print("Warning: Could not load camera parameters. Using COLMAP auto-calibration.")
            
        # Initialize result variables
        self.point_cloud = None
        
    def load_camera_parameters(self):
        """
        Load camera parameters from JSON file
        """
        params_path = os.path.join(self.data_dir, 'camera_parameters.json')
        
        if not os.path.exists(params_path):
            raise FileNotFoundError(f"Camera parameters file not found: {params_path}")
            
        with open(params_path, 'r') as f:
            params = json.load(f)
            
        # Extract camera parameters
        self.intrinsic_matrix = np.array(params['intrinsic_matrix'])
        self.distortion_coeffs = np.array(params['distortion_coeffs'])
        self.resolution = tuple(params['resolution'])
        self.views = params.get('views', [])
        
    def extract_camera_parameters(self):
        """
        Extract camera parameters in COLMAP format from loaded parameters
        Returns dictionary with camera parameters in COLMAP format
        """
        if not hasattr(self, 'intrinsic_matrix'):
            return None
            
        fx = self.intrinsic_matrix[0, 0]
        fy = self.intrinsic_matrix[1, 1]
        cx = self.intrinsic_matrix[0, 2]
        cy = self.intrinsic_matrix[1, 2]
        width, height = self.resolution
        
        return {
            "model": "PINHOLE",
            "width": width,
            "height": height,
            "params": [fx, fy, cx, cy]
        }
        
    def run_command(self, command, verbose=True):
        """
        Run a command and handle errors
        
        Args:
            command (list): Command to run
            verbose (bool): Whether to print command output
            
        Returns:
            bool: True if command succeeded, False otherwise
        """
        if verbose:
            print(f"Running: {' '.join(command)}")
            
        try:
            result = subprocess.run(
                command,
                stdout=subprocess.PIPE if not verbose else None,
                stderr=subprocess.PIPE if not verbose else None,
                text=True,
                check=True
            )
            return True
        except subprocess.CalledProcessError as e:
            print(f"Command failed with return code {e.returncode}")
            if not verbose and e.stdout:
                print(f"stdout: {e.stdout}")
            if not verbose and e.stderr:
                print(f"stderr: {e.stderr}")
            return False
        
    def feature_extraction(self):
        """
        Run COLMAP feature extraction
        
        Returns:
            bool: True if successful, False otherwise
        """
        print("Extracting features...")
        
        command = [
            self.colmap_path, "feature_extractor",
            "--database_path", self.database_path,
            "--image_path", self.images_dir,
            "--ImageReader.single_camera", "1"
        ]
        
        # Add camera parameters if available
        if self.camera_params_loaded:
            camera_params = self.extract_camera_parameters()
            if camera_params:
                command.extend([
                    "--ImageReader.camera_model", camera_params["model"],
                    "--ImageReader.camera_params", 
                    ",".join(map(str, camera_params["params"]))
                ])
                
        return self.run_command(command)
    
    def feature_matching(self):
        """
        Run COLMAP feature matching
        
        Returns:
            bool: True if successful, False otherwise
        """
        print("Matching features...")
        
        command = [
            self.colmap_path, "exhaustive_matcher",
            "--database_path", self.database_path
        ]
        
        return self.run_command(command)
    
    def sparse_reconstruction(self):
        """
        Run COLMAP sparse reconstruction
        
        Returns:
            bool: True if successful, False otherwise
        """
        print("Performing sparse reconstruction...")
        
        command = [
            self.colmap_path, "mapper",
            "--database_path", self.database_path,
            "--image_path", self.images_dir,
            "--output_path", self.sparse_dir
        ]
        
        return self.run_command(command)
    
    def dense_reconstruction(self):
        """
        Run COLMAP dense reconstruction
        
        Returns:
            bool: True if successful, False otherwise
        """
        print("Preparing for dense reconstruction...")
        
        # Get the first model (usually model 0)
        model_dirs = os.listdir(self.sparse_dir)
        if not model_dirs:
            print("No sparse models found, reconstruction failed.")
            return False
            
        model_dir = model_dirs[0]
        if model_dir.isdigit():
            sparse_model_path = os.path.join(self.sparse_dir, model_dir)
        else:
            # If not a number, probably the model is directly in sparse_dir
            sparse_model_path = self.sparse_dir
        
        # Image undistortion
        print("Undistorting images...")
        undistort_command = [
            self.colmap_path, "image_undistorter",
            "--image_path", self.images_dir,
            "--input_path", sparse_model_path,
            "--output_path", self.dense_dir,
            "--output_type", "COLMAP"
        ]
        
        if not self.run_command(undistort_command):
            return False
            
        # Patch matching stereo
        print("Running patch matching stereo...")
        stereo_command = [
            self.colmap_path, "patch_match_stereo",
            "--workspace_path", self.dense_dir,
            "--workspace_format", "COLMAP",
            "--PatchMatchStereo.depth_min", "0.1",
            "--PatchMatchStereo.depth_max", "100",
            "--PatchMatchStereo.window_radius", str(self.quality_params[self.quality]['patch_match_window_radius']),
            "--PatchMatchStereo.window_step", "1",
            "--PatchMatchStereo.num_samples", "15",
            "--PatchMatchStereo.num_iterations", str(self.quality_params[self.quality]['patch_match_num_iterations']),
            "--PatchMatchStereo.geom_consistency", "true"
        ]
        
        if not self.run_command(stereo_command):
            return False
            
        # Stereo fusion
        print("Performing stereo fusion...")
        fusion_command = [
            self.colmap_path, "stereo_fusion",
            "--workspace_path", self.dense_dir,
            "--workspace_format", "COLMAP",
            "--input_type", "geometric",
            "--output_path", os.path.join(self.dense_dir, "fused.ply"),
            "--StereoFusion.min_num_pixels", str(self.quality_params[self.quality]['depth_fusion_num_samples']),
            "--StereoFusion.max_reproj_error", "2",
            "--StereoFusion.max_depth_error", "0.1"
        ]
        
        return self.run_command(fusion_command)
    
    def meshing(self):
        """
        Create a mesh from the dense point cloud using Poisson surface reconstruction
        
        Returns:
            bool: True if successful, False otherwise
        """
        print("Creating mesh...")
        
        dense_point_cloud_path = os.path.join(self.dense_dir, "fused.ply")
        meshed_model_path = os.path.join(self.dense_dir, "meshed.ply")
        
        if not os.path.exists(dense_point_cloud_path):
            print(f"Dense point cloud not found at {dense_point_cloud_path}")
            return False
            
        # Load point cloud
        try:
            pcd = o3d.io.read_point_cloud(dense_point_cloud_path)
            
            # Estimate normals if not already computed
            if not pcd.has_normals():
                print("Estimating normals...")
                pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
                pcd.orient_normals_consistent_tangent_plane(100)
            
            # Apply Poisson surface reconstruction
            print("Applying Poisson surface reconstruction...")
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=9, width=0, scale=1.1, linear_fit=False
            )
            
            # Remove low-density vertices
            vertices_to_remove = densities < np.quantile(densities, 0.01)
            mesh.remove_vertices_by_mask(vertices_to_remove)
            
            # Save the mesh
            o3d.io.write_triangle_mesh(meshed_model_path, mesh)
            print(f"Mesh saved to {meshed_model_path}")
            
            # Copy final results to output directory
            output_cloud_path = os.path.join(self.output_dir, "dense_point_cloud.ply")
            output_mesh_path = os.path.join(self.output_dir, "dense_mesh.ply")
            
            shutil.copy(dense_point_cloud_path, output_cloud_path)
            shutil.copy(meshed_model_path, output_mesh_path)
            
            print(f"Results copied to: {output_cloud_path} and {output_mesh_path}")
            
            # Save point cloud for visualization
            self.point_cloud = pcd
            
            return True
            
        except Exception as e:
            print(f"Error during meshing: {e}")
            return False
    
    def run_pipeline(self):
        """
        Run the complete dense reconstruction pipeline
        
        Returns:
            bool: True if all steps succeeded, False otherwise
        """
        start_time = time.time()
        
        print(f"Starting dense reconstruction with quality: {self.quality}")
        print(f"Images directory: {self.images_dir}")
        print(f"Workspace directory: {self.workspace_dir}")
        
        # Check if images exist
        if not os.path.exists(self.images_dir):
            print(f"Images directory not found: {self.images_dir}")
            return False
            
        # Check if COLMAP is available
        try:
            version_cmd = [self.colmap_path, "help"]
            result = subprocess.run(version_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if result.returncode != 0:
                print(f"COLMAP not found at {self.colmap_path}")
                print("Please install COLMAP or provide the correct path")
                return False
        except Exception as e:
            print(f"Error checking COLMAP: {e}")
            return False
            
        # Run the pipeline steps
        if not self.feature_extraction():
            print("Feature extraction failed")
            return False
            
        if not self.feature_matching():
            print("Feature matching failed")
            return False
            
        if not self.sparse_reconstruction():
            print("Sparse reconstruction failed")
            return False
            
        if not self.dense_reconstruction():
            print("Dense reconstruction failed")
            return False
            
        if not self.meshing():
            print("Meshing failed")
            return False
            
        elapsed_time = time.time() - start_time
        print(f"Dense reconstruction completed in {elapsed_time:.2f} seconds")
        
        return True
    
    def visualize(self):
        """
        Visualize the reconstruction results
        """
        dense_point_cloud_path = os.path.join(self.dense_dir, "fused.ply")
        mesh_path = os.path.join(self.dense_dir, "meshed.ply")
        
        if os.path.exists(mesh_path):
            print(f"Visualizing mesh from {mesh_path}")
            mesh = o3d.io.read_triangle_mesh(mesh_path)
            mesh.compute_vertex_normals()
            o3d.visualization.draw_geometries([mesh])
        elif os.path.exists(dense_point_cloud_path):
            print(f"Visualizing point cloud from {dense_point_cloud_path}")
            pcd = o3d.io.read_point_cloud(dense_point_cloud_path)
            o3d.visualization.draw_geometries([pcd])
        elif self.point_cloud is not None:
            print("Visualizing point cloud from memory")
            o3d.visualization.draw_geometries([self.point_cloud])
        else:
            print("No reconstruction results available for visualization")
            return False
            
        return True

def main():
    parser = argparse.ArgumentParser(description='Dense Multi-View Stereo reconstruction using COLMAP')
    parser.add_argument('--data_dir', required=True, help='Directory containing images and camera parameters')
    parser.add_argument('--colmap_path', default=None, help='Path to COLMAP executable')
    parser.add_argument('--workspace_dir', default=None, help='Directory for COLMAP workspace')
    parser.add_argument('--quality', choices=['low', 'medium', 'high', 'extreme'], default='high',
                        help='Quality level for dense reconstruction')
    parser.add_argument('--visualize', action='store_true', help='Visualize the results')
    
    args = parser.parse_args()
    
    try:
        # Create the dense reconstruction object
        mvs = DenseMultiViewStereo(
            args.data_dir,
            colmap_path=args.colmap_path,
            workspace_dir=args.workspace_dir,
            quality=args.quality
        )
        
        # Run the reconstruction pipeline
        print("Starting dense reconstruction...")
        success = mvs.run_pipeline()
        
        # Visualize the result if requested
        if success and args.visualize:
            mvs.visualize()
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    return 0

if __name__ == "__main__":
    main() 