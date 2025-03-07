#!/usr/bin/env python
import os
import numpy as np
import open3d as o3d
import cv2
import json
import argparse
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import time
import copy

class TSDFReconstruction:
    """
    3D model reconstruction using Truncated Signed Distance Function (TSDF) 
    with RGB-D information and camera parameters
    """
    
    def __init__(self, data_dir, voxel_size=0.01, sdf_trunc=0.04, max_depth=3.0, min_depth=0.1):
        """
        Initialize the TSDF reconstruction system
        
        Args:
            data_dir (str): Directory containing the capture data (RGB-D images and camera parameters)
            voxel_size (float): Size of voxels for the TSDF volume (in meters)
            sdf_trunc (float): Truncation value for signed distance function (in meters)
            max_depth (float): Maximum depth value to use from depth images (in meters)
            min_depth (float): Minimum depth value to use from depth images (in meters)
        """
        self.data_dir = data_dir
        self.images_dir = os.path.join(data_dir, "images")
        self.output_dir = os.path.join(data_dir, "reconstruction_tsdf")
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # TSDF parameters
        self.voxel_size = voxel_size
        self.sdf_trunc = sdf_trunc
        self.max_depth = max_depth
        self.min_depth = min_depth
        
        # Load camera parameters
        self.load_camera_parameters()
        
        # Initialize reconstruction parameters
        self.volume = None
        self.mesh = None
        
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
        self.views = params['views']
        
        # Create Open3D intrinsic object
        self.intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=self.resolution[0],
            height=self.resolution[1],
            fx=self.intrinsic_matrix[0, 0],
            fy=self.intrinsic_matrix[1, 1],
            cx=self.intrinsic_matrix[0, 2],
            cy=self.intrinsic_matrix[1, 2]
        )
        
    def create_tsdf_volume(self):
        """
        Create a TSDF volume for integration
        """
        # We're using the scalable TSDF volume for larger scenes
        print(f"Creating TSDF volume with voxel_size={self.voxel_size}, sdf_trunc={self.sdf_trunc}")
        self.volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=self.voxel_size,
            sdf_trunc=self.sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )
        
    def load_rgbd_images(self, view):
        """
        Load RGB and depth images for a given view
        
        Args:
            view (dict): View information containing file paths and pose
            
        Returns:
            tuple: (RGBD image, 4x4 extrinsic matrix)
        """
        # Get view_id for constructing file paths if needed
        view_id = view.get('view_id', 0)
        
        # Extract capture directory name for path processing
        capture_dir_name = os.path.basename(self.data_dir)
        
        # Load RGB image - extract path from view info
        if 'image_path' in view:
            # Get the relative path from the view info
            img_rel_path = view['image_path']
            
            # Normalize path separators
            img_rel_path = img_rel_path.replace('\\', '/').replace('//', '/')
            
            # Check if the path already contains the capture directory
            if capture_dir_name in img_rel_path:
                # Extract just the part after the capture directory name
                parts = img_rel_path.split('/')
                try:
                    capture_index = parts.index(capture_dir_name)
                    # Use only the part after capture_dir_name
                    img_rel_path = '/'.join(parts[capture_index+1:])
                except ValueError:
                    # If exact match not found, just use the basename
                    img_rel_path = os.path.basename(img_rel_path)
            
            # Construct the full path
            color_path = os.path.join(self.data_dir, img_rel_path)
        else:
            # Fallback to a default naming convention
            color_path = os.path.join(self.images_dir, f"view_{view_id:03d}.jpg")
        
        # Load depth image - extract path from view info using the same approach
        if 'depth_path' in view:
            depth_rel_path = view['depth_path']
            
            # Normalize path separators
            depth_rel_path = depth_rel_path.replace('\\', '/').replace('//', '/')
            
            # Check if the path already contains the capture directory
            if capture_dir_name in depth_rel_path:
                # Extract just the part after the capture directory name
                parts = depth_rel_path.split('/')
                try:
                    capture_index = parts.index(capture_dir_name)
                    # Use only the part after capture_dir_name
                    depth_rel_path = '/'.join(parts[capture_index+1:])
                except ValueError:
                    # If exact match not found, just use the basename
                    depth_rel_path = os.path.basename(depth_rel_path)
            
            # Construct the full path
            depth_path = os.path.join(self.data_dir, depth_rel_path)
        else:
            # Fallback to a default naming convention
            depth_path = os.path.join(self.data_dir, "depth", f"depth_{view_id:03d}.npy")
        
        # Check if the files exist
        if not os.path.exists(color_path):
            # Try an alternative path if the first attempt failed
            alt_color_path = os.path.join(self.data_dir, "images", f"view_{view_id:03d}.jpg")
            if os.path.exists(alt_color_path):
                color_path = alt_color_path
                print(f"Using alternative RGB path: {color_path}")
            else:
                raise FileNotFoundError(f"RGB image not found: {color_path}")
            
        if not os.path.exists(depth_path):
            # Try an alternative path if the first attempt failed
            alt_depth_path = os.path.join(self.data_dir, "depth", f"depth_{view_id:03d}.npy")
            if os.path.exists(alt_depth_path):
                depth_path = alt_depth_path
                print(f"Using alternative depth path: {depth_path}")
            else:
                raise FileNotFoundError(f"Depth map not found: {depth_path}")
        
        # Load RGB image
        color_image = cv2.imread(color_path)
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        
        # Load depth image
        if depth_path.endswith('.npy'):
            depth_image = np.load(depth_path)
        else:
            depth_image = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
        
        # Analyze the depth map to determine scale
        if depth_image.size > 0:
            depth_max = np.max(depth_image[depth_image > 0]) if np.any(depth_image > 0) else 0
            depth_min = np.min(depth_image[depth_image > 0]) if np.any(depth_image > 0) else 0
            print(f"View {view_id} - Depth range: [{depth_min}, {depth_max}], with {np.count_nonzero(depth_image > 0)} non-zero values")
            
            # If depth values are very large (likely in millimeters), convert to meters
            if depth_max > 100:  # Assuming depth shouldn't be more than 100 meters
                depth_scale = 1000.0  # millimeters to meters
                print(f"View {view_id} - Converting depth from millimeters to meters (scale: 1/1000)")
            else:
                depth_scale = 1.0
        else:
            depth_scale = 1.0
        
        # Convert to meters if needed
        depth_image = depth_image.astype(np.float32) / depth_scale
        
        # Clip depth values based on valid range
        depth_image[depth_image < self.min_depth] = 0
        depth_image[depth_image > self.max_depth] = 0
        
        # Create Open3D images
        color_o3d = o3d.geometry.Image(color_image)
        depth_o3d = o3d.geometry.Image(depth_image)
        
        # Create RGBD image - depth_scale=1.0 because we've already scaled the depth
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d, depth_o3d,
            depth_scale=1.0,  # Depth is already in correct units
            depth_trunc=self.max_depth,
            convert_rgb_to_intensity=False
        )
        
        # Get extrinsic matrix (camera pose)
        if 'extrinsic_matrix' in view:
            extrinsic = np.array(view['extrinsic_matrix'])
        else:
            extrinsic = np.array(view['extrinsic'])
        
        return rgbd, extrinsic
        
    def integrate_views(self):
        """
        Integrate all views into the TSDF volume
        """
        print(f"Integrating {len(self.views)} views into TSDF volume...")
        
        # Create TSDF volume
        self.create_tsdf_volume()
        
        # Track valid integration count
        valid_integrations = 0
        
        # Integrate each view
        for i, view in enumerate(tqdm(self.views)):
            try:
                # Load RGBD image and camera pose
                rgbd, extrinsic = self.load_rgbd_images(view)
                
                # Debug: Check if depth image has valid data
                depth_np = np.asarray(rgbd.depth)
                num_valid_depth = np.count_nonzero(depth_np > 0)
                if num_valid_depth < 100:  # Arbitrary threshold for a meaningful depth image
                    print(f"WARNING: View {i} has very few valid depth pixels ({num_valid_depth})")
                    continue
                
                # Integrate into volume
                self.volume.integrate(rgbd, self.intrinsic, extrinsic)
                valid_integrations += 1
            except Exception as e:
                print(f"Error integrating view {i}: {str(e)}")
                continue
                
        print(f"Successfully integrated {valid_integrations} out of {len(self.views)} views")
        
    def extract_mesh(self):
        """
        Extract mesh from TSDF volume
        
        Returns:
            o3d.geometry.TriangleMesh: Reconstructed mesh
        """
        print("Extracting mesh from TSDF volume...")
        mesh = self.volume.extract_triangle_mesh()
        
        # Check if mesh has vertices
        if len(mesh.vertices) == 0:
            print("ERROR: Extracted mesh has 0 vertices!")
            print("Trying to visualize the TSDF volume directly...")
            
            # Try to visualize the TSDF volume as a point cloud
            pc = self.volume.extract_point_cloud()
            if len(pc.points) > 0:
                print(f"Successfully extracted point cloud with {len(pc.points)} points.")
                print("Visualizing point cloud instead of mesh...")
                o3d.visualization.draw_geometries([pc])
            else:
                print("ERROR: TSDF volume is empty. No points or mesh could be extracted.")
                print("This is likely due to:")
                print("1. Depth values are out of range or invalid")
                print("2. Camera extrinsic matrices are incorrect")
                print("3. TSDF parameters (voxel_size/sdf_trunc) are inappropriate")
                
                # Try to create a simple point cloud from the first valid view
                # to diagnose camera and depth issues
                for i, view in enumerate(self.views):
                    try:
                        rgbd, extrinsic = self.load_rgbd_images(view)
                        depth_np = np.asarray(rgbd.depth)
                        if np.count_nonzero(depth_np > 0) > 100:
                            print(f"Creating debug point cloud from view {i}...")
                            pcd = o3d.geometry.PointCloud.create_from_depth_image(
                                rgbd.depth, 
                                self.intrinsic, 
                                extrinsic,
                                depth_scale=1.0,
                                depth_trunc=self.max_depth
                            )
                            o3d.visualization.draw_geometries([pcd])
                            break
                    except Exception as e:
                        continue
        else:
            # Optional: Refine mesh
            print(f"Mesh extracted with {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles.")
            print("Refining mesh...")
            mesh.compute_vertex_normals()
            mesh.remove_degenerate_triangles()
            mesh.remove_duplicated_vertices()
            mesh.remove_duplicated_triangles()
            mesh.remove_unreferenced_vertices()
            print(f"After refinement: {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles.")
        
        self.mesh = mesh
        return mesh
    
    def save_mesh(self, filename="tsdf_mesh.ply"):
        """
        Save the reconstructed mesh to file
        
        Args:
            filename (str): Filename for the mesh (default: "tsdf_mesh.ply")
        """
        output_path = os.path.join(self.output_dir, filename)
        print(f"Saving mesh to {output_path}")
        o3d.io.write_triangle_mesh(output_path, self.mesh)
        
    def visualize_mesh(self):
        """
        Visualize the reconstructed mesh
        """
        if self.mesh is None:
            print("No mesh to visualize. Run reconstruct() first.")
            return
        
        if len(self.mesh.vertices) == 0:
            print("Mesh has 0 vertices. Nothing to visualize.")
            return
            
        print("Visualizing mesh...")
        print(f"Mesh has {len(self.mesh.vertices)} vertices and {len(self.mesh.triangles)} triangles")
        
        # Create a new visualizer for better control
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="TSDF Reconstruction", width=1280, height=720)
        
        # Add a coordinate frame to show origin and orientation
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        vis.add_geometry(frame)
        
        # Add a sphere at origin for reference
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
        sphere.paint_uniform_color([1, 0, 0])  # Red
        vis.add_geometry(sphere)
        
        # Get mesh bounds to help with camera positioning
        mesh_min_bound = self.mesh.get_min_bound()
        mesh_max_bound = self.mesh.get_max_bound()
        mesh_center = self.mesh.get_center()
        mesh_scale = np.linalg.norm(mesh_max_bound - mesh_min_bound)
        
        print(f"Mesh bounds: min={mesh_min_bound}, max={mesh_max_bound}")
        print(f"Mesh center: {mesh_center}")
        print(f"Mesh scale: {mesh_scale}")
        
        # If mesh is outside the typical viewing area, move it to origin
        if np.linalg.norm(mesh_center) > 10:
            print(f"Mesh is far from origin, translating for better visualization")
            self.mesh.translate(-mesh_center)
        
        # Paint the mesh to ensure it's visible (optional)
        if not self.mesh.has_vertex_colors():
            print("Mesh doesn't have colors, adding default coloring for visibility")
            self.mesh.paint_uniform_color([0.7, 0.7, 0.7])  # Light gray
        
        # Add the mesh
        vis.add_geometry(self.mesh)
        
        # Improve rendering options
        opt = vis.get_render_option()
        opt.mesh_show_back_face = True
        opt.background_color = np.array([0.1, 0.1, 0.1])  # Dark gray
        opt.point_size = 3.0
        opt.line_width = 2.0
        opt.show_coordinate_frame = True
        
        # Add scene lighting
        opt.light_on = True
        
        # Set reasonable viewpoint based on mesh bounds
        vis.poll_events()
        vis.update_renderer()
        ctr = vis.get_view_control()
        ctr.set_zoom(0.8)
        
        # Try to set the camera to look at the mesh
        ctr.set_lookat(mesh_center)
        
        print("Visualization window should be open now.")
        print("If you see only a gray screen, try rotating the view with your mouse.")
        print("Press 'H' in the visualization window to see keyboard/mouse controls.")
        
        # Run the visualization
        vis.run()
        vis.destroy_window()
        
    def visualize_camera_poses(self):
        """
        Visualize the camera poses used for reconstruction
        """
        # Create a visualization window
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        
        # Add a coordinate frame at the origin
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        vis.add_geometry(frame)
        
        # Create and add camera frustrums for each view
        for i, view in enumerate(self.views):
            # Get the camera extrinsic matrix
            if 'extrinsic_matrix' in view:
                extrinsic = np.array(view['extrinsic_matrix'])
            else:
                extrinsic = np.array(view['extrinsic'])
            
            # Create a small sphere at the camera location
            # The camera position is the inverse of the extrinsic translation
            cam_to_world = np.linalg.inv(extrinsic)
            cam_pos = cam_to_world[:3, 3]
            
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
            sphere.translate(cam_pos)
            
            # Use different colors for each camera
            color = plt.cm.jet(i / len(self.views))[:3]
            sphere.paint_uniform_color(color)
            
            vis.add_geometry(sphere)
        
        # Run the visualization
        vis.run()
        vis.destroy_window()
        
    def visualize_debug(self):
        """
        Visualize debug information to help diagnose rendering issues
        """
        print("Showing debug visualization...")
        
        # Create a visualization window
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Debug Visualization", width=1280, height=720)
        
        # Add a coordinate frame
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
        vis.add_geometry(frame)
        
        # Add camera positions
        for i, view in enumerate(self.views[:5]):  # Just first 5 for clarity
            if 'extrinsic_matrix' in view:
                extrinsic = np.array(view['extrinsic_matrix'])
            else:
                extrinsic = np.array(view['extrinsic'])
            
            # Create a small sphere at the camera position
            cam_to_world = np.linalg.inv(extrinsic)
            cam_pos = cam_to_world[:3, 3]
            
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
            sphere.translate(cam_pos)
            color = plt.cm.jet(i / min(5, len(self.views)))[:3]
            sphere.paint_uniform_color(color)
            
            vis.add_geometry(sphere)
            
            # Add a text label with the camera index
            # (This isn't directly supported in Open3D, so we'd need a custom solution)

        # If we have a mesh, add it
        if self.mesh is not None and len(self.mesh.vertices) > 0:
            # Make a copy of the mesh
            debug_mesh = copy.deepcopy(self.mesh)
            # Paint it a distinct color
            debug_mesh.paint_uniform_color([0, 1, 0])  # Green
            vis.add_geometry(debug_mesh)
                
        # Add a simple demo mesh to verify rendering works
        demo_mesh = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)
        demo_mesh.translate([3, 0, 0])  # Move to the right
        demo_mesh.paint_uniform_color([0, 0, 1])  # Blue
        vis.add_geometry(demo_mesh)
        
        # Set rendering options
        opt = vis.get_render_option()
        opt.background_color = np.array([0.1, 0.1, 0.1])
        opt.point_size = 5.0
        opt.show_coordinate_frame = True
        
        # Run the visualization
        vis.run()
        vis.destroy_window()
        
    def reconstruct(self, visualize=True):
        """
        Perform TSDF reconstruction from RGB-D images
        
        Args:
            visualize (bool): Whether to visualize the mesh after reconstruction
            
        Returns:
            o3d.geometry.TriangleMesh: Reconstructed mesh
        """
        start_time = time.time()
        
        # Integrate views into TSDF volume
        self.integrate_views()
        
        # Extract mesh
        mesh = self.extract_mesh()
        
        # Save mesh
        self.save_mesh()
        
        elapsed = time.time() - start_time
        print(f"Reconstruction completed in {elapsed:.2f} seconds")
        
        # Check if mesh has vertices
        if len(mesh.vertices) == 0:
            print("WARNING: Reconstructed mesh has no vertices.")
            print("This might be due to incorrect depth values or camera parameters.")
            print("Try adjusting the voxel size, truncation distance, or depth range parameters.")
        else:
            print(f"Mesh reconstructed with {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles.")
            
            # Visualize if requested
            if visualize:
                print("Visualizing reconstructed mesh...")
                self.visualize_mesh()
                
                # If we're still having issues, show debug visualization
                print("Do you want to see a debug visualization? (yes/no)")
                response = input().strip().lower()
                if response == "yes" or response == "y":
                    self.visualize_debug()
        
        return mesh

def main():
    """
    Main function for running TSDF reconstruction
    """
    parser = argparse.ArgumentParser(description="3D reconstruction using TSDF with RGB-D and camera parameters")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing RGB-D data and camera parameters")
    parser.add_argument("--voxel_size", type=float, default=0.01, help="Size of voxels for TSDF volume (in meters)")
    parser.add_argument("--sdf_trunc", type=float, default=0.03, help="Truncation value for signed distance function (in meters)")
    parser.add_argument("--max_depth", type=float, default=10.0, help="Maximum depth value to use (in meters)")
    parser.add_argument("--min_depth", type=float, default=0.05, help="Minimum depth value to use (in meters)")
    parser.add_argument("--visualize", action="store_true", default=True, help="Visualize the reconstructed mesh (on by default)")
    parser.add_argument("--no_visualize", action="store_false", dest="visualize", help="Don't visualize the reconstructed mesh")
    parser.add_argument("--visualize_cameras", action="store_true", help="Visualize camera positions")
    
    args = parser.parse_args()
    
    print(f"Starting TSDF reconstruction with parameters:")
    print(f"  Data directory: {args.data_dir}")
    print(f"  Voxel size: {args.voxel_size} meters")
    print(f"  SDF truncation: {args.sdf_trunc} meters")
    print(f"  Depth range: {args.min_depth} to {args.max_depth} meters")
    
    # Create TSDF reconstruction object
    tsdf = TSDFReconstruction(
        args.data_dir, 
        voxel_size=args.voxel_size, 
        sdf_trunc=args.sdf_trunc,
        max_depth=args.max_depth,
        min_depth=args.min_depth
    )
    
    # Visualize camera positions if requested
    if args.visualize_cameras:
        tsdf.visualize_camera_poses()
    
    # Perform reconstruction
    tsdf.reconstruct(visualize=args.visualize)
    
if __name__ == "__main__":
    main() 