#!/usr/bin/env python
import sys
import open3d as o3d

def visualize_point_cloud(ply_path):
    """
    Load and visualize a PLY point cloud file using Open3D
    
    Args:
        ply_path (str): Path to the PLY file
    """
    try:
        # Load the point cloud from the PLY file
        point_cloud = o3d.io.read_point_cloud(ply_path)
        
        if not point_cloud.has_points():
            print(f"Error: No points found in {ply_path}")
            sys.exit(1)
        
        # Print basic information about the point cloud
        print(f"Point cloud loaded: {ply_path}")
        print(f"Number of points: {len(point_cloud.points)}")
        print(f"Point cloud has colors: {point_cloud.has_colors()}")
        print(f"Point cloud has normals: {point_cloud.has_normals()}")
        
        # Create a coordinate frame for reference (RGB = XYZ)
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=max(point_cloud.get_max_bound() - point_cloud.get_min_bound()) * 0.1,
            origin=[0, 0, 0]
        )
        
        # Visualize the point cloud
        o3d.visualization.draw_geometries([point_cloud, coordinate_frame],
                                          window_name=f"Point Cloud Viewer - {ply_path}",
                                          width=1024,
                                          height=768)
    except Exception as e:
        print(f"Error visualizing point cloud: {e}")
        sys.exit(1)

def main():
    # Check if a PLY file is provided as a command-line argument
    if len(sys.argv) != 2:
        print("Usage: python visualize_pointcloud.py <path_to_ply_file>")
        sys.exit(1)
    
    ply_path = sys.argv[1]
    
    # Check if the file has a .ply extension
    if not ply_path.lower().endswith('.ply'):
        print("Warning: The input file does not have a .ply extension.")
        response = input("Do you want to continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(0)
    
    visualize_point_cloud(ply_path)

if __name__ == "__main__":
    main() 