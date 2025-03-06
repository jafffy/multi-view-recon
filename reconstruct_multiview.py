#!/usr/bin/env python
import os
import cv2
import numpy as np
import json
import argparse
import open3d as o3d
from tqdm import tqdm
from pathlib import Path

class MultiViewReconstruction:
    def __init__(self, data_dir):
        """
        Initialize the multi-view reconstruction system
        
        Args:
            data_dir (str): Directory containing the capture data (images and camera parameters)
        """
        self.data_dir = data_dir
        self.images_dir = os.path.join(data_dir, "images")
        self.output_dir = os.path.join(data_dir, "reconstruction")
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load camera parameters
        self.load_camera_parameters()
        
        # Initialize reconstruction parameters
        self.point_cloud = None
        self.point_colors = None
        
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
        
        print(f"Loaded {len(self.views)} camera views from {params_path}")
        print(f"Camera intrinsic matrix:\n{self.intrinsic_matrix}")
    
    def detect_features(self, image):
        """
        Detect features in an image
        
        Args:
            image: Input image
            
        Returns:
            keypoints, descriptors
        """
        # Convert image to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Create SIFT detector
        sift = cv2.SIFT_create()
        
        # Detect and compute keypoints and descriptors
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        
        return keypoints, descriptors
    
    def match_features(self, desc1, desc2):
        """
        Match features between two images
        
        Args:
            desc1, desc2: Feature descriptors
            
        Returns:
            list of matches
        """
        # Use FLANN for fast matching
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        # Find 2 best matches for each descriptor
        matches = flann.knnMatch(desc1, desc2, k=2)
        
        # Apply Lowe's ratio test to filter good matches
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
                
        return good_matches
    
    def triangulate_points(self, img1, img2, K, dist_coef, R1, t1, R2, t2):
        """
        Triangulate 3D points from two views
        
        Args:
            img1, img2: Input images
            K: Camera intrinsic matrix
            dist_coef: Distortion coefficients
            R1, t1: Rotation and translation for first camera
            R2, t2: Rotation and translation for second camera
            
        Returns:
            3D points, corresponding point colors
        """
        # Detect features
        kp1, desc1 = self.detect_features(img1)
        kp2, desc2 = self.detect_features(img2)
        
        if desc1 is None or desc2 is None or len(kp1) < 5 or len(kp2) < 5:
            print("Not enough features detected")
            return None, None
        
        # Match features
        matches = self.match_features(desc1, desc2)
        
        if len(matches) < 8:
            print("Not enough good matches")
            return None, None
        
        # Extract matched points
        points1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        points2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        
        # Undistort points
        points1_undist = cv2.undistortPoints(points1.reshape(-1, 1, 2), K, dist_coef)
        points2_undist = cv2.undistortPoints(points2.reshape(-1, 1, 2), K, dist_coef)
        
        # Create projection matrices
        P1 = np.hstack((R1, t1.reshape(-1, 1)))
        P2 = np.hstack((R2, t2.reshape(-1, 1)))
        
        # Triangulate points
        points_4d = cv2.triangulatePoints(
            P1, P2, 
            points1_undist.reshape(-1, 2).T,
            points2_undist.reshape(-1, 2).T
        )
        
        # Convert to 3D points
        points_3d = cv2.convertPointsFromHomogeneous(points_4d.T).reshape(-1, 3)
        
        # Get colors from first image
        points_colors = []
        for pt in points1.astype(int):
            x, y = pt
            if 0 <= x < img1.shape[1] and 0 <= y < img1.shape[0]:
                if len(img1.shape) == 3:
                    color = img1[y, x]
                    # Convert from BGR to RGB
                    color = color[::-1]
                else:
                    # Grayscale image
                    color = np.array([img1[y, x], img1[y, x], img1[y, x]])
                points_colors.append(color)
            else:
                points_colors.append(np.array([128, 128, 128]))  # Default gray
        
        return points_3d, np.array(points_colors)
    
    def filter_points(self, points_3d, min_distance=0.01, max_distance=10.0):
        """
        Filter out outlier points
        
        Args:
            points_3d: 3D points
            min_distance: Minimum distance from origin
            max_distance: Maximum distance from origin
            
        Returns:
            Filtered points, corresponding indices
        """
        if points_3d is None or len(points_3d) == 0:
            return None, []
            
        # Calculate distances from origin
        distances = np.linalg.norm(points_3d, axis=1)
        
        # Filter points
        valid_indices = np.where((distances >= min_distance) & (distances <= max_distance))[0]
        filtered_points = points_3d[valid_indices]
        
        return filtered_points, valid_indices
    
    def reconstruct(self):
        """
        Perform multi-view reconstruction
        
        Returns:
            True if reconstruction was successful
        """
        if len(self.views) < 2:
            print("At least two views are required for reconstruction")
            return False
            
        all_points_3d = []
        all_point_colors = []
        
        print(f"Reconstructing from {len(self.views)} views...")
        
        # Process pairs of views
        for i in tqdm(range(len(self.views) - 1)):
            # Get current and next view
            view1 = self.views[i]
            view2 = self.views[i + 1]
            
            # Load images
            img1_path = os.path.join(self.data_dir, view1['image_path'])
            img2_path = os.path.join(self.data_dir, view2['image_path'])
            
            if not os.path.exists(img1_path) or not os.path.exists(img2_path):
                print(f"Warning: Image not found, skipping pair {i}, {i+1}")
                continue
                
            img1 = cv2.imread(img1_path)
            img2 = cv2.imread(img2_path)
            
            if img1 is None or img2 is None:
                print(f"Warning: Failed to load images, skipping pair {i}, {i+1}")
                continue
            
            # Get camera poses
            extrinsic1 = np.array(view1['extrinsic_matrix'])
            extrinsic2 = np.array(view2['extrinsic_matrix'])
            
            # Extract rotation and translation
            R1 = extrinsic1[:3, :3]
            t1 = extrinsic1[:3, 3]
            
            R2 = extrinsic2[:3, :3]
            t2 = extrinsic2[:3, 3]
            
            # Triangulate points between this pair of views
            points_3d, point_colors = self.triangulate_points(
                img1, img2, 
                self.intrinsic_matrix, self.distortion_coeffs,
                R1, t1, R2, t2
            )
            
            if points_3d is not None and len(points_3d) > 0:
                # Filter points
                filtered_points, valid_indices = self.filter_points(points_3d)
                
                if filtered_points is not None and len(filtered_points) > 0:
                    all_points_3d.append(filtered_points)
                    all_point_colors.append(point_colors[valid_indices])
        
        if not all_points_3d:
            print("No points could be reconstructed")
            return False
            
        # Combine all points
        combined_points = np.vstack(all_points_3d)
        combined_colors = np.vstack(all_point_colors)
        
        # Normalize colors to 0-1 range
        combined_colors = combined_colors.astype(np.float64) / 255.0
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(combined_points)
        pcd.colors = o3d.utility.Vector3dVector(combined_colors)
        
        # Remove statistical outliers
        print("Removing outliers...")
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        
        # Estimate normals
        print("Estimating normals...")
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        
        # Save the point cloud
        output_ply = os.path.join(self.output_dir, "reconstruction.ply")
        o3d.io.write_point_cloud(output_ply, pcd)
        
        self.point_cloud = pcd
        print(f"Reconstruction completed. Point cloud saved to: {output_ply}")
        print(f"Number of points: {len(pcd.points)}")
        
        return True
    
    def visualize(self):
        """
        Visualize the reconstructed point cloud
        """
        if self.point_cloud is None:
            print("No point cloud to visualize")
            return
            
        # Create coordinate frame for reference
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=max(self.point_cloud.get_max_bound() - self.point_cloud.get_min_bound()) * 0.1,
            origin=[0, 0, 0]
        )
        
        # Visualize the point cloud
        o3d.visualization.draw_geometries(
            [self.point_cloud, coordinate_frame],
            window_name="Multi-View Reconstruction",
            width=1024,
            height=768
        )

def main():
    parser = argparse.ArgumentParser(description='Multi-view reconstruction tool')
    parser.add_argument('--data', '-d', type=str, required=True,
                        help='Directory containing capture data (images and camera parameters)')
    parser.add_argument('--visualize', '-v', action='store_true',
                        help='Visualize the reconstruction result')
    
    args = parser.parse_args()
    
    try:
        # Initialize reconstruction
        mvr = MultiViewReconstruction(data_dir=args.data)
        
        # Perform reconstruction
        if mvr.reconstruct() and args.visualize:
            # Visualize the result
            mvr.visualize()
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 