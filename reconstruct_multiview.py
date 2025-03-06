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
        
        # Optional: Improve image contrast
        gray = cv2.equalizeHist(gray)
            
        # Create SIFT detector with more sensitive parameters
        sift = cv2.SIFT_create(
            nfeatures=0,          # Maximum number of features (0 = unlimited)
            nOctaveLayers=5,      # Default is 3
            contrastThreshold=0.02,  # Default is 0.04 (lower = more features)
            edgeThreshold=15,      # Default is 10
            sigma=1.6             # Default is 1.6
        )
        
        # Detect and compute keypoints and descriptors
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        
        print(f"Detected {len(keypoints) if keypoints is not None else 0} features")
        
        return keypoints, descriptors
    
    def match_features(self, desc1, desc2):
        """
        Match features between two images
        
        Args:
            desc1, desc2: Feature descriptors
            
        Returns:
            list of matches
        """
        if desc1 is None or desc2 is None:
            return []
            
        if len(desc1) < 2 or len(desc2) < 2:
            return []
            
        # Use FLANN for fast matching
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=100)  # Increased from 50
        
        try:
            # Try FLANN matcher first
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            
            # Find 2 best matches for each descriptor
            matches = flann.knnMatch(desc1, desc2, k=2)
            
            # Apply Lowe's ratio test with less strict threshold
            good_matches = []
            for m, n in matches:
                if m.distance < 0.8 * n.distance:  # Changed from 0.7 to 0.8
                    good_matches.append(m)
        except Exception as e:
            print(f"FLANN matcher failed: {e}. Trying Brute Force matcher...")
            # Fall back to Brute Force matcher
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(desc1, desc2, k=2)
            
            good_matches = []
            for m, n in matches:
                if m.distance < 0.85 * n.distance:  # Even less strict for BF matcher
                    good_matches.append(m)
        
        print(f"Found {len(good_matches)} good matches")
                
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
        # Save debug images
        debug_dir = os.path.join(self.output_dir, "debug")
        os.makedirs(debug_dir, exist_ok=True)
        
        # Detect features
        kp1, desc1 = self.detect_features(img1)
        kp2, desc2 = self.detect_features(img2)
        
        # Reduced minimum feature requirement from 5 to 4
        if desc1 is None or desc2 is None or len(kp1) < 4 or len(kp2) < 4:
            print("Not enough features detected")
            return None, None
        
        # Match features
        matches = self.match_features(desc1, desc2)
        
        # Reduced minimum matches requirement from 8 to 5
        if len(matches) < 5:
            print("Not enough good matches")
            return None, None
        
        # Draw matches for debugging
        try:
            match_img = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, 
                                        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            cv2.imwrite(os.path.join(debug_dir, f"matches_{len(matches)}.jpg"), match_img)
        except Exception as e:
            print(f"Could not save match debug image: {e}")
        
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
        print(f"Triangulated {len(points_3d)} points")
        
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
    
    def filter_points(self, points_3d, min_distance=0.0001, max_distance=100.0):
        """
        Filter out outlier points
        
        Args:
            points_3d: 3D points
            min_distance: Minimum distance from origin (reduced from 0.001)
            max_distance: Maximum distance from origin (increased from 20.0)
            
        Returns:
            Filtered points, corresponding indices
        """
        if points_3d is None or len(points_3d) == 0:
            return None, []
            
        # Calculate distances from origin
        distances = np.linalg.norm(points_3d, axis=1)
        
        # Filter points with a much more relaxed criteria
        valid_indices = np.where((distances >= min_distance) & (distances <= max_distance))[0]
        filtered_points = points_3d[valid_indices]
        
        # Apply statistical outlier removal to eliminate extreme outliers while keeping most points
        if len(filtered_points) > 10:  # Need sufficient points for statistical analysis
            # Calculate median distance and median absolute deviation
            median_distance = np.median(distances[valid_indices])
            mad = np.median(np.abs(distances[valid_indices] - median_distance))
            
            # Use a generous threshold of median ± 10 * MAD
            # This will keep most points while removing extreme outliers
            inlier_indices = np.where(
                np.abs(distances[valid_indices] - median_distance) < 10.0 * mad
            )[0]
            
            filtered_points = filtered_points[inlier_indices]
            valid_indices = valid_indices[inlier_indices]
        
        print(f"Filtered points: {len(filtered_points)}/{len(points_3d)} points kept")
        
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
        print(f"Images directory: {self.images_dir}")
        
        # Try different pairs, not just consecutive ones
        pairs_to_try = []
        
        # Add consecutive pairs (i, i+1)
        for i in range(len(self.views) - 1):
            pairs_to_try.append((i, i+1))
            
        # Add some pairs with larger baseline (i, i+2) and (i, i+3)
        for i in range(len(self.views) - 2):
            pairs_to_try.append((i, i+2))
            
        for i in range(len(self.views) - 3):
            pairs_to_try.append((i, i+3))
        
        # Process pairs of views
        for i, j in tqdm(pairs_to_try):
            # Get current and next view
            view1 = self.views[i]
            view2 = self.views[j]
            
            print(f"\nProcessing view pair ({i}, {j})")
            
            # Load images
            # Check if the image_path contains the data_dir already
            img1_path = view1['image_path']
            img2_path = view2['image_path']
            
            # If the path is relative (doesn't start with data_dir), join with data_dir
            if self.data_dir not in img1_path:
                img1_path = os.path.join(self.images_dir, os.path.basename(img1_path))
            if self.data_dir not in img2_path:
                img2_path = os.path.join(self.images_dir, os.path.basename(img2_path))
            
            if not os.path.exists(img1_path) or not os.path.exists(img2_path):
                print(f"Warning: Image not found, skipping pair {i}, {j}")
                print(f"Searched for: {img1_path} and {img2_path}")
                continue
                
            img1 = cv2.imread(img1_path)
            img2 = cv2.imread(img2_path)
            
            if img1 is None or img2 is None:
                print(f"Warning: Failed to load images, skipping pair {i}, {j}")
                continue
                
            print(f"Loaded images: {os.path.basename(img1_path)} ({img1.shape}) and {os.path.basename(img2_path)} ({img2.shape})")
            
            # Get camera poses
            extrinsic1 = np.array(view1['extrinsic_matrix'])
            extrinsic2 = np.array(view2['extrinsic_matrix'])
            
            # Extract rotation and translation
            R1 = extrinsic1[:3, :3]
            t1 = extrinsic1[:3, 3]
            
            R2 = extrinsic2[:3, :3]
            t2 = extrinsic2[:3, 3]
            
            # Calculate baseline (distance between cameras)
            baseline = np.linalg.norm(t2 - t1)
            print(f"Baseline distance: {baseline:.4f}")
            
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
        
        # Create a cache directory for normals if it doesn't exist
        cache_dir = os.path.join(self.output_dir, "cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        # Cache filename for the point cloud with normals
        normals_cache_path = os.path.join(cache_dir, "reconstruction_with_normals.ply")
        
        # Check if we have a cached version with normals
        if os.path.exists(normals_cache_path):
            print(f"Loading point cloud with cached normals from: {normals_cache_path}")
            try:
                cached_pcd = o3d.io.read_point_cloud(normals_cache_path)
                
                # Verify the cached point cloud has normals and points
                if cached_pcd.has_normals() and len(cached_pcd.points) > 0:
                    # Make sure the cached cloud has the same number of points
                    if len(cached_pcd.points) == len(pcd.points):
                        print("Using cached normals")
                        pcd = cached_pcd
                    else:
                        print("Cached point cloud has different number of points. Recomputing normals...")
                        pcd = self.compute_and_cache_normals(pcd, normals_cache_path)
                else:
                    print("Cached point cloud doesn't have normals or points. Recomputing...")
                    pcd = self.compute_and_cache_normals(pcd, normals_cache_path)
            except Exception as e:
                print(f"Error loading cached normals: {e}. Recomputing...")
                pcd = self.compute_and_cache_normals(pcd, normals_cache_path)
        else:
            # Compute and cache normals
            pcd = self.compute_and_cache_normals(pcd, normals_cache_path)
        
        # Save the point cloud
        output_ply = os.path.join(self.output_dir, "reconstruction.ply")
        o3d.io.write_point_cloud(output_ply, pcd)
        
        self.point_cloud = pcd
        print(f"Reconstruction completed. Point cloud saved to: {output_ply}")
        print(f"Number of points: {len(pcd.points)}")
        
        # Add information about point cloud scale to help diagnose the "too far" issue
        if len(pcd.points) > 0:
            # Calculate bounding box
            bbox = pcd.get_axis_aligned_bounding_box()
            min_bound = bbox.min_bound
            max_bound = bbox.max_bound
            extent = bbox.get_extent()
            
            print(f"Point cloud bounding box:")
            print(f"  Min bounds: [{min_bound[0]:.2f}, {min_bound[1]:.2f}, {min_bound[2]:.2f}]")
            print(f"  Max bounds: [{max_bound[0]:.2f}, {max_bound[1]:.2f}, {max_bound[2]:.2f}]")
            print(f"  Size: [{extent[0]:.2f}, {extent[1]:.2f}, {extent[2]:.2f}]")
            
            # Calculate distance from origin to help understand scale issues
            distances = np.linalg.norm(np.asarray(pcd.points), axis=1)
            avg_distance = np.mean(distances)
            print(f"Average distance from origin: {avg_distance:.2f}")
            print(f"Min distance from origin: {np.min(distances):.2f}")
            print(f"Max distance from origin: {np.max(distances):.2f}")
            
        return True
    
    def compute_and_cache_normals(self, pcd, cache_path):
        """
        Compute normals for a point cloud and cache the result
        
        Args:
            pcd: Point cloud to compute normals for
            cache_path: Path to save the cached point cloud with normals
        """
        print("Estimating normals...")
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        
        # Save to cache
        print(f"Saving point cloud with normals to cache: {cache_path}")
        o3d.io.write_point_cloud(cache_path, pcd)
        
        return pcd
    
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