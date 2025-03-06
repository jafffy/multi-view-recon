#!/usr/bin/env python
import os
import cv2
import numpy as np
import json
import argparse
import open3d as o3d
from tqdm import tqdm
from pathlib import Path
import copy

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
    
    def enhance_rendered_image(self, image):
        """
        Enhance a rendered image to make feature detection more effective
        
        Args:
            image: Input image
            
        Returns:
            Enhanced image
        """
        if image is None:
            return None
            
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # Apply a series of enhancements to make features more detectable
        
        # 1. Increase contrast
        gray = cv2.equalizeHist(gray)
        
        # 2. Apply morphological operations to enhance point cloud dots
        kernel = np.ones((3, 3), np.uint8)
        gray = cv2.dilate(gray, kernel, iterations=1)
        
        # 3. Apply bilateral filter to smooth while preserving edges
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # 4. Apply adaptive thresholding to enhance edges and features
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
                                     
        # 5. Find contours to enhance edges
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        edge_image = np.zeros_like(gray)
        cv2.drawContours(edge_image, contours, -1, 255, 1)
        
        # 6. Combine original with edge-enhanced version
        enhanced = cv2.addWeighted(gray, 0.7, edge_image, 0.3, 0)
        
        # Save debug images
        debug_dir = os.path.join(self.output_dir, "debug")
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(os.path.join(debug_dir, "original.jpg"), image)
        cv2.imwrite(os.path.join(debug_dir, "enhanced.jpg"), enhanced)
        
        return enhanced
        
    def detect_features(self, image):
        """
        Detect features in an image
        
        Args:
            image: Input image
            
        Returns:
            keypoints, descriptors
        """
        # First enhance the image if it's a rendered point cloud
        enhanced_image = self.enhance_rendered_image(image)
        
        # Convert image to grayscale
        if len(enhanced_image.shape) == 3:
            gray = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = enhanced_image
        
        # Create SIFT detector with much more sensitive parameters for point cloud renders
        sift = cv2.SIFT_create(
            nfeatures=0,          # Maximum number of features (0 = unlimited)
            nOctaveLayers=5,      # Default is 3
            contrastThreshold=0.01,  # Reduced from 0.02 (lower = more features)
            edgeThreshold=20,      # Increased from 15 (higher = less edge filtering)
            sigma=1.4             # Slightly reduced from 1.6 to detect finer details
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
            
        # Only require a minimum of 1 descriptor per image (reduced from 2)
        if len(desc1) < 1 or len(desc2) < 1:
            return []
            
        # Use FLANN for fast matching
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=150)  # Increased from 100 for better quality matches
        
        try:
            # Try FLANN matcher first
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            
            # Find 2 best matches for each descriptor
            matches = flann.knnMatch(desc1, desc2, k=2)
            
            # Apply Lowe's ratio test with much less strict threshold for point cloud renders
            good_matches = []
            for m, n in matches:
                if m.distance < 0.9 * n.distance:  # Changed from 0.8 to 0.9 (more lenient)
                    good_matches.append(m)
                    
            # If we have very few matches, try with an even less strict threshold
            if len(good_matches) < 10:
                good_matches = []
                for m, n in matches:
                    if m.distance < 0.95 * n.distance:  # Very lenient for sparse features
                        good_matches.append(m)
                        
        except Exception as e:
            print(f"FLANN matcher failed: {e}. Trying Brute Force matcher...")
            # Fall back to Brute Force matcher
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(desc1, desc2, k=2)
            
            good_matches = []
            for m, n in matches:
                if m.distance < 0.9 * n.distance:  # Even less strict for BF matcher
                    good_matches.append(m)
                    
            # If we have very few matches, try with an even less strict threshold
            if len(good_matches) < 10:
                good_matches = []
                for m, n in matches:
                    if m.distance < 0.95 * n.distance:  # Very lenient for sparse features
                        good_matches.append(m)
        
        print(f"Found {len(good_matches)} good matches")
                
        return good_matches
    
    def triangulate_points(self, img1, img2, K, dist_coef, R1, t1, R2, t2):
        """
        Triangulate 3D points from two views with robust RANSAC-based essential matrix estimation
        
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
        
        # Reduced minimum feature requirement to absolute minimum
        if desc1 is None or desc2 is None or len(kp1) < 3 or len(kp2) < 3:
            print("Not enough features detected")
            return None, None
        
        # Match features
        matches = self.match_features(desc1, desc2)
        
        # Reduced minimum matches requirement to absolute minimum for triangulation
        if len(matches) < 4:
            print("Not enough good matches")
            return None, None
        
        # Save original images with detected keypoints for debugging
        img1_kp = cv2.drawKeypoints(img1, kp1, None, color=(0,255,0), 
                                   flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
        img2_kp = cv2.drawKeypoints(img2, kp2, None, color=(0,255,0), 
                                   flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
        
        cv2.imwrite(os.path.join(debug_dir, f"keypoints_img1_{len(kp1)}.jpg"), img1_kp)
        cv2.imwrite(os.path.join(debug_dir, f"keypoints_img2_{len(kp2)}.jpg"), img2_kp)
        
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
        
        # Convert undistorted points to the right format for essential matrix estimation
        points1_norm = points1_undist.reshape(-1, 2)
        points2_norm = points2_undist.reshape(-1, 2)
        
        # Add RANSAC-based essential matrix estimation
        try:
            # Use undistorted normalized points for essential matrix estimation
            E, mask = cv2.findEssentialMat(
                points1_norm, points2_norm, 
                np.eye(3),  # Identity since points are already normalized
                method=cv2.RANSAC, 
                prob=0.999, 
                threshold=0.005  # Adjust based on your data
            )
            
            # Get inlier matches
            inlier_mask = mask.ravel() == 1
            inlier_points1 = points1_norm[inlier_mask]
            inlier_points2 = points2_norm[inlier_mask]
            inlier_matches = [m for i, m in enumerate(matches) if inlier_mask[i]]
            
            print(f"Essential matrix inliers: {np.sum(inlier_mask)} out of {len(matches)} matches")
            
            # Draw inlier matches for debugging
            try:
                inlier_match_img = cv2.drawMatches(img1, kp1, img2, kp2, inlier_matches, None, 
                                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                cv2.imwrite(os.path.join(debug_dir, f"inlier_matches_{len(inlier_matches)}.jpg"), inlier_match_img)
            except Exception as e:
                print(f"Could not save inlier match debug image: {e}")
            
            # Check if we have enough inliers
            if len(inlier_points1) < 4:
                print("Not enough inlier matches after RANSAC, falling back to all matches")
                # Fall back to using all matches if not enough inliers
                inlier_points1 = points1_norm
                inlier_points2 = points2_norm
        except Exception as e:
            print(f"Essential matrix estimation failed: {e}. Using all matches.")
            inlier_points1 = points1_norm
            inlier_points2 = points2_norm
            
        # Create projection matrices
        P1 = np.hstack((R1, t1.reshape(-1, 1)))
        P2 = np.hstack((R2, t2.reshape(-1, 1)))
        
        # Triangulate points using inliers
        points_4d = cv2.triangulatePoints(
            P1, P2, 
            inlier_points1.T,
            inlier_points2.T
        )
        
        # Convert to 3D points
        points_3d = cv2.convertPointsFromHomogeneous(points_4d.T).reshape(-1, 3)
        print(f"Triangulated {len(points_3d)} points from inliers")
        
        # Get original image indices for colors (need to map back to original points)
        if inlier_points1.shape[0] != points1.shape[0]:
            # We're using inliers, so we need to get the original indices
            inlier_indices = np.where(inlier_mask)[0]
            original_points = points1[inlier_indices].astype(int)
        else:
            # We're using all points
            original_points = points1.astype(int)
        
        # Get colors from first image
        points_colors = []
        for pt in original_points:
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
    
    def filter_points(self, points_3d, min_distance=0.000001, max_distance=1000.0):
        """
        Filter out outlier points
        
        Args:
            points_3d: 3D points
            min_distance: Minimum distance from origin (drastically reduced)
            max_distance: Maximum distance from origin (drastically increased)
            
        Returns:
            Filtered points, corresponding indices
        """
        if points_3d is None or len(points_3d) == 0:
            return None, []

        # Temporarily disable filtering for debug purposes if you want to see all points
        # return points_3d, np.arange(len(points_3d))
        
        # Calculate distances from origin
        distances = np.linalg.norm(points_3d, axis=1)
        
        # Filter points with extremely relaxed criteria
        valid_indices = np.where((distances >= min_distance) & (distances <= max_distance))[0]
        filtered_points = points_3d[valid_indices]
        
        # Only apply statistical outlier removal if we have a lot of points
        # For sparse point clouds, we want to keep almost everything
        if len(filtered_points) > 50:  # Only filter if we have a good number of points
            # Calculate median distance and median absolute deviation
            median_distance = np.median(distances[valid_indices])
            mad = np.median(np.abs(distances[valid_indices] - median_distance))
            
            # Use an extremely generous threshold of median ± 20 * MAD
            # This will only filter the most extreme outliers
            inlier_indices = np.where(
                np.abs(distances[valid_indices] - median_distance) < 20.0 * mad
            )[0]
            
            filtered_points = filtered_points[inlier_indices]
            valid_indices = valid_indices[inlier_indices]
        
        print(f"Filtered points: {len(filtered_points)}/{len(points_3d)} points kept ({len(filtered_points)/len(points_3d)*100:.1f}%)")
        
        return filtered_points, valid_indices
    
    def reconstruct(self, use_ransac_pose=True):
        """
        Perform multi-view reconstruction
        
        Args:
            use_ransac_pose: Whether to use RANSAC-based pose estimation as a fallback
            
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
            
        # Add pairs with larger baseline (i, i+2), (i, i+3), (i, i+4)
        for i in range(len(self.views) - 2):
            pairs_to_try.append((i, i+2))
            
        for i in range(len(self.views) - 3):
            pairs_to_try.append((i, i+3))
            
        for i in range(len(self.views) - 4):
            pairs_to_try.append((i, i+4))
            
        # Add some pairs with even larger baselines for better triangulation
        # These pairs are more likely to have good parallax
        step = max(1, len(self.views) // 8)  # Divide the view set into roughly 8 segments
        for i in range(0, len(self.views) - step, step):
            pairs_to_try.append((i, i + step))
            
        # Add some pairs from beginning to end for loop closure
        if len(self.views) >= 8:
            pairs_to_try.append((0, len(self.views) - 1))
            pairs_to_try.append((0, len(self.views) - 2))
            pairs_to_try.append((1, len(self.views) - 1))
        
        print(f"Will try {len(pairs_to_try)} view pairs for reconstruction")
        
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
            
            # Get camera poses from provided extrinsics
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
            
            # If triangulation failed with provided poses or yielded too few points, try RANSAC pose estimation
            if (points_3d is None or len(points_3d) < 10) and use_ransac_pose:
                print("Triangulation with provided poses failed or yielded too few points.")
                print("Trying RANSAC-based pose estimation...")
                
                # Detect features and match
                kp1, desc1 = self.detect_features(img1)
                kp2, desc2 = self.detect_features(img2)
                
                if desc1 is not None and desc2 is not None and len(kp1) >= 8 and len(kp2) >= 8:
                    matches = self.match_features(desc1, desc2)
                    
                    if len(matches) >= 8:  # Minimum points for essential matrix
                        # Extract matched points
                        points1 = np.float32([kp1[m.queryIdx].pt for m in matches])
                        points2 = np.float32([kp2[m.trainIdx].pt for m in matches])
                        
                        # Use RANSAC to recover pose
                        try:
                            # Get pose from essential matrix
                            R_ransac, t_ransac, inlier_mask = self.recover_pose_from_essential_matrix(
                                points1, points2, self.intrinsic_matrix
                            )
                            
                            print("RANSAC pose estimation successful, using recovered pose for triangulation")
                            
                            # Keep R1 and t1 as the reference camera (usually identity rotation and zero translation)
                            # Set R2 and t2 based on the recovered relative pose
                            R2_ransac = R1 @ R_ransac  # Compose rotations
                            t2_ransac = t1 + R1 @ t_ransac.ravel()  # Translate and rotate
                            
                            # Try triangulation again with the recovered pose
                            points_3d, point_colors = self.triangulate_points(
                                img1, img2, 
                                self.intrinsic_matrix, self.distortion_coeffs,
                                R1, t1, R2_ransac, t2_ransac
                            )
                            
                            print(f"Triangulation with RANSAC pose yielded {len(points_3d) if points_3d is not None else 0} points")
                        except Exception as e:
                            print(f"RANSAC pose estimation failed: {e}")
            
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
        
        # Print point cloud information
        print(f"Point cloud has {len(self.point_cloud.points)} points")
        print(f"Point cloud bounds: {self.point_cloud.get_min_bound()} to {self.point_cloud.get_max_bound()}")
        
        # Increase point size for better visibility
        self.point_cloud.paint_uniform_color([0.8, 0.8, 0.8])  # Light gray default color
        
        # Create coordinate frame for reference
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=max(self.point_cloud.get_max_bound() - self.point_cloud.get_min_bound()) * 0.1,
            origin=[0, 0, 0]
        )
        
        # Visualize the point cloud with custom settings for better visibility
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Multi-View Reconstruction", width=1024, height=768)
        vis.add_geometry(self.point_cloud)
        vis.add_geometry(coordinate_frame)
        
        # Get render options and set point size larger
        render_option = vis.get_render_option()
        render_option.point_size = 5.0  # Larger point size (default is 1.0)
        render_option.background_color = np.array([0.1, 0.1, 0.1])  # Dark background
        
        # Set camera position
        view_control = vis.get_view_control()
        view_control.set_zoom(0.8)
        
        vis.run()
        vis.destroy_window()

    def capture_enhanced_views(self, point_cloud, num_views=24, output_dir=None):
        """
        Capture enhanced views of a point cloud for better feature detection
        
        Args:
            point_cloud: The point cloud to capture views from
            num_views: Number of views to capture
            output_dir: Directory to save the captured views
            
        Returns:
            List of view dictionaries
        """
        if output_dir is None:
            output_dir = os.path.join(self.output_dir, "enhanced_views")
            
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Capturing {num_views} enhanced views of the point cloud...")
        
        # Create a copy of the point cloud for rendering
        render_cloud = copy.deepcopy(point_cloud)
        
        # Increase point size for better visibility in rendered images
        render_cloud.paint_uniform_color([1.0, 1.0, 1.0])  # White for better contrast
        
        # Get the bounding box of the point cloud
        bbox = render_cloud.get_axis_aligned_bounding_box()
        center = bbox.get_center()
        extent = bbox.get_extent()
        
        # Calculate the radius for the camera positions
        radius = max(extent) * 1.5
        
        # Create a visualizer for rendering
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False, width=1024, height=1024)
        vis.add_geometry(render_cloud)
        
        # Set render options for better feature detection
        render_option = vis.get_render_option()
        render_option.point_size = 5.0  # Larger point size
        render_option.background_color = np.array([0.0, 0.0, 0.0])  # Black background
        
        # Get the view control
        view_control = vis.get_view_control()
        
        views = []
        
        # Capture views from different angles
        for i in range(num_views):
            # Calculate the camera position on a sphere
            theta = i * 2 * np.pi / num_views
            for phi in [np.pi/4, np.pi/2, 3*np.pi/4]:  # Multiple elevation angles
                # Convert spherical to Cartesian coordinates
                x = radius * np.sin(phi) * np.cos(theta)
                y = radius * np.sin(phi) * np.sin(theta)
                z = radius * np.cos(phi)
                
                # Set the camera position
                camera_pos = center + np.array([x, y, z])
                up_vector = np.array([0, 0, 1])  # Up is Z-axis
                
                # Look at the center of the point cloud
                view_control.set_lookat(center)
                view_control.set_front(camera_pos - center)
                view_control.set_up(up_vector)
                view_control.set_zoom(0.7)
                
                # Update the renderer
                vis.poll_events()
                vis.update_renderer()
                
                # Capture the image
                image = vis.capture_screen_float_buffer(do_render=True)
                image = np.asarray(image) * 255
                image = image.astype(np.uint8)
                
                # Convert to BGR for OpenCV
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                # Apply image enhancements for better feature detection
                enhanced = self.enhance_rendered_image(image_bgr)
                
                # Save the image
                view_idx = len(views)
                image_path = os.path.join(output_dir, f"view_{view_idx:03d}.jpg")
                cv2.imwrite(image_path, enhanced)
                
                # Create the view dictionary
                R, t = self.calculate_camera_pose(camera_pos, center, up_vector)
                
                # Create extrinsic matrix (4x4 [R|t; 0 0 0 1]) from rotation and translation
                extrinsic = np.eye(4)  # Start with identity matrix
                extrinsic[:3, :3] = R  # Set upper-left 3x3 block to rotation matrix
                extrinsic[:3, 3] = t   # Set right column to translation vector
                
                view = {
                    'image_path': image_path,
                    'R': R,
                    't': t,
                    'extrinsic_matrix': extrinsic,  # Add extrinsic matrix for reconstruct method
                    'camera_pos': camera_pos,
                    'center': center,
                    'up': up_vector
                }
                
                views.append(view)
                
                print(f"Captured enhanced view {view_idx} from position {camera_pos}")
        
        vis.destroy_window()
        
        return views
        
    def calculate_camera_pose(self, camera_pos, target, up):
        """
        Calculate camera rotation and translation from position, target and up vector
        
        Args:
            camera_pos: Camera position
            target: Look-at target
            up: Up vector
            
        Returns:
            R, t: Rotation matrix and translation vector
        """
        # Calculate camera axes
        forward = target - camera_pos
        forward = forward / np.linalg.norm(forward)
        
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        
        up = np.cross(right, forward)
        up = up / np.linalg.norm(up)
        
        # Create rotation matrix
        R = np.array([
            [right[0], up[0], -forward[0]],
            [right[1], up[1], -forward[1]],
            [right[2], up[2], -forward[2]]
        ])
        
        # Translation is just the negative camera position in the camera coordinate system
        t = -R @ camera_pos
        
        return R, t

    def second_pass_reconstruction(self, input_point_cloud):
        """
        Perform a second pass reconstruction using enhanced views of the input point cloud
        
        Args:
            input_point_cloud: Input point cloud to capture enhanced views from
            
        Returns:
            True if reconstruction was successful
        """
        print("\n=== Starting Second Pass Reconstruction ===\n")
        print("Capturing enhanced views of the input point cloud...")
        
        # Set default intrinsic parameters if none are loaded
        if not hasattr(self, 'intrinsic_matrix') or self.intrinsic_matrix is None:
            print("No intrinsic matrix found, using default values for 1024x1024 images")
            # Create a default intrinsic matrix for 1024x1024 images
            # Focal length = image_size * 0.8, principal point = image_center
            fx = fy = 1024 * 0.8  # Focal length
            cx = cy = 1024 / 2    # Principal point (image center)
            self.intrinsic_matrix = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ])
            self.distortion_coeffs = np.zeros(5)  # No distortion
            self.resolution = (1024, 1024)
        
        # Capture enhanced views of the input point cloud
        enhanced_views = self.capture_enhanced_views(input_point_cloud)
        
        if len(enhanced_views) < 2:
            print("Failed to capture enough enhanced views")
            return False
            
        # Save the original views
        original_views = self.views
        
        # Use the enhanced views for reconstruction
        self.views = enhanced_views
        
        # Output debugging information
        print(f"Using {len(enhanced_views)} enhanced views for second pass reconstruction")
        print(f"Intrinsic matrix:\n{self.intrinsic_matrix}")
        
        # Perform reconstruction with enhanced views
        success = self.reconstruct()
        
        # Restore original views
        self.views = original_views
        
        return success

    def recover_pose_from_essential_matrix(self, points1, points2, K):
        """
        Recover the relative camera pose from an essential matrix
        
        Args:
            points1, points2: Matched points in image coordinates (not normalized)
            K: Camera intrinsic matrix
            
        Returns:
            R, t: Estimated rotation and translation
            inlier_mask: Binary mask indicating inliers
        """
        # Normalize points using intrinsic matrix
        points1_undist = cv2.undistortPoints(points1.reshape(-1, 1, 2), K, self.distortion_coeffs)
        points2_undist = cv2.undistortPoints(points2.reshape(-1, 1, 2), K, self.distortion_coeffs)
        
        points1_norm = points1_undist.reshape(-1, 2)
        points2_norm = points2_undist.reshape(-1, 2)
        
        # Find essential matrix with RANSAC
        E, mask = cv2.findEssentialMat(
            points1_norm, points2_norm, 
            np.eye(3),  # Identity matrix since points are already normalized
            method=cv2.RANSAC, 
            prob=0.999, 
            threshold=0.005
        )
        
        # Recover pose from essential matrix
        _, R, t, pose_mask = cv2.recoverPose(E, points1_norm, points2_norm)
        
        # Combine masks from both operations
        inlier_mask = mask.ravel() == 1
        pose_inliers = pose_mask.ravel() == 1
        final_mask = np.logical_and(inlier_mask, pose_inliers)
        
        print(f"Recovered pose with {np.sum(final_mask)} inliers out of {len(points1)} points")
        
        return R, t, final_mask

def main():
    parser = argparse.ArgumentParser(description='Multi-view reconstruction tool')
    parser.add_argument('--data', '-d', type=str, required=True,
                      help='Directory containing capture data (images and camera parameters)')
    parser.add_argument('--visualize', '-v', action='store_true',
                      help='Visualize the reconstruction result')
    parser.add_argument('--second_pass', action='store_true', 
                      help='Perform a second pass reconstruction')
    parser.add_argument('--enhance-views', action='store_true', 
                      help='Generate enhanced views of the model')
    parser.add_argument('--num-views', type=int, default=24, 
                      help='Number of enhanced views to generate')
    parser.add_argument('--use-ransac-pose', action='store_true', default=True, 
                      help='Use RANSAC-based pose estimation when provided poses fail (default: True)')
    parser.add_argument('--no-ransac-pose', action='store_false', dest='use_ransac_pose',
                      help='Disable RANSAC-based pose estimation')
    
    args = parser.parse_args()
    
    try:
        # Initialize reconstruction
        mvr = MultiViewReconstruction(data_dir=args.data)
        
        # Perform reconstruction
        print("Starting reconstruction...")
        success = mvr.reconstruct(use_ransac_pose=args.use_ransac_pose)
        
        # Perform second pass if requested and first pass was successful
        if success and args.second_pass and mvr.point_cloud is not None:
            print("\nPerforming second pass reconstruction...")
            mvr.second_pass_reconstruction(mvr.point_cloud)
        
        # Visualize the result if requested
        if args.visualize:
            mvr.visualize()
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    return 0

if __name__ == "__main__":
    main() 