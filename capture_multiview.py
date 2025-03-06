#!/usr/bin/env python
import os
import cv2
import numpy as np
import json
import argparse
from datetime import datetime

class MultiViewCapture:
    def __init__(self, output_dir=None, camera_id=0, resolution=(1280, 720)):
        """
        Initialize the multi-view capture system
        
        Args:
            output_dir (str): Directory to save captured images and camera parameters
            camera_id (int): Camera device ID (default: 0)
            resolution (tuple): Camera resolution (width, height)
        """
        self.camera_id = camera_id
        self.resolution = resolution
        
        # Create output directory with timestamp if not specified
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"capture_{timestamp}"
            
        self.output_dir = output_dir
        self.images_dir = os.path.join(output_dir, "images")
        
        # Create directories if they don't exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)
        
        # Initialize camera
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        
        if not self.cap.isOpened():
            raise ValueError(f"Failed to open camera (ID: {camera_id})")
            
        # Initialize camera parameters
        self.intrinsic_matrix = None
        self.distortion_coeffs = None
        self.extrinsic_matrices = []
        self.captured_frames = []
        
    def calibrate_camera(self, chessboard_size=(9, 6), square_size=0.025):
        """
        Calibrate camera to get intrinsic parameters
        
        Args:
            chessboard_size (tuple): Number of internal corners in the chessboard (width, height)
            square_size (float): Size of each square in meters
        """
        print("Starting camera calibration...")
        print("Please show the chessboard pattern from different angles.")
        print("Press 'c' to capture calibration image, 'q' to finish calibration.")
        
        # Prepare object points (3D points in real world space)
        objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size
        
        # Arrays to store object points and image points
        objpoints = []  # 3D points in real world space
        imgpoints = []  # 2D points in image plane
        
        captured_count = 0
        min_captures = 10
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                break
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            display_frame = frame.copy()
            
            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
            
            # If found, add object points and image points
            if ret:
                cv2.drawChessboardCorners(display_frame, chessboard_size, corners, ret)
                
                # Display instruction to capture this frame
                cv2.putText(display_frame, "Press 'c' to capture this frame", 
                           (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Show the number of captured frames
            cv2.putText(display_frame, f"Captured: {captured_count}/{min_captures}", 
                       (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('Camera Calibration', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                if captured_count < min_captures:
                    print(f"Warning: Only {captured_count} calibration images. Recommended: {min_captures}")
                    confirm = input("Continue with calibration anyway? (y/n): ")
                    if confirm.lower() != 'y':
                        continue
                break
                
            if key == ord('c') and ret:
                objpoints.append(objp)
                imgpoints.append(corners)
                captured_count += 1
                print(f"Captured calibration image {captured_count}")
        
        cv2.destroyAllWindows()
        
        if captured_count == 0:
            print("No calibration images captured. Using default parameters.")
            # Default intrinsic parameters (estimate based on resolution)
            focal_length = max(self.resolution) * 1.2
            self.intrinsic_matrix = np.array([
                [focal_length, 0, self.resolution[0]/2],
                [0, focal_length, self.resolution[1]/2],
                [0, 0, 1]
            ])
            self.distortion_coeffs = np.zeros(5)
            return
        
        print("Calculating camera calibration parameters...")
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None
        )
        
        if ret:
            self.intrinsic_matrix = mtx
            self.distortion_coeffs = dist
            print("Camera calibration successful!")
            print(f"Intrinsic matrix:\n{self.intrinsic_matrix}")
            print(f"Distortion coefficients: {self.distortion_coeffs.ravel()}")
        else:
            print("Camera calibration failed. Using default parameters.")
            # Default intrinsic parameters
            focal_length = max(self.resolution) * 1.2
            self.intrinsic_matrix = np.array([
                [focal_length, 0, self.resolution[0]/2],
                [0, focal_length, self.resolution[1]/2],
                [0, 0, 1]
            ])
            self.distortion_coeffs = np.zeros(5)
    
    def capture_view(self, view_id):
        """
        Capture a single view (image and estimated pose)
        
        Args:
            view_id (int): Identifier for this view
        
        Returns:
            bool: True if capture was successful
        """
        ret, frame = self.cap.read()
        if not ret:
            print("Failed to grab frame")
            return False
            
        # Save the image
        image_path = os.path.join(self.images_dir, f"view_{view_id:03d}.jpg")
        cv2.imwrite(image_path, frame)
        
        # For this example, we'll create a simple extrinsic matrix
        # In a real scenario, this would come from a tracking system or user input
        # Here we're creating a synthetic circular path around the origin
        angle = (view_id / 20) * 2 * np.pi  # Circular path
        radius = 1.0
        
        # Camera position
        tx = radius * np.cos(angle)
        ty = 0.2  # Slight elevation
        tz = radius * np.sin(angle)
        
        # Camera orientation (looking at origin)
        rotation_matrix = self._look_at([tx, ty, tz], [0, 0, 0], [0, 1, 0])
        
        # Construct the extrinsic matrix [R|t]
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = rotation_matrix
        extrinsic[:3, 3] = [tx, ty, tz]
        
        # Store the extrinsic matrix
        self.extrinsic_matrices.append(extrinsic)
        self.captured_frames.append({
            'view_id': view_id,
            'image_path': image_path,
            'extrinsic_matrix': extrinsic.tolist()
        })
        
        print(f"Captured view {view_id}: {image_path}")
        return True
    
    def _look_at(self, eye, target, up):
        """
        Create a rotation matrix for a camera looking at a target point
        
        Args:
            eye: Camera position
            target: Target position to look at
            up: Up direction
            
        Returns:
            3x3 rotation matrix
        """
        eye = np.array(eye)
        target = np.array(target)
        up = np.array(up)
        
        forward = target - eye
        forward = forward / np.linalg.norm(forward)
        
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        
        new_up = np.cross(right, forward)
        new_up = new_up / np.linalg.norm(new_up)
        
        # Construct the rotation matrix
        rotation_matrix = np.array([
            [right[0], right[1], right[2]],
            [new_up[0], new_up[1], new_up[2]],
            [-forward[0], -forward[1], -forward[2]]
        ])
        
        return rotation_matrix
        
    def capture_multiview(self, num_views=20):
        """
        Capture multiple views by rotating around the object
        
        Args:
            num_views (int): Number of views to capture
        """
        print(f"Starting multi-view capture ({num_views} views)")
        print("Position the object in the center of the view")
        print("Press 's' to start capturing, 'q' to quit")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                break
                
            # Display instructions
            cv2.putText(frame, "Position object in center", 
                       (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "Press 's' to start capturing", 
                       (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('Multi-View Capture', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                return False
            elif key == ord('s'):
                break
        
        # Start capturing views
        for view_id in range(num_views):
            # Show live preview with instructions
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break
                    
                # Display current view number
                cv2.putText(frame, f"View {view_id+1}/{num_views}", 
                           (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, "Position camera, then press 'c' to capture", 
                           (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                cv2.imshow('Multi-View Capture', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    return False
                elif key == ord('c'):
                    break
            
            self.capture_view(view_id)
            
        cv2.destroyAllWindows()
        return True
        
    def save_camera_parameters(self):
        """
        Save camera parameters to JSON file
        """
        params = {
            'intrinsic_matrix': self.intrinsic_matrix.tolist() if self.intrinsic_matrix is not None else None,
            'distortion_coeffs': self.distortion_coeffs.tolist() if self.distortion_coeffs is not None else None,
            'resolution': self.resolution,
            'views': self.captured_frames
        }
        
        params_path = os.path.join(self.output_dir, 'camera_parameters.json')
        with open(params_path, 'w') as f:
            json.dump(params, f, indent=4)
            
        print(f"Camera parameters saved to: {params_path}")
        
    def close(self):
        """
        Release resources
        """
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Multi-view image capture tool')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output directory for captured data')
    parser.add_argument('--camera', '-c', type=int, default=0,
                        help='Camera device ID (default: 0)')
    parser.add_argument('--resolution', '-r', type=str, default='1280x720',
                        help='Camera resolution, format: WIDTHxHEIGHT (default: 1280x720)')
    parser.add_argument('--views', '-v', type=int, default=20,
                        help='Number of views to capture (default: 20)')
    parser.add_argument('--no-calibration', action='store_true',
                        help='Skip camera calibration')
    
    args = parser.parse_args()
    
    # Parse resolution
    width, height = map(int, args.resolution.split('x'))
    resolution = (width, height)
    
    try:
        # Initialize multi-view capture
        mvc = MultiViewCapture(
            output_dir=args.output,
            camera_id=args.camera,
            resolution=resolution
        )
        
        # Calibrate camera (unless skipped)
        if not args.no_calibration:
            mvc.calibrate_camera()
        
        # Capture multiple views
        if mvc.capture_multiview(num_views=args.views):
            # Save camera parameters
            mvc.save_camera_parameters()
            print(f"Multi-view capture completed. Data saved to: {mvc.output_dir}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'mvc' in locals():
            mvc.close()

if __name__ == "__main__":
    main() 