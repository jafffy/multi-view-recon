import numpy as np
import open3d as o3d
import cv2
import os
import json
import glob
from pathlib import Path
import copy
import time
import matplotlib.pyplot as plt
from IPython.display import display, clear_output

def back_project_depth(depth_map, intrinsic_matrix, extrinsic_matrix=None, depth_threshold=0.0):
    """
    Back-Projection을 통해 depth map으로부터 3D 포인트 클라우드를 생성합니다.
    
    Args:
        depth_map (np.ndarray): (H, W) shape의 depth map (실제 거리 단위로 저장되어야 함)
        intrinsic_matrix (np.ndarray): 3x3 카메라 내부 파라미터 행렬 (fx, fy, cx, cy 포함)
        extrinsic_matrix (np.ndarray, optional): 4x4 카메라 extrinsic 행렬 (camera->world 변환)
        depth_threshold (float): 유효 depth로 간주할 최소 값. 이 값 이하의 depth는 무시합니다.
    
    Returns:
        points (np.ndarray): (N, 3) shape의 포인트 클라우드 (world 좌표계)
    """
    height, width = depth_map.shape
    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    cx = intrinsic_matrix[0, 2]
    cy = intrinsic_matrix[1, 2]
    
    # 픽셀 좌표 grid 생성 (u: 가로, v: 세로)
    u = np.arange(0, width)
    v = np.arange(0, height)
    uu, vv = np.meshgrid(u, v)
    
    # depth 값이 유효한 마스크 (depth > threshold)
    valid_mask = depth_map > depth_threshold
    if np.count_nonzero(valid_mask) == 0:
        print("Warning: No valid depth pixels found!")
        return np.empty((0, 3))
    
    # 유효한 픽셀 좌표와 depth 값 추출
    uu_valid = uu[valid_mask]
    vv_valid = vv[valid_mask]
    depth_valid = depth_map[valid_mask]
    
    # 카메라 좌표계에서의 3D 포인트 계산: 
    # X = (u - cx) * depth / fx, Y = (v - cy) * depth / fy, Z = depth
    x = (uu_valid - cx) * depth_valid / fx
    y = (vv_valid - cy) * depth_valid / fy
    z = depth_valid
    points_camera = np.stack((x, y, z), axis=1)  # shape: (N, 3)
    
    # 만약 extrinsic_matrix가 주어졌다면, world 좌표계로 변환
    if extrinsic_matrix is not None:
        # 동차 좌표(homogeneous coordinates)로 변환
        ones = np.ones((points_camera.shape[0], 1))
        points_homogeneous = np.concatenate([points_camera, ones], axis=1)  # (N, 4)
        # extrinsic_matrix를 적용 (4x4 행렬과의 곱)
        points_world_homogeneous = (extrinsic_matrix @ points_homogeneous.T).T  # (N, 4)
        # 동차 좌표를 일반 좌표로 변환
        points = points_world_homogeneous[:, :3] / points_world_homogeneous[:, 3:]
    else:
        points = points_camera
    
    return points

def load_camera_parameters(output_dir):
    """
    virtual_camera_capture.py에서 저장한 카메라 파라미터를 불러옵니다.
    
    Args:
        output_dir (str): virtual_camera_capture.py의 output_dir 경로
    
    Returns:
        dict: 카메라 파라미터들을 포함하는 딕셔너리
    """
    params_path = os.path.join(output_dir, 'camera_parameters.json')
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"Camera parameters file not found at {params_path}")
    
    with open(params_path, 'r') as f:
        params = json.load(f)
    
    # Convert lists back to numpy arrays
    params['intrinsic_matrix'] = np.array(params['intrinsic_matrix'])
    params['distortion_coeffs'] = np.array(params['distortion_coeffs'])
    params['center'] = np.array(params['center'])
    params['extent'] = np.array(params['extent'])
    
    # Convert extrinsic matrices in each view
    for view in params['views']:
        view['extrinsic_matrix'] = np.array(view['extrinsic_matrix'])
        view['camera_position'] = np.array(view['camera_position'])
    
    return params

def process_single_view(view_info, intrinsic_matrix, output_dir=None, depth_threshold=0.0, remove_outliers=True):
    """
    단일 뷰의 depth map을 포인트 클라우드로 변환합니다.
    
    Args:
        view_info (dict): 뷰 정보 (captured_frames의 요소)
        intrinsic_matrix (np.ndarray): 3x3 카메라 내부 파라미터 행렬
        output_dir (str, optional): 결과 포인트 클라우드를 저장할 경로
        depth_threshold (float): 유효 depth로 간주할 최소 값
        remove_outliers (bool): 통계적 이상치를 제거할지 여부
    
    Returns:
        o3d.geometry.PointCloud: 처리된 포인트 클라우드
    """
    view_id = view_info['view_id']
    depth_path = view_info['depth_path']
    extrinsic_matrix = view_info['extrinsic_matrix']
    
    # Load depth map
    if not os.path.exists(depth_path):
        print(f"Warning: Depth file not found at {depth_path}")
        return None
    
    depth_map = np.load(depth_path)
    
    # 별도로 저장된 depth min/max가 있다면 그것을 이용하여 depth threshold 조정
    if 'depth_min' in view_info:
        # depth_min보다 조금 높은 값으로 threshold 설정
        # (노이즈나 배경의 매우 먼 점들을 제외)
        depth_threshold = max(depth_threshold, view_info['depth_min'] * 1.05)
    
    # Back-project to 3D point cloud
    points = back_project_depth(
        depth_map=depth_map, 
        intrinsic_matrix=intrinsic_matrix,
        extrinsic_matrix=extrinsic_matrix,
        depth_threshold=depth_threshold
    )
    
    if len(points) == 0:
        print(f"Warning: No valid points generated for view {view_id}")
        return None
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Optional: Load RGB image and add colors to point cloud
    if 'image_path' in view_info and os.path.exists(view_info['image_path']):
        img = cv2.imread(view_info['image_path'])
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            height, width = depth_map.shape
            
            # 유효한 depth 값이 있는 픽셀의 좌표 추출
            valid_mask = depth_map > depth_threshold
            if np.count_nonzero(valid_mask) > 0:
                y_coords, x_coords = np.where(valid_mask)
                
                # 유효한 픽셀의 색상 추출
                colors = img[y_coords, x_coords] / 255.0
                pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Statistical outlier removal
    if remove_outliers and len(points) > 20:  # 최소 20개 이상 포인트가 필요
        try:
            pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        except Exception as e:
            print(f"Warning: Failed to remove outliers: {e}")
    
    # Save point cloud to file if output_dir provided
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"pcd_{view_id:03d}.ply")
        o3d.io.write_point_cloud(output_path, pcd)
        print(f"Saved point cloud for view {view_id} to {output_path}")
    
    return pcd

def process_all_views(output_dir, pcds_output_dir=None, depth_threshold=0.0, remove_outliers=True):
    """
    virtual_camera_capture.py에서 저장한 모든 뷰를 처리하여 포인트 클라우드를 생성합니다.
    
    Args:
        output_dir (str): virtual_camera_capture.py의 output_dir 경로
        pcds_output_dir (str, optional): 생성된 포인트 클라우드를 저장할 경로
        depth_threshold (float): 유효 depth로 간주할 최소 값
        remove_outliers (bool): 통계적 이상치를 제거할지 여부
    
    Returns:
        list: 처리된 o3d.geometry.PointCloud 객체들의 리스트
    """
    # Load camera parameters
    params = load_camera_parameters(output_dir)
    intrinsic_matrix = params['intrinsic_matrix']
    
    # Process each view
    point_clouds = []
    
    for view_info in params['views']:
        pcd = process_single_view(
            view_info=view_info,
            intrinsic_matrix=intrinsic_matrix,
            output_dir=pcds_output_dir,
            depth_threshold=depth_threshold,
            remove_outliers=remove_outliers
        )
        
        if pcd is not None:
            point_clouds.append(pcd)
    
    print(f"Processed {len(point_clouds)} point clouds from {len(params['views'])} views")
    return point_clouds

def preprocess_point_cloud(pcd, voxel_size=0.01):
    """
    Point cloud를 전처리합니다. Voxel downsampling과 normal 계산을 수행합니다.
    
    Args:
        pcd (o3d.geometry.PointCloud): 입력 포인트 클라우드
        voxel_size (float): 다운샘플링에 사용할 복셀 크기
        
    Returns:
        o3d.geometry.PointCloud: 전처리된 포인트 클라우드
    """
    # Voxel downsampling
    pcd_down = pcd.voxel_down_sample(voxel_size)
    
    # Normal 계산 (radius는 voxel_size의 2배로 설정)
    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    
    return pcd_down

def pairwise_registration(source, target, max_correspondence_distance, 
                          with_scaling=False, robust_kernel=True):
    """
    두 포인트 클라우드 간의 정밀 정합(ICP)을 수행합니다.
    
    Args:
        source (o3d.geometry.PointCloud): 소스 포인트 클라우드
        target (o3d.geometry.PointCloud): 타겟 포인트 클라우드
        max_correspondence_distance (float): 대응점 간 최대 거리
        with_scaling (bool): 스케일링 변환 포함 여부
        robust_kernel (bool): 로버스트 커널 사용 여부
        
    Returns:
        튜플: (변환 행렬, 정보 행렬)
    """
    # Point-to-plane ICP
    if with_scaling:
        # Use similarity registration (rotation, translation, and scaling)
        result = o3d.pipelines.registration.registration_icp(
            source, target, max_correspondence_distance,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=100, relative_fitness=1e-6, relative_rmse=1e-6)
        )
    else:
        # Use weighted ICP with robust kernel for better accuracy
        if robust_kernel:
            loss = o3d.pipelines.registration.HuberLoss(k=0.1)
            
            # Robust point-to-plane ICP
            result = o3d.pipelines.registration.registration_icp(
                source, target, max_correspondence_distance,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(loss),
                criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                    max_iteration=100, relative_fitness=1e-6, relative_rmse=1e-6)
            )
        else:
            # Standard point-to-plane ICP
            result = o3d.pipelines.registration.registration_icp(
                source, target, max_correspondence_distance,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                    max_iteration=100, relative_fitness=1e-6, relative_rmse=1e-6)
            )
    
    # Information 행렬 계산 (대응점들에 기반한 불확실성)
    correspondence_set = result.correspondence_set
    if len(correspondence_set) == 0:
        # 대응점이 없으면 단위 행렬 반환
        information = np.eye(6)
    else:
        information = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
            source, target, max_correspondence_distance, result.transformation)
        
    return (result.transformation, information)

def register_point_clouds(point_clouds, voxel_size=0.005, refine=True, face_specific=True):
    """
    다중 포인트 클라우드를 정합(registration)합니다.
    
    Args:
        point_clouds (list): o3d.geometry.PointCloud 객체들의 리스트
        voxel_size (float): 다운샘플링 및 정합에 사용할 복셀 크기
        refine (bool): Multi-way registration 적용 여부
        face_specific (bool): 얼굴 포인트 클라우드 특화 처리 적용 여부
        
    Returns:
        list: 정합된 o3d.geometry.PointCloud 객체들의 리스트
    """
    if len(point_clouds) <= 1:
        return point_clouds
    
    print(f"Registering {len(point_clouds)} point clouds...")
    
    # 얼굴 특화 처리: 포인트 클라우드의 방향 정규화 (전면이 +Z 방향 바라보도록)
    if face_specific:
        for i, pcd in enumerate(point_clouds):
            # 1. 포인트 클라우드의 중심을 계산
            center = pcd.get_center()
            
            # 2. 주성분 분석을 통해 얼굴의 주 방향 추정
            covariance = np.cov(np.asarray(pcd.points).T)
            eigvals, eigvecs = np.linalg.eig(covariance)
            
            # 3. 가장 작은 고유값에 해당하는 고유벡터가 얼굴의 법선 방향
            smallest_eigval_idx = np.argmin(eigvals)
            normal_vec = eigvecs[:, smallest_eigval_idx]
            
            # 4. 법선 방향이 +Z를 향하도록 회전 행렬 계산
            z_axis = np.array([0, 0, 1])
            rotation_axis = np.cross(normal_vec, z_axis)
            
            # 회전축이 0이 아닌 경우에만 회전 적용
            if np.linalg.norm(rotation_axis) > 1e-6:
                rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
                dot_product = np.dot(normal_vec, z_axis)
                angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
                
                # 회전 행렬 생성 및 적용
                rotation = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis * angle)
                pcd.rotate(rotation, center=center)
        
        print("Applied face-specific orientation normalization")
    
    # 전처리 (다운샘플링 및 normal 계산)
    processed_point_clouds = [preprocess_point_cloud(pcd, voxel_size) for pcd in point_clouds]
    
    # 첫 번째 포인트 클라우드에 다른 포인트 클라우드들을 정합
    registered_pcds = [processed_point_clouds[0]]
    
    # ICP 파라미터는 복셀 크기에 따라 조정
    max_correspondence_distance = voxel_size * 5  # Starting with a larger distance
    
    # 포인트 클라우드의 각 쌍 간의 대응 관계 (변환 행렬, 정보 행렬)
    odometry = []
    
    # 모든 인접한 포인트 클라우드 쌍에 대해 Pairwise registration 수행
    for i in range(len(processed_point_clouds) - 1):
        source = processed_point_clouds[i]
        target = processed_point_clouds[i + 1]
        
        # 초기 정합을 위해 먼저 특징 기반 정합 수행 (FPFH 특징 사용)
        source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            source, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100))
        target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            target, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100))
        
        result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source, target, source_fpfh, target_fpfh, True,
            max_correspondence_distance * 1.5,  # Use larger distance for RANSAC
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            3,  # minimum_correspondence_points
            [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(max_correspondence_distance * 1.5)
            ],
            o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)
        )
        
        # RANSAC 결과를 이용하여 초기 변환 적용
        source_tmp = copy.deepcopy(source)
        source_tmp.transform(result_ransac.transformation)
        
        # ICP를 통한 정밀 정합 (서로 다른 거리 값으로 다단계 ICP 수행)
        current_transformation = result_ransac.transformation
        
        # Multi-stage ICP refinement
        distances = [max_correspondence_distance, voxel_size * 2, voxel_size]
        
        for dist in distances:
            # Copy the current source cloud and apply the transformation so far
            source_tmp = copy.deepcopy(source)
            source_tmp.transform(current_transformation)
            
            # Run ICP with current distance parameter
            transformation, information = pairwise_registration(
                source_tmp, target, dist, 
                with_scaling=face_specific,  # 얼굴일 경우 스케일링 허용
                robust_kernel=True  # 로버스트 커널 사용
            )
            
            # Update the current transformation
            current_transformation = transformation @ current_transformation
        
        # Store the final transformation and information
        odometry.append((current_transformation, information))
        
        # Transform the original point cloud and add to registered list
        transformed_cloud = copy.deepcopy(point_clouds[i + 1])
        transformed_cloud.transform(current_transformation)
        registered_pcds.append(transformed_cloud)
        
        print(f"Registered pair {i+1}/{len(processed_point_clouds)-1}")
    
    # 고급: MultiWay registration으로 refinement (global optimization)
    if refine and len(odometry) >= 2:
        print("Performing global optimization for multi-way registration...")
        
        # 포즈 그래프 구성
        pose_graph = o3d.pipelines.registration.PoseGraph()
        
        # 첫 번째 노드는 원점에 위치
        pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.eye(4)))
        
        # 나머지 노드 추가
        for i in range(len(odometry)):
            pose_graph.nodes.append(
                o3d.pipelines.registration.PoseGraphNode(odometry[i][0]))
        
        # Odometry 엣지 추가
        for i in range(len(odometry)):
            pose_graph.edges.append(
                o3d.pipelines.registration.PoseGraphEdge(
                    i, i+1, odometry[i][0], odometry[i][1], uncertain=False))
        
        # 리프 노드가 아닌 노드들 간의 루프 클로저 엣지 추가
        n_pcds = len(processed_point_clouds)
        
        # 시간 복잡도를 줄이기 위해 인접하지 않은 일부 페어만 검사
        # (모든 조합은 O(n²)으로 느릴 수 있음)
        for i in range(n_pcds):
            for j in range(i + 2, min(i + 5, n_pcds)):  # 5개 범위 내 검사
                # 두 노드 간의 변환 행렬 계산
                source_tmp = copy.deepcopy(processed_point_clouds[i])
                if i > 0:
                    source_tmp.transform(pose_graph.nodes[i].pose)
                
                target_tmp = copy.deepcopy(processed_point_clouds[j])
                if j > 0:
                    target_tmp.transform(pose_graph.nodes[j].pose)
                
                # 현재 변환된 상태에서 ICP 수행
                transformation, information = pairwise_registration(
                    source_tmp, target_tmp, voxel_size * 2, robust_kernel=True
                )
                
                # 충분한 대응점이 있는 경우에만 루프 클로저 엣지 추가
                if np.trace(information) > 0.1:
                    print(f"Add loop closure edge between {i} and {j}")
                    pose_graph.edges.append(
                        o3d.pipelines.registration.PoseGraphEdge(
                            i, j, transformation, information, uncertain=True))
        
        # Global optimization
        option = o3d.pipelines.registration.GlobalOptimizationOption(
            max_correspondence_distance=voxel_size * 2,
            edge_prune_threshold=0.25,
            reference_node=0,
            preference_loop_closure=10.0)
        
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            o3d.pipelines.registration.global_optimization(
                pose_graph,
                o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
                o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
                option)
        
        # 최적화된 포즈로 포인트 클라우드 변환
        registered_pcds = []
        for i in range(n_pcds):
            pcd_transformed = copy.deepcopy(point_clouds[i])
            pcd_transformed.transform(pose_graph.nodes[i].pose)
            registered_pcds.append(pcd_transformed)
        
        print("Global optimization completed")
    
    return registered_pcds

def combine_point_clouds(point_clouds, voxel_size=0.01, save_path=None, register=True, face_specific=True):
    """
    여러 포인트 클라우드를 결합하고 다운샘플링합니다.
    
    Args:
        point_clouds (list): o3d.geometry.PointCloud 객체들의 리스트
        voxel_size (float): 다운샘플링에 사용할 복셀 크기
        save_path (str, optional): 결합된 포인트 클라우드를 저장할 경로
        register (bool): 포인트 클라우드 정합 수행 여부
        face_specific (bool): 얼굴 특화 처리 적용 여부
    
    Returns:
        o3d.geometry.PointCloud: 결합된 포인트 클라우드
    """
    if not point_clouds:
        print("No point clouds to combine")
        return None
    
    # 정합 수행 (필요한 경우)
    if register and len(point_clouds) > 1:
        point_clouds = register_point_clouds(
            point_clouds, voxel_size=voxel_size, 
            refine=True, face_specific=face_specific
        )
    
    # Combine all point clouds
    combined_pcd = o3d.geometry.PointCloud()
    for pcd in point_clouds:
        combined_pcd += pcd
    
    # Voxel downsampling to reduce density
    combined_pcd = combined_pcd.voxel_down_sample(voxel_size)
    
    # 법선 재계산
    combined_pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    
    # 얼굴 포인트 클라우드인 경우, 추가 처리 (ex: bilateral smoothing)
    if face_specific:
        try:
            # 얼굴 포인트 클라우드 평활화를 위한 bilateral filter
            print("Applying bilateral smoothing for face geometry...")
            combined_pcd = combined_pcd.filter_smooth_simple(number_of_iterations=5, 
                                                            neighbor_size=voxel_size*4)
            combined_pcd.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
        except Exception as e:
            print(f"Warning: Could not apply face-specific smoothing: {e}")
    
    # Save combined point cloud if path is provided
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        o3d.io.write_point_cloud(save_path, combined_pcd)
        print(f"Combined point cloud saved to {save_path}")
    
    return combined_pcd

###########################################
# Enhanced Visualization Functionality
###########################################

class VisualizationUtility:
    """
    포인트 클라우드 시각화에 관련된 유틸리티 기능을 제공하는 클래스
    """
    
    @staticmethod
    def create_coordinate_frame(size=0.1, origin=[0, 0, 0]):
        """좌표계 프레임 생성"""
        return o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=origin)
    
    @staticmethod
    def color_by_height(pcd, min_height=None, max_height=None, cmap_name='viridis'):
        """높이(z 좌표)에 따라 포인트 클라우드에 색상을 적용"""
        import matplotlib.pyplot as plt
        import numpy as np
        
        points = np.asarray(pcd.points)
        z_values = points[:, 2]
        
        # 높이 범위가 지정되지 않은 경우 자동으로 계산
        if min_height is None:
            min_height = np.min(z_values)
        if max_height is None:
            max_height = np.max(z_values)
        
        # 정규화된 높이 값 계산
        normalized_z = (z_values - min_height) / (max_height - min_height)
        normalized_z = np.clip(normalized_z, 0, 1)
        
        # 색상맵 적용
        cmap = plt.get_cmap(cmap_name)
        colors = cmap(normalized_z)[:, :3]  # RGBA to RGB
        
        colored_pcd = copy.deepcopy(pcd)
        colored_pcd.colors = o3d.utility.Vector3dVector(colors)
        return colored_pcd
    
    @staticmethod
    def color_by_normal(pcd, component='z', cmap_name='coolwarm'):
        """법선 벡터의 특정 성분에 따라 포인트 클라우드에 색상을 적용"""
        import matplotlib.pyplot as plt
        import numpy as np
        
        # 법선 벡터 계산이 안된 경우 계산
        if not pcd.has_normals():
            pcd.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        
        normals = np.asarray(pcd.normals)
        
        if component == 'x':
            values = normals[:, 0]
        elif component == 'y':
            values = normals[:, 1]
        elif component == 'z':
            values = normals[:, 2]
        else:
            raise ValueError(f"Unknown normal component: {component}")
        
        # [-1, 1] 범위의 법선 벡터 성분을 [0, 1] 범위로 정규화
        normalized_values = (values + 1) / 2
        
        # 색상맵 적용
        cmap = plt.get_cmap(cmap_name)
        colors = cmap(normalized_values)[:, :3]  # RGBA to RGB
        
        colored_pcd = copy.deepcopy(pcd)
        colored_pcd.colors = o3d.utility.Vector3dVector(colors)
        return colored_pcd
    
    @staticmethod
    def visualize_point_cloud(pcd, window_name="Point Cloud", width=1280, height=720, 
                              background_color=[0.1, 0.1, 0.1], point_size=2,
                              show_coordinate_frame=True, show_normals=False):
        """포인트 클라우드 시각화"""
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name, width=width, height=height)
        
        # 포인트 클라우드 추가
        vis.add_geometry(pcd)
        
        # 좌표계 추가
        if show_coordinate_frame:
            coordinate_frame = VisualizationUtility.create_coordinate_frame(size=0.2)
            vis.add_geometry(coordinate_frame)
        
        # 렌더링 옵션 설정
        render_option = vis.get_render_option()
        render_option.background_color = np.array(background_color)
        render_option.point_size = point_size
        
        # 법선 표시 설정
        if show_normals and pcd.has_normals():
            render_option.point_show_normal = True
        
        # 뷰 컨트롤 설정
        ctrl = vis.get_view_control()
        ctrl.set_zoom(0.8)
        
        # 실행
        vis.run()
        vis.destroy_window()
    
    @staticmethod
    def visualize_registration_result(source, target, transformation=None, 
                                     window_name="Registration Result"):
        """정합 결과 시각화"""
        # 소스와 타겟 복사
        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)
        
        # 소스에 변환 행렬 적용 (제공된 경우)
        if transformation is not None:
            source_temp.transform(transformation)
        
        # 색상 설정
        source_temp.paint_uniform_color([1, 0.706, 0])  # 노란색
        target_temp.paint_uniform_color([0, 0.651, 0.929])  # 파란색
        
        # 시각화
        VisualizationUtility.visualize_point_cloud(
            source_temp + target_temp, 
            window_name=window_name
        )
    
    @staticmethod
    def visualize_before_after(point_clouds_before, point_clouds_after, indices=None):
        """정합 전/후 비교 시각화"""
        if indices is None:
            if len(point_clouds_before) <= 3:
                indices = list(range(len(point_clouds_before)))
            else:
                # 시작, 중간, 끝 포인트 클라우드 선택
                indices = [0, len(point_clouds_before)//2, len(point_clouds_before)-1]
        
        # 정합 전 포인트 클라우드 (하나로 결합)
        before_combined = o3d.geometry.PointCloud()
        for i in indices:
            if i < len(point_clouds_before):
                pcd = copy.deepcopy(point_clouds_before[i])
                # 각 클라우드에 고유한 색상 부여
                color = [0.7, 0.3, 0.3]  # 기본 붉은색
                if i == 0:
                    color = [1.0, 0.0, 0.0]  # 빨강
                elif i == 1:
                    color = [0.0, 1.0, 0.0]  # 초록
                elif i == 2:
                    color = [0.0, 0.0, 1.0]  # 파랑
                pcd.paint_uniform_color(color)
                before_combined += pcd
        
        # 정합 후 포인트 클라우드 (하나로 결합)
        after_combined = o3d.geometry.PointCloud()
        for i in indices:
            if i < len(point_clouds_after):
                pcd = copy.deepcopy(point_clouds_after[i])
                # 정합 전과 동일한 색상 부여
                color = [0.7, 0.3, 0.3]  # 기본 붉은색
                if i == 0:
                    color = [1.0, 0.0, 0.0]  # 빨강
                elif i == 1:
                    color = [0.0, 1.0, 0.0]  # 초록
                elif i == 2:
                    color = [0.0, 0.0, 1.0]  # 파랑
                pcd.paint_uniform_color(color)
                after_combined += pcd
        
        # 시각화 (2개의 창으로 분리)
        print("Visualizing before registration. Close window to continue...")
        VisualizationUtility.visualize_point_cloud(
            before_combined, 
            window_name="Before Registration"
        )
        
        print("Visualizing after registration. Close window to continue...")
        VisualizationUtility.visualize_point_cloud(
            after_combined, 
            window_name="After Registration"
        )
    
    @staticmethod
    def visualize_registration_progress(point_clouds, voxel_size=0.005, max_iters=30):
        """정합 과정을 애니메이션으로 시각화"""
        if len(point_clouds) < 2:
            print("At least 2 point clouds are required to visualize registration progress")
            return
        
        # 창 생성
        vis = o3d.visualization.Visualizer()
        vis.create_window("Registration Progress", width=1280, height=720)
        
        # 렌더링 옵션 설정
        render_option = vis.get_render_option()
        render_option.background_color = np.array([0.1, 0.1, 0.1])
        render_option.point_size = 2
        
        # 좌표계 추가
        coordinate_frame = VisualizationUtility.create_coordinate_frame(size=0.2)
        vis.add_geometry(coordinate_frame)
        
        # 각 포인트 클라우드의 복사본 생성 및 색상 할당
        processed_pcds = []
        for i, pcd in enumerate(point_clouds):
            # 다운샘플링 및 normal 계산
            pcd_copy = copy.deepcopy(pcd)
            pcd_down = pcd_copy.voxel_down_sample(voxel_size)
            pcd_down.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30))
            
            # 색상 할당 (인덱스에 따라 다른 색상)
            color = [0.2, 0.2, 0.2]  # 기본 회색
            if i == 0:
                color = [1.0, 0.0, 0.0]  # 첫 번째: 빨강
            elif i == 1:
                color = [0.0, 1.0, 0.0]  # 두 번째: 초록
            else:
                # 기타: 인덱스에 따라 색상 변화
                import matplotlib.cm as cm
                color = cm.jet(i / (len(point_clouds)-1))[:3]
            
            pcd_down.paint_uniform_color(color)
            processed_pcds.append(pcd_down)
        
        # 첫 번째 포인트 클라우드를 기준으로 설정
        source = processed_pcds[0]
        vis.add_geometry(source)
        
        # 각 추가 포인트 클라우드에 대해 정합 수행 및 시각화
        for i in range(1, len(processed_pcds)):
            target = processed_pcds[i]
            vis.add_geometry(target)
            
            # 현재 상태 업데이트
            vis.update_geometry(target)
            vis.poll_events()
            vis.update_renderer()
            
            # ICP 반복 수행
            current_transformation = np.identity(4)
            
            for iter in range(max_iters):
                # 현재 변환 적용
                target_temp = copy.deepcopy(target)
                target_temp.transform(current_transformation)
                
                # ICP 한 단계 실행
                result_icp = o3d.pipelines.registration.registration_icp(
                    source, target_temp, voxel_size * 3, np.identity(4),
                    o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1)
                )
                
                # 변환 갱신
                current_transformation = result_icp.transformation @ current_transformation
                
                # 결과 시각화
                target.transform(result_icp.transformation)
                vis.update_geometry(target)
                vis.poll_events()
                vis.update_renderer()
            
            # 이 포인트 클라우드를 새로운 소스로 설정
            source = target
        
        # 마지막 상태 보여주기
        vis.run()
        vis.destroy_window()
    
    @staticmethod
    def visualize_camera_positions(camera_params, point_cloud=None):
        """카메라 위치와 방향 시각화"""
        if 'views' not in camera_params:
            print("Camera views not found in parameters")
            return
        
        # 시각화할 객체들 목록
        geometries = []
        
        # 포인트 클라우드가 있다면 추가
        if point_cloud is not None:
            geometries.append(point_cloud)
        
        # 좌표계 추가
        origin_frame = VisualizationUtility.create_coordinate_frame(size=0.3)
        geometries.append(origin_frame)
        
        # 각 카메라 뷰에 대한 표시 추가
        for i, view in enumerate(camera_params['views']):
            if 'extrinsic_matrix' not in view or 'camera_position' not in view:
                continue
            
            # 카메라 위치
            pos = np.array(view['camera_position'])
            
            # 카메라 방향 (extrinsic matrix에서 추출)
            extrinsic = np.array(view['extrinsic_matrix'])
            
            # 카메라 표시용 좌표계 생성
            camera_frame = VisualizationUtility.create_coordinate_frame(size=0.1)
            camera_frame.translate(pos)
            
            # 카메라 번호 표시용 구 추가
            camera_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.03)
            camera_sphere.translate(pos)
            
            # 카메라 번호에 따라 색상 할당
            import matplotlib.cm as cm
            color = cm.jet(i / max(1, len(camera_params['views'])-1))[:3]
            camera_sphere.paint_uniform_color(color)
            
            geometries.append(camera_frame)
            geometries.append(camera_sphere)
        
        # 모든 객체 시각화
        VisualizationUtility.visualize_point_cloud(
            o3d.geometry.PointCloud(),  # Empty point cloud as a container
            window_name="Camera Positions",
            show_coordinate_frame=False  # Already added explicitly
        )
        
        # 각 지오메트리 추가
        vis = o3d.visualization.Visualizer()
        vis.create_window("Camera Positions", width=1280, height=720)
        
        for geom in geometries:
            vis.add_geometry(geom)
        
        vis.run()
        vis.destroy_window()

# 예제: virtual_camera_capture.py의 출력을 처리하여 포인트 클라우드 생성
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Process depth maps from virtual_camera_capture.py')
    parser.add_argument('--input', type=str, required=True, 
                        help='Input directory (output_dir from virtual_camera_capture.py)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory for point clouds (defaults to input_dir/point_clouds)')
    parser.add_argument('--combined', type=str, default=None,
                        help='Output path for combined point cloud (defaults to output_dir/combined.ply)')
    parser.add_argument('--depth_threshold', type=float, default=0.0,
                        help='Minimum valid depth threshold (defaults to 0.0)')
    parser.add_argument('--voxel_size', type=float, default=0.01,
                        help='Voxel size for downsampling combined cloud (defaults to 0.01)')
    parser.add_argument('--no_outlier_removal', action='store_true',
                        help='Disable statistical outlier removal')
    parser.add_argument('--no_registration', action='store_true',
                        help='Disable point cloud registration')
    parser.add_argument('--face_specific', action='store_true', default=True,
                        help='Apply face-specific enhancements (default: True)')
    
    # Add visualization options
    vis_group = parser.add_argument_group('Visualization Options')
    vis_group.add_argument('--visualize', action='store_true',
                        help='Visualize the combined point cloud')
    vis_group.add_argument('--visualize_progress', action='store_true',
                        help='Visualize the registration progress')
    vis_group.add_argument('--visualize_before_after', action='store_true',
                        help='Visualize before/after registration comparison')
    vis_group.add_argument('--visualize_cameras', action='store_true',
                        help='Visualize camera positions from capture')
    vis_group.add_argument('--color_by_height', action='store_true',
                        help='Color point cloud by height (Z-coordinate)')
    vis_group.add_argument('--color_by_normal', choices=['x', 'y', 'z'],
                        help='Color point cloud by normal component (x, y, or z)')
    vis_group.add_argument('--show_normals', action='store_true',
                        help='Show normal vectors in visualization')
    
    args = parser.parse_args()
    
    # Set default output directories if not specified
    if args.output is None:
        args.output = os.path.join(args.input, 'point_clouds')
    
    if args.combined is None:
        args.combined = os.path.join(args.output, 'combined.ply')
    
    try:
        # Load camera parameters (for some visualizations)
        camera_params = None
        try:
            camera_params = load_camera_parameters(args.input)
        except Exception as e:
            print(f"Warning: Could not load camera parameters: {e}")
        
        # Visualize camera positions if requested
        if args.visualize_cameras and camera_params is not None:
            print("Visualizing camera positions...")
            VisualizationUtility.visualize_camera_positions(camera_params)
        
        # Process all views to generate individual point clouds
        print("Processing individual point clouds...")
        original_point_clouds = process_all_views(
            output_dir=args.input,
            pcds_output_dir=args.output,
            depth_threshold=args.depth_threshold,
            remove_outliers=not args.no_outlier_removal
        )
        
        if not original_point_clouds:
            print("No valid point clouds generated")
            exit(1)
        
        # Keep a copy of original (before registration) for visualization
        point_clouds = copy.deepcopy(original_point_clouds)
        
        # Visualize registration progress if requested
        if args.visualize_progress and not args.no_registration and len(point_clouds) > 1:
            print("Visualizing registration progress...")
            VisualizationUtility.visualize_registration_progress(
                point_clouds, voxel_size=args.voxel_size
            )
        
        # Register and combine point clouds
        if not args.no_registration and len(point_clouds) > 1:
            print("Registering point clouds...")
            registered_point_clouds = register_point_clouds(
                point_clouds, 
                voxel_size=args.voxel_size,
                refine=True, 
                face_specific=args.face_specific
            )
            
            # Visualize before/after registration if requested
            if args.visualize_before_after:
                print("Visualizing before/after registration comparison...")
                VisualizationUtility.visualize_before_after(
                    original_point_clouds, registered_point_clouds
                )
            
            print("Combining point clouds...")
            combined_pcd = combine_point_clouds(
                point_clouds=registered_point_clouds,
                voxel_size=args.voxel_size,
                save_path=args.combined,
                register=False,  # Already registered
                face_specific=args.face_specific
            )
        else:
            print("Combining point clouds (without registration)...")
            combined_pcd = combine_point_clouds(
                point_clouds=point_clouds,
                voxel_size=args.voxel_size,
                save_path=args.combined,
                register=False,
                face_specific=args.face_specific
            )
        
        # Final visualization of combined point cloud
        if combined_pcd is not None:
            # Prepare visualization
            vis_pcd = copy.deepcopy(combined_pcd)
            
            # Color by height if requested
            if args.color_by_height:
                print("Coloring point cloud by height...")
                vis_pcd = VisualizationUtility.color_by_height(vis_pcd)
            
            # Color by normal if requested
            elif args.color_by_normal:
                print(f"Coloring point cloud by normal {args.color_by_normal} component...")
                vis_pcd = VisualizationUtility.color_by_normal(
                    vis_pcd, component=args.color_by_normal
                )
            
            # Regular visualization if requested
            if args.visualize:
                print("Visualizing combined point cloud...")
                VisualizationUtility.visualize_point_cloud(
                    vis_pcd,
                    window_name="Combined Point Cloud",
                    show_normals=args.show_normals
                )
        
        print(f"Processing completed successfully!")
        print(f"- Individual point clouds: {args.output}")
        print(f"- Combined point cloud: {args.combined}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc() 