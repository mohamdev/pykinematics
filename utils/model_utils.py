import hppfcl as fcl
import pinocchio as pin
import numpy as np
from scipy.spatial.transform import Rotation as R
import pinocchio as pin
from typing import List, Tuple, Dict

#Build inertia matrix from 6 inertia components
def make_inertia_matrix(ixx:float, ixy:float, ixz:float, iyy:float, iyz:float, izz:float)->np.ndarray:
    return np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])

#Function that takes as input a matrix and orthogonalizes it
#Its mainly used to orthogonalize rotation matrices constructed by hand
def orthogonalize_matrix(matrix:np.ndarray)->np.ndarray:
    # Perform Singular Value Decomposition
    U, _, Vt = np.linalg.svd(matrix)
    # Reconstruct the orthogonal matrix
    orthogonal_matrix = U @ Vt
    # Ensure the determinant is 1
    if np.linalg.det(orthogonal_matrix) < 0:
        U[:, -1] *= -1
        orthogonal_matrix = U @ Vt
    return orthogonal_matrix


#construct torso frame and get its pose from a dictionnary of mks positions and names
def get_torso_pose(mocap_mks_positions):
    pose = np.eye(4,4)
    trunk_center = mocap_mks_positions['Neck']
    
    Y = (mocap_mks_positions['Neck'] - mocap_mks_positions['midHip']).reshape(3,1)
    Y = Y/np.linalg.norm(Y)
    X = (mocap_mks_positions['Neck'] - mocap_mks_positions['C7_study']).reshape(3,1)
    X = X/np.linalg.norm(Y)
    Z = np.cross(X, Y, axis=0)
    X = np.cross(Y, Z, axis=0)

    pose[:3,0] = X.reshape(3,)
    pose[:3,1] = Y.reshape(3,)
    pose[:3,2] = Z.reshape(3,)
    pose[:3,3] = trunk_center.reshape(3,)
    pose[:3,:3] = orthogonalize_matrix(pose[:3,:3])
    return pose

#construct upperarm frame and get its pose
def get_upperarm_pose(mocap_mks_positions):
    pose = np.eye(4,4)
    shoulder_center = mocap_mks_positions['RShoulder'].reshape(3,1)
    elbow_center = (mocap_mks_positions['r_melbow_study'] + mocap_mks_positions['r_lelbow_study']).reshape(3,1)/2.0
    
    Y = shoulder_center - elbow_center
    Y = Y/np.linalg.norm(Y)
    Z = (mocap_mks_positions['r_lelbow_study'] - mocap_mks_positions['r_melbow_study']).reshape(3,1)
    Z = Z/np.linalg.norm(Z)
    X = np.cross(Y, Z, axis=0)
    Z = np.cross(X, Y, axis=0)

    pose[:3,0] = X.reshape(3,)
    pose[:3,1] = Y.reshape(3,)
    pose[:3,2] = Z.reshape(3,)
    pose[:3,3] = shoulder_center.reshape(3,)
    pose[:3,:3] = orthogonalize_matrix(pose[:3,:3])

    return pose

#construct lowerarm frame and get its pose
def get_lowerarm_pose(mocap_mks_positions):
    pose = np.eye(4,4)
    elbow_center = (mocap_mks_positions['r_melbow_study'] + mocap_mks_positions['r_lelbow_study']).reshape(3,1)/2.0
    wrist_center = (mocap_mks_positions['r_mwrist_study'] + mocap_mks_positions['r_lwrist_study']).reshape(3,1)/2.0
    
    Y = elbow_center - wrist_center
    Y = Y/np.linalg.norm(Y)
    Z = (mocap_mks_positions['r_lwrist_study'] - mocap_mks_positions['r_mwrist_study']).reshape(3,1)
    Z = Z/np.linalg.norm(Z)
    X = np.cross(Y, Z, axis=0)
    Z = np.cross(X, Y, axis=0)

    pose[:3,0] = X.reshape(3,)
    pose[:3,1] = Y.reshape(3,)
    pose[:3,2] = Z.reshape(3,)
    pose[:3,3] = elbow_center.reshape(3,)
    pose[:3,:3] = orthogonalize_matrix(pose[:3,:3])
    return pose

#construct pelvis frame and get its pose
def get_pelvis_pose(mocap_mks_positions):
    pose = np.eye(4,4)
    center_PSIS = (mocap_mks_positions['r.PSIS_study'] + mocap_mks_positions['L.PSIS_study']).reshape(3,1)/2.0
    center_ASIS = (mocap_mks_positions['r.ASIS_study'] + mocap_mks_positions['L.ASIS_study']).reshape(3,1)/2.0
    center_right_ASIS_PSIS = (mocap_mks_positions['r.PSIS_study'] + mocap_mks_positions['r.ASIS_study']).reshape(3,1)/2.0
    center_left_ASIS_PSIS = (mocap_mks_positions['L.PSIS_study'] + mocap_mks_positions['L.ASIS_study']).reshape(3,1)/2.0

    X = center_ASIS - center_PSIS
    X = X/np.linalg.norm(X)
    Z = center_right_ASIS_PSIS - center_left_ASIS_PSIS
    Z = Z/np.linalg.norm(Z)
    Y = np.cross(Z, X, axis=0)
    Z = np.cross(X, Y, axis=0)

    pose[:3,0] = X.reshape(3,)
    pose[:3,1] = Y.reshape(3,)
    pose[:3,2] = Z.reshape(3,)
    pose[:3,3] = ((center_right_ASIS_PSIS + center_left_ASIS_PSIS)/2.0).reshape(3,)
    pose[:3,:3] = orthogonalize_matrix(pose[:3,:3])

    return pose

#construct thigh frame and get its pose
def get_thigh_pose(mocap_mks_positions):
    pose = np.eye(4,4)
    hip_center = mocap_mks_positions['RHip'].reshape(3,1)
    knee_center = (mocap_mks_positions['r_knee_study'] + mocap_mks_positions['r_mknee_study']).reshape(3,1)/2.0
    
    Y = hip_center - knee_center
    Y = Y/np.linalg.norm(Y)
    Z = (mocap_mks_positions['r_knee_study'] - mocap_mks_positions['r_mknee_study']).reshape(3,1)
    Z = Z/np.linalg.norm(Z)
    X = np.cross(Y, Z, axis=0)
    Z = np.cross(X, Y, axis=0)

    pose[:3,0] = X.reshape(3,)
    pose[:3,1] = Y.reshape(3,)
    pose[:3,2] = Z.reshape(3,)
    pose[:3,3] = hip_center.reshape(3,)
    pose[:3,:3] = orthogonalize_matrix(pose[:3,:3])

    return pose

#construct shank frame and get its pose
def get_shank_pose(mocap_mks_positions):
    pose = np.eye(4,4)

    knee_center = (mocap_mks_positions['r_knee_study'] + mocap_mks_positions['r_mknee_study']).reshape(3,1)/2.0
    ankle_center = (mocap_mks_positions['r_mankle_study'] + mocap_mks_positions['r_ankle_study']).reshape(3,1)/2.0
    
    Y = knee_center - ankle_center
    Y = Y/np.linalg.norm(Y)
    Z = (mocap_mks_positions['r_knee_study'] - mocap_mks_positions['r_mknee_study']).reshape(3,1)
    Z = Z/np.linalg.norm(Z)
    X = np.cross(Y, Z, axis=0)
    Z = np.cross(X, Y, axis=0)

    pose[:3,0] = X.reshape(3,)
    pose[:3,1] = Y.reshape(3,)
    pose[:3,2] = Z.reshape(3,)
    pose[:3,3] = knee_center.reshape(3,)
    pose[:3,:3] = orthogonalize_matrix(pose[:3,:3])
    return pose

#construct foot frame and get its pose
def get_foot_pose(mocap_mks_positions):
    pose = np.eye(4,4)

    ankle_center = (mocap_mks_positions['r_mankle_study'] + mocap_mks_positions['r_ankle_study']).reshape(3,1)/2.0
    
    X = (mocap_mks_positions['r_toe_study'] - mocap_mks_positions['r_calc_study']).reshape(3,1)
    X = X/np.linalg.norm(X)
    Z = (mocap_mks_positions['r_ankle_study'] - mocap_mks_positions['r_mankle_study']).reshape(3,1)
    Z = Z/np.linalg.norm(Z)
    Y = np.cross(Z, X, axis=0)
    Z = np.cross(X, Y, axis=0)

    pose[:3,0] = X.reshape(3,)
    pose[:3,1] = Y.reshape(3,)
    pose[:3,2] = Z.reshape(3,)
    pose[:3,3] = ankle_center.reshape(3,)
    pose[:3,:3] = orthogonalize_matrix(pose[:3,:3])
    return pose



#Construct challenge segments frames from mocap mks
# - mocap_mks_positions is a dictionnary of mocap mks names and 3x1 global positions
# - returns sgts_poses which correspond to a dictionnary to segments poses and names, constructed from mks global positions
def construct_segments_frames_challenge(mocap_mks_positions):    
    torso_pose = get_torso_pose(mocap_mks_positions)
    upperarm_pose = get_upperarm_pose(mocap_mks_positions)
    lowerarm_pose = get_lowerarm_pose(mocap_mks_positions)
    pelvis_pose = get_pelvis_pose(mocap_mks_positions)
    thigh_pose = get_thigh_pose(mocap_mks_positions)
    shank_pose = get_shank_pose(mocap_mks_positions)
    foot_pose = get_foot_pose(mocap_mks_positions)
    
    # Constructing the dictionary to store segment poses
    sgts_poses = {
        "torso": torso_pose,
        "upperarm": upperarm_pose,
        "lowerarm": lowerarm_pose,
        "pelvis": pelvis_pose,
        "thigh": thigh_pose,
        "shank": shank_pose,
        "foot": foot_pose
    }
    # for name, pose in sgts_poses.items():
    #     print(name, " rot det : ", np.linalg.det(pose[:3,:3]))
    return sgts_poses

def get_segments_lstm_mks_dict_challenge()->Dict:
    #This fuction returns a dictionnary containing the segments names, and the corresponding list of lstm
    #mks names attached to the segment
    # Constructing the dictionary to store segment poses
    sgts_mks_dict = {
        "torso": ['RShoulder', 'r_shoulder_study', 'L_shoulder_study', 'LShoulder', 'Neck', 'C7_study'],
        "upperarm": ['r_melbow_study', 'r_lelbow_study', 'RElbow'],
        "lowerarm": ['r_lwrist_study', 'r_mwrist_study', 'RWrist',],
        "pelvis": ['r.PSIS_study', 'L.PSIS_study', 'r.ASIS_study', 'L.ASIS_study', 'RHip', 'LHip', 'LHJC_study', 'RHJC_study', 'midHip'],
        "thigh": ['r_knee_study', 'r_mknee_study', 'RKnee', 'r_thigh2_study', 'r_thigh3_study', 'r_thigh1_study'],
        "shank": ['r_sh3_study', 'r_sh2_study', 'r_sh1_study'],
        "foot": ['r_ankle_study', 'r_mankle_study', 'RAnkle', 'r_calc_study', 'RHeel', 'r_5meta_study', 'RSmallToe', 'r_toe_study', 'RBigToe']
    }
    return sgts_mks_dict

def get_subset_challenge_mks_names()->List:
    """_This function returns the subset of markers used to track the right body side kinematics with pinocchio_

    Returns:
        List: _the subset of markers used to track the right body side kinematics with pinocchio_
    """
    mks_names = ['RShoulder', 'r_shoulder_study', 'L_shoulder_study', 'LShoulder', 'Neck', 'C7_study', 'r_melbow_study', 'r_lelbow_study', 'RElbow',
                 'r_lwrist_study', 'r_mwrist_study', 'RWrist','r.PSIS_study', 'L.PSIS_study', 'r.ASIS_study', 'L.ASIS_study', 'RHip', 'LHip', 'LHJC_study', 'RHJC_study', 'midHip',
                 'r_knee_study', 'r_mknee_study', 'RKnee', 'r_thigh2_study', 'r_thigh3_study', 'r_thigh1_study','r_sh3_study', 'r_sh2_study', 'r_sh1_study',
                 'r_ankle_study', 'r_mankle_study', 'RAnkle', 'r_calc_study', 'RHeel', 'r_5meta_study', 'RSmallToe', 'r_toe_study', 'RBigToe']
    return mks_names

def get_local_lstm_mks_positions(sgts_poses: Dict, lstm_mks_positions: Dict, sgts_mks_dict: Dict)-> Dict:
    """_Get the local 3D position of the lstms markers_

    Args:
        sgts_poses (Dict): _sgts_poses corresponds to a dictionnary to segments poses and names, constructed from global mks positions_
        lstm_mks_positions (Dict): _lstm_mks_positions is a dictionnary of lstm mks names and 3x1 global positions_
        sgts_mks_dict (Dict): _sgts_mks_dict a dictionnary containing the segments names, and the corresponding list of lstm mks names attached to the segment_

    Returns:
        Dict: _returns a dictionnary of lstm mks names and their 3x1 local positions_
    """
    lstm_mks_local_positions = {}

    for segment, markers in sgts_mks_dict.items():
        # Get the segment's transformation matrix
        segment_pose = sgts_poses[segment]
        
        # Compute the inverse of the segment's transformation matrix
        segment_pose_inv = np.eye(4,4)
        segment_pose_inv[:3,:3] = np.transpose(segment_pose[:3,:3])
        segment_pose_inv[:3,3] = -np.transpose(segment_pose[:3,:3]) @ segment_pose[:3,3]
        for marker in markers:
            if marker in lstm_mks_positions:
                # Get the marker's global position
                marker_global_pos = np.append(lstm_mks_positions[marker], 1)  # Convert to homogeneous coordinates

                marker_local_pos_hom = segment_pose_inv @ marker_global_pos  # Transform to local coordinates
                marker_local_pos = marker_local_pos_hom[:3]  # Convert back to 3x1 coordinates
                # Store the local position in the dictionary
                lstm_mks_local_positions[marker] = marker_local_pos

    return lstm_mks_local_positions

def get_local_segments_positions(sgts_poses: Dict)->Dict:
    """_Get the local positions of the segments_

    Args:
        sgts_poses (Dict): _a dictionnary of segment poses_

    Returns:
        Dict: _returns a dictionnary of local positions for each segment except pelvis_
    """
    # Initialize the dictionary to store local positions
    local_positions = {}

    # Pelvis is the base, so it does not have a local position
    pelvis_pose = sgts_poses["pelvis"]

    # Define a helper function to get the translation part from a 4x4 transformation matrix
    def get_translation(matrix):
        return matrix[:3, 3]

    # Compute local positions for each segment
    # Torso with respect to pelvis
    if "torso" in sgts_poses:
        torso_global = sgts_poses["torso"]
        local_positions["torso"] = (np.linalg.inv(pelvis_pose) @ torso_global @ np.array([0, 0, 0, 1]))[:3]

    # Upperarm with respect to torso
    if "upperarm" in sgts_poses:
        upperarm_global = sgts_poses["upperarm"]
        torso_global = sgts_poses["torso"]
        local_positions["upperarm"] = (np.linalg.inv(torso_global) @ upperarm_global @ np.array([0, 0, 0, 1]))[:3]

    # Lowerarm with respect to upperarm
    if "lowerarm" in sgts_poses:
        lowerarm_global = sgts_poses["lowerarm"]
        upperarm_global = sgts_poses["upperarm"]
        local_positions["lowerarm"] = (np.linalg.inv(upperarm_global) @ lowerarm_global @ np.array([0, 0, 0, 1]))[:3]

    # Thigh with respect to pelvis
    if "thigh" in sgts_poses:
        thigh_global = sgts_poses["thigh"]
        local_positions["thigh"] = (np.linalg.inv(pelvis_pose) @ thigh_global @ np.array([0, 0, 0, 1]))[:3]

    # Shank with respect to thigh
    if "shank" in sgts_poses:
        shank_global = sgts_poses["shank"]
        thigh_global = sgts_poses["thigh"]
        local_positions["shank"] = (np.linalg.inv(thigh_global) @ shank_global @ np.array([0, 0, 0, 1]))[:3]

    # Foot with respect to shank
    if "foot" in sgts_poses:
        foot_global = sgts_poses["foot"]
        shank_global = sgts_poses["shank"]
        local_positions["foot"] = (np.linalg.inv(shank_global) @ foot_global @ np.array([0, 0, 0, 1]))[:3]
    
    return local_positions


def build_model_challenge(mocap_mks_positions: Dict, lstm_mks_positions: Dict, meshes_folder_path: str)->Tuple[pin.Model,pin.Model, Dict]:
    """_Build the biomechanical model associated to one exercise for one subject_

    Args:
        mocap_mks_positions (Dict): _mocap_mks_positions is a dictionnary of mocap mks names and 3x1 global positions_
        lstm_mks_positions (Dict): _lstm_mks_positions is a dictionnary of lstm mks names and 3x1 global positions_
        meshes_folder_path (str): _meshes_folder_path is the path to the folder containing the meshes_

    Returns:
        Tuple[pin.Model,pin.GeomModel, Dict]: _returns the pinocchio model, geometry model, and a dictionnary with visuals._
    """
    sgts_poses = construct_segments_frames_challenge(mocap_mks_positions)
    sgts_mks_dict = get_segments_lstm_mks_dict_challenge()
    lstm_mks_local_positions = get_local_lstm_mks_positions(sgts_poses, lstm_mks_positions, sgts_mks_dict)
    local_segments_positions = get_local_segments_positions(sgts_poses)
    visuals_dict = {}


    # Meshes rotations
    rtorso = R.from_rotvec(np.pi/2 * np.array([0, 1, 0]))
    rupperarm = R.from_rotvec(np.pi/2 * np.array([0, 1, 0]))
    rlowerarm = R.from_rotvec(np.pi/2 * np.array([0, 1, 0]))
    rhand = R.from_rotvec(np.pi * np.array([0, 1, 0]))

    # Mesh loader
    mesh_loader = fcl.MeshLoader()

    # MODEL GENERATION 
    inertia = pin.Inertia.Zero()
    model= pin.Model()
    geom_model = pin.GeometryModel()

    # Torso with Freeflyer
    IDX_PELV_JF = model.addJoint(0,pin.JointModelFreeFlyer(),pin.SE3(np.array([[1,0,0],[0,0,-1],[0,1,0]]), np.matrix([0,0,0]).T),'pelvis_freeflyer')
    pelvis = pin.Frame('pelvis',IDX_PELV_JF,0,pin.SE3(np.eye(3), np.matrix([0,0,0]).T),pin.FrameType.OP_FRAME, inertia)
    IDX_PELV_SF = model.addFrame(pelvis,False)
    # Add markers data
    idx_frame = IDX_PELV_SF
    for i in sgts_mks_dict["pelvis"]:
        frame = pin.Frame(i,IDX_PELV_JF,idx_frame,pin.SE3(np.eye(3,3), np.matrix(lstm_mks_local_positions[i]).T),pin.FrameType.OP_FRAME, inertia) 
        idx_frame = model.addFrame(frame,False)
    
    pelvis_visual = pin.GeometryObject('pelvis', IDX_PELV_SF, IDX_PELV_JF, mesh_loader.load(meshes_folder_path+'/pelvis_mesh.STL'), pin.SE3(rtorso.as_matrix(), np.matrix([-0.15, -0.17, 0.13]).T), meshes_folder_path+'/pelvis_mesh.STL', np.array([0.0065, 0.0065, 0.0065]), False, np.array([0, 1, 1, 1]))
    geom_model.addGeometryObject(pelvis_visual)
    visuals_dict["pelvis"] = pelvis_visual

    # Lombaire L5-S1 flexion/extension
    IDX_L5S1_JF = model.addJoint(IDX_PELV_JF,pin.JointModelRZ(),pin.SE3(np.eye(3), np.matrix([0, 0, 0]).T),'L5S1_FE') 
    torso = pin.Frame('torso',IDX_L5S1_JF,idx_frame,pin.SE3(np.eye(3), np.matrix(local_segments_positions['torso']).T),pin.FrameType.OP_FRAME, inertia)
    IDX_TORSO_SF = model.addFrame(torso,False)
    idx_frame = IDX_TORSO_SF
    for i in sgts_mks_dict["torso"]:
        frame = pin.Frame(i,IDX_L5S1_JF,idx_frame,pin.SE3(np.eye(3,3), np.matrix(lstm_mks_local_positions[i]+ local_segments_positions['torso']).T),pin.FrameType.OP_FRAME, inertia) 
        idx_frame = model.addFrame(frame,False)

    torso_visual = pin.GeometryObject('torso', IDX_TORSO_SF, IDX_L5S1_JF, mesh_loader.load(meshes_folder_path+'/torso_mesh.STL'), pin.SE3(rtorso.as_matrix(), np.matrix([-0.15, 0.17, 0.13]).T), meshes_folder_path+'/torso_mesh.STL', np.array([0.0065, 0.0065, 0.0065]), False, np.array([0, 1, 1, 1]))
    geom_model.addGeometryObject(torso_visual)
    visuals_dict["torso"] = torso_visual

    abdomen_visual = pin.GeometryObject('abdomen', IDX_TORSO_SF, IDX_L5S1_JF, mesh_loader.load(meshes_folder_path+'/abdomen_mesh.STL'), pin.SE3(rtorso.as_matrix(), np.matrix([-0.12, 0.05, 0.09]).T), meshes_folder_path+'/abdomen_mesh.STL', np.array([0.0065, 0.0065, 0.0065]), False, np.array([0, 1, 1, 1]))
    geom_model.addGeometryObject(abdomen_visual)
    visuals_dict["abdomen"] = abdomen_visual

    # Shoulder YXY
    IDX_SH_Y1_JF = model.addJoint(IDX_L5S1_JF,pin.JointModelRY(),pin.SE3(np.eye(3), np.matrix(local_segments_positions['upperarm'] + local_segments_positions['torso']).T),'Shoulder_Y1') 
    upperarm = pin.Frame('upperarm_y1',IDX_SH_Y1_JF,idx_frame,pin.SE3(np.eye(3), np.matrix([0,0,0]).T),pin.FrameType.OP_FRAME, inertia)
    IDX_UPA_SF = model.addFrame(upperarm,False)
    idx_frame = IDX_UPA_SF

    shoulder_visual = pin.GeometryObject('shoulder', IDX_UPA_SF, IDX_SH_Y1_JF, mesh_loader.load(meshes_folder_path+'/shoulder_mesh.STL'), pin.SE3(np.eye(3), np.matrix([-0.16, -0.045, -0.045]).T), meshes_folder_path+'/shoulder_mesh.STL',np.array([0.0055, 0.0055, 0.0055]), False , np.array([0,1,1,0.5]))
    geom_model.addGeometryObject(shoulder_visual)
    visuals_dict["shoulder"] = shoulder_visual

    IDX_SH_X_JF = model.addJoint(IDX_SH_Y1_JF,pin.JointModelRX(),pin.SE3(np.eye(3), np.matrix([0,0,0]).T),'Shoulder_X') 
    upperarm = pin.Frame('upperarm_x',IDX_SH_X_JF,idx_frame,pin.SE3(np.eye(3), np.matrix([0,0,0]).T),pin.FrameType.OP_FRAME, inertia)
    IDX_UPA_SF = model.addFrame(upperarm,False)
    idx_frame = IDX_UPA_SF

    IDX_SH_Y_JF = model.addJoint(IDX_SH_X_JF,pin.JointModelRY(),pin.SE3(np.eye(3), np.matrix([0,0,0]).T),'Shoulder_Y2') 
    upperarm = pin.Frame('upperarm',IDX_SH_Y_JF,idx_frame,pin.SE3(np.eye(3), np.matrix([0,0,0]).T),pin.FrameType.OP_FRAME, inertia)
    IDX_UPA_SF = model.addFrame(upperarm,False)
    idx_frame = IDX_UPA_SF
    for i in sgts_mks_dict["upperarm"]:
        frame = pin.Frame(i,IDX_SH_Y_JF,idx_frame,pin.SE3(np.eye(3,3), np.matrix(lstm_mks_local_positions[i]).T),pin.FrameType.OP_FRAME, inertia) 
        idx_frame = model.addFrame(frame,False)

    upperarm_visual = pin.GeometryObject('upperarm', IDX_UPA_SF, IDX_SH_Y_JF, mesh_loader.load(meshes_folder_path+'/upperarm_mesh.STL'), pin.SE3(rupperarm.as_matrix(), np.matrix([-0.07, -0.29, 0.18]).T), meshes_folder_path+'/upperarm_mesh.STL',np.array([0.0063, 0.0060, 0.007]), False , np.array([0,1,1,0.5]))
    geom_model.addGeometryObject(upperarm_visual)
    visuals_dict["upperarm"] = upperarm_visual

    # Elbow ZY
    IDX_EL_Z_JF = model.addJoint(IDX_SH_Y_JF,pin.JointModelRZ(),pin.SE3(np.eye(3), np.matrix(local_segments_positions['lowerarm']).T),'Elbow_Z') 
    lowerarm = pin.Frame('lowerarm_z',IDX_EL_Z_JF,idx_frame,pin.SE3(np.eye(3), np.matrix([0,0,0]).T),pin.FrameType.OP_FRAME, inertia)
    IDX_LOA_SF = model.addFrame(lowerarm,False)
    idx_frame = IDX_LOA_SF

    elbow_visual = pin.GeometryObject('elbow', IDX_LOA_SF, IDX_EL_Z_JF, mesh_loader.load(meshes_folder_path+'/elbow_mesh.STL'), pin.SE3(np.eye(3), np.matrix([-0.15, -0.02, -0.035]).T), meshes_folder_path+'/elbow_mesh.STL',np.array([0.0055, 0.0055, 0.0055]), False , np.array([0,1,1,0.5]))
    geom_model.addGeometryObject(elbow_visual)
    visuals_dict["elbow"] = elbow_visual

    IDX_EL_Y_JF = model.addJoint(IDX_EL_Z_JF,pin.JointModelRY(),pin.SE3(np.eye(3), np.matrix([0,0,0]).T),'Elbow_Y') 
    lowerarm = pin.Frame('lowerarm',IDX_EL_Y_JF,idx_frame,pin.SE3(np.eye(3), np.matrix([0,0,0]).T),pin.FrameType.OP_FRAME, inertia)
    IDX_LOA_SF = model.addFrame(lowerarm,False)
    idx_frame = IDX_LOA_SF

    for i in sgts_mks_dict["lowerarm"]:
        frame = pin.Frame(i,IDX_EL_Y_JF,idx_frame,pin.SE3(np.eye(3,3), np.matrix(lstm_mks_local_positions[i]).T),pin.FrameType.OP_FRAME, inertia) 
        idx_frame = model.addFrame(frame,False)

    lowerarm_visual = pin.GeometryObject('lowerarm',IDX_LOA_SF, IDX_EL_Y_JF, mesh_loader.load(meshes_folder_path+'/lowerarm_mesh.STL'), pin.SE3(rupperarm.as_matrix(), np.matrix([-0.05, -0.25, 0.17]).T), meshes_folder_path+'/lowerarm_mesh.STL',np.array([0.0060, 0.0060, 0.0060]), False , np.array([0,1,1,0.5]))
    geom_model.addGeometryObject(lowerarm_visual)
    visuals_dict["lowerarm"] = lowerarm_visual


    # Hip ZY
    IDX_HIP_Z_JF = model.addJoint(IDX_PELV_JF,pin.JointModelRZ(),pin.SE3(np.eye(3), np.matrix(local_segments_positions['thigh']).T),'Hip_Z') 
    thigh = pin.Frame('thigh_z',IDX_HIP_Z_JF,idx_frame,pin.SE3(np.eye(3), np.matrix([0,0,0]).T),pin.FrameType.OP_FRAME, inertia)
    IDX_THIGH_SF = model.addFrame(thigh,False)
    idx_frame = IDX_THIGH_SF

    IDX_HIP_X_JF = model.addJoint(IDX_HIP_Z_JF,pin.JointModelRY(),pin.SE3(np.eye(3), np.matrix([0,0,0]).T),'Hip_X') 
    thigh = pin.Frame('thigh',IDX_HIP_X_JF,idx_frame,pin.SE3(np.eye(3), np.matrix([0,0,0]).T),pin.FrameType.OP_FRAME, inertia)
    IDX_THIGH_SF = model.addFrame(thigh,False)
    idx_frame = IDX_THIGH_SF

    for i in sgts_mks_dict["thigh"]:
        frame = pin.Frame(i,IDX_HIP_X_JF,idx_frame,pin.SE3(np.eye(3,3), np.matrix(lstm_mks_local_positions[i]).T),pin.FrameType.OP_FRAME, inertia) 
        idx_frame = model.addFrame(frame,False)

    upperleg_visual = pin.GeometryObject('upperleg',IDX_THIGH_SF, IDX_HIP_Z_JF, mesh_loader.load(meshes_folder_path+'/upperleg_mesh.STL'), pin.SE3(rupperarm.as_matrix(), np.matrix([-0.13, -0.37, 0.1]).T), meshes_folder_path+'/upperleg_mesh.STL',np.array([0.0060, 0.0060, 0.0060]), False , np.array([0,1,1,0.5]))
    geom_model.addGeometryObject(upperleg_visual)
    visuals_dict["upperleg"] = upperleg_visual


    # Knee Z
    IDX_KNEE_Z_JF = model.addJoint(IDX_HIP_X_JF,pin.JointModelRZ(),pin.SE3(np.eye(3), np.matrix(local_segments_positions['shank']).T),'Knee_Z') 
    shank = pin.Frame('shank',IDX_KNEE_Z_JF,idx_frame,pin.SE3(np.eye(3), np.matrix([0,0,0]).T),pin.FrameType.OP_FRAME, inertia)
    IDX_SHANK_SF = model.addFrame(shank,False)
    idx_frame = IDX_SHANK_SF

    for i in sgts_mks_dict["shank"]:
        frame = pin.Frame(i,IDX_KNEE_Z_JF,idx_frame,pin.SE3(np.eye(3,3), np.matrix(lstm_mks_local_positions[i]).T),pin.FrameType.OP_FRAME, inertia) 
        idx_frame = model.addFrame(frame,False)
    
    knee_visual = pin.GeometryObject('knee',IDX_SHANK_SF, IDX_KNEE_Z_JF, mesh_loader.load(meshes_folder_path+'/knee_mesh.STL'), pin.SE3(np.eye(3), np.matrix([-0.13, 0, -0.015]).T), meshes_folder_path+'/knee_mesh.STL',np.array([0.0060, 0.0060, 0.0060]), False , np.array([0,1,1,0.5]))
    geom_model.addGeometryObject(knee_visual)
    visuals_dict["knee"] = knee_visual

    lowerleg_visual = pin.GeometryObject('lowerleg',IDX_SHANK_SF, IDX_KNEE_Z_JF, mesh_loader.load(meshes_folder_path+'/lowerleg_mesh.STL'), pin.SE3(rupperarm.as_matrix(), np.matrix([-0.11, -0.40, 0.1]).T), meshes_folder_path+'/lowerleg_mesh.STL',np.array([0.0060, 0.0060, 0.0060]), False , np.array([0,1,1,0.5]))
    geom_model.addGeometryObject(lowerleg_visual)
    visuals_dict["lowerleg"] = lowerleg_visual

    # Ankle Z
    IDX_ANKLE_Z_JF = model.addJoint(IDX_KNEE_Z_JF,pin.JointModelRZ(),pin.SE3(np.eye(3), np.matrix(local_segments_positions['foot']).T),'Ankle_Z') 
    foot = pin.Frame('foot',IDX_ANKLE_Z_JF,idx_frame,pin.SE3(np.eye(3), np.matrix([0,0,0]).T),pin.FrameType.OP_FRAME, inertia)
    IDX_SFOOT_SF = model.addFrame(foot,False)
    idx_frame = IDX_SFOOT_SF

    for i in sgts_mks_dict["foot"]:
        frame = pin.Frame(i,IDX_ANKLE_Z_JF,idx_frame,pin.SE3(np.eye(3,3), np.matrix(lstm_mks_local_positions[i]).T),pin.FrameType.OP_FRAME, inertia) 
        idx_frame = model.addFrame(frame,False)
    
    foot_visual = pin.GeometryObject('foot',IDX_SFOOT_SF, IDX_ANKLE_Z_JF, mesh_loader.load(meshes_folder_path+'/foot_mesh.STL'), pin.SE3(rupperarm.as_matrix(), np.matrix([-0.11, -0.07, 0.09]).T), meshes_folder_path+'/foot_mesh.STL',np.array([0.0060, 0.0060, 0.0060]), False , np.array([0,1,1,0.5]))
    geom_model.addGeometryObject(foot_visual)
    visuals_dict["foot"] = foot_visual

    # data     = model.createData()
    # # Sample a random configuration
    # q        = zero(model.nq)
    # print('q: %s' % q.T)
    # # Perform the forward kinematics over the kinematic tree
    # pin.forwardKinematics(model,data,q)
    # # Print out the placement of each joint of the kinematic tree
    # for name, oMi in zip(model.names, data.oMi):
    #     print(("{:<24} : {: .2f} {: .2f} {: .2f}"
    #         .format( name, *oMi.translation.T.flat )))

    model.upperPositionLimit[7:] = np.array([0.393,3.142,3.142,1.22,2.53, 1.57, 2.27,0.52,2.36,0.87])
    model.lowerPositionLimit[7:] = np.array([-0.305,-1.047,-0.698,-1.57,0, -1.57, -0.52,-0.52,0,-0.35])

    return model, geom_model, visuals_dict




