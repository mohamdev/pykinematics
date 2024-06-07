import eigenpy
import hppfcl as fcl
import pinocchio as pin
import sys
from pinocchio.visualize import GepettoVisualizer
import numpy as np
from scipy.spatial.transform import Rotation as R
import time
from sklearn.preprocessing import normalize

#Build inertia matrix from 6 inertia components
def make_inertia_matrix(ixx, ixy, ixz, iyy, iyz, izz):
    return np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])

def orthogonalize_matrix(matrix):
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
    for name, pose in sgts_poses.items():
        print(name, " rot det : ", np.linalg.det(pose[:3,:3]))
    return sgts_poses

def get_segments_lstm_mks_dict_challenge():
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

#Build model
# - sgts_poses corresponds to a dictionnary to segments poses and names, constructed from global mks positions
# - lstm_mks_positions is a dictionnary of lstm mks names and 3x1 global positions
# - sgts_mks_dict a dictionnary containing the segments names, and the corresponding list of lstm mks names attached to the segment
# - returns a dictionnary of lstm mks names and their 3x1 local positions 
def get_local_lstm_mks_positions(sgts_poses, lstm_mks_positions, sgts_mks_dict):
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

                if marker == 'RShoulder':
                    print("segment:", segment)
                    print("position:", marker_global_pos[:3] )
                    print("torso position:", segment_pose[:3,3])
                    print("torso to RShoulder:", marker_global_pos[:3] - segment_pose[:3,3])
                    print("local pos: ", marker_local_pos)
                    print("rot det :", np.linalg.det(segment_pose_inv[:3,:3]))

                # Store the local position in the dictionary
                lstm_mks_local_positions[marker] = marker_local_pos

    return lstm_mks_local_positions

#Build model
# - mocap_mks_positions is a dictionnary of mocap mks names and 3x1 global positions
# - lstm_mks_positions is a dictionnary of lstm mks names and 3x1 global positions
def build_model_challenge(mocap_mks_positions, lstm_mks_positions):
    model = pin.Model()
    sgts_poses = construct_segments_frames_challenge(mocap_mks_positions)
    lstm_mks_local_positions = get_local_lstm_mks_positions(sgts_poses, lstm_mks_positions)

