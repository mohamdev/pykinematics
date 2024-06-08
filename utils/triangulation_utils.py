import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import linalg
import os
import time
import pandas as pd
from scipy.spatial.transform import Rotation as R

def DLT(projections, points):
    num_camera=len(projections)
    A=[]
    for i in range (num_camera):
        P=projections[i]
        point = points[i]
        # print ('point',point)

        for j in range (len(point)):
            A.append(point[j][1]*P[2,:] - P[1,:])
            A.append(P[0,:] - point[j][0]*P[2,:])

    A = np.array(A).reshape((-1,4))
    #print('A: ')
    #print(A)
    B = A.transpose() @ A
    _, _, Vh = linalg.svd(B, full_matrices = False)

    # print('Triangulated point: ')
    # print(Vh[3,0:3]/Vh[3,3])
    return Vh[3,0:3]/Vh[3,3]

def DLT_adaptive(projections, points, idx_cams_used: list):
    A=[]
    for i in range(len(idx_cams_used)):
        P=projections[idx_cams_used[i]]
        point = points[i]
        # print ('point',point)

        for j in range (len(point)):
            A.append(point[j][1]*P[2,:] - P[1,:])
            A.append(P[0,:] - point[j][0]*P[2,:])

    A = np.array(A).reshape((-1,4))
    #print('A: ')
    #print(A)
    B = A.transpose() @ A
    _, _, Vh = linalg.svd(B, full_matrices = False)

    # print('Triangulated point: ')
    # print(Vh[3,0:3]/Vh[3,3])
    return Vh[3,0:3]/Vh[3,3]

def is_camera_used(score: float, threshold: float)->bool:
    if score > threshold:
        return True
    else:
        return False

def which_cameras_used(scores: list, threshold: float)->list:
    which_cam_used_list = []
    for score in scores:
        which_cam_used_list.append(is_camera_used(score, threshold))
    return which_cam_used_list

def index_cameras_used(which_cam_used_list: list):
    index_cameras_used = []
    for i in range(len(which_cam_used_list)):
        if which_cam_used_list[i] == True:
            index_cameras_used.append(i)
    return index_cameras_used


def triangulate_points(uvs, mtxs, dists, projections):
    num_frames = len(uvs[0])  # Nombre de frames, basé sur la première caméra
    num_points = len(uvs[0][0])  # Nombre de points, basé sur la première frame de la première caméra
    p3ds_frames = []

    for frame_idx in range(num_frames):

        p3ds_frame=[]
        points_2d_per_frame = [uv[frame_idx] for uv in uvs] 

        undistorted_points = []

        for cam_idx in range(len(points_2d_per_frame)):
            points = points_2d_per_frame[cam_idx]
            distCoeffs_mat = np.array([dists[cam_idx]]).reshape(-1, 1)
            points_undistorted = cv.undistortPoints(np.array(points).reshape(-1, 1, 2), mtxs[cam_idx], distCoeffs_mat)

            undistorted_points.append(points_undistorted)

        for point_idx in range(num_points):
            points_per_point = [undistorted_points[cam_idx][point_idx] for cam_idx in range(len(undistorted_points))]
            _p3d = DLT(projections, points_per_point)
            p3ds_frame.append(_p3d)

        p3ds_frames.append(p3ds_frame)

    return np.array(p3ds_frames)

def triangulate_points_adaptive(uvs, mtxs, dists, projections, scores: list, threshold: float):
    num_frames = len(uvs[0])  # Nombre de frames, basé sur la première caméra
    num_points = len(uvs[0][0])  # Nombre de points, basé sur la première frame de la première caméra
    p3ds_frames = []

    for frame_idx in range(num_frames):
        which_cam_used_list = which_cameras_used(scores[frame_idx], threshold)
        p3ds_frame=[]
        points_2d_per_frame = [uv[frame_idx] for uv in uvs] 

        undistorted_points = []

        idx_cams_used = index_cameras_used(which_cam_used_list)
        print(idx_cams_used)
        for cam_idx in idx_cams_used:
            points = points_2d_per_frame[cam_idx]
            distCoeffs_mat = np.array([dists[cam_idx]]).reshape(-1, 1)
            points_undistorted = cv.undistortPoints(np.array(points).reshape(-1, 1, 2), mtxs[cam_idx], distCoeffs_mat)
            undistorted_points.append(points_undistorted)

        for point_idx in range(num_points):
            points_per_point = [undistorted_points[i][point_idx] for i in range(len(undistorted_points))]
            _p3d = DLT_adaptive(projections, points_per_point, idx_cams_used)
            p3ds_frame.append(_p3d)

        p3ds_frames.append(p3ds_frame)

    return np.array(p3ds_frames)