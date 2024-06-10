import sys
import os
import pinocchio as pin 
import time
from pinocchio.visualize import GepettoVisualizer
import numpy as np
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from utils.read_write_utils import read_lstm_data,get_lstm_mks_names,read_mocap_data,convert_to_list_of_dicts
from utils.model_utils import *
from utils.viz_utils import place

fichier_csv_lstm_mks = "./data/sujet_1/Marche/jcp_coordinates_ncameras_augmented.csv"
fichier_csv_mocap_mks = "./data/mks_coordinates_3D_sujet1.trc"
lstm_mks_dict, mapping = read_lstm_data(fichier_csv_lstm_mks)
lstm_mks_names = get_lstm_mks_names(fichier_csv_lstm_mks)
mocap_mks_dict = read_mocap_data(fichier_csv_mocap_mks)
lstm_mks_dict = convert_to_list_of_dicts(lstm_mks_dict)

viz = GepettoVisualizer()

try:
    viz.initViewer()
except ImportError as err:
    print("Error while initializing the viewer. It seems you should install gepetto-viewer")
    print(err)
    sys.exit(0)

try:
    viz.loadViewerModel("pinocchio")
except AttributeError as err:
    print("Error while loading the viewer model. It seems you should start gepetto-viewer")
    print(err)
    sys.exit(0)


viz.viewer.gui.addXYZaxis(f'world/base_frame', [255, 0., 0, 1.], 0.015, 0.11)
place(viz, 'world/base_frame', pin.SE3(np.eye(3), np.matrix([0, 0, 0]).T))
for name in mocap_mks_dict:
    sphere_name = f'world/{name}'
    viz.viewer.gui.addSphere(sphere_name, 0.015, [0, 0., 255, 1.])
for name in lstm_mks_dict[0]:
    sphere_name = f'world/{name}'
    viz.viewer.gui.addSphere(sphere_name, 0.01, [255, 0., 0., 1.])

pelvis_pose_lstm = get_pelvis_pose(lstm_mks_dict[0])
pelvis_pose = get_pelvis_pose(mocap_mks_dict)
thigh_pose = get_thigh_pose(mocap_mks_dict)
shank_pose = get_shank_pose(mocap_mks_dict)
foot_pose = get_foot_pose(mocap_mks_dict)
torso_pose = get_torso_pose(mocap_mks_dict)
upperarm_pose = get_upperarm_pose(mocap_mks_dict)
lowerarm_pose = get_lowerarm_pose(mocap_mks_dict)
offset = pelvis_pose[:3,3] - pelvis_pose_lstm[:3,3]

compare_offsets(mocap_mks_dict, lstm_mks_dict[0])
# Iterate over all points in first sample of lstm_mks_dict and add them to the viewer

viz.viewer.gui.addXYZaxis(f'world/pelvis', [255, 0., 0, 1.], 0.008, 0.08)
viz.viewer.gui.addXYZaxis(f'world/thigh', [255, 0., 0, 1.], 0.008, 0.08)
viz.viewer.gui.addXYZaxis(f'world/shank', [255, 0., 0, 1.], 0.008, 0.08)
viz.viewer.gui.addXYZaxis(f'world/foot', [255, 0., 0, 1.], 0.008, 0.08)
viz.viewer.gui.addXYZaxis(f'world/torso', [255, 0., 0, 1.], 0.008, 0.08)
viz.viewer.gui.addXYZaxis(f'world/upperarm', [255, 0., 0, 1.], 0.008, 0.08)
viz.viewer.gui.addXYZaxis(f'world/lowerarm', [255, 0., 0, 1.], 0.008, 0.08)
place(viz, f'world/pelvis',  pin.SE3(pelvis_pose[:3, :3], np.matrix(pelvis_pose[:3,3].reshape(3,)).T))
place(viz, f'world/thigh',  pin.SE3(thigh_pose[:3, :3], np.matrix(thigh_pose[:3,3].reshape(3,)).T))
place(viz, f'world/shank',  pin.SE3(shank_pose[:3, :3], np.matrix(shank_pose[:3,3].reshape(3,)).T))
place(viz, f'world/foot',  pin.SE3(foot_pose[:3, :3], np.matrix(foot_pose[:3,3].reshape(3,)).T))
place(viz, f'world/torso',  pin.SE3(torso_pose[:3, :3], np.matrix(torso_pose[:3,3].reshape(3,)).T))
place(viz, f'world/upperarm',  pin.SE3(upperarm_pose[:3, :3], np.matrix(upperarm_pose[:3,3].reshape(3,)).T))
place(viz, f'world/lowerarm',  pin.SE3(lowerarm_pose[:3, :3], np.matrix(lowerarm_pose[:3,3].reshape(3,)).T))
# Iterate over all points in mocap_mks_dict and add them to the viewer

for key, value in mocap_mks_dict.items():
    sphere_name = f'world/{key}'
    # Place the sphere at the corresponding 3D position
    place(viz, sphere_name, pin.SE3(np.eye(3), np.matrix(value.reshape(3,)).T))

# for key, value in lstm_mks_dict[0].items():
#     if key != 'Time':
#         sphere_name = f'world/{key}'
#         # Place the sphere at the corresponding 3D position
#         place(viz, sphere_name, pin.SE3(np.eye(3), np.matrix(value.reshape(3,)+offset).T))
