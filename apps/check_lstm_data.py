import pandas as pd
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from utils.read_write_utils import *
from utils.model_utils import *
from utils.viz_utils import *

fichier_csv_lstm_mks = "../data/jcp_coordinates_ncameras_augmented.csv"
fichier_csv_mocap_mks = "../data/mks_coordinates_3D.trc"
lstm_mks_dict, mapping = read_lstm_data(fichier_csv_lstm_mks)
lstm_mks_names = get_lstm_mks_names(fichier_csv_lstm_mks)
mocap_mks_dict = read_mocap_data(fichier_csv_mocap_mks)
lstm_mks_dict = convert_to_list_of_dicts(lstm_mks_dict)


# print(lstm_mks_dict)

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


viz.viewer.gui.addXYZaxis('world/base_frame', [255, 0., 0, 1.], 0.04, 0.2)
viz.viewer.gui.addXYZaxis('world/torso', [255, 0., 0, 1.], 0.01, 0.11)
viz.viewer.gui.addXYZaxis('world/upperarm', [255, 0., 0, 1.], 0.01, 0.11)
viz.viewer.gui.addXYZaxis('world/lowerarm', [255, 0., 0, 1.], 0.01, 0.11)
viz.viewer.gui.addXYZaxis('world/pelvis', [255, 0., 0, 1.], 0.01, 0.11)
viz.viewer.gui.addXYZaxis('world/thigh', [255, 0., 0, 1.], 0.01, 0.11)
viz.viewer.gui.addXYZaxis('world/shank', [255, 0., 0, 1.], 0.01, 0.11)
viz.viewer.gui.addXYZaxis('world/foot', [255, 0., 0, 1.], 0.01, 0.11)
place(viz, 'world/base_frame', pin.SE3(np.eye(3), np.matrix([0, 0, 0]).T))

for name in lstm_mks_names:
    sphere_name = f'world/{name}'
    viz.viewer.gui.addSphere(sphere_name, 0.015, [0, 0., 255, 1.])

# Iterate over all points in first sample of lstm_mks_dict and add them to the viewer
for i in range(len(lstm_mks_dict)):
# for i in range(25):
    torso_pose = get_torso_pose(lstm_mks_dict[i])
    upperarm_pose = get_upperarm_pose(lstm_mks_dict[i])
    lowerarm_pose = get_lowerarm_pose(lstm_mks_dict[i])
    pelvis_pose = get_pelvis_pose(lstm_mks_dict[i])
    thigh_pose = get_thigh_pose(lstm_mks_dict[i])
    shank_pose = get_shank_pose(lstm_mks_dict[i])
    foot_pose = get_foot_pose(lstm_mks_dict[i])
    for name in lstm_mks_names:
        sphere_name = f'world/{name}'
        place(viz, sphere_name, pin.SE3(np.eye(3), np.matrix(lstm_mks_dict[i][name].reshape(3,)).T))
        place(viz, 'world/torso', pin.SE3(torso_pose[:3,:3], np.matrix(torso_pose[:3,3].reshape(3,)).T))
        place(viz, 'world/upperarm', pin.SE3(upperarm_pose[:3,:3], np.matrix(upperarm_pose[:3,3].reshape(3,)).T))
        place(viz, 'world/lowerarm', pin.SE3(lowerarm_pose[:3,:3], np.matrix(lowerarm_pose[:3,3].reshape(3,)).T))
        place(viz, 'world/pelvis', pin.SE3(pelvis_pose[:3,:3], np.matrix(pelvis_pose[:3,3].reshape(3,)).T))
        place(viz, 'world/thigh', pin.SE3(thigh_pose[:3,:3], np.matrix(thigh_pose[:3,3].reshape(3,)).T))
        place(viz, 'world/shank', pin.SE3(shank_pose[:3,:3], np.matrix(shank_pose[:3,3].reshape(3,)).T))
        place(viz, 'world/foot', pin.SE3(foot_pose[:3,:3], np.matrix(foot_pose[:3,3].reshape(3,)).T))
    time.sleep(0.03)

# # Iterate over all points in mocap_mks_dict and add them to the viewer
# for key, value in mocap_mks_dict.items():
#     # Add a sphere for each point with the given key as the name
#     sphere_name = f'world/{key}'
#     viz.viewer.gui.addSphere(sphere_name, 0.01, [0, 0., 0, 1.])
    
#     # Place the sphere at the corresponding 3D position
#     place(viz, sphere_name, pin.SE3(np.eye(3), np.matrix(value.reshape(3,)).T))

