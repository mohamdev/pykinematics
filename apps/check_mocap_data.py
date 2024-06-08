import sys
import os
import pinocchio as pin 
import time
from pinocchio.visualize import GepettoVisualizer
import numpy as np
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from utils.read_write_utils import read_lstm_data,get_lstm_mks_names,read_mocap_data,convert_to_list_of_dicts
from utils.model_utils import get_torso_pose,get_thigh_pose,get_foot_pose,get_pelvis_pose,get_shank_pose,get_upperarm_pose,get_lowerarm_pose
from utils.viz_utils import place

fichier_csv_lstm_mks = "data/jcp_coordinates_ncameras_augmented.csv"
fichier_csv_mocap_mks = "data/mks_coordinates_3D_sujet1.trc"
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


place(viz, 'world/base_frame', pin.SE3(np.eye(3), np.matrix([0, 0, 0]).T))

for name in mocap_mks_dict:
    sphere_name = f'world/{name}'
    viz.viewer.gui.addSphere(sphere_name, 0.015, [0, 0., 255, 1.])

# Iterate over all points in first sample of lstm_mks_dict and add them to the viewer

# Iterate over all points in mocap_mks_dict and add them to the viewer
for key, value in mocap_mks_dict.items():
    # Add a sphere for each point with the given key as the name
    sphere_name = f'world/{key}'
    viz.viewer.gui.addSphere(sphere_name, 0.01, [0, 0., 0, 1.])
    
    # Place the sphere at the corresponding 3D position
    place(viz, sphere_name, pin.SE3(np.eye(3), np.matrix(value.reshape(3,)).T))
    time.sleep(0.03)

