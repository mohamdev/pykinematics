import sys
import os
import pinocchio as pin 
import time
from pinocchio.visualize import GepettoVisualizer
import numpy as np
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from utils.read_write_utils import read_lstm_data,get_lstm_mks_names,read_mocap_data,convert_to_list_of_dicts, read_joint_positions
from utils.model_utils import get_torso_pose,get_thigh_pose,get_foot_pose,get_pelvis_pose,get_shank_pose,get_upperarm_pose,get_lowerarm_pose
from utils.viz_utils import place, vizualise_triangulated_landmarks

no_sujet="1"
trial="marche"
trial_folder = "Marche"

fichier_csv_lstm_mks = f'./data/sujet_{no_sujet}/{trial_folder}/jcp_coordinates_ncameras.trc'
joints_pos_dict, nb_samples = read_joint_positions(fichier_csv_lstm_mks)

# print(joints_pos_dict)

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

vizualise_triangulated_landmarks(joints_pos_dict, nb_samples, 0.03, viz)
