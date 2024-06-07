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

lstm_mks_positions = lstm_mks_dict[0]
sgts_poses = construct_segments_frames_challenge(lstm_mks_positions)
sgts_mks_dict = get_segments_lstm_mks_dict_challenge()
local_positions = get_local_lstm_mks_positions(sgts_poses, lstm_mks_positions, sgts_mks_dict)
print(local_positions)