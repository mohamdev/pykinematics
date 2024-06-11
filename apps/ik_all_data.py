from utils.read_write_utils import read_lstm_data, get_lstm_mks_names, read_mocap_data, convert_to_list_of_dicts, write_joint_angle_results
from utils.ik_utils import IK_Casadi
import pinocchio as pin 
from utils.model_utils import get_subset_challenge_mks_names, get_segments_lstm_mks_dict_challenge, build_model_challenge
import numpy as np 

subjects = ['sujet_1', 'sujet_2']
tasks = ['Assis-debout','Exotique','Manutention','Marche']
meshes_folder_path = "meshes/" #Changes le par ton folder de meshes

for subject in subjects :
    for task in tasks: 
        fichier_csv_lstm_mks = "data/"+subject+"/"+task+"/jcp_coordinates_ncameras_augmented.csv"
        fichier_csv_mocap_mks = "data/mks_coordinates_3D_"+subject+".trc"
        fichier_csv_lstm_mks_calib = "data/"+subject+"/Marche/jcp_coordinates_ncameras_augmented.csv"

        #Read data
        lstm_mks_dict, mapping = read_lstm_data(fichier_csv_lstm_mks)
        lstm_mks_dict_calib, mapping_calib = read_lstm_data(fichier_csv_lstm_mks_calib)
        lstm_mks_names = get_lstm_mks_names(fichier_csv_lstm_mks) #Liste des noms des mks du lstm (totalité des mks)
        subset_challenge_mks_names = get_subset_challenge_mks_names() #Cette fonction te retourne les noms des markers dont on a besoin pour le challenge
        mocap_mks_dict = read_mocap_data(fichier_csv_mocap_mks) #Markers mocap, pas utilisés ici car merdiques pour le moment
        lstm_mks_dict_calib = convert_to_list_of_dicts(lstm_mks_dict_calib)
        lstm_mks_dict = convert_to_list_of_dicts(lstm_mks_dict) #Je convertis ton dictionnaire de trajectoires (arrays) en une "trajectoire de dictionnaires", c'est plus facile à manipuler pour la calib
        lstm_mks_positions_calib = lstm_mks_dict_calib[0] #Je prends la première frame de la trajectoire pour construire le modèle
        seg_names_mks = get_segments_lstm_mks_dict_challenge() #Dictionnaire contenant les noms des segments + les mks correspondnat à chaque segment



        #C'est normal qu'il y ait deux fois le même argument, normalement le 1er argument c'est les mks mocap. 
        model, geom_model, visuals_dict = build_model_challenge(lstm_mks_positions_calib, lstm_mks_positions_calib, meshes_folder_path)

        q0 = pin.neutral(model)

        ### IK 

        ik_problem = IK_Casadi(model, lstm_mks_dict, q0)

        q = ik_problem.solve_ik()

        q=np.array(q)
        directory_name = "results/calib_lstm/"+subject+"/"+task
        write_joint_angle_results(directory_name,q)

