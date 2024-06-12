import pinocchio as pin 
import numpy as np
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from utils.read_write_utils import plot_joint_angle_results, read_joint_angles

subject = 'sujet_1'
task = 'Exotique'

fichier_csv_lstm_mks = "./data/"+subject+"/"+task+"/jcp_coordinates_ncameras_augmented.csv"
results_directory = "./results/challenge/"+subject+"/"+task
# results_directory = "results/test/"

# # Plots 
plot_joint_angle_results(results_directory)