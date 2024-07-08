import pinocchio as pin 
import numpy as np
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from utils.read_write_utils import plot_joint_angle_results_lowerbody, read_joint_angles_lowerbody, read_mocap_data
import matplotlib.pyplot as plt


# subject = 'sujet_1'
# task = 'Exotique'

# fichier_csv_lstm_mks = "./data/"+subject+"/"+task+"/jcp_coordinates_ncameras_augmented.csv"
# results_directory = "./results/challenge/"+subject+"/"+task
# # results_directory = "results/test/"

# # # Plots 
# plot_joint_angle_results(results_directory)

trial = "trial_02"
tache = "squat_8kg"
nom_sujet = "Maxime"

fichier_csv_mocap_mks = "./data/mocap_data/"+ trial + "/mks_"+ tache + "_" + nom_sujet + ".csv" #positions des mks _ref_ 
# fichier_csv_JA_ipopt= "./results/lowerbody_ik/"+ trial + "/joint_angles_"+ tache + "_" + nom_sujet + "_ipopt.csv"
fichier_csv_JA_qp= "./results/lowerbody_ik/"+ trial + "/joint_angles_"+ tache + "_" + nom_sujet + ".csv"


# q_est_ipopt= read_joint_angles_lowerbody(fichier_csv_JA_ipopt)
q_est_qp= read_joint_angles_lowerbody(fichier_csv_JA_qp)

# Diviser les colonnes en deux groupes de 6
columns_part1 = q_est_qp.columns[:6]
columns_part2 = q_est_qp.columns[6:]

# columns_part_1 = q_est_ipopt.columns[:6]
# columns_part_2 = q_est_ipopt.columns[6:]
# Tracer le premier groupe de colonnes
fig1, axs1 = plt.subplots(len(columns_part1), figsize=(12, 18))
fig1.suptitle('Trajectoires des angles articulaires (Partie 1)')

for i, col in enumerate(columns_part1):
    axs1[i].plot(q_est_qp[col], label='QP', color='blue')  # QP en bleu
    # axs1[i].plot(q_est_ipopt[col], label='IPOPT', color='orange')  # IPOPT en orange
    axs1[i].set_title(col)
    axs1[i].set_title(col)
    axs1[i].set_xlabel('Index')
    axs1[i].set_ylabel('Angle (rad)')

plt.tight_layout()
plt.show()

# Tracer le second groupe de colonnes
fig2, axs2 = plt.subplots(len(columns_part2), figsize=(12, 18))
fig2.suptitle('Trajectoires des angles articulaires (Partie 2)')

for i, col in enumerate(columns_part2):
    axs2[i].plot(q_est_qp[col], label='QP', color='blue')  # QP en bleu
    # axs2[i].plot(q_est_ipopt[col], label='IPOPT', color='orange')  # IPOPT en orange
    axs2[i].set_title(col)
    axs2[i].set_title(col)
    axs2[i].set_xlabel('Index')
    axs2[i].set_ylabel('Angle (rad)')

plt.tight_layout()
plt.show()