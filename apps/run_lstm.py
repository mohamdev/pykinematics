from utils.data_augmentation_utils import *


os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
subject="sujet1"
trial="exotique"
no_sujet = 1
trial_folder = "Exotique"

#Modification du fichier résultat pour compatibilité avec LSTM___________________________________________________________________________

len_csv = 114
# Chemins vers les fichiers
csv_file_path = "./data/a9fd6740-1c9d-40df-beca-15e6eecf08d7.csv"
trc_file_path = f'./data/sujet_{no_sujet}/{trial_folder}/jcp_coordinates_ncameras_for_lstm.trc'

# Lire les fichiers
csv_data = pd.read_csv(csv_file_path, header=None)
trc_data = pd.read_csv(trc_file_path)

# Colonnes à garder dans le fichier .trc et leur nouvel ordre
joints_csv = ['Neck', 'RShoulder', 'RElbow', 'RWrist', 'LShoulder', 'LElbow', 'LWrist', 'midHip', 'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle', 'LBigToe', 'LSmallToe', 'LHeel', 'RBigToe', 'RSmallToe', 'RHeel']
joints_trc = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle', 'head', 'neck', 'hip', 'left_big_toe', 'right_big_toe', 'left_small_toe', 'right_small_toe', 'left_heel', 'right_heel']

joint_mapping = get_joint_mapping()

print('le trc a ', len(trc_data.columns), 'colonnes')

# Indices des colonnes à garder
columns_to_keep = []
for joint in joint_mapping.keys():
    print(joint)
    idx = (joints_trc.index(joint)* 3)
    print(idx)
    columns_to_keep.extend([idx, idx + 1, idx + 2])
print ('colonne à garder = ', columns_to_keep)

# Garder et réorganiser les colonnes dans le fichier .trc
trc_filtered_data = trc_data.iloc[:, columns_to_keep]

# Renommer les colonnes pour correspondre à celles du fichier .csv
new_column_names = []
for joint in joints_trc:
    if joint in joint_mapping:
        new_column_names.extend([joint_mapping[joint] + '_X', joint_mapping[joint] + '_Y', joint_mapping[joint] + '_Z'])

trc_filtered_data.columns = new_column_names

# Ajouter une colonne avec une numérotation des lignes
trc_filtered_data.insert(0, 'Frame', range(1, len(trc_filtered_data) + 1))

# Ajouter une colonne avec une indentation de 1/60 en commençant à 0
trc_filtered_data.insert(1, 'Time', np.arange(0, len(trc_filtered_data))/60)


# Ajouter des lignes supplémentaires pour correspondre au nombre de lignes du fichier .csv si nécessaire
num_rows_to_add = len(csv_data) - len(trc_filtered_data)
print("num rows to add:", num_rows_to_add)
print(len(csv_data))
print(len(trc_filtered_data))
if num_rows_to_add > 0:
    additional_rows = pd.DataFrame(0.0, index=range(num_rows_to_add), columns=trc_filtered_data.columns)
    trc_filtered_data = pd.concat([trc_filtered_data, additional_rows], ignore_index=True)

pathInputTRCFile=f'./data/sujet_{no_sujet}/{trial_folder}/jcp_coordinates_ncameras_transformed.trc'
# Sauvegarder le fichier .trc transformé
trc_filtered_data.to_csv(f'./data/sujet_{no_sujet}/{trial_folder}/jcp_coordinates_ncameras_transformed.trc', header=False, index=False)

if subject =='sujet1':
    subject_mass=60.0
    subject_height=1.57
elif subject =='sujet2':
    subject_mass=58.0
    subject_height=1.74
pathOutputTRCFile=f'./data/sujet_{no_sujet}/{trial_folder}/jcp_coordinates_ncameras_filtered_augmented.trc'
augmenterDir=f'./augmentation_model/'


augmentTRC(pathInputTRCFile, subject_mass, subject_height, pathOutputTRCFile, augmenterDir, augmenterModelName="LSTM", augmenter_model='v0.3', offset=True)







   
