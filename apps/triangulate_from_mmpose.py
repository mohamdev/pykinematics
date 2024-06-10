from utils.triangulation_utils import *
from utils.read_write_utils import *
from utils.linear_algebra_utils import *

mmpose_model="body26"
no_sujet="1"
trial="marche"
trial_folder = "Marche"

liste_fichiers = [
    f'./data/sujet_{no_sujet}/{trial_folder}/result_{trial}_26585_sujet{no_sujet}.txt',
    f'./data/sujet_{no_sujet}/{trial_folder}/result_{trial}_26587_sujet{no_sujet}.txt',
    f'./data/sujet_{no_sujet}/{trial_folder}/result_{trial}_26578_sujet{no_sujet}.txt',
    f'./data/sujet_{no_sujet}/{trial_folder}/result_{trial}_26579_sujet{no_sujet}.txt',
    f'./data/sujet_{no_sujet}/{trial_folder}/result_{trial}_26580_sujet{no_sujet}.txt',
    f'./data/sujet_{no_sujet}/{trial_folder}/result_{trial}_26582_sujet{no_sujet}.txt',
    f'./data/sujet_{no_sujet}/{trial_folder}/result_{trial}_26583_sujet{no_sujet}.txt',
    f'./data/sujet_{no_sujet}/{trial_folder}/result_{trial}_26584_sujet{no_sujet}.txt',
    f'./data/sujet_{no_sujet}/{trial_folder}/result_{trial}_26586_sujet{no_sujet}.txt'
]

donnees_cameras=[]
for fichier in liste_fichiers :
    donnees_cameras.append(read_mmpose_file(fichier))

print(len(donnees_cameras), 'caméras sont utilisées pour cette triangulation')

# Nombre de points joints (JCP)
if mmpose_model=='body26':
    nombre_points = 26
else:
    nombre_points = 133

uvs=[]
for donnee_camera in donnees_cameras:
    uvs_camera = np.array([[ligne[2*i], ligne[2*i + 1]] for ligne in donnee_camera for i in range(nombre_points)])
    # uvs_camera = low_pass_filter_data(uvs_camera)
    uvs_camera = uvs_camera.reshape(-1, nombre_points, 2)
    uvs.append(uvs_camera)

# #Matrices intrinsèques recopiées à partir du fichier calib dans data.
donnees = get_cams_params_challenge()

for cam in donnees :
    R_extrinseque = np.zeros(shape=(3,3))
    rotation=np.array(donnees[cam]["rotation"])
    # print(rotation)
    translation=np.array([donnees[cam]["translation"]]).reshape(3,1)
    cv.Rodrigues(rotation.reshape(3,1), R_extrinseque)
    # print(R_extrinseque)
    projection = np.concatenate([R_extrinseque, translation], axis=-1)
    # print(projection)
    donnees[cam]["projection"] = projection
    # print("avec boucle pour la cam : ", cam , donnees[cam]["projection"])

rotations=[]
translations=[]
dists=[]
mtxs=[]
projections=[]

# Créer un dictionnaire pour stocker les données correspondant aux numéros de fichier
donnees_correspondantes = {}

# Parcourir les fichiers
for fichier in liste_fichiers:
    # Extraire le numéro de série de la caméra
    numero_fichier = fichier.split('_')[-2]
    
    # Vérifier si le numéro de fichier correspond à une entrée dans le dictionnaire de données
    if numero_fichier in donnees:
        donnees_correspondantes[fichier] = donnees[numero_fichier]
        # Ajouter la matrice mtx à la liste mtxs
        mtxs.append(donnees[numero_fichier]["mtx"])
        dists.append(donnees[numero_fichier]["dist"])
        translations.append(donnees[numero_fichier]["translation"])
        rotations.append(donnees[numero_fichier]["rotation"])
        projections.append(donnees[numero_fichier]["projection"])


scores = read_mmpose_scores(liste_fichiers)
threshold = 0.7
p3ds_frames = triangulate_points_adaptive(uvs, mtxs, dists, projections, scores, threshold)
p3ds_frames = butterworth_filter(p3ds_frames, 5.0)
# p3ds_frames = triangulate_points(uvs, mtxs, dists, projections)

if mmpose_model == 'body26':
    joints_trc = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle', 'head', 'neck', 'hip', 'left_big_toe', 'right_big_toe', 'left_small_toe', 'right_small_toe', 'left_heel', 'right_heel']
else:
    joints_trc = [
        "Nose", "left_eye", "right_eye", "left_ear", "right_ear", 
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", 
        "left_wrist", "right_wrist", "left_hip", "right_hip", 
        "left_knee", "right_knee", "left_ankle", "right_ankle", 
        "left_big_toe", "left_small_toe", "left_heel", 
        "right_big_toe", "right_small_toe", "right_heel",
        "face-0", "face-1", "face-2", "face-3", "face-4", "face-5", 
        "face-6", "face-7", "face-8", "face-9", "face-10", "face-11", 
        "face-12", # Continue for face points up to face-67
        *["face-" + str(i) for i in range(13, 68)],
        "left_hand_root", "left_thumb1", "left_thumb2", "left_thumb3", "left_thumb4",
        "left_forefinger1", "left_forefinger2", "left_forefinger3", "left_forefinger4",
        "left_middle_finger1", "left_middle_finger2", "left_middle_finger3", "left_middle_finger4",
        "left_ring_finger1", "left_ring_finger2", "left_ring_finger3", "left_ring_finger4",
        "left_pinky_finger1", "left_pinky_finger2", "left_pinky_finger3", "left_pinky_finger4",
        "right_hand_root", "right_thumb1", "right_thumb2", "right_thumb3", "right_thumb4",
        "right_forefinger1", "right_forefinger2", "right_forefinger3", "right_forefinger4",
        "right_middle_finger1", "right_middle_finger2", "right_middle_finger3", "right_middle_finger4",
        "right_ring_finger1", "right_ring_finger2", "right_ring_finger3", "right_ring_finger4",
        "right_pinky_finger1", "right_pinky_finger2", "right_pinky_finger3", "right_pinky_finger4"
    ]

# Écrire les résultats dans un fichier TRC
with open(f'./data/sujet_{no_sujet}/{trial_folder}/jcp_coordinates_ncameras.trc', 'w') as f:
    f.write(','.join(joints_trc) + '\n')
    for frame in p3ds_frames:
        frame_flat = frame.flatten()
        f.write(','.join(map(str, frame_flat)) + '\n')

with open(f'./data/sujet_{no_sujet}/{trial_folder}/jcp_coordinates_ncameras_for_lstm.trc', 'w') as f:
    for frame in p3ds_frames:
        frame = frame.flatten()
        f.write(','.join(map(str, frame)) + '\n')


