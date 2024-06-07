import pinocchio as pin 
from pinocchio.visualize import GepettoVisualizer
import numpy as np
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from utils.read_write_utils import read_lstm_data, get_lstm_mks_names, read_mocap_data, convert_to_list_of_dicts
from utils.model_utils import get_subset_challenge_mks_names, get_segments_lstm_mks_dict_challenge, build_model_challenge
from utils.viz_utils import place

fichier_csv_lstm_mks = "data/jcp_coordinates_ncameras_augmented.csv"
fichier_csv_mocap_mks = "data/mks_coordinates_3D.trc"
meshes_folder_path = "meshes/" #Changes le par ton folder de meshes

#Read data
lstm_mks_dict, mapping = read_lstm_data(fichier_csv_lstm_mks)
lstm_mks_names = get_lstm_mks_names(fichier_csv_lstm_mks) #Liste des noms des mks du lstm (totalité des mks)
subset_challenge_mks_names = get_subset_challenge_mks_names() #Cette fonction te retourne les noms des markers dont on a besoin pour le challenge
mocap_mks_dict = read_mocap_data(fichier_csv_mocap_mks) #Markers mocap, pas utilisés ici car merdiques pour le moment
lstm_mks_dict = convert_to_list_of_dicts(lstm_mks_dict) #Je convertis ton dictionnaire de trajectoires (arrays) en une "trajectoire de dictionnaires", c'est plus facile à manipuler pour la calib
lstm_mks_positions_calib = lstm_mks_dict[0] #Je prends la première frame de la trajectoire pour construire le modèle
seg_names_mks = get_segments_lstm_mks_dict_challenge() #Dictionnaire contenant les noms des segments + les mks correspondnat à chaque segment


#C'est normal qu'il y ait deux fois le même argument, normalement le 1er argument c'est les mks mocap. 
model, geom_model, visuals_dict = build_model_challenge(lstm_mks_positions_calib, lstm_mks_positions_calib, meshes_folder_path)


visual_model = geom_model
viz = GepettoVisualizer(model, geom_model, visual_model)

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

for name, visual in visuals_dict.items():
    viz.viewer.gui.setColor(viz.getViewerNodeName(visual, pin.GeometryType.VISUAL), [0, 1, 1, 0.5])

# Set color for other visual objects similarly
data = model.createData()

q0 = pin.neutral(model)

viz.display(q0)

# viz.viewer.gui.addXYZaxis('world/base_frame', [255, 0., 0, 1.], 0.02, 0.15)
# place(viz, 'world/base_frame', pin.SE3(np.eye(3), np.matrix([0, 0, 0]).T))

for seg_name, mks in seg_names_mks.items():
    viz.viewer.gui.addXYZaxis(f'world/{seg_name}', [255, 0., 0, 1.], 0.008, 0.08)
    for mk_name in mks:
        sphere_name = f'world/{mk_name}'
        viz.viewer.gui.addSphere(sphere_name, 0.01, [0, 0., 255, 1.])


pin.forwardKinematics(model, data, q0)
pin.updateFramePlacements(model, data)

for seg_name, mks in seg_names_mks.items():
    #Display markers from model
    for mk_name in mks:
        sphere_name = f'world/{mk_name}'
        mk_position = data.oMf[model.getFrameId(mk_name)].translation
        place(viz, sphere_name, pin.SE3(np.eye(3), np.matrix(mk_position.reshape(3,)).T))
    
    #Display frames from model
    frame_name = f'world/{seg_name}'
    frame_se3= data.oMf[model.getFrameId(seg_name)]
    place(viz, frame_name, frame_se3)
        
