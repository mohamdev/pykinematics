# import eigenpy
from utils.read_write_utils import read_lstm_data, get_lstm_mks_names, read_mocap_data, convert_to_list_of_dicts, write_joint_angle_results
from utils.ik_utils import IK_Casadi
import pinocchio as pin 
from utils.model_utils import get_subset_challenge_mks_names, get_segments_lstm_mks_dict_challenge, build_model_challenge, get_segments_mocap_mks
import sys
from pinocchio.visualize import GepettoVisualizer
from utils.viz_utils import place
import numpy as np 
import time 


subject = 'subject1'
type = 'train'
task= 'balancing'
#fichier_csv_mocap_mks = "./data/mocap_data/Lowerbody_Cal_.csv"  #just to check the markers
fichier_csv_mocap_mks = "./data/mocap_data/"+ subject +"/mks_data_"+ type +"_"+task +".csv"

meshes_folder_path = "meshes/" #Changes le par ton folder de meshes

#Read data
mocap_mks_list = read_mocap_data(fichier_csv_mocap_mks)
mocap_mks_dict_sample0 = mocap_mks_list[0] 

seg_names_mks = get_segments_mocap_mks()


#C'est normal qu'il y ait deux fois le même argument, normalement le 1er argument c'est les mks mocap. 
model, geom_model, visuals_dict = build_model_challenge(mocap_mks_dict_sample0, mocap_mks_dict_sample0, meshes_folder_path)

q0 = pin.neutral(model)
q0[7:]=0.0001*np.ones(model.nq-7)

### IK 

###### FOR MANUTENTION ONLY TRY TO MITIGATE MOONWALK ######

# for jj in range(1,len(lstm_mks_dict)):
#     lstm_mks_dict[jj]["r_toe_study"]=lstm_mks_dict[0]["r_toe_study"]
#     lstm_mks_dict[jj]["r_ankle_study"]=lstm_mks_dict[0]["r_ankle_study"]
#     lstm_mks_dict[jj]["r_mankle_study"]=lstm_mks_dict[0]["r_mankle_study"]
#     lstm_mks_dict[jj]["r_5meta_study"]=lstm_mks_dict[0]["r_5meta_study"]
#     lstm_mks_dict[jj]["r_calc_study"]=lstm_mks_dict[0]["r_calc_study"]


ik_problem = IK_Casadi(model, mocap_mks_list, q0)

q = ik_problem.solve_ik()

q=np.array(q)
directory_name = "results/lowerbody/"+subject+"/"+task
write_joint_angle_results(directory_name,q)

### Visualisation of the obtained trajectory 

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

for seg_name, mks in seg_names_mks.items():
    viz.viewer.gui.addXYZaxis(f'world/{seg_name}', [0, 255., 0, 1.], 0.008, 0.08)
    for mk_name in mks:
        sphere_name_model = f'world/{mk_name}_model'
        sphere_name_raw = f'world/{mk_name}_raw'
        viz.viewer.gui.addSphere(sphere_name_model, 0.01, [0, 0., 255, 1.])
        viz.viewer.gui.addSphere(sphere_name_raw, 0.01, [255, 0., 0, 1.])

# Set color for other visual objects similarly
data = model.createData()

for i in range(len(q)):
    q_i = q[i]
    viz.display(q_i)

    pin.forwardKinematics(model, data, q_i)
    pin.updateFramePlacements(model, data)

    for seg_name, mks in seg_names_mks.items():
        #Display markers from model
        for mk_name in mks:
            sphere_name_model = f'world/{mk_name}_model'
            sphere_name_raw = f'world/{mk_name}_raw'
            mk_position = data.oMf[model.getFrameId(mk_name)].translation
            place(viz, sphere_name_model, pin.SE3(np.eye(3), np.matrix(mk_position.reshape(3,)).T))
            place(viz, sphere_name_raw, pin.SE3(np.eye(3), np.matrix(mocap_mks_list[i][mk_name].reshape(3,)).T))
        
        #Display frames from model
        frame_name = f'world/{seg_name}'
        frame_se3= data.oMf[model.getFrameId(seg_name)]
        place(viz, frame_name, frame_se3)
    
    if i == 0:
        input("Ready?")
    else:
        time.sleep(0.016)