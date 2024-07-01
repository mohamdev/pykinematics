import sys
import os
import pinocchio as pin 
import time
from pinocchio.visualize import GepettoVisualizer
import numpy as np
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from utils.read_write_utils import read_lstm_data,get_lstm_mks_names,read_mocap_data,convert_to_list_of_dicts
from utils.model_utils import *
from utils.viz_utils import place

subject = 'subject1'
type = 'train'
task= 'balancing'
#fichier_csv_mocap_mks = "./data/mocap_data/Lowerbody_Cal_.csv"  #just to check the markers
fichier_csv_mocap_mks = "./data/mocap_data/"+ subject +"/mks_data_"+ type +"_"+task +".csv"
mocap_mks_list= read_mocap_data(fichier_csv_mocap_mks)

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


#Create and place base frame
viz.viewer.gui.addXYZaxis(f'world/base_frame', [255, 0., 0, 1.], 0.015, 0.11)
place(viz, 'world/base_frame', pin.SE3(np.eye(3), np.matrix([0, 0, 0]).T))

#Create markers
for name in mocap_mks_list[0]:
    sphere_name = f'world/{name}'
    viz.viewer.gui.addSphere(sphere_name, 0.015, [0, 0., 255, 1.])

#Create sergments frames
viz.viewer.gui.addXYZaxis(f'world/pelvis', [255, 0., 0, 1.], 0.008, 0.08)
viz.viewer.gui.addXYZaxis(f'world/thighR', [255, 0., 0, 1.], 0.008, 0.08)
viz.viewer.gui.addXYZaxis(f'world/shankR', [255, 0., 0, 1.], 0.008, 0.08)
viz.viewer.gui.addXYZaxis(f'world/footR', [255, 0., 0, 1.], 0.008, 0.08)
viz.viewer.gui.addXYZaxis(f'world/thighL', [255, 0., 0, 1.], 0.008, 0.08)
viz.viewer.gui.addXYZaxis(f'world/shankL', [255, 0., 0, 1.], 0.008, 0.08)
viz.viewer.gui.addXYZaxis(f'world/footL', [255, 0., 0, 1.], 0.008, 0.08)


#Animation
for i in range (len(mocap_mks_list)):

    pelvis_pose = get_pelvis_pose(mocap_mks_list[i])

    thighR_pose = get_thighR_pose(mocap_mks_list[i])
    shankR_pose = get_shankR_pose(mocap_mks_list[i])
    footR_pose = get_footR_pose(mocap_mks_list[i])

    thighL_pose = get_thighL_pose(mocap_mks_list[i])
    shankL_pose = get_shankL_pose(mocap_mks_list[i])
    footL_pose = get_footL_pose(mocap_mks_list[i])
    place(viz, f'world/pelvis',  pin.SE3(pelvis_pose[:3, :3], np.matrix(pelvis_pose[:3,3].reshape(3,)).T))
    place(viz, f'world/thighR',  pin.SE3(thighR_pose[:3, :3], np.matrix(thighR_pose[:3,3].reshape(3,)).T))
    place(viz, f'world/shankR',  pin.SE3(shankR_pose[:3, :3], np.matrix(shankR_pose[:3,3].reshape(3,)).T))
    place(viz, f'world/footR',  pin.SE3(footR_pose[:3, :3], np.matrix(footR_pose[:3,3].reshape(3,)).T)) 
    place(viz, f'world/thighL',  pin.SE3(thighL_pose[:3, :3], np.matrix(thighL_pose[:3,3].reshape(3,)).T))
    place(viz, f'world/shankL',  pin.SE3(shankL_pose[:3, :3], np.matrix(shankL_pose[:3,3].reshape(3,)).T))
    place(viz, f'world/footL',  pin.SE3(footL_pose[:3, :3], np.matrix(footL_pose[:3,3].reshape(3,)).T))

    # Iterate over all points in mocap_mks_dict and add them to the viewer
    for key, value in mocap_mks_list[i].items():
        sphere_name = f'world/{key}'
        # Place the sphere at the corresponding 3D position
        place(viz, sphere_name, pin.SE3(np.eye(3), np.matrix(value.reshape(3,)).T))
    
    time.sleep(0.005)
