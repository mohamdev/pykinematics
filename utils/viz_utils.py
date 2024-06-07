import pinocchio as pin
import numpy as np
import time 

def place(viz, name, M):
    viz.viewer.gui.applyConfiguration(name, pin.SE3ToXYZQUAT(M).tolist())
    viz.viewer.gui.refresh()

def visualize_joint_angle_results(directory_name:str, viz, model):
    dofs_names = ['FF_TX','FF_TY','FF_TZ','FF_Rquat0','FF_Rquat1','FF_Rquat2','FF_Rquat3','L5S1_FE','RShoulder_FE','RShoulder_AA','RShoulder_RIE','RElbow_FE','RElbow_PS','RHip_FE','RHip_AA','RKnee_FE','RAnkle_FE']
    q=[]
    for name in dofs_names: 
        q_i = np.loadtxt(directory_name+'/'+name+'.csv')
        q.append(q_i)
    
    q=np.array(q)

    input('Are you ready to record the results ?')
    for ii in range(q.shape[1]):
        q_ii=q[:,ii]
        viz.display(q_ii)
        time.sleep(0.016)
        


