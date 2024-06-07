import pinocchio as pin
import numpy as np

def place(viz, name, M):
    viz.viewer.gui.applyConfiguration(name, pin.SE3ToXYZQUAT(M).tolist())
    viz.viewer.gui.refresh()

def visualize_joint_angle_results(directory_name:str, viz, model):
    dofs_names = ['L5S1_FE','RShoulder_FE','RShoulder_AA','RShoulder_RIE','RElbow_FE','RElbow_PS','RHip_FE','RHip_AA','RKnee_FE','RAnkle_FE']
    q=[]
    for name in dofs_names: 
        q_i = np.loadtxt(directory_name+'/'+name+'.csv')
        q.append(q_i)
    
    q=np.array(q)

    for ii in range(q.shape[0]):
        q_ii=pin.neutral(model)
        q_ii[7:]=q[ii,:]
        viz.display(q_ii)
        input()


