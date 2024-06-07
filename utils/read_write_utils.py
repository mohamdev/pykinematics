import pandas as pd 
import numpy as np
from typing import Dict, Tuple, List
import matplotlib.pyplot as plt 

def read_lstm_data(file_name: str)->Tuple[Dict, Dict]:
    """_Creates two dictionnaries, one containing the 3D positions of all the markers output by the LSTM, another to map the number of the marks to the JC associated_
    Args:
        file_name (str): _The name of the file to process_

    Returns:
        Tuple[Dict, Dict]: _3D positions of all markers, mapping to JCP_
    """
    data = pd.read_csv(file_name).to_numpy()
    data = data[:,1:-1]
    time_vector = data[2:,0].astype(float)

    x = data[1,:] # Get label names
    labels = x[~pd.isnull(x)].tolist() #removes nan values

    x = data[0,:] # Get JCP names
    JCP = x[~pd.isnull(x)] #removes nan values
    JCP = JCP[1:].tolist() # removes time

    # Assert that the length of labels is three times the length of JCP
    assert len(labels) == 3 * len(JCP), "The length of labels must be three times the length of JCP."

    # Create the dictionary
    mapping = {}
    for i in range(len(JCP)):
        mapping[JCP[i]] = labels[i*3:(i*3)+3]

    labels = labels # add Time label

    data = data[2:,1:].astype(float)

    d=dict(zip(labels,data.T))
    
    # Create an empty dictionary for the combined arrays
    d3 = {'Time': time_vector}

    # Iterate over each key-value pair in d2
    for key, value in mapping.items():
        # Extract the arrays corresponding to the markers in value from d1
        arrays = [d[marker] for marker in value]
        # Combine the arrays into a single 3D array
        combined_array = np.array(arrays)
        # Transpose the array to have the shape (3, n), where n is the number of data points
        combined_array = np.transpose(combined_array)

        # Store the combined array in d3 with the key from d2
        d3[key] = combined_array

    return d3,mapping

def convert_to_list_of_dicts(dict_mks_data: Dict)-> List:
    """_This function converts a dictionnary of data outputed from read_lstm_data(), to a list of dictionnaries for each sample._

    Args:
        dict_mks_data (Dict): _ dictionnary of data outputed from read_lstm_data()_

    Returns:
        List: _ list of dictionnaries for each sample._
    """
    list_of_dicts = []
    for i in range(len(dict_mks_data['Time'])):
        curr_dict = {}
        for name in dict_mks_data:
            curr_dict[name] = dict_mks_data[name][i]
            # print(dict_mks_data[name][i])
        list_of_dicts.append(curr_dict)
    return list_of_dicts

def get_lstm_mks_names(file_name: str):
    """_Gets the lstm mks names_
    Args:
        file_name (str): _The name of the file to process_

    Returns:
        mk_names (list): _lstm mks names_
    """
    mk_data = pd.read_csv(file_name)
    row = mk_data.iloc[0]#on chope la deuxième ligne
    mk_names = row[2:].tolist() #on enlève les deux premieres valeurs
    mk_names = [mot for mot in mk_names if pd.notna(mot)] #On enlève les nan correspondant aux cases vides du fichier csv
    return mk_names

def read_mocap_data(file_path: str)->Dict:
    """_Gets the lstm mks names_
    Args:
        file_path (str): _The name of the file to process_

    Returns:
        mocap_mks_positions (list): _mocap mks positions and names dict_
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Extracting the anatomical landmarks names from the first line
    landmarks = lines[0].strip().split(',')
    
    # Extracting the 3D positions from the second line
    positions = list(map(float, lines[1].strip().split(',')))
    
    # Creating a dictionary to store the 3D positions of the landmarks
    mocap_mks_positions = {}
    for i, landmark in enumerate(landmarks):
        # Each landmark has 3 positions (x, y, z)
        mocap_mks_positions[landmark] = np.array(positions[3*i:3*i+3]).reshape(3,1)
    
    return mocap_mks_positions

def write_joint_angle_results(directory_name: str, q:np.ndarray):
    """_Write the joint angles obtained from the ik as asked by the challenge moderators_

    Args:
        directory_name (str): _Name of the directory to store the results_
        q (np.ndarray): _Joint angle results_
    """
    dofs_names = ['FF_TX','FF_TY','FF_TZ','FF_Rquat0','FF_Rquat1','FF_Rquat2','FF_Rquat3','L5S1_FE','RShoulder_FE','RShoulder_AA','RShoulder_RIE','RElbow_FE','RElbow_PS','RHip_FE','RHip_AA','RKnee_FE','RAnkle_FE']
    for ii in range(q.shape[1]):
        open(directory_name+'/'+dofs_names[ii]+'.csv', 'w').close() # clear the file 
        np.savetxt(directory_name+'/'+dofs_names[ii]+'.csv', q[:,ii])

def plot_joint_angle_results(directory_name:str):
    """_Plots the corresponding joint angles_

    Args:
        directory_name (str): _Directory name where the data to plot are stored_
    """
    dofs_names = ['FF_TX','FF_TY','FF_TZ','FF_Rquat0','FF_Rquat1','FF_Rquat2','FF_Rquat3','L5S1_FE','RShoulder_FE','RShoulder_AA','RShoulder_RIE','RElbow_FE','RElbow_PS','RHip_FE','RHip_AA','RKnee_FE','RAnkle_FE']
    for name in dofs_names: 
        q_i = np.loadtxt(directory_name+'/'+name+'.csv')
        plt.plot(q_i)
        plt.title(name)
        plt.show()

def read_joint_angles(directory_name:str)->np.ndarray:
    dofs_names = ['FF_TX','FF_TY','FF_TZ','FF_Rquat0','FF_Rquat1','FF_Rquat2','FF_Rquat3','L5S1_FE','RShoulder_FE','RShoulder_AA','RShoulder_RIE','RElbow_FE','RElbow_PS','RHip_FE','RHip_AA','RKnee_FE','RAnkle_FE']
    q=[]
    for name in dofs_names: 
        q_i = np.loadtxt(directory_name+'/'+name+'.csv')
        q.append(q_i)
    
    q=np.array(q)
    return q
