import pandas as pd 
import numpy as np
from typing import Dict, Tuple, List

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
