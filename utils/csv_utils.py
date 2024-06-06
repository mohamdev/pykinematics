import pandas as pd 
import numpy as np
from typing import Dict, Tuple 

def create_data(file_name: str)->Tuple[Dict, Dict]:
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

    data = data[2:,:].astype(float)

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






