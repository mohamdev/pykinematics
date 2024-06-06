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
    values = []

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

    labels = ['Time']+labels # add Time label

    data = data[2:,:]

    d=dict(zip(labels,data.T))

    return d,mapping




