from utils.csv_utils import create_data

file_name = 'data/jcp_coordinates_ncameras_augmented.csv'

d, mapping = create_data(file_name)

print(d,mapping)