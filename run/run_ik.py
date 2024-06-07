from utils.csv_utils import create_data
from utils.ik_utils import IK_Casadi
import pinocchio as pin 

file_name = 'data/jcp_coordinates_ncameras_augmented.csv'
d, mapping = create_data(file_name)

### CODE TO LOAD THE MODEL 

model = pin.Model()
q0 = pin.neutral(model)

### IK 

ik_problem = IK_Casadi(model, d, q0)

q = ik_problem.solve_ik()