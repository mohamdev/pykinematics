import pinocchio as pin 
import casadi 
import pinocchio.casadi as cpin 
from typing import Dict, List
import numpy as np 


class IK_Casadi:
    """ Class to generate an IK problem using pinocchio casadi 
    """
    def __init__(self,model: pin.Model, dict_m: Dict, q0: np.ndarray):
        """_Init of the class _

        Args:
            model (pin.Model): _Pinocchio biomechanical model_
            dict_m (Dict): _a dictionnary containing the measures of the landmarks_
            q0 (np.ndarray): _initial configuration_
        """
        self._model = model
        self._dict_m = dict_m
        self._q0 = q0
        self._cmodel = cpin.Model(self._model)
        self._cdata = self._cmodel.createData()

        # Create a list of keys excluding the specified key
        self._keys_list = [key for key in self._dict_m[0].keys() if key !='Time']

        ### CASADI FRAMEWORK
        self._nq = self._cmodel.nq
        self._nv = self._cmodel.nv

        cq = casadi.SX.sym("q",self._nq,1)
        cdq = casadi.SX.sym("dq",self._nv,1)

        cpin.framesForwardKinematics(self._cmodel, self._cdata, cq)
        self._integrate = casadi.Function('integrate',[ cq,cdq ],[cpin.integrate(self._cmodel,cq,cdq) ])

        cfunction_list = []
        self._new_key_list = [] # Only take the frames that are in the model 

        for key in self._keys_list:
            index_mk = self._cmodel.getFrameId(key)
            if index_mk < len(self._model.frames.tolist()): # Check that the frame is in the model
                new_key = key.replace('.','')
                self._new_key_list.append(key)
                function_mk = casadi.Function(f'f_{new_key}',[cq],[self._cdata.oMf[index_mk].translation])
                cfunction_list.append(function_mk)

        self._cfunction_dict=dict(zip(self._new_key_list,cfunction_list))
        
    def create_meas_list(self)-> List[Dict]:
        """_Create a list with each element is a dictionnary of measurements referencing a given sample_

        Returns:
            List[Dict]: _List of dictionnary of measures_
        """
        d_list = []
        for i in range(len(self._dict_m)):
            meas_i = []
            for key in self._keys_list:
                meas_i.append(self._dict_m[i][key])
            d_i = dict(zip(self._keys_list,meas_i))
            d_list.append(d_i)
        return d_list

    def solve_ik_sample(self, ii: int, meas: Dict)->np.ndarray:
        """_Solve the ik optimisation problem : q* = argmin(||P_m - P_e||^2 + lambda|q_init - q|) st to q_min <= q <= q_max for a given sample _

        Args:
            ii (int): _number of sample_
            meas (Dict): _Dictionnary of landmark measurements_

        Returns:
            np.ndarray: _q_i joint angle at the i-th sample_
        """

        # Casadi optimization class
        opti = casadi.Opti()

        # Variables MX type
        DQ = opti.variable(self._nv)
        Q = self._integrate(self._q0,DQ)

        cost = 0
        for key in self._cfunction_dict.keys():
            cost+=100*casadi.sumsqr(meas[key]-self._cfunction_dict[key](Q)) + 0.01*casadi.sum1(self._q0-Q) #LASSO 

        # Set the constraint for the joint limits
        for i in range(7,self._nq):
            opti.subject_to(opti.bounded(self._model.lowerPositionLimit[i],Q[i],self._model.upperPositionLimit[i]))
        
        opti.minimize(cost)

        # Set Ipopt options to suppress output
        opts = {
            "ipopt.print_level": 0,
            "ipopt.sb": "yes",
            "ipopt.max_iter": 100,
            "ipopt.linear_solver": "mumps"
        }

        opti.solver("ipopt", opts)

        print('Solving for ' + str(ii) +'...')
        sol = opti.solve_limited()
        
        q_i = sol.value(Q)
        return q_i 

    def solve_ik(self)->List:
        """_Returns a list of joint angle configuration over the whole trajectory _

        Returns:
            List: _List of q_i_
        """
        q_list = []
        meas_list = self.create_meas_list()
        for i in range(len(meas_list)):
            self._q0 = self.solve_ik_sample(i, meas_list[i])
            q_list.append(self._q0)
        return q_list

        



        



