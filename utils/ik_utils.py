import pinocchio as pin 
import casadi 
import pinocchio.casadi as cpin 
from typing import Dict, List
import numpy as np 


class IK_Casadi:
    def __init__(self,model: pin.Model, dict_m: Dict, q0: np.ndarray):
        self._model = model
        self._dict_m = dict_m
        self._q0 = q0
        self._cmodel = cpin.Model(self._model)
        self._cdata = self._cmodel.createData()

        # Create a list of keys excluding the specified key
        self._keys_list = [key for key in self._dict_m[0].keys() if key !='Time']

        ### CASADI FRAMEWORK
        self._nq = self._cmodel.nq
        cq = casadi.SX.sym("q", self._nq, 1)

        cpin.framesForwardKinematics(self._cmodel, self._cdata, cq)

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
        d_list = []
        for i in range(len(self._dict_m)):
            meas_i = []
            for key in self._keys_list:
                meas_i.append(self._dict_m[i][key])
            d_i = dict(zip(self._keys_list,meas_i))
            d_list.append(d_i)
        return d_list

    def solve_ik_sample(self, ii: int, meas: Dict)->np.ndarray:

        # Casadi optimization class
        opti = casadi.Opti()

        # Variables MX type
        Q = opti.variable(self._nq, 1)   # state trajectory

        cost = 0
        for key in self._cfunction_dict.keys():
            cost+=casadi.sumsqr(meas[key]-self._cfunction_dict[key](Q))
        
        # Initial values for solver
        opti.set_initial(Q, self._q0)

        # TODO: Set the constraint for the joint limits
        for i in range(self._nq):
            opti.subject_to(opti.bounded(self._model.lowerPositionLimit[i],Q[i],self._model.upperPositionLimit[i]))
        
        opti.minimize(cost)

        # Options
        opts={}
        opts['ipopt']={'max_iter':100, 'linear_solver':'mumps'}

        opti.solver("ipopt", opts) # set numerical backend

        print('Solving for ' + str(ii) +'...')
        sol = opti.solve_limited()
        
        q_i = sol.value(Q)
        return q_i 

    def solve_ik(self)->List:
        q_list = []
        meas_list = self.create_meas_list()
        for i in range(len(meas_list)):
            self._q0 = self.solve_ik_sample(i, meas_list[i])
            q_list.append(self._q0)
        return q_list

        



        



