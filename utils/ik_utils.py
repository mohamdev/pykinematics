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
        self._dt = np.mean(np.diff(self._dict_m['Time']))

        # Create a list of keys excluding the specified key
        self._keys_list = [key for key in self._dict_m.keys() if key !='Time']

        ### CASADI FRAMEWORK
        self._nq = self._cmodel.nq
        cq = casadi.SX.sym("q", self._nq, 1)

        cpin.framesForwardKinematics(self._cmodel, self._cdata, cq)

        cfunction_list = []
        for key in self._keys_list:
            index_mk = self._cmodel.getFrameId(key)
            function_mk = casadi.Function(f'f_{key}',[cq],[self._cdata.oMf[index_mk].translation])
            cfunction_list.append(function_mk)
        self._cfunction_dict=dict(zip(self._keys_list,cfunction_list))
        
    def create_meas_list(self)-> List[Dict]:
        d_list = []
        for i in range(self._dict_m['Time'].shape[0]):
            meas_i = []
            for key in self._keys_list:
                meas_i.append(self._dict_m[key][i])
            d_i = dict(zip(self._keys_list,meas_i))
            d_list.append(d_i)
        return d_list

    def solve_ik_sample(self, i: int, meas: Dict)->np.ndarray:

        # Casadi optimization class
        opti = casadi.Opti()

        # Variables MX type
        Q = opti.variable(self._nq, 1)   # state trajectory

        cost = 0
        for key, array in meas.items():
            cost+=casadi.sumsqr(array-self._cfunction_dict[key](Q))
        
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

        print(f'Solving for {i}...')
        sol = opti.solve_limited()
        
        q_i = sol.value(Q)
        return q_i 

    def solve_ik(self)->np.ndarray:
        q_list = []
        meas_list = self.create_meas_list(self)
        for i in range(len(meas_list)):
            self._q0 = self.solve_ik_sample(self, i, meas_list[i])
            q_list.append(self._q0)
        q=np.array(q_list)
        return q 

        



        



