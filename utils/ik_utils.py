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

        ### CASADI FRAMEWORK
        self._nq = self._cmodel.nq
        cq = casadi.SX.sym("q", self._nq, 1)

        cpin.framesForwardKinematics(self._cmodel, self._cdata, cq)

        self._cfunction_list = []
        for key in self._dict_m.keys():
            if key != 'Time':
                index_mk = self._cmodel.getFrameId(key)
                function_mk = casadi.Function(f'f_{key}',[cq],[self._cdata.oMf[index_mk].translation])
                self._cfunction_list.append(function_mk)

    def create_meas_list(self)-> List:
        meas_list = []
        for i in range(self._dict_m['Time'].shape[0]):
            meas_i = []
            for key in self._dict_m.keys():
                if key != 'Time':
                    meas_i.append(self._dict_m[key][i])
            meas_list.append(meas_i)
        return meas_list

    def solve_ik_sample(self, meas):
        for i in range(len(meas)):
            
        return q_i 

    def solve_ik(self)->np.ndarray:
        q_list = []
        meas_list = self.create_meas_list(self)
        for i in range(len(meas_list)):
            self._q0 = self.solve_ik_sample(self, meas_list[i])
            q_list.append(self._q0)
        q=np.array(q_list)
        return q 

        



        



