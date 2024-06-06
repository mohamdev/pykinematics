import eigenpy
import hppfcl as fcl
import pinocchio as pin
import sys
from pinocchio.visualize import GepettoVisualizer
import numpy as np
from scipy.spatial.transform import Rotation as R
import time


def place(viz, name, M):
    viz.viewer.gui.applyConfiguration(name, pin.SE3ToXYZQUAT(M).tolist())
    viz.viewer.gui.refresh()

