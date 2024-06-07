from model_utils import *
from viz_utils import *

# MODEL GENERATION 

model = pin.Model()
geom_model = pin.GeometryModel()

# Pelvis with Freeflyer
IDX_JFF = model.addJoint(0, pin.JointModelFreeFlyer(), pin.SE3(np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]), np.matrix([0, 0, 0]).T), 'pelvis_freeflyer')
inertia_pelvis = pin.Inertia(13.3963, np.array([-0.000189, -0.027216, -0.000567]), make_inertia_matrix(0.134392, 0.0068219, 0.00172127, 0.134423, 0.000558921, 0.120176))
pelvis = pin.Frame('pelvis', IDX_JFF, 0, pin.SE3(np.eye(3), np.matrix([0, 0, 0]).T), pin.FrameType.OP_FRAME, inertia_pelvis)
IDX_PF = model.addFrame(pelvis, True)

# Left Hip Spherical ZYX
IDX_JLH = model.addJoint(IDX_JFF, pin.JointModelSphericalZYX(), pin.SE3(np.eye(3), np.matrix([0.053375, -0.0749, -0.079975]).T), 'left_hip_ZYX')
inertia_left_upperleg = pin.Inertia(13.3963, np.array([-0.000189, -0.027216, -0.000567]), make_inertia_matrix(0.134392, 0.0068219, 0.00172127, 0.134423, 0.000558921, 0.120176))
left_upperleg = pin.Frame('left_upperleg', IDX_JLH, IDX_PF, pin.SE3(np.eye(3), np.matrix([0, 0, 0]).T), pin.FrameType.OP_FRAME, inertia_left_upperleg)
IDX_LULF = model.addFrame(left_upperleg, False)

# Define other joints and frames for left leg
# Similar definitions for right leg, trunk, upper limbs, etc.

# ADD MESHES TO THE MODEL 
rtorso = R.from_rotvec(np.pi/2 * np.array([0, 1, 0]))
rupperarm = R.from_rotvec(np.pi/2 * np.array([0, 1, 0]))
rlowerarm = R.from_rotvec(np.pi/2 * np.array([0, 1, 0]))
rhand = R.from_rotvec(np.pi * np.array([0, 1, 0]))

mesh_loader = fcl.MeshLoader()

torso_visual = pin.GeometryObject('torso', IDX_PF, IDX_JFF, mesh_loader.load('/home/aladinedev2/Projects/swika/meshes/torso_mesh.STL'), pin.SE3(rtorso.as_matrix(), np.matrix([-0.15, -0.27, 0.13]).T), '/home/aladinedev2/Projects/swika/meshes/torso_mesh.STL', np.array([0.0065, 0.0065, 0.0065]), False, np.array([0, 1, 1, 1]))
# Add other visual geometries similarly

geom_model.addGeometryObject(torso_visual)
# Add other geometries to geom_model

# VISUALISATION 

visual_model = geom_model

viz = GepettoVisualizer(model, geom_model, visual_model)

try:
    viz.initViewer()
except ImportError as err:
    print("Error while initializing the viewer. It seems you should install gepetto-viewer")
    print(err)
    sys.exit(0)

try:
    viz.loadViewerModel("pinocchio")
except AttributeError as err:
    print("Error while loading the viewer model. It seems you should start gepetto-viewer")
    print(err)
    sys.exit(0)

# CHECK THAT THE MODEL IS PROPERLY CREATED

viz.viewer.gui.setColor(viz.getViewerNodeName(torso_visual, pin.GeometryType.VISUAL), [0, 1, 1, 0.5])
# Set color for other visual objects similarly

viz.display(pin.neutral(model))
print(pin.neutral(model))
pin.forwardKinematics(model, data, pin.neutral(model))
pin.updateFramePlacements(model, data)

Mposition_goal_temp = data.oMf[IDX_PF]
viz.viewer.gui.addXYZaxis('world/PELVISframe', [0, 0., 0, 1.], 0.01, 0.1)
place(viz, 'world/PELVISframe', Mposition_goal_temp)
