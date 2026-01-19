import argparse
from pathlib import Path
from utils import YAM_URDF_DIR

parser = argparse.ArgumentParser()
parser.add_argument("--urdf_path", type=Path, default=YAM_URDF_DIR)

args = parser.parse_args()

urdf_path: Path = args.urdf_path.resolve()
assert urdf_path.exists()

# pyroki code
import os
import yourdfpy
import pyroki as pk
from utils import solve_ik
import numpy as np
import jax.numpy as jnp
from jaxlie import SE3, SO3

np.set_printoptions(suppress=True, precision=4)

# urdf = yourdfpy.URDF.load(
#     "/home/zetans/Documents/i2rt/i2rt/robot_models/yam/yam.urdf",
#     mesh_dir="/home/zetans/Documents/i2rt/i2rt/robot_models/yam/assets/",
# )
urdf = yourdfpy.URDF.load(urdf_path, mesh_dir=urdf_path.parent)
robot = pk.Robot.from_urdf(urdf)


def ik(goal: SE3):
    solution = solve_ik(
        robot=robot,
        target_link_name="link_6",
        target_position=np.array(goal.wxyz_xyz[-3:]),
        target_wxyz=np.array(goal.wxyz_xyz[:4]),
    )
    return solution


# isaacsim code

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.api.objects import VisualCuboid
from pxr import UsdGeom
import spatial_utils as su

world = World()
world.scene.add_default_ground_plane()  # type: ignore

lead_cube_prim_path = "/World/lead_cube"
lead_cube = VisualCuboid(
    prim_path=lead_cube_prim_path,
    name="lead_cube",
    position=[0.0, 0.0, 2.0],
    size=0.08,
    color=np.array([1.0, 0.0, 1.0]),
)
world.scene.add(lead_cube)
lead_cube_xform = UsdGeom.Xformable(lead_cube.prim)

robot_prim_path = "/World/yam"
# add_reference_to_stage("/home/zetans/Documents/i2rt/i2rt/robot_models/yam/yam.usd", robot_prim_path)
add_reference_to_stage("yam_ros.usd", robot_prim_path)

articulation = SingleArticulation(prim_path=robot_prim_path)

world.reset()
articulation.initialize()
while True:
    world.step(render=True)
    lead_pose = np.array(lead_cube_xform.GetLocalTransformation()).T
    print(lead_pose)

    tcp_offset = su.ty(0.04) @ su.tz(-0.13)
    goal = SE3.from_matrix(lead_pose @ tcp_offset)

    solution = ik(goal)

    articulation.set_joint_positions(positions=solution)
    # articulation.set_joint_positions(positions=np.pad(np.array(solution), (0, 2)))

