import argparse
from pathlib import Path
from utils import solve_trajopt, YAM_URDF_DIR

parser = argparse.ArgumentParser()
parser.add_argument("--urdf_path", type=Path, default=YAM_URDF_DIR)

args = parser.parse_args()

urdf_path: Path = args.urdf_path.resolve()
assert urdf_path.exists()

# pyroki code

import yourdfpy
import pyroki as pk
import numpy as np
import jax.numpy as jnp
from jaxlie import SE3, SO3

np.set_printoptions(suppress=True, precision=4)

urdf = yourdfpy.URDF.load(urdf_path, mesh_dir=urdf_path.parent)
robot = pk.Robot.from_urdf(urdf)
robot_coll = pk.collision.RobotCollision.from_urdf(urdf)

# Define the trajectory problem:
# - number of timesteps, timestep size
timesteps, dt = 50, 0.1
# - the start and end poses.
start_pos, end_pos = np.array([0.5, -0.3, 0.2]), np.array([0.5, 0.3, 0.2])
target_link_name = "link_6"


wall_height = 0.0
wall_width = 0.0
wall_length = 0.0
wall_coll = pk.collision.Box.from_extent(
    extent=np.array([wall_length, wall_width, wall_height]),
    position=np.array([0.5, 0.0, wall_height / 2]),
)

world_coll = [wall_coll]

traj = solve_trajopt(
    robot,
    robot_coll,
    world_coll,
    target_link_name,
    start_pos,
    np.array([0, 0, 0, 1]),
    end_pos,
    np.array([0, 0, 1, 0]),
    timesteps,
    dt,
)

traj = np.array(traj)
print(traj)
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

robot_prim_path = "/World/robot"
add_reference_to_stage("/home/zetans/mjlab/src/mjlab/asset_zoo/robots/i2rt_yam/yam.usd", robot_prim_path)

articulation = SingleArticulation(prim_path=robot_prim_path)

world.reset()
articulation.initialize()

i=0
count=1
while True:
    world.step(render=True)

    articulation.set_joint_positions(positions=np.array(traj[i]))

    i += count
    if i == len(traj) or i == -1:
        count *= -1
        i += count

