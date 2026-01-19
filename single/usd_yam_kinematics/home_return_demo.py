import argparse
from pathlib import Path
import threading
import sys
import select
import time
from utils import YAM_URDF_DIR

parser = argparse.ArgumentParser()
parser.add_argument("--urdf_path", type=Path, default=YAM_URDF_DIR)

args = parser.parse_args()

urdf_path: Path = args.urdf_path.resolve()
assert urdf_path.exists()

# pyroki code

import yourdfpy
import pyroki as pk
from utils import solve_trajopt_home, solve_traj_from_cfgs, HOME_POSITION_CFG, TCP_OFFSET
import numpy as np
import jax.numpy as jnp
from jaxlie import SE3, SO3
import spatial_utils as su

np.set_printoptions(suppress=True, precision=4)
# urdf = yourdfpy.URDF.load(
#     "/home/zetans/Documents/i2rt/i2rt/robot_models/yam/yam.urdf",
#     mesh_dir="/home/zetans/Documents/i2rt/i2rt/robot_models/yam/assets/",
# )
urdf = yourdfpy.URDF.load(urdf_path, mesh_dir=urdf_path.parent)
robot = pk.Robot.from_urdf(urdf)
robot_coll = pk.collision.RobotCollision.from_urdf(urdf)

# Define the trajectory problem:
# - number of timesteps, timestep size
timesteps, dt = 25, 0.1
target_link_name = "link_6"

# Collision
wall_height = 0.3
wall_width = 0.1
wall_length = 0.4
wall_coll = pk.collision.Box.from_extent(
    extent=np.array([wall_length, wall_width, wall_height]),
    position=np.array([0.5, 0.0, wall_height / 2]),
)

table_height = 0.1
table_width = 2.0
table_length = 2.0
table_coll = pk.collision.Box.from_extent(
    extent=np.array([table_length, table_width, table_height]),
    position=np.array([1.0, 0.0, -table_height/2]),
)
world_coll = [table_coll]

# Shared state for interactive control
class ControlState:
    def __init__(self):
        self.computing_trajectory = False
        self.current_traj = None
        self.traj_to_target = None
        self.traj_to_home = None
        self.traj_index = 0
        self.state = "home"  # "home", "going_to_target", "computing_return", "returning_home"
        self.target_pos = None
        self.target_wxyz = None
        self.lock = threading.Lock()
control_state = ControlState()

# Isaacsim code
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.api.objects import VisualCuboid
from pxr import UsdGeom

world = World()
robot_prim_path = "/World/yam"

lead_cube_prim_path = "/World/lead_cube"
lead_cube = VisualCuboid(
    prim_path=lead_cube_prim_path,
    name="lead_cube",
    position=[0.5, 0.0, 0.5], #[0.25, 0.33, -0.25]
    size=0.025,
    color=np.array([1.0, 0.0, 1.0]),
)
world.scene.add(lead_cube)
lead_cube_xform = UsdGeom.Xformable(lead_cube.prim)

add_reference_to_stage("yam_collision_world.usd", robot_prim_path)

articulation = SingleArticulation(prim_path=robot_prim_path)

world.reset()
articulation.initialize()

# Start at home configuration
articulation.set_joint_positions(positions=HOME_POSITION_CFG)

print("\n" + "="*60)
print("Interactive Home Return Demo")
print("="*60)
print("Instructions:")
print("  - Press ENTER in the terminal to execute trajectory to lead_cube")
print("  - Once target is reached, return trajectory will be computed automatically")
print("  - Trajectory computation takes ~15 seconds (simulation continues)")
print("  - Press Ctrl+C to exit")
print("="*60 + "\n")


def compute_trajectory_to_target():
    """Compute trajectory to current lead_cube pose in a separate thread."""
    with control_state.lock:
        if control_state.computing_trajectory:
            return
        control_state.computing_trajectory = True
    
    try:
        # Get current lead_cube pose
        cube_pose = np.array(lead_cube_xform.GetLocalTransformation()).T
        tcp_offset = su.tx(TCP_OFFSET[0]) @ su.ty(TCP_OFFSET[1]) @ su.tz(TCP_OFFSET[2])
        lead_pose = cube_pose @ tcp_offset

        # Extract position and orientation from the transformation matrix
        target_pos = lead_pose[:3, 3]
        rotation_matrix = lead_pose[:3, :3]
        # Convert rotation matrix to quaternion (w, x, y, z)
        from scipy.spatial.transform import Rotation as R
        rot = R.from_matrix(rotation_matrix)
        target_wxyz = rot.as_quat()  # Returns (x, y, z, w)
        # Convert to (w, x, y, z) format
        target_wxyz = np.array([target_wxyz[3], target_wxyz[0], target_wxyz[1], target_wxyz[2]])
        
        print(f"\nComputing trajectory to target: pos={target_pos}, wxyz={target_wxyz}")
        print("This will take ~15 seconds. Simulation continues running...")
        
        t0 = time.time()
        # Compute trajectory from home to target only
        traj_to_target = solve_trajopt_home(
            robot,
            robot_coll,
            world_coll,
            target_link_name,
            target_pos,
            target_wxyz,
            is_home_end=False,  # home -> target
            timesteps=timesteps,
            dt=dt,
        )
        traj_to_target = np.array(traj_to_target)
        print(f"Trajectory to target computed in {time.time() - t0:.2f} seconds")
        
        with control_state.lock:
            control_state.traj_to_target = traj_to_target
            control_state.current_traj = traj_to_target
            control_state.traj_index = 0
            control_state.state = "going_to_target"
            control_state.target_pos = target_pos
            control_state.target_wxyz = target_wxyz
            control_state.computing_trajectory = False
        
        print("\nTrajectory ready! Executing...\n")
    except Exception as e:
        print(f"Error computing trajectory: {e}")
        import traceback
        traceback.print_exc()
        with control_state.lock:
            control_state.computing_trajectory = False


def compute_trajectory_to_home():
    """Compute trajectory from current target back to home in a separate thread."""
    with control_state.lock:
        if control_state.computing_trajectory:
            return
        if control_state.target_pos is None or control_state.target_wxyz is None:
            print("Error: Target pose not available for return trajectory")
            return
        control_state.computing_trajectory = True
        target_pos = control_state.target_pos
        target_wxyz = control_state.target_wxyz
    
    try:
        print("\nComputing trajectory back to home...")
        print("This will take ~15 seconds. Simulation continues running...")
        
        t0 = time.time()
        # Compute trajectory from target back to home
        # traj_to_home = solve_trajopt_home(
        #     robot,
        #     robot_coll,
        #     world_coll,
        #     target_link_name,
        #     target_pos,
        #     target_wxyz,
        #     is_home_end=True,  # target -> home
        #     timesteps=timesteps,
        #     dt=dt,
        # )
        start_cfg = np.array(articulation.get_joint_positions())
        traj_to_home = solve_traj_from_cfgs(robot, robot_coll, world_coll, start_cfg, HOME_POSITION_CFG, timesteps, dt)
        print(f"Trajectory to home computed in {time.time() - t0:.2f} seconds")
        
        with control_state.lock:
            control_state.traj_to_home = traj_to_home
            control_state.current_traj = traj_to_home
            control_state.traj_index = 0
            control_state.state = "returning_home"
            control_state.computing_trajectory = False
        
        print("\nReturn trajectory ready! Executing...\n")
    except Exception as e:
        print(f"Error computing return trajectory: {e}")
        import traceback
        traceback.print_exc()
        with control_state.lock:
            control_state.computing_trajectory = False
            control_state.state = "home"


def check_input():
    """Check for user input in a non-blocking way."""
    try:
        if sys.stdin.isatty():
            if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
                line = sys.stdin.readline()
                if line.strip() == "" or line.strip().lower() == "go":
                    return True
    except (OSError, ValueError):
        pass
    return False


# Main simulation loop
while True:
    world.step(render=True)
    
    # Check for user input
    if check_input():
        with control_state.lock:
            if not control_state.computing_trajectory and control_state.state == "home":
                # Start trajectory computation in a separate thread
                thread = threading.Thread(target=compute_trajectory_to_target, daemon=True)
                thread.start()
    
    # Execute trajectory if available
    with control_state.lock:
        if control_state.state == "computing_return":
            # Keep robot at target position while computing return trajectory
            if control_state.traj_to_target is not None and len(control_state.traj_to_target) > 0:
                # Stay at the last position of the trajectory to target
                articulation.set_joint_positions(positions=np.array(control_state.traj_to_target[-1]))
        elif control_state.current_traj is not None:
            if control_state.traj_index < len(control_state.current_traj):
                # Execute current step
                articulation.set_joint_positions(positions=np.array(control_state.current_traj[control_state.traj_index]))
                control_state.traj_index += 1
            else:
                # Trajectory completed, switch to next phase
                if control_state.state == "going_to_target":
                    print("\nReached target! Computing return trajectory...\n")
                    # Start computing return trajectory
                    control_state.current_traj = None  # Stop executing trajectory
                    control_state.state = "computing_return"
                    # Start trajectory computation in a separate thread
                    thread = threading.Thread(target=compute_trajectory_to_home, daemon=True)
                    thread.start()
                elif control_state.state == "returning_home":
                    print("\nReturned to home. Ready for next command.\n")
                    control_state.current_traj = None
                    control_state.traj_to_target = None
                    control_state.traj_to_home = None
                    control_state.traj_index = 0
                    control_state.target_pos = None
                    control_state.target_wxyz = None
                    control_state.state = "home"
        else:
            # No trajectory, stay at home
            articulation.set_joint_positions(positions=HOME_POSITION_CFG)

