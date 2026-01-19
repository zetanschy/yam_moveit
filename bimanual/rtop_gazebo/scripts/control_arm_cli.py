#!/usr/bin/env python3

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import time
import math


class ArmControllerCLI(Node):
    """A CLI tool to send joint trajectory goals to the RTOP bimanual robot."""

    def __init__(self, arm_side="left"):
        """
        Initialize the arm controller CLI.

        Args:
            arm_side: "left" or "right" to select which arm to control
        """
        super().__init__(f"arm_controller_cli_{arm_side}")
        self.arm_side = arm_side
        self.controller_name = f"{arm_side}_arm_controller"
        self.action_topic = f"/{self.controller_name}/follow_joint_trajectory"
        
        self.action_client = ActionClient(
            self,
            FollowJointTrajectory,
            self.action_topic,
        )

        self.joint_names = [
            f"{arm_side}_joint1",
            f"{arm_side}_joint2",
            f"{arm_side}_joint3",
            f"{arm_side}_joint4",
            f"{arm_side}_joint5",
            f"{arm_side}_joint6",
        ]

        self.predefined_poses = {
            "home": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "ready": [0.0, -0.785, 0.0, -1.57, 0.0, 0.785],  # Example "ready" pose
            "stretch_up": [
                0.0,
                math.pi / 2,
                math.pi,
                0.0,
                0.0,
                0.0,
            ],  # Pointing upwards
            "forward_low": [
                0.0,
                math.pi / 2,
                math.pi / 2,
                -math.pi / 2,
                0.0,
                0.0,
            ],  # Example forward and slightly down
        }
        self.get_logger().info(f"RTOP {arm_side.upper()} Arm Controller CLI started. Waiting for action server...")
        if not self.action_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error(
                f"Action server '{self.action_topic}' not available after 10 seconds. "
                f"Please ensure the {self.controller_name} is running."
            )
            raise RuntimeError(
                f"Action server '{self.action_topic}' not available. Initialization failed."
            )
        self.get_logger().info(f"Action server found for {self.controller_name}.")

    def send_goal(self, joint_positions, duration_sec=5.0):
        """
        Send a joint trajectory goal to the action server.

        Args:
        ----
        joint_positions : list of float
            A list of target joint positions.
        duration_sec : float
            The time in seconds to reach the target.

        """
        if len(joint_positions) != len(self.joint_names):
            self.get_logger().error(
                f"Incorrect number of joint positions. Expected {len(self.joint_names)}, "
                f"got {len(joint_positions)}"
            )
            return False

        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = self.joint_names

        point = JointTrajectoryPoint()
        point.positions = [float(pos) for pos in joint_positions]
        point.time_from_start = Duration(
            sec=int(duration_sec), nanosec=int((duration_sec % 1) * 1e9)
        )
        # Velocities and accelerations can be left empty for simple position control
        # point.velocities = [0.0] * len(self.joint_names)
        # point.accelerations = [0.0] * len(self.joint_names)

        goal_msg.trajectory.points.append(point)

        self.get_logger().info(f"Sending goal: {joint_positions} to be reached in {duration_sec}s")

        self.action_client.send_goal_async(goal_msg)

        # Optional: Add feedback and result handling if needed
        # future = self.action_client.send_goal_async(goal_msg)
        # rclpy.spin_until_future_complete(self, future)
        # goal_handle = future.result()

        # if not goal_handle.accepted:
        #     self.get_logger().info('Goal rejected :(')
        #     return False

        # self.get_logger().info('Goal accepted :)')
        # result_future = goal_handle.get_result_async()
        # rclpy.spin_until_future_complete(self, result_future)
        # result = result_future.result().result
        # if result.error_code == FollowJointTrajectory.Result.SUCCESSFUL:
        #     self.get_logger().info('Trajectory execution successful!')
        # else:
        #     self.get_logger().info(f'Trajectory execution failed '
        #                             f'with error code: {result.error_code}')
        # return True

        self.get_logger().info("Goal sent. The robot should be moving.")
        return True

    def run_cli(self):
        """Run the command-line interface loop."""
        while rclpy.ok():
            print(f"\nRTOP {self.arm_side.upper()} ARM Control CLI")
            print("=" * 40)
            print("Available predefined poses:")
            for i, name in enumerate(self.predefined_poses.keys()):
                print(f"  {i+1}. {name}")
            print("  c. Custom joint angles")
            print("  q. Quit")

            choice = input("Enter your choice: ").strip().lower()

            if choice == "q":
                break
            elif choice == "c":
                custom_angles_str = input(
                    f"Enter {len(self.joint_names)} joint angles (space-separated, in radians): "
                ).strip()
                try:
                    custom_angles = [float(x) for x in custom_angles_str.split()]
                    if len(custom_angles) == len(self.joint_names):
                        duration = float(
                            input("Enter duration to reach target (seconds, e.g., 5.0): ").strip()
                        )
                        self.send_goal(custom_angles, duration_sec=duration)
                    else:
                        self.get_logger().warning(
                            f"Please enter exactly {len(self.joint_names)} angles."
                        )
                except ValueError:
                    self.get_logger().warning("Invalid input. Please enter numbers only.")
            else:
                try:
                    pose_idx = int(choice) - 1
                    pose_names = list(self.predefined_poses.keys())
                    if 0 <= pose_idx < len(pose_names):
                        selected_pose_name = pose_names[pose_idx]
                        target_positions = self.predefined_poses[selected_pose_name]
                        duration = float(
                            input(
                                f"Enter duration to reach '{selected_pose_name}' "
                                f"(seconds, e.g., 5.0): "
                            ).strip()
                        )
                        self.send_goal(target_positions, duration_sec=duration)
                    else:
                        self.get_logger().warning("Invalid pose number.")
                except ValueError:
                    self.get_logger().warning(
                        "Invalid choice. Please enter a number or 'c' or 'q'."
                    )

            time.sleep(0.1)  # Small delay to allow ROS 2 to process


def main(args=None):
    import sys
    
    rclpy.init(args=args)
    
    # Parse command line argument for arm side
    arm_side = "left"  # default
    if len(sys.argv) > 1:
        arm_side = sys.argv[1].lower()
        if arm_side not in ["left", "right"]:
            print("Usage: control_arm_cli.py [left|right]")
            print(f"Invalid arm side: {arm_side}. Using default: left")
            arm_side = "left"
    
    arm_controller_cli_node = ArmControllerCLI(arm_side=arm_side)

    try:
        arm_controller_cli_node.run_cli()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        arm_controller_cli_node.get_logger().error(f"CLI loop error: {e}")
    finally:
        arm_controller_cli_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
