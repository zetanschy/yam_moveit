#!/usr/bin/env python3

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import sys
import termios
import tty
import select
import math
import time
import os


class ArmTeleop(Node):
    """A CLI tool for real-time teleoperation of the YAM arm robot using keyboard."""

    def __init__(self):
        """Initialize the arm teleoperation node."""
        super().__init__("yam_arm_teleop")
        self.controller_name = "arm_controller"
        self.action_topic = f"/{self.controller_name}/follow_joint_trajectory"
        
        self.action_client = ActionClient(
            self,
            FollowJointTrajectory,
            self.action_topic,
        )

        self.joint_names = [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
        ]

        # Current joint positions (in radians)
        self.current_positions = [0.0] * len(self.joint_names)
        
        # Step size for joint angle changes (in radians)
        self.step_size = math.pi / 36  # 5 degrees
        
        # Duration for trajectory execution (seconds)
        self.trajectory_duration = 0.5

        self.get_logger().info("YAM Arm Teleoperation started. Waiting for action server...")
        if not self.action_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error(
                f"Action server '{self.action_topic}' not available after 10 seconds. "
                f"Please ensure the {self.controller_name} is running."
            )
            raise RuntimeError(
                f"Action server '{self.action_topic}' not available. Initialization failed."
            )
        self.get_logger().info(f"Action server found for {self.controller_name}.")
        
        # ANSI escape codes for terminal control
        self.CLEAR_LINE = '\033[2K'  # Clear entire line
        self.CURSOR_UP = '\033[1A'   # Move cursor up one line
        self.CURSOR_HOME = '\033[H'  # Move cursor to home position
        self.CLEAR_SCREEN = '\033[2J'  # Clear entire screen
        self.RESET = '\033[0m'       # Reset formatting

    def get_key(self):
        """Get a single character from stdin without blocking."""
        if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
            return sys.stdin.read(1)
        return None
    
    def clear_screen(self):
        """Clear the terminal screen."""
        os.system('clear' if os.name != 'nt' else 'cls')

    def send_goal(self, joint_positions, duration_sec=None):
        """
        Send a joint trajectory goal to the action server.

        Args:
        ----
        joint_positions : list of float
            A list of target joint positions.
        duration_sec : float
            The time in seconds to reach the target. If None, uses self.trajectory_duration.

        """
        if duration_sec is None:
            duration_sec = self.trajectory_duration

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

        goal_msg.trajectory.points.append(point)
        self.action_client.send_goal_async(goal_msg)
        return True

    def print_status(self):
        """Print the current status display."""
        # Clear screen and print status
        self.clear_screen()
        print("=" * 70)
        print("YAM ARM TELEOPERATION")
        print("=" * 70)
        print("\nKey Mappings:")
        print("  Joint 1 (Base Rotation):     Q / A  - Increase / Decrease")
        print("  Joint 2 (Shoulder Pitch):    W / S  - Increase / Decrease")
        print("  Joint 3 (Elbow):              E / D  - Increase / Decrease")
        print("  Joint 4 (Wrist Pitch):        R / F  - Increase / Decrease")
        print("  Joint 5 (Wrist Roll):         T / G  - Increase / Decrease")
        print("  Joint 6 (End Effector):       Y / H  - Increase / Decrease")
        print("\nOther Controls:")
        print("    + / -  - Increase / Decrease step size")
        print("    0      - Reset all joints to zero")
        print("    ?      - Show this help message")
        print("    q      - Quit")
        print("=" * 70)
        print(f"\nCurrent Step Size: {math.degrees(self.step_size):.1f} degrees")
        print("\nCurrent Joint Positions (degrees):")
        for i, (name, pos) in enumerate(zip(self.joint_names, self.current_positions)):
            print(f"  {name}: {math.degrees(pos):7.2f}°")
        print("\n" + "=" * 70)
        print("Press keys to control the arm (or 'q' to quit)...")
        sys.stdout.flush()

    def reset_joints(self):
        """Reset all joints to zero position."""
        self.current_positions = [0.0] * len(self.joint_names)
        self.send_goal(self.current_positions, duration_sec=2.0)
        self.print_status()

    def update_joint(self, joint_index, direction):
        """
        Update a joint position.

        Args:
        ----
        joint_index : int
            Index of the joint to update (0-5)
        direction : int
            +1 to increase, -1 to decrease

        """
        if 0 <= joint_index < len(self.joint_names):
            self.current_positions[joint_index] += direction * self.step_size
            # Keep angles in reasonable range (optional: you can adjust limits)
            # self.current_positions[joint_index] = max(-math.pi, min(math.pi, self.current_positions[joint_index]))
            self.send_goal(self.current_positions)
            self.print_status()

    def run_teleop(self):
        """Run the teleoperation loop."""
        # Save terminal settings
        old_settings = termios.tcgetattr(sys.stdin)
        try:
            # Set terminal to cbreak mode (allows some terminal processing but still gets keys immediately)
            tty.setcbreak(sys.stdin.fileno())
            
            self.print_status()
            
            while rclpy.ok():
                # Process ROS callbacks
                rclpy.spin_once(self, timeout_sec=0.01)
                
                # Check for keyboard input
                key = self.get_key()
                
                if key is not None:
                    key_lower = key.lower()
                    
                    # Joint controls
                    if key_lower == 'q':
                        if key.isupper():
                            # Q - Increase joint1
                            self.update_joint(0, 1)
                        else:
                            # q - Quit
                            break
                    elif key_lower == 'a':
                        # A - Decrease joint1
                        self.update_joint(0, -1)
                    elif key_lower == 'w':
                        # W - Increase joint2
                        self.update_joint(1, 1)
                    elif key_lower == 's':
                        # S - Decrease joint2
                        self.update_joint(1, -1)
                    elif key_lower == 'e':
                        # E - Increase joint3
                        self.update_joint(2, 1)
                    elif key_lower == 'd':
                        # D - Decrease joint3
                        self.update_joint(2, -1)
                    elif key_lower == 'r':
                        # R - Increase joint4
                        self.update_joint(3, 1)
                    elif key_lower == 'f':
                        # F - Decrease joint4
                        self.update_joint(3, -1)
                    elif key_lower == 't':
                        # T - Increase joint5
                        self.update_joint(4, 1)
                    elif key_lower == 'g':
                        # G - Decrease joint5
                        self.update_joint(4, -1)
                    elif key_lower == 'y':
                        # Y - Increase joint6
                        self.update_joint(5, 1)
                    elif key_lower == 'h':
                        # H - Decrease joint6 (since Y is increase)
                        self.update_joint(5, -1)
                    elif key == '?':
                        # ? - Show help
                        self.print_status()
                    # Step size controls
                    elif key == '+' or key == '=':
                        self.step_size = min(math.pi, self.step_size + math.pi / 180)  # Increase by 1 degree
                        self.print_status()
                    elif key == '-':
                        self.step_size = max(math.pi / 180, self.step_size - math.pi / 180)  # Decrease by 1 degree
                        self.print_status()
                    # Reset
                    elif key == '0':
                        self.reset_joints()
                
                time.sleep(0.01)  # Small delay to prevent excessive CPU usage
                
        except KeyboardInterrupt:
            pass
        finally:
            # Restore terminal settings
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            self.clear_screen()
            print("Exiting teleoperation...")
            print("Terminal restored.")


def main(args=None):
    rclpy.init(args=args)
    
    arm_teleop_node = ArmTeleop()

    try:
        arm_teleop_node.run_teleop()
    except Exception as e:
        arm_teleop_node.get_logger().error(f"Teleoperation error: {e}")
    finally:
        arm_teleop_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

