#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import time


class ArmTester(Node):
    def __init__(self, arm_side="left"):
        """
        Initialize the arm tester.

        Args:
            arm_side: "left" or "right" to select which arm to test
        """
        super().__init__(f"arm_tester_{arm_side}")
        self.arm_side = arm_side
        self.controller_name = f"{arm_side}_arm_controller"
        self.topic = f"/{self.controller_name}/joint_trajectory"
        
        self.publisher = self.create_publisher(
            JointTrajectory, self.topic, 10
        )

        # Joint names for RTOP 6-DOF arm
        self.joint_names = [
            f"{arm_side}_joint1",
            f"{arm_side}_joint2",
            f"{arm_side}_joint3",
            f"{arm_side}_joint4",
            f"{arm_side}_joint5",
            f"{arm_side}_joint6",
        ]

        self.get_logger().info(f"RTOP {arm_side.upper()} Arm Tester initialized")
        # Wait for the publisher to be ready
        time.sleep(1)

    def send_trajectory(self, positions, duration_sec=3.0):
        """Send a joint trajectory command."""
        msg = JointTrajectory()
        msg.joint_names = self.joint_names

        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start = Duration(
            sec=int(duration_sec), nanosec=int((duration_sec % 1) * 1e9)
        )

        msg.points = [point]

        self.publisher.publish(msg)
        self.get_logger().info(f"Sent trajectory: {positions}")

    def test_sequence(self):
        """Run a test sequence of movements."""
        self.get_logger().info(f"Starting {self.arm_side} arm test sequence...")

        # Home position
        self.get_logger().info("Moving to home position...")
        self.send_trajectory([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 3.0)
        time.sleep(4)

        # Test position 1
        self.get_logger().info("Moving to test position 1...")
        self.send_trajectory([0.5, -0.5, 0.3, -0.7, 0.2, 0.4], 3.0)
        time.sleep(4)

        # Test position 2
        self.get_logger().info("Moving to test position 2...")
        self.send_trajectory([-0.3, 0.7, -0.4, 0.5, -0.2, 0.6], 3.0)
        time.sleep(4)

        # Back to home
        self.get_logger().info("Returning to home...")
        self.send_trajectory([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 3.0)
        time.sleep(4)

        self.get_logger().info(f"{self.arm_side.upper()} arm test sequence complete!")


def main(args=None):
    import sys
    
    rclpy.init(args=args)
    
    # Parse command line argument for arm side
    arm_side = "left"  # default
    if len(sys.argv) > 1:
        arm_side = sys.argv[1].lower()
        if arm_side not in ["left", "right"]:
            print("Usage: test_arm_movement.py [left|right]")
            print(f"Invalid arm side: {arm_side}. Using default: left")
            arm_side = "left"

    tester = ArmTester(arm_side=arm_side)

    try:
        # Run the test sequence
        tester.test_sequence()

    except KeyboardInterrupt:
        pass
    finally:
        tester.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
