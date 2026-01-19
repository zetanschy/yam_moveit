#!/usr/bin/env python3
"""
Gazebo Simulation Launch File for yam_arm Robot
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    Command,
    LaunchConfiguration,
    PathJoinSubstitution,
    FindExecutable,
)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    package_name = "yam_arm_description"

    # --- Paths / shares ---
    pkg_share = FindPackageShare(package=package_name)
    ros_gz_sim_dir = get_package_share_directory("ros_gz_sim")

    yam_arm_description_share = get_package_share_directory("yam_arm_description")
    yam_arm_description_share = get_package_share_directory("yam_arm_description")

    # --- Gazebo resource paths for model:// resolution ---
    yam_arm_description_share_parent = os.path.dirname(yam_arm_description_share)  # .../share
    yam_arm_description_share_parent = os.path.dirname(yam_arm_description_share)  # .../share
    install_dir = os.path.dirname(yam_arm_description_share_parent)  # .../install

    gz_resource_path = f"{yam_arm_description_share_parent}:{yam_arm_description_share_parent}:{install_dir}"
    if "GZ_SIM_RESOURCE_PATH" in os.environ and os.environ["GZ_SIM_RESOURCE_PATH"]:
        gz_resource_path = f"{os.environ['GZ_SIM_RESOURCE_PATH']}:{gz_resource_path}"

    set_gz_resource_path = SetEnvironmentVariable("GZ_SIM_RESOURCE_PATH", gz_resource_path)
    set_ign_resource_path = SetEnvironmentVariable("IGN_GAZEBO_RESOURCE_PATH", gz_resource_path)

    # --- Launch args ---
    world_arg = DeclareLaunchArgument(
        "world",
        default_value="",
        description="Full path to world SDF file (empty for default empty world)",
    )
    robot_x_arg = DeclareLaunchArgument("robot_x", default_value="0.0", description="Robot spawn X position")
    robot_y_arg = DeclareLaunchArgument("robot_y", default_value="0.0", description="Robot spawn Y position")
    robot_z_arg = DeclareLaunchArgument("robot_z", default_value="0.0", description="Robot spawn Z position")
    robot_yaw_arg = DeclareLaunchArgument("robot_yaw", default_value="0.0", description="Robot spawn yaw angle")

    controllers_arg = DeclareLaunchArgument(
        "ros2_controllers_path",
        default_value=PathJoinSubstitution(
            [FindPackageShare("yam_arm_moveit_config"), "config", "ros2_controllers.yaml"]
        ),
        description="Path to the ros2_controllers.yaml file",
    )


    # --- URDF / robot_description (xacro) ---
    # urdf_file_path = PathJoinSubstitution([FindPackageShare(package="yam_arm_moveit_config"), "urdf", "yam_arm_calib.urdf.xacro"])
    urdf_file_path = PathJoinSubstitution([pkg_share, "urdf", "yam.urdf"])

    # If your xacro accepts ros2_controllers_path:=..., pass it (safe even if unused in xacro)
    robot_description_command = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            urdf_file_path,
            " ",
            "ros2_controllers_path:=",
            LaunchConfiguration("ros2_controllers_path"),
        ]
    )

    robot_description_content = ParameterValue(robot_description_command, value_type=str)

    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[{"use_sim_time": True, "robot_description": robot_description_content}],
    )

    # --- Gazebo Sim ---
    gazebo_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(ros_gz_sim_dir, "launch", "gz_sim.launch.py")),
        launch_arguments={
            "gz_args": ["-r -v 4 ", LaunchConfiguration("world")],
            "on_exit_shutdown": "true",
        }.items(),
    )


    # --- Spawn robot in Gazebo ---
    # Use -topic with leading slash (as per bcr_arm example)
    spawn_robot_node = Node(
        package="ros_gz_sim",
        executable="create",
        name="spawn_yam_arm",
        output="screen",
        arguments=[
            "-topic",
            "/robot_description",
            "-name",
            "yam_arm",
            "-allow_renaming",
            "true",
            "-x",
            LaunchConfiguration("robot_x"),
            "-y",
            LaunchConfiguration("robot_y"),
            "-z",
            LaunchConfiguration("robot_z"),
            "-Y",
            LaunchConfiguration("robot_yaw"),
        ],
    )

    # --- Controllers ---
    # Controllers are automatically loaded by the gz_ros2_control plugin
    # from the ros2_controllers.yaml file specified in yam_arm_gazebo.urdf.xacro
    # No manual spawning needed!

    # --- Bridges (keep as-is if you still need them) ---
    bridge_joint_states = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="bridge_joint_states",
        arguments=[
            "/joint_states@sensor_msgs/msg/JointState[ignition.msgs.Model"
        ],
        output="screen",
    )

    bridge_clock = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="bridge_clock",
        arguments=["/clock@rosgraph_msgs/msg/Clock@ignition.msgs.Clock"],
        output="screen",
    )

    return LaunchDescription(
        [
            world_arg,
            robot_x_arg,
            robot_y_arg,
            robot_z_arg,
            robot_yaw_arg,
            controllers_arg,
            set_gz_resource_path,
            set_ign_resource_path,
            gazebo_sim,
            robot_state_publisher_node,
            spawn_robot_node,
            bridge_joint_states,
            bridge_clock,
        ]
    )
