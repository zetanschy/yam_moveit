#!/usr/bin/env python3
"""
Gazebo Simulation Launch File for RTOP Robot
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
    TimerAction,
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
    package_name = "rtop_description"

    # --- Paths / shares ---
    pkg_share = FindPackageShare(package=package_name)
    ros_gz_sim_dir = get_package_share_directory("ros_gz_sim")

    rtop_description_share = get_package_share_directory("rtop_description")
    yam_arm_description_share = get_package_share_directory("yam_arm_description")

    # --- Gazebo resource paths for model:// resolution ---
    rtop_description_share_parent = os.path.dirname(rtop_description_share)  # .../share
    yam_arm_description_share_parent = os.path.dirname(yam_arm_description_share)  # .../share
    install_dir = os.path.dirname(rtop_description_share_parent)  # .../install

    gz_resource_path = f"{rtop_description_share_parent}:{yam_arm_description_share_parent}:{install_dir}"
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
            [FindPackageShare("rtop_moveit_config"), "config", "ros2_controllers.yaml"]
        ),
        description="Path to the ros2_controllers.yaml file",
    )


    # --- URDF / robot_description (xacro) ---
    # urdf_file_path = PathJoinSubstitution([FindPackageShare(package="rtop_moveit_config"), "urdf", "rtop_calib.urdf.xacro"])
    urdf_file_path = PathJoinSubstitution([pkg_share, "urdf", "rtop_gazebo.urdf.xacro"])

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
        name="spawn_rtop",
        output="screen",
        arguments=[
            "-topic",
            "/robot_description",
            "-name",
            "rtop",
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

    # --- Controller Manager ---
    # Start the ros2_control_node (controller_manager) with controllers config
    # robot_description is obtained from /robot_description topic via remapping
    controller_manager_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        name="controller_manager",
        output="screen",
        parameters=[LaunchConfiguration("ros2_controllers_path")],
        remappings=[
            ("/controller_manager/robot_description", "/robot_description"),
        ],
    )

    # --- Spawn Controllers ---
    # Spawn controllers directly (avoiding MoveIt config name mismatch)
    # Use TimerAction to wait for controller_manager to be ready
    spawn_joint_state_broadcaster = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_state_broadcaster",
            "--controller-manager",
            "/controller_manager",
        ],
        output="screen",
    )

    spawn_left_arm_controller = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "left_arm_controller",
            "--controller-manager",
            "/controller_manager",
        ],
        output="screen",
    )

    spawn_right_arm_controller = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "right_arm_controller",
            "--controller-manager",
            "/controller_manager",
        ],
        output="screen",
    )

    spawn_left_gripper_controller = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "left_gripper_controller",
            "--controller-manager",
            "/controller_manager",
        ],
        output="screen",
    )

    spawn_right_gripper_controller = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "right_gripper_controller",
            "--controller-manager",
            "/controller_manager",
        ],
        output="screen",
    )

    # Delay controller spawning until after controller_manager is ready
    load_controllers = TimerAction(
        period=3.0,  # Wait for controller_manager to initialize
        actions=[
            spawn_joint_state_broadcaster,
            spawn_left_arm_controller,
            spawn_right_arm_controller,
            spawn_left_gripper_controller,
            spawn_right_gripper_controller,
        ],
    )

    # --- Bridges ---
    # Note: Joint states are published by ros2_control/joint_state_broadcaster, so we don't bridge them
    # Only bridge clock if needed
    bridge_clock = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="bridge_clock",
        arguments=["/clock@rosgraph_msgs/msg/Clock@gz.msgs.Clock"],
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
            controller_manager_node,
            load_controllers,
            bridge_clock,
        ]
    )
