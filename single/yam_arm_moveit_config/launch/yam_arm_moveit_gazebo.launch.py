#!/usr/bin/env python3

"""
Combined Launch File for YAM Arm with Gazebo Simulation and MoveIt!
This launch file sets up the YAM arm with MoveIt! and Gazebo simulation together,
avoiding duplicate controller spawning.
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from moveit_configs_utils import MoveItConfigsBuilder
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    declared_arguments = []
    declared_arguments.append(
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="true",
            description="Use simulation (Gazebo) clock if true",
        )
    )

    use_sim_time = LaunchConfiguration("use_sim_time")

    yam_arm_gazebo_pkg = FindPackageShare("yam_arm_gazebo")

    # Include Gazebo launch file
    # Note: yam_ctrl.gazebo.launch.py already uses use_sim_time=True internally
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([yam_arm_gazebo_pkg, "launch", "yam_ctrl.gazebo.launch.py"])
        ),
    )

    # Build MoveIt configuration
    moveit_config = (
        MoveItConfigsBuilder("yam_calib", package_name="yam_arm_moveit_config")
        .robot_description(
            os.path.join(
                get_package_share_directory("yam_arm_moveit_config"),
                "config",
                "yam_calib.urdf.xacro",
            )
        )
        .robot_description_semantic(
            os.path.join(
                get_package_share_directory("yam_arm_moveit_config"),
                "config",
                "yam_calib.srdf",
            )
        )
        .trajectory_execution(
            os.path.join(
                get_package_share_directory("yam_arm_moveit_config"),
                "config",
                "moveit_controllers.yaml",
            )
        )
        .to_moveit_configs()
    )

    # Move Group Node
    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[moveit_config.to_dict(), {"use_sim_time": use_sim_time}],
        arguments=["--ros-args", "--log-level", "info"],
    )

    # RViz Node
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=[
            "-d",
            os.path.join(
                get_package_share_directory("yam_arm_moveit_config"),
                "config",
                "moveit.rviz",
            ),
        ],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            {"use_sim_time": use_sim_time},
        ],
    )

    return LaunchDescription(
        declared_arguments
        + [
            gazebo_launch,
            move_group_node,
            rviz_node,
            # Controllers are already spawned by the Gazebo launch file,
            # so we don't need to spawn them again here
        ]
    )

