#!/usr/bin/env python3

"""
Combined Launch File for YAM Arm with Gazebo Simulation and MoveIt!
This launch file sets up the YAM arm with MoveIt! and Gazebo simulation together,
avoiding duplicate controller spawning.
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterValue
from moveit_configs_utils import MoveItConfigsBuilder
from moveit_configs_utils.launch_utils import DeclareBooleanLaunchArg
from moveit_configs_utils.launches import generate_moveit_rviz_launch
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    yam_arm_gazebo_pkg = FindPackageShare("yam_arm_gazebo")
    yam_arm_description_pkg = get_package_share_directory("yam_arm_description")
    yam_arm_moveit_config_pkg = get_package_share_directory("yam_arm_moveit_config")
    
    ld = LaunchDescription()
    
    # Include Gazebo launch file FIRST (starts robot_state_publisher, controllers, etc.)
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([yam_arm_gazebo_pkg, "launch", "yam_ctrl.gazebo.launch.py"])
        ),
    )
    ld.add_action(gazebo_launch)
    
    # Build MoveIt config using the SAME URDF as Gazebo (yam_arm_gazebo.urdf.xacro)
    moveit_config = (
        MoveItConfigsBuilder("yam_calib", package_name="yam_arm_moveit_config")
        .robot_description(
            file_path=os.path.join(yam_arm_description_pkg, "urdf", "yam_arm_gazebo.urdf.xacro"),
        )
        .robot_description_semantic(
            file_path=os.path.join(yam_arm_moveit_config_pkg, "config", "yam_calib.srdf")
        )
        .robot_description_kinematics(
            file_path=os.path.join(yam_arm_moveit_config_pkg, "config", "kinematics.yaml")
        )
        .trajectory_execution(
            file_path=os.path.join(yam_arm_moveit_config_pkg, "config", "moveit_controllers.yaml")
        )
        .joint_limits(
            file_path=os.path.join(yam_arm_moveit_config_pkg, "config", "joint_limits.yaml")
        )
        .planning_pipelines(pipelines=["ompl"])
        .planning_scene_monitor(
            publish_robot_description=True,
            publish_robot_description_semantic=True,
        )
        .to_moveit_configs()
    )
    
    # Move Group Node with explicit use_sim_time (like xarm does)
    ld.add_action(DeclareBooleanLaunchArg("debug", default_value=False))
    ld.add_action(DeclareBooleanLaunchArg("allow_trajectory_execution", default_value=True))
    ld.add_action(DeclareBooleanLaunchArg("publish_monitored_planning_scene", default_value=True))
    
    should_publish = LaunchConfiguration("publish_monitored_planning_scene")
    move_group_configuration = {
        "publish_robot_description_semantic": True,
        "allow_trajectory_execution": LaunchConfiguration("allow_trajectory_execution"),
        "capabilities": ParameterValue(
            LaunchConfiguration("capabilities"), value_type=str
        ),
        "disable_capabilities": ParameterValue(
            LaunchConfiguration("disable_capabilities"), value_type=str
        ),
        "publish_planning_scene": should_publish,
        "publish_geometry_updates": should_publish,
        "publish_state_updates": should_publish,
        "publish_transforms_updates": should_publish,
        "monitor_dynamics": False,
    }
    
    ld.add_action(DeclareLaunchArgument(
        "capabilities",
        default_value=moveit_config.move_group_capabilities["capabilities"],
    ))
    ld.add_action(DeclareLaunchArgument(
        "disable_capabilities",
        default_value=moveit_config.move_group_capabilities["disable_capabilities"],
    ))
    
    move_group_params = [
        moveit_config.to_dict(),
        move_group_configuration,
        {"use_sim_time": True},  # Explicitly set use_sim_time like xarm
    ]
    
    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=move_group_params,
        additional_env={"DISPLAY": os.environ.get("DISPLAY", "")},
    )
    # Delay move_group to ensure controllers are ready
    move_group_timer = TimerAction(
        period=8.0,
        actions=[move_group_node],
    )
    ld.add_action(move_group_timer)
    
    # RViz - Build a SEPARATE moveit_config for RViz (like the working two-file approach)
    rviz_moveit_config = (
        MoveItConfigsBuilder("yam_calib", package_name="yam_arm_moveit_config")
        .robot_description(
            file_path=os.path.join(yam_arm_description_pkg, "urdf", "yam_arm_gazebo.urdf.xacro"),
        )
        .robot_description_semantic(
            file_path=os.path.join(yam_arm_moveit_config_pkg, "config", "yam_calib.srdf")
        )
        .robot_description_kinematics(
            file_path=os.path.join(yam_arm_moveit_config_pkg, "config", "kinematics.yaml")
        )
        .planning_pipelines(pipelines=["ompl"])
        .to_moveit_configs()
    )
    
    # Use the exact same function as the working approach
    rviz_launch = generate_moveit_rviz_launch(rviz_moveit_config)
    for action in rviz_launch.entities:
        ld.add_action(action)
    
    return ld
