#!/usr/bin/env python3

"""
Launch file for robot_state_publisher of YAM Arm.
Publishes the robot's state and TF2 transforms.

Usage:
    # Without camera (default)
    ros2 launch yam_arm_description robot_state_publisher.launch.py
    
    # With camera
    ros2 launch yam_arm_description robot_state_publisher.launch.py use_camera:=true
    
    # With simulation time
    ros2 launch yam_arm_description robot_state_publisher.launch.py use_sim_time:=true
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # Declare launch arguments
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation time if true'
    )
    
    use_camera_arg = DeclareLaunchArgument(
        'use_camera',
        default_value='false',
        description='Include camera in robot description if true (uses yam_cam.urdf.xacro)'
    )
    
    # Use OpaqueFunction to evaluate arguments at launch time
    return LaunchDescription([
        use_sim_time_arg,
        use_camera_arg,
        OpaqueFunction(function=launch_setup),
    ])


def launch_setup(context, *args, **kwargs):
    # Get package share directory
    yam_arm_description_share = get_package_share_directory('yam_arm_description')
    
    # Get launch argument values
    use_camera = context.launch_configurations.get('use_camera', 'false').lower() == 'true'
    use_sim_time = LaunchConfiguration('use_sim_time')
    
    # Determine which URDF file to use
    if use_camera:
        urdf_file = os.path.join(yam_arm_description_share, 'urdf', 'yam_cam.urdf.xacro')
    else:
        urdf_file = os.path.join(yam_arm_description_share, 'urdf', 'yam.urdf.xacro')
    
    # Process xacro to get URDF
    robot_description_content = ParameterValue(
        Command(['xacro ', urdf_file]),
        value_type=str
    )
    
    # Robot state publisher node
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_description_content,
            'use_sim_time': use_sim_time,
        }]
    )
    
    # Joint state publisher (for visualization/testing without hardware)
    # This publishes fake joint states - useful for visualization
    joint_state_publisher_node = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
        }]
    )
    
    return [
        robot_state_publisher_node,
        joint_state_publisher_node,
    ]
