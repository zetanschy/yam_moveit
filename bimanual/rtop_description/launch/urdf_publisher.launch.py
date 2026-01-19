from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    """
    Launch file for Isaac Sim URDF importer.
    Publishes robot_description on /robot_description topic.
    """
    # Get package share directory
    rtop_description_share = get_package_share_directory('rtop_description')
    
    # URDF file path
    urdf_file = os.path.join(rtop_description_share, 'urdf', 'rtop.urdf.xacro')
    
    # Process xacro to get URDF
    robot_description_content = ParameterValue(
        Command(['xacro ', urdf_file]),
        value_type=str
    )
    
    # Robot state publisher node - publishes robot_description
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_description_content,
            'use_sim_time': LaunchConfiguration('use_sim_time'),
        }]
    )
    
    # Declare launch arguments
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation time (set to true for Isaac Sim)'
    )
    
    return LaunchDescription([
        use_sim_time_arg,
        robot_state_publisher_node,
    ])

