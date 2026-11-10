"""Headless simulation: robot_state_publisher + joint_state_publisher + sim node.

No RViz and no GUI sliders -- useful for quick physics checks and CI.
The initial arm pose is the demo pose: the magnet hovers above the vessel.

    ros2 launch magnetic_robots_sim sim_headless.launch.py
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.substitutions import Command, FindExecutable
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    desc_share = get_package_share_directory('magnetic_robots_description')
    xacro_file = os.path.join(desc_share, 'urdf', 'magnetic_cell.xacro')
    params_file = os.path.join(
        get_package_share_directory('magnetic_robots_sim'),
        'config', 'sim_params.yaml')

    robot_description = ParameterValue(
        Command([FindExecutable(name='xacro'), ' ', xacro_file]),
        value_type=str)

    # initial arm pose: magnet hovers 11 cm above the vessel, pointing down
    initial_zeros = {
        'zeros.shoulder_joint': 0.2229,
        'zeros.upperArm_joint': -0.1852,
        'zeros.foreArm_joint': 1.6317,
        'zeros.wrist1_joint': 0.2369,
        'zeros.wrist2_joint': 1.5620,
        'zeros.wrist3_joint': 0.0,
    }

    return LaunchDescription([
        Node(package='robot_state_publisher', executable='robot_state_publisher',
             parameters=[{'robot_description': robot_description}],
             output='screen'),
        Node(package='joint_state_publisher', executable='joint_state_publisher',
             parameters=[initial_zeros], output='screen'),
        Node(package='magnetic_robots_sim', executable='continuum_sim_node',
             parameters=[params_file], output='screen'),
    ])
