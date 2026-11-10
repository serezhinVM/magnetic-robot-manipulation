"""Visualize the magnetic manipulation cell: AUBO i5 arm + magnet + scene.

Starts robot_state_publisher, joint_state_publisher_gui and RViz2.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.substitutions import Command, FindExecutable
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg_share = get_package_share_directory('magnetic_robots_description')
    xacro_file = os.path.join(pkg_share, 'urdf', 'magnetic_cell.xacro')
    rviz_file = os.path.join(pkg_share, 'rviz', 'magnetic_cell.rviz')

    robot_description = ParameterValue(
        Command([FindExecutable(name='xacro'), ' ', xacro_file]),
        value_type=str,
    )

    # initial arm pose: magnet hovers 11 cm above the vessel, pointing down
    # -> the rod bends and the point robots move right after launch
    initial_zeros = {
        'zeros.shoulder_joint': 0.2229,
        'zeros.upperArm_joint': -0.1852,
        'zeros.foreArm_joint': 1.6317,
        'zeros.wrist1_joint': 0.2369,
        'zeros.wrist2_joint': 1.5620,
        'zeros.wrist3_joint': 0.0,
    }

    return LaunchDescription([
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters=[{'robot_description': robot_description}],
            output='screen',
        ),
        Node(
            package='joint_state_publisher_gui',
            executable='joint_state_publisher_gui',
            parameters=[initial_zeros],
            output='screen',
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            arguments=['-d', rviz_file],
            output='screen',
        ),
    ])
