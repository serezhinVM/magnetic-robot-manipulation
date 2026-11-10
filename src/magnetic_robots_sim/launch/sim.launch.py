"""Full magnetic manipulation simulation.

Starts robot_state_publisher, joint_state_publisher_gui, RViz2 and the
closed-loop simulation node (continuum_sim_node):
move the arm with the joint sliders -> the magnetic field changes ->
the continuum rod bends and the point robots move inside the vessel.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    desc_share = get_package_share_directory('magnetic_robots_description')
    sim_share = get_package_share_directory('magnetic_robots_sim')
    params_file = os.path.join(sim_share, 'config', 'sim_params.yaml')

    # arm + magnet + scene + joint sliders + RViz
    display = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(desc_share, 'launch', 'display.launch.py')))

    # closed-loop physics node: TF -> field -> rod + point robots -> RViz
    sim_node = Node(
        package='magnetic_robots_sim',
        executable='continuum_sim_node',
        name='continuum_sim_node',
        output='screen',
        parameters=[params_file],
    )

    return LaunchDescription([display, sim_node])