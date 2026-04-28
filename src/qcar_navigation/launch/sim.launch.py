from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='qcar_navigation',
            executable='undistorted_node_sim',
            name='undistort_node',
            output='screen',
        ),
        Node(
            package='qcar_navigation',
            executable='lane_detection_sw_node_sim',
            name='lane_detection_node',
            output='screen',
        ),
        Node(
            package='qcar_navigation',
            executable='lane_visualizer_node_sim',
            name='lane_visualizer_node',
            output='screen',
        ),
        Node(
            package='qcar_navigation',
            executable='lidar_kalman_node_amh19_sim',
            name='lidar_listener_node',
            output='screen',
        ),
        Node(
            package='qcar_navigation',
            executable='qcar_lidar_alert_2_sim',
            name='obstacle_detector',
            output='screen',
        ),
    ])
