from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='qcar_navigation',
            executable='undistorted_node',
            name='undistort_node',
            output='screen',
        ),
        Node(
            package='qcar_navigation',
            executable='lane_detection_sw_node',
            name='lane_detection_node',
            output='screen',
        ),
        Node(
            package='qcar_navigation',
            executable='lane_visualizer_node',
            name='lane_visualizer_node',
            output='screen',
        ),
        Node(
            package='qcar_navigation',
            executable='lidar_kalman_node_amh19',
            name='lidar_listener_node',
            output='screen',
        ),
        Node(
            package='qcar_navigation',
            executable='qcar_lidar_alert_2',
            name='obstacle_detector',
            output='screen',
        ),
    ])
