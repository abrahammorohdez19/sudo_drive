from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """
    Percepción + odometría únicamente.
    Para lanzar el control (movimiento) usa control.launch.py.

    Uso:
        ros2 launch qcar_navigation physical.launch.py
        ros2 launch qcar_navigation physical.launch.py show_map:=true
        ros2 launch qcar_navigation physical.launch.py show_map:=true path_csv:=/ruta/waypoints.csv
    """
    return LaunchDescription([

        DeclareLaunchArgument(
            'show_map',
            default_value='false',
            description='Lanzar ventana matplotlib con trayectoria en tiempo real.'
        ),
        DeclareLaunchArgument(
            'path_csv',
            default_value='',
            description='CSV de referencia para superponer en el mapa (opcional).'
        ),

        # ── Visión ───────────────────────────────────────────────────
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

        # ── LiDAR / obstáculos ───────────────────────────────────────
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

        # ── Odometría IMU + encoder ──────────────────────────────────
        Node(
            package='qcar_navigation',
            executable='pose_ekf_qcar_2',
            name='pose_estimator',
            output='screen',
        ),

        # ── Mapa en tiempo real (opcional, requiere display) ─────────
        Node(
            package='qcar_navigation',
            executable='pose_monitor_node',
            name='pose_monitor',
            output='screen',
            condition=IfCondition(LaunchConfiguration('show_map')),
            parameters=[{
                'path_csv': LaunchConfiguration('path_csv'),
            }],
        ),
    ])
