from setuptools import find_packages, setup
from glob import glob 
import os 

package_name = 'qcar_navigation'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), 
         glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'rviz'),   # AGREGAR ESTO
         glob('rviz/*.rviz')),                           # AGREGAR ESTO
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='abrahammh19 nataliahaha',
    maintainer_email='abrahammorohdez@gmail.com',
    description='Package from Sudo Drive team for 8th semester Autonomous Taxi Task Project',
    license='Apache 2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            # ── Hardware (QCar real) ──────────────────────────────────────
            'undistorted_node = qcar_navigation.undistorted_node:main',
            'lane_visualizer_node = qcar_navigation.lane_visualizer_node:main',
            'lane_detection_sw_node = qcar_navigation.lane_detection_sw_node:main',
            'pure_pursuit_vision_node = qcar_navigation.pure_pursuit_vision_node:main',
            'lidar_kalman_node_amh19 = qcar_navigation.lidar_kalman_node_amh19:main',
            'qcar_lidar_alert_2 = qcar_navigation.qcar_lidar_alert_2:main',
            # ── Simulación (/qcar_sim/*) ──────────────────────────────────
            'undistorted_node_sim = qcar_navigation.undistorted_node_sim:main',
            'lane_detection_sw_node_sim = qcar_navigation.lane_detection_sw_node_sim:main',
            'lane_visualizer_node_sim = qcar_navigation.lane_visualizer_node_sim:main',
            'pure_pursuit_vision_node_sim = qcar_navigation.pure_pursuit_vision_node_sim:main',
            'lidar_kalman_node_amh19_sim = qcar_navigation.lidar_kalman_node_amh19_sim:main',
            'qcar_lidar_alert_2_sim = qcar_navigation.qcar_lidar_alert_2_sim:main',
            # ── Odometría IMU (BNO055) ────────────────────────────────────
            'imu_external = qcar_navigation.imu_external:main',
            'pose_ekf_qcar_2 = qcar_navigation.pose_ekf_qcar_2:main',
            'pose_final_qcar = qcar_navigation.pose_final_qcar:main',
            # ── Trayectoria grabada + mux ─────────────────────────────────
            'trayectoria_grabar_csv = qcar_navigation.trayectoria_grabar_csv_node:main',
            'qcar_pure_pursuit = qcar_navigation.qcar_pure_pursuit:main',
            'cmd_mux_node = qcar_navigation.cmd_mux_node:main',
            'pose_monitor_node = qcar_navigation.pose_monitor_node:main',
            'lidar_loc_node = qcar_navigation.lidar_loc_node:main',
        ],
    },
)
