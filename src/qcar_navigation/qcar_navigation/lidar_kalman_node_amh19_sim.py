#!/usr/bin/env python3
"""
=========================================================
AMH19 - ROS2 LiDAR Listener and Polar Visualization Tool (SIM)
---------------------------------------------------------
Author: Abraham Moro-Hernandez (AMH19)
=========================================================
Description:
    Versión simulación de lidar_kalman_node_amh19.py.
    Cambio principal: suscripción a /qcar_sim/scan
    en lugar de /qcar/scan.

Features:
    - QoS configurado para datos de sensor (Best Effort)
    - Consume /qcar_sim/scan
    - Visualización polar en tiempo real a 10 Hz con matplotlib
    - Reporta distancia mínima y promedio
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import LaserScan
import numpy as np
import matplotlib.pyplot as plt


class LidarListener(Node):
    """ROS2 node that listens to a LaserScan topic and visualizes LiDAR data."""

    def __init__(self):
        super().__init__('lidar_listener_node')

        qos_lidar = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.VOLATILE
        )

        # Tópico del simulador
        self.create_subscription(
            LaserScan,
            '/qcar_sim/scan',
            self.lidar_callback,
            qos_profile=qos_lidar
        )

        self.timer_period = 0.1
        self.timer = self.create_timer(self.timer_period, self.timer_callback)
        self.latest_scan = None

        plt.ion()
        self.fig, self.ax = plt.subplots(subplot_kw={'projection': 'polar'})
        self.ax.set_title("QCar LiDAR View — SIM (10 Hz)", va='bottom')
        self.ax.set_rmax(6.0)
        self.scatter = None

        self.get_logger().info("LiDAR Listener node (SIM) active and listening to /qcar_sim/scan")

    def lidar_callback(self, msg: LaserScan):
        self.latest_scan = msg

    def timer_callback(self):
        if self.latest_scan is not None:
            self.plot_scan(self.latest_scan)

    def plot_scan(self, msg: LaserScan):
        ranges = np.array(msg.ranges, dtype=float)
        valid = np.isfinite(ranges)

        if not np.any(valid):
            self.get_logger().warn("LiDAR produced no valid range data")
            return

        angles = np.linspace(msg.angle_min, msg.angle_max, len(ranges))
        ranges = np.clip(ranges, msg.range_min, msg.range_max)

        self.ax.clear()
        self.ax.scatter(angles[valid], ranges[valid], s=5, c='cyan')
        self.ax.set_theta_zero_location('N')
        self.ax.set_theta_direction(-1)
        self.ax.set_rmax(6.0)
        self.ax.set_title("QCar LiDAR View — SIM (10 Hz)", va='bottom')
        plt.pause(0.001)

        d_min  = np.min(ranges[valid])
        d_mean = np.mean(ranges[valid])

        self.get_logger().info(
            f"Minimum distance = {d_min:.2f} m | Mean distance = {d_mean:.2f} m"
        )


def main(args=None):
    rclpy.init(args=args)
    node = LidarListener()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
