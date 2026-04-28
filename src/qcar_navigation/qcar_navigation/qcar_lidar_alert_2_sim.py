#!/usr/bin/env python3
"""
=======================================================================
 QCar Obstacle Detector Node (SIM) - Smart Mobility
 Author: Abraham Moro-Hernandez (AMH19)

-----------------------------------------------------------------------
 Versión simulación de qcar_lidar_alert_2.py.
 Cambios respecto al nodo de hardware:
   - LiDAR:    /qcar/scan      → /qcar_sim/scan
   - Velocidad: /qcar/velocity (Vector3Stamped)
               → /qcar_sim/odom (nav_msgs/Odometry)
   - Alerta:   /qcar/obstacle_alert → /qcar_sim/obstacle_alert
   - front_angle_offset: 4.71 rad (270°, montaje físico del QCar)
               → 0.0 rad (LiDAR simulado apunta hacia adelante)
=======================================================================
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool
from nav_msgs.msg import Odometry
import numpy as np


class ObstacleDetector(Node):
    """Obstacle detection module for the QCar simulation using LiDAR."""

    def __init__(self):
        super().__init__('obstacle_detector')

        # ----------------------------------------------------------
        # Parameters
        # ----------------------------------------------------------
        self.distance_threshold = 0.35  # meters

        self.angle_range_low    = 22.5  # deg  (velocity <= 1 m/s)
        self.angle_range_high   = 30.0  # deg  (velocity > 1 m/s)
        self.velocity_threshold = 1.0   # m/s

        self.angle_range         = self.angle_range_low
        self.current_velocity_x  = 0.0

        # En simulación el LiDAR apunta directamente hacia adelante (0 rad)
        self.front_angle_offset = 0.0

        self.debug_mode = True

        # ----------------------------------------------------------
        # Publishers
        # ----------------------------------------------------------
        self.alert_pub = self.create_publisher(Bool, '/qcar_sim/obstacle_alert', 10)

        # ----------------------------------------------------------
        # Subscribers
        # ----------------------------------------------------------
        self.lidar_sub = self.create_subscription(
            LaserScan, '/qcar_sim/scan', self.lidar_callback, 10)

        # Velocidad desde el odómetro simulado
        self.odom_sub = self.create_subscription(
            Odometry, '/qcar_sim/odom', self.odom_callback, 10)

        self.get_logger().info("QCar Obstacle Detector Node (SIM) initialized")
        self.get_logger().info(f"  Distance threshold: {self.distance_threshold} m")
        self.get_logger().info(f"  Angle range (<= {self.velocity_threshold} m/s): ±{self.angle_range_low}°")
        self.get_logger().info(f"  Angle range (>  {self.velocity_threshold} m/s): ±{self.angle_range_high}°")
        self.get_logger().info(f"  LiDAR front offset: {np.degrees(self.front_angle_offset):.1f}° (sim = 0°)")
        self.get_logger().info(f"  LiDAR topic  : /qcar_sim/scan")
        self.get_logger().info(f"  Odom topic   : /qcar_sim/odom")
        self.get_logger().info(f"  Alert topic  : /qcar_sim/obstacle_alert")
        if self.debug_mode:
            self.get_logger().info("  Debug Mode: ON")

    def odom_callback(self, msg: Odometry):
        self.current_velocity_x = abs(msg.twist.twist.linear.x)
        if self.current_velocity_x > self.velocity_threshold:
            self.angle_range = self.angle_range_high
        else:
            self.angle_range = self.angle_range_low

        if self.debug_mode:
            self.get_logger().info(
                f"Velocity X: {self.current_velocity_x:.2f} m/s -> Cone: ±{self.angle_range}°",
                throttle_duration_sec=2.0
            )

    def lidar_callback(self, msg: LaserScan):
        angle_min       = msg.angle_min
        angle_increment = msg.angle_increment
        ranges          = np.array(msg.ranges)

        if self.debug_mode:
            min_dist  = float('inf')
            min_angle = 0
            for i, d in enumerate(ranges):
                if msg.range_min < d < min_dist:
                    min_dist  = d
                    min_angle = angle_min + i * angle_increment
            self.get_logger().info(
                f"Closest obstacle: {min_dist:.2f} m at {min_angle:.3f} rad ({np.degrees(min_angle):.1f}°)",
                throttle_duration_sec=1.0
            )

        center_index  = int((self.front_angle_offset - angle_min) / angle_increment)
        range_indices = int((self.angle_range * np.pi / 180) / angle_increment)

        start_idx = max(0, center_index - range_indices)
        end_idx   = min(len(ranges) - 1, center_index + range_indices)

        obstacle_detected = False
        min_distance      = float('inf')

        for i in range(start_idx, end_idx + 1):
            distance = ranges[i]
            if msg.range_min < distance < self.distance_threshold:
                obstacle_detected = True
                min_distance = min(min_distance, distance)

        alert_msg      = Bool()
        alert_msg.data = obstacle_detected
        self.alert_pub.publish(alert_msg)

        if obstacle_detected:
            self.get_logger().warn(
                f"Obstacle detected at {min_distance:.2f} m "
                f"(velocity: {self.current_velocity_x:.2f} m/s, cone: ±{self.angle_range}°)"
            )


def main(args=None):
    rclpy.init(args=args)
    node = ObstacleDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
