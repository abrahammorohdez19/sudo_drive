#!/usr/bin/env python3
"""
=======================================================================
 LiDAR Visualizer Node — QCar Sudo Drive
 Author: Abraham Moro-Hernandez (AMH19)
-----------------------------------------------------------------------
 Visualización cartesiana del LiDAR. El frente del carro apunta
 siempre hacia arriba (+Y / Norte). La flecha amarilla indica el
 heading según el EKF (/qcar/pose).

 Subscriptions:
   /qcar/scan   (sensor_msgs/LaserScan)
   /qcar/pose   (geometry_msgs/Vector3Stamped)  x, y, theta

 Parámetros:
   view_half_m         — rango visible ±m (default 5.0)
   lidar_front_offset  — ángulo del frente en el scan (default 4.71 rad)

 Uso:
   ros2 run qcar_navigation lidar_map_overlay_node
=======================================================================
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSProfile, ReliabilityPolicy,
                        HistoryPolicy, DurabilityPolicy)
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Vector3Stamped


class LidarMapOverlayNode(Node):

    def __init__(self):
        super().__init__('lidar_map_overlay_node')

        # ── Parámetros ────────────────────────────────────────────────
        self.declare_parameter('view_half_m',        5.0)
        self.declare_parameter('lidar_front_offset', 1.57)

        self.view_half       = float(self.get_parameter('view_half_m').value)
        self.lidar_front_off = float(self.get_parameter('lidar_front_offset').value)

        # ── Pose del carro ────────────────────────────────────────────
        self.pose_x      = 0.0
        self.pose_y      = 0.0
        self.pose_yaw    = 0.0
        self._pose_rcvd  = False

        # ── QoS ───────────────────────────────────────────────────────
        qos_lidar = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.create_subscription(
            LaserScan, '/qcar/scan', self._cb_scan, qos_lidar)
        self.create_subscription(
            Vector3Stamped, '/qcar/pose', self._cb_pose, 10)

        self.latest_scan = None
        self.create_timer(0.1, self._timer_cb)

        # ── Figura matplotlib ─────────────────────────────────────────
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(7, 7))
        self.fig.canvas.manager.set_window_title('QCar — LiDAR Visualizer')
        self.fig.patch.set_facecolor('#111111')

        self.ax.set_xlim(-self.view_half, self.view_half)
        self.ax.set_ylim(-self.view_half, self.view_half)
        self.ax.set_aspect('equal')
        self.ax.set_facecolor('#1a1a1a')
        self.ax.tick_params(colors='#888888')
        self.ax.set_xlabel('X — derecha (m)', color='#888888', fontsize=9)
        self.ax.set_ylabel('Y — adelante (m)', color='#888888', fontsize=9)
        for spine in self.ax.spines.values():
            spine.set_edgecolor('#444444')

        # Círculos de distancia
        for r in range(1, int(self.view_half) + 1):
            self.ax.add_patch(plt.Circle(
                (0, 0), r, color='#444444', fill=False,
                linewidth=0.6, linestyle='--', zorder=2))
            self.ax.text(0.05, r + 0.1, f'{r} m', color='#555555',
                         fontsize=7, va='bottom', zorder=2)

        self.ax.axhline(0, color='#333333', linewidth=0.6, zorder=2)
        self.ax.axvline(0, color='#333333', linewidth=0.6, zorder=2)

        v = self.view_half * 0.93
        for label, xy in [('N', (0, v)), ('S', (0, -v)),
                           ('E', (v, 0)), ('W', (-v, 0))]:
            self.ax.text(*xy, label, color='#777777', fontsize=9,
                         ha='center', va='center', zorder=2)

        # ── Artistas dinámicos ────────────────────────────────────────
        self._scan_scatter = self.ax.scatter(
            [], [], s=4, c='cyan', alpha=0.85, zorder=4)

        self._car_dot, = self.ax.plot(
            [0], [0], 'o', color='red', markersize=7, zorder=6)

        self._heading_arrow = self.ax.annotate(
            '', xy=(0.0, 0.55), xytext=(0.0, 0.0),
            arrowprops=dict(arrowstyle='->', color='yellow',
                            lw=2.0, mutation_scale=15),
            zorder=6)

        self.get_logger().info(
            f'LidarMapOverlayNode listo  '
            f'view={self.view_half}m  '
            f'lidar_front_offset={self.lidar_front_off:.2f} rad')

    # ════════════════════════════════════════════════════════════════
    #  CALLBACKS
    # ════════════════════════════════════════════════════════════════

    def _cb_scan(self, msg: LaserScan):
        self.latest_scan = msg

    def _cb_pose(self, msg: Vector3Stamped):
        self.pose_x     = float(msg.vector.x)
        self.pose_y     = float(msg.vector.y)
        self.pose_yaw   = float(msg.vector.z)
        self._pose_rcvd = True

    # ════════════════════════════════════════════════════════════════
    #  TIMER
    # ════════════════════════════════════════════════════════════════

    def _timer_cb(self):
        if self.latest_scan is not None:
            self._plot_scan(self.latest_scan)

        # Flecha de heading rota con el yaw del EKF
        # Negado porque el frente del carro apunta en -cos(yaw) en el frame display
        self._heading_arrow.xy = (
            -np.sin(self.pose_yaw) * 0.6,
            -np.cos(self.pose_yaw) * 0.6,
        )

        pose_str = (f'x={self.pose_x:.2f}  y={self.pose_y:.2f}  '
                    f'yaw={np.degrees(self.pose_yaw):.1f}°'
                    if self._pose_rcvd else '⚠ /qcar/pose NO recibido')
        self.ax.set_title(f'QCar LiDAR — {pose_str}',
                          color='white', fontsize=9)
        plt.pause(0.001)

    # ════════════════════════════════════════════════════════════════
    #  PLOT LiDAR — polar → cartesiano
    # ════════════════════════════════════════════════════════════════

    def _plot_scan(self, msg: LaserScan):
        ranges = np.array(msg.ranges, dtype=float)
        valid  = (np.isfinite(ranges)
                  & (ranges >= msg.range_min)
                  & (ranges <= msg.range_max))

        if not np.any(valid):
            self._scan_scatter.set_offsets(np.empty((0, 2)))
            return

        angles = np.linspace(msg.angle_min, msg.angle_max, len(ranges))
        r      = np.clip(ranges[valid], msg.range_min, msg.range_max)

        # Restar offset de montaje: θ=4.71 (frente) → θ=0 → Norte/+Y
        tc = angles[valid] - self.lidar_front_off
        x  = r * np.sin(tc)
        y  = r * np.cos(tc)

        self._scan_scatter.set_offsets(np.column_stack([x, y]))

        self.get_logger().info(
            f'x={self.pose_x:.2f}  y={self.pose_y:.2f}  '
            f'yaw={np.degrees(self.pose_yaw):.1f}°  |  '
            f'Min={float(np.min(r)):.2f}m  Pts={valid.sum()}',
            throttle_duration_sec=1.0)


# ════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════

def main(args=None):
    rclpy.init(args=args)
    node = LidarMapOverlayNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        plt.close('all')
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
