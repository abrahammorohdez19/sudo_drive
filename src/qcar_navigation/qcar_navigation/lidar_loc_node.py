#!/usr/bin/env python3
"""
=======================================================================
 LiDAR Localization Node — QCar Sudo Drive  (STANDALONE / OPCIONAL)
 Author: Abraham Moro-Hernandez (AMH19)
-----------------------------------------------------------------------
 Localización geométrica 2-D en cuarto rectangular conocido.
 NO modifica ningún otro nodo ni topic existente.

 Algoritmo:
   1. Histograma de normales de pares consecutivos de scan → heading
   2. Proyectar puntos en ejes del cuarto → distancias a 4 paredes
   3. Posición = (dist_pared_izq, dist_pared_inferior)

 Subscriptions:  /qcar/scan        (sensor_msgs/LaserScan)
 Published:      /qcar/lidar_pose  (geometry_msgs/PoseStamped)
                 /qcar/lidar_pose_simple (std_msgs/String)  — HUD legible

 Parámetros:
   room_x           — ancho del cuarto en X  (metros)
   room_y           — largo del cuarto en Y  (metros)
   lidar_front_rad  — ángulo LiDAR que apunta al frente del QCar (rad)
                      QCar = 4.712 (270°)
   scan_topic       — topic del scan (default /qcar/scan)
=======================================================================
"""

import math
import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSProfile, ReliabilityPolicy,
                        HistoryPolicy, DurabilityPolicy)
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
import numpy as np


_QOS_BE = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=5,
    durability=DurabilityPolicy.VOLATILE,
)


class LidarLocNode(Node):

    def __init__(self):
        super().__init__('lidar_loc_node')

        # ── Parámetros ────────────────────────────────────────────────
        self.declare_parameter('room_x',          4.0)    # ancho cuarto (m)
        self.declare_parameter('room_y',          5.0)    # largo cuarto (m)
        self.declare_parameter('lidar_front_rad', 4.712)  # 270° frente QCar
        self.declare_parameter('scan_topic',      '/qcar/scan')
        self.declare_parameter('max_range',       6.0)    # descartar lecturas > X m
        self.declare_parameter('wall_gap_m',      0.25)   # gap max entre pts misma pared

        p = lambda n: self.get_parameter(n).value
        self.room_x       = float(p('room_x'))
        self.room_y       = float(p('room_y'))
        self.lidar_front  = float(p('lidar_front_rad'))
        self.max_range    = float(p('max_range'))
        self.wall_gap     = float(p('wall_gap_m'))

        # ── I/O ───────────────────────────────────────────────────────
        self.create_subscription(
            LaserScan, p('scan_topic'), self._cb_scan, _QOS_BE)

        self._pub_pose = self.create_publisher(
            PoseStamped, '/qcar/lidar_pose', 10)
        self._pub_hud = self.create_publisher(
            String, '/qcar/lidar_pose_simple', 10)

        g = self.get_logger().info
        g('=' * 58)
        g(' LIDAR LOC NODE  [standalone — no afecta navegación]')
        g(f'  Cuarto: {self.room_x:.1f} m × {self.room_y:.1f} m')
        g(f'  LiDAR front offset: {math.degrees(self.lidar_front):.1f}°')
        g('  Output: /qcar/lidar_pose  /qcar/lidar_pose_simple')
        g('=' * 58)

    # ════════════════════════════════════════════════════════════════
    #  CALLBACK PRINCIPAL
    # ════════════════════════════════════════════════════════════════

    def _cb_scan(self, msg: LaserScan):
        angles = (msg.angle_min
                  + np.arange(len(msg.ranges)) * msg.angle_increment)
        r = np.array(msg.ranges, dtype=np.float32)

        # Filtrar puntos inválidos
        valid = (r > msg.range_min) & (r < min(msg.range_max, self.max_range))
        if valid.sum() < 30:
            return

        r = r[valid];  a = angles[valid]
        px = r * np.cos(a)
        py = r * np.sin(a)

        # ── 1. Heading del robot ──────────────────────────────────────
        yaw_robot = self._heading_from_walls(px, py)

        # ── 2. Posición en el cuarto ──────────────────────────────────
        x_pos, y_pos = self._position_from_walls(px, py, yaw_robot)

        # ── 3. Publicar PoseStamped ───────────────────────────────────
        pose = PoseStamped()
        pose.header.stamp    = msg.header.stamp
        pose.header.frame_id = 'room'
        pose.pose.position.x = float(x_pos)
        pose.pose.position.y = float(y_pos)
        half = yaw_robot / 2.0
        pose.pose.orientation.z = math.sin(half)
        pose.pose.orientation.w = math.cos(half)
        self._pub_pose.publish(pose)

        # ── 4. HUD de texto ───────────────────────────────────────────
        hud = String()
        hud.data = (f'x={x_pos:.2f}m  y={y_pos:.2f}m  '
                    f'yaw={math.degrees(yaw_robot):+.1f}°')
        self._pub_hud.publish(hud)

        self.get_logger().info(hud.data, throttle_duration_sec=0.5)

    # ════════════════════════════════════════════════════════════════
    #  HEADING — histograma de normales de pares consecutivos
    # ════════════════════════════════════════════════════════════════

    def _heading_from_walls(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Los puntos consecutivos en la misma pared forman segmentos cuyas
        normales se acumulan en el histograma.  El pico dominante da la
        orientación de la pared más cercana → heading del robot.
        """
        dx = np.diff(x);  dy = np.diff(y)
        gap = np.hypot(dx, dy)

        # Solo pares que pertenecen a la misma pared (gap pequeño)
        same_wall = gap < self.wall_gap
        if same_wall.sum() < 5:
            return 0.0

        # Normal de cada segmento, colapsada a [0, π) (paredes no tienen sentido)
        normals = np.arctan2(dy[same_wall], dx[same_wall]) % math.pi

        # Histograma suavizado
        n_bins = 360
        hist, edges = np.histogram(normals, bins=n_bins, range=(0.0, math.pi))
        hist = np.convolve(hist, np.ones(9) / 9, mode='same')

        peak      = int(np.argmax(hist))
        wall_dir  = (edges[peak] + edges[peak + 1]) / 2.0

        # El frente del QCar está a `lidar_front` en el frame LiDAR.
        # heading_room = ángulo de la pared + corrección de montaje
        heading = (wall_dir - self.lidar_front + math.pi / 2.0) % (2 * math.pi)
        return heading

    # ════════════════════════════════════════════════════════════════
    #  POSICIÓN — proyección sobre ejes del cuarto
    # ════════════════════════════════════════════════════════════════

    def _position_from_walls(self, x: np.ndarray, y: np.ndarray,
                             yaw: float) -> tuple:
        """
        Rota el scan al frame del cuarto y busca los picos de densidad
        de puntos sobre cada eje (= posición de las paredes).
        Posición del robot = distancia a la pared de origen (0,0).
        """
        c, s   = math.cos(-yaw), math.sin(-yaw)
        xr     =  c * x - s * y   # puntos en frame del cuarto (eje X)
        yr     =  s * x + c * y   # puntos en frame del cuarto (eje Y)

        x_pos = self._wall_distance(xr, self.room_x)
        y_pos = self._wall_distance(yr, self.room_y)
        return x_pos, y_pos

    def _wall_distance(self, proj: np.ndarray, room_dim: float) -> float:
        """
        En la proyección 1D, las dos paredes opuestas aparecen como picos
        a ≈ -d1  y  ≈ +d2,  donde d1+d2 ≈ room_dim.
        La posición del robot sobre ese eje = d1 (distancia a la pared origen).
        """
        margin = room_dim * 0.15       # ignorar lecturas muy cercanas (ruido)
        hist, edges = np.histogram(
            proj, bins=300, range=(-room_dim * 1.2, room_dim * 1.2))
        centers = (edges[:-1] + edges[1:]) / 2.0
        hist    = np.convolve(hist, np.ones(5) / 5, mode='same')

        # Pico en el lado negativo (pared "detrás / izquierda")
        neg_mask = centers < -margin
        if neg_mask.any() and hist[neg_mask].max() > 0:
            d_neg = float(-centers[neg_mask][np.argmax(hist[neg_mask])])
        else:
            d_neg = room_dim / 2.0

        # Clamp a dimensión del cuarto
        d_neg = float(np.clip(d_neg, 0.0, room_dim))
        return d_neg


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════

def main(args=None):
    rclpy.init(args=args)
    node = LidarLocNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
