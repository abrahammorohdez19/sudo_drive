#!/usr/bin/env python3
"""
=======================================================================
 Pose Monitor Node — QCar Sudo Drive
 Author: Abraham Moro-Hernandez (AMH19)
-----------------------------------------------------------------------
 Visualización en tiempo real de la posición del QCar sobre el mapa.
 Muestra pose actual, trayectoria recorrida, referencia CSV y waypoint
 objetivo del pure pursuit, todo en una ventana matplotlib en vivo.

 Subscriptions:
   /qcar/pose          (geometry_msgs/Vector3Stamped)  x, y, yaw
   /qcar/mux_source    (std_msgs/String)               'vision'|'trajectory'

 Parámetros:
   path_csv     — CSV de referencia (columnas x, y). Vacío = sin referencia.
   update_hz    — frecuencia de actualización del plot (default 5 Hz)
   trail_len    — máximo de puntos de traza guardados (default 2000)
=======================================================================
"""

import csv
import math
import signal
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('TkAgg')          # cambia a 'Qt5Agg' si TkAgg no está disponible
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3Stamped
from std_msgs.msg import String


class PoseMonitorNode(Node):

    def __init__(self):
        super().__init__('pose_monitor_node')

        self.declare_parameter('path_csv',   '')
        self.declare_parameter('update_hz',  5.0)
        self.declare_parameter('trail_len',  2000)

        path_csv   = self.get_parameter('path_csv').value
        update_hz  = self.get_parameter('update_hz').value
        self.trail = self.get_parameter('trail_len').value

        # ── Estado ──────────────────────────────────────────────────────
        self.pose_x    = 0.0
        self.pose_y    = 0.0
        self.pose_yaw  = 0.0
        self.mux_src   = 'vision'
        self.trail_x   = []
        self.trail_y   = []
        self.ref_x     = []
        self.ref_y     = []
        self.pose_rcvd = False

        # ── Trayectoria de referencia ────────────────────────────────────
        if path_csv.strip():
            p = Path(path_csv)
            if p.exists():
                with p.open() as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        try:
                            self.ref_x.append(float(row['x']))
                            self.ref_y.append(float(row['y']))
                        except (KeyError, ValueError):
                            pass
                self.get_logger().info(f'Referencia cargada: {len(self.ref_x)} puntos de {path_csv}')
            else:
                self.get_logger().warn(f'CSV no encontrado: {path_csv}')

        # ── Suscripciones ────────────────────────────────────────────────
        self.create_subscription(Vector3Stamped, '/qcar/pose',
                                 self._cb_pose, 10)
        self.create_subscription(String, '/qcar/mux_source',
                                 self._cb_mux, 10)

        # ── Plot ─────────────────────────────────────────────────────────
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.fig.canvas.manager.set_window_title('QCar — Pose Monitor')
        self._init_plot()

        self.create_timer(1.0 / update_hz, self._update_plot)

        self._save_dir = Path.home() / 'Workspace' / 'sudo_drive' / 'trayectoria'
        self._save_dir.mkdir(parents=True, exist_ok=True)

        self.get_logger().info(f'Pose Monitor iniciado — update @ {update_hz} Hz')
        self.get_logger().info(f'Trayectoria se guardará en: {self._save_dir}')

    # ════════════════════════════════════════════════════════════════════
    #  CALLBACKS
    # ════════════════════════════════════════════════════════════════════

    def _cb_pose(self, msg: Vector3Stamped):
        self.pose_x   = msg.vector.x
        self.pose_y   = msg.vector.y
        self.pose_yaw = msg.vector.z
        self.pose_rcvd = True
        self.trail_x.append(self.pose_x)
        self.trail_y.append(self.pose_y)
        if len(self.trail_x) > self.trail:
            self.trail_x.pop(0)
            self.trail_y.pop(0)

    def _cb_mux(self, msg: String):
        self.mux_src = msg.data

    # ════════════════════════════════════════════════════════════════════
    #  PLOT
    # ════════════════════════════════════════════════════════════════════

    def _init_plot(self):
        ax = self.ax
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_title('QCar — Posición en tiempo real')

        if self.ref_x:
            ax.plot(self.ref_x, self.ref_y, '--', color='gray',
                    linewidth=1.5, label='Referencia CSV', zorder=1)

        self._ln_trail,  = ax.plot([], [], '-', color='steelblue',
                                   linewidth=1.2, label='Trayectoria real', zorder=2)
        self._sc_pos     = ax.scatter([], [], s=80, color='red',
                                      zorder=5, label='Posición actual')
        self._arrow      = None
        self._txt_info   = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                                   va='top', fontsize=9, family='monospace',
                                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax.legend(loc='lower right', fontsize=8)
        self.fig.tight_layout()

    def _update_plot(self):
        if not self.pose_rcvd:
            return

        ax = self.ax

        # Traza
        self._ln_trail.set_data(self.trail_x, self.trail_y)

        # Posición actual
        self._sc_pos.set_offsets([[self.pose_x, self.pose_y]])

        # Flecha de orientación
        if self._arrow is not None:
            self._arrow.remove()
        arrow_len = 0.10
        dx = arrow_len * math.cos(self.pose_yaw)
        dy = arrow_len * math.sin(self.pose_yaw)
        self._arrow = ax.annotate('', xy=(self.pose_x + dx, self.pose_y + dy),
                                  xytext=(self.pose_x, self.pose_y),
                                  arrowprops=dict(arrowstyle='->', color='red',
                                                  lw=2.0))

        # HUD texto
        src_color = '🟢' if self.mux_src == 'vision' else '🟡'
        self._txt_info.set_text(
            f'X:   {self.pose_x:+.3f} m\n'
            f'Y:   {self.pose_y:+.3f} m\n'
            f'Yaw: {math.degrees(self.pose_yaw):+.1f}°\n'
            f'{src_color} Fuente: {self.mux_src}\n'
            f'Puntos traza: {len(self.trail_x)}'
        )

        # Auto-zoom centrado en el carro
        if self.trail_x:
            all_x = list(self.ref_x) + self.trail_x
            all_y = list(self.ref_y) + self.trail_y
            pad = 0.5
            ax.set_xlim(min(all_x) - pad, max(all_x) + pad)
            ax.set_ylim(min(all_y) - pad, max(all_y) + pad)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()


    def save_trajectory(self):
        if len(self.trail_x) < 2:
            return
        ts  = datetime.now().strftime('%Y%m%d_%H%M%S')
        out = self._save_dir / f'trayectoria_{ts}.csv'
        with out.open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['x', 'y'])
            for x, y in zip(self.trail_x, self.trail_y):
                writer.writerow([f'{x:.4f}', f'{y:.4f}'])
        self.get_logger().info(
            f'Trayectoria guardada: {out}  ({len(self.trail_x)} puntos)')


def main(args=None):
    rclpy.init(args=args)
    node = PoseMonitorNode()

    signal.signal(signal.SIGINT, lambda *_: rclpy.shutdown())

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.save_trajectory()
        plt.close('all')
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
