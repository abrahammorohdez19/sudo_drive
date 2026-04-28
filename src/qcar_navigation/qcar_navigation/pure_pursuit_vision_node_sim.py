#!/usr/bin/env python3
"""
=======================================================================
 Pure Pursuit Controller (SIM) — Vision-based | QCar Sudo Drive
 Author: Abraham Moro-Hernandez (AMH19)
-----------------------------------------------------------------------
 Versión simulación de pure_pursuit_vision_node.py.
 Cambios respecto al nodo de hardware:
   - Velocidad leída desde /qcar_sim/odom  (nav_msgs/Odometry)
     en lugar de /qcar/velocity (Vector3Stamped)
   - Comando enviado a /qcar_sim/user_command en lugar de
     /qcar/user_command
   - Alerta de obstáculo desde /qcar_sim/obstacle_alert

 Subscriptions:
   /amh19/lane/lines         (std_msgs/Float32MultiArray)  [a, b, c]
   /amh19/lane/centroid      (geometry_msgs/Point)
   /qcar_sim/odom            (nav_msgs/Odometry)  velocidad lineal x
   /qcar_sim/obstacle_alert  (std_msgs/Bool)

 Published Topics:
   /qcar_sim/user_command    (geometry_msgs/Vector3Stamped)
                              vector.x = throttle (m/s)
                              vector.y = -steering (rad)
=======================================================================
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3Stamped, Point
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, Float32MultiArray

import math
import csv
import signal
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt


class PurePursuitVisionNode(Node):

    def __init__(self):
        super().__init__('pure_pursuit_vision_node')

        self.declare_parameter('lookahead',       0.20)
        self.declare_parameter('k_gain',          0.50)
        self.declare_parameter('v_ref',           0.04)
        self.declare_parameter('wheelbase',       0.256)
        self.declare_parameter('max_steer',       0.50)

        self.declare_parameter('img_width',       640)
        self.declare_parameter('img_height',      480)

        self.declare_parameter('roi_bottom',      0.97)
        self.declare_parameter('lookahead_rows',  80)

        # Topics — apuntando a la simulación
        self.declare_parameter('lines_topic',    '/amh19/lane/lines')
        self.declare_parameter('centroid_topic', '/amh19/lane/centroid')
        self.declare_parameter('odom_topic',     '/qcar_sim/odom')
        self.declare_parameter('cmd_topic',      '/qcar_sim/user_command')
        self.declare_parameter('alert_topic',    '/qcar_sim/obstacle_alert')

        p = lambda n: self.get_parameter(n).value
        self.Lfc            = p('lookahead')
        self.k_gain         = p('k_gain')
        self.v_ref          = p('v_ref')
        self.L              = p('wheelbase')
        self.max_steer_cmd  = p('max_steer')
        self.img_w          = p('img_width')
        self.img_h          = p('img_height')
        self.roi_bottom     = p('roi_bottom')
        self.lookahead_rows = p('lookahead_rows')

        lines_topic    = p('lines_topic')
        centroid_topic = p('centroid_topic')
        odom_topic     = p('odom_topic')
        cmd_topic      = p('cmd_topic')
        alert_topic    = p('alert_topic')

        self.poly   = None
        self.cx     = None
        self.v_enc  = 0.0
        self.paro   = False
        self.n_frames = 0

        self.log_t      = []
        self.log_steer  = []
        self.log_err    = []
        self.log_v      = []
        self.start_time = self.get_clock().now()

        self.lines_sub = self.create_subscription(
            Float32MultiArray, lines_topic, self.lines_callback, 10)
        self.centroid_sub = self.create_subscription(
            Point, centroid_topic, self.centroid_callback, 10)
        # La velocidad viene del odómetro simulado (nav_msgs/Odometry)
        self.odom_sub = self.create_subscription(
            Odometry, odom_topic, self.odom_callback, 10)
        self.paro_sub = self.create_subscription(
            Bool, alert_topic, self.paro_callback, 10)

        self.pub = self.create_publisher(Vector3Stamped, cmd_topic, 10)

        self.timer = self.create_timer(0.02, self.control_loop)

        g = self.get_logger().info
        g('=' * 62)
        g(' PURE PURSUIT VISION NODE (SIM)  [v1 — línea central]')
        g('=' * 62)
        g(f'  v_ref={self.v_ref} m/s   wheelbase={self.L} m')
        g(f'  Lfc={self.Lfc}  k_gain={self.k_gain}  max_steer=±{self.max_steer_cmd} rad')
        g(f'  lookahead_rows={self.lookahead_rows}  roi_bottom={self.roi_bottom}')
        g(f'  Imagen: {self.img_w}x{self.img_h}')
        g(f'  Odom  : {odom_topic}')
        g(f'  Cmd   : {cmd_topic}')
        g('=' * 62)

    def lines_callback(self, msg: Float32MultiArray):
        data = msg.data
        if len(data) == 3 and not all(v == -1.0 for v in data):
            self.poly = np.array(data, dtype=float)
        else:
            self.poly = None

    def centroid_callback(self, msg: Point):
        self.cx = float(msg.x)

    def odom_callback(self, msg: Odometry):
        # Velocidad lineal longitudinal del simulador
        self.v_enc = float(msg.twist.twist.linear.x)

    def paro_callback(self, msg: Bool):
        old = self.paro
        self.paro = bool(msg.data)
        if self.paro != old:
            if self.paro:
                self.get_logger().warn('Obstáculo detectado: activando PARO.')
            else:
                self.get_logger().info('Obstáculo despejado: reanudando movimiento.')

    def control_loop(self):
        now = self.get_clock().now()

        if self.paro:
            self.stop_qcar()
            return

        if self.poly is None or self.cx is None:
            self.stop_qcar()
            return

        self.n_frames += 1

        delta     = self.compute_pure_pursuit_delta()
        steer_cmd = float(np.clip(delta, -self.max_steer_cmd, self.max_steer_cmd))

        cmd = Vector3Stamped()
        cmd.header.stamp    = now.to_msg()
        cmd.header.frame_id = 'base_link'
        cmd.vector.x        = float(self.v_ref)
        cmd.vector.y        = float(-steer_cmd)
        cmd.vector.z        = 0.0
        self.pub.publish(cmd)

        t       = (now.nanoseconds - self.start_time.nanoseconds) * 1e-9
        err_px  = self.cx - self.img_w / 2.0
        self.log_t.append(t)
        self.log_steer.append(math.degrees(steer_cmd))
        self.log_err.append(err_px)
        self.log_v.append(self.v_ref)

        if self.n_frames % 50 == 0:
            self.get_logger().info(
                f'[Frame {self.n_frames}]  '
                f'cx={self.cx:.0f}px  err={err_px:+.0f}px  '
                f'delta={math.degrees(delta):+.1f}°  v={self.v_ref:.3f} m/s')

    def compute_pure_pursuit_delta(self):
        y_ref  = int(self.img_h * self.roi_bottom)
        y_look = max(0, y_ref - self.lookahead_rows)

        x_look = float(np.polyval(self.poly, y_look))
        dx     = x_look - self.img_w / 2.0

        alpha  = math.atan2(dx, float(self.lookahead_rows))
        Lf     = max(self.k_gain * abs(self.v_ref) + self.Lfc, 1e-3)
        delta  = math.atan2(2.0 * self.L * math.sin(alpha), Lf)
        return delta

    def stop_qcar(self):
        cmd = Vector3Stamped()
        cmd.header.stamp    = self.get_clock().now().to_msg()
        cmd.header.frame_id = 'base_link'
        cmd.vector.x        = 0.0
        cmd.vector.y        = 0.0
        cmd.vector.z        = 0.0
        self.pub.publish(cmd)

    def report_performance(self):
        if not self.log_err:
            return

        errors_abs = [abs(e) for e in self.log_err]
        avg_err    = float(np.mean(errors_abs))
        max_err    = float(np.max(errors_abs))
        std_err    = float(np.std(errors_abs))

        tol1 = 15.0
        tol2 = 30.0
        sim1 = sum(1 for e in errors_abs if e <= tol1) / len(errors_abs) * 100
        sim2 = sum(1 for e in errors_abs if e <= tol2) / len(errors_abs) * 100

        if avg_err < tol1:
            assessment = 'EXCELENTE'
        elif avg_err < tol2:
            assessment = 'BUENO'
        elif avg_err < 60.0:
            assessment = 'ACEPTABLE'
        else:
            assessment = 'DEFICIENTE'

        print('\n========== EVALUACIÓN DE SEGUIMIENTO (SIM) ==========')
        print(f'Error lateral promedio:    {avg_err:.1f} px')
        print(f'Error lateral máximo:      {max_err:.1f} px')
        print(f'Desviación estándar:       {std_err:.1f} px')
        print(f'% dentro de {tol1:.0f} px:         {sim1:.1f}%')
        print(f'% dentro de {tol2:.0f} px:         {sim2:.1f}%')
        print(f'Frames procesados:         {self.n_frames}')
        print(f'\nEvaluación general: {assessment}')
        print('======================================================\n')

    def plot_results(self):
        if not self.log_t:
            print('No hay datos para graficar.')
            return

        self.report_performance()

        out_dir = (Path.home() / 'Workspace' / 'sudo_drive'
                   / 'resultados' / 'pure_pursuit_vision_sim')
        out_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        tag       = f'Lfc{self.Lfc}_V{self.v_ref}_lkr{self.lookahead_rows}_sim'

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].plot(self.log_t, self.log_err, '-b', linewidth=1.2)
        axes[0].axhline(0, color='r', linestyle='--', alpha=0.5)
        axes[0].set_xlabel('Tiempo [s]')
        axes[0].set_ylabel('Error lateral [px]')
        axes[0].set_title('Error lateral vs tiempo (SIM)')
        axes[0].grid(True)

        axes[1].plot(self.log_t, self.log_steer, '-g', linewidth=1.2)
        axes[1].set_xlabel('Tiempo [s]')
        axes[1].set_ylabel('Steering [°]')
        axes[1].set_title('Steering vs tiempo (SIM)')
        axes[1].grid(True)

        axes[2].plot(self.log_t, self.log_v, '-m', linewidth=1.5)
        axes[2].set_xlabel('Tiempo [s]')
        axes[2].set_ylabel('Velocidad [m/s]')
        axes[2].set_title('Velocidad (SIM)')
        axes[2].grid(True)

        plt.tight_layout()
        fig_path = out_dir / f'analisis_{timestamp}_{tag}.png'
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f'Gráficas guardadas en: {fig_path}')

        csv_path = out_dir / f'log_{timestamp}_{tag}.csv'
        with csv_path.open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['index', 't', 'err_px', 'steer_deg', 'v_cmd'])
            for i, t in enumerate(self.log_t):
                writer.writerow([
                    i, t,
                    self.log_err[i]   if i < len(self.log_err)   else '',
                    self.log_steer[i] if i < len(self.log_steer) else '',
                    self.log_v[i]     if i < len(self.log_v)     else '',
                ])
        print(f'CSV guardado en: {csv_path}')


def main(args=None):
    rclpy.init(args=args)
    node = PurePursuitVisionNode()

    _rclpy_handler = signal.getsignal(signal.SIGINT)

    def _sigint_handler(signum, frame):
        node.stop_qcar()
        if callable(_rclpy_handler):
            _rclpy_handler(signum, frame)

    signal.signal(signal.SIGINT, _sigint_handler)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.plot_results()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
