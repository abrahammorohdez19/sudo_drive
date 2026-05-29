#!/usr/bin/env python3
"""
=======================================================================
 Pure Pursuit Controller — Vision-based | QCar Sudo Drive
 Author: Abraham Moro-Hernandez (AMH19)
-----------------------------------------------------------------------
 Basado en pure_pursuit_vision_node_sim.py — mismo algoritmo,
 topics del hardware físico.

 Algoritmo:
   y_ref    = fila inferior del ROI (img_h * roi_bottom)
   y_look   = y_ref - lookahead_rows
   x_look   = polyval([a,b,c], y_look)
   dx       = x_look - img_w/2 + lateral_offset_px + curv_offset
   alpha    = atan2(dx, lookahead_rows)
   delta    = atan2(2 * steer_gain * sin(alpha), lookahead_rows)

 Subscriptions:
   /amh19/lane/lines     (Float32MultiArray)  [a, b, c]
   /amh19/lane/centroid  (Point)
   /qcar/velocity        (Vector3Stamped)     encoder
   /qcar/obstacle_alert  (Bool)

 Published:
   /qcar/user_command    (Vector3Stamped)
       vector.x = velocidad (m/s)
       vector.y = -steering (rad)
=======================================================================
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import Vector3Stamped, Point
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

        self.declare_parameter('lookahead',           0.20)
        self.declare_parameter('k_gain',              0.50)
        self.declare_parameter('v_ref',               0.05)
        self.declare_parameter('wheelbase',           0.256)
        self.declare_parameter('max_steer',           0.50)
        self.declare_parameter('max_steer_rate',      0.05)
        self.declare_parameter('steer_alpha',         0.50)
        self.declare_parameter('steer_gain',          15.0)
        self.declare_parameter('warmup_frames',       30)
        self.declare_parameter('xlook_tol_px',       200.0)
        self.declare_parameter('startup_cap_frames',  60)
        self.declare_parameter('startup_max_steer',   0.20)
        self.declare_parameter('k_curv_offset',     4000.0)
        self.declare_parameter('img_width',           640)
        self.declare_parameter('img_height',          306)
        self.declare_parameter('roi_bottom',          0.97)
        self.declare_parameter('lookahead_rows',      60)
        self.declare_parameter('lateral_offset_px',  180.0)

        self.declare_parameter('lines_topic',    '/amh19/lane/lines')
        self.declare_parameter('centroid_topic', '/amh19/lane/centroid')
        self.declare_parameter('encoder_topic',  '/qcar/velocity')
        self.declare_parameter('cmd_topic',      '/qcar/user_command')
        self.declare_parameter('alert_topic',    '/qcar/obstacle_alert')

        p = lambda n: self.get_parameter(n).value
        self.Lfc                = p('lookahead')
        self.k_gain             = p('k_gain')
        self.v_ref              = p('v_ref')
        self.L                  = p('wheelbase')
        self.max_steer_cmd      = p('max_steer')
        self.max_steer_rate     = p('max_steer_rate')
        self.steer_alpha        = p('steer_alpha')
        self.steer_gain         = p('steer_gain')
        self.warmup_frames      = p('warmup_frames')
        self.xlook_tol_px       = p('xlook_tol_px')
        self.startup_cap_frames = p('startup_cap_frames')
        self.startup_max_steer  = p('startup_max_steer')
        self.k_curv_offset      = p('k_curv_offset')
        self.img_w              = p('img_width')
        self.img_h              = p('img_height')
        self.roi_bottom         = p('roi_bottom')
        self.lookahead_rows     = p('lookahead_rows')
        self.lateral_offset_px  = p('lateral_offset_px')

        lines_topic    = p('lines_topic')
        centroid_topic = p('centroid_topic')
        encoder_topic  = p('encoder_topic')
        cmd_topic      = p('cmd_topic')
        alert_topic    = p('alert_topic')

        self.poly         = None
        self.last_poly    = None
        self.poly_age     = 0
        self.max_poly_age = 30
        self.cx           = None
        self.last_cx      = None
        self.v_enc        = 0.0
        self.paro         = False
        self.n_frames     = 0
        self.prev_steer   = 0.0

        self.log_t      = []
        self.log_steer  = []
        self.log_err    = []
        self.log_v      = []
        self.start_time = self.get_clock().now()

        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)

        self.create_subscription(Float32MultiArray, lines_topic,    self.lines_callback,   qos)
        self.create_subscription(Point,             centroid_topic, self.centroid_callback, qos)
        self.create_subscription(Vector3Stamped,    encoder_topic,  self.encoder_callback,  qos)
        self.create_subscription(Bool,              alert_topic,    self.paro_callback,     qos)

        self.pub   = self.create_publisher(Vector3Stamped, cmd_topic, qos)
        self.timer = self.create_timer(0.02, self.control_loop)

        g = self.get_logger().info
        g('=' * 62)
        g(' PURE PURSUIT VISION NODE  [físico — basado en sim]')
        g('=' * 62)
        g(f'  v_ref={self.v_ref} m/s   steer_gain={self.steer_gain}')
        g(f'  max_steer_rate={self.max_steer_rate}  steer_alpha={self.steer_alpha}')
        g(f'  lookahead_rows={self.lookahead_rows}  lateral_offset={self.lateral_offset_px} px')
        g(f'  Imagen: {self.img_w}x{self.img_h}')
        g('=' * 62)

    # ────────────────────────────────────────────────────────────────
    #  CALLBACKS
    # ────────────────────────────────────────────────────────────────

    def lines_callback(self, msg: Float32MultiArray):
        data = msg.data
        if len(data) == 3 and not all(v == -1.0 for v in data):
            self.poly      = np.array(data, dtype=float)
            self.last_poly = self.poly.copy()
            self.poly_age  = 0
        else:
            self.poly = None

    def centroid_callback(self, msg: Point):
        self.cx      = float(msg.x)
        self.last_cx = self.cx

    def encoder_callback(self, msg: Vector3Stamped):
        self.v_enc = float(msg.vector.x)

    def paro_callback(self, msg: Bool):
        old = self.paro
        self.paro = bool(msg.data)
        if self.paro != old:
            if self.paro:
                self.get_logger().warn('Obstáculo detectado: PARO.')
            else:
                self.get_logger().info('Obstáculo despejado: reanudando.')

    # ────────────────────────────────────────────────────────────────
    #  CONTROL LOOP  50 Hz
    # ────────────────────────────────────────────────────────────────

    def control_loop(self):
        now = self.get_clock().now()

        if self.paro:
            self.stop_qcar()
            return

        self.n_frames += 1
        if self.n_frames <= self.warmup_frames:
            self.stop_qcar()
            if self.n_frames == self.warmup_frames:
                self.get_logger().info(
                    f'Warmup completado ({self.warmup_frames} frames). Iniciando control.')
            return

        current_poly = self.poly
        if current_poly is None:
            self.poly_age += 1
            if self.last_poly is not None and self.poly_age <= self.max_poly_age:
                current_poly = self.last_poly
            else:
                self.stop_qcar()
                return
        else:
            self.poly_age = 0

        current_cx = self.cx if self.cx is not None else self.last_cx
        if current_cx is None:
            self.stop_qcar()
            return

        delta, x_look_used = self.compute_pure_pursuit_delta(current_poly, current_cx)
        delta = float(np.clip(delta, -self.max_steer_cmd, self.max_steer_cmd))

        delta = self.steer_alpha * delta + (1.0 - self.steer_alpha) * self.prev_steer

        change    = float(np.clip(delta - self.prev_steer,
                                  -self.max_steer_rate, self.max_steer_rate))
        steer_cmd       = self.prev_steer + change
        self.prev_steer = steer_cmd

        frames_since_warmup = self.n_frames - self.warmup_frames
        if frames_since_warmup <= self.startup_cap_frames:
            steer_cmd = float(np.clip(steer_cmd,
                                      -self.startup_max_steer,
                                       self.startup_max_steer))

        cmd = Vector3Stamped()
        cmd.header.stamp    = now.to_msg()
        cmd.header.frame_id = 'base_link'
        cmd.vector.x        = float(self.v_ref)
        cmd.vector.y        = float(-steer_cmd)
        cmd.vector.z        = 0.0
        self.pub.publish(cmd)

        t      = (now.nanoseconds - self.start_time.nanoseconds) * 1e-9
        err_px = x_look_used - self.img_w / 2.0 + self.lateral_offset_px
        self.log_t.append(t)
        self.log_steer.append(math.degrees(steer_cmd))
        self.log_err.append(err_px)
        self.log_v.append(self.v_ref)

        if self.n_frames % 50 == 0:
            self.get_logger().info(
                f'[Frame {self.n_frames:5d}]  '
                f'cx={current_cx:.0f}px  x_look={x_look_used:.0f}px  '
                f'err={err_px:+.0f}px  '
                f'steer={math.degrees(steer_cmd):+.1f}°  v={self.v_ref:.3f} m/s')

    # ────────────────────────────────────────────────────────────────
    #  PURE PURSUIT — ESPACIO IMAGEN
    # ────────────────────────────────────────────────────────────────

    def compute_pure_pursuit_delta(self, poly, cx: float):
        y_ref  = int(self.img_h * self.roi_bottom)
        y_look = max(0, y_ref - self.lookahead_rows)
        x_look = float(np.polyval(poly, y_look))

        if not (0.0 <= x_look <= float(self.img_w)):
            x_look = cx
        elif abs(x_look - cx) > self.xlook_tol_px:
            x_look = cx

        curv_offset      = min(self.k_curv_offset * abs(float(poly[0])), 100.0)
        effective_offset = self.lateral_offset_px + curv_offset

        dx    = x_look - self.img_w / 2.0 + effective_offset
        alpha = math.atan2(dx, float(self.lookahead_rows))
        delta = math.atan2(2.0 * self.steer_gain * math.sin(alpha),
                           float(self.lookahead_rows))
        return delta, x_look

    # ────────────────────────────────────────────────────────────────
    #  STOP
    # ────────────────────────────────────────────────────────────────

    def stop_qcar(self):
        self.prev_steer = 0.0
        cmd = Vector3Stamped()
        cmd.header.stamp    = self.get_clock().now().to_msg()
        cmd.header.frame_id = 'base_link'
        cmd.vector.x        = 0.0
        cmd.vector.y        = 0.0
        cmd.vector.z        = 0.0
        self.pub.publish(cmd)

    # ────────────────────────────────────────────────────────────────
    #  ANÁLISIS Y PLOTS
    # ────────────────────────────────────────────────────────────────

    def report_performance(self):
        if not self.log_err:
            return
        errors_abs = [abs(e) for e in self.log_err]
        avg_err    = float(np.mean(errors_abs))
        max_err    = float(np.max(errors_abs))
        std_err    = float(np.std(errors_abs))
        tol1, tol2 = 15.0, 30.0
        sim1 = sum(1 for e in errors_abs if e <= tol1) / len(errors_abs) * 100
        sim2 = sum(1 for e in errors_abs if e <= tol2) / len(errors_abs) * 100
        assessment = ('EXCELENTE' if avg_err < tol1 else
                      'BUENO'     if avg_err < tol2 else
                      'ACEPTABLE' if avg_err < 60   else 'DEFICIENTE')
        print('\n========== EVALUACIÓN ==========')
        print(f'Error lateral promedio: {avg_err:.1f} px')
        print(f'Error lateral máximo:   {max_err:.1f} px')
        print(f'Desv. estándar:         {std_err:.1f} px')
        print(f'Dentro de {tol1:.0f} px: {sim1:.1f}%   Dentro de {tol2:.0f} px: {sim2:.1f}%')
        print(f'Frames: {self.n_frames}   Evaluación: {assessment}')
        print('=================================\n')

    def plot_results(self):
        if not self.log_t:
            return
        self.report_performance()
        out = (Path.home() / 'Workspace' / 'sudo_drive'
               / 'resultados' / 'pure_pursuit_vision')
        out.mkdir(parents=True, exist_ok=True)
        ts  = datetime.now().strftime('%Y%m%d_%H%M%S')
        tag = f'off{int(self.lateral_offset_px)}_gain{self.steer_gain}'

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].plot(self.log_t, self.log_err, '-b', lw=1.2, label='Error lateral')
        axes[0].axhline(0, color='r', ls='--', alpha=0.5)
        axes[0].set(title='Error lateral vs tiempo', xlabel='t [s]', ylabel='err [px]')
        axes[0].legend(); axes[0].grid(True)
        axes[1].plot(self.log_t, self.log_steer, '-g', lw=1.2)
        axes[1].set(title='Steering', xlabel='t [s]', ylabel='steer [°]')
        axes[1].grid(True)
        axes[2].plot(self.log_t, self.log_v, '-m', lw=1.5)
        axes[2].set(title='Velocidad', xlabel='t [s]', ylabel='v [m/s]')
        axes[2].grid(True)
        plt.tight_layout()
        fig.savefig(out / f'analisis_{ts}_{tag}.png', dpi=300, bbox_inches='tight')
        plt.close(fig)

        csv_path = out / f'log_{ts}_{tag}.csv'
        with csv_path.open('w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['t', 'err_px', 'steer_deg', 'v_cmd'])
            for i, t in enumerate(self.log_t):
                w.writerow([t,
                             self.log_err[i]   if i < len(self.log_err)   else '',
                             self.log_steer[i] if i < len(self.log_steer) else '',
                             self.log_v[i]     if i < len(self.log_v)     else ''])
        self.get_logger().info(f'Log guardado: {csv_path}')


# ════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════

def main(args=None):
    import time
    rclpy.init(args=args)
    node = PurePursuitVisionNode()

    def _send_stop():
        for _ in range(5):
            node.stop_qcar()
        time.sleep(0.15)

    _rclpy_handler = signal.getsignal(signal.SIGINT)

    def _sigint_handler(signum, frame):
        _send_stop()
        if callable(_rclpy_handler):
            _rclpy_handler(signum, frame)

    def _sigterm_handler(_signum, _frame):
        _send_stop()
        rclpy.shutdown()

    signal.signal(signal.SIGINT,  _sigint_handler)
    signal.signal(signal.SIGTERM, _sigterm_handler)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        _send_stop()
        node.plot_results()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
