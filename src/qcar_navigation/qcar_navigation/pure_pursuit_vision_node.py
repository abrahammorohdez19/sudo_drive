#!/usr/bin/env python3
"""
=======================================================================
 Pure Pursuit Controller — Vision-based | QCar Sudo Drive  v2
 Author: Abraham Moro-Hernandez (AMH19)
-----------------------------------------------------------------------
 Línea amarilla = BORDE IZQUIERDO del carril derecho.
 El QCar debe circular a la DERECHA de la línea amarilla.

 Target point:
   x_target = x_line + lane_offset_px   (a la derecha de la línea)
   dx       = x_target - img_w / 2      (error respecto al centro)

 Mejoras v2:
   - Lookahead adaptativo por curvatura (máx en recta, mín en curva)
   - Velocidad adaptativa (nominal en recta, reduce en curvas)
   - Anticipación y boost de steering en curvas izquierda
   - Log detallado cada N frames para ajuste experimental

 Subscriptions:
   /amh19/lane/lines     (Float32MultiArray)  [a, b, c] poly
   /amh19/lane/centroid  (Point)              centroide línea
   /qcar/velocity        (Vector3Stamped)     encoder
   /qcar/obstacle_alert  (Bool)               paro de emergencia

 Published:
   /qcar/user_command    (Vector3Stamped)
       vector.x = velocidad (m/s)
       vector.y = -steering (rad)
=======================================================================
"""

import math
import csv
import signal
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import Vector3Stamped, Point
from std_msgs.msg import Bool, Float32MultiArray


class PurePursuitVisionNode(Node):

    def __init__(self):
        super().__init__('pure_pursuit_vision_node')

        # ── Velocidad adaptativa ──────────────────────────────────────
        self.declare_parameter('nominal_speed',           0.055)
        self.declare_parameter('min_speed',               0.042)
        self.declare_parameter('max_speed',               0.065)
        self.declare_parameter('curvature_slowdown_gain', 1.0)

        # ── Steering ──────────────────────────────────────────────────
        self.declare_parameter('max_steer',       0.50)
        self.declare_parameter('max_steer_rate',  0.10)
        self.declare_parameter('steer_alpha',     0.65)
        self.declare_parameter('steer_gain',      11.0)

        # ── Lookahead adaptativo ──────────────────────────────────────
        self.declare_parameter('lookahead_rows_min',  45)
        self.declare_parameter('lookahead_rows_max',  80)
        self.declare_parameter('k_curv_lookahead',   1200.0)

        # ── Lane offset ───────────────────────────────────────────────
        # Desplazamiento a la DERECHA de la línea amarilla.
        # Sube → más separación de la línea.  Baja → más cerca.
        self.declare_parameter('lane_offset_px',   160.0)

        # ── Anticipación curva izquierda ──────────────────────────────
        self.declare_parameter('left_curve_threshold',          0.003)
        self.declare_parameter('left_curve_lookahead_reduction', 10)
        self.declare_parameter('left_curve_steer_boost',        0.15)

        # ── Imagen ────────────────────────────────────────────────────
        self.declare_parameter('img_width',    640)
        self.declare_parameter('img_height',   480)
        self.declare_parameter('roi_bottom',   0.97)
        self.declare_parameter('xlook_tol_px', 200.0)
        self.declare_parameter('max_poly_age', 15)
        self.declare_parameter('warmup_frames',       30)
        self.declare_parameter('startup_cap_frames',  40)
        self.declare_parameter('startup_max_steer',   0.15)

        # ── Topics ────────────────────────────────────────────────────
        self.declare_parameter('lines_topic',    '/amh19/lane/lines')
        self.declare_parameter('centroid_topic', '/amh19/lane/centroid')
        self.declare_parameter('encoder_topic',  '/qcar/velocity')
        self.declare_parameter('cmd_topic',      '/qcar/user_command')
        self.declare_parameter('alert_topic',    '/qcar/obstacle_alert')

        p = lambda n: self.get_parameter(n).value

        self.nominal_speed      = float(p('nominal_speed'))
        self.min_speed          = float(p('min_speed'))
        self.max_speed          = float(p('max_speed'))
        self.curv_slowdown      = float(p('curvature_slowdown_gain'))
        self.max_steer_cmd      = float(p('max_steer'))
        self.max_steer_rate     = float(p('max_steer_rate'))
        self.steer_alpha        = float(p('steer_alpha'))
        self.steer_gain         = float(p('steer_gain'))
        self.look_rows_min      = int(p('lookahead_rows_min'))
        self.look_rows_max      = int(p('lookahead_rows_max'))
        self.k_curv_look        = float(p('k_curv_lookahead'))
        self.lane_offset_px     = float(p('lane_offset_px'))
        self.left_curv_thresh   = float(p('left_curve_threshold'))
        self.left_curv_look_red = int(p('left_curve_lookahead_reduction'))
        self.left_curv_boost    = float(p('left_curve_steer_boost'))
        self.img_w              = int(p('img_width'))
        self.img_h              = int(p('img_height'))
        self.roi_bottom         = float(p('roi_bottom'))
        self.xlook_tol_px       = float(p('xlook_tol_px'))
        self.max_poly_age       = int(p('max_poly_age'))
        self.warmup_frames      = int(p('warmup_frames'))
        self.startup_cap_frames = int(p('startup_cap_frames'))
        self.startup_max_steer  = float(p('startup_max_steer'))

        # ── Estado ────────────────────────────────────────────────────
        self.poly       = None
        self.last_poly  = None
        self.poly_age   = 0
        self.cx         = None
        self.cy         = None
        self.last_cx    = None
        self.last_cy    = None
        self.v_enc      = 0.0
        self.paro       = False
        self.n_frames        = 0
        self.prev_steer      = 0.0
        self._warmup_done    = False
        self._valid_poly_cnt = 0

        self.log_t      = []
        self.log_steer  = []
        self.log_err    = []
        self.log_v      = []
        self.start_time = self.get_clock().now()

        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)

        self.create_subscription(Float32MultiArray, p('lines_topic'),
                                 self._cb_lines,    qos)
        self.create_subscription(Point,             p('centroid_topic'),
                                 self._cb_centroid, qos)
        self.create_subscription(Vector3Stamped,    p('encoder_topic'),
                                 self._cb_encoder,  qos)
        self.create_subscription(Bool,              p('alert_topic'),
                                 self._cb_paro,     qos)

        self.pub   = self.create_publisher(Vector3Stamped, p('cmd_topic'), qos)
        self.timer = self.create_timer(0.02, self._control_loop)

        g = self.get_logger().info
        g('=' * 64)
        g(' PURE PURSUIT VISION  v2 — carril derecho, línea amarilla')
        g('=' * 64)
        g(f'  Speed   nominal={self.nominal_speed}  min={self.min_speed}  max={self.max_speed}')
        g(f'  Steer   gain={self.steer_gain}  rate={self.max_steer_rate}  alpha={self.steer_alpha}')
        g(f'  Look    [{self.look_rows_min}..{self.look_rows_max}] rows  k={self.k_curv_look}')
        g(f'  Offset  {self.lane_offset_px:.0f} px a la derecha de la línea')
        g(f'  LCurve  thresh={self.left_curv_thresh}  red={self.left_curv_look_red}  boost={self.left_curv_boost}')
        g('=' * 64)

    # ════════════════════════════════════════════════════════════════
    #  CALLBACKS
    # ════════════════════════════════════════════════════════════════

    def _cb_lines(self, msg: Float32MultiArray):
        data = msg.data
        if len(data) == 3 and not all(v == -1.0 for v in data):
            self.poly      = np.array(data, dtype=float)
            self.last_poly = self.poly.copy()
            self.poly_age  = 0
        else:
            self.poly = None

    def _cb_centroid(self, msg: Point):
        self.cx      = float(msg.x)
        self.cy      = float(msg.y)
        self.last_cx = self.cx
        self.last_cy = self.cy

    def _cb_encoder(self, msg: Vector3Stamped):
        self.v_enc = float(msg.vector.x)

    def _cb_paro(self, msg: Bool):
        prev = self.paro
        self.paro = bool(msg.data)
        if self.paro != prev:
            if self.paro:
                self.get_logger().warn('Obstáculo: PARO activado.')
            else:
                self.get_logger().info('Obstáculo despejado: reanudando.')

    # ════════════════════════════════════════════════════════════════
    #  CONTROL LOOP  (50 Hz)
    # ════════════════════════════════════════════════════════════════

    def _control_loop(self):
        now = self.get_clock().now()

        if self.paro:
            self.stop_qcar()
            return

        self.n_frames += 1

        if self.poly is not None:
            self._valid_poly_cnt += 1

        # Warmup: esperar tiempo mínimo Y al menos 20 polys válidos
        if not self._warmup_done:
            self.stop_qcar()
            ready = (self.n_frames >= self.warmup_frames
                     and self._valid_poly_cnt >= 20)
            if ready:
                self._warmup_done = True
                self.get_logger().info(
                    f'Warmup completo — {self.n_frames} frames, '
                    f'{self._valid_poly_cnt} polys válidos. Iniciando control.')
            else:
                if self.n_frames % 25 == 0:
                    self.get_logger().info(
                        f'Warmup: {self.n_frames}/{self.warmup_frames} frames, '
                        f'{self._valid_poly_cnt}/20 polys...')
            return

        # Polinomio activo (con fallback a último válido)
        poly = self.poly
        if poly is None:
            self.poly_age += 1
            if self.last_poly is not None and self.poly_age <= self.max_poly_age:
                poly = self.last_poly
            else:
                self.stop_qcar()
                return
        else:
            self.poly_age = 0

        cx = self.cx if self.cx is not None else self.last_cx
        cy = self.cy if self.cy is not None else self.last_cy
        if cx is None or cy is None:
            self.stop_qcar()
            return

        delta, x_look, x_target, look_rows, curv, is_left = \
            self._compute_delta(poly, cx, cy)

        # Velocidad adaptativa: reduce en curvas
        speed = float(np.clip(
            self.nominal_speed - self.curv_slowdown * curv,
            self.min_speed, self.max_speed))

        # Clip → EMA → rate limiter
        delta     = float(np.clip(delta, -self.max_steer_cmd, self.max_steer_cmd))
        delta_ema = self.steer_alpha * delta + (1.0 - self.steer_alpha) * self.prev_steer
        change    = float(np.clip(delta_ema - self.prev_steer,
                                  -self.max_steer_rate, self.max_steer_rate))
        steer_cmd       = self.prev_steer + change
        self.prev_steer = steer_cmd

        # Cap de steering durante los primeros frames post-warmup
        frames_since_warmup = self.n_frames - self.warmup_frames
        if 0 < frames_since_warmup <= self.startup_cap_frames:
            steer_cmd = float(np.clip(steer_cmd,
                                      -self.startup_max_steer,
                                       self.startup_max_steer))

        cmd = Vector3Stamped()
        cmd.header.stamp    = now.to_msg()
        cmd.header.frame_id = 'base_link'
        cmd.vector.x        = speed
        cmd.vector.y        = float(-steer_cmd)
        self.pub.publish(cmd)

        dx = x_target - self.img_w / 2.0
        t  = (now.nanoseconds - self.start_time.nanoseconds) * 1e-9
        self.log_t.append(t)
        self.log_steer.append(math.degrees(steer_cmd))
        self.log_err.append(dx)
        self.log_v.append(speed)

        if self.n_frames % 50 == 0:
            self.get_logger().info(
                f'[{self.n_frames:5d}]  '
                f'x_line={cx:.0f}  x_look={x_look:.0f}  x_tgt={x_target:.0f}  '
                f'dx={dx:+.0f}px  '
                f'lkr={look_rows}  curv={curv:.4f}({"L" if is_left else "R"})  '
                f'spd={speed:.3f}  '
                f'raw={math.degrees(delta):+.1f}°  '
                f'out={math.degrees(steer_cmd):+.1f}°')

    # ════════════════════════════════════════════════════════════════
    #  PURE PURSUIT EN ESPACIO IMAGEN
    # ════════════════════════════════════════════════════════════════

    def _compute_delta(self, poly, cx: float, cy: float):
        curv    = abs(float(poly[0]))
        is_left = float(poly[0]) < -self.left_curv_thresh

        # Lookahead adaptativo: más largo en recta, más corto en curva
        look_rows = int(np.clip(
            self.look_rows_max - self.k_curv_look * curv,
            self.look_rows_min, self.look_rows_max))

        # Reducción extra en curva izquierda → anticipa antes
        if is_left:
            look_rows = max(self.look_rows_min,
                            look_rows - self.left_curv_look_red)

        y_ref  = int(cy)
        y_look = max(0, y_ref - look_rows)
        x_look = float(np.polyval(poly, y_look))

        if not (0.0 <= x_look <= float(self.img_w)):
            x_look = cx
        elif abs(x_look - cx) > self.xlook_tol_px:
            x_look = cx

        # Target: DERECHA de la línea amarilla
        x_target = x_look + self.lane_offset_px
        x_target = float(np.clip(x_target, 0.0, float(self.img_w)))

        dx    = x_target - self.img_w / 2.0
        alpha = math.atan2(dx, float(look_rows))
        delta = math.atan2(2.0 * self.steer_gain * math.sin(alpha),
                           float(look_rows))

        # Boost de steering en curva izquierda pronunciada
        if is_left and curv > self.left_curv_thresh:
            delta = float(np.clip(
                delta * (1.0 + self.left_curv_boost),
                -self.max_steer_cmd, self.max_steer_cmd))

        return delta, x_look, x_target, look_rows, curv, is_left

    # ════════════════════════════════════════════════════════════════
    #  STOP
    # ════════════════════════════════════════════════════════════════

    def stop_qcar(self):
        self.prev_steer = 0.0
        cmd = Vector3Stamped()
        cmd.header.stamp    = self.get_clock().now().to_msg()
        cmd.header.frame_id = 'base_link'
        cmd.vector.x        = 0.0
        cmd.vector.y        = 0.0
        self.pub.publish(cmd)

    # ════════════════════════════════════════════════════════════════
    #  ANÁLISIS Y PLOT  (al salir)
    # ════════════════════════════════════════════════════════════════

    def report_performance(self):
        if not self.log_err:
            return
        errs   = [abs(e) for e in self.log_err]
        avg    = float(np.mean(errs))
        mx     = float(np.max(errs))
        grade  = ('EXCELENTE' if avg < 15 else
                  'BUENO'     if avg < 30 else
                  'ACEPTABLE' if avg < 60 else 'DEFICIENTE')
        print('\n========== EVALUACIÓN ==========')
        print(f'Error lateral promedio: {avg:.1f} px')
        print(f'Error lateral máximo:   {mx:.1f} px')
        print(f'Frames totales:         {self.n_frames}')
        print(f'Evaluación:             {grade}')
        print('=================================\n')

    def plot_results(self):
        if not self.log_t:
            return
        self.report_performance()
        out = (Path.home() / 'Workspace' / 'sudo_drive'
               / 'resultados' / 'pure_pursuit_vision')
        out.mkdir(parents=True, exist_ok=True)
        ts  = datetime.now().strftime('%Y%m%d_%H%M%S')
        tag = f'spd{self.nominal_speed}_off{int(self.lane_offset_px)}'

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].plot(self.log_t, self.log_err, '-b', lw=1.2)
        axes[0].axhline(0, color='r', ls='--', alpha=0.5)
        axes[0].set(title='Error lateral dx', xlabel='t [s]', ylabel='dx [px]')
        axes[0].grid(True)

        axes[1].plot(self.log_t, self.log_steer, '-g', lw=1.2)
        axes[1].set(title='Steering', xlabel='t [s]', ylabel='steer [°]')
        axes[1].grid(True)

        axes[2].plot(self.log_t, self.log_v, '-m', lw=1.5)
        axes[2].set(title='Velocidad comando', xlabel='t [s]', ylabel='v [m/s]')
        axes[2].grid(True)

        plt.tight_layout()
        fig.savefig(out / f'analisis_{ts}_{tag}.png', dpi=300, bbox_inches='tight')
        plt.close(fig)

        csv_path = out / f'log_{ts}_{tag}.csv'
        with csv_path.open('w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['t', 'dx_px', 'steer_deg', 'v_cmd'])
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
