#!/usr/bin/env python3
"""
=======================================================================
 Supervisor Node — QCar Sudo Drive
 Author: Natalia Montserrat Cruz Pérez
-----------------------------------------------------------------------
 Árbitro centralizado de señales de parada.
 Lee:
   /qcar/obstacle_alert     (Bool)         — nodo obstacle_detection
   /compe/traffic_state     (String)       — nodo de visión
   /amh19/pp/cmd_raw        (Vector3Stamped) — salida del Pure Pursuit

 Publica:
   /qcar/user_command       (Vector3Stamped) — al hardware

 Lógica:
   Si obstacle_alert=True  → v=0, steer=0
   Si traffic_state ∈ {'stop','red'} → v=0, steer mantenido
   Si traffic_state = 'yellow' → v *= factor_amarillo
   Si todo libre → relay directo del Pure Pursuit
=======================================================================
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import Vector3Stamped
from std_msgs.msg import Bool, String
import time


# Estados de tráfico que implican parada total
STOP_STATES  = {'stop', 'red'}
# Estados que implican reducción de velocidad
SLOW_STATES  = {'yellow'}


class SupervisorNode(Node):

    def __init__(self):
        super().__init__('supervisor_node')

        # ── Parámetros ────────────────────────────────────────────
        self.declare_parameter('pp_cmd_topic',      '/qcar/user_command_raw')
        self.declare_parameter('obstacle_topic',    '/qcar/obstacle_alert')
        self.declare_parameter('traffic_topic',     '/compe/traffic_state')
        self.declare_parameter('output_topic',      '/qcar/user_command')
        self.declare_parameter('yellow_factor',     0.5)   # vel * factor en amarillo
        self.declare_parameter('traffic_timeout_s', 1.0)   # s sin msg antes de ignorar estado

        p = lambda n: self.get_parameter(n).value
        self.yellow_factor      = p('yellow_factor')
        self.traffic_timeout_s  = p('traffic_timeout_s')

        # ── Estado interno ────────────────────────────────────────
        self.obstacle_stop   = False          # señal de obstacle_detection
        self.traffic_state   = 'free'         # último estado de visión
        self.traffic_stamp   = 0.0            # tiempo del último msg de visión
        self.last_pp_cmd     = None           # último comando del Pure Pursuit

        # ── QoS ──────────────────────────────────────────────────
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)

        # ── Subscribers ───────────────────────────────────────────
        self.create_subscription(
            Bool, p('obstacle_topic'), self._obstacle_cb, qos)
        self.create_subscription(
            String, p('traffic_topic'), self._traffic_cb, qos)
        self.create_subscription(
            Vector3Stamped, p('pp_cmd_topic'), self._pp_cmd_cb, qos)

        # ── Publisher ─────────────────────────────────────────────
        self.pub = self.create_publisher(Vector3Stamped, p('output_topic'), qos)

        # ── Timer de supervisión a 50 Hz ──────────────────────────
        self.timer = self.create_timer(0.02, self._supervise)

        self.get_logger().info('Supervisor node activo.')
        self.get_logger().info(
            f'  Escuchando PP en: {p("pp_cmd_topic")}')
        self.get_logger().info(
            f'  Publicando en:   {p("output_topic")}')

    # ── CALLBACKS ─────────────────────────────────────────────────

    def _obstacle_cb(self, msg: Bool):
        prev = self.obstacle_stop
        self.obstacle_stop = bool(msg.data)
        if self.obstacle_stop != prev:
            level = self.get_logger().warn
            level(f'obstacle_alert → {self.obstacle_stop}')

    def _traffic_cb(self, msg: String):
        state = msg.data.strip().lower()
        self.traffic_state = state
        self.traffic_stamp = time.monotonic()

        if state in STOP_STATES:
            self.get_logger().warn(f'traffic_state={state} → PARADA')
        elif state in SLOW_STATES:
            self.get_logger().info(f'traffic_state={state} → reduciendo velocidad')

    def _pp_cmd_cb(self, msg: Vector3Stamped):
        self.last_pp_cmd = msg

    # ── LÓGICA DE SUPERVISIÓN ─────────────────────────────────────

    def _supervise(self):
        now = time.monotonic()

        # Sin comando del Pure Pursuit → silencio (no publicar basura)
        if self.last_pp_cmd is None:
            return

        cmd = Vector3Stamped()
        cmd.header.stamp    = self.get_clock().now().to_msg()
        cmd.header.frame_id = 'base_link'

        # ── Prioridad 1: obstacle físico (más urgente) ─────────────
        if self.obstacle_stop:
            cmd.vector.x = 0.0
            cmd.vector.y = 0.0
            self.pub.publish(cmd)
            return

        # ── Prioridad 2: estado de tráfico actual (si msg es fresco) ─
        traffic_fresh = (now - self.traffic_stamp) < self.traffic_timeout_s

        if traffic_fresh and self.traffic_state in STOP_STATES:
            cmd.vector.x = 0.0
            cmd.vector.y = self.last_pp_cmd.vector.y
            self.pub.publish(cmd)
            return

        if traffic_fresh and self.traffic_state in SLOW_STATES:
            cmd.vector.x = self.last_pp_cmd.vector.x * self.yellow_factor
            cmd.vector.y = self.last_pp_cmd.vector.y
            self.pub.publish(cmd)
            return

        # ── Sin razón de parada: relay directo ────────────────────
        self.pub.publish(self.last_pp_cmd)


# ── MAIN ──────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = SupervisorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()