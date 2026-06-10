#!/usr/bin/env python3
"""
=======================================================================
 Traffic Gate Node — Interconexión señales de tránsito | QCar Sudo Drive
-----------------------------------------------------------------------
 Actúa como "guardián" entre el Pure Pursuit y el hardware del QCar.
 Recibe los comandos del controlador y los reenvía o bloquea según
 el estado de la señal de tránsito detectada.

 Flujo:
   [pure_pursuit] ──> /qcar/user_command_raw
                               │
              [traffic_gate_node] <── /traffic_detection (std_msgs/String)
                               │
                    /qcar/user_command ──> [QCar físico]

 Estados reconocidos en /traffic_detection:
   "green"  → reenvía el comando sin modificar
   "yellow" → reenvía el comando reduciendo la velocidad (factor yellow_speed_factor)
   "red"    → publica velocidad 0, conserva steering
   "stop"   → publica velocidad 0, conserva steering (igual que red)
   (otros)  → reenvía sin modificar (comportamiento seguro por defecto)

 Subscriptions:
   /qcar/user_command_raw  (Vector3Stamped)  — viene del pure pursuit
   /traffic_detection      (String)          — viene del paquete de detección

 Published:
   /qcar/user_command      (Vector3Stamped)  — va al hardware del QCar
=======================================================================
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from geometry_msgs.msg import Vector3Stamped
from std_msgs.msg import String


class TrafficGateNode(Node):

    # Estados que implican detención
    STOP_STATES  = {'red', 'stop'}
    # Estados que implican reducción de velocidad
    SLOW_STATES  = {'yellow'}

    def __init__(self):
        super().__init__('interconnection_node')

        # ── Parámetros ────────────────────────────────────────────── MODIFICAR ESTO
        self.declare_parameter('input_topic',       '/qcar/user_command_raw')
        self.declare_parameter('output_topic',      '/qcar/user_command')
        self.declare_parameter('detection_topic',   '/compe/traffic_state')
        self.declare_parameter('yellow_speed_factor', 0.5)   # fracción de v_ref para amarillo
        self.declare_parameter('state_timeout',     3.0)     # segundos sin detección → ignorar estado

        p = lambda n: self.get_parameter(n).value
        input_topic       = p('input_topic')
        output_topic      = p('output_topic')
        detection_topic   = p('detection_topic')
        self.yellow_factor = float(p('yellow_speed_factor'))
        self.state_timeout = float(p('state_timeout'))

        # ── Estado interno ───────────────────────────────────────────
        self.traffic_state      = 'Green'   # estado por defecto: avanzar
        self.last_detection_time = None     # timestamp de la última detección

        # ── QoS ─────────────────────────────────────────────────────
        qos_sub = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        qos_pub = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
        
        # ── Suscripciones ────────────────────────────────────────────
        self.create_subscription(
            Vector3Stamped,
            input_topic,
            self.command_callback,
            qos_sub
        )
        self.create_subscription(
            String,
            detection_topic,
            self.detection_callback,
            QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        )

        # ── Publicador ───────────────────────────────────────────────
        self.pub = self.create_publisher(Vector3Stamped, output_topic, qos_pub)
        self.create_timer(1.0, self._republish_state)

        # ── Log de inicio ────────────────────────────────────────────
        g = self.get_logger().info
        g('=' * 62)
        g(' TRAFFIC GATE NODE  [sudo_drive — QCar físico]')
        g('=' * 62)
        g(f'  input  : {input_topic}')
        g(f'  output : {output_topic}')
        g(f'  detect : {detection_topic}')
        g(f'  yellow_speed_factor = {self.yellow_factor}')
        g(f'  state_timeout       = {self.state_timeout} s')
        g('=' * 62)

    # ────────────────────────────────────────────────────────────────
    #  CALLBACK — detección de señal de tránsito
    # ────────────────────────────────────────────────────────────────

    def detection_callback(self, msg: String):
        new_state = msg.data.strip().lower()
        self.last_detection_time = self.get_clock().now()

        if new_state != self.traffic_state:
            self.get_logger().info(
                f'[TrafficGate] Estado: {self.traffic_state.upper()} → {new_state.upper()}'
            )
            self.traffic_state = new_state

    # ────────────────────────────────────────────────────────────────
    #  CALLBACK — comando del pure pursuit
    # ────────────────────────────────────────────────────────────────

    def command_callback(self, msg: Vector3Stamped):
        state = self._effective_state()
        out   = self._apply_state(msg, state)
        self.pub.publish(out)

    # ────────────────────────────────────────────────────────────────
    #  LÓGICA DE ESTADO
    # ────────────────────────────────────────────────────────────────

    def _effective_state(self) -> str:
        if self.last_detection_time is None:
            return 'green'
        return self.traffic_state        

    def _apply_state(self, msg: Vector3Stamped, state: str) -> Vector3Stamped:
        self.get_logger().info(f'[_apply_state] state={state}  vx_in={msg.vector.x:.4f}')
        out = Vector3Stamped()
        out.header   = msg.header
        out.vector.y = msg.vector.y
        out.vector.z = msg.vector.z

        if state in self.STOP_STATES:
            out.vector.x = 0.0
            self.get_logger().info('[_apply_state] DETENIENDO')
        elif state in self.SLOW_STATES:
            out.vector.x = msg.vector.x * self.yellow_factor
        else:
            out.vector.x = msg.vector.x

        return out
    
    def _republish_state(self):
        self.get_logger().info(f'[republish] estado actual: {self.traffic_state}')

"""
    def _apply_state(self, msg: Vector3Stamped, state: str) -> Vector3Stamped:
        out = Vector3Stamped()
        out.header   = msg.header
        out.vector.y = msg.vector.y  # steering siempre pasa sin modificar
        out.vector.z = msg.vector.z

        if state in self.STOP_STATES:
            out.vector.x = 0.0

        elif state in self.SLOW_STATES:
            out.vector.x = msg.vector.x * self.yellow_factor

        else:
            out.vector.x = msg.vector.x

        return out
    """

# ════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════

def main(args=None):
    rclpy.init(args=args)
    node = TrafficGateNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()