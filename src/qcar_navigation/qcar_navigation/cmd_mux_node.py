#!/usr/bin/env python3
"""
=======================================================================
 Command Mux Node — Sudo Drive QCar
 Author: Abraham Moro-Hernandez (AMH19)
-----------------------------------------------------------------------
 VISION primary / TRAJECTORY fallback con dead-reckoning en curvas.

 Cuando visión pierde la línea:
   1. Dead-reckoning: sigue con el ÚLTIMO comando de visión válido
      (mismo steering + velocidad) mientras no pase timeout_cycles.
   2. Fallback: después de timeout_cycles sin visión activa, cambia
      a trayectoria grabada.
   3. Recovery: cuando visión recupera la línea (vision_confirm frames
      consecutivos), vuelve a visión inmediatamente.

 "vision activa" = último cmd de visión tenía v > 0.
=======================================================================
"""

import signal
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3Stamped
from std_msgs.msg import String, Bool


class CmdMuxNode(Node):

    def __init__(self):
        super().__init__('cmd_mux_node')

        self.declare_parameter('timeout_cycles',        60)
        self.declare_parameter('vision_confirm',         5)
        self.declare_parameter('control_hz',            50.0)
        self.declare_parameter('flip_trajectory_steer', True)

        self._timeout = self.get_parameter('timeout_cycles').value
        self._confirm = self.get_parameter('vision_confirm').value
        hz            = self.get_parameter('control_hz').value
        self._flip    = self.get_parameter('flip_trajectory_steer').value

        self._cmd_vision          = None
        self._cmd_trajectory      = None
        self._last_good_vision    = None   # último cmd con v > 0
        self._vision_alive        = False
        self._last_vision_stamp   = None   # watchdog: cuándo llegó el último msg de vision
        self._obstacle            = False  # paro de emergencia — bloquea TODAS las fuentes

        self._no_vision_count = 0
        self._vision_count    = 0
        self._source          = 'vision'
        self._vision_watchdog_sec = 2.0    # s sin mensaje de vision → limpia dead-reckoning

        self.create_subscription(
            Vector3Stamped, '/qcar/cmd_vision',
            self._cb_vision, 10)
        self.create_subscription(
            Vector3Stamped, '/qcar/cmd_trajectory',
            self._cb_trajectory, 10)
        self.create_subscription(
            Bool, '/qcar/obstacle_alert',
            self._cb_obstacle, 10)

        self._pub_cmd = self.create_publisher(
            Vector3Stamped, '/qcar/user_command', 10)
        self._pub_src = self.create_publisher(
            String, '/qcar/mux_source', 10)

        self.create_timer(1.0 / hz, self._loop)

        self.get_logger().info('=' * 60)
        self.get_logger().info(' CMD MUX  vision→primary / dead-reckoning / trajectory→fallback')
        self.get_logger().info(f' timeout={self._timeout}  confirm={self._confirm}  flip={self._flip}')
        self.get_logger().info('=' * 60)

    def _cb_vision(self, msg: Vector3Stamped):
        self._cmd_vision     = msg
        self._vision_alive   = msg.vector.x > 0.0
        self._last_vision_stamp = self.get_clock().now()
        if self._vision_alive:
            self._last_good_vision = msg   # guardar último comando válido

    def _cb_trajectory(self, msg: Vector3Stamped):
        self._cmd_trajectory = msg

    def _cb_obstacle(self, msg: Bool):
        prev = self._obstacle
        self._obstacle = bool(msg.data)
        if self._obstacle != prev:
            if self._obstacle:
                self.get_logger().warn('MUX: OBSTACLE → paro de emergencia en TODAS las fuentes')
            else:
                self.get_logger().info('MUX: obstáculo despejado — reanudando')

    def _loop(self):
        # Watchdog: si el nodo de vision murió sin enviar v=0, limpiar dead-reckoning
        if self._last_vision_stamp is not None:
            age = (self.get_clock().now() - self._last_vision_stamp).nanoseconds * 1e-9
            if age > self._vision_watchdog_sec:
                if self._last_good_vision is not None:
                    self.get_logger().warn(
                        f'Vision watchdog: sin mensaje {age:.1f}s — limpiando dead-reckoning')
                self._vision_alive      = False
                self._last_good_vision  = None
                self._last_vision_stamp = None  # no repetir el warn

        # Contadores de histéresis
        if self._vision_alive:
            self._vision_count    += 1
            self._no_vision_count  = 0
        else:
            self._no_vision_count += 1
            self._vision_count     = 0

        # Transiciones de estado
        if self._source == 'vision':
            if self._no_vision_count >= self._timeout:
                self._source = 'trajectory'
                self.get_logger().warn(
                    f'Vision lost {self._timeout} cycles — TRAJECTORY fallback')
        else:
            if self._vision_count >= self._confirm:
                self._source = 'vision'
                self.get_logger().info(
                    f'Vision recovered ({self._confirm} frames) — back to VISION')

        # Paro de emergencia tiene prioridad absoluta sobre cualquier fuente
        if self._obstacle:
            self._pub_cmd.publish(self._zero_cmd())
            src = String(); src.data = 'obstacle_stop'
            self._pub_src.publish(src)
            return

        # Seleccionar comando
        if self._source == 'vision':
            if self._vision_alive and self._cmd_vision is not None:
                # Visión activa: usar comando directo
                cmd = self._cmd_vision
            elif self._last_good_vision is not None:
                # Visión perdida temporalmente: dead-reckoning con último comando válido
                cmd = self._last_good_vision
            else:
                cmd = self._zero_cmd()
        else:
            # Fallback a trayectoria
            if self._cmd_trajectory is not None:
                raw = self._cmd_trajectory
                if self._flip:
                    cmd = Vector3Stamped()
                    cmd.header   = raw.header
                    cmd.vector.x = raw.vector.x
                    cmd.vector.y = -raw.vector.y
                    cmd.vector.z = raw.vector.z
                else:
                    cmd = raw
            else:
                cmd = self._zero_cmd()

        self._pub_cmd.publish(cmd)

        src = String()
        src.data = self._source
        self._pub_src.publish(src)

    def _zero_cmd(self) -> Vector3Stamped:
        msg = Vector3Stamped()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        return msg


def main(args=None):
    import time

    rclpy.init(args=args)
    node = CmdMuxNode()

    def _send_stop():
        """Publica zero varias veces y espera a que DDS lo flush."""
        for _ in range(5):
            node._pub_cmd.publish(node._zero_cmd())
        time.sleep(0.15)   # da tiempo al DDS para entregar antes de morir

    _rclpy_handler = signal.getsignal(signal.SIGINT)

    def _sigint_handler(signum, frame):
        _send_stop()
        if callable(_rclpy_handler):
            _rclpy_handler(signum, frame)

    def _sigterm_handler(signum, frame):
        # ros2 launch manda SIGTERM a los hijos — mismo tratamiento que SIGINT
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
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
