
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import Vector3Stamped

class MoveNode(Node):
    def __init__(self):
        super().__init__('move_node')
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
        self.pub = self.create_publisher(Vector3Stamped, '/qcar/user_command_raw', qos)
        self.create_timer(0.02, self.loop)
        self.get_logger().info('Moviendo QCar...')

    def loop(self):
        cmd = Vector3Stamped()
        cmd.header.stamp    = self.get_clock().now().to_msg()
        cmd.header.frame_id = 'base_link'
        cmd.vector.x        = 0.0
        cmd.vector.y        = 0.0
        cmd.vector.z        = 0.0
        self.pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = MoveNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        stop = Vector3Stamped()
        stop.header.stamp = node.get_clock().now().to_msg()
        stop.header.frame_id = 'base_link'
        node.pub.publish(stop)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()