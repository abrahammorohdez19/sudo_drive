#!/usr/bin/env python3
"""
=======================================================================
 Undistort Node (SIM) — QCar Navigation | Sudo Drive
 Authors: Abraham Moro-Hernandez (AMH19)
-----------------------------------------------------------------------
 Versión simulación: passthrough sin corrección de distorsión.
 La cámara simulada ya entrega imágenes sin distorsión óptica,
 por lo que este nodo simplemente retransmite la imagen al tópico
 esperado por los nodos de visión downstream.

 Subscribed Topics:
     /qcar_sim/csi_front/image_raw  (sensor_msgs/Image)

 Published Topics:
     /amh19/undistorted/csi_front   (sensor_msgs/Image)
=======================================================================
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


DEFAULT_CAMERA_TOPIC = '/qcar_sim/csi_front/image_raw'
DEFAULT_OUT_TOPIC    = '/amh19/undistorted/csi_front'


class UndistortNodeSim(Node):

    def __init__(self):
        super().__init__('undistort_node')

        self.declare_parameter('camera_topic', DEFAULT_CAMERA_TOPIC)
        self.declare_parameter('out_topic',    DEFAULT_OUT_TOPIC)

        camera_topic = self.get_parameter('camera_topic').get_parameter_value().string_value
        out_topic    = self.get_parameter('out_topic').get_parameter_value().string_value

        self.bridge   = CvBridge()
        self.n_frames = 0

        self.sub = self.create_subscription(
            Image, camera_topic, self.image_callback, qos_profile_sensor_data)
        self.pub = self.create_publisher(
            Image, out_topic, qos_profile_sensor_data)

        self.get_logger().info('=' * 50)
        self.get_logger().info(' UNDISTORT NODE (SIM) — passthrough')
        self.get_logger().info('=' * 50)
        self.get_logger().info(f'Subscribed to : {camera_topic}')
        self.get_logger().info(f'Publishing to : {out_topic}')
        self.get_logger().info('=' * 50)

    def image_callback(self, msg: Image):
        self.pub.publish(msg)
        self.n_frames += 1
        if self.n_frames % 100 == 0:
            self.get_logger().info(f'Frames pasados: {self.n_frames}')


def main(args=None):
    rclpy.init(args=args)
    node = UndistortNodeSim()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print('\n' + '=' * 50)
        print(' UNDISTORT NODE (SIM) STOPPED')
        print('=' * 50)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
