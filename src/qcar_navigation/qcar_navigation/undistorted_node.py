#!/usr/bin/env python3
"""
=======================================================================
 Undistort Node — QCar Navigation | Sudo Drive
 Authors: Abraham Moro-Hernandez (AMH19)
-----------------------------------------------------------------------
 Subscribed Topics:
     /qcar/decompressed/csi_front  (sensor_msgs/Image)

 Published Topics:
     /amh19/undistorted/csi_front  (sensor_msgs/Image)
=======================================================================
"""

import os
import time
import cv2
import numpy as np
import yaml

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge

_QOS_LATEST = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
    durability=QoSDurabilityPolicy.VOLATILE,
)

DEFAULT_CALIB_YAML   = '/home/abrahammh19/Workspace/sudo_drive/qcar_calibration/calibration/qcar_front.yaml'
DEFAULT_CAMERA_TOPIC = '/qcar/csi_front'   # topic comprimido directo del QCar
DEFAULT_OUT_TOPIC    = '/amh19/undistorted/csi_front'


def load_calibration(path: str):
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    K = np.array(data['camera_matrix']['data'], np.float64).reshape(3, 3)
    D = np.array(data['distortion_coefficients']['data'], np.float64)
    return K, D


class UndistortNode(Node):

    def __init__(self):
        super().__init__('undistort_node')

        self.declare_parameter('calib_yaml',   DEFAULT_CALIB_YAML)
        self.declare_parameter('camera_topic', DEFAULT_CAMERA_TOPIC)
        self.declare_parameter('out_topic',    DEFAULT_OUT_TOPIC)
        self.declare_parameter('scale', 0.5)

        calib_path   = self.get_parameter('calib_yaml').get_parameter_value().string_value
        camera_topic = self.get_parameter('camera_topic').get_parameter_value().string_value
        out_topic    = self.get_parameter('out_topic').get_parameter_value().string_value
        self._scale  = float(self.get_parameter('scale').get_parameter_value().double_value)

        if not os.path.exists(calib_path):
            self.get_logger().error(f'Calibration file not found: {calib_path}')
            raise SystemExit(1)

        self.K, self.D = load_calibration(calib_path)
        self._map1    = None
        self._map2    = None
        self.bridge   = CvBridge()
        self._n       = 0
        self._t_start = time.monotonic()
        self._last_frame_t = None

        self.sub = self.create_subscription(
            CompressedImage, camera_topic, self.image_callback, _QOS_LATEST)
        self.pub_undist = self.create_publisher(
            Image, out_topic, _QOS_LATEST)

        # Watchdog: avisa si no llegan frames
        self.create_timer(5.0, self._watchdog)

        self.get_logger().info('=' * 56)
        self.get_logger().info(' UNDISTORT NODE')
        self.get_logger().info('=' * 56)
        self.get_logger().info(f'  Input : {camera_topic}')
        self.get_logger().info(f'  Output: {out_topic}')
        self.get_logger().info(f'  Scale : {self._scale}')
        self.get_logger().info('  Esperando frames...')
        self.get_logger().info('=' * 56)

    def _watchdog(self):
        if self._last_frame_t is None:
            self.get_logger().warn(
                'NO se han recibido frames aún. '
                'Verifica que el topic existe: '
                'ros2 topic hz /qcar/decompressed/csi_front')
        else:
            dt = time.monotonic() - self._last_frame_t
            if dt > 3.0:
                self.get_logger().warn(
                    f'Último frame hace {dt:.1f}s — topic puede estar caído')

    def image_callback(self, msg: CompressedImage):
        self._last_frame_t = time.monotonic()

        buf = np.frombuffer(msg.data, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            return

        h, w = img.shape[:2]

        if self._map1 is None:
            newK, roi = cv2.getOptimalNewCameraMatrix(
                self.K, self.D, (w, h), alpha=0)

            rx, ry, rw, rh = roi
            crop_w = rw if rw > 0 else w
            crop_h = rh if rh > 0 else h
            out_w  = int(crop_w * self._scale)
            out_h  = int(crop_h * self._scale)

            newK[0, 2] -= rx
            newK[1, 2] -= ry
            newK[0, :] *= out_w / crop_w
            newK[1, :] *= out_h / crop_h

            self._map1, self._map2 = cv2.initUndistortRectifyMap(
                self.K, self.D, None, newK,
                (out_w, out_h), cv2.CV_16SC2)

            self.get_logger().info(
                f'Remap listo | {w}x{h} → {out_w}x{out_h} (scale={self._scale})')

        dst = cv2.remap(img, self._map1, self._map2, cv2.INTER_LINEAR)

        out_msg = self.bridge.cv2_to_imgmsg(dst, encoding='bgr8')
        out_msg.header = msg.header
        if msg.header.stamp.sec == 0 and msg.header.stamp.nanosec == 0:
            out_msg.header.stamp = self.get_clock().now().to_msg()
        self.pub_undist.publish(out_msg)

        self._n += 1
        if self._n % 60 == 0:
            fps = self._n / (time.monotonic() - self._t_start)
            self.get_logger().info(f'FPS={fps:.1f}  res={dst.shape[1]}x{dst.shape[0]}')


def main(args=None):
    rclpy.init(args=args)
    node = UndistortNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
