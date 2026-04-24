#!/usr/bin/env python3
"""
=======================================================================
 Undistort Node — QCar Navigation | Sudo Drive
 Authors: Abraham Moro-Hernandez (AMH19)
-----------------------------------------------------------------------
 Camera Undistortion Node
 Runs on the laptop. Subscribes to the decompressed image topic
 (already handled by a separate decompression package), applies
 lens distortion correction using the calibration YAML stored
 locally in the workspace, and republishes the corrected image.

 Workflow:
     QCar        -> /qcar/compressed/csi_front      (CompressedImage)
     Decomp pkg  -> /qcar/decompressed/csi_front    (Image)
     This node   -> /amh19/undistorted/csi_front    (Image)

 Subscribed Topics:
     /qcar/decompressed/csi_front  (sensor_msgs/Image)
         Decompressed image from QCar front CSI camera

 Published Topics:
     /amh19/undistorted/csi_front  (sensor_msgs/Image)
         Undistorted BGR image ready for downstream vision processing
=======================================================================
"""

import os
import cv2
import numpy as np
import yaml

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


# ── Defaults (overridable via CLI or launch file) ──────────────────────
DEFAULT_CALIB_YAML   = '/home/abrahammh19/Workspace/sudo_drive/qcar_calibration/calibration/qcar_front.yaml'
DEFAULT_CAMERA_TOPIC = '/qcar/decompressed/csi_front'
DEFAULT_OUT_TOPIC    = '/amh19/undistorted/csi_front'


def load_calibration(path: str):
    """
    Loads intrinsic camera matrix (K) and distortion coefficients (D)
    from an OpenCV/ROS-compatible YAML calibration file.
    """
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    K = np.array(data['camera_matrix']['data'], np.float64).reshape(3, 3)
    D = np.array(data['distortion_coefficients']['data'], np.float64)
    return K, D


class UndistortNode(Node):
    """
    ROS2 node designed to run on the developer laptop.
    Receives decompressed images from the QCar over the network,
    applies lens undistortion using a local calibration file,
    and publishes the corrected image for downstream vision nodes.

    Requirements:
        - Same ROS_DOMAIN_ID on laptop and QCar
        - Both machines on the same network
        - Calibration YAML at:
          ~/Workspace/sudo_drive/qcar_calibration/calibration/qcar_front.yaml
    """

    def __init__(self):
        super().__init__('undistort_node')

        # ── ROS2 parameters ───────────────────────────────────────────────
        self.declare_parameter('calib_yaml',   DEFAULT_CALIB_YAML)
        self.declare_parameter('camera_topic', DEFAULT_CAMERA_TOPIC)
        self.declare_parameter('out_topic',    DEFAULT_OUT_TOPIC)

        calib_path   = self.get_parameter('calib_yaml').get_parameter_value().string_value
        camera_topic = self.get_parameter('camera_topic').get_parameter_value().string_value
        out_topic    = self.get_parameter('out_topic').get_parameter_value().string_value

        # ── Validate calibration file ─────────────────────────────────────
        if not os.path.exists(calib_path):
            self.get_logger().error(f'Calibration file not found: {calib_path}')
            self.get_logger().error(
                'Expected: ~/Workspace/sudo_drive/qcar_calibration/calibration/qcar_front.yaml'
            )
            raise SystemExit(1)

        # ── Load calibration ──────────────────────────────────────────────
        self.K, self.D = load_calibration(calib_path)
        self.newK      = None   # computed once on first frame
        self.roi       = None
        self.bridge    = CvBridge()
        self.n_frames  = 0

        # ── Subscriber — decompressed Image from upstream package ─────────
        self.sub = self.create_subscription(
            Image,
            camera_topic,
            self.image_callback,
            qos_profile_sensor_data
        )

        # ── Publisher — undistorted Image for downstream vision nodes ─────
        self.pub_undist = self.create_publisher(
            Image,
            out_topic,
            qos_profile_sensor_data
        )

        # ── Startup log ───────────────────────────────────────────────────
        self.get_logger().info('=' * 50)
        self.get_logger().info(' UNDISTORT NODE STARTED  [running on laptop]')
        self.get_logger().info('=' * 50)
        self.get_logger().info(f'Calibration  : {calib_path}')
        self.get_logger().info('Subscribed to:')
        self.get_logger().info(f'  {camera_topic} (sensor_msgs/Image)')
        self.get_logger().info('Publishing:')
        self.get_logger().info(f'  {out_topic} (sensor_msgs/Image)')
        self.get_logger().info('=' * 50)
        self.get_logger().info('Waiting for frames from QCar...')
        self.get_logger().info('  Check: ros2 topic hz /qcar/decompressed/csi_front')

    # ── Image callback ────────────────────────────────────────────────
    def image_callback(self, msg: Image):
        """
        Receives a decompressed Image, applies undistortion correction,
        and publishes the result preserving the original header.
        """

        # Convert ROS Image message to OpenCV BGR array
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge conversion failed: {e}')
            return

        if img is None or img.size == 0:
            self.get_logger().warn('Received empty frame, skipping.')
            return

        h, w = img.shape[:2]

        # Compute optimal new camera matrix once on the first frame
        if self.newK is None:
            self.newK, self.roi = cv2.getOptimalNewCameraMatrix(
                self.K, self.D, (w, h), alpha=0
            )
            self.get_logger().info(
                f'New camera matrix computed | resolution={w}x{h} | roi={self.roi}'
            )

        # Apply undistortion
        dst = cv2.undistort(img, self.K, self.D, None, self.newK)

        # Crop black borders using valid ROI
        x, y, rw, rh = self.roi
        if rw > 0 and rh > 0:
            dst = dst[y:y + rh, x:x + rw]

        # Publish corrected image preserving original timestamp and frame_id
        out_msg = self.bridge.cv2_to_imgmsg(dst, encoding='bgr8')
        out_msg.header = msg.header
        self.pub_undist.publish(out_msg)

        # Periodic frame count log
        self.n_frames += 1
        if self.n_frames % 30 == 0:
            self.get_logger().info(f'Frames processed: {self.n_frames}')


def main(args=None):
    rclpy.init(args=args)
    node = UndistortNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print('\n' + '=' * 50)
        print(' UNDISTORT NODE STOPPED')
        print('=' * 50)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()