#!/usr/bin/env python3
"""
Captura imágenes del QCar1 para calibración
Ejecutar en el Jetson vía SSH con: python3 collect_images.py
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os

# ── AJUSTA ESTE TOPIC al que viste en ros2 topic list ──
CAMERA_TOPIC = '/qcar/csi_front'
SAVE_DIR     = '/home/nvidia/calibration/images'

class ImageCollector(Node):
    def __init__(self):
        super().__init__('image_collector')
        os.makedirs(SAVE_DIR, exist_ok=True)
        self.bridge = CvBridge()
        self.count  = 0
        self.latest_frame = None

        self.sub = self.create_subscription(
            Image, CAMERA_TOPIC, self.cb, 10)

        # Timer para mostrar preview cada 100ms
        self.timer = self.create_timer(0.1, self.show_preview)
        self.get_logger().info(f'Escuchando: {CAMERA_TOPIC}')
        self.get_logger().info('SPACE = capturar | Q = salir | necesitas 25-30 fotos')

    def cb(self, msg):
        try:
            self.latest_frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'CvBridge error: {e}')

    def show_preview(self):
        if self.latest_frame is None:
            return

        frame = self.latest_frame.copy()
        # Overlay con instrucciones
        cv2.putText(frame, f'Capturas: {self.count}/25  |  SPACE=foto  Q=salir',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('QCar1 - Calibracion', frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            path = f'{SAVE_DIR}/img_{self.count:03d}.png'
            cv2.imwrite(path, self.latest_frame)
            self.count += 1
            self.get_logger().info(f'[{self.count}] Guardada: {path}')

        elif key == ord('q'):
            self.get_logger().info(f'Listo. {self.count} imágenes en {SAVE_DIR}')
            cv2.destroyAllWindows()
            rclpy.shutdown()

def main():
    rclpy.init()
    node = ImageCollector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()