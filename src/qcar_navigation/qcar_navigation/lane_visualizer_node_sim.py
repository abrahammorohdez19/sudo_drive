#!/usr/bin/env python3
"""
=======================================================================
 Lane Visualizer Node (SIM) — QCar Navigation | Sudo Drive
 Authors: Abraham Moro-Hernandez (AMH19)
-----------------------------------------------------------------------
 Versión simulación: idéntica a lane_visualizer_node.py.
 Los tópicos de entrada/salida son los mismos (no dependen del hardware).

 Subscribed Topics:
     /amh19/lane/lines     (std_msgs/Float32MultiArray)
     /amh19/lane/centroid  (geometry_msgs/Point)

 Published Topics:
     /amh19/rviz/lane_lines  (visualization_msgs/MarkerArray)
=======================================================================
"""

import math
import numpy as np

import rclpy
from rclpy.node import Node

from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray


DEFAULT_LINES_TOPIC    = '/amh19/lane/lines'
DEFAULT_CENTROID_TOPIC = '/amh19/lane/centroid'
DEFAULT_MARKER_TOPIC   = '/amh19/rviz/lane_lines'

DEFAULT_CAMERA_HEIGHT_M  = 0.12
DEFAULT_CAMERA_PITCH_DEG = 20.0
DEFAULT_CAMERA_FOV_H_DEG = 80.0
DEFAULT_IMAGE_WIDTH_PX   = 640
DEFAULT_IMAGE_HEIGHT_PX  = 480
DEFAULT_MARKER_LIFETIME_S = 0.2
DEFAULT_FIXED_FRAME      = 'base_link'

MARKER_NS = 'lane_detection'


class LaneVisualizerNode(Node):

    def __init__(self):
        super().__init__('lane_visualizer_node')

        self.declare_parameter('lines_topic',       DEFAULT_LINES_TOPIC)
        self.declare_parameter('centroid_topic',    DEFAULT_CENTROID_TOPIC)
        self.declare_parameter('marker_topic',      DEFAULT_MARKER_TOPIC)
        self.declare_parameter('camera_height_m',   DEFAULT_CAMERA_HEIGHT_M)
        self.declare_parameter('camera_pitch_deg',  DEFAULT_CAMERA_PITCH_DEG)
        self.declare_parameter('camera_fov_h_deg',  DEFAULT_CAMERA_FOV_H_DEG)
        self.declare_parameter('image_width_px',    DEFAULT_IMAGE_WIDTH_PX)
        self.declare_parameter('image_height_px',   DEFAULT_IMAGE_HEIGHT_PX)
        self.declare_parameter('marker_lifetime_s', DEFAULT_MARKER_LIFETIME_S)
        self.declare_parameter('fixed_frame',       DEFAULT_FIXED_FRAME)

        p = lambda name: self.get_parameter(name).value
        self.lines_topic      = p('lines_topic')
        self.centroid_topic   = p('centroid_topic')
        self.marker_topic     = p('marker_topic')
        self.camera_height    = p('camera_height_m')
        self.camera_pitch_rad = math.radians(p('camera_pitch_deg'))
        self.fov_h_rad        = math.radians(p('camera_fov_h_deg'))
        self.img_w            = p('image_width_px')
        self.img_h            = p('image_height_px')
        self.lifetime_s       = p('marker_lifetime_s')
        self.fixed_frame      = p('fixed_frame')

        self.fx = (self.img_w / 2.0) / math.tan(self.fov_h_rad / 2.0)
        self.fy = self.fx
        self.cx = self.img_w / 2.0
        self.cy = self.img_h / 2.0

        self._latest_lines    = None
        self._latest_centroid = None

        self.sub_lines = self.create_subscription(
            Float32MultiArray, self.lines_topic, self._lines_callback, 10)
        self.sub_centroid = self.create_subscription(
            Point, self.centroid_topic, self._centroid_callback, 10)

        self.pub_markers = self.create_publisher(MarkerArray, self.marker_topic, 10)
        self.pub_timer   = self.create_timer(1.0 / 15.0, self._publish_markers)

        g = self.get_logger().info
        g('=' * 60)
        g(' LANE VISUALIZER NODE (SIM) STARTED  [RViz Markers]')
        g('=' * 60)
        g(f'Subscribed : {self.lines_topic}')
        g(f'           : {self.centroid_topic}')
        g(f'Publishing : {self.marker_topic}')
        g(f'Frame      : {self.fixed_frame}')
        g('=' * 60)

    def _lines_callback(self, msg: Float32MultiArray):
        if len(msg.data) == 8:
            self._latest_lines = list(msg.data)

    def _centroid_callback(self, msg: Point):
        self._latest_centroid = (msg.x, msg.y)

    def _publish_markers(self):
        marker_array = MarkerArray()
        now = self.get_clock().now().to_msg()

        delete_all = Marker()
        delete_all.action = Marker.DELETEALL
        delete_all.ns = MARKER_NS
        marker_array.markers.append(delete_all)

        marker_id = 0

        if self._latest_lines is not None:
            lx1, ly1, lx2, ly2, rx1, ry1, rx2, ry2 = self._latest_lines
            has_left  = lx1 >= 0
            has_right = rx1 >= 0

            if has_left:
                m = self._make_lane_line_marker(
                    marker_id, now, [(lx1, ly1), (lx2, ly2)],
                    r=0.0, g=1.0, b=0.0)
                if m: marker_array.markers.append(m)
                marker_id += 1

            if has_right:
                m = self._make_lane_line_marker(
                    marker_id, now, [(rx1, ry1), (rx2, ry2)],
                    r=1.0, g=0.0, b=0.0)
                if m: marker_array.markers.append(m)
                marker_id += 1

            if has_left and has_right:
                m = self._make_lane_corridor_marker(
                    marker_id, now,
                    (lx1, ly1), (lx2, ly2), (rx1, ry1), (rx2, ry2))
                if m:
                    marker_array.markers.append(m)
                    marker_id += 1

        if self._latest_centroid is not None:
            cx_px, cy_px = self._latest_centroid
            gx, gy = self._pixel_to_ground(cx_px, cy_px)

            sphere = self._make_sphere_marker(marker_id, now, gx, gy, 0.04,
                                              r=1.0, g=1.0, b=0.0)
            marker_array.markers.append(sphere)
            marker_id += 1

            arrow = self._make_error_arrow_marker(marker_id, now, gx, gy)
            marker_array.markers.append(arrow)
            marker_id += 1

            label = self._make_text_marker(
                marker_id, now, gx, gy + 0.05, f'err: {gy:+.3f}m')
            marker_array.markers.append(label)
            marker_id += 1

        self.pub_markers.publish(marker_array)

    def _pixel_to_ground(self, u: float, v: float):
        alpha    = math.atan2(v - self.cy, self.fy)
        theta_vert = self.camera_pitch_rad - alpha
        if theta_vert <= 0:
            return 1.0, 0.0
        x_fwd = self.camera_height / math.tan(theta_vert)
        beta  = math.atan2(u - self.cx, self.fx)
        y_lat = -x_fwd * math.tan(beta)
        x_fwd = max(0.1, min(x_fwd, 5.0))
        y_lat = max(-2.0, min(y_lat, 2.0))
        return x_fwd, y_lat

    def _base_marker(self, marker_id: int, now, mtype: int) -> Marker:
        m = Marker()
        m.header.frame_id = self.fixed_frame
        m.header.stamp    = now
        m.ns              = MARKER_NS
        m.id              = marker_id
        m.type            = mtype
        m.action          = Marker.ADD
        lt = self.lifetime_s
        m.lifetime.sec     = int(lt)
        m.lifetime.nanosec = int((lt - int(lt)) * 1e9)
        return m

    def _make_lane_line_marker(self, mid, now, px_pts, r, g, b):
        m = self._base_marker(mid, now, Marker.LINE_STRIP)
        m.scale.x = 0.03
        m.color.r = r; m.color.g = g; m.color.b = b; m.color.a = 0.9
        for (u, v) in px_pts:
            gx, gy = self._pixel_to_ground(u, v)
            p = Point(); p.x = gx; p.y = gy; p.z = 0.01
            m.points.append(p)
        return m if len(m.points) >= 2 else None

    def _make_lane_corridor_marker(self, mid, now, lp1, lp2, rp1, rp2):
        m = self._base_marker(mid, now, Marker.TRIANGLE_LIST)
        m.scale.x = 1.0; m.scale.y = 1.0; m.scale.z = 1.0
        m.color.r = 0.0; m.color.g = 0.8; m.color.b = 0.8; m.color.a = 0.15
        corners_3d = []
        for (u, v) in [lp1, lp2, rp2, rp1]:
            gx, gy = self._pixel_to_ground(u, v)
            p = Point(); p.x = gx; p.y = gy; p.z = 0.005
            corners_3d.append(p)
        m.points.extend([corners_3d[0], corners_3d[1], corners_3d[2]])
        m.points.extend([corners_3d[0], corners_3d[2], corners_3d[3]])
        return m

    def _make_sphere_marker(self, mid, now, x, y, size, r, g, b):
        m = self._base_marker(mid, now, Marker.SPHERE)
        m.pose.position.x = x; m.pose.position.y = y; m.pose.position.z = 0.05
        m.pose.orientation.w = 1.0
        m.scale.x = size; m.scale.y = size; m.scale.z = size
        m.color.r = r; m.color.g = g; m.color.b = b; m.color.a = 1.0
        return m

    def _make_error_arrow_marker(self, mid, now, x_fwd, y_lat):
        m = self._base_marker(mid, now, Marker.ARROW)
        m.scale.x = 0.02; m.scale.y = 0.04; m.scale.z = 0.06
        m.color.r = 0.2; m.color.g = 0.5; m.color.b = 1.0; m.color.a = 0.85
        start = Point(); start.x = x_fwd; start.y = 0.0;   start.z = 0.05
        end   = Point(); end.x   = x_fwd; end.y   = y_lat; end.z   = 0.05
        m.points = [start, end]
        return m

    def _make_text_marker(self, mid, now, x, y, text):
        m = self._base_marker(mid, now, Marker.TEXT_VIEW_FACING)
        m.pose.position.x = x; m.pose.position.y = y; m.pose.position.z = 0.15
        m.pose.orientation.w = 1.0
        m.scale.z = 0.08
        m.color.r = 1.0; m.color.g = 1.0; m.color.b = 1.0; m.color.a = 1.0
        m.text = text
        return m


def main(args=None):
    rclpy.init(args=args)
    node = LaneVisualizerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print('\n' + '=' * 60)
        print(' LANE VISUALIZER NODE (SIM) STOPPED')
        print('=' * 60)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
