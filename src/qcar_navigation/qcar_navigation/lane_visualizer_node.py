#!/usr/bin/env python3
"""
=======================================================================
 Lane Visualizer Node — QCar Navigation | Sudo Drive
 Authors: Abraham Moro-Hernandez (AMH19)
-----------------------------------------------------------------------
 RViz Lane Visualization Node
 Converts lane detection output (Float32MultiArray + Point centroid)
 into RViz-compatible Marker messages for 3D visualization in
 the vehicle's base_link frame.

 This node performs a simplified inverse perspective mapping (IPM)
 to project image-space lane line endpoints into a ground-plane
 estimate in the vehicle frame. Full camera-to-ground projection
 requires the camera's extrinsic parameters (height, pitch angle).

 Coordinate frames:
     image frame  : x→right, y↓down  (OpenCV pixels)
     vehicle frame: x→forward, y←left, z↑up  (REP-103 / base_link)
     ground plane : z = 0  (flat world assumption)

 Published Markers:
     /amh19/rviz/lane_lines    (visualization_msgs/MarkerArray)
         - LINE_STRIP for left lane (green)
         - LINE_STRIP for right lane (red)
         - SPHERE for lane centroid (yellow)
         - ARROW for lateral error (blue)
         - TEXT for HUD overlay

 Subscribed Topics:
     /amh19/lane/lines     (std_msgs/Float32MultiArray)
     /amh19/lane/centroid  (geometry_msgs/Point)

 RViz Setup:
     1. Add MarkerArray display → topic: /amh19/rviz/lane_lines
     2. Fixed Frame: base_link (or map if you have localization)
     3. Lane lines appear as 3D markers on the ground plane

 Camera model (simplified IPM):
     The QCar CSI camera is approximately:
       height above ground: ~0.12 m (12 cm)
       pitch angle: ~20° downward
     These values are set as ROS2 parameters and must be calibrated
     to your actual mounting. Adjust camera_height and camera_pitch_deg.
=======================================================================
"""

import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.duration import Duration

from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray
from builtin_interfaces.msg import Duration as DurationMsg


# ── Defaults ──────────────────────────────────────────────────────────
DEFAULT_LINES_TOPIC   = '/amh19/lane/lines'
DEFAULT_CENTROID_TOPIC = '/amh19/lane/centroid'
DEFAULT_MARKER_TOPIC  = '/amh19/rviz/lane_lines'

# Camera mounting parameters (MUST be calibrated for your QCar setup)
DEFAULT_CAMERA_HEIGHT_M   = 0.12   # camera height above ground plane (meters)
DEFAULT_CAMERA_PITCH_DEG  = 20.0   # downward pitch angle (degrees)
DEFAULT_CAMERA_FOV_H_DEG  = 80.0   # horizontal field of view (degrees)
DEFAULT_IMAGE_WIDTH_PX    = 640    # expected image width (pixels)
DEFAULT_IMAGE_HEIGHT_PX   = 480    # expected image height (pixels)

# Marker lifetime (seconds). 0 = permanent (useful for static viz)
DEFAULT_MARKER_LIFETIME_S = 0.2

# Fixed frame for RViz
DEFAULT_FIXED_FRAME = 'base_link'

# Namespace for markers (prevents collision with other plugins)
MARKER_NS = 'lane_detection'


class LaneVisualizerNode(Node):
    """
    Subscribes to lane detection output and publishes RViz Marker messages.

    Inverse Perspective Mapping (simplified):
        Given a pixel (u, v) in the image plane, estimate its position
        in the vehicle frame using the pinhole camera model + known
        camera height and pitch angle.

        y_ground (forward distance) = camera_height / tan(pitch + alpha)
        x_ground (lateral offset)   = y_ground * tan(beta)

        where:
            alpha = vertical angle from optical center = atan((v_center - v) / fy)
            beta  = horizontal angle from optical center = atan((u - u_center) / fx)

        This is an approximation assuming a flat ground plane.
        For higher accuracy, use full extrinsic calibration with cv2.solvePnP.

    Note: This node uses SYNCHRONIZED subscribers (message_filters) to ensure
    lines and centroid arrive together. A simple timer-based approach is also
    provided as fallback.
    """

    def __init__(self):
        super().__init__('lane_visualizer_node')

        # ── Declare parameters ─────────────────────────────────────────
        self.declare_parameter('lines_topic',          DEFAULT_LINES_TOPIC)
        self.declare_parameter('centroid_topic',       DEFAULT_CENTROID_TOPIC)
        self.declare_parameter('marker_topic',         DEFAULT_MARKER_TOPIC)
        self.declare_parameter('camera_height_m',      DEFAULT_CAMERA_HEIGHT_M)
        self.declare_parameter('camera_pitch_deg',     DEFAULT_CAMERA_PITCH_DEG)
        self.declare_parameter('camera_fov_h_deg',     DEFAULT_CAMERA_FOV_H_DEG)
        self.declare_parameter('image_width_px',       DEFAULT_IMAGE_WIDTH_PX)
        self.declare_parameter('image_height_px',      DEFAULT_IMAGE_HEIGHT_PX)
        self.declare_parameter('marker_lifetime_s',    DEFAULT_MARKER_LIFETIME_S)
        self.declare_parameter('fixed_frame',          DEFAULT_FIXED_FRAME)

        p = lambda name: self.get_parameter(name).value
        self.lines_topic       = p('lines_topic')
        self.centroid_topic    = p('centroid_topic')
        self.marker_topic      = p('marker_topic')
        self.camera_height     = p('camera_height_m')
        self.camera_pitch_rad  = math.radians(p('camera_pitch_deg'))
        self.fov_h_rad         = math.radians(p('camera_fov_h_deg'))
        self.img_w             = p('image_width_px')
        self.img_h             = p('image_height_px')
        self.lifetime_s        = p('marker_lifetime_s')
        self.fixed_frame       = p('fixed_frame')

        # Derived camera intrinsics (simplified pinhole)
        # fx = (img_w / 2) / tan(fov_h / 2)
        self.fx = (self.img_w / 2.0) / math.tan(self.fov_h_rad / 2.0)
        self.fy = self.fx  # assume square pixels
        self.cx = self.img_w  / 2.0
        self.cy = self.img_h  / 2.0

        # State cache (updated by respective subscribers)
        self._latest_lines    = None   # [lx1, ly1, lx2, ly2, rx1, ry1, rx2, ry2]
        self._latest_centroid = None   # (cx_px, cy_px)

        # ── Subscribers ────────────────────────────────────────────────
        self.sub_lines = self.create_subscription(
            Float32MultiArray,
            self.lines_topic,
            self._lines_callback,
            10
        )
        self.sub_centroid = self.create_subscription(
            Point,
            self.centroid_topic,
            self._centroid_callback,
            10
        )

        # ── Publisher ──────────────────────────────────────────────────
        self.pub_markers = self.create_publisher(MarkerArray, self.marker_topic, 10)

        # ── Publish timer (decoupled from perception frequency) ────────
        # Markers are published at 15 Hz regardless of detection rate.
        # This prevents RViz from flickering when detections are intermittent.
        self.pub_timer = self.create_timer(1.0 / 15.0, self._publish_markers)

        self._log_startup()

    def _log_startup(self):
        log = self.get_logger().info
        log('=' * 60)
        log(' LANE VISUALIZER NODE STARTED  [RViz Markers]')
        log('=' * 60)
        log(f'Subscribed : {self.lines_topic}')
        log(f'           : {self.centroid_topic}')
        log(f'Publishing : {self.marker_topic}  (visualization_msgs/MarkerArray)')
        log(f'Frame      : {self.fixed_frame}')
        log(f'Camera     : h={self.camera_height}m  pitch={math.degrees(self.camera_pitch_rad):.1f}°')
        log(f'Intrinsics : fx={self.fx:.1f}  cx={self.cx}  cy={self.cy}')
        log('=' * 60)
        log('RViz setup:')
        log('  1. Add → MarkerArray → topic: /amh19/rviz/lane_lines')
        log('  2. Fixed Frame: base_link')
        log('=' * 60)

    # ── Subscriber callbacks ──────────────────────────────────────────
    def _lines_callback(self, msg: Float32MultiArray):
        if len(msg.data) == 8:
            self._latest_lines = list(msg.data)

    def _centroid_callback(self, msg: Point):
        self._latest_centroid = (msg.x, msg.y)

    # ── Main publish timer callback ───────────────────────────────────
    def _publish_markers(self):
        marker_array = MarkerArray()
        now = self.get_clock().now().to_msg()

        # Deletion marker — clears all previous markers each frame
        # (avoids ghost markers when lines disappear)
        delete_all = Marker()
        delete_all.action = Marker.DELETEALL
        delete_all.ns = MARKER_NS
        marker_array.markers.append(delete_all)

        marker_id = 0

        if self._latest_lines is not None:
            lx1, ly1, lx2, ly2, rx1, ry1, rx2, ry2 = self._latest_lines
            has_left  = lx1 >= 0
            has_right = rx1 >= 0

            # Left lane line (green)
            if has_left:
                m = self._make_lane_line_marker(
                    marker_id, now,
                    [(lx1, ly1), (lx2, ly2)],
                    r=0.0, g=1.0, b=0.0, label='LEFT'
                )
                if m: marker_array.markers.append(m)
                marker_id += 1

            # Right lane line (red)
            if has_right:
                m = self._make_lane_line_marker(
                    marker_id, now,
                    [(rx1, ry1), (rx2, ry2)],
                    r=1.0, g=0.0, b=0.0, label='RIGHT'
                )
                if m: marker_array.markers.append(m)
                marker_id += 1

            # Lane corridor fill (translucent polygon between lines)
            if has_left and has_right:
                m = self._make_lane_corridor_marker(
                    marker_id, now,
                    (lx1, ly1), (lx2, ly2),
                    (rx1, ry1), (rx2, ry2)
                )
                if m:
                    marker_array.markers.append(m)
                    marker_id += 1

        # Centroid sphere + error arrow
        if self._latest_centroid is not None:
            cx_px, cy_px = self._latest_centroid

            # Project centroid to ground plane
            gx, gy = self._pixel_to_ground(cx_px, cy_px)

            # Sphere at centroid ground point
            sphere = self._make_sphere_marker(marker_id, now, gx, gy, 0.04,
                                              r=1.0, g=1.0, b=0.0)
            marker_array.markers.append(sphere)
            marker_id += 1

            # Lateral error arrow (centroid x vs vehicle center x=0)
            lateral_error = gy   # in vehicle frame, y = lateral offset
            arrow = self._make_error_arrow_marker(marker_id, now, gx, lateral_error)
            marker_array.markers.append(arrow)
            marker_id += 1

            # Text label
            label = self._make_text_marker(
                marker_id, now, gx, gy + 0.05,
                f'err: {lateral_error:+.3f}m'
            )
            marker_array.markers.append(label)
            marker_id += 1

        self.pub_markers.publish(marker_array)

    # ── IPM: pixel → ground plane ─────────────────────────────────────
    def _pixel_to_ground(self, u: float, v: float):
        """
        Projects image pixel (u, v) to vehicle frame ground coordinates (x_fwd, y_lat).

        Model:
            - Camera at height h above flat ground
            - Camera pitched downward by camera_pitch_rad
            - Pinhole model with focal length fx, fy

        Returns:
            x_fwd : forward distance from camera origin (meters)
            y_lat : lateral offset (positive = left, negative = right) (meters)

        WARNING: This is a simplified flat-earth IPM. It does NOT account for:
            - Road slope / banking
            - Camera roll
            - Lens distortion (should be corrected by UndistortNode first ✓)
        For production use, replace with full cv2.solvePnP + homography approach.
        """
        # Vertical angle below optical axis
        alpha = math.atan2(v - self.cy, self.fy)

        # Effective vertical angle from horizontal (accounting for camera pitch)
        theta_vert = self.camera_pitch_rad - alpha

        if theta_vert <= 0:
            # Point is above horizon — cannot project to ground
            return 1.0, 0.0

        x_fwd = self.camera_height / math.tan(theta_vert)

        # Horizontal angle from optical axis
        beta  = math.atan2(u - self.cx, self.fx)
        y_lat = -x_fwd * math.tan(beta)   # negative: image-right → vehicle-right → negative y in REP-103

        # Clamp to reasonable road range
        x_fwd = max(0.1, min(x_fwd, 5.0))
        y_lat = max(-2.0, min(y_lat, 2.0))

        return x_fwd, y_lat

    # ── Marker builders ───────────────────────────────────────────────
    def _base_marker(self, marker_id: int, now, mtype: int) -> Marker:
        m = Marker()
        m.header.frame_id = self.fixed_frame
        m.header.stamp    = now
        m.ns              = MARKER_NS
        m.id              = marker_id
        m.type            = mtype
        m.action          = Marker.ADD
        # Lifetime
        lt = self.lifetime_s
        m.lifetime.sec     = int(lt)
        m.lifetime.nanosec = int((lt - int(lt)) * 1e9)
        return m

    def _make_lane_line_marker(self, mid, now, px_pts, r, g, b, label=''):
        """
        LINE_STRIP marker for a lane boundary line.
        px_pts: list of (u, v) pixel coordinates (2 points for a line segment).
        """
        m = self._base_marker(mid, now, Marker.LINE_STRIP)
        m.scale.x = 0.03   # line width in meters

        m.color.r = r
        m.color.g = g
        m.color.b = b
        m.color.a = 0.9

        for (u, v) in px_pts:
            gx, gy = self._pixel_to_ground(u, v)
            p = Point()
            p.x = gx
            p.y = gy
            p.z = 0.01   # slightly above ground to prevent z-fighting
            m.points.append(p)

        return m if len(m.points) >= 2 else None

    def _make_lane_corridor_marker(self, mid, now, lp1, lp2, rp1, rp2):
        """
        TRIANGLE_LIST marker filling the lane corridor (translucent).
        Uses two triangles to fill the quadrilateral lane area.
        """
        m = self._base_marker(mid, now, Marker.TRIANGLE_LIST)
        m.scale.x = 1.0
        m.scale.y = 1.0
        m.scale.z = 1.0
        m.color.r = 0.0
        m.color.g = 0.8
        m.color.b = 0.8
        m.color.a = 0.15   # very transparent

        corners_px = [lp1, lp2, rp2, rp1]
        corners_3d = []
        for (u, v) in corners_px:
            gx, gy = self._pixel_to_ground(u, v)
            p = Point()
            p.x = gx
            p.y = gy
            p.z = 0.005
            corners_3d.append(p)

        # Triangle 1: lp1, lp2, rp2
        m.points.extend([corners_3d[0], corners_3d[1], corners_3d[2]])
        # Triangle 2: lp1, rp2, rp1
        m.points.extend([corners_3d[0], corners_3d[2], corners_3d[3]])

        return m

    def _make_sphere_marker(self, mid, now, x, y, size, r, g, b):
        m = self._base_marker(mid, now, Marker.SPHERE)
        m.pose.position.x = x
        m.pose.position.y = y
        m.pose.position.z = 0.05
        m.pose.orientation.w = 1.0
        m.scale.x = size
        m.scale.y = size
        m.scale.z = size
        m.color.r = r
        m.color.g = g
        m.color.b = b
        m.color.a = 1.0
        return m

    def _make_error_arrow_marker(self, mid, now, x_fwd, y_lat):
        """
        ARROW from vehicle center (0,0) to centroid ground projection.
        Blue arrow = lateral error direction.
        """
        m = self._base_marker(mid, now, Marker.ARROW)
        m.scale.x = 0.02   # shaft diameter
        m.scale.y = 0.04   # head diameter
        m.scale.z = 0.06   # head length
        m.color.r = 0.2
        m.color.g = 0.5
        m.color.b = 1.0
        m.color.a = 0.85

        start = Point(); start.x = x_fwd; start.y = 0.0;   start.z = 0.05
        end   = Point(); end.x   = x_fwd; end.y   = y_lat; end.z   = 0.05
        m.points = [start, end]
        return m

    def _make_text_marker(self, mid, now, x, y, text):
        m = self._base_marker(mid, now, Marker.TEXT_VIEW_FACING)
        m.pose.position.x = x
        m.pose.position.y = y
        m.pose.position.z = 0.15
        m.pose.orientation.w = 1.0
        m.scale.z = 0.08   # text height in meters
        m.color.r = 1.0
        m.color.g = 1.0
        m.color.b = 1.0
        m.color.a = 1.0
        m.text = text
        return m


# ── Entry point ───────────────────────────────────────────────────────
def main(args=None):
    rclpy.init(args=args)
    node = LaneVisualizerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print('\n' + '=' * 60)
        print(' LANE VISUALIZER NODE STOPPED')
        print('=' * 60)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()