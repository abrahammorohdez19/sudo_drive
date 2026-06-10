#!/usr/bin/env python3
"""
=======================================================================
 QCar Traffic Detector Node - Smart Mobility
 
-----------------------------------------------------------------------
 Bridges the vision pipeline with the vehicle control layer.
 Subscribes to the traffic state topic published by the YOLO vision
 node and publishes two independent boolean alerts:
 
   /qcar/obstacle_alert  — True when a stop sign OR red light is
                           detected  →  full stop required.
   /qcar/slow_down_alert — True when a yellow light is detected
                           →  reduce speed.
 
 Mirrors the interface and coding style of the LiDAR-based
 ObstacleDetector node.
=======================================================================
"""
 
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String
 
 
class TrafficDetector(Node):
    """Translates vision-based traffic detections into vehicle control alerts."""
 
    # ------------------------------------------------------------------
    # Labels published by the vision node  (traffic_unified_node.py)
    # ------------------------------------------------------------------
    _LABEL_STOP   = "Stop"
    _LABEL_RED    = "Red"
    _LABEL_YELLOW = "Yellow"
    _LABEL_GREEN  = "Green"
 
    def __init__(self):
        super().__init__('traffic_detector')
 
        # ----------------------------------------------------------
        # Parameters
        # ----------------------------------------------------------
        self.debug_mode = False     # True = log every incoming state message
 
        # ----------------------------------------------------------
        # Internal state  (track changes to avoid redundant logs)
        # ----------------------------------------------------------
        self._obstacle_active  = False   # Stop sign or Red light
        self._slow_down_active = False   # Yellow light
 
        # ----------------------------------------------------------
        # Publishers
        # ----------------------------------------------------------
        self.obstacle_pub  = self.create_publisher(Bool, '/qcar/obstacle_alert',  10)
        self.slow_down_pub = self.create_publisher(Bool, '/qcar/slow_down_alert', 10)
 
        # ----------------------------------------------------------
        # Subscribers
        # ----------------------------------------------------------
        self.traffic_sub = self.create_subscription(
            String,
            '/compe/traffic_state',
            self.traffic_state_callback,
            10,
        )
 
        self.get_logger().info("QCar Traffic Detector Node initialized")
        self.get_logger().info(f"  Listening on       : /compe/traffic_state")
        self.get_logger().info(f"  Obstacle alert     : /qcar/obstacle_alert  (Stop | Red)")
        self.get_logger().info(f"  Slow-down alert    : /qcar/slow_down_alert (Yellow)")
        if self.debug_mode:
            self.get_logger().info("  Debug Mode: ON")
 
    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
 
    def traffic_state_callback(self, msg: String) -> None:
        """
        Maps incoming traffic state labels to boolean alert topics.
 
        Decision table
        ──────────────────────────────────────────────────────────
          State    │ obstacle_alert │ slow_down_alert │ Action
        ───────────┼────────────────┼─────────────────┼──────────
          "Stop"   │   True         │   False         │ Full stop
          "Red"    │   True         │   False         │ Full stop
          "Yellow" │   False        │   True          │ Slow down
          "Green"  │   False        │   False         │ Continue
          other    │   False        │   False         │ Continue
        ──────────────────────────────────────────────────────────
        """
        state = msg.data
 
        obstacle  = state in (self._LABEL_STOP, self._LABEL_RED)
        slow_down = state == self._LABEL_YELLOW
 
        if self.debug_mode:
            self.get_logger().info(
                f"Traffic state: '{state}' → obstacle={obstacle}, slow_down={slow_down}"
            )
 
        # Log only on state transitions to keep the terminal clean
        if obstacle != self._obstacle_active:
            self._obstacle_active = obstacle
            if obstacle:
                self.get_logger().warn(
                    f"FULL STOP: '{state}' detected — obstacle alert ON"
                )
            else:
                self.get_logger().info(
                    f"Obstacle cleared (state: '{state}') — obstacle alert OFF"
                )
 
        if slow_down != self._slow_down_active:
            self._slow_down_active = slow_down
            if slow_down:
                self.get_logger().warn(
                    "SLOW DOWN: Yellow light detected — slow-down alert ON"
                )
            else:
                self.get_logger().info(
                    f"Yellow cleared (state: '{state}') — slow-down alert OFF"
                )
 
        # Publish a steady stream on both topics every callback
        obstacle_msg       = Bool()
        obstacle_msg.data  = obstacle
        self.obstacle_pub.publish(obstacle_msg)
 
        slow_down_msg      = Bool()
        slow_down_msg.data = slow_down
        self.slow_down_pub.publish(slow_down_msg)
 
 
# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
 
def main(args=None):
    rclpy.init(args=args)
    node = TrafficDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
 
 
if __name__ == '__main__':
    main()
 
