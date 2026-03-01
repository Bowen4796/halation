# Rerouting Node (takes segmented LIDAR output and generates a new path to avoid obstacles)
import rclpy
from rclpy.lifecycle import Node as LifecycleNode
from rclpy.lifecycle import State, TransitionCallbackReturn
from std_msgs.msg import String
from slg_msgs.msg import SegmentArray
import json

class ReroutingNode(LifecycleNode):
    def __init__(self):
        super().__init__('rerouting_node')
        
        self.subscription = None
        self.publisher_ = None
        self.segment_watchdog = None
        self.last_segment_time = None
        
        # Safety Parameters
        self.stop_distance = 1.5  # Meters: How close can an object get?
        self.lane_width = 0.5     # Meters: How wide is our path (center to edge)?
        
        self.get_logger().info("Rerouting Node (V2) lifecycle initialized")

    def on_configure(self, state: State) -> TransitionCallbackReturn:
        """Configure subscriptions and publishers."""
        self.get_logger().info('on_configure called')
        try:
            # Subscribe to the processed segments (green boxes)
            self.subscription = self.create_subscription(
                SegmentArray,
                '/segments',
                self.listener_callback,
                10)
            
            # Publish commands to the motor interface
            self.publisher_ = self.create_publisher(String, '/motor_command', 10)
            
            self.get_logger().info("Rerouting Node configured")
            return TransitionCallbackReturn.SUCCESS
        except Exception as e:
            self.get_logger().error(f'Configure failed: {e}')
            return TransitionCallbackReturn.FAILURE

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        """Activate rerouting and start watchdog."""
        self.get_logger().info('on_activate called')
        try:
            self.last_segment_time = self.get_clock().now()
            
            # Start watchdog timer
            self.segment_watchdog = self.create_timer(0.5, self.watchdog_callback)
            self.get_logger().info('Rerouting Node activated')
            return TransitionCallbackReturn.SUCCESS
        except Exception as e:
            self.get_logger().error(f'Activate failed: {e}')
            return TransitionCallbackReturn.FAILURE

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        """Deactivate rerouting and stop watchdog."""
        self.get_logger().info('on_deactivate called')
        try:
            if self.segment_watchdog:
                self.destroy_timer(self.segment_watchdog)
                self.segment_watchdog = None
            
            self.get_logger().info('Rerouting Node deactivated')
            return TransitionCallbackReturn.SUCCESS
        except Exception as e:
            self.get_logger().error(f'Deactivate failed: {e}')
            return TransitionCallbackReturn.FAILURE

    def on_cleanup(self, state: State) -> TransitionCallbackReturn:
        """Clean up subscriptions and publishers."""
        self.get_logger().info('on_cleanup called')
        try:
            if self.subscription:
                self.destroy_subscription(self.subscription)
                self.subscription = None
            
            if self.publisher_:
                # Publisher cleanup if needed
                pass
            
            self.get_logger().info('Rerouting Node cleaned up')
            return TransitionCallbackReturn.SUCCESS
        except Exception as e:
            self.get_logger().error(f'Cleanup failed: {e}')
            return TransitionCallbackReturn.FAILURE

    def on_shutdown(self, state: State) -> TransitionCallbackReturn:
        """Shutdown the node."""
        self.get_logger().info('on_shutdown called')
        return TransitionCallbackReturn.SUCCESS

    def watchdog_callback(self):
        """Detect if segment stream has stopped."""
        now = self.get_clock().now()
        elapsed = (now - self.last_segment_time).nanoseconds / 1e9
        
        if elapsed > 1.0:
            self.get_logger().warn(f'Watchdog: No segments for {elapsed:.1f}s')
            # Send STOP command when segments are stale
            self.publish_command("stop")

    def listener_callback(self, msg):
        """Process incoming segments."""
        self.last_segment_time = self.get_clock().now()
        
        action = "forwards"
        
        # Loop through every object the segmentation node found
        for segment in msg.segments:
            # Calculate the centroid (geometric center) of the object
            if not segment.points:
                continue
                
            avg_x = sum(p.x for p in segment.points) / len(segment.points)
            avg_y = sum(p.y for p in segment.points) / len(segment.points)

            # Decision Logic: Is this object in my way?
            # X is forward distance, Y is left/right offset
            if 0 < avg_x < self.stop_distance:  # Is it close?
                if abs(avg_y) < self.lane_width: # Is it in my lane?
                    self.get_logger().warn(f"Avoidance Triggered! Object {segment.id} at ({avg_x:.2f}, {avg_y:.2f})")
                    
                    # Simple Avoidance: Steer away from the center of the object
                    if avg_y > 0:
                        action = "turn_right" # Object is on left -> go right
                    else:
                        action = "turn_left"  # Object is on right -> go left
                    
                    # Once we find one threat, we react and stop checking others
                    break

        self.publish_command(action)

    def publish_command(self, command):
        msg = String()
        msg.data = json.dumps({"command": command})
        self.publisher_.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = ReroutingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


