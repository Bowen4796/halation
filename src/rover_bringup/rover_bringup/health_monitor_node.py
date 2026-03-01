"""
Health Monitor Node - Continuously monitors critical ROS topics and resources
for the rover recovery system. Publishes system health status on /system_health.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32
from sensor_msgs.msg import LaserScan
from slg_msgs.msg import SegmentArray
import json
import time
from collections import deque


class HealthMonitor(Node):
    def __init__(self):
        super().__init__('health_monitor')
        
        # Topic freshness tracking
        self.topic_timestamps = {
            '/scan': None,
            '/segments': None,
            '/motor_command': None,
            '/lidar_pitch': None,
        }
        
        # Startup grace period (topics may not publish immediately)
        self.startup_time = time.time()
        self.startup_grace_period = 5.0  # seconds
        
        # Freshness thresholds (seconds)
        self.freshness_thresholds = {
            '/scan': 0.2,           # LIDAR should update every 100ms
            '/segments': 0.5,       # Allow time for segmentation processing
            '/motor_command': 1.0,  # Commands less frequent
            '/lidar_pitch': 0.5,    # Servo position feedback
        }
        
        # Resource thresholds
        self.cpu_threshold = 80.0      # Percent
        self.memory_threshold = 85.0   # Percent
        
        # Topic frequency tracking
        self.scan_frequency_history = deque(maxlen=10)
        
        # Last known health status
        self.last_health_status = {'overall': 'INITIALIZING'}
        
        # Create subscriptions to monitored topics
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.create_subscription(SegmentArray, '/segments', self.segments_callback, 10)
        self.create_subscription(String, '/motor_command', self.cmd_callback, 10)
        self.create_subscription(Float32, '/lidar_pitch', self.servo_callback, 10)
        
        # Publisher for health status
        self.health_pub = self.create_publisher(String, '/system_health', 10)
        
        # Health check timer (100ms)
        self.create_timer(0.1, self.health_check)
        
        self.get_logger().info('Health Monitor initialized')
    
    def scan_callback(self, msg):
        """Track LIDAR scan arrivals."""
        now = self.get_clock().now()
        self.topic_timestamps['/scan'] = now
        
        # Track scan frequency for anomaly detection
        if len(self.scan_frequency_history) > 0:
            last_time = self.scan_frequency_history[-1]
            delta = (now - last_time).nanoseconds / 1e9
            
            if delta > 0.2:  # Should be ~100ms, 0.2s is concerning
                self.get_logger().warn(f'LIDAR slow: {delta:.3f}s since last scan')
                pass
        self.scan_frequency_history.append(now)
    
    def segments_callback(self, msg):
        """Track segmentation output arrivals."""
        self.topic_timestamps['/segments'] = self.get_clock().now()
    
    def cmd_callback(self, msg):
        """Track motor command arrivals."""
        self.topic_timestamps['/motor_command'] = self.get_clock().now()
    
    def servo_callback(self, msg):
        """Track servo position feedback."""
        self.topic_timestamps['/lidar_pitch'] = self.get_clock().now()
    
    def health_check(self):
        """Periodic health assessment."""
        health_status = {
            'timestamp': self.get_clock().now().nanoseconds,
            'topics': {},
            'resources': {},
            'overall': 'HEALTHY',
        }
        
        now = self.get_clock().now()
        critical_count = 0
        warning_count = 0
        
        # Check each monitored topic
        for topic_name, threshold in self.freshness_thresholds.items():
            last_time = self.topic_timestamps[topic_name]
            
            if last_time is None:
                # Check if we're still in startup grace period
                elapsed_startup = time.time() - self.startup_time
                
                if elapsed_startup < self.startup_grace_period:
                    # During startup, NEVER_RECEIVED is only a warning
                    health_status['topics'][topic_name] = {
                        'status': 'NEVER_RECEIVED',
                        'age_sec': None,
                        'severity': 'WARNING',
                        'note': f'Startup grace period ({elapsed_startup:.1f}s / {self.startup_grace_period}s)'
                    }
                    warning_count += 1
                else:
                    # After startup, NEVER_RECEIVED is critical (node probably doesn't publish this topic)
                    health_status['topics'][topic_name] = {
                        'status': 'NEVER_RECEIVED',
                        'age_sec': None,
                        'severity': 'CRITICAL'
                    }
                    critical_count += 1
            else:
                age = (now - last_time).nanoseconds / 1e9
                
                if age > threshold * 5:  # Very stale
                    status = 'STALE'
                    severity = 'CRITICAL'
                    critical_count += 1
                elif age > threshold:  # Slightly stale
                    status = 'STALE'
                    severity = 'WARNING'
                    warning_count += 1
                else:
                    status = 'FRESH'
                    severity = 'OK'
                
                health_status['topics'][topic_name] = {
                    'status': status,
                    'age_sec': age,
                    'threshold_sec': threshold,
                    'severity': severity
                }
        
        # Check resource usage
        try:
            import psutil
            process = psutil.Process()
            
            cpu_percent = process.cpu_percent(interval=0.01)
            memory_percent = process.memory_percent()
            
            health_status['resources'] = {
                'cpu_percent': cpu_percent,
                'memory_percent': memory_percent,
                'cpu_ok': cpu_percent < self.cpu_threshold,
                'memory_ok': memory_percent < self.memory_threshold,
            }
            
            if cpu_percent > self.cpu_threshold:
                warning_count += 1
            
            if memory_percent > self.memory_threshold:
                warning_count += 1
        except ImportError:
            self.get_logger().debug('psutil not available for resource monitoring')
            health_status['resources'] = {'error': 'psutil not available'}
        
        # Determine overall health
        if critical_count > 0:
            health_status['overall'] = 'CRITICAL'
        elif warning_count > 0:
            health_status['overall'] = 'DEGRADED'
        
        # Store for recovery manager
        self.last_health_status = health_status
        
        # Publish health status
        msg = String()
        msg.data = json.dumps(health_status, default=str)
        self.health_pub.publish(msg)
        
        # Log if degraded or critical
        if health_status['overall'] in ['DEGRADED', 'CRITICAL']:
            self.get_logger().warn(f"System Health: {health_status['overall']}")
            for topic, status in health_status['topics'].items():
                if status.get('severity') != 'OK':
                    age = status.get('age_sec')
                    age_str = f"{age:.2f}s" if age is not None else "N/A"
                    self.get_logger().warn(f"  {topic}: {status['status']} (age: {age_str})")


def main(args=None):
    rclpy.init(args=args)
    monitor = HealthMonitor()
    rclpy.spin(monitor)
    monitor.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
