"""
Recovery Manager Node - Monitors system health and orchestrates recovery
of failed lifecycle nodes. Implements 3-level recovery strategy.
"""

import rclpy
from rclpy.node import Node
from rclpy.lifecycle import State
from lifecycle_msgs.srv import GetState, ChangeState
from lifecycle_msgs.msg import Transition
from std_msgs.msg import String
import json
import time
from enum import Enum
from threading import Thread


class RecoveryState(Enum):
    INITIALIZING = "INITIALIZING"
    NORMAL = "NORMAL"
    DEGRADED = "DEGRADED"
    RECOVERY_1 = "RECOVERY_1"      # Light recovery
    RECOVERY_2 = "RECOVERY_2"      # Medium recovery
    RECOVERY_3 = "RECOVERY_3"      # Heavy recovery
    MANUAL_MODE = "MANUAL_MODE"    # Operator only


class RecoveryManager(Node):
    def __init__(self):
        super().__init__('recovery_manager')
        
        # Recovery state machine
        self.state = RecoveryState.INITIALIZING
        self.last_state_change = time.time()
        
        # Failure tracking
        self.failure_type = None
        self.failure_types = []  # Can be multiple
        self.affected_nodes = []
        self.recovery_attempts = 0
        self.max_recovery_attempts = 3
        
        # Critical nodes that can be recovered
        self.critical_nodes = {
            'lidar_servo': {
                'full_name': '/lidar_servo',
                'recovery_order': 1
            },
            'laser_segmentation': {
                'full_name': '/laser_segmentation',
                'recovery_order': 2
            },
            'rerouting_node': {
                'full_name': '/rerouting_node',
                'recovery_order': 3
            },
            'motor_control': {
                'full_name': '/motor_control',
                'recovery_order': 4
            }
        }
        
        # Lifecycle service clients
        self.lifecycle_clients = {}
        for node_name, config in self.critical_nodes.items():
            self.lifecycle_clients[node_name] = {
                'change_state': self.create_client(
                    ChangeState,
                    f'{config["full_name"]}/change_state'
                ),
                'get_state': self.create_client(
                    GetState,
                    f'{config["full_name"]}/get_state'
                )
            }
        
        # Subscribe to health status
        self.health_sub = self.create_subscription(
            String,
            '/system_health',
            self.health_callback,
            10
        )
        
        # Publisher for recovery alerts
        self.alert_pub = self.create_publisher(String, '/recovery_alerts', 10)
        
        # Recovery state machine timer
        self.create_timer(0.5, self.recovery_logic)
        
        self.get_logger().info('Recovery Manager initialized')
    
    def health_callback(self, msg):
        """Receive and process health status updates."""
        try:
            health = json.loads(msg.data)
            
            # Check for critical issues
            critical_issues = []
            for topic, status in health.get('topics', {}).items():
                if status.get('severity') == 'CRITICAL':
                    critical_issues.append({
                        'topic': topic,
                        'status': status
                    })
            
            if critical_issues:
                self.detect_failure(critical_issues)
            else:
                # No critical issues - system is healthy
                if self.state != RecoveryState.NORMAL:
                    self.state = RecoveryState.NORMAL
                    self.get_logger().info('System recovered (no critical issues detected)')
        
        except json.JSONDecodeError as e:
            self.get_logger().error(f'Failed to parse health status: {e}')
    
    def detect_failure(self, issues):
        """Identify failure types and affected nodes."""
        
        # Topic to failure configuration mapping
        topic_configs = {
            '/scan': {
                'failure_type': 'LIDAR_FAILURE',
                'affected_nodes': ['lidar_servo', 'laser_segmentation', 'rerouting_node']
            },
            '/segments': {
                'failure_type': 'SEGMENTATION_FAILURE',
                'affected_nodes': ['laser_segmentation', 'rerouting_node']
            },
            '/motor_command': {
                'failure_type': 'DECISION_FAILURE',
                'affected_nodes': ['rerouting_node']
            },
            '/lidar_pitch': {
                'failure_type': 'SERVO_FAILURE',
                'affected_nodes': ['lidar_servo']
            }
        }
        
        # Track all failures detected
        failure_types = []
        affected_nodes = set()
        failed_topics = []
        
        # Check each monitored topic
        for topic, config in topic_configs.items():
            if any(issue['topic'] == topic for issue in issues):
                failure_types.append(config['failure_type'])
                affected_nodes.update(config['affected_nodes'])
                failed_topics.append(topic)
        
        # Handle unknown failures
        if not failure_types:
            failure_types.append('UNKNOWN')
            affected_nodes = set()
            failed_topics = [issue['topic'] for issue in issues]
        
        # Store failure information
        self.failure_types = failure_types
        self.failure_type = failure_types[0] if failure_types else 'UNKNOWN'  # Primary failure
        self.affected_nodes = list(affected_nodes)
        
        # Log detailed failure information
        self.get_logger().error(
            f'FAILURE DETECTED: {self.failure_type}\n'
            f'All failures: {self.failure_types}\n'
            f'Failed topics: {failed_topics}\n'
            f'Affected nodes: {self.affected_nodes}'
        )
        
        # Publish alert with primary failure type
        self.publish_alert(self.failure_type, 'DETECTED', self.affected_nodes)
        
        # Transition to degraded state
        if self.state == RecoveryState.NORMAL:
            self.state = RecoveryState.DEGRADED
            self.recovery_attempts = 0
    
    def recovery_logic(self):
        """Main recovery state machine."""
        
        if self.state == RecoveryState.INITIALIZING:
            # Wait for system to report health
            pass
        
        elif self.state == RecoveryState.NORMAL:
            # All good
            pass
        
        elif self.state == RecoveryState.DEGRADED:
            # Transition to recovery attempt 1
            self.state = RecoveryState.RECOVERY_1
            self.recovery_attempts = 0
            self.last_state_change = time.time()
            self.get_logger().warn('Starting recovery attempt 1 (light recovery)')
            self.publish_alert(self.failure_type, 'RECOVERY_START_1', self.affected_nodes)
        
        elif self.state == RecoveryState.RECOVERY_1:
            elapsed = time.time() - self.last_state_change
            
            if elapsed < 0.5:
                # Deactivate affected nodes only
                self.deactivate_nodes(self.affected_nodes)
            elif elapsed < 1.5:
                # Wait for deactivation to complete and nodes to settle
                # If nodes were killed and respawned, they need time to restart
                pass
            elif elapsed < 2.0:
                # Reactivate affected nodes (now they should be in unconfigured state)
                # configure_nodes will be called first by activate_nodes
                self.activate_nodes(self.affected_nodes)
            else:
                # Check if recovered (would get health callback)
                # After 2.5 seconds, move to next attempt
                self.recovery_attempts += 1
                self.state = RecoveryState.RECOVERY_2
                self.last_state_change = time.time()
                self.get_logger().warn(f'Recovery attempt 1 inconclusive, trying recovery 2')
                self.publish_alert(self.failure_type, 'RECOVERY_START_2', self.affected_nodes)
        
        elif self.state == RecoveryState.RECOVERY_2:
            elapsed = time.time() - self.last_state_change
            
            if elapsed < 1.0:
                # Deactivate more broadly
                self.deactivate_nodes(list(self.critical_nodes.keys()))
                self.send_motor_command('STOP')
            elif elapsed < 4.0:
                # Wait longer for settling and potential node respawn
                # If nodes were killed, they need time to restart
                pass
            elif elapsed < 4.5:
                # Reactivate in proper order (configure will be called automatically)
                self.activate_nodes(list(self.critical_nodes.keys()))
            elif elapsed < 5.0:
                # Send neutral command to resume motion
                self.send_motor_command('FORWARD')
            else:
                # Check recovery
                self.recovery_attempts += 1
                self.state = RecoveryState.RECOVERY_3
                self.last_state_change = time.time()
                self.get_logger().warn(f'Recovery attempt 2 inconclusive, trying recovery 3')
                self.publish_alert(self.failure_type, 'RECOVERY_START_3', self.affected_nodes)
        
        elif self.state == RecoveryState.RECOVERY_3:
            elapsed = time.time() - self.last_state_change
            
            if elapsed < 1.0:
                # Full stop
                self.send_motor_command('STOP')
                self.deactivate_all_nodes()
            elif elapsed < 6.0:
                # Wait longer for nodes to settle and potentially be killed/respawned
                pass
            elif elapsed < 6.5:
                # Try full reactivation (configure will be called automatically)
                self.activate_all_nodes()
            elif elapsed < 7.0:
                # Resume motion
                self.send_motor_command('FORWARD')
            else:
                self.recovery_attempts += 1
                if self.recovery_attempts >= self.max_recovery_attempts:
                    self.get_logger().critical(
                        f'RECOVERY FAILED after {self.max_recovery_attempts} attempts!\n'
                        f'Failure type: {self.failure_type}\n'
                        f'Affected nodes: {self.affected_nodes}\n'
                        f'Switching to MANUAL MODE ONLY'
                    )
                    self.state = RecoveryState.MANUAL_MODE
                    self.send_motor_command('STOP')
                    self.deactivate_all_nodes()
                    self.publish_alert(self.failure_type, 'RECOVERY_FAILED', self.affected_nodes)
        
        elif self.state == RecoveryState.MANUAL_MODE:
            # Only respond to explicit commands
            pass
    
    def deactivate_nodes(self, node_names):
        """Deactivate specific nodes."""
        for node_name in node_names:
            if node_name not in self.critical_nodes:
                continue
            
            client = self.lifecycle_clients[node_name]['change_state']
            request = ChangeState.Request()
            request.transition.id = Transition.TRANSITION_DEACTIVATE
            
            try:
                future = client.call_async(request)
                self.get_logger().info(f'Deactivating {node_name}')
            except Exception as e:
                # Log as warning rather than error since node might not be responsive
                self.get_logger().warn(f'Could not deactivate {node_name}: {e}')
            
            time.sleep(0.2)  # Small delay between deactivation calls
    
    def configure_nodes(self, node_names):
        """Configure specific nodes (transition from unconfigured to inactive)."""
        sorted_nodes = sorted(
            node_names,
            key=lambda n: self.critical_nodes.get(n, {}).get('recovery_order', 999)
        )
        
        for node_name in sorted_nodes:
            if node_name not in self.critical_nodes:
                continue
            
            try:
                client = self.lifecycle_clients[node_name]['change_state']
                request = ChangeState.Request()
                request.transition.id = Transition.TRANSITION_CONFIGURE
                
                future = client.call_async(request)
                self.get_logger().info(f'Configuring {node_name}')
            except Exception as e:
                self.get_logger().warn(f'Issue configuring {node_name}: {e}')
            
            time.sleep(0.3)  # Stagger configuration calls
    
    def activate_nodes(self, node_names):
        """Activate specific nodes in order."""
        sorted_nodes = sorted(
            node_names,
            key=lambda n: self.critical_nodes.get(n, {}).get('recovery_order', 999)
        )
        
        # First ensure all nodes are configured
        self.configure_nodes(sorted_nodes)
        
        # Add delay to allow configuration to complete
        time.sleep(0.5)
        
        # Then activate them
        for node_name in sorted_nodes:
            if node_name not in self.critical_nodes:
                continue
            
            client = self.lifecycle_clients[node_name]['change_state']
            request = ChangeState.Request()
            request.transition.id = Transition.TRANSITION_ACTIVATE
            
            try:
                future = client.call_async(request)
                self.get_logger().info(f'Activating {node_name}')
            except Exception as e:
                self.get_logger().error(f'Failed to activate {node_name}: {e}')
            
            time.sleep(0.5)  # Stagger activations
    
    def deactivate_all_nodes(self):
        """Deactivate all critical nodes."""
        self.deactivate_nodes(list(self.critical_nodes.keys()))
    
    def activate_all_nodes(self):
        """Activate all critical nodes."""
        self.activate_nodes(list(self.critical_nodes.keys()))
    
    def send_motor_command(self, command):
        """Send command to motor control."""
        msg = String()
        msg.data = json.dumps({'command': command})
        # This would need to be published somewhere
        self.get_logger().info(f'Motor command: {command}')
    
    def publish_alert(self, failure_type, alert_type, affected_nodes):
        """Publish recovery alert."""
        alert = {
            'timestamp': self.get_clock().now().nanoseconds,
            'failure_type': failure_type,
            'alert_type': alert_type,
            'affected_nodes': affected_nodes,
            'recovery_state': self.state.value,
            'recovery_attempts': self.recovery_attempts
        }
        
        msg = String()
        msg.data = json.dumps(alert)
        self.alert_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    recovery_mgr = RecoveryManager()
    rclpy.spin(recovery_mgr)
    recovery_mgr.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
