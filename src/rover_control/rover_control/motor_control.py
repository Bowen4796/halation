from rclpy.lifecycle import Node as LifecycleNode
from rclpy.lifecycle import State, TransitionCallbackReturn
from std_msgs.msg import String
import json
import rclpy
import time

# Try to import real DAC hardware, fall back to mock
try:
    import adafruit_dacx578
    import board
    import busio
    HARDWARE_AVAILABLE = True
except Exception:
    HARDWARE_AVAILABLE = False
    
    # Mock I2C and DAC classes for testing
    class MockBusIO:
        def I2C(self, scl, sda):
            return MockI2C()
    
    class MockI2C:
        pass
    
    class MockDAC:
        def __init__(self, i2c, address):
            self.channels = [MockChannel() for _ in range(8)]
    
    class MockChannel:
        def __init__(self):
            self.raw_value = 0
    
    class MockAdafruitDAC:
        def DACx578(self, i2c, address):
            return MockDAC(i2c, address)
    
    # Create mock objects
    adafruit_dacx578 = MockAdafruitDAC()
    
    class MockBoard:
        SCL = 'SCL'
        SDA = 'SDA'
    
    board = MockBoard()
    busio = MockBusIO()

i2c = busio.I2C(board.SCL, board.SDA) if HARDWARE_AVAILABLE else None

VOLTAGE_MAX = 4095  # DAC max value for speed

I2C_ADDRESS = 0x4c

# Motor channel configuration
# Left side: motors 1 & 3 (channels 0,1 and 4,5)
# Right side: motors 2 & 4 (channels 2,3 and 6,7)
MOTORS = {
    "left_front": {"speed_channel": 0, "direction_channel": 1, "inverted": False},
    "right_front": {"speed_channel": 2, "direction_channel": 3, "inverted": True},
    "left_back": {"speed_channel": 4, "direction_channel": 5, "inverted": False},
    "right_back": {"speed_channel": 6, "direction_channel": 7, "inverted": True},
}


class MotorControl(LifecycleNode):
    def __init__(self):
        super().__init__("motor_control")
        
        self.dac = None
        self.vector_subscriber = None
        self.estop_subscriber = None
        self.command_watchdog = None
        self.last_command_time = None
        
        self.get_logger().info('MotorControl lifecycle node initialized')

    def on_configure(self, state: State) -> TransitionCallbackReturn:
        """Configure hardware and subscriptions."""
        self.get_logger().info('on_configure called')
        try:
            if HARDWARE_AVAILABLE:
                self.dac = adafruit_dacx578.DACx578(i2c, address=I2C_ADDRESS)
                self.get_logger().info(f"Motor control configured with I2C address {I2C_ADDRESS}")
            else:
                self.dac = adafruit_dacx578.DACx578(None, address=I2C_ADDRESS)
                self.get_logger().warn('Running in MOCK MODE (no real DAC available)')
            
            # Create subscriptions
            self.vector_subscriber = self.create_subscription(
                String, "/motor_vector", self.handle_vector, 10
            )
            
            self.estop_subscriber = self.create_subscription(
                String, "/motor_estop", self.handle_estop, 10
            )
            
            return TransitionCallbackReturn.SUCCESS
        except Exception as e:
            self.get_logger().error(f'Configure failed: {e}')
            return TransitionCallbackReturn.FAILURE

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        """Activate motor control and start watchdog."""
        self.get_logger().info('on_activate called')
        try:
            self.stop_all()
            self.last_command_time = self.get_clock().now()
            
            # Start watchdog timer
            self.command_watchdog = self.create_timer(2.0, self.watchdog_callback)
            self.get_logger().info('Motor control activated')
            return TransitionCallbackReturn.SUCCESS
        except Exception as e:
            self.get_logger().error(f'Activate failed: {e}')
            return TransitionCallbackReturn.FAILURE

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        """Deactivate motor control and stop watchdog."""
        self.get_logger().info('on_deactivate called')
        try:
            # Stop watchdog
            if self.command_watchdog:
                self.destroy_timer(self.command_watchdog)
                self.command_watchdog = None
            
            # Safety: stop all motors
            self.stop_all()
            self.get_logger().info('Motor control deactivated')
            return TransitionCallbackReturn.SUCCESS
        except Exception as e:
            self.get_logger().error(f'Deactivate failed: {e}')
            return TransitionCallbackReturn.FAILURE

    def on_cleanup(self, state: State) -> TransitionCallbackReturn:
        """Clean up subscriptions."""
        self.get_logger().info('on_cleanup called')
        try:
            if self.vector_subscriber:
                self.destroy_subscription(self.vector_subscriber)
                self.vector_subscriber = None
            
            if self.estop_subscriber:
                self.destroy_subscription(self.estop_subscriber)
                self.estop_subscriber = None
            
            self.get_logger().info('Motor control cleaned up')
            return TransitionCallbackReturn.SUCCESS
        except Exception as e:
            self.get_logger().error(f'Cleanup failed: {e}')
            return TransitionCallbackReturn.FAILURE

    def on_shutdown(self, state: State) -> TransitionCallbackReturn:
        """Shutdown the node."""
        self.get_logger().info('on_shutdown called')
        try:
            self.stop_all()
            return TransitionCallbackReturn.SUCCESS
        except Exception as e:
            self.get_logger().error(f'Shutdown failed: {e}')
            return TransitionCallbackReturn.FAILURE

    def watchdog_callback(self):
        """Detect if command stream has stopped."""
        now = self.get_clock().now()
        elapsed = (now - self.last_command_time).nanoseconds / 1e9
        
        if elapsed > 2.0:
            self.get_logger().warn(f'Watchdog: No commands for {elapsed:.1f}s')

    def set_motor(self, motor_name: str, speed: float):
        """Set a motor's speed and direction.
        
        Args:
            motor_name: Key from MOTORS dict
            speed: -1.0 to 1.0 (negative = backward, positive = forward)
        """
        motor = MOTORS[motor_name]
        
        # Apply inversion if needed
        actual_speed = speed if not motor["inverted"] else -speed
        
        # Determine direction
        forward = actual_speed >= 0
        
        # Set speed via DAC (absolute value)
        dac_value = int(abs(actual_speed) * VOLTAGE_MAX)
        dac_value = min(dac_value, VOLTAGE_MAX)  # Clamp to max
        self.dac.channels[motor["speed_channel"]].raw_value = dac_value
        
        # Set direction via DAC channel (VOLTAGE_MAX = forward, 0 = backward)
        direction_value = VOLTAGE_MAX if forward else 0
        self.dac.channels[motor["direction_channel"]].raw_value = direction_value
        
        self.get_logger().debug(f"{motor_name}: speed={dac_value}, forward={forward}")

    def stop_all(self):
        """Stop all motors immediately."""
        for motor_name in MOTORS:
            self.set_motor(motor_name, 0.0)
        self.get_logger().info("All motors stopped")

    def drive(self, x: float, y: float):
        """Drive using differential steering.
        
        Args:
            x: -1.0 to 1.0 (negative = left, positive = right)
            y: -1.0 to 1.0 (negative = backward, positive = forward)
        """
        # Differential drive calculation
        # y controls forward/backward, x controls turning
        left_speed = y + x
        right_speed = y - x
        
        # Clamp speeds to -1.0 to 1.0
        max_magnitude = max(abs(left_speed), abs(right_speed), 1.0)
        left_speed /= max_magnitude
        right_speed /= max_magnitude
        
        # Apply to motors
        self.set_motor("left_front", left_speed)
        self.set_motor("left_back", left_speed)
        self.set_motor("right_front", right_speed)
        self.set_motor("right_back", right_speed)
        
        self.get_logger().debug(f"Drive: x={x:.2f}, y={y:.2f} -> L={left_speed:.2f}, R={right_speed:.2f}")

    def handle_vector(self, msg: String):
        """Handle incoming vector commands from joystick."""
        self.last_command_time = self.get_clock().now()
        
        try:
            outer_data = json.loads(msg.data)
            # Handle nested JSON from rosbridge
            inner_data = json.loads(outer_data.get("data", "{}"))
            x = float(inner_data.get("x", 0))
            y = float(inner_data.get("y", 0))
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            self.get_logger().warn(f"Invalid vector data: {msg.data} - {e}")
            return
        
        # Clamp values to valid range
        x = max(-1.0, min(1.0, x))
        y = max(-1.0, min(1.0, y))
        
        self.get_logger().info(f"Vector: x={x:.2f}, y={y:.2f}")
        self.drive(x, y)

    def handle_estop(self, msg: String):
        """Handle emergency stop."""
        self.get_logger().warn("E-STOP ACTIVATED")
        self.stop_all()


def main(args=None):
    rclpy.init(args=args)
    motor_node = MotorControl()
    rclpy.spin(motor_node)
    motor_node.destroy_node()
    rclpy.shutdown()
