#!/usr/bin/env python3
import rclpy
from rclpy.lifecycle import Node as LifecycleNode
from rclpy.lifecycle import State, TransitionCallbackReturn
from std_msgs.msg import Float32

try:
    import Jetson.GPIO as GPIO
    HARDWARE_AVAILABLE = True
except Exception:
    # Mock GPIO for testing without Jetson hardware
    GPIO = None
    HARDWARE_AVAILABLE = False
    
    class MockGPIO:
        BOARD = 'BOARD'
        OUT = 'OUT'
        def setmode(self, mode):
            pass
        def setup(self, pin, mode):
            pass
        def PWM(self, pin, freq):
            return MockPWM()
    
    class MockPWM:
        def start(self, duty):
            pass
        def ChangeDutyCycle(self, duty):
            pass
        def stop(self):
            pass
    
    GPIO = MockGPIO()

SERVO_PIN = 33
PWM_FREQ = 50

# Pitch angle corresponding to duty cycle positions
PITCH_LOW = 0.0   # duty_low = 0
PITCH_HIGH = 20.0   # duty_high = 10


class ServoDriver(LifecycleNode):
    def __init__(self):
        super().__init__('lidar_servo')
        
        self.pwm = None
        self.timer = None
        self.angle_pub = None
        self.duty_low = 0
        self.duty_high = 10
        self.current_high = False
        
        self.get_logger().info("Servo Driver lifecycle node initialized")

    def on_configure(self, state: State) -> TransitionCallbackReturn:
        """Configure GPIO and publisher."""
        self.get_logger().info('on_configure called')
        try:
            GPIO.setmode(GPIO.BOARD)
            GPIO.setup(SERVO_PIN, GPIO.OUT)
            self.pwm = GPIO.PWM(SERVO_PIN, PWM_FREQ)
            self.pwm.start(3.0)
            
            # Publisher for current pitch angle
            self.angle_pub = self.create_publisher(Float32, '/lidar_pitch', 10)
            
            self.get_logger().info("Servo Driver configured")
            return TransitionCallbackReturn.SUCCESS
        except Exception as e:
            self.get_logger().error(f'Configure failed: {e}')
            return TransitionCallbackReturn.FAILURE

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        """Start sweep timer."""
        self.get_logger().info('on_activate called')
        try:
            self.timer = self.create_timer(0.1, self.sweep_callback)
            self.get_logger().info('Servo Driver activated - sweeping')
            return TransitionCallbackReturn.SUCCESS
        except Exception as e:
            self.get_logger().error(f'Activate failed: {e}')
            return TransitionCallbackReturn.FAILURE

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        """Stop sweep timer."""
        self.get_logger().info('on_deactivate called')
        try:
            if self.timer:
                self.destroy_timer(self.timer)
                self.timer = None
            self.get_logger().info('Servo Driver deactivated')
            return TransitionCallbackReturn.SUCCESS
        except Exception as e:
            self.get_logger().error(f'Deactivate failed: {e}')
            return TransitionCallbackReturn.FAILURE

    def on_cleanup(self, state: State) -> TransitionCallbackReturn:
        """Clean up GPIO."""
        self.get_logger().info('on_cleanup called')
        try:
            if self.pwm:
                self.pwm.stop()
            GPIO.cleanup()
            self.get_logger().info('GPIO cleanup done')
            return TransitionCallbackReturn.SUCCESS
        except Exception as e:
            self.get_logger().error(f'Cleanup failed: {e}')
            return TransitionCallbackReturn.FAILURE

    def on_shutdown(self, state: State) -> TransitionCallbackReturn:
        """Shutdown the node."""
        self.get_logger().info('on_shutdown called')
        return TransitionCallbackReturn.SUCCESS

    def sweep_callback(self):
        """Sweep servo and publish pitch angle."""
        if self.current_high:
            self.pwm.ChangeDutyCycle(self.duty_low)
            pitch = PITCH_LOW
        else:
            self.pwm.ChangeDutyCycle(self.duty_high)
            pitch = PITCH_HIGH
        self.current_high = not self.current_high

        # Publish pitch angle (timestamp is when message is published)
        msg = Float32()
        msg.data = pitch
        self.angle_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    driver = ServoDriver()
    rclpy.spin(driver)
    driver.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
