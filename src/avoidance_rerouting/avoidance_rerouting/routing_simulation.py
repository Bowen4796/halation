import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import numpy as np
import math

class BoundingBox:
    """Class to store bounding box data for detected objects"""
    def __init__(self, height, width, distance, angle, rover_pos):
        """
        Parameters:
        - height: Height of the cube (z-dimension)
        - width: Width of the cube (perpendicular to distance vector)
        - distance: Distance from rover to object center (to the middle of width and height)
        - angle: Angle of the distance vector from rover's heading (degrees)
                 The width will be perpendicular to this angle
        - rover_pos: Tuple of (x, y, theta) for rover position
        """
        rover_x, rover_y, rover_theta = rover_pos
        
        # Store dimensions
        self.height = height
        self.width = width
        self.length = width  # Assuming square base, length = width
        
        # Convert angle to radians and make it relative to rover's heading
        # This is the angle of the distance vector
        distance_angle_rad = math.radians(angle) + rover_theta
        
        # Calculate global position (center of the box)
        self.x = rover_x + distance * math.cos(distance_angle_rad)
        self.y = rover_y + distance * math.sin(distance_angle_rad)
        
        # The box rotation: width is perpendicular to the distance vector
        # So the box's rotation angle is the distance angle
        self.angle = distance_angle_rad

class RoutingSimulation(Node):
    def __init__(self):
        super().__init__('routing_simulation')
        
        # Publisher for bounding box markers
        self.marker_publisher = self.create_publisher(
            MarkerArray, 
            '/object_bounding_boxes', 
            10
        )
        
        # Publisher for rover position marker
        self.rover_marker_publisher = self.create_publisher(
            Marker,
            '/rover_position',
            10
        )
        
        # Timer to publish at regular intervals
        self.timer = self.create_timer(0.1, self.publish_markers)
        
        # Rover position (x, y, theta)
        self.rover_x = 0.0
        self.rover_y = 0.0
        self.rover_theta = 0.0  # Rover's heading angle (radians)
        
        # List to store all detected bounding boxes
        self.bounding_boxes = []
        
        # Create test bounding boxes
        self.create_test_bounding_boxes()
        
        self.get_logger().info('Routing Simulation Node Started')

    def define_obstacle(self, height, width, distance, angle):
        """
        Define and add a new obstacle bounding box.
        
        Parameters:
        - height: Height of the cube (z-dimension)
        - width: Width of the cube
        - distance: Distance from rover to object center
        - angle: Angle relative to rover's heading (degrees)
        """
        rover_pos = (self.rover_x, self.rover_y, self.rover_theta)
        
        bbox = BoundingBox(
            height=height,
            width=width,
            distance=distance,
            angle=angle,
            rover_pos=rover_pos
        )
        self.bounding_boxes.append(bbox)

    def create_test_bounding_boxes(self):
        """
        Create initial test bounding boxes for simulation.
        These are stored in self.bounding_boxes list.
        """
        self.rover_theta = math.pi / 6
        self.rover_x = .5
        
        self.define_obstacle(1.0, 0.5, 3.0, -30)
        
        self.define_obstacle(0.8, 0.6, 1.0, 0)

    def create_bounding_box_marker(self, marker_id, bbox):
        """
        Create a bounding box marker for a detected object.
        
        Parameters:
        - marker_id: Unique ID for this marker
        - bbox: BoundingBox object with global position and dimensions
        
        Returns:
        - Marker object representing the bounding box
        """
        marker = Marker()
        marker.header.frame_id = "laser"  # Same frame as lidar
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "object_bounding_boxes"
        marker.id = marker_id
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        
        # Use bbox global position
        obj_center_x = bbox.x
        obj_center_y = bbox.y
        obj_center_z = bbox.height / 2.0
        
        # Define the 8 corners of the cube in local frame
        # Local frame: length is along the distance direction, width is perpendicular
        half_width = bbox.width / 2.0
        half_length = bbox.length / 2.0
        half_height = bbox.height / 2.0
        
        # Corners in local object frame
        # Length (x-local) points along the distance vector from rover
        # Width (y-local) is perpendicular to the distance vector
        corners_local = [
            [-half_length, -half_width, -half_height],  # 0: bottom-back-left
            [half_length, -half_width, -half_height],   # 1: bottom-front-left
            [half_length, half_width, -half_height],    # 2: bottom-front-right
            [-half_length, half_width, -half_height],   # 3: bottom-back-right
            [-half_length, -half_width, half_height],   # 4: top-back-left
            [half_length, -half_width, half_height],    # 5: top-front-left
            [half_length, half_width, half_height],     # 6: top-front-right
            [-half_length, half_width, half_height],    # 7: top-back-right
        ]
        
        # Rotate corners around z-axis by bbox.angle and translate to global position
        corners_global = []
        cos_theta = math.cos(bbox.angle)
        sin_theta = math.sin(bbox.angle)
        
        for corner in corners_local:
            # Rotate in XY plane around z-axis
            x_rot = corner[0] * cos_theta - corner[1] * sin_theta
            y_rot = corner[0] * sin_theta + corner[1] * cos_theta
            
            # Translate to global position
            point = Point()
            point.x = obj_center_x + x_rot
            point.y = obj_center_y + y_rot
            point.z = obj_center_z + corner[2]
            corners_global.append(point)
        
        # Create edges for the bounding box (12 edges for a cube)
        edges = [
            # Bottom face
            (0, 1), (1, 2), (2, 3), (3, 0),
            # Top face
            (4, 5), (5, 6), (6, 7), (7, 4),
            # Vertical edges
            (0, 4), (1, 5), (2, 6), (3, 7)
        ]
        
        # Draw the 12 edges as solid lines
        # LINE_LIST requires pairs of points: [start1, end1, start2, end2, ...]
        marker.points = []
        for edge in edges:
            p1 = corners_global[edge[0]]
            p2 = corners_global[edge[1]]
            
            # Add start and end point of this edge
            marker.points.append(p1)
            marker.points.append(p2)
        
        # Set marker properties
        marker.scale.x = 0.02  # Line width
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        
        return marker

    def create_rover_marker(self):
        """
        Create a bounding box marker for the rover position.
        Rover is a 0.5m x 0.5m x 0.5m cube.
        
        Returns:
        - Marker object representing the rover
        """
        marker = Marker()
        marker.header.frame_id = "laser"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "rover"
        marker.id = 0
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        
        # Rover position
        rover_center_x = self.rover_x
        rover_center_y = self.rover_y
        rover_center_z = 0.25  # Half of 0.5m height
        
        # Define the 8 corners of the rover cube
        half_size = 0.25  # 0.5m / 2
        
        # Corners in local rover frame
        # Length (x-local) points along rover's heading
        # Width (y-local) is perpendicular to heading
        corners_local = [
            [-half_size, -half_size, -half_size],  # 0: bottom-back-left
            [half_size, -half_size, -half_size],   # 1: bottom-front-left
            [half_size, half_size, -half_size],    # 2: bottom-front-right
            [-half_size, half_size, -half_size],   # 3: bottom-back-right
            [-half_size, -half_size, half_size],   # 4: top-back-left
            [half_size, -half_size, half_size],    # 5: top-front-left
            [half_size, half_size, half_size],     # 6: top-front-right
            [-half_size, half_size, half_size],    # 7: top-back-right
        ]
        
        # Rotate corners around z-axis by rover heading
        corners_global = []
        cos_theta = math.cos(self.rover_theta)
        sin_theta = math.sin(self.rover_theta)
        
        for corner in corners_local:
            # Rotate in XY plane around z-axis
            x_rot = corner[0] * cos_theta - corner[1] * sin_theta
            y_rot = corner[0] * sin_theta + corner[1] * cos_theta
            
            # Translate to global position
            point = Point()
            point.x = rover_center_x + x_rot
            point.y = rover_center_y + y_rot
            point.z = rover_center_z + corner[2]
            corners_global.append(point)
        
        # Create edges for the bounding box (12 edges for a cube)
        edges = [
            # Bottom face
            (0, 1), (1, 2), (2, 3), (3, 0),
            # Top face
            (4, 5), (5, 6), (6, 7), (7, 4),
            # Vertical edges
            (0, 4), (1, 5), (2, 6), (3, 7)
        ]
        
        # Draw the 12 edges as solid lines
        marker.points = []
        for edge in edges:
            p1 = corners_global[edge[0]]
            p2 = corners_global[edge[1]]
            
            # Add start and end point of this edge
            marker.points.append(p1)
            marker.points.append(p2)
        
        # Set marker properties - different color for rover (blue)
        marker.scale.x = 0.02  # Line width
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.color.a = 1.0
        
        return marker

    def publish_markers(self):
        """
        Main publishing function that creates and publishes both
        rover position and bounding boxes.
        """
        # Publish rover position
        rover_marker = self.create_rover_marker()
        self.rover_marker_publisher.publish(rover_marker)
        
        # Publish bounding boxes
        marker_array = MarkerArray()
        
        # Create markers for all stored bounding boxes
        for i, bbox in enumerate(self.bounding_boxes):
            marker = self.create_bounding_box_marker(
                marker_id=i,
                bbox=bbox
            )
            marker_array.markers.append(marker)
        
        # Publish the marker array
        self.marker_publisher.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    node = RoutingSimulation()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()