"""
Avoidance System Launch File

Launches LIDAR-based obstacle avoidance with segmentation and rerouting.
Uses mock LIDAR for testing (no hardware required).

LifecycleNodes auto-activate via event handlers:
  - laser_segmentation: UNCONFIGURED → INACTIVE → ACTIVE (via launch events)
  - rerouting_node: UNCONFIGURED → INACTIVE → ACTIVE (via launch events)

Can also be manually controlled:
  ros2 lifecycle set /rerouting_node configure
  ros2 lifecycle set /rerouting_node activate

Nodes:
  - mock_lidar_publisher: Synthetic LIDAR data for testing
  - laser_segmentation: Converts /scan to /segments (LifecycleNode)
  - rerouting_node: Obstacle avoidance decision maker (LifecycleNode)
  - foxglove_bridge: 3D visualization on port 8765
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, EmitEvent, RegisterEventHandler
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node, LifecycleNode
from launch_ros.event_handlers import OnStateTransition
from launch_ros.events.lifecycle import ChangeState
import launch.events
import lifecycle_msgs.msg


def generate_launch_description():
    # Include laser_segmentation launch file
    segmentation_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('laser_segmentation'),
                'launch',
                'segmentation.launch.py'
            )
        ),
        launch_arguments={
            'log_level': 'info',
        }.items()
    )

    # Rerouting node (LifecycleNode for recovery support)
    rerouting_node = LifecycleNode(
        package='avoidance_rerouting',
        executable='rerouting',
        name='rerouting_node',
        output='screen',
        namespace='',
        emulate_tty=True,
        respawn=True,
        respawn_delay=2,
    )

    # When rerouting_node reaches INACTIVE, automatically activate it
    register_event_handler_rerouting_inactive = RegisterEventHandler(
        OnStateTransition(
            target_lifecycle_node=rerouting_node,
            goal_state='inactive',
            entities=[
                EmitEvent(event=ChangeState(
                    lifecycle_node_matcher=launch.events.matches_action(rerouting_node),
                    transition_id=lifecycle_msgs.msg.Transition.TRANSITION_ACTIVATE,
                )),
            ],
        )
    )

    # Immediately configure rerouting_node when launch starts
    emit_event_rerouting_configure = EmitEvent(
        event=ChangeState(
            lifecycle_node_matcher=launch.events.matches_action(rerouting_node),
            transition_id=lifecycle_msgs.msg.Transition.TRANSITION_CONFIGURE,
        )
    )

    return LaunchDescription([
        # Mock LIDAR publisher (synthetic data for testing)
        Node(
            package='avoidance_rerouting',
            executable='mock_lidar_publisher',
            name='mock_lidar_publisher',
            output='screen',
        ),

        # Rerouting node event handlers and activation
        register_event_handler_rerouting_inactive,
        emit_event_rerouting_configure,
        rerouting_node,

        # Foxglove visualization bridge
        Node(
            package='foxglove_bridge',
            executable='foxglove_bridge',
            name='foxglove_bridge',
            output='screen',
            parameters=[{'port': 8765}],
        ),

        # Laser segmentation (external package - already LifecycleNode)
        segmentation_launch,
    ])
