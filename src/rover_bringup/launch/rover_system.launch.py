from launch import LaunchDescription
from launch.actions import EmitEvent, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch_ros.actions import Node, LifecycleNode
from launch_ros.event_handlers import OnStateTransition
from launch_ros.events.lifecycle import ChangeState
import launch.events
import lifecycle_msgs.msg


def generate_launch_description():
    # Motor control lifecycle node
    motor_control = LifecycleNode(
        package='rover_control',
        executable='motor_control',
        name='motor_control',
        output='screen',
        namespace='',
        emulate_tty=True,
        respawn=True,
        respawn_delay=2,
    )
    
    # When motor_control is respawned (process exits), configure it
    register_event_handler_motor_exit = RegisterEventHandler(
        OnProcessExit(
            target_action=motor_control,
            on_exit=[
                EmitEvent(event=ChangeState(
                    lifecycle_node_matcher=launch.events.matches_action(motor_control),
                    transition_id=lifecycle_msgs.msg.Transition.TRANSITION_CONFIGURE,
                )),
            ],
        )
    )
    
    # When motor_control reaches INACTIVE, automatically activate it
    register_event_handler_motor_inactive = RegisterEventHandler(
        OnStateTransition(
            target_lifecycle_node=motor_control,
            goal_state='inactive',
            entities=[
                EmitEvent(event=ChangeState(
                    lifecycle_node_matcher=launch.events.matches_action(motor_control),
                    transition_id=lifecycle_msgs.msg.Transition.TRANSITION_ACTIVATE,
                )),
            ],
        )
    )
    
    # Immediately configure motor_control when launch starts
    emit_event_motor_configure = EmitEvent(
        event=ChangeState(
            lifecycle_node_matcher=launch.events.matches_action(motor_control),
            transition_id=lifecycle_msgs.msg.Transition.TRANSITION_CONFIGURE,
        )
    )
    
    # Lidar servo lifecycle node
    lidar_servo = LifecycleNode(
        package='rover_sensing',
        executable='lidar_servo',
        name='lidar_servo',
        output='screen',
        namespace='',
        emulate_tty=True,
        respawn=True,
        respawn_delay=2,
    )
    
    # When lidar_servo is respawned (process exits), configure it
    register_event_handler_servo_exit = RegisterEventHandler(
        OnProcessExit(
            target_action=lidar_servo,
            on_exit=[
                EmitEvent(event=ChangeState(
                    lifecycle_node_matcher=launch.events.matches_action(lidar_servo),
                    transition_id=lifecycle_msgs.msg.Transition.TRANSITION_CONFIGURE,
                )),
            ],
        )
    )
    
    # When lidar_servo reaches INACTIVE, automatically activate it
    register_event_handler_servo_inactive = RegisterEventHandler(
        OnStateTransition(
            target_lifecycle_node=lidar_servo,
            goal_state='inactive',
            entities=[
                EmitEvent(event=ChangeState(
                    lifecycle_node_matcher=launch.events.matches_action(lidar_servo),
                    transition_id=lifecycle_msgs.msg.Transition.TRANSITION_ACTIVATE,
                )),
            ],
        )
    )
    
    # Immediately configure lidar_servo when launch starts
    emit_event_servo_configure = EmitEvent(
        event=ChangeState(
            lifecycle_node_matcher=launch.events.matches_action(lidar_servo),
            transition_id=lifecycle_msgs.msg.Transition.TRANSITION_CONFIGURE,
        )
    )

    return LaunchDescription([
        # ===== INFRASTRUCTURE NODES =====
        Node(
            package='rover_bringup',
            executable='rover_bringup',
            name='bringup_node',
            output='screen',
        ),
        Node(
            package='rover_bringup',
            executable='memory_monitor',
            name='memory_monitor',
            output='screen',
        ),
        Node(
            package='rosbridge_server',
            executable='rosbridge_websocket',
            name='rosbridge_websocket',
            output='screen',
            parameters=[{'port': 9095}],
        ),
        Node(
            package='rover_station_sync',
            executable='fastapi_proxy_node',
            name='fastapi_proxy_node',
            output='screen',
        ),
        
        # ===== LIFECYCLE NODES (Critical System Components) =====
        # Motor control with auto-activation and respawn
        register_event_handler_motor_exit,
        register_event_handler_motor_inactive,
        emit_event_motor_configure,
        motor_control,
        
        # Lidar servo with auto-activation and respawn
        register_event_handler_servo_exit,
        register_event_handler_servo_inactive,
        emit_event_servo_configure,
        lidar_servo,
        
        # ===== RECOVERY SYSTEM NODES =====
        Node(
            package='rover_bringup',
            executable='health_monitor',
            name='health_monitor',
            output='screen',
            emulate_tty=True,
        ),
        
        Node(
            package='rover_bringup',
            executable='recovery_manager',
            name='recovery_manager',
            output='screen',
            emulate_tty=True,
        ),
    ])
