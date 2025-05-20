import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    package_name = 'my_robot'

    # Подключаем robot_state_publisher из rsp.launch.py
    rsp = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(get_package_share_directory(package_name), 'launch', 'launch_robot.launch.py')
        ]),
        launch_arguments={'use_sim_time': 'true'}.items()
    )

    # Запуск Gazebo
    gazebo_params_file = os.path.join(get_package_share_directory(package_name), 'config', 'gazebo_params.yaml')
    world_path = os.path.join(get_package_share_directory(package_name), 'worlds', 'empty.world')

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(get_package_share_directory('gazebo_ros'), 'launch', 'gazebo.launch.py')
        ]),
        launch_arguments={
            'extra_gazebo_args': '--ros-args --params-file ' + gazebo_params_file,
            'world': world_path
        }.items()
    )

    # Спавним модель робота в Gazebo
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=['-topic', 'robot_description', '-entity', 'my_bot'],
        output='screen'
    )

    # Контроллеры
    diff_drive_spawner = ExecuteProcess(
        cmd=['ros2', 'run', 'controller_manager', 'spawner', 'diff_cont'],
        output='screen'
    )

    joint_broad_spawner = ExecuteProcess(
        cmd=['ros2', 'run', 'controller_manager', 'spawner', 'joint_broad'],
        output='screen'
    )

    # SLAM Toolbox
    slam_toolbox_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(get_package_share_directory("slam_toolbox"), 'launch', 'online_async_launch.py')
        ]),
        launch_arguments={
            'use_sim_time': 'true',
            'params_file': os.path.join(get_package_share_directory(package_name), 'config', 'mapper_params_online_async.yaml')
        }.items()
    )

    # === РЕГИСТРАЦИЯ ОБРАБОТЧИКОВ СОБЫТИЙ ===
    delayed_slam_toolbox = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=spawn_entity,
            on_exit=[slam_toolbox_node],
        )
    )

    delayed_diff_drive_spawner = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=spawn_entity,
            on_exit=[diff_drive_spawner],
        )
    )

    delayed_joint_broad_spawner = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=diff_drive_spawner,
            on_exit=[joint_broad_spawner],
        )
    )

    # === LAUNCH DESCRIPTION ===
    return LaunchDescription([
        rsp,
        gazebo,
        spawn_entity,
        delayed_slam_toolbox,
        delayed_diff_drive_spawner,
        delayed_joint_broad_spawner,
    ])
