import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess, RegisterEventHandler
from launch.event_handlers import OnProcessStart, OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    package_name = 'articubot_one'

    # Подключаем robot_state_publisher
    rsp = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [os.path.join(get_package_share_directory(package_name), 'launch', 'rsp.launch.py')]
        ),
        launch_arguments={'use_sim_time': 'true'}.items()
    )

    # Создаем Node для robot_state_publisher отдельно
    # robot_state_publisher_node = Node(
    #     package='robot_state_publisher',
    #     executable='robot_state_publisher',
    #     name='robot_state_publisher',
    #     output='screen',
    #     parameters=[{
    #         'use_sim_time': True
    #     }]
    # )

    # Подключаем Gazebo
    gazebo_params_file = os.path.join(get_package_share_directory(package_name),'config','gazebo_params.yaml')
    world_path = os.path.join(get_package_share_directory(package_name), 'worlds', 'WareHouseV2.world')
    # Include the Gazebo launch file, provided by the gazebo_ros package
    gazebo = IncludeLaunchDescription(
                PythonLaunchDescriptionSource([os.path.join(
                    get_package_share_directory('gazebo_ros'), 'launch', 'gazebo.launch.py')]),
                    launch_arguments={'extra_gazebo_args': '--ros-args --params-file ' + gazebo_params_file,
                                      'world': world_path}.items()
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
        shell=True,
        output='screen'
    )

    joint_broad_spawner = ExecuteProcess(
        cmd=['ros2', 'run', 'controller_manager', 'spawner', 'joint_broad'],
        shell=True,
        output='screen'
    )

    # === РЕГИСТРАЦИЯ ОБРАБОТЧИКОВ СОБЫТИЙ ===
    # Спавним модель после запуска robot_state_publisher_node
    # delayed_spawn_entity = RegisterEventHandler(
    #     event_handler=OnProcessStart(
    #         target_action=robot_state_publisher_node,
    #         on_start=[spawn_entity],
    #     )
    # )

    # diff_drive после спавна модели
    delayed_diff_drive_spawner = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=spawn_entity,
            on_exit=[diff_drive_spawner],
        )
    )

    # joint_broad после diff_drive
    delayed_joint_broad_spawner = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=diff_drive_spawner,
            on_exit=[joint_broad_spawner],
        )
    )

    # return LaunchDescription([
    #     # Добавляем основные действия
    #     robot_state_publisher_node,
    #     gazebo,

    #     # Регистрируем обработчики событий
    #     delayed_spawn_entity,
    #     delayed_diff_drive_spawner,
    #     delayed_joint_broad_spawner,
    # ])
    # === ДОБАВЛЯЕМ SLAM TOOLBOX С ЗАДЕРЖКОЙ ===
    slam_toolbox_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [os.path.join(get_package_share_directory("slam_toolbox"), 'launch', 'online_async_launch.py')]
        ),
        launch_arguments={
            'use_sim_time': 'true',
            'params_file': os.path.join(get_package_share_directory(package_name), 'config', 'mapper_params_online_async.yaml')
        }.items()
    )

    # Запускаем SLAM после `joint_broad`
    delayed_slam_toolbox = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=spawn_entity,
            on_exit=[slam_toolbox_node],
        )
    )
    # Launch them all!
    return LaunchDescription([
        rsp,
        # joystick,
        # twist_mux,
        #robot_state_publisher_node,
        gazebo,
        spawn_entity,
        #delayed_slam_toolbox,
        #delayed_diff_drive_spawner,
        #delayed_joint_broad_spawner,
    ])