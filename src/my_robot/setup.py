from setuptools import setup

package_name = 'my_robot'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Добавьте эти строки:
        ('share/' + package_name + '/launch', [
        'launch/launch_robot.launch.py',
        'launch/launch_sim.launch.py',
        'launch/rsp.launch.py'
    ]),
        ('share/' + package_name + '/description', [
        'description/robot.urdf.xacro',
        'description/robot_core.xacro',
        'description/lidar.xacro',
        'description/inertial_macros.xacro',
        'description/gazebo_control.xacro',
        'description/ros2_control.xacro'
        ]),
        ('share/' + package_name + '/worlds', ['world/WareHouseV2.world']),
        ('share/' + package_name + '/config', ['config/my_controllers.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='your_email@example.com',
    description='My diff drive robot package',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        ],
    },
)
