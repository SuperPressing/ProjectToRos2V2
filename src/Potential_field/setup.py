from setuptools import find_packages, setup

package_name = 'Potential_field'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='neo',
    maintainer_email='rok199200@gmail.com',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        'Reguluator = Potential_field.Reguluator:main',
        'Trac = Potential_field.Traectory:main',
        'A_start = Potential_field.A_star:main',
        'Mark = Potential_field.Markov_solutions:main',
        'Find = Potential_field.Finding_obstacles:main',
        'Update_map = Potential_field.Update_map:main',
        ],
    },
)
