from setuptools import find_packages, setup
from glob import glob 
import os 

package_name = 'qcar_navigation'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), 
         glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'rviz'),   # AGREGAR ESTO
         glob('rviz/*.rviz')),                           # AGREGAR ESTO
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='abrahammh19 nataliahaha',
    maintainer_email='abrahammorohdez@gmail.com',
    description='Package from Sudo Drive team for 8th semester Autonomous Taxi Task Project',
    license='Apache 2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'undistorted_node = qcar_navigation.undistorted_node:main',
            'lane_visualizer_node = qcar_navigation.lane_visualizer_node:main',
            'lane_detection_sw_node = qcar_navigation.lane_detection_sw_node:main'

        ],
    },
)
