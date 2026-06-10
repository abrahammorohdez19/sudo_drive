from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'trafficlight_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # ESTA LÍNEA ES CRUCIAL: Instala los archivos de lanzamiento (.launch.py) en el share del workspace
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='evan',
    maintainer_email='evan@todo.todo',
    description='Paquete de deteccion de semaforos con YOLOv8 para QCar',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # Registra el comando para que ROS sepa dónde buscar la función main()
            'traffic_test_node = trafficlight_pkg.traffic_test_node:main'
        ],
    },
)
