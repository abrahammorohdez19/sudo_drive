import os
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    
    # RUTA CORREGIDA CON LA SUBCARPETA DEL DATASET
    traffic_node_params = {
        'model_path': '/home/evan/workspaces/quanser_acc_ws/src/compe/entrenamiento_yolo/YOYO.v1i.yolov8/best.pt',
        'confidence_threshold': 0.50,
        'xbee_port': '/dev/ttyUSB0',
        'xbee_enabled': True
    }

    # Declaramos el nodo para que el launch lo ejecute
    traffic_detector_node = Node(
        package='trafficlight_pkg',
        executable='traffic_test_node',
        name='traffic_unified_node',
        output='screen',
        parameters=[traffic_node_params]
    )

    return LaunchDescription([
        traffic_detector_node
    ])