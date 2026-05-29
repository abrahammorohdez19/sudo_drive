from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """
    Control — pure pursuit visión → /qcar/user_command
    Requiere physical.launch.py corriendo.

    Parámetros principales:
        v_ref              — velocidad (m/s)
        lateral_offset_px  — px desde la línea amarilla hacia la derecha
                             sube = más a la derecha, baja = más cerca
        steer_gain         — ganancia (sube = más ángulo en curvas)
        lookahead_rows     — filas de lookahead (sube = más suave)

    Ejemplos:
        ros2 launch qcar_navigation control.launch.py
        ros2 launch qcar_navigation control.launch.py lateral_offset_px:=200.0
        ros2 launch qcar_navigation control.launch.py v_ref:=0.06 steer_gain:=18.0
    """
    return LaunchDescription([

        DeclareLaunchArgument('v_ref',             default_value='0.05',
            description='Velocidad de referencia (m/s).'),
        DeclareLaunchArgument('steer_gain',        default_value='18.0',
            description='Ganancia steering. Sube para curvas más cerradas.'),
        DeclareLaunchArgument('lookahead_rows',    default_value='60',
            description='Filas de lookahead. Sube = más suave, baja = más reactivo.'),
        DeclareLaunchArgument('max_steer_rate',    default_value='0.10',
            description='Rate limiter rad/ciclo.'),
        DeclareLaunchArgument('steer_alpha',       default_value='0.55',
            description='Filtro EMA. 1=sin filtro, 0=sin cambio.'),
        DeclareLaunchArgument('lateral_offset_px', default_value='220.0',
            description='Px a la derecha de la línea amarilla. '
                        'Sube = más separación.'),
        DeclareLaunchArgument('startup_cap_frames', default_value='60',
            description='Frames con steering limitado al arranque.'),
        DeclareLaunchArgument('startup_max_steer',  default_value='0.20',
            description='Steering máximo durante startup_cap_frames (rad).'),

        Node(
            package='qcar_navigation',
            executable='pure_pursuit_vision_node',
            name='pure_pursuit_vision',
            output='screen',
            parameters=[{
                'v_ref':              LaunchConfiguration('v_ref'),
                'steer_gain':         LaunchConfiguration('steer_gain'),
                'lookahead_rows':     LaunchConfiguration('lookahead_rows'),
                'max_steer_rate':     LaunchConfiguration('max_steer_rate'),
                'steer_alpha':        LaunchConfiguration('steer_alpha'),
                'lateral_offset_px':  LaunchConfiguration('lateral_offset_px'),
                'startup_cap_frames': LaunchConfiguration('startup_cap_frames'),
                'startup_max_steer':  LaunchConfiguration('startup_max_steer'),
                'k_curv_offset':      0.0,
            }],
        ),
    ])
