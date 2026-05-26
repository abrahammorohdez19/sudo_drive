from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """
    Nodo de movimiento: visión pura → /qcar/user_command.
    Requiere que physical.launch.py ya esté corriendo.

    Uso:
        ros2 launch qcar_navigation control.launch.py
        ros2 launch qcar_navigation control.launch.py v_ref:=0.05 steer_gain:=18.0
    """
    return LaunchDescription([

        DeclareLaunchArgument(
            'v_ref',
            default_value='0.05',
            description='Velocidad de referencia (m/s).'
        ),
        DeclareLaunchArgument(
            'steer_gain',
            default_value='16.0',
            description='Ganancia steering pure pursuit en px. Baja = menos agresivo.'
        ),
        DeclareLaunchArgument(
            'lookahead_rows',
            default_value='60',
            description='Filas de lookahead. Sube para suavizar, baja para más reactividad.'
        ),
        DeclareLaunchArgument(
            'max_steer_rate',
            default_value='0.10',
            description='Rate limiter steering rad/ciclo. Baja = cambios más lentos.'
        ),
        DeclareLaunchArgument(
            'steer_alpha',
            default_value='0.55',
            description='EMA filter: 1=sin filtro, 0=sin cambio.'
        ),
        DeclareLaunchArgument(
            'k_curv_offset',
            default_value='3000.0',
            description='Offset lateral adaptivo por curvatura (0 = desactivado).'
        ),
        DeclareLaunchArgument(
            'lateral_offset_px',
            default_value='220.0',
            description='Desplazamiento lateral respecto a la línea amarilla (px). '
                        'Sube para alejarte más a la derecha, baja para acercarte.'
        ),

        # ── Pure Pursuit visión → /qcar/user_command ─────────────────
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
                'k_curv_offset':      LaunchConfiguration('k_curv_offset'),
                'lateral_offset_px':  LaunchConfiguration('lateral_offset_px'),
                'startup_cap_frames': 0,
            }],
        ),
    ])
