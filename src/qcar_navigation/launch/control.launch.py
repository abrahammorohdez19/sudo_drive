from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """
    Control — visión pura → /qcar/user_command
    Requiere physical.launch.py corriendo.

    Parámetros de ajuste rápido:
        nominal_speed     — velocidad en recta (m/s)
        lane_offset_px    — px a la derecha de la línea (sube = más a la derecha)
        steer_gain        — ganancia de steering (sube = más ángulo)
        lookahead_rows_max — lookahead en recta (sube = más suave)
        left_curve_steer_boost — boost extra en curvas izquierda

    Ejemplo:
        ros2 launch qcar_navigation control.launch.py nominal_speed:=0.06 lane_offset_px:=170.0
    """
    return LaunchDescription([

        # ── Velocidad ─────────────────────────────────────────────────
        DeclareLaunchArgument('nominal_speed',           default_value='0.055',
            description='Velocidad en recta (m/s).'),
        DeclareLaunchArgument('min_speed',               default_value='0.047',
            description='Velocidad mínima en curva pronunciada (m/s).'),
        DeclareLaunchArgument('max_speed',               default_value='0.065',
            description='Velocidad máxima absoluta (m/s).'),
        DeclareLaunchArgument('curvature_slowdown_gain', default_value='0.4',
            description='Ganancia de reducción de vel. por curvatura.'),

        # ── Steering ──────────────────────────────────────────────────
        DeclareLaunchArgument('steer_gain',      default_value='18.0',
            description='Ganancia de steering (px). Sube = más ángulo.'),
        DeclareLaunchArgument('max_steer_rate',  default_value='0.03',
            description='Rate limiter rad/ciclo.'),
        DeclareLaunchArgument('steer_alpha',     default_value='0.30',
            description='EMA filter: 1=sin filtro, 0=sin cambio.'),

        # ── Lookahead adaptativo ──────────────────────────────────────
        DeclareLaunchArgument('lookahead_rows_min', default_value='45',
            description='Lookahead mínimo (curva máxima).'),
        DeclareLaunchArgument('lookahead_rows_max', default_value='80',
            description='Lookahead máximo (recta).'),
        DeclareLaunchArgument('k_curv_lookahead',   default_value='1200.0',
            description='Reducción de lookahead por unidad de curvatura.'),

        # ── Lane offset ───────────────────────────────────────────────
        DeclareLaunchArgument('lane_offset_px', default_value='155.0',
            description='Px a la derecha de la línea amarilla. '
                        'Sube = más separación. Baja = más cerca.'),

        # ── Anticipación curva izquierda ──────────────────────────────
        DeclareLaunchArgument('left_curve_steer_boost', default_value='0.0',
            description='Boost de steering en curva izquierda (0=desactivado).'),
        DeclareLaunchArgument('left_curve_lookahead_reduction', default_value='10',
            description='Filas extra reducidas en curva izquierda.'),

        # ── Nodo ──────────────────────────────────────────────────────
        Node(
            package='qcar_navigation',
            executable='pure_pursuit_vision_node',
            name='pure_pursuit_vision',
            output='screen',
            parameters=[{
                'nominal_speed':               LaunchConfiguration('nominal_speed'),
                'min_speed':                   LaunchConfiguration('min_speed'),
                'max_speed':                   LaunchConfiguration('max_speed'),
                'curvature_slowdown_gain':     LaunchConfiguration('curvature_slowdown_gain'),
                'steer_gain':                  LaunchConfiguration('steer_gain'),
                'max_steer_rate':              LaunchConfiguration('max_steer_rate'),
                'steer_alpha':                 LaunchConfiguration('steer_alpha'),
                'lookahead_rows_min':          LaunchConfiguration('lookahead_rows_min'),
                'lookahead_rows_max':          LaunchConfiguration('lookahead_rows_max'),
                'k_curv_lookahead':            LaunchConfiguration('k_curv_lookahead'),
                'lane_offset_px':              LaunchConfiguration('lane_offset_px'),
                'left_curve_steer_boost':      LaunchConfiguration('left_curve_steer_boost'),
                'left_curve_lookahead_reduction': LaunchConfiguration('left_curve_lookahead_reduction'),
                'startup_cap_frames':          40,
                'startup_max_steer':           0.15,
                'warmup_frames':               60,
            }],
        ),
    ])
