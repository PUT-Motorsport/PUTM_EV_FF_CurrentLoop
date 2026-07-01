from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    pkg_share   = FindPackageShare('current_predictor')
    config_file = PathJoinSubstitution([pkg_share, 'config', 'params.yaml'])
    weights_dir = PathJoinSubstitution([pkg_share, 'model_weights'])

    return LaunchDescription([
        DeclareLaunchArgument(
            'current_topic',
            default_value='current_sensors_data',
            description='Topic with raw inverter currents [I_FL, I_FR, I_RL, I_RR]',
        ),
        DeclareLaunchArgument(
            'torque_topic',
            default_value='torque_setpoints',
            description='Topic with torque setpoints [T_FL, T_FR, T_RL, T_RR] in Nm',
        ),
        DeclareLaunchArgument(
            'voltage_topic',
            default_value='bus_voltage',
            description='Topic with DC bus voltage in Volts',
        ),
        Node(
            package='current_predictor',
            executable='current_predictor_node',
            name='current_predictor',
            parameters=[
                config_file,
                {
                    'current_topic': LaunchConfiguration('current_topic'),
                    'torque_topic':  LaunchConfiguration('torque_topic'),
                    'voltage_topic': LaunchConfiguration('voltage_topic'),
                    'weights_dir':   weights_dir,
                },
            ],
            output='screen',
            emulate_tty=True,
        ),
    ])
