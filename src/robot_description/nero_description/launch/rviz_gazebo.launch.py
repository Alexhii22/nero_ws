# Copyright 2025. RViz2 启动文件，配合 Gazebo 仿真使用。
# 预配置：RobotModel (/robot_description)、TF、MarkerArray (/reach_target_markers)、Grid。
# 用法: ros2 launch nero_description rviz_gazebo.launch.py
# 需先启动 Gazebo: ros2 launch nero_description gazebo_control.launch.py

from ament_index_python.packages import get_package_share_path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_path("nero_description")
    default_rviz_config = str(pkg_share / "rviz" / "bi_nero_gazebo.rviz")

    rviz_config_arg = DeclareLaunchArgument(
        "rviz_config",
        default_value=default_rviz_config,
        description="Path to RViz config file",
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", LaunchConfiguration("rviz_config")],
        parameters=[{"use_sim_time": True}],
    )

    return LaunchDescription([
        rviz_config_arg,
        rviz_node,
    ])
