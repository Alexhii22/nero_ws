# Copyright 2025. PPO sim2sim 集成启动：Gazebo + 控制 + 目标发布 + RViz
# 用法: ros2 launch nero_description ppo_sim2sim.launch.py
# 启动 policy: ros2 run nero_description gazebo_ppo_play -p /path/to/policy.pt

from ament_index_python.packages import get_package_share_path

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    pkg_share = str(get_package_share_path("nero_description"))

    # 1. Gazebo + robot + controllers + spawn_reach_targets（已包含在 gazebo_control 中）
    gazebo_control_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([pkg_share, "/launch/gazebo_control.launch.py"]),
    )

    # 2. RViz（MarkerArray 已配置为 /reach_target_markers，bi_nero_gazebo.rviz）
    rviz_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([pkg_share, "/launch/rviz_gazebo.launch.py"]),
    )


    # RViz 延迟 8s 启动，等 Gazebo、robot 和 TF 树完全就绪
    # 增加延迟时间以避免时间倒退警告
    rviz_delayed = TimerAction(
        period=8.0,
        actions=[rviz_launch],
    )

    return LaunchDescription([
        gazebo_control_launch,
        rviz_delayed,
    ])
