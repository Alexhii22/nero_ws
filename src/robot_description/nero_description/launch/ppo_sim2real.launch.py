# Copyright 2025. PPO sim2real 集成启动：实机关节状态发布 + robot_state_publisher + RViz
# 用法: ros2 launch nero_description ppo_sim2real.launch.py
# 用法: ros2 launch nero_description ppo_sim2real.launch.py can_left:=can0 can_right:=can1
# 用法: ros2 launch nero_description ppo_sim2real.launch.py no_right:=true  # 只连接左臂

from ament_index_python.packages import get_package_share_path

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    pkg_share = str(get_package_share_path("nero_description"))

    # 实机关节状态发布 + robot_state_publisher + RViz（已包含在 rviz_real 中）
    rviz_real_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([pkg_share, "/launch/rviz_real.launch.py"]),
    )

    return LaunchDescription([
        rviz_real_launch,
    ])

