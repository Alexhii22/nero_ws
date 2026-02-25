# Copyright 2025. RViz2 启动文件，配合实机使用。
# 包含：实机关节状态发布 + robot_state_publisher + RViz
# 用法: ros2 launch nero_description rviz_real.launch.py
# 用法: ros2 launch nero_description rviz_real.launch.py can_left:=can0 can_right:=can1

import os
import re
import tempfile

from ament_index_python.packages import get_package_share_path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_path("nero_description")
    urdf_path = pkg_share / "urdf" / "bi_nero_description.urdf"
    default_rviz_config = str(pkg_share / "rviz" / "bi_nero_gazebo.rviz")
    
    # 读取并处理 URDF
    with open(urdf_path, "r") as f:
        urdf_str = f.read()
    # Gazebo 不识别 package://，将 mesh 路径改为绝对路径（file://）
    pkg_share_str = str(pkg_share).replace("\\", "/")
    urdf_str = urdf_str.replace("package://nero_description/", f"file://{pkg_share_str}/")
    # 去掉 XML 注释，避免 --param 解析时因注释中含 -- 报错
    urdf_str = re.sub(r'<!--.*?-->', '', urdf_str, flags=re.DOTALL)
    # 去掉 XML 声明行，避免解析时报错
    urdf_str = re.sub(r'<\?xml[^>]*\?>\s*', '', urdf_str)
    # 压缩为单行，避免解析时因换行/多行 XML 报错
    urdf_str = re.sub(r'\s+', ' ', urdf_str).strip()
    
    # 写入临时 YAML，通过 parameters_file 传给 robot_state_publisher
    escaped = urdf_str.replace("\\", "\\\\").replace('"', '\\"')
    yaml_lines = [
        "/**:",
        "  ros__parameters:",
        "    use_sim_time: false",  # 实机不使用仿真时间
        "    publish_frequency: 50.0",
        '    robot_description: "' + escaped + '"',
    ]
    params_file = os.path.join(tempfile.gettempdir(), "bi_nero_real_robot_description_params.yaml")
    with open(params_file, "w", encoding="utf-8") as f:
        f.write("\n".join(yaml_lines))
    
    # 声明启动参数
    rviz_config_arg = DeclareLaunchArgument(
        "rviz_config",
        default_value=default_rviz_config,
        description="Path to RViz config file",
    )
    can_left_arg = DeclareLaunchArgument(
        "can_left",
        default_value="can_left",
        description="左臂 CAN 通道名称",
    )
    can_right_arg = DeclareLaunchArgument(
        "can_right",
        default_value="can_right",
        description="右臂 CAN 通道名称（默认: can_right，连接右臂）",
    )
    no_left_arg = DeclareLaunchArgument(
        "no_left",
        default_value="false",
        description="是否不连接左臂（true/false）",
    )
    no_right_arg = DeclareLaunchArgument(
        "no_right",
        default_value="false",
        description="是否不连接右臂（true/false，默认连接右臂）",
    )
    publish_hz_arg = DeclareLaunchArgument(
        "publish_hz",
        default_value="50.0",
        description="关节状态发布频率 (Hz)",
    )
    
    # 1. 实机关节状态发布节点（从实机读取关节状态并发布到 /joint_states）
    real_joint_states_node = Node(
        package="nero_description",
        executable="gain_real_joint_positions",
        name="gain_real_joint_positions",
        output="screen",
        parameters=[{"use_sim_time": False}],
        arguments=[
            "--can-left", LaunchConfiguration("can_left"),
            "--can-right", LaunchConfiguration("can_right"),
            "--hz", LaunchConfiguration("publish_hz"),
        ],
    )
    
    # 2. robot_state_publisher（订阅 /joint_states，发布 TF）
    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[params_file],
        output="screen",
    )
    
    # 3. RViz（不使用仿真时间）
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", LaunchConfiguration("rviz_config")],
        parameters=[{"use_sim_time": False}],  # 实机不使用仿真时间
    )
    
    return LaunchDescription([
        rviz_config_arg,
        can_left_arg,
        can_right_arg,
        no_left_arg,
        no_right_arg,
        publish_hz_arg,
        real_joint_states_node,
        robot_state_publisher_node,
        rviz_node,
    ])

