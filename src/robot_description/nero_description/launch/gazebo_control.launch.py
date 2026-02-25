# Copyright 2025. Gazebo 中加载 bi_nero 模型，并通过 ros2_control 进行关节控制。
# 控制接口：joint_trajectory_controller (JTC，内部插值平滑，非瞬移)
#   Topic: /joint_trajectory_controller/joint_trajectory (trajectory_msgs/JointTrajectory)
# 或运行: ros2 run nero_description publish_joint_positions
# 目标位姿 Marker: /reach_target_markers (RViz Add MarkerArray)，/reach_targets (PoseArray)
# 使用 parameters_file 传递 robot_description，避免 --param override 导致 gazebo_ros2_control 解析失败。

import os
import re
import tempfile

from ament_index_python.packages import get_package_share_path
from launch import LaunchDescription
from launch.actions import ExecuteProcess, RegisterEventHandler, TimerAction
from launch.event_handlers import OnProcessExit
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_path("nero_description")
    urdf_path = pkg_share / "urdf" / "bi_nero_description.urdf"
    config_path = pkg_share / "config" / "bi_nero_controllers.yaml"
    

    with open(urdf_path, "r") as f:
        urdf_str = f.read()
    urdf_str = urdf_str.replace("GAZEBO_ROS2_CONTROL_CONFIG_PATH", str(config_path))
    # Gazebo 不识别 package://，将 mesh 路径改为绝对路径（file://）
    pkg_share_str = str(pkg_share).replace("\\", "/")
    urdf_str = urdf_str.replace("package://nero_description/", f"file://{pkg_share_str}/")
    # 去掉 XML 注释，避免 --param 解析时因注释中含 -- 报错
    urdf_str = re.sub(r'<!--.*?-->', '', urdf_str, flags=re.DOTALL)
    # 去掉 XML 声明行，避免 spawn_entity 用 lxml 解析时报错
    urdf_str = re.sub(r'<\?xml[^>]*\?>\s*', '', urdf_str)
    # 压缩为单行，避免 gazebo_ros2_control 内部按 --param 解析时因换行/多行 XML 报错
    urdf_str = re.sub(r'\s+', ' ', urdf_str).strip()

    # 写入临时 YAML，通过 parameters_file 传给 robot_state_publisher，避免用 --param 传大字符串
    # YAML 双引号字符串内需转义 \ 和 "
    escaped = urdf_str.replace("\\", "\\\\").replace('"', '\\"')
    yaml_lines = [
        "/**:",
        "  ros__parameters:",
        "    use_sim_time: true",
        "    publish_frequency: 15.0",
        '    robot_description: "' + escaped + '"',
    ]
    params_file = os.path.join(tempfile.gettempdir(), "bi_nero_robot_description_params.yaml")
    with open(params_file, "w", encoding="utf-8") as f:
        f.write("\n".join(yaml_lines))

    robot_name = "bi_nero"

    # controller_manager 在全局命名空间 /controller_manager（非 /bi_nero/controller_manager）
    controller_manager_node = "controller_manager"

    # Gazebo
    start_gazebo_cmd = ExecuteProcess(
        cmd=[
            "gazebo",
            "--verbose",
            "-s", "libgazebo_ros_init.so",
            "-s", "libgazebo_ros_factory.so",
        ],
        output="screen",
    )

    # robot_state_publisher 通过 parameters_file 加载参数，避免 --param 传 robot_description 导致插件解析失败
    # 需要先启动来发布 /robot_description topic（供 spawn_entity 使用）
    # 然后会自动订阅 /joint_states（当 joint_state_broadcaster 启动后）来发布 TF
    node_robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[params_file],
        output="screen",
    )
    
    # 延迟 robot_state_publisher 启动 2 秒，等待 Gazebo 时钟发布
    robot_state_publisher_delayed = TimerAction(
        period=2.0,
        actions=[node_robot_state_publisher],
    )

    # 在 Gazebo 中生成机器人（URDF 中含 gazebo_ros2_control 插件）
    spawn_entity_cmd = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        arguments=["-entity", robot_name, "-topic", "robot_description"],
        output="screen",
    )

    # 等待 controller_manager 服务出现（插件加载较慢或机器慢时 8s 不够），最多等 60s
    wait_and_load_joint_state_broadcaster = (
        "for i in $(seq 1 60); do "
        "ros2 service list 2>/dev/null | grep -q 'controller_manager/load_controller' && break; "
        "sleep 1; "
        "done; "
        "sleep 2; "
        f"ros2 control load_controller -c {controller_manager_node} -s --set-state active joint_state_broadcaster"
    )
    load_joint_state_broadcaster = ExecuteProcess(
        cmd=["bash", "-c", wait_and_load_joint_state_broadcaster],
        output="screen",
    )

    # 加载 joint_trajectory_controller（JTC，内部插值平滑到目标位置，非瞬移）
    load_joint_trajectory_controller = ExecuteProcess(
        cmd=[
            "bash", "-c",
            f"sleep 5 && ros2 control load_controller -c {controller_manager_node} -s --set-state active joint_trajectory_controller",
        ],
        output="screen",
    )

    # 确保启动顺序：robot_state_publisher -> spawn_entity -> load_joint_state_broadcaster -> load_joint_trajectory_controller
    # robot_state_publisher 先启动发布 /robot_description（供 spawn_entity 使用）
    # 当 joint_state_broadcaster 启动后，robot_state_publisher 会自动订阅 /joint_states 并发布 TF
    close_evt1 = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=spawn_entity_cmd,
            on_exit=[load_joint_state_broadcaster],
        )
    )
    close_evt2 = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=load_joint_state_broadcaster,
            on_exit=[load_joint_trajectory_controller],
        )
    )

    # 目标位姿 Marker 节点（发布 /reach_target_markers、/reach_targets，每 6s 随机更新）
    node_reach_targets = Node(
        package="nero_description",
        executable="spawn_reach_targets",
        name="spawn_reach_targets",
        output="screen",
        parameters=[{"use_sim_time": True}],
    )

    ld = LaunchDescription()
    ld.add_action(close_evt1)
    ld.add_action(close_evt2)
    ld.add_action(start_gazebo_cmd)
    ld.add_action(robot_state_publisher_delayed)  # 先启动，发布 /robot_description topic
    ld.add_action(spawn_entity_cmd)  # 从 /robot_description topic 读取 URDF
    ld.add_action(node_reach_targets)
    return ld
