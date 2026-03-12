nero-ros2包：
 git clone https://github.com/Alexhii22/nero_ws.git
 
 src colcon:
 colcon build
 source install/setup.bash
 sudo apt update
 sudo apt install ros-humble-gazebo-ros-pkgs
 sudo apt install ros-humble-gazebo-ros2-control
 sudo apt install ros-$ROS_DISTRO-ros2-control ros-$ROS_DISTRO-ros2-controllers #关节control
 
#######################################################################################################
 
 sim2sim启用：
 source后：
ros2 launch nero_description ppo_sim2sim.launch.py #启动 binero仿真gazebo及rviz环境 controllers
ros2 run nero_description spawn_reach_targets #发布目标、关键点、末端点  rviz中markerarray订阅一下/reach话题可视化
关节限制调整前：
ros2 run nero_description publish_joint_positions #机械臂回零点 
ros2 run nero_description gazebo_ppo_play --policy /home/alexlin1/nero_ws/src/robot_description/nero_description/config/policy.pt #路径下权重文件sim


┌─────────────────────────────────────────────────────────────┐
│                    Sim2Sim 数据流                            │
└─────────────────────────────────────────────────────────────┘

[1] spawn_reach_targets.py
    ├─ 发布 /reach_targets (PoseArray)  ← 随机目标位姿
    └─ 发布 /reach_target_markers (MarkerArray)  ← RViz 可视化

[2] Gazebo 仿真器
    ├─ 发布 /joint_states (JointState)  ← 仿真关节状态
    └─ 订阅 /joint_trajectory_controller/joint_trajectory

[3] robot_state_publisher
    ├─ 订阅 /joint_states
    └─ 发布 TF 树 (world → left_link7, right_link7)

[4] gazebo_ppo_play.py
    ├─ 订阅 /joint_states  ← 获取关节位置（速度用差分计算）
    ├─ 订阅 /reach_targets  ← 获取目标位姿
    ├─ 查询 TF (left_link7, right_link7)  ← 获取末端位姿
    ├─ 构建 60 维 obs
    ├─ Policy 推理 → 14 维 action
    ├─ 解码 + 插值 → 关节位置
    └─ 发布 /joint_trajectory_controller/joint_trajectory 
#######################################################################################################

nero机械臂与pc can通信
单臂：
cd nero_sdk/pyAgxArm/pyAgxArm/scripts/ubuntu/

bash find_all_can_port.sh 寻找can

bash can_activate.sh can0 1000000 "1-3:1.0"

bash can_activate.sh can0 1000000 
#can0名字可以改  激活

双臂：
bash find_all_can_port.sh 
打印如下：
Both ethtool and can-utils are installed.
Interface can0 is connected to USB port 1-1:1.0
Interface can1 is connected to USB port 1-3:1.0（记录端口号）
修改并执行
bash can_muti_activate.sh 

notice:(连接时web端打开can接收后需要关闭，偶尔与can端会冲突)
#######################################################################################################
sim2real启用：

加载完毕后可启动ros包功能：
sdk获取关节位置，发布为state、velocity及rviz订阅
ros2 launch nero_description ppo_sim2real.launch.py #rviz可视化
ros2 run nero_description publish_real_joint_positions #机械臂使能去零位
ros2 run nero_description spawn_reach_targets #发布目标及末端关键点
ros2 run nero_description sim_ppo_play --policy /home/alexlin1/nero_ws/src/robot_description/nero_description/config/policy.pt

┌─────────────────────────────────────────────────────────────┐
│                    Sim2Real 数据流                           │
└─────────────────────────────────────────────────────────────┘

[1] spawn_reach_targets.py
    ├─ 发布 /reach_targets (PoseArray)  ← 随机目标位姿
    └─ 发布 /reach_target_markers (MarkerArray)  ← RViz 可视化

[2] gain_real_joint_positions.py
    ├─ 通过 pyAgxArm 读取实机关节状态
    │  ├─ get_joint_states()  → 关节位置
    │  └─ get_motor_states()  → 关节速度（电机反馈）
    └─ 发布 /joint_states (JointState)  ← 实机关节状态

[3] robot_state_publisher
    ├─ 订阅 /joint_states
    └─ 发布 TF 树 (world → left_link7, right_link7)

[4] sim_ppo_play.py
    ├─ 订阅 /joint_states  ← 获取关节位置和速度（直接读取）
    ├─ 订阅 /reach_targets  ← 获取目标位姿
    ├─ 查询 TF (left_link7, right_link7)  ← 获取末端位姿
    ├─ 构建 60 维 obs
    ├─ Policy 推理 → 14 维 action
    ├─ 解码 + 插值 → 关节位置
    └─ 直接调用 robot.move_j()  ← 控制实机机器人（通过 CAN）


















