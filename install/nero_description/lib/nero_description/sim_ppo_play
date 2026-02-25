#!/usr/bin/env python3
# Copyright 2025. Sim2Real: 按 Isaac Lab Bi-Nero Reach 的 60 维状态输入加载 policy.pt，
# 推理后直接控制实机机械臂（与 Isaac Lab set_joint_position_target 一致）。
# 用法: ros2 run nero_description sim_ppo_play --policy /path/to/policy.pt --can-left can_left --can-right can_right
# 依赖: gain_real_joint_positions（发布 /joint_states）、robot_state_publisher（发布 TF）、spawn_reach_targets（发布 /reach_targets）

import argparse
import math
import sys
import time

try:
    import numpy as np
    import torch
    import rclpy
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from sensor_msgs.msg import JointState
    from geometry_msgs.msg import PoseArray
    from std_msgs.msg import Float32MultiArray
    from tf2_ros import Buffer, TransformListener, TransformException
except ImportError as e:
    print(f"缺少依赖: {e}")
    sys.exit(1)

try:
    from pyAgxArm import create_agx_arm_config, AgxArmFactory
except ImportError as e:
    print(f"需要 pyAgxArm 库: {e}")
    sys.exit(1)

# ---------- 与 Isaac Lab 严格对齐（reach_env_cfg + joint_pos_env_cfg）----------
# 60 维 obs 顺序（PolicyCfg 属性定义顺序，concatenate_terms=True）:
#   [0:9]   left_keypoints_error_world   (target_kp - ee_kp 世界系, 9D, m)
#   [9:18]  right_keypoints_error_world  (同上)
#   [18:25] left_joint_pos               (obs_joint_pos_absolute, rad)
#   [25:32] left_joint_vel               (obs_joint_vel, rad/s)
#   [32:39] left_joint_prev_pos          (default + scale*last_action, rad)
#   [39:46] right_joint_pos
#   [46:53] right_joint_vel
#   [53:60] right_joint_prev_pos
# 关键点误差: 与 spawn_reach_targets 同一套计算（scale=0.25, 3 点 X+/Y+/Z+，四元数 x,y,z,w），
# 即目标关键点与末端关键点用同一公式，再做 target_kps - ee_kps 得到 9D 误差输入。
# Action 解码: joint_pos = default + scale*action；joint_pos_env_cfg 中 scale=0.6。
DEFAULT_LEFT = np.array([1.6, 1.2, 0.52, 0.52, -0.6, 0.0, 0.0], dtype=np.float32)
DEFAULT_RIGHT = np.array([-1.6, 1.2, -0.52, 0.52, 0.6, 0.0, 0.0], dtype=np.float32)  # 零点设置
SCALE = 0.6  # scale设置 动作解码需与isaaclab动作限幅一致
KEYPOINT_SCALE = 0.25  # 关键点误差计算与spawn_reach_targets一致

JOINT_NAMES = [
    "left_joint1", "left_joint2", "left_joint3", "left_joint4",
    "left_joint5", "left_joint6", "left_joint7",
    "right_joint1", "right_joint2", "right_joint3", "right_joint4",
    "right_joint5", "right_joint6", "right_joint7",
]

# 关节位置/速度合理范围（/joint_states 速度裁剪）
JOINT_VEL_CLIP = 3.0  # rad/s

# 机械臂关节限位 (rad)，左右臂同构。J1~J7 对应 left/right_joint1..7
# J1:-157°~157°, J2:-15°~190°, J3:-160°~160°, J4:-60°~125°, J5:-160°~160°, J6:-43°~58°, J7:-90°~90°
_deg = np.pi / 180.0
JOINT_LIMITS_LOW = np.array([
    -157 * _deg, -15 * _deg, -160 * _deg, -60 * _deg, -160 * _deg, -43 * _deg, -90 * _deg,
], dtype=np.float32)
JOINT_LIMITS_HIGH = np.array([
    157 * _deg, 190 * _deg, 160 * _deg, 125 * _deg, 160 * _deg, 58 * _deg, 90 * _deg,
], dtype=np.float32)

# ---------- 关键点计算与 spawn_reach_targets 完全一致（scale=0.25, 3 点 X+/Y+/Z+）----------
# 这样状态空间的 keypoints_error 与 RViz 里看到的目标/末端关键点小球一致，再转为误差输入。


def quat_rotate_vector(q, v):
    """四元数 q (x,y,z,w) 旋转向量 v，与 spawn_reach_targets 一致（与 observations/rewards 一致）"""
    qx, qy, qz, qw = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    vx, vy, vz = float(v[0]), float(v[1]), float(v[2])
    return np.array([
        (1 - 2 * qy * qy - 2 * qz * qz) * vx + (2 * qx * qy - 2 * qz * qw) * vy + (2 * qx * qz + 2 * qy * qw) * vz,
        (2 * qx * qy + 2 * qz * qw) * vx + (1 - 2 * qx * qx - 2 * qz * qz) * vy + (2 * qy * qz - 2 * qx * qw) * vz,
        (2 * qx * qz - 2 * qy * qw) * vx + (2 * qy * qz + 2 * qx * qw) * vy + (1 - 2 * qx * qx - 2 * qy * qy) * vz,
    ], dtype=np.float32)


def get_keypoints_world(pos, quat_xyzw, keypoint_scale=KEYPOINT_SCALE):
    """3 个正轴关键点世界坐标 (9,) = 3 点×(x,y,z)。与 spawn_reach_targets.get_target_keypoints_world 一致。
    pos: (3,), quat_xyzw: (x,y,z,w) 来自 ROS/TF。"""
    corners = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    out = []
    for c in corners:
        offset = [c[0] * keypoint_scale, c[1] * keypoint_scale, c[2] * keypoint_scale]
        kp = np.asarray(pos, dtype=np.float32) + quat_rotate_vector(quat_xyzw, offset)
        out.extend(kp.tolist())
    return np.array(out, dtype=np.float32)


def keypoints_error_world(ee_pos, ee_quat, tgt_pos, tgt_quat):
    """关键点误差（世界系）：target - ee，与 obs_keypoints_error_world 一致。(9,)
    使用与 spawn_reach_targets 相同的关键点计算，转为状态空间输入。"""
    ee_kps = get_keypoints_world(ee_pos, ee_quat)
    tgt_kps = get_keypoints_world(tgt_pos, tgt_quat)
    return tgt_kps - ee_kps


def pose_to_pos_quat(pose):
    """Pose -> pos(3), quat(4) x,y,z,w"""
    p = np.array([pose.position.x, pose.position.y, pose.position.z], dtype=np.float32)
    q = np.array([
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w
    ], dtype=np.float32)
    return p, q


def _fmt(arr, fmt=".4f", max_show=7):
    """格式化数组，过长时只显示前 max_show 个 + ..."""
    a = np.asarray(arr).ravel()
    if len(a) <= max_show:
        return " ".join(f"{x:{fmt}}" for x in a)
    return " ".join(f"{x:{fmt}}" for x in a[:max_show]) + f" ... (len={len(a)})"


class SimPpoPlayNode(Node):
    def __init__(self, policy_path: str, can_left: str = None, can_right: str = None,
                 control_hz: float = 100.0, debug: bool = False, debug_interval: int = 30,
                 print_io: bool = False, print_io_interval: int = 30,
                 interp_alpha: float = 0.02, speed_percent: int = 50,
                 interp_dead_zone: float = 0.001, action_ema_alpha: float = 0.3,
                 use_joint_relative: bool = False, max_joint_delta_per_step: float = 0.1):
        super().__init__(
            "sim_ppo_play",
            parameter_overrides=[Parameter("use_sim_time", value=False)],  # 实机不使用仿真时间
        )
        self.control_hz = control_hz
        self.policy_path = policy_path
        self.debug = debug
        self.debug_interval = max(1, debug_interval)
        self.print_io = print_io
        self.print_io_interval = max(1, print_io_interval)
        # 与 Isaac 对齐：alpha=1 每步直接发 policy 目标，最终位姿一致。alpha<1 时 sent 平滑逼近 target，最终位姿可能略异
        self.interp_alpha = max(0.01, min(1.0, float(interp_alpha)))
        self.speed_percent = speed_percent
        self.interp_dead_zone = interp_dead_zone
        # Action EMA滤波：对policy输出的action进行指数滑动平均（一阶低通滤波）
        # action_ema_alpha=0.0 表示不滤波，1.0 表示完全使用上一次的值
        # 推荐值：0.1-0.3（轻微平滑），0.3-0.5（中等平滑），0.5-0.8（强平滑）
        self.action_ema_alpha = max(0.0, min(1.0, float(action_ema_alpha)))
        # 关节相对控制：将policy输出与当前关节位置关联，限制单步变化量，避免跳变
        # use_joint_relative=True 时，计算目标位置与当前关节位置的差值，限制单步变化量
        # max_joint_delta_per_step: 每个控制步允许的最大关节位置变化量（rad），推荐0.05-0.2
        self.use_joint_relative = bool(use_joint_relative)
        self.max_joint_delta_per_step = float(max_joint_delta_per_step)

        # 初始化实机机器人（支持同时连接左臂和右臂）
        self.robot_left = None
        self.robot_right = None
        
        if can_left:
            self.get_logger().info(f"初始化左臂机器人 (CAN: {can_left})...")
            try:
                cfg_left = create_agx_arm_config(robot="nero", channel=can_left)
                self.robot_left = AgxArmFactory.create_arm(cfg_left)
                self.robot_left.connect()
                while not self.robot_left.enable():
                    time.sleep(0.01)
                self.robot_left.set_motion_mode(self.robot_left.MOTION_MODE.J)
                self.robot_left.set_speed_percent(self.speed_percent)
                self.get_logger().info("左臂机器人连接成功")
            except Exception as e:
                self.get_logger().error(f"左臂机器人连接失败: {e}")
                self.robot_left = None
        
        if can_right:
            self.get_logger().info(f"初始化右臂机器人 (CAN: {can_right})...")
            try:
                cfg_right = create_agx_arm_config(robot="nero", channel=can_right)
                self.robot_right = AgxArmFactory.create_arm(cfg_right)
                self.robot_right.connect()
                while not self.robot_right.enable():
                    time.sleep(0.01)
                self.robot_right.set_motion_mode(self.robot_right.MOTION_MODE.J)
                self.robot_right.set_speed_percent(self.speed_percent)
                self.get_logger().info("右臂机器人连接成功")
            except Exception as e:
                self.get_logger().error(f"右臂机器人连接失败: {e}")
                self.robot_right = None
        
        if not self.robot_left and not self.robot_right:
            error_msg = "没有可用的机器人实例，请检查 CAN 通道配置"
            self.get_logger().error(error_msg)
            raise RuntimeError(error_msg)

        self.joint_positions = {n: 0.0 for n in JOINT_NAMES}
        self.joint_velocities = {n: 0.0 for n in JOINT_NAMES}  # 在 _control_step 里用控制步长差分更新
        
        # 调试用：发布 policy 命令到 topic
        self._pub_policy_cmd = self.create_publisher(JointState, "policy_cmd", 10)
        # 发布原始 action（14维，policy 直接输出，裁剪后，相对于默认位置的偏移量）
        self._pub_raw_action = self.create_publisher(Float32MultiArray, "policy_raw_action", 10)
        # 发布EMA滤波后的action（14维，EMA滤波后，相对于默认位置的偏移量）
        self._pub_action_ema = self.create_publisher(Float32MultiArray, "policy_action_ema", 10)
        # 发布理想关节位置（14维，DEFAULT + SCALE*action，解码后，未插值、未限位裁剪，便于 plotjuggler 查看）
        self._pub_ideal_joints = self.create_publisher(Float32MultiArray, "policy_ideal_joints", 10)
        
        # 上一拍指令关节位置（用于 obs 的 joint_prev_pos）。Isaac 中 joint_prev_pos = default + processed_action；
        # 第一拍尚未执行过 action，processed_action=0，故 上一拍指令 = default。
        self._left_prev_pos = DEFAULT_LEFT.copy()
        self._right_prev_pos = DEFAULT_RIGHT.copy()
        # 插值用：当前实际下发的关节目标（从当前点向 policy 目标平滑移动）
        self._left_sent = DEFAULT_LEFT.copy()
        self._right_sent = DEFAULT_RIGHT.copy()
        # 初始化上次发送位置（用于节流）
        self._last_sent_pos_left = DEFAULT_LEFT.copy()
        self._last_sent_pos_right = DEFAULT_RIGHT.copy()
        # 用于自适应插值：记录上一次的目标，用于检测目标变化幅度
        self._left_prev_target = DEFAULT_LEFT.copy()
        self._right_prev_target = DEFAULT_RIGHT.copy()
        # Action EMA滤波用：记录上一次的action（用于EMA滤波）
        self._prev_action_ema = np.zeros(14, dtype=np.float32)

        # /reach_targets: PoseArray, frame_id=world, poses[0]=右臂目标, poses[1]=左臂目标
        self._right_target_pos = None
        self._right_target_quat = None
        self._left_target_pos = None
        self._left_target_quat = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.create_subscription(JointState, "/joint_states", self._joint_states_cb, 10)
        self.create_subscription(PoseArray, "/reach_targets", self._target_cb, 10)

        # 加载 policy.pt（含 normalizer 时直接喂原始 60 维）
        self.get_logger().info(f"加载 policy: {policy_path}")
        self._policy = torch.jit.load(policy_path, map_location="cpu")
        self._policy.eval()

        self._timer = self.create_timer(1.0 / control_hz, self._control_step)
        self._step_count = 0
        # 实机速度直接从 /joint_states 读取（由 gain_real_joint_positions.py 通过 get_motor_states 获取）
        # 不需要差分计算，因为实机已经有准确的电机速度反馈
        self._control_dt = 1.0 / control_hz
        

    def _joint_states_cb(self, msg):
        """从 /joint_states 更新关节位置和速度（实机速度来自 get_motor_states，更准确）"""
        for i, name in enumerate(msg.name):
            if name not in self.joint_positions:
                continue
            pos = msg.position[i] if i < len(msg.position) else 0.0
            vel = msg.velocity[i] if i < len(msg.velocity) else 0.0
            self.joint_positions[name] = pos
            self.joint_velocities[name] = vel

    def _target_cb(self, msg):
        if len(msg.poses) < 2:
            return
        # poses[0]=右臂目标, poses[1]=左臂目标（与 spawn_reach_targets 一致）
        self._right_target_pos, self._right_target_quat = pose_to_pos_quat(msg.poses[0])
        self._left_target_pos, self._left_target_quat = pose_to_pos_quat(msg.poses[1])

    def _get_link_pose_world(self, child_frame: str):
        """从 TF 获取 link 在 world 下的位姿 pos(3), quat(4)"""
        try:
            t = self.tf_buffer.lookup_transform(
                "world", child_frame, rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.05)
            )
            p = np.array([
                t.transform.translation.x, t.transform.translation.y, t.transform.translation.z
            ], dtype=np.float32)
            q = np.array([
                t.transform.rotation.x, t.transform.rotation.y,
                t.transform.rotation.z, t.transform.rotation.w
            ], dtype=np.float32)
            return p, q
        except (TransformException, Exception):
            return None, None

    def _build_obs_60(self):
        """构建 60 维状态，与 Isaac PolicyCfg 顺序一致。"""
        if self._left_target_pos is None or self._right_target_pos is None:
            return None

        left_ee_pos, left_ee_quat = self._get_link_pose_world("left_link7")
        right_ee_pos, right_ee_quat = self._get_link_pose_world("right_link7")
        if left_ee_pos is None or right_ee_pos is None:
            return None

        left_kp_err = keypoints_error_world(
            left_ee_pos, left_ee_quat,
            self._left_target_pos, self._left_target_quat
        )
        right_kp_err = keypoints_error_world(
            right_ee_pos, right_ee_quat,
            self._right_target_pos, self._right_target_quat
        )

        # 与 reach_env_cfg PolicyCfg 一致：obs_joint_pos_absolute / obs_joint_vel，joint_names 顺序 left_joint1..7, right_joint1..7
        left_joint_pos = np.array(
            [self.joint_positions[f"left_joint{i}"] for i in range(1, 8)], dtype=np.float32
        )
        left_joint_vel = np.array(
            [self.joint_velocities[f"left_joint{i}"] for i in range(1, 8)], dtype=np.float32
        )
        right_joint_pos = np.array(
            [self.joint_positions[f"right_joint{i}"] for i in range(1, 8)], dtype=np.float32
        )
        right_joint_vel = np.array(
            [self.joint_velocities[f"right_joint{i}"] for i in range(1, 8)], dtype=np.float32
        )
        # 第一步（step 0）与 Isaac reset 对齐：Isaac reset 后 joint_vel 为 0，实机无 reset 故用 0 代替
        if self._step_count == 0:
            left_joint_vel = np.zeros(7, dtype=np.float32)
            right_joint_vel = np.zeros(7, dtype=np.float32)

        obs = np.concatenate([
            left_kp_err,           # 9
            right_kp_err,          # 9
            left_joint_pos,        # 7
            left_joint_vel,        # 7
            self._left_prev_pos,   # 7
            right_joint_pos,       # 7
            right_joint_vel,       # 7
            self._right_prev_pos,  # 7
        ], dtype=np.float32)
        return obs

    def _decode_action_to_joint_positions(self, action: np.ndarray):
        """与 Isaac Lab JointPositionAction 一致：processed = default + scale*action（scale=0.6, use_default_offset=True）。
        policy 输出 (14,) -> 左 7 + 右 7 关节位置 (rad)。"""
        if action.ndim == 2:
            action = action[0]
        left = DEFAULT_LEFT + SCALE * action[:7].astype(np.float32)
        right = DEFAULT_RIGHT + SCALE * action[7:14].astype(np.float32)
        return left, right

    @staticmethod
    def _clip_arm_to_limits(pos_7: np.ndarray) -> np.ndarray:
        """将单臂 7 关节位置 (rad) 裁剪到机械臂关节限位内。"""
        return np.clip(pos_7, JOINT_LIMITS_LOW, JOINT_LIMITS_HIGH)
    
    def _send_joint_positions_to_robot(self, left_pos: np.ndarray, right_pos: np.ndarray):
        """直接发送关节位置到实机机械臂（通过 pyAgxArm）"""
        if self.robot_left is not None:
            left_pos_list = left_pos.tolist()
            try:
                self.robot_left.move_j(left_pos_list)
            except Exception as e:
                self.get_logger().warn(f"左臂发送失败: {e}")
        
        if self.robot_right is not None:
            right_pos_list = right_pos.tolist()
            try:
                self.robot_right.move_j(right_pos_list)
            except Exception as e:
                self.get_logger().warn(f"右臂发送失败: {e}")

    def _debug_print_step(self, step: int, obs: np.ndarray, action: np.ndarray, left_cmd: np.ndarray, right_cmd: np.ndarray):
        """逐步骤打印: 原始输入 -> 60 维各段 -> policy 输出 -> 解码关节位置（便于对照 Isaac）"""
        left_ee_pos, left_ee_quat = self._get_link_pose_world("left_link7")
        right_ee_pos, right_ee_quat = self._get_link_pose_world("right_link7")
        left_joint_pos = np.array([self.joint_positions[f"left_joint{i}"] for i in range(1, 8)], dtype=np.float32)
        left_joint_vel = np.array([self.joint_velocities[f"left_joint{i}"] for i in range(1, 8)], dtype=np.float32)
        right_joint_pos = np.array([self.joint_positions[f"right_joint{i}"] for i in range(1, 8)], dtype=np.float32)
        right_joint_vel = np.array([self.joint_velocities[f"right_joint{i}"] for i in range(1, 8)], dtype=np.float32)
        left_kp_err = keypoints_error_world(
            left_ee_pos, left_ee_quat, self._left_target_pos, self._left_target_quat
        ) if left_ee_pos is not None else np.zeros(9, dtype=np.float32)
        right_kp_err = keypoints_error_world(
            right_ee_pos, right_ee_quat, self._right_target_pos, self._right_target_quat
        ) if right_ee_pos is not None else np.zeros(9, dtype=np.float32)

        print("\n" + "=" * 70)
        print(f"[DEBUG step {step}] 数据流: 原始输入 -> 60 维 obs -> policy -> 解码关节")
        print("=" * 70)
        print("(1) 原始输入")
        print(f"  /reach_targets 左臂目标 pos: {_fmt(self._left_target_pos)}  quat: {_fmt(self._left_target_quat)}")
        print(f"  /reach_targets 右臂目标 pos: {_fmt(self._right_target_pos)}  quat: {_fmt(self._right_target_quat)}")
        print(f"  TF left_link7  pos: {_fmt(left_ee_pos) if left_ee_pos is not None else 'None'}")
        print(f"  TF right_link7 pos: {_fmt(right_ee_pos) if right_ee_pos is not None else 'None'}")
        print(f"  /joint_states 左臂 pos(7): {_fmt(left_joint_pos)}  vel(7): {_fmt(left_joint_vel)}")
        print(f"  /joint_states 右臂 pos(7): {_fmt(right_joint_pos)}  vel(7): {_fmt(right_joint_vel)}")
        print(f"  上一拍指令  left_prev(7): {_fmt(self._left_prev_pos)}  right_prev(7): {_fmt(self._right_prev_pos)}")
        print("(2) 中间量（拼进 60 维的顺序）")
        print(f"  left_keypoints_error_world(9):  {_fmt(left_kp_err)}")
        print(f"  right_keypoints_error_world(9): {_fmt(right_kp_err)}")
        print(f"  left_joint_pos(7):  {_fmt(left_joint_pos)}")
        print(f"  left_joint_vel(7):  {_fmt(left_joint_vel)}")
        print(f"  left_joint_prev_pos(7): {_fmt(self._left_prev_pos)}")
        print(f"  right_joint_pos(7): {_fmt(right_joint_pos)}")
        print(f"  right_joint_vel(7): {_fmt(right_joint_vel)}")
        print(f"  right_joint_prev_pos(7): {_fmt(self._right_prev_pos)}")
        print(f"  obs 60 维 [0:9] left_kp_err: {_fmt(obs[0:9])}  [9:18] right_kp_err: {_fmt(obs[9:18])}")
        print(f"  obs [18:25] left_pos: {_fmt(obs[18:25])}  [25:32] left_vel: {_fmt(obs[25:32])}")
        print(f"  obs [32:39] left_prev: {_fmt(obs[32:39])}  [39:46] right_pos: {_fmt(obs[39:46])}")
        print(f"  obs [46:53] right_vel: {_fmt(obs[46:53])}  [53:60] right_prev: {_fmt(obs[53:60])}")
        print("(3) Policy 输入与输出")
        print(f"  obs_tensor shape: (1, 60)  dtype float32")
        print(f"  action(14): {_fmt(action)}")
        print("(4) 解码后关节位置（已按 J1~J7 限位裁剪，发往实机）")
        print(f"  left_cmd(7)  = clip(DEFAULT_LEFT + {SCALE}*action[:7]):  {_fmt(left_cmd)}")
        print(f"  right_cmd(7) = clip(DEFAULT_RIGHT + {SCALE}*action[7:14]): {_fmt(right_cmd)}")
        print("=" * 70 + "\n")

    def _control_step(self):
        # 实机速度直接从 /joint_states 读取（由 gain_real_joint_positions.py 通过 get_motor_states 获取）
        # 不需要差分计算，因为实机已经有准确的电机速度反馈
        # 只需对速度进行限幅，确保在合理范围内
        for name in JOINT_NAMES:
            vel = self.joint_velocities[name]
            self.joint_velocities[name] = max(-JOINT_VEL_CLIP, min(JOINT_VEL_CLIP, vel))

        obs = self._build_obs_60()
        if obs is None:
            self.get_logger().warn(
                "等待 /reach_targets 与 TF (left_link7, right_link7)",
                throttle_duration_sec=2.0
            )
            return

        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)
            action = self._policy(obs_tensor).cpu().numpy()

        action_flat = np.asarray(action[0] if action.ndim == 2 else action, dtype=np.float32)
        # 与 Isaac 一致：policy 输出通常被约束在 [-1,1]（如 tanh），解码前裁剪
        action_flat = np.clip(action_flat, -1.0, 1.0)

        # 发布原始 action（14维，policy 直接输出，裁剪后，相对于默认位置的偏移量）
        raw_action_msg = Float32MultiArray()
        raw_action_msg.data = action_flat.tolist()
        self._pub_raw_action.publish(raw_action_msg)
        
        # Action EMA滤波：对policy输出的action进行指数滑动平均（一阶低通滤波）
        # 简单的一阶低通滤波：filtered = alpha * prev_filtered + (1 - alpha) * current
        if self.action_ema_alpha > 0.0:
            action_flat = self.action_ema_alpha * self._prev_action_ema + (1.0 - self.action_ema_alpha) * action_flat
            # 确保仍在 [-1, 1] 范围内
            action_flat = np.clip(action_flat, -1.0, 1.0)
            self._prev_action_ema = action_flat.copy()
            
            # 发布EMA滤波后的action（14维，EMA滤波后，相对于默认位置的偏移量）
            action_ema_msg = Float32MultiArray()
            action_ema_msg.data = action_flat.tolist()
            self._pub_action_ema.publish(action_ema_msg)

        # 与 Isaac JointPositionAction 一致：target = default + scale*action；joint_prev_pos = 该 target（用于下一拍 obs）
        left_decoded, right_decoded = self._decode_action_to_joint_positions(action_flat)
        
        # 关节相对控制：将policy输出与当前关节位置关联，限制单步变化量，避免跳变
        if self.use_joint_relative:
            # 获取当前关节位置
            left_current = np.array([self.joint_positions[f"left_joint{i}"] for i in range(1, 8)], dtype=np.float32)
            right_current = np.array([self.joint_positions[f"right_joint{i}"] for i in range(1, 8)], dtype=np.float32)
            
            # 计算目标位置与当前位置的差值
            left_delta = left_decoded - left_current
            right_delta = right_decoded - right_current
            
            # 限制单步变化量（避免大跳变）
            left_delta_magnitude = np.abs(left_delta)
            right_delta_magnitude = np.abs(right_delta)
            
            # 如果变化量超过限制，按比例缩放
            left_scale = np.where(left_delta_magnitude > self.max_joint_delta_per_step,
                                 self.max_joint_delta_per_step / left_delta_magnitude,
                                 1.0)
            right_scale = np.where(right_delta_magnitude > self.max_joint_delta_per_step,
                                  self.max_joint_delta_per_step / right_delta_magnitude,
                                  1.0)
            
            # 应用缩放后的差值
            left_delta_clamped = left_delta * left_scale
            right_delta_clamped = right_delta * right_scale
            
            # 从当前位置加上限制后的差值，得到新的目标位置
            left_target = left_current + left_delta_clamped
            right_target = right_current + right_delta_clamped
        else:
            # 传统方式：直接使用解码后的目标位置
            left_target = left_decoded
            right_target = right_decoded
        
        # 限位裁剪：对目标位置进行限位
        left_target = self._clip_arm_to_limits(left_target)
        right_target = self._clip_arm_to_limits(right_target)
        
        # 发布理想关节位置（14维，经过关节相对控制和限位裁剪后，未插值，与实际控制指令一致）
        ideal_joints_msg = Float32MultiArray()
        ideal_joints_msg.data = np.concatenate([left_target, right_target]).tolist()
        self._pub_ideal_joints.publish(ideal_joints_msg)
        
        self._left_prev_pos = left_target.copy()
        self._right_prev_pos = right_target.copy()

        # 改进的插值方法：固定 alpha + 死区，更好地处理 policy 输出跳变
        # 1. 计算目标变化幅度
        left_target_diff = np.abs(left_target - self._left_prev_target)
        right_target_diff = np.abs(right_target - self._right_prev_target)
        max_left_diff = np.max(left_target_diff)
        max_right_diff = np.max(right_target_diff)
        
        # 2. 死区：如果目标变化很小，不更新（避免微小跳变）
        if max_left_diff < self.interp_dead_zone and max_right_diff < self.interp_dead_zone:
            # 目标几乎没变化，保持当前 sent 不变
            left_sent = self._left_sent.copy()
            right_sent = self._right_sent.copy()
        else:
            # 3. 固定 alpha 应用插值（sent = sent + alpha*(target - sent)）
            left_alpha = self.interp_alpha
            right_alpha = self.interp_alpha

            left_delta = left_target - self._left_sent
            right_delta = right_target - self._right_sent
            left_sent = self._left_sent + left_alpha * left_delta
            right_sent = self._right_sent + right_alpha * right_delta
        
        left_pos = self._clip_arm_to_limits(left_sent)
        right_pos = self._clip_arm_to_limits(right_sent)
        self._left_sent = left_pos.copy()
        self._right_sent = right_pos.copy()
        self._left_prev_target = left_target.copy()
        self._right_prev_target = right_target.copy()

        # 直接发送到实机
        self._send_joint_positions_to_robot(left_pos, right_pos)

        # 发布到 topic 用于调试
        policy_msg = JointState()
        policy_msg.header.stamp = self.get_clock().now().to_msg()
        policy_msg.name = list(JOINT_NAMES)
        policy_msg.position = np.concatenate([left_pos, right_pos]).tolist()
        self._pub_policy_cmd.publish(policy_msg)

        if self.debug and (self._step_count % self.debug_interval == 0):
            act = action[0] if action.ndim == 2 else action
            self._debug_print_step(self._step_count, obs, act, left_pos, right_pos)

        # 运行时打印 PPO 输入(60维)与输出(14维)，便于排查问题
        if self.print_io and (self._step_count % self.print_io_interval == 0):
            act = action_flat
            print(f"[PPO_IO step={self._step_count}] obs_60: " + ",".join(f"{x:.6f}" for x in obs))
            print(f"[PPO_IO step={self._step_count}] action_14: " + ",".join(f"{x:.6f}" for x in act))

        self._step_count += 1

    def cleanup(self):
        """清理资源：退出时自动失能机器人"""
        if self.robot_left is not None:
            try:
                self.get_logger().info("失能左臂机器人...")
                self.robot_left.disable()
                self.get_logger().info("左臂机器人已失能")
            except Exception as e:
                self.get_logger().warn(f"左臂失能失败: {e}")
        
        if self.robot_right is not None:
            try:
                self.get_logger().info("失能右臂机器人...")
                self.robot_right.disable()
                self.get_logger().info("右臂机器人已失能")
            except Exception as e:
                self.get_logger().warn(f"右臂失能失败: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Sim2Real: 60 维状态输入 policy.pt，直接控制实机机械臂"
    )
    parser.add_argument(
        "--policy", "-p", type=str, required=True,
        help="policy.pt 路径（如 logs/rsl_rl/.../exported/policy.pt）"
    )
    parser.add_argument(
        "--can-left",
        type=str,
        default="can_left",
        help="左臂 CAN 通道名称（默认: can_left）",
    )
    parser.add_argument(
        "--can-right",
        type=str,
        default="can_right",
        help="右臂 CAN 通道名称（默认: can_right，默认连接双臂）",
    )
    parser.add_argument(
        "--no-right",
        action="store_true",
        help="不连接右臂（只使用左臂）",
    )
    parser.add_argument(
        "--hz", type=float, default=100.0,
        help="控制频率 (Hz)，与 Isaac step_dt 一致时取 30"
    )
    parser.add_argument(
        "--speed", type=int, default=30,
        help="运动速度百分比（默认: 10）"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="每 N 步打印一次数据流: 原始输入 -> 60 维各段 -> policy 输出 -> 解码关节（便于逐项核对）"
    )
    parser.add_argument(
        "--debug-interval", type=int, default=30,
        help="--debug 时每多少个控制步打印一次（默认 30）"
    )
    parser.add_argument(
        "--print-io", action="store_true",
        help="运行时打印输入 PPO 的 60 维 obs 与输出的 14 维 action（便于排查问题）"
    )
    parser.add_argument(
        "--print-io-interval", type=int, default=30,
        help="--print-io 时每多少个控制步打印一次（默认 30，即约 1 秒一次@30Hz）"
    )
    parser.add_argument(
        "--interp-alpha", type=float, default=0.015,
        help="当前动作→下一动作插值系数 (0,1]。默认 0.015 适合实机（避免震荡）；1.0 与 Isaac 一致（每步直接发目标）。注意：实机建议 0.01-0.1，过大易震荡（实机执行延迟+速度限制导致过冲）"
    )
    parser.add_argument(
        "--interp-dead-zone", type=float, default=0.0,
        help="插值死区阈值 (rad)。如果 policy 目标变化小于此值，不更新（避免微小跳变）。默认 0.001 rad (约 0.06°)"
    )
    parser.add_argument(
        "--action-ema-alpha", type=float, default=0.1,
        help="Action EMA滤波系数 (0,1]。对policy输出的action进行指数滑动平均。0.0表示不滤波，1.0表示完全使用上一次的值。推荐值：0.1-0.3（轻微平滑），0.3-0.5（中等平滑），0.5-0.8（强平滑）"
    )
    parser.add_argument(
        "--use-joint-relative", action="store_true",
        help="启用关节相对控制。针对ideal joints（解码后的关节位置）进行限制，将policy输出与当前关节位置关联，限制单步变化量，避免跳变。计算ideal joints与当前关节位置的差值，限制单步变化量在max_joint_delta_per_step内"
    )
    parser.add_argument(
        "--max-joint-delta-per-step", type=float, default=0.05,
        help="关节相对控制时，每个控制步允许的最大关节位置变化量（rad）。针对ideal joints（解码后的关节位置）进行限制。默认0.3 rad，当policy输出大跳变时，将ideal joints的单步变化限制在此值内。推荐0.1-0.3。值越小，平滑效果越强，但响应也越慢"
    )
    args = parser.parse_args()

    # 处理参数：如果指定了 --no-right，则不连接右臂
    can_left = args.can_left if args.can_left else None
    can_right = None if args.no_right else (args.can_right if args.can_right else None)

    rclpy.init()
    try:
        node = SimPpoPlayNode(
            policy_path=args.policy,
            can_left=can_left,
            can_right=can_right,
            control_hz=args.hz,
            debug=args.debug,
            debug_interval=args.debug_interval,
            print_io=args.print_io,
            print_io_interval=args.print_io_interval,
            interp_alpha=args.interp_alpha,
            speed_percent=args.speed,
            interp_dead_zone=args.interp_dead_zone,
            action_ema_alpha=args.action_ema_alpha,
            use_joint_relative=args.use_joint_relative,
            max_joint_delta_per_step=args.max_joint_delta_per_step,
        )

        interp_info = f" [插值 alpha={args.interp_alpha}"
        if args.interp_dead_zone > 0:
            interp_info += f", 死区={args.interp_dead_zone:.4f}rad"
        interp_info += "]"
        
        action_ema_info = ""
        if args.action_ema_alpha > 0.0:
            action_ema_info = f" [Action EMA alpha={args.action_ema_alpha:.3f}]"
        
        joint_relative_info = ""
        if args.use_joint_relative:
            joint_relative_info = f" [关节相对控制 max_delta={args.max_joint_delta_per_step:.3f}rad/step]"
        
        node.get_logger().info(
            f"sim_ppo_play 已启动: policy={args.policy}, hz={args.hz}, speed={args.speed}%"
            + (f" [DEBUG 每 {args.debug_interval} 步打印]" if args.debug else "")
            + (f" [PRINT_IO 每 {args.print_io_interval} 步打印 obs_60 + action_14]" if args.print_io else "")
            + (interp_info if args.interp_alpha < 1.0 or args.interp_dead_zone > 0 else "")
            + action_ema_info
            + joint_relative_info
        )
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
        node.get_logger().info("退出中...")
        node.cleanup()
        node.destroy_node()
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1
    finally:
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
