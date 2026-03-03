#!/usr/bin/env python3
# Copyright 2025. Gazebo sim2sim: 按 Isaac Lab Bi-Nero Reach 的 60 维状态输入加载 policy.pt，
# 推理后发布关节位置轨迹（与 Isaac Lab set_joint_position_target 一致）。
# 用法: ros2 run nero_description gazebo_ppo_play --policy /path/to/policy.pt [--debug]
# 依赖: joint_trajectory_controller (JTC，内部插值平滑，非瞬移)

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
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
    from builtin_interfaces.msg import Duration
    from geometry_msgs.msg import PoseArray
    from std_msgs.msg import Float32MultiArray
    from tf2_ros import Buffer, TransformListener, TransformException
except ImportError as e:
    print(f"缺少依赖: {e}")
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
DEFAULT_RIGHT = np.array([-1.6, 1.2, -0.52, 0.52, 0.6, 0.0, 0.0], dtype=np.float32)#零点设置
SCALE = 0.5#scale设置 动作解码需与isaaclab动作限幅一致
KEYPOINT_SCALE = 0.25#关键点误差计算与spawn_reach_targets一致

JOINT_NAMES = [
    "left_joint1", "left_joint2", "left_joint3", "left_joint4",
    "left_joint5", "left_joint6", "left_joint7",
    "right_joint1", "right_joint2", "right_joint3", "right_joint4",
    "right_joint5", "right_joint6", "right_joint7",
]

# 关节位置/速度合理范围（/joint_states 速度裁剪）
JOINT_VEL_CLIP = 3.0  # rad/s

# 直接关节位置控制：每步都发布（不做 EMA，保持 policy 输出不变）

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

########################################################
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


class GazeboPpoPlayNode(Node):
    def __init__(self, policy_path: str, control_hz: float = 30.0, debug: bool = False, debug_interval: int = 30,
                 use_sim_time: bool = True, print_io: bool = False, print_io_interval: int = 30,
                 interp_alpha: float = 0.075):
        super().__init__(
            "gazebo_ppo_play",
            parameter_overrides=[Parameter("use_sim_time", value=use_sim_time)],
        )
        self.control_hz = control_hz
        self.policy_path = policy_path
        self.debug = debug
        self.debug_interval = max(1, debug_interval)
        self.print_io = print_io
        self.print_io_interval = max(1, print_io_interval)
        # 与 Isaac 对齐：alpha=1 每步直接发 policy 目标，最终位姿一致。alpha<1 时 sent 平滑逼近 target，最终位姿可能略异
        self.interp_alpha = max(0.01, min(1.0, float(interp_alpha)))

        self.joint_positions = {n: 0.0 for n in JOINT_NAMES}
        self.joint_velocities = {n: 0.0 for n in JOINT_NAMES}  # 在 _control_step 里用控制步长差分更新
        # 发布关节轨迹到 joint_trajectory_controller（JTC 内部插值平滑，非瞬移）
        self._pub_joint_cmd = self.create_publisher(
            JointTrajectory, "/joint_trajectory_controller/joint_trajectory", 10
        )
        # 调试用
        self._pub_policy_cmd = self.create_publisher(JointState, "policy_cmd", 10)#创建关节发布者，最多缓存十条
        # 发布解码后的关节位置（14维，DEFAULT + SCALE*action，裁剪限位前，便于与实际控制命令对比）
        self._pub_raw_action = self.create_publisher(Float32MultiArray, "policy_raw_action", 10)
        # 发布实际关节状态（14维，从/joint_states获取，便于对比策略输出与实际执行）
        self._pub_actual_joints = self.create_publisher(Float32MultiArray, "actual_joint_states", 10)
        # 上一拍指令关节位置（用于 obs 的 joint_prev_pos）。Isaac 中 joint_prev_pos = default + processed_action；
        # 第一拍尚未执行过 action，processed_action=0，故 上一拍指令 = default。
        self._left_prev_pos = DEFAULT_LEFT.copy()
        self._right_prev_pos = DEFAULT_RIGHT.copy()
        # 插值用：当前实际下发的关节目标（从当前点向 policy 目标平滑移动）
        self._left_sent = DEFAULT_LEFT.copy()
        self._right_sent = DEFAULT_RIGHT.copy()

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
        # 速度与 publish_joint_positions --vel-check 一致：控制步长差分 vel=(pos-prev)/control_dt，避免 joint_states 连续同值导致 vel 恒 0
        self._control_dt = 1.0 / control_hz
        self._prev_control_pos = None  # 上一控制步的 joint 位置，首步为 None 时 obs 中 vel 为 0

    def _joint_states_cb(self, msg):
        """只更新关节位置；速度在 _control_step 里用控制步长差分计算。"""
        for i, name in enumerate(msg.name):
            if name not in self.joint_positions:
                continue
            pos = msg.position[i] if i < len(msg.position) else 0.0
            self.joint_positions[name] = pos

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
        # 第一步（step 0）与 Isaac reset 对齐：Isaac reset 后 joint_vel 为 0，Gazebo 无 reset 故用 0 代替
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

    def _publish_joint_positions(self, left_pos: np.ndarray, right_pos: np.ndarray):
        """发布关节轨迹到 joint_trajectory_controller，JTC 在指定时间内平滑插值到目标。
        轨迹时间根据 interp_alpha 调整：alpha 越小，目标变化越平滑，可以给控制器更多时间。
        """
        positions = np.concatenate([left_pos, right_pos]).tolist()
        msg = JointTrajectory()
        msg.joint_names = list(JOINT_NAMES)
        point = JointTrajectoryPoint()
        point.positions = [float(x) for x in positions]
        # 轨迹时间：根据 interp_alpha 调整
        # alpha 越小，目标变化越平滑，可以给控制器更多时间到达目标
        # 最小为 1 个控制周期，最大为 2 个控制周期（给控制器更多缓冲时间）
        if self.interp_alpha < 0.3:
            # 小 alpha：目标变化平滑，给控制器更多时间
            duration_sec = self._control_dt * 1.5
        else:
            # 大 alpha：目标变化快，但至少给 1 个控制周期
            duration_sec = self._control_dt
        sec = int(duration_sec)
        nanosec = int((duration_sec - sec) * 1e9)
        point.time_from_start = Duration(sec=sec, nanosec=nanosec)
        msg.points = [point]
        self._pub_joint_cmd.publish(msg)

    def _publish_home_positions(self):
        """退出时发送默认姿态"""
        self._publish_joint_positions(DEFAULT_LEFT, DEFAULT_RIGHT)

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

        sep = "  "
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
        print("(4) 解码后关节位置（已按 J1~J7 限位裁剪，发往 joint_trajectory_controller）")
        print(f"  left_cmd(7)  = clip(DEFAULT_LEFT + {SCALE}*action[:7]):  {_fmt(left_cmd)}")
        print(f"  right_cmd(7) = clip(DEFAULT_RIGHT + {SCALE}*action[7:14]): {_fmt(right_cmd)}")
        print("=" * 70 + "\n")

    def _control_step(self):
        # 与 publish_joint_positions --vel-check 相同：vel = (当前 pos - 上一步 pos) / control_dt，再限幅
        if self._prev_control_pos is not None:
            for name in JOINT_NAMES:
                dp = self.joint_positions[name] - self._prev_control_pos[name]
                vel = max(-JOINT_VEL_CLIP, min(JOINT_VEL_CLIP, dp / self._control_dt))
                self.joint_velocities[name] = vel
        self._prev_control_pos = {n: self.joint_positions[n] for n in JOINT_NAMES}

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

        # 与 Isaac JointPositionAction 一致：target = default + scale*action；joint_prev_pos = 该 target（用于下一拍 obs）
        left_decoded, right_decoded = self._decode_action_to_joint_positions(action_flat)
        
        # 发布解码后的关节位置（14维，DEFAULT + SCALE*action，裁剪限位前，便于与实际控制命令对比）
        decoded_joint_pos = np.concatenate([left_decoded, right_decoded])
        raw_action_msg = Float32MultiArray()
        raw_action_msg.data = decoded_joint_pos.tolist()
        self._pub_raw_action.publish(raw_action_msg)
        left_target = self._clip_arm_to_limits(left_decoded)
        right_target = self._clip_arm_to_limits(right_decoded)
        self._left_prev_pos = left_target.copy()
        self._right_prev_pos = right_target.copy()

        # 当前动作→下一动作插值：从当前下发的点平滑移动到 policy 目标点
        # sent = sent + alpha*(target - sent)；alpha=1 时不插值，直接 target
        alpha = self.interp_alpha
        left_sent = self._left_sent + alpha * (left_target - self._left_sent)
        right_sent = self._right_sent + alpha * (right_target - self._right_sent)
        left_pos = self._clip_arm_to_limits(left_sent)
        right_pos = self._clip_arm_to_limits(right_sent)
        self._left_sent = left_pos.copy()
        self._right_sent = right_pos.copy()

        self._publish_joint_positions(left_pos, right_pos)

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


def main():
    parser = argparse.ArgumentParser(
        description="Gazebo sim2sim: 60 维状态输入 policy.pt，发布关节位置"
    )
    parser.add_argument(
        "--policy", "-p", type=str, required=True,
        help="policy.pt 路径（如 logs/rsl_rl/.../exported/policy.pt）"
    )
    parser.add_argument(
        "--hz", type=float, default=30.0,
        help="控制频率 (Hz)，与 Isaac step_dt 一致时取 30"
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
        "--interp-alpha", type=float, default=0.05,
        help="当前动作→下一动作插值系数 (0,1]。默认 0.1 平衡响应速度与控制器跟踪能力；1.0 与 Isaac 一致但可能导致容差错误；建议 0.05-0.2"
    )
    parser.add_argument(
        "--use-sim-time", action="store_true", default=True,
        help="使用仿真时间（Gazebo 下必须，默认 True）"
    )
    parser.add_argument(
        "--no-sim-time", dest="use_sim_time", action="store_false",
        help="不使用仿真时间（真机时可加此选项）"
    )
    args = parser.parse_args()

    rclpy.init()
    node = GazeboPpoPlayNode(
        policy_path=args.policy,
        control_hz=args.hz,
        debug=args.debug,
        debug_interval=args.debug_interval,
        use_sim_time=args.use_sim_time,
        print_io=args.print_io,
        print_io_interval=args.print_io_interval,
        interp_alpha=args.interp_alpha,
    )

    node.get_logger().info(
        f"gazebo_ppo_play 已启动: policy={args.policy}, hz={args.hz}"
        + (f" [DEBUG 每 {args.debug_interval} 步打印]" if args.debug else "")
        + (f" [PRINT_IO 每 {args.print_io_interval} 步打印 obs_60 + action_14]" if args.print_io else "")
        + (f" [插值 alpha={args.interp_alpha}]" if args.interp_alpha < 1.0 else "")
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.get_logger().info("退出中，发送默认姿态...")
    node._publish_home_positions()
    time.sleep(0.5)
    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
