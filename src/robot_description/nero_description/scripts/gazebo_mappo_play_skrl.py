#!/usr/bin/env python3
# Copyright 2025. 在 ROS 中直接加载 SKRL MAPPO checkpoint，用 agent.act(obs) 推理，无需导出脚本。
# 归一化与策略前向与 Isaac Lab play.py 完全同一套代码路径，避免手动拆分带来的潜在 gap。
#
# 用法: ros2 run nero_description gazebo_mappo_play_skrl --checkpoint /path/to/agent_XXXXX.pt [--debug]
# 依赖: torch, gymnasium, skrl（pip install gymnasium skrl）；与 gazebo_ppo_play 相同的 ROS 话题/TF。

import argparse
import os
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

try:
    import gymnasium as gym
    from skrl.multi_agents.torch.mappo import MAPPO
    from skrl.resources.preprocessors.torch import RunningStandardScaler
    from skrl.utils.model_instantiators.torch import deterministic_model, gaussian_model
except ImportError as e:
    print("ROS 内直接跑 SKRL 需要安装: pip install gymnasium skrl")
    print(f"错误: {e}")
    sys.exit(1)

# ---------- 与 Isaac Lab MAPPO train.py / play.py 严格一致 ----------
OBS_LEFT_DIM = 30
OBS_RIGHT_DIM = 30
CRITIC_DIM = 60
ACT_DIM = 7
HIDDEN_DIMS = [256, 256, 128]
POSSIBLE_AGENTS = ["left", "right"]

# 动作解码与 joint_pos_env_cfg 一致
DEFAULT_LEFT = np.array([1.0, 1.0, -1.0, 1.5, 1.0, 0.0, 0.0], dtype=np.float32)
DEFAULT_RIGHT = np.array([-1.0, 1.0, 1.0, 1.5, -1.0, 0.0, 0.0], dtype=np.float32)
SCALE = 0.5
KEYPOINT_SCALE = 0.25

JOINT_NAMES = [
    "left_joint1", "left_joint2", "left_joint3", "left_joint4",
    "left_joint5", "left_joint6", "left_joint7",
    "right_joint1", "right_joint2", "right_joint3", "right_joint4",
    "right_joint5", "right_joint6", "right_joint7",
]
JOINT_VEL_CLIP = 3.0
_deg = np.pi / 180.0
JOINT_LIMITS_LOW = np.array([
    -157 * _deg, -15 * _deg, -160 * _deg, -60 * _deg, -160 * _deg, -43 * _deg, -90 * _deg,
], dtype=np.float32)
JOINT_LIMITS_HIGH = np.array([
    157 * _deg, 190 * _deg, 160 * _deg, 125 * _deg, 160 * _deg, 58 * _deg, 90 * _deg,
], dtype=np.float32)


def build_policy(obs_dim: int, act_dim: int, device: str):
    """与 train.py 完全一致。"""
    return gaussian_model(
        observation_space=gym.spaces.Box(-np.inf, np.inf, (obs_dim,), dtype=np.float32),
        action_space=gym.spaces.Box(-1.0, 1.0, (act_dim,), dtype=np.float32),
        device=device,
        clip_actions=False,
        clip_log_std=True,
        min_log_std=-20.0,
        max_log_std=2.0,
        initial_log_std=0.0,
        network=[{"name": "net", "input": "STATES", "layers": HIDDEN_DIMS, "activations": "elu"}],
        output="ACTIONS",
    )


def build_value(state_dim: int, device: str):
    """与 train.py 完全一致。"""
    return deterministic_model(
        observation_space=gym.spaces.Box(-np.inf, np.inf, (state_dim,), dtype=np.float32),
        action_space=gym.spaces.Box(-1.0, 1.0, (1,), dtype=np.float32),
        device=device,
        clip_actions=False,
        network=[{"name": "net", "input": "STATES", "layers": HIDDEN_DIMS, "activations": "elu"}],
        output="ONE",
    )


def quat_rotate_vector(q, v):
    qx, qy, qz, qw = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    vx, vy, vz = float(v[0]), float(v[1]), float(v[2])
    return np.array([
        (1 - 2 * qy * qy - 2 * qz * qz) * vx + (2 * qx * qy - 2 * qz * qw) * vy + (2 * qx * qz + 2 * qy * qw) * vz,
        (2 * qx * qy + 2 * qz * qw) * vx + (1 - 2 * qx * qx - 2 * qz * qz) * vy + (2 * qy * qz - 2 * qx * qw) * vz,
        (2 * qx * qz - 2 * qy * qw) * vx + (2 * qy * qz + 2 * qx * qw) * vy + (1 - 2 * qx * qx - 2 * qy * qy) * vz,
    ], dtype=np.float32)


def get_keypoints_world(pos, quat_xyzw, keypoint_scale=KEYPOINT_SCALE):
    corners = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    out = []
    for c in corners:
        offset = [c[0] * keypoint_scale, c[1] * keypoint_scale, c[2] * keypoint_scale]
        kp = np.asarray(pos, dtype=np.float32) + quat_rotate_vector(quat_xyzw, offset)
        out.extend(kp.tolist())
    return np.array(out, dtype=np.float32)


def keypoints_error_world(ee_pos, ee_quat, tgt_pos, tgt_quat):
    ee_kps = get_keypoints_world(ee_pos, ee_quat)
    tgt_kps = get_keypoints_world(tgt_pos, tgt_quat)
    return tgt_kps - ee_kps


def pose_to_pos_quat(pose):
    p = np.array([pose.position.x, pose.position.y, pose.position.z], dtype=np.float32)
    q = np.array([
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w
    ], dtype=np.float32)
    return p, q


def _fmt(arr, fmt=".4f", max_show=7):
    a = np.asarray(arr).ravel()
    if len(a) <= max_show:
        return " ".join(f"{x:{fmt}}" for x in a)
    return " ".join(f"{x:{fmt}}" for x in a[:max_show]) + f" ... (len={len(a)})"


def obs_60_to_obs_left_right(obs_60: np.ndarray):
    obs_left = np.concatenate([
        obs_60[0:9], obs_60[18:25], obs_60[25:32], obs_60[32:39],
    ], dtype=np.float32)
    obs_right = np.concatenate([
        obs_60[9:18], obs_60[39:46], obs_60[46:53], obs_60[53:60],
    ], dtype=np.float32)
    return obs_left, obs_right


def _build_skrl_agent(checkpoint_path: str, device: str = "cpu"):
    """在 ROS 侧构建 MAPPO agent 并加载 checkpoint，不依赖 Isaac/gym 环境。"""
    inf = float("inf")
    observation_spaces = {
        "left": gym.spaces.Box(-inf, inf, (OBS_LEFT_DIM,), dtype=np.float32),
        "right": gym.spaces.Box(-inf, inf, (OBS_RIGHT_DIM,), dtype=np.float32),
    }
    action_spaces = {
        "left": gym.spaces.Box(-1.0, 1.0, (ACT_DIM,), dtype=np.float32),
        "right": gym.spaces.Box(-1.0, 1.0, (ACT_DIM,), dtype=np.float32),
    }
    shared_observation_spaces = {
        "left": gym.spaces.Box(-inf, inf, (CRITIC_DIM,), dtype=np.float32),
        "right": gym.spaces.Box(-inf, inf, (CRITIC_DIM,), dtype=np.float32),
    }

    policy_left = build_policy(OBS_LEFT_DIM, ACT_DIM, device)
    policy_right = build_policy(OBS_RIGHT_DIM, ACT_DIM, device)
    value_left = build_value(CRITIC_DIM, device)
    value_right = build_value(CRITIC_DIM, device)
    models = {
        "left": {"policy": policy_left, "value": value_left},
        "right": {"policy": policy_right, "value": value_right},
    }
    agent_cfg = {
        "state_preprocessor": RunningStandardScaler,
        "state_preprocessor_kwargs": {"size": observation_spaces["left"], "device": device},
        "shared_state_preprocessor": RunningStandardScaler,
        "shared_state_preprocessor_kwargs": {"size": shared_observation_spaces["left"], "device": device},
        "value_preprocessor": RunningStandardScaler,
        "value_preprocessor_kwargs": {"size": 1, "device": device},
        "experiment": {"write_interval": 0, "checkpoint_interval": 0},
    }
    agent = MAPPO(
        possible_agents=POSSIBLE_AGENTS,
        models=models,
        memories=None,
        cfg=agent_cfg,
        observation_spaces=observation_spaces,
        action_spaces=action_spaces,
        device=device,
        shared_observation_spaces=shared_observation_spaces,
    )
    agent.init(trainer_cfg={"timesteps": 0, "headless": True})
    agent.load(os.path.abspath(checkpoint_path))
    agent.set_running_mode("eval")
    return agent


class GazeboMappoPlaySkrlNode(Node):
    def __init__(self, checkpoint_path: str, control_hz: float = 30.0, debug: bool = False, debug_interval: int = 30,
                 use_sim_time: bool = True, print_io: bool = False, print_io_interval: int = 30,
                 interp_alpha: float = 0.05, device: str = "cpu"):
        super().__init__(
            "gazebo_mappo_play_skrl",
            parameter_overrides=[Parameter("use_sim_time", value=use_sim_time)],
        )
        self.control_hz = control_hz
        self.debug = debug
        self.debug_interval = max(1, debug_interval)
        self.print_io = print_io
        self.print_io_interval = max(1, print_io_interval)
        self.interp_alpha = max(0.01, min(1.0, float(interp_alpha)))

        self.joint_positions = {n: 0.0 for n in JOINT_NAMES}
        self.joint_velocities = {n: 0.0 for n in JOINT_NAMES}
        self._pub_joint_cmd = self.create_publisher(
            JointTrajectory, "/joint_trajectory_controller/joint_trajectory", 10
        )
        self._pub_policy_cmd = self.create_publisher(JointState, "policy_cmd", 10)
        self._pub_raw_action = self.create_publisher(Float32MultiArray, "policy_raw_action", 10)
        self._left_prev_pos = DEFAULT_LEFT.copy()
        self._right_prev_pos = DEFAULT_RIGHT.copy()
        self._left_sent = DEFAULT_LEFT.copy()
        self._right_sent = DEFAULT_RIGHT.copy()
        self._right_target_pos = None
        self._right_target_quat = None
        self._left_target_pos = None
        self._left_target_quat = None
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.create_subscription(JointState, "/joint_states", self._joint_states_cb, 10)
        self.create_subscription(PoseArray, "/reach_targets", self._target_cb, 10)

        self.get_logger().info(f"加载 SKRL MAPPO checkpoint: {checkpoint_path}")
        self._agent = _build_skrl_agent(checkpoint_path, device=device)
        self.get_logger().info("SKRL agent 已加载，归一化与推理与 Isaac play.py 同一套代码路径")

        self._timer = self.create_timer(1.0 / control_hz, self._control_step)
        self._step_count = 0
        self._control_dt = 1.0 / control_hz
        self._prev_control_pos = None

    def _joint_states_cb(self, msg):
        for i, name in enumerate(msg.name):
            if name not in self.joint_positions:
                continue
            self.joint_positions[name] = msg.position[i] if i < len(msg.position) else 0.0

    def _target_cb(self, msg):
        if len(msg.poses) < 2:
            return
        self._right_target_pos, self._right_target_quat = pose_to_pos_quat(msg.poses[0])
        self._left_target_pos, self._left_target_quat = pose_to_pos_quat(msg.poses[1])

    def _get_link_pose_world(self, child_frame: str):
        try:
            t = self.tf_buffer.lookup_transform(
                "world", child_frame, rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.05)
            )
            p = np.array([t.transform.translation.x, t.transform.translation.y, t.transform.translation.z], dtype=np.float32)
            q = np.array([
                t.transform.rotation.x, t.transform.rotation.y,
                t.transform.rotation.z, t.transform.rotation.w
            ], dtype=np.float32)
            return p, q
        except (TransformException, Exception):
            return None, None

    def _build_obs_60(self):
        if self._left_target_pos is None or self._right_target_pos is None:
            return None
        left_ee_pos, left_ee_quat = self._get_link_pose_world("left_link7")
        right_ee_pos, right_ee_quat = self._get_link_pose_world("right_link7")
        if left_ee_pos is None or right_ee_pos is None:
            return None
        left_kp_err = keypoints_error_world(
            left_ee_pos, left_ee_quat, self._left_target_pos, self._left_target_quat
        )
        right_kp_err = keypoints_error_world(
            right_ee_pos, right_ee_quat, self._right_target_pos, self._right_target_quat
        )
        left_joint_pos = np.array([self.joint_positions[f"left_joint{i}"] for i in range(1, 8)], dtype=np.float32)
        left_joint_vel = np.array([self.joint_velocities[f"left_joint{i}"] for i in range(1, 8)], dtype=np.float32)
        right_joint_pos = np.array([self.joint_positions[f"right_joint{i}"] for i in range(1, 8)], dtype=np.float32)
        right_joint_vel = np.array([self.joint_velocities[f"right_joint{i}"] for i in range(1, 8)], dtype=np.float32)
        if self._step_count == 0:
            left_joint_vel = np.zeros(7, dtype=np.float32)
            right_joint_vel = np.zeros(7, dtype=np.float32)
        obs = np.concatenate([
            left_kp_err, right_kp_err,
            left_joint_pos, left_joint_vel, self._left_prev_pos,
            right_joint_pos, right_joint_vel, self._right_prev_pos,
        ], dtype=np.float32)
        return obs

    def _decode_action_to_joint_positions(self, action: np.ndarray):
        if action.ndim == 2:
            action = action[0]
        left = DEFAULT_LEFT + SCALE * action[:7].astype(np.float32)
        right = DEFAULT_RIGHT + SCALE * action[7:14].astype(np.float32)
        return left, right

    @staticmethod
    def _clip_arm_to_limits(pos_7: np.ndarray) -> np.ndarray:
        return np.clip(pos_7, JOINT_LIMITS_LOW, JOINT_LIMITS_HIGH)

    def _publish_joint_positions(self, left_pos: np.ndarray, right_pos: np.ndarray):
        msg = JointTrajectory()
        msg.joint_names = list(JOINT_NAMES)
        point = JointTrajectoryPoint()
        point.positions = np.concatenate([left_pos, right_pos]).tolist()
        duration_sec = self._control_dt * 1.5 if self.interp_alpha < 0.3 else self._control_dt
        sec, nanosec = int(duration_sec), int((duration_sec - int(duration_sec)) * 1e9)
        point.time_from_start = Duration(sec=sec, nanosec=nanosec)
        msg.points = [point]
        self._pub_joint_cmd.publish(msg)

    def _publish_home_positions(self):
        self._publish_joint_positions(DEFAULT_LEFT, DEFAULT_RIGHT)

    def _control_step(self):
        if self._prev_control_pos is not None:
            for name in JOINT_NAMES:
                dp = self.joint_positions[name] - self._prev_control_pos[name]
                vel = max(-JOINT_VEL_CLIP, min(JOINT_VEL_CLIP, dp / self._control_dt))
                self.joint_velocities[name] = vel
        self._prev_control_pos = {n: self.joint_positions[n] for n in JOINT_NAMES}

        obs_60 = self._build_obs_60()
        if obs_60 is None:
            self.get_logger().warn(
                "等待 /reach_targets 与 TF (left_link7, right_link7)",
                throttle_duration_sec=2.0
            )
            return

        obs_left, obs_right = obs_60_to_obs_left_right(obs_60)
        # 直接喂 SKRL agent：内部会做与 train/play 一致的归一化与 policy 前向
        obs_dict = {
            "left": torch.from_numpy(obs_left).float().unsqueeze(0),
            "right": torch.from_numpy(obs_right).float().unsqueeze(0),
        }
        with torch.inference_mode():
            actions_raw, _, outputs = self._agent.act(obs_dict, timestep=0, timesteps=0)
            action_left = outputs["left"].get("mean_actions", actions_raw["left"]).squeeze(0).cpu().numpy()
            action_right = outputs["right"].get("mean_actions", actions_raw["right"]).squeeze(0).cpu().numpy()
        action_flat = np.concatenate([np.clip(action_left, -1.0, 1.0), np.clip(action_right, -1.0, 1.0)]).astype(np.float32)

        left_decoded, right_decoded = self._decode_action_to_joint_positions(action_flat)
        raw_action_msg = Float32MultiArray()
        raw_action_msg.data = np.concatenate([left_decoded, right_decoded]).tolist()
        self._pub_raw_action.publish(raw_action_msg)
        left_target = self._clip_arm_to_limits(left_decoded)
        right_target = self._clip_arm_to_limits(right_decoded)
        self._left_prev_pos = left_target.copy()
        self._right_prev_pos = right_target.copy()

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
            print(f"[MAPPO_SKRL step {self._step_count}] obs_left: {_fmt(obs_left)} -> action_left: {_fmt(action_left)}")
            print(f"  obs_right: {_fmt(obs_right)} -> action_right: {_fmt(action_right)}")
        if self.print_io and (self._step_count % self.print_io_interval == 0):
            print(f"[MAPPO_IO step={self._step_count}] obs_left: " + ",".join(f"{x:.6f}" for x in obs_left))
            print(f"[MAPPO_IO step={self._step_count}] obs_right: " + ",".join(f"{x:.6f}" for x in obs_right))
            print(f"[MAPPO_IO step={self._step_count}] action_14: " + ",".join(f"{x:.6f}" for x in action_flat))
        self._step_count += 1


def main():
    parser = argparse.ArgumentParser(
        description="Gazebo MAPPO：直接加载 SKRL checkpoint，用 agent.act(obs) 推理，与 Isaac play 同一套代码路径。"
    )
    parser.add_argument(
        "--checkpoint", "-c", type=str, default=None,
        help="SKRL MAPPO checkpoint 路径。不填时使用 config/best_agent.pt（包 share 或当前目录）"
    )
    parser.add_argument("--hz", type=float, default=30.0)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug-interval", type=int, default=30)
    parser.add_argument("--print-io", action="store_true")
    parser.add_argument("--print-io-interval", type=int, default=30)
    parser.add_argument("--interp-alpha", type=float, default=0.2)
    parser.add_argument("--use-sim-time", action="store_true", default=True)
    parser.add_argument("--no-sim-time", dest="use_sim_time", action="store_false")
    parser.add_argument("--device", type=str, default="cpu", help="torch device（ROS 下通常 cpu）")
    args = parser.parse_args()

    checkpoint_path = args.checkpoint
    if checkpoint_path is None or checkpoint_path == "":
        try:
            from ament_index_python.packages import get_package_share_directory
            pkg_share = get_package_share_directory("nero_description")
            checkpoint_path = os.path.join(pkg_share, "config", "best_agent.pt")
        except Exception:
            checkpoint_path = os.path.join(os.path.dirname(__file__), "..", "config", "best_agent.pt")
        checkpoint_path = os.path.abspath(checkpoint_path)
        if not os.path.isfile(checkpoint_path):
            print("未指定 --checkpoint 且未找到 config/best_agent.pt，请指定 checkpoint 路径。")
            print("  例: ros2 run nero_description gazebo_mappo_play_skrl -c /path/to/best_agent.pt")
            return 1
    else:
        checkpoint_path = os.path.abspath(checkpoint_path)
        if not os.path.isfile(checkpoint_path):
            print(f"Checkpoint 不存在: {checkpoint_path}")
            return 1

    rclpy.init()
    node = GazeboMappoPlaySkrlNode(
        checkpoint_path=checkpoint_path,
        control_hz=args.hz,
        debug=args.debug,
        debug_interval=args.debug_interval,
        use_sim_time=args.use_sim_time,
        print_io=args.print_io,
        print_io_interval=args.print_io_interval,
        interp_alpha=args.interp_alpha,
        device=args.device,
    )
    node.get_logger().info(
        f"gazebo_mappo_play_skrl 已启动: checkpoint={checkpoint_path}, hz={args.hz}"
        + (f" [DEBUG 每 {args.debug_interval} 步]" if args.debug else "")
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
