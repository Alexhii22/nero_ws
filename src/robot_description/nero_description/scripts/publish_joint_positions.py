#!/usr/bin/env python3
# Copyright 2025. 通过 joint_trajectory_controller (JTC) 发布关节位置目标（rad）。
# JTC 内部做插值平滑，不会瞬移。
# 用法:
#   单次发布: ros2 run nero_description publish_joint_positions [left1 ... right7]
#   单次+指定时间: ros2 run nero_description publish_joint_positions --duration 2.0 [left1 ... right7]
#   插值平滑: ros2 run nero_description publish_joint_positions --interp-alpha 0.2 [left1 ... right7]
#   速度诊断(臂不动): ros2 run nero_description publish_joint_positions --vel-check
#   插值+速度(臂动且打印速度): ros2 run nero_description publish_joint_positions --interp-alpha 0.2 --vel-check

import argparse
import sys

try:
    import numpy as np
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
    from builtin_interfaces.msg import Duration
except ImportError as e:
    print(f"需要 ROS2 环境与依赖: {e}")
    sys.exit(1)

JOINT_NAMES = [
    "left_joint1", "left_joint2", "left_joint3", "left_joint4",
    "left_joint5", "left_joint6", "left_joint7",
    "right_joint1", "right_joint2", "right_joint3", "right_joint4",
    "right_joint5", "right_joint6", "right_joint7",
]

JTC_TOPIC = "/joint_trajectory_controller/joint_trajectory"

# 差分速度估计与 gazebo_ppo_play 一致
VEL_CLIP = 3.0  # rad/s

DEFAULT_POSITIONS = [
     1.6, 1.2, 0.52, 0.52, -0.6, 0.0, 0.0,
     -1.6, 1.2, -0.52, 0.52, 0.6, 0.0, 0.0,
]

# 关节限位 (rad)，与 gazebo_ppo_play 一致
_deg = np.pi / 180.0
JOINT_LIMITS_LOW = np.array([
    -157 * _deg, -15 * _deg, -160 * _deg, -60 * _deg, -160 * _deg, -43 * _deg, -90 * _deg,
    -157 * _deg, -15 * _deg, -160 * _deg, -60 * _deg, -160 * _deg, -43 * _deg, -90 * _deg,
])
JOINT_LIMITS_HIGH = np.array([
    157 * _deg, 190 * _deg, 160 * _deg, 125 * _deg, 160 * _deg, 58 * _deg, 90 * _deg,
    157 * _deg, 190 * _deg, 160 * _deg, 125 * _deg, 160 * _deg, 58 * _deg, 90 * _deg,
])


def clip_to_limits(pos: np.ndarray) -> np.ndarray:
    return np.clip(pos, JOINT_LIMITS_LOW, JOINT_LIMITS_HIGH).astype(np.float64)


def make_trajectory_msg(positions: list[float], duration_sec: float) -> JointTrajectory:
    """构造 JointTrajectory 消息：单个目标点 + 到达时间。
    JTC 会从当前位置平滑插值到 positions，在 duration_sec 内完成。"""
    msg = JointTrajectory()
    msg.joint_names = list(JOINT_NAMES)
    point = JointTrajectoryPoint()
    point.positions = [float(x) for x in positions]
    sec = int(duration_sec)
    nanosec = int((duration_sec - sec) * 1e9)
    point.time_from_start = Duration(sec=sec, nanosec=nanosec)
    msg.points = [point]
    return msg


class VelCheckNode(Node):
    """诊断：用「控制步长」差分算速度（与 gazebo_ppo_play 一致），避免 joint_states 连续同值导致 vel 恒 0。"""

    def __init__(self, print_hz: float = 2.0, control_hz: float = 30.0):
        super().__init__("vel_check")
        self.print_hz = print_hz
        self.control_hz = control_hz
        self.control_dt = 1.0 / control_hz
        self._positions = {n: 0.0 for n in JOINT_NAMES}
        self._velocities = {n: 0.0 for n in JOINT_NAMES}
        self._prev_control_pos = None  # 上一「控制步」位置，用于 vel = (pos - prev) / control_dt
        self._cb_count = 0

        self._sub = self.create_subscription(
            JointState, "/joint_states", self._joint_states_cb, 10
        )
        self._timer = self.create_timer(self.control_dt, self._control_step)  # 与控制步同频
        self._print_timer = self.create_timer(1.0 / self.print_hz, self._print_cb)
        self.get_logger().info(
            f"速度诊断: 控制步长差分 vel=(pos-prev)/{self.control_dt:.3f}s, 每 {1/self.print_hz}s 打印 (Ctrl+C 退出)"
        )

    def _joint_states_cb(self, msg):
        """只更新位置；速度在 _control_step 里用控制步长算。"""
        self._cb_count += 1
        for i, name in enumerate(msg.name):
            if name not in self._positions:
                continue
            self._positions[name] = msg.position[i] if i < len(msg.position) else 0.0

    def _control_step(self):
        """与 gazebo_ppo_play 一致：vel = (当前 pos - 上一步 pos) / control_dt。"""
        if self._prev_control_pos is not None:
            for name in JOINT_NAMES:
                dp = self._positions[name] - self._prev_control_pos[name]
                vel = dp / self.control_dt
                vel = max(-VEL_CLIP, min(VEL_CLIP, vel))
                self._velocities[name] = vel
        self._prev_control_pos = dict(self._positions)

    def _print_cb(self):
        left_pos = [self._positions[f"left_joint{i}"] for i in range(1, 8)]
        left_vel = [self._velocities[f"left_joint{i}"] for i in range(1, 8)]
        right_pos = [self._positions[f"right_joint{i}"] for i in range(1, 8)]
        right_vel = [self._velocities[f"right_joint{i}"] for i in range(1, 8)]
        print(
            f"[vel_check] callbacks={self._cb_count} control_dt={self.control_dt:.3f}s\n"
            f"  left_pos(7):  {_fmt(left_pos)}\n"
            f"  left_vel(7):  {_fmt(left_vel)}\n"
            f"  right_pos(7): {_fmt(right_pos)}\n"
            f"  right_vel(7): {_fmt(right_vel)}"
        )


def _fmt(arr, fmt=".4f"):
    return " ".join(f"{x:{fmt}}" for x in arr)


class InterpPublisherNode(Node):
    """当前姿态 → 目标姿态 插值发布（通过 JTC trajectory 消息）。可选同时做控制步长速度诊断并打印。"""

    def __init__(self, target: np.ndarray, interp_alpha: float, hz: float, vel_check: bool = False, vel_print_hz: float = 2.0):
        super().__init__("publish_joint_positions")
        self.target = np.asarray(target, dtype=np.float64)
        self.alpha = max(0.01, min(1.0, interp_alpha))
        self.hz = hz
        self.vel_check = vel_check
        self.control_dt = 1.0 / hz
        # 当前下发的设定点，先设为默认，收到 joint_states 后会用当前实际位置覆盖一次
        self._sent = np.array(DEFAULT_POSITIONS, dtype=np.float64)
        self._got_states = False
        # 速度诊断：当前/上一步位置，用于 vel=(pos-prev)/control_dt
        self._positions = {n: 0.0 for n in JOINT_NAMES}
        self._velocities = {n: 0.0 for n in JOINT_NAMES}
        self._prev_control_pos = None

        self._pub = self.create_publisher(
            JointTrajectory, JTC_TOPIC, 10
        )
        self._sub = self.create_subscription(
            JointState, "/joint_states", self._joint_states_cb, 10
        )
        self._timer = self.create_timer(1.0 / self.hz, self._timer_cb)
        if self.vel_check:
            self._print_timer = self.create_timer(1.0 / vel_print_hz, self._print_vel_cb)
        self.get_logger().info(
            f"插值发布: alpha={self.alpha}, hz={self.hz}, 目标(前3)={self.target[:3].tolist()}..."
            + (" [同时打印控制步长速度]" if self.vel_check else "")
        )

    def _joint_states_cb(self, msg):
        pos_by_name = {}
        for i, name in enumerate(msg.name):
            if name in JOINT_NAMES and i < len(msg.position):
                pos_by_name[name] = msg.position[i]
                if self.vel_check:
                    self._positions[name] = msg.position[i]
        if len(pos_by_name) == 14 and not self._got_states:
            self._sent = np.array([pos_by_name[n] for n in JOINT_NAMES], dtype=np.float64)
            self._sent = clip_to_limits(self._sent)
            self._got_states = True
            self.get_logger().info("已用 /joint_states 当前姿态作为插值起点")

    def _timer_cb(self):
        # 控制步长速度（与 gazebo_ppo_play 一致）
        if self.vel_check and self._prev_control_pos is not None:
            for name in JOINT_NAMES:
                dp = self._positions[name] - self._prev_control_pos[name]
                vel = max(-VEL_CLIP, min(VEL_CLIP, dp / self.control_dt))
                self._velocities[name] = vel
        if self.vel_check:
            self._prev_control_pos = dict(self._positions)
        # 插值发布
        self._sent = self._sent + self.alpha * (self.target - self._sent)
        self._sent = clip_to_limits(self._sent)
        # 通过 JTC 发布：目标位置 + 到达时间 = 1 个控制周期
        msg = make_trajectory_msg(self._sent.tolist(), self.control_dt)
        self._pub.publish(msg)

    def _print_vel_cb(self):
        left_pos = [self._positions[f"left_joint{i}"] for i in range(1, 8)]
        left_vel = [self._velocities[f"left_joint{i}"] for i in range(1, 8)]
        right_pos = [self._positions[f"right_joint{i}"] for i in range(1, 8)]
        right_vel = [self._velocities[f"right_joint{i}"] for i in range(1, 8)]
        print(
            f"[vel_check] control_dt={self.control_dt:.3f}s\n"
            f"  left_pos(7):  {_fmt(left_pos)}\n  left_vel(7):  {_fmt(left_vel)}\n"
            f"  right_pos(7): {_fmt(right_pos)}\n  right_vel(7): {_fmt(right_vel)}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="发布关节位置；可选插值平滑（当前→目标）"
    )
    parser.add_argument(
        "positions", nargs="*", type=float,
        help="14 个关节位置 (rad): left1..7 right1..7，不传则用默认姿态"
    )
    parser.add_argument(
        "--interp-alpha", type=float, default=None,
        help="插值系数 (0,1]，启用后周期性发布 sent=sent+alpha*(target-sent)，便于观察平滑效果"
    )
    parser.add_argument(
        "--hz", type=float, default=30.0,
        help="插值模式下的发布频率 (默认 30)"
    )
    parser.add_argument(
        "--duration", type=float, default=2.0,
        help="单次发布模式：到达目标的时间 (秒，默认 2.0)。JTC 在此时间内平滑插值"
    )
    parser.add_argument(
        "--vel-check", action="store_true",
        help="打印控制步长速度。单独使用时仅诊断不发布指令(臂不动)；与 --interp-alpha 同用则边发布边打印(臂会动)"
    )
    parser.add_argument(
        "--vel-check-hz", type=float, default=2.0,
        help="--vel-check 时打印频率 (默认 2)"
    )
    args = parser.parse_args()

    # 仅 --vel-check 且未开插值：只诊断不发布，臂不会动
    if args.vel_check and (args.interp_alpha is None or args.interp_alpha >= 1.0):
        rclpy.init()
        node = VelCheckNode(print_hz=args.vel_check_hz)
        node.get_logger().info("仅速度诊断，不发布关节指令，机械臂不会动；要让臂动同时看速度请加 --interp-alpha 0.2")
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
        node.destroy_node()
        rclpy.shutdown()
        return 0

    if len(args.positions) >= 14:
        target = np.array(args.positions[:14], dtype=np.float64)
    else:
        target = np.array(DEFAULT_POSITIONS, dtype=np.float64)

    # 插值模式：周期性发布，从当前点平滑到目标；可同时 --vel-check 打印速度
    if args.interp_alpha is not None and args.interp_alpha < 1.0:
        rclpy.init()
        node = InterpPublisherNode(
            target, args.interp_alpha, args.hz,
            vel_check=args.vel_check, vel_print_hz=args.vel_check_hz,
        )
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
        node.destroy_node()
        rclpy.shutdown()
        return 0

    # 单次发布：发送一条轨迹，JTC 在 duration 内平滑插值到目标
    rclpy.init()
    node = Node("publish_joint_positions")
    pub = node.create_publisher(JointTrajectory, JTC_TOPIC, 10)
    target_clipped = clip_to_limits(target)
    msg = make_trajectory_msg(target_clipped.tolist(), args.duration)

    # 等待至少一个订阅者连接（最多 5 秒）
    import time
    waited = 0.0
    while pub.get_subscription_count() == 0 and waited < 5.0:
        time.sleep(0.1)
        rclpy.spin_once(node, timeout_sec=0.01)
        waited += 0.1
    if pub.get_subscription_count() == 0:
        node.get_logger().warn("未检测到订阅者，joint_trajectory_controller 可能未启动")

    # 连续发布几次，确保送达
    for _ in range(5):
        pub.publish(msg)
        rclpy.spin_once(node, timeout_sec=0.05)

    node.get_logger().info(
        f"已发布轨迹到 {JTC_TOPIC}，目标将在 {args.duration}s 内平滑到达"
    )
    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
