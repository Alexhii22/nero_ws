#!/usr/bin/env python3
# Copyright 2025. 在 world 系下发布左右臂**移动目标**位姿，与 reach_env_cfg.py 的 DynamicSweepPoseCommand 逻辑一致。
# 用 RViz Marker 可视化；发布 /reach_targets (PoseArray) 与目标关键点，与 PPO obs_target_keypoints_world 一致。
# 用法: ros2 run nero_description spawn_reach_targets
# 需启动 Gazebo 仿真 + RViz（Add -> MarkerArray -> Topic: /reach_target_markers, Fixed Frame: world）

import argparse
import math
import random
import sys

try:
    import rclpy
    from rclpy.node import Node
    from visualization_msgs.msg import MarkerArray, Marker
    from geometry_msgs.msg import PoseArray, Pose
    from tf2_ros import Buffer, TransformListener
    from tf2_ros import TransformException
except ImportError as e:
    print(f"需要 ROS2 环境: {e}")
    sys.exit(1)

EE_FRAME = "world"
KEYPOINT_SCALE = 0.25

# ---------- 与 reach_env_cfg.py DynamicSweepPoseCommandCfg 完全一致 ----------
# Left (左臂): Y 从 -0.05 向 -0.4 匀速移动，到达后切到 rest，等待 0.5~2s 后新目标在 start_pos_y 随机 X 出现
LEFT_CFG = {
    "pos_x_min": -0.4,
    "pos_x_max": -0.1,
    "start_pos_y": -0.05,
    "end_pos_y": -0.4,
    "fixed_pos_z": 0.40,
    "velocity": 0.20,
    "rest_pos_x": -0.45,
    "rest_pos_y": -0.05,
    "rest_pos_z": 0.55,
    "wait_time_min": 1.0,
    "wait_time_max": 3.0,
    "roll": (-math.pi / 9, math.pi / 9),
    "pitch": (3 * math.pi / 2 - math.pi / 2, 3 * math.pi / 2 - math.pi / 2),  # (pi, pi)
    "yaw": (math.pi + 9.5 * math.pi / 10, 2 * math.pi),
}

# Right (右臂): Y 从 0.4 向 0.05 匀速移动（代码里统一用 y -= velocity*dt）
RIGHT_CFG = {
    "pos_x_min": -0.4,
    "pos_x_max": -0.1,
    "start_pos_y": 0.5,
    "end_pos_y": 0.05,
    "fixed_pos_z": 0.40,
    "velocity": 0.20,
    "rest_pos_x": -0.45,
    "rest_pos_y": 0.30,
    "rest_pos_z": 0.55,
    "wait_time_min": 1.0,
    "wait_time_max": 3.0,
    "roll": (-math.pi / 9, math.pi / 9),
    "pitch": (math.pi + 3 * math.pi / 2 - math.pi / 2, math.pi + 3 * math.pi / 2 - math.pi / 2),  # (2*pi, 2*pi)
    "yaw": (9.6 * math.pi / 10, 10.4 * math.pi / 10),
}


def rpy_to_quaternion(roll, pitch, yaw):
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return (x, y, z, w)


def quat_rotate_vector(q, v):
    qx, qy, qz, qw = q[0], q[1], q[2], q[3]
    vx, vy, vz = v[0], v[1], v[2]
    return (
        (1 - 2 * qy * qy - 2 * qz * qz) * vx + (2 * qx * qy - 2 * qz * qw) * vy + (2 * qx * qz + 2 * qy * qw) * vz,
        (2 * qx * qy + 2 * qz * qw) * vx + (1 - 2 * qx * qx - 2 * qz * qz) * vy + (2 * qy * qz - 2 * qx * qw) * vz,
        (2 * qx * qz - 2 * qy * qw) * vx + (2 * qy * qz + 2 * qx * qw) * vy + (1 - 2 * qx * qx - 2 * qy * qy) * vz,
    )


def get_target_keypoints_world(pos, quat, keypoint_scale=KEYPOINT_SCALE):
    corners = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    kps = []
    for c in corners:
        offset = [c[0] * keypoint_scale, c[1] * keypoint_scale, c[2] * keypoint_scale]
        kp = [pos[i] + quat_rotate_vector(quat, offset)[i] for i in range(3)]
        kps.append(kp)
    return kps


class MovingTargetState:
    """单臂移动目标状态机，与 DynamicSweepPoseCommand 一致。"""

    def __init__(self, cfg, is_left=True):
        self.cfg = cfg
        self.is_left = is_left
        self.pos_x = 0.0
        self.pos_y = 0.0
        self.pos_z = cfg["fixed_pos_z"]
        self.roll = random.uniform(*cfg["roll"])
        self.pitch = cfg["pitch"][0]
        self.yaw = random.uniform(*cfg["yaw"])
        self.is_moving = True
        self.wait_timer = 0.0
        self._reset_target()

    def _reset_target(self):
        self.pos_x = random.uniform(self.cfg["pos_x_min"], self.cfg["pos_x_max"])
        self.pos_y = self.cfg["start_pos_y"]
        self.pos_z = self.cfg["fixed_pos_z"]
        self.roll = random.uniform(*self.cfg["roll"])
        self.pitch = self.cfg["pitch"][0]
        self.yaw = random.uniform(*self.cfg["yaw"])
        self.is_moving = True

    def _set_rest_target(self):
        self.pos_x = self.cfg["rest_pos_x"]
        self.pos_y = self.cfg["rest_pos_y"]
        self.pos_z = self.cfg["rest_pos_z"]
        self.is_moving = False
        self.wait_timer = random.uniform(self.cfg["wait_time_min"], self.cfg["wait_time_max"])

    def step(self, dt):
        if self.is_moving:
            self.pos_y -= self.cfg["velocity"] * dt
            if self.pos_y <= self.cfg["end_pos_y"]:
                self._set_rest_target()
        else:
            self.wait_timer -= dt
            if self.wait_timer <= 0.0:
                self._reset_target()

    def pose(self):
        return (self.pos_x, self.pos_y, self.pos_z, self.roll, self.pitch, self.yaw)


class ReachTargetMarkerNode(Node):
    """发布左右臂移动目标位姿（与 reach_env_cfg DynamicSweepPoseCommand 一致）及 link7 关键点。"""

    def __init__(self, update_hz=50.0, seed=None):
        super().__init__("spawn_reach_targets")
        if seed is not None:
            random.seed(seed)

        self.marker_pub = self.create_publisher(MarkerArray, "/reach_target_markers", 10)
        self.pose_pub = self.create_publisher(PoseArray, "/reach_targets", 10)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self._left_state = MovingTargetState(LEFT_CFG, is_left=True)
        self._right_state = MovingTargetState(RIGHT_CFG, is_left=False)
        self._dt = 1.0 / update_hz
        self._update_timer = self.create_timer(self._dt, self._update_moving_targets)
        self._log_count = 0

    def _make_marker(self, marker_id, pose, color, label, lifetime_sec=1):
        m = Marker()
        m.header.frame_id = "world"
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = "reach_targets"
        m.id = marker_id
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x, m.pose.position.y, m.pose.position.z = pose[0], pose[1], pose[2]
        q = rpy_to_quaternion(pose[3], pose[4], pose[5])
        m.pose.orientation.x, m.pose.orientation.y, m.pose.orientation.z, m.pose.orientation.w = q
        m.scale.x = m.scale.y = m.scale.z = 0.04
        m.color.r, m.color.g, m.color.b, m.color.a = color
        m.lifetime.sec = max(1, int(lifetime_sec))
        m.lifetime.nanosec = 0
        m.text = label
        return m

    def _make_keypoint_marker(self, marker_id, pos_xyz, color, label, ns="reach_target_keypoints", lifetime_sec=1):
        m = Marker()
        m.header.frame_id = "world"
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = ns
        m.id = marker_id
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x, m.pose.position.y, m.pose.position.z = pos_xyz[0], pos_xyz[1], pos_xyz[2]
        m.pose.orientation.x, m.pose.orientation.y, m.pose.orientation.z, m.pose.orientation.w = 0.0, 0.0, 0.0, 1.0
        m.scale.x = m.scale.y = m.scale.z = 0.02
        m.color.r, m.color.g, m.color.b, m.color.a = color
        m.lifetime.sec = max(1, int(lifetime_sec))
        m.lifetime.nanosec = 0
        m.text = label
        return m

    def _get_link_pose(self, child_frame):
        try:
            t = self.tf_buffer.lookup_transform(
                EE_FRAME, child_frame, rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1)
            )
            p = (t.transform.translation.x, t.transform.translation.y, t.transform.translation.z)
            q = (t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w)
            return p, q
        except (TransformException, Exception):
            return None, None

    def _update_moving_targets(self):
        self._left_state.step(self._dt)
        self._right_state.step(self._dt)

        right_pose = self._right_state.pose()
        left_pose = self._left_state.pose()
        right_pos = (right_pose[0], right_pose[1], right_pose[2])
        right_quat = rpy_to_quaternion(right_pose[3], right_pose[4], right_pose[5])
        left_pos = (left_pose[0], left_pose[1], left_pose[2])
        left_quat = rpy_to_quaternion(left_pose[3], left_pose[4], left_pose[5])
        right_kps = get_target_keypoints_world(right_pos, right_quat)
        left_kps = get_target_keypoints_world(left_pos, left_quat)

        # Markers: 目标球 + 关键点
        lifetime = 2
        arr = MarkerArray()
        arr.markers.append(self._make_marker(0, right_pose, (1.0, 0.0, 0.0, 0.9), "right_target", lifetime))
        arr.markers.append(self._make_marker(1, left_pose, (0.0, 0.0, 1.0, 0.9), "left_target", lifetime))
        for i, kp in enumerate(right_kps):
            arr.markers.append(self._make_keypoint_marker(i, kp, (1.0, 0.5, 0.0, 0.95), f"right_tgt_kp[{i}]", lifetime_sec=lifetime))
        for i, kp in enumerate(left_kps):
            arr.markers.append(self._make_keypoint_marker(10 + i, kp, (0.0, 0.8, 1.0, 0.95), f"left_tgt_kp[{i}]", lifetime_sec=lifetime))

        # 末端 link7 关键点（若 TF 可用）
        left_ee_pos, left_ee_quat = self._get_link_pose("left_link7")
        right_ee_pos, right_ee_quat = self._get_link_pose("right_link7")
        if left_ee_pos is not None and right_ee_pos is not None:
            left_ee_kps = get_target_keypoints_world(left_ee_pos, left_ee_quat)
            right_ee_kps = get_target_keypoints_world(right_ee_pos, right_ee_quat)
            for i, kp in enumerate(right_ee_kps):
                arr.markers.append(self._make_keypoint_marker(i, kp, (1.0, 0.3, 0.0, 0.9), f"right_ee_kp[{i}]", ns="ee_keypoints", lifetime_sec=1))
            for i, kp in enumerate(left_ee_kps):
                arr.markers.append(self._make_keypoint_marker(10 + i, kp, (0.3, 0.6, 1.0, 0.9), f"left_ee_kp[{i}]", ns="ee_keypoints", lifetime_sec=1))

        self.marker_pub.publish(arr)

        # PoseArray: [right, left] 与 Isaac 命令顺序一致
        pa = PoseArray()
        pa.header.frame_id = "world"
        pa.header.stamp = self.get_clock().now().to_msg()
        for pose in (right_pose, left_pose):
            p = Pose()
            p.position.x, p.position.y, p.position.z = pose[0], pose[1], pose[2]
            q = rpy_to_quaternion(pose[3], pose[4], pose[5])
            p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = q
            pa.poses.append(p)
        self.pose_pub.publish(pa)

        self._log_count += 1
        if self._log_count % 100 == 0:
            self.get_logger().info(
                f"移动目标: 右 ({right_pose[0]:.2f},{right_pose[1]:.2f},{right_pose[2]:.2f}) "
                f"左 ({left_pose[0]:.2f},{left_pose[1]:.2f},{left_pose[2]:.2f})"
            )

def main():
    parser = argparse.ArgumentParser(
        description="发布左右臂移动目标位姿（与 reach_env_cfg DynamicSweepPoseCommand 一致），RViz Marker 可视化"
    )
    parser.add_argument("--update-hz", type=float, default=50.0, help="目标状态更新频率 (Hz)")
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    args, _ = parser.parse_known_args()

    rclpy.init()
    node = ReachTargetMarkerNode(update_hz=args.update_hz, seed=args.seed)
    print("=" * 60)
    print("reach 移动目标节点已启动（与 reach_env_cfg.py DynamicSweepPoseCommand 一致）")
    print("  Topic: /reach_target_markers (MarkerArray)")
    print("  Topic: /reach_targets (PoseArray)")
    print(f"  更新频率: {args.update_hz} Hz")
    print("  左臂: Y 从 -0.05 → -0.4，到达后 rest 再等待 0.5~2s 后新目标")
    print("  右臂: Y 从 0.4 → 0.05，同上")
    print("  RViz: Add -> MarkerArray -> Topic: /reach_target_markers, Fixed Frame: world")
    print("=" * 60)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
