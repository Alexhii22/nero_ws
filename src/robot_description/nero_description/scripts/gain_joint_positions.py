#!/usr/bin/env python3
# Copyright 2025. 输出与 reach_env_cfg PolicyCfg 一致的状态向量（30 维，单臂左）：keypoints_error_world(9)+joint_pos(7)+joint_vel(7)+joint_prev_pos(7)。
# 用法: ros2 run nero_description gain_joint_positions
# 需: Gazebo + /joint_states + /reach_targets + TF (base_link, left_link7)

import sys
import math

try:
    import numpy as np
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    from geometry_msgs.msg import PoseArray
    from tf2_ros import Buffer, TransformListener, TransformException
except ImportError as e:
    print(f"缺少依赖: {e}")
    sys.exit(1)

JOINT_NAMES = [
    "left_joint1", "left_joint2", "left_joint3", "left_joint4",
    "left_joint5", "left_joint6", "left_joint7",
    "right_joint1", "right_joint2", "right_joint3", "right_joint4",
    "right_joint5", "right_joint6", "right_joint7",
]
# 与 joint_pos_env_cfg / bi_nero 默认位姿一致，用于 joint_pos_rel = current - default
DEFAULT_LEFT = np.array([1.6, 1.2, 0.52, 0.52, -0.6, 0.0, 0.0], dtype=np.float32)
DEFAULT_RIGHT = np.array([-1.6, 1.2, -0.52, 0.52, 0.6, 0.0, 0.0], dtype=np.float32)
KEYPOINT_SCALE = 0.25
OBS_FRAME = "base_link"


def quat_inv(q):
    """单位四元数逆 (x,y,z,w)"""
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float32)


def quat_mul(q1, q2):
    """四元数乘法 q1*q2 (x,y,z,w)"""
    x1, y1, z1, w1 = q1[0], q1[1], q1[2], q1[3]
    x2, y2, z2, w2 = q2[0], q2[1], q2[2], q2[3]
    return np.array([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ], dtype=np.float32)


def quat_rotate_vector(q, v):
    """四元数 q (x,y,z,w) 旋转向量 v"""
    qv = np.array([v[0], v[1], v[2], 0.0], dtype=np.float32)
    qv_rot = quat_mul(quat_mul(q, qv), quat_inv(q))
    return qv_rot[:3]


# 与 observations.py 一致：3 个正轴关键点 0.25*[1,0,0],[0,1,0],[0,0,1]
OFFSETS_3 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32) * KEYPOINT_SCALE


def keypoints_in_world(pos, quat):
    """3 个正轴关键点从位姿系变换到世界系。返回 (9,) = 3 点×(x,y,z)。"""
    out = []
    for o in OFFSETS_3:
        pt = quat_rotate_vector(quat, o) + pos
        out.extend(pt.tolist())
    return np.array(out, dtype=np.float32)


def keypoints_error_world(ee_pos, ee_quat, tgt_pos, tgt_quat):
    """关键点误差（世界系）：target_keypoints - ee_keypoints，与 obs_keypoints_error_world 一致。(9,)"""
    ee_kps = keypoints_in_world(ee_pos, ee_quat)
    tgt_kps = keypoints_in_world(tgt_pos, tgt_quat)
    return tgt_kps - ee_kps


def pose_to_pos_quat(pose):
    """geometry_msgs Pose -> pos(3), quat(4) x,y,z,w"""
    p = np.array([pose.position.x, pose.position.y, pose.position.z], dtype=np.float32)
    q = np.array([
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w
    ], dtype=np.float32)
    return p, q


def transform_pose_to_frame(buf, target_frame, source_frame, pos, quat, now):
    """将 pos/quat（在 source_frame）变换到 target_frame"""
    try:
        t = buf.lookup_transform(
            target_frame, source_frame, now, rclpy.duration.Duration(seconds=0.1)
        )
        q_t = np.array([
            t.transform.rotation.x, t.transform.rotation.y,
            t.transform.rotation.z, t.transform.rotation.w
        ], dtype=np.float32)
        t_vec = np.array([
            t.transform.translation.x, t.transform.translation.y, t.transform.translation.z
        ], dtype=np.float32)
        pos_t = quat_rotate_vector(q_t, pos) + t_vec
        quat_t = quat_mul(q_t, quat)
        return pos_t, quat_t
    except (TransformException, Exception):
        return None, None


class GainJointPositionsNode(Node):
    def __init__(self):
        super().__init__("gain_joint_positions")
        self.declare_parameter("print_rate", 1.0)
        self.declare_parameter("obs_frame", OBS_FRAME)
        print_rate = self.get_parameter("print_rate").value
        self.obs_frame = self.get_parameter("obs_frame").value

        self.joint_positions = {n: 0.0 for n in JOINT_NAMES}
        self.joint_velocities = {n: 0.0 for n in JOINT_NAMES}
        self._prev_pos = {n: 0.0 for n in JOINT_NAMES}
        self._prev_time = None
        self.target_poses = None  # (left_pos, left_quat, right_pos, right_quat) in obs_frame

        self.js_sub = self.create_subscription(
            JointState, "/joint_states", self._joint_states_cb, 10
        )
        self.target_sub = self.create_subscription(
            PoseArray, "/reach_targets", self._target_cb, 10
        )
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.timer = self.create_timer(1.0 / print_rate, self._print_state)

    def _joint_states_cb(self, msg):
        now = self.get_clock().now()
        for i, name in enumerate(msg.name):
            if name not in self.joint_positions:
                continue
            pos = msg.position[i] if i < len(msg.position) else 0.0
            self.joint_positions[name] = pos
            if self._prev_time is not None:
                dt = (now - self._prev_time).nanoseconds * 1e-9
                if dt > 0:
                    self.joint_velocities[name] = (pos - self._prev_pos[name]) / dt
            else:
                self.joint_velocities[name] = msg.velocity[i] if i < len(msg.velocity) else 0.0
            self._prev_pos[name] = pos
        self._prev_time = now

    def _target_cb(self, msg):
        if len(msg.poses) < 2:
            return
        src = msg.header.frame_id or "world"
        now = rclpy.time.Time()
        rp, rq = pose_to_pos_quat(msg.poses[0])
        lp, lq = pose_to_pos_quat(msg.poses[1])
        if src and src != self.obs_frame:
            rp, rq = transform_pose_to_frame(self.tf_buffer, self.obs_frame, src, rp, rq, now)
            lp, lq = transform_pose_to_frame(self.tf_buffer, self.obs_frame, src, lp, lq, now)
            if rp is None or lp is None:
                return
        self.target_poses = (lp, lq, rp, rq)

    def _get_link_pose(self, child_frame):
        try:
            t = self.tf_buffer.lookup_transform(
                self.obs_frame, child_frame, rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1)
            )
            p = np.array([
                t.transform.translation.x, t.transform.translation.y, t.transform.translation.z
            ], dtype=np.float32)
            q = np.array([
                t.transform.rotation.x, t.transform.rotation.y,
                t.transform.rotation.z, t.transform.rotation.w
            ], dtype=np.float32)
            return p, q
        except Exception:
            return None, None

    def _build_obs_vector(self):
        """构建与 reach_env_cfg PolicyCfg 一致：keypoints_error_world(9)+joint_pos(7)+joint_vel(7)+joint_prev_pos(7)=30D。"""
        if self.target_poses is None:
            return None
        lp_t, lq_t, _, _ = self.target_poses
        left_pos, left_quat = self._get_link_pose("left_link7")
        if left_pos is None:
            return None

        left_kp_err = keypoints_error_world(left_pos, left_quat, lp_t, lq_t)
        left_joint_pos = np.array(
            [self.joint_positions[f"left_joint{i}"] for i in range(1, 8)], dtype=np.float32
        )
        left_joint_vel = np.array(
            [self.joint_velocities[f"left_joint{i}"] for i in range(1, 8)], dtype=np.float32
        )
        # joint_prev_pos = default + scale*last_action；本节点无 action，用 default 占位
        left_joint_prev_pos = DEFAULT_LEFT.copy()

        obs = np.concatenate([
            left_kp_err, left_joint_pos, left_joint_vel, left_joint_prev_pos,
        ], dtype=np.float32)
        return obs

    def _print_state(self):
        if self.target_poses is None:
            self.get_logger().warn("等待 /reach_targets ...", throttle_duration_sec=2.0)
            return
        obs = self._build_obs_vector()
        if obs is None:
            self.get_logger().warn("无法获取 TF (left_link7)", throttle_duration_sec=2.0)
            return

        print("\n" + "=" * 60)
        print("状态向量 (30 维，与 reach PolicyCfg 一致，相对 {})".format(self.obs_frame))
        print("顺序: left_keypoints_error_world(9) left_joint_pos(7) left_joint_vel(7) left_joint_prev_pos(7)")
        print("一行浮点数 (policy 输入):")
        print(" ".join(f"{x:.6f}" for x in obs))
        print("=" * 60)


def main():
    rclpy.init()
    node = GainJointPositionsNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
