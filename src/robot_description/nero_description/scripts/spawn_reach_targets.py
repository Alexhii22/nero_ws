#!/usr/bin/env python3
# Copyright 2025. 在 world 系下生成左右臂随机目标位姿，用 RViz Marker 可视化（无重力，纯虚拟）。
# 区间与 reach_env_cfg.py 一致。发布目标位姿、关键点世界坐标，与 PPO obs_target_keypoints_world 一致。
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

EE_FRAME = "world"  # TF 父坐标系，left_link7/right_link7 将变换到此系

# 与 observations.py obs_target_keypoints_world 一致：keypoint_scale=0.25, add_negative_axes=False → 3 点 (X+,Y+,Z+)
KEYPOINT_SCALE = 0.25

# 右臂目标位姿区间 (reach_env_cfg.py 80-85)
RIGHT_RANGES = {
    "pos_x": (0.15, 0.3),
    "pos_y": (0.15, 0.25),
    "pos_z": (0.3, 0.4),
    "roll": (-math.pi / 9, math.pi / 9),
    "pitch": (math.pi+3 * math.pi / 2 + math.pi/2, math.pi + 3 * math.pi / 2 + math.pi/2),
    "yaw": (9.6 * math.pi / 10, 10.4 * math.pi / 10),
}

# 左臂目标位姿区间 (reach_env_cfg.py 95-100)
LEFT_RANGES = {
    "pos_x": (0.15, 0.3),
    "pos_y": (-0.25, -0.15),
    "pos_z": (0.3, 0.4),
    "roll": (-math.pi / 9, math.pi / 9),
    "pitch": (3 * math.pi / 2+math.pi/2, 3 * math.pi / 2+math.pi/2),
    "yaw": (math.pi+9.5 * math.pi / 10, 2*math.pi),
}


def sample_pose(ranges):
    """在区间内随机采样位姿 (x,y,z, roll,pitch,yaw)"""
    pos_x = random.uniform(*ranges["pos_x"])
    pos_y = random.uniform(*ranges["pos_y"])
    pos_z = random.uniform(*ranges["pos_z"])
    roll = random.uniform(*ranges["roll"])
    pitch = ranges["pitch"][0]
    yaw = random.uniform(*ranges["yaw"])
    return (pos_x, pos_y, pos_z, roll, pitch, yaw)


def rpy_to_quaternion(roll, pitch, yaw):
    """欧拉角 (roll, pitch, yaw) 转四元数 (x, y, z, w)"""
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return (x, y, z, w)


def quat_rotate_vector(q, v):
    """四元数 q (x,y,z,w) 旋转向量 v，与 observations/rewards 一致"""
    qx, qy, qz, qw = q[0], q[1], q[2], q[3]
    vx, vy, vz = v[0], v[1], v[2]
    return (
        (1 - 2 * qy * qy - 2 * qz * qz) * vx + (2 * qx * qy - 2 * qz * qw) * vy + (2 * qx * qz + 2 * qy * qw) * vz,
        (2 * qx * qy + 2 * qz * qw) * vx + (1 - 2 * qx * qx - 2 * qz * qz) * vy + (2 * qy * qz - 2 * qx * qw) * vz,
        (2 * qx * qz - 2 * qy * qw) * vx + (2 * qy * qz + 2 * qx * qw) * vy + (1 - 2 * qx * qx - 2 * qy * qy) * vz,
    )


def get_target_keypoints_world(pos, quat, keypoint_scale=KEYPOINT_SCALE, add_negative_axes=False):
    """目标关键点世界坐标，与 observations.obs_target_keypoints_world 一致。
    pos: (x,y,z), quat: (x,y,z,w)。返回 3 点 9D 或 6 点 18D。"""
    corners = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    if add_negative_axes:
        corners += [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]
    kps = []
    for c in corners:
        offset = [c[0] * keypoint_scale, c[1] * keypoint_scale, c[2] * keypoint_scale]
        kp = [pos[i] + quat_rotate_vector(quat, offset)[i] for i in range(3)]
        kps.append(kp)
    return kps


class ReachTargetMarkerNode(Node):
    """发布左右臂目标位姿与左右 link7 关键点的 RViz Marker"""

    def __init__(self, interval=6.0, seed=None, ee_publish_hz=10.0):
        super().__init__("spawn_reach_targets")
        if seed is not None:
            random.seed(seed)

        self.marker_pub = self.create_publisher(
            MarkerArray, "/reach_target_markers", 10
        )
        self.pose_pub = self.create_publisher(
            PoseArray, "/reach_targets", 10
        )
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.interval = interval
        self._target_markers = []  # 目标+目标关键点 markers，供 _publish_ee 合并
        self.timer = self.create_timer(interval, self._update)
        self.ee_timer = self.create_timer(1.0 / ee_publish_hz, self._publish_ee_keypoints)

        self._update()

    def _make_marker(self, marker_id, pose, color, label):
        """创建单个球体 Marker"""
        m = Marker()
        m.header.frame_id = "world"
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = "reach_targets"
        m.id = marker_id
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x = pose[0]
        m.pose.position.y = pose[1]
        m.pose.position.z = pose[2]
        q = rpy_to_quaternion(pose[3], pose[4], pose[5])
        m.pose.orientation.x, m.pose.orientation.y, m.pose.orientation.z, m.pose.orientation.w = q
        m.scale.x = m.scale.y = m.scale.z = 0.04
        m.color.r, m.color.g, m.color.b, m.color.a = color
        m.lifetime.sec = int(self.interval) + 1
        m.lifetime.nanosec = 0
        m.text = label
        return m

    def _make_keypoint_marker(self, marker_id, pos_xyz, color, label, ns="reach_target_keypoints", lifetime_sec=2):
        """创建关键点小球 Marker（世界坐标）"""
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
        m.lifetime.sec = int(lifetime_sec)
        m.lifetime.nanosec = 0
        m.text = label
        return m

    def _get_link_pose(self, child_frame):
        """从 TF 获取 link 在 EE_FRAME 下的位姿 pos(x,y,z), quat(x,y,z,w)"""
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

    def _publish_ee_keypoints(self):
        """定时发布左右 link7 关键点，与目标 markers 合并后发布"""
        left_pos, left_quat = self._get_link_pose("left_link7")
        right_pos, right_quat = self._get_link_pose("right_link7")

        arr = MarkerArray()
        arr.markers.extend(self._target_markers)

        if left_pos is not None and right_pos is not None:
            left_kps = get_target_keypoints_world(left_pos, left_quat)
            right_kps = get_target_keypoints_world(right_pos, right_quat)
            for i, kp in enumerate(right_kps):
                arr.markers.append(self._make_keypoint_marker(
                    i, kp, (1.0, 0.3, 0.0, 0.9), f"right_ee_kp[{i}]",
                    ns="ee_keypoints", lifetime_sec=1
                ))
            for i, kp in enumerate(left_kps):
                arr.markers.append(self._make_keypoint_marker(
                    10 + i, kp, (0.3, 0.6, 1.0, 0.9), f"left_ee_kp[{i}]",
                    ns="ee_keypoints", lifetime_sec=1
                ))

        if arr.markers:
            self.marker_pub.publish(arr)

    def _update(self):
        right_pose = sample_pose(RIGHT_RANGES)
        left_pose = sample_pose(LEFT_RANGES)

        right_pos = (right_pose[0], right_pose[1], right_pose[2])
        right_quat = rpy_to_quaternion(right_pose[3], right_pose[4], right_pose[5])
        left_pos = (left_pose[0], left_pose[1], left_pose[2])
        left_quat = rpy_to_quaternion(left_pose[3], left_pose[4], left_pose[5])

        right_kps = get_target_keypoints_world(right_pos, right_quat)
        left_kps = get_target_keypoints_world(left_pos, left_quat)

        # 终端打印目标关键点世界坐标
        print("\n" + "=" * 60)
        print("目标关键点世界坐标 (world, 与 obs_target_keypoints_world 一致)")
        print(f"右臂目标 pos=({right_pos[0]:.4f}, {right_pos[1]:.4f}, {right_pos[2]:.4f})")
        print(f"  X+ kp: ({right_kps[0][0]:.4f}, {right_kps[0][1]:.4f}, {right_kps[0][2]:.4f})")
        print(f"  Y+ kp: ({right_kps[1][0]:.4f}, {right_kps[1][1]:.4f}, {right_kps[1][2]:.4f})")
        print(f"  Z+ kp: ({right_kps[2][0]:.4f}, {right_kps[2][1]:.4f}, {right_kps[2][2]:.4f})")
        print(f"右臂 9D: [{', '.join(f'{x:.4f}' for p in right_kps for x in p)}]")
        print(f"左臂目标 pos=({left_pos[0]:.4f}, {left_pos[1]:.4f}, {left_pos[2]:.4f})")
        print(f"  X+ kp: ({left_kps[0][0]:.4f}, {left_kps[0][1]:.4f}, {left_kps[0][2]:.4f})")
        print(f"  Y+ kp: ({left_kps[1][0]:.4f}, {left_kps[1][1]:.4f}, {left_kps[1][2]:.4f})")
        print(f"  Z+ kp: ({left_kps[2][0]:.4f}, {left_kps[2][1]:.4f}, {left_kps[2][2]:.4f})")
        print(f"左臂 9D: [{', '.join(f'{x:.4f}' for p in left_kps for x in p)}]")

        # 若 TF 可用，打印 link7 关键点
        left_ee_pos, left_ee_quat = self._get_link_pose("left_link7")
        right_ee_pos, right_ee_quat = self._get_link_pose("right_link7")
        if left_ee_pos is not None and right_ee_pos is not None:
            left_ee_kps = get_target_keypoints_world(left_ee_pos, left_ee_quat)
            right_ee_kps = get_target_keypoints_world(right_ee_pos, right_ee_quat)
            print("---")
            print("末端 link7 关键点世界坐标 (与 obs_ee_keypoints_world 一致)")
            print(f"右 link7 pos=({right_ee_pos[0]:.4f}, {right_ee_pos[1]:.4f}, {right_ee_pos[2]:.4f})")
            print(f"  9D: [{', '.join(f'{x:.4f}' for p in right_ee_kps for x in p)}]")
            print(f"左 link7 pos=({left_ee_pos[0]:.4f}, {left_ee_pos[1]:.4f}, {left_ee_pos[2]:.4f})")
            print(f"  9D: [{', '.join(f'{x:.4f}' for p in left_ee_kps for x in p)}]")
        else:
            print("--- (link7 TF 暂不可用，待 Gazebo+robot_state_publisher 发布)")
        print("=" * 60 + "\n")

        # 存储目标 markers，供 _publish_ee_keypoints 合并发布
        self._target_markers = []
        self._target_markers.append(self._make_marker(0, right_pose, (1.0, 0.0, 0.0, 0.9), "right_target"))
        self._target_markers.append(self._make_marker(1, left_pose, (0.0, 0.0, 1.0, 0.9), "left_target"))
        for i, kp in enumerate(right_kps):
            self._target_markers.append(self._make_keypoint_marker(
                i, kp, (1.0, 0.5, 0.0, 0.95), f"right_tgt_kp[{i}]",
                lifetime_sec=int(self.interval) + 1
            ))
        for i, kp in enumerate(left_kps):
            self._target_markers.append(self._make_keypoint_marker(
                10 + i, kp, (0.0, 0.8, 1.0, 0.95), f"left_tgt_kp[{i}]",
                lifetime_sec=int(self.interval) + 1
            ))
        self._publish_ee_keypoints()

        # PoseArray
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

        self.get_logger().info(
            f"已更新目标位姿: 右臂 ({right_pose[0]:.2f},{right_pose[1]:.2f},{right_pose[2]:.2f}) "
            f"左臂 ({left_pose[0]:.2f},{left_pose[1]:.2f},{left_pose[2]:.2f})"
        )


def main():
    parser = argparse.ArgumentParser(
        description="发布左右臂随机目标位姿 Marker（RViz 可视化，无重力，每 6 秒更新）"
    )
    parser.add_argument("--interval", type=float, default=7.0, help="随机更新间隔（秒）")
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    parser.add_argument("--ee-hz", type=float, default=10.0, help="末端 link7 关键点发布频率 (Hz)")
    # 使用 parse_known_args() 忽略 ROS 2 标准参数（--ros-args, -r, --params-file 等）
    args, unknown = parser.parse_known_args()

    rclpy.init()
    node = ReachTargetMarkerNode(interval=args.interval, seed=args.seed, ee_publish_hz=args.ee_hz)
    print("=" * 60)
    print("reach 目标位姿 Marker 节点已启动")
    print("  Topic: /reach_target_markers (MarkerArray)")
    print("  Topic: /reach_targets (PoseArray)")
    print(f"  更新间隔: {args.interval}s")
    print("  RViz: Add -> MarkerArray -> Topic: /reach_target_markers")
    print("        Fixed Frame: world")
    print("  红球=右臂目标, 蓝球=左臂目标")
    print("  橙球=目标关键点, 深橙=右 link7 关键点, 浅蓝=左 link7 关键点")
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
