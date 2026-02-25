#!/usr/bin/env python3
# Copyright 2025. 从实机读取关节状态并发布到 ROS 2 /joint_states topic
# 用法: ros2 run nero_description publish_real_joint_states [--can-left CAN_LEFT] [--can-right CAN_RIGHT]

import argparse
import sys
import time

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
except ImportError as e:
    print(f"需要 ROS2 环境与依赖: {e}")
    sys.exit(1)

try:
    from pyAgxArm import create_agx_arm_config, AgxArmFactory
except ImportError as e:
    print(f"需要 pyAgxArm 库: {e}")
    sys.exit(1)

# 关节名称顺序（与 URDF 一致）
JOINT_NAMES = [
    "left_joint1", "left_joint2", "left_joint3", "left_joint4",
    "left_joint5", "left_joint6", "left_joint7",
    "right_joint1", "right_joint2", "right_joint3", "right_joint4",
    "right_joint5", "right_joint6", "right_joint7",
]


class RealJointStatesPublisher(Node):
    """从实机读取关节状态并发布到 ROS 2 /joint_states topic"""

    def __init__(self, can_left: str = None, can_right: str = None, publish_hz: float = 50.0):
        super().__init__("real_joint_states_publisher")
        
        self.publish_hz = publish_hz
        self.publish_dt = 1.0 / publish_hz
        
        # 初始化机器人实例
        self.robot_left = None
        self.robot_right = None
        
        if can_left:
            self.get_logger().info(f"初始化左臂机器人 (CAN: {can_left})...")
            try:
                cfg_left = create_agx_arm_config(robot="nero", channel=can_left)
                self.robot_left = AgxArmFactory.create_arm(cfg_left)
                self.robot_left.connect()
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
                self.get_logger().info("右臂机器人连接成功")
            except Exception as e:
                self.get_logger().error(f"右臂机器人连接失败: {e}")
                self.robot_right = None
        
        if not self.robot_left and not self.robot_right:
            self.get_logger().error("没有可用的机器人实例，退出")
            sys.exit(1)
        
        # 创建发布者
        self.pub = self.create_publisher(JointState, "/joint_states", 10)
        
        # 创建定时器
        self.timer = self.create_timer(self.publish_dt, self.publish_joint_states)
        
        # 存储上一时刻的位置，用于计算速度
        self.prev_positions = {name: 0.0 for name in JOINT_NAMES}
        self.prev_time = None
        
        self.get_logger().info(
            f"实机关节状态发布节点已启动 (发布频率: {publish_hz} Hz)"
        )
        if self.robot_left:
            self.get_logger().info("  - 左臂: 已连接")
        if self.robot_right:
            self.get_logger().info("  - 右臂: 已连接")
    
    def publish_joint_states(self):
        """读取实机关节状态并发布"""
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(JOINT_NAMES)
        msg.position = [0.0] * len(JOINT_NAMES)
        msg.velocity = [0.0] * len(JOINT_NAMES)
        msg.effort = []
        
        # 读取左臂关节状态
        if self.robot_left:
            js_left = self.robot_left.get_joint_states()
            if js_left is not None and len(js_left.msg) >= 7:
                for i in range(7):
                    msg.position[i] = float(js_left.msg[i])
        
        # 读取右臂关节状态
        if self.robot_right:
            js_right = self.robot_right.get_joint_states()
            if js_right is not None and len(js_right.msg) >= 7:
                for i in range(7):
                    msg.position[i + 7] = float(js_right.msg[i])
        
        # 计算速度（差分）
        now = self.get_clock().now()
        if self.prev_time is not None:
            dt = (now - self.prev_time).nanoseconds * 1e-9
            if dt > 0:
                for i, name in enumerate(JOINT_NAMES):
                    if msg.position[i] != 0.0 or self.prev_positions[name] != 0.0:
                        msg.velocity[i] = (msg.position[i] - self.prev_positions[name]) / dt
                        # 限幅速度
                        msg.velocity[i] = max(-10.0, min(10.0, msg.velocity[i]))
        
        # 更新历史数据
        for i, name in enumerate(JOINT_NAMES):
            self.prev_positions[name] = msg.position[i]
        self.prev_time = now
        
        # 发布消息
        self.pub.publish(msg)


def main():
    parser = argparse.ArgumentParser(
        description="从实机读取关节状态并发布到 ROS 2 /joint_states topic"
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
        help="右臂 CAN 通道名称（默认: can_right，连接右臂）",
    )
    parser.add_argument(
        "--hz",
        type=float,
        default=50.0,
        help="发布频率 (Hz，默认: 50.0)",
    )
    parser.add_argument(
        "--no-left",
        action="store_true",
        help="不连接左臂",
    )
    parser.add_argument(
        "--no-right",
        action="store_true",
        help="不连接右臂（使用此选项时默认不连接右臂）",
    )
    
    args = parser.parse_args()
    
    # 处理参数：如果指定了 --no-left 或 --no-right，则设为 None
    # 如果 can_right 是空字符串，也设为 None
    can_left = None if args.no_left else (args.can_left if args.can_left else None)
    # 默认连接右臂（can_right="can_right"），除非指定了 --no-right 或 can_right 为空字符串
    if args.no_right:
        can_right = None  # 明确指定不连接右臂
    elif args.can_right and args.can_right.strip():
        can_right = args.can_right.strip()  # 使用提供的 CAN 通道
    else:
        can_right = None  # 空字符串或 None，不连接右臂
    
    rclpy.init()
    node = RealJointStatesPublisher(
        can_left=can_left,
        can_right=can_right,
        publish_hz=args.hz,
    )
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    
    node.get_logger().info("退出中...")
    if node.robot_left:
        try:
            node.robot_left.disable()
        except:
            pass
    if node.robot_right:
        try:
            node.robot_right.disable()
        except:
            pass
    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())

