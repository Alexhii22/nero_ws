#!/usr/bin/env python3
# Copyright 2025. 从实机读取关节状态并发布到 ROS 2 /joint_states topic
# 用法: ros2 run nero_description gain_real_joint_positions [--can-left CAN_LEFT] [--can-right CAN_RIGHT]

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


class GainRealJointPositionsNode(Node):
    """从实机读取关节状态并发布到 ROS 2 /joint_states topic"""

    def __init__(self, can_left: str = None, can_right: str = None, publish_hz: float = 50.0):
        super().__init__("gain_real_joint_positions")
        
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
            error_msg = "没有可用的机器人实例，请检查 CAN 通道配置"
            self.get_logger().error(error_msg)
            raise RuntimeError(error_msg)
        
        # 创建发布者
        self.pub = self.create_publisher(JointState, "/joint_states", 10)
        
        # 创建定时器
        self.timer = self.create_timer(self.publish_dt, self.publish_joint_states)
        
        # 用于终端显示速度的计数器
        self.print_counter = 0
        self.print_interval = int(publish_hz / 30.0)  # 每 0.5 秒打印一次（默认 2 Hz）
        
        self.get_logger().info(
            f"实机关节状态发布节点已启动 (发布频率: {publish_hz} Hz)"
        )
        if self.robot_left:
            self.get_logger().info("  - 左臂: 已连接")
        if self.robot_right:
            self.get_logger().info("  - 右臂: 已连接")
        self.get_logger().info("  终端将每 0.5 秒显示一次关节速度")
    
    def publish_joint_states(self):
        """读取实机关节状态并发布（使用 get_motor_states 获取速度）"""
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(JOINT_NAMES)
        msg.position = [0.0] * len(JOINT_NAMES)
        msg.velocity = [0.0] * len(JOINT_NAMES)
        msg.effort = []
        
        # 读取左臂关节状态和速度
        if self.robot_left:
            js_left = self.robot_left.get_joint_states()
            if js_left is not None and len(js_left.msg) >= 7:
                for i in range(7):
                    msg.position[i] = float(js_left.msg[i])
            
            # 使用 get_motor_states 读取每个关节的电机速度
            for joint_idx in range(1, 8):  # 关节 1-7
                ms = self.robot_left.get_motor_states(joint_idx)
                if ms is not None:
                    msg.velocity[joint_idx - 1] = float(ms.msg.motor_speed)
        
        # 读取右臂关节状态和速度
        if self.robot_right:
            js_right = self.robot_right.get_joint_states()
            if js_right is not None and len(js_right.msg) >= 7:
                for i in range(7):
                    msg.position[i + 7] = float(js_right.msg[i])
            
            # 使用 get_motor_states 读取每个关节的电机速度
            for joint_idx in range(1, 8):  # 关节 1-7
                ms = self.robot_right.get_motor_states(joint_idx)
                if ms is not None:
                    msg.velocity[joint_idx - 1 + 7] = float(ms.msg.motor_speed)
        
        # 发布消息
        self.pub.publish(msg)
        
        # 终端实时显示速度（每 print_interval 次打印一次）
        self.print_counter += 1
        if self.print_counter >= self.print_interval:
            self.print_counter = 0
            self._print_velocities(msg)
    
    def _print_velocities(self, msg: JointState):
        """在终端打印关节速度"""
        left_vel = [msg.velocity[i] for i in range(7)]
        right_vel = [msg.velocity[i + 7] for i in range(7)]
        
        # 格式化输出
        left_str = " ".join(f"{v:7.4f}" for v in left_vel)
        right_str = " ".join(f"{v:7.4f}" for v in right_vel)
        
        print("\n" + "=" * 80)
        print("关节速度 (rad/s):")
        print(f"  左臂: {left_str}")
        print(f"  右臂: {right_str}")
        print("=" * 80)


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
    
    # 使用 parse_known_args() 来忽略 ROS 2 launch 系统自动添加的参数（如 --ros-args）
    args, unknown = parser.parse_known_args()
    if unknown:
        # 可以记录未知参数，但不影响程序运行
        pass
    
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
    
    try:
        rclpy.init()
    except Exception as e:
        print(f"ROS 2 初始化失败: {e}", file=sys.stderr)
        return 1
    
    try:
        node = GainRealJointPositionsNode(
            can_left=can_left,
            can_right=can_right,
            publish_hz=args.hz,
        )
    except Exception as e:
        print(f"节点创建失败: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        rclpy.shutdown()
        return 2
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        node.get_logger().error(f"运行时错误: {e}")
        import traceback
        traceback.print_exc()
        return 2
    finally:
        node.get_logger().info("退出中...")
        # 不调用 disable()，保持机械臂使能状态
        pass
        node.destroy_node()
        rclpy.shutdown()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

