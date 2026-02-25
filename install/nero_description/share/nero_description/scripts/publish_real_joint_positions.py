#!/usr/bin/env python3
# Copyright 2025. 直接控制实机机械臂：发布关节位置和运动速度
# 用法:
#   单次发布: ros2 run nero_description publish_real_joint_positions --can-left can_left [left1 ... left7]
#   单次发布（右臂）: ros2 run nero_description publish_real_joint_positions --can-right can_right [right1 ... right7]
#   循环发布: ros2 run nero_description publish_real_joint_positions --can-left can_left --loop [left1 ... left7]
#   设置速度: ros2 run nero_description publish_real_joint_positions --can-left can_left --speed 50 [left1 ... left7]

import argparse
import sys
import time

try:
    from pyAgxArm import create_agx_arm_config, AgxArmFactory
except ImportError as e:
    print(f"需要 pyAgxArm 库: {e}")
    sys.exit(1)

# 默认关节位置（单位：rad）
# [底座， 大俯仰， 大臂旋转， 小臂俯仰， 小臂旋转， 末端云台roll， 云台pitch]
DEFAULT_LEFT_POSITIONS = [1.6, 1.2, 0.52, 0.52, -0.6, 0.0, 0.0]
DEFAULT_RIGHT_POSITIONS = [-1.6, 1.2, -0.52, 0.52, 0.6, 0.0, 0.0]


def wait_motion_done(robot, timeout_s: float = 5.0) -> bool:
    """等待运动结束，超时返回 False。"""
    time.sleep(0.5)
    start_t = time.monotonic()
    while True:
        status = robot.get_arm_status()
        if status is not None and status.msg.motion_status == 0:
            print("已到达目标位置")
            return True
        if time.monotonic() - start_t > timeout_s:
            print(f"等待运动结束超时（{timeout_s:.0f}s）")
            return False
        time.sleep(0.1)


def get_current_velocities(robot) -> list[float]:
    """获取当前关节速度（使用 get_motor_states）"""
    velocities = []
    for joint_idx in range(1, 8):  # 关节 1-7
        ms = robot.get_motor_states(joint_idx)
        if ms is not None:
            velocities.append(float(ms.msg.motor_speed))
        else:
            velocities.append(0.0)
    return velocities


def print_status(robot, arm_name: str, target_pos: list = None):
    """打印当前状态（位置和速度）
    
    Args:
        robot: 机器人实例
        arm_name: 臂名称（"左" 或 "右"）
        target_pos: 目标位置（可选），用于对比实际位置和目标位置
    """
    js = robot.get_joint_states()
    if js is None or len(js.msg) < 7:
        print(f"{arm_name} 臂: 无法获取位置")
        return
    
    pos = list(js.msg)
    vel = get_current_velocities(robot)
    
    pos_str = " ".join(f"{p:7.4f}" for p in pos)
    vel_str = " ".join(f"{v:7.4f}" for v in vel)
    print(f"\n{arm_name} 臂状态:")
    print(f"  位置 (rad): {pos_str}")
    print(f"  速度 (rad/s): {vel_str}")
    
    # 如果有目标位置，显示误差
    if target_pos is not None and len(target_pos) == 7:
        errors = [abs(p - t) for p, t in zip(pos, target_pos)]
        max_error = max(errors)
        error_str = " ".join(f"{e:7.4f}" for e in errors)
        print(f"  目标位置: {' '.join(f'{t:7.4f}' for t in target_pos)}")
        print(f"  位置误差: {error_str} (最大: {max_error:.4f} rad)")
        if max_error > 0.01:  # 如果误差大于 0.01 rad，提示可能还在运动中
            print(f"  注意: 位置误差较大，可能还在运动中或未到达目标位置")


def main():
    parser = argparse.ArgumentParser(
        description="直接控制实机机械臂：发布关节位置和运动速度（参照 joint_control.py）"
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
        "--speed",
        type=float,
        default=30.0,
        help="运动速度百分比（默认: 100.0）",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="循环发送目标位置",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="循环发送间隔（秒，默认: 1.0）",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="等待运动完成的超时时间（秒，默认: 5.0）",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="不等待运动完成",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="只显示当前状态，不发送命令",
    )
    parser.add_argument(
        "positions",
        nargs="*",
        type=float,
        help="关节位置 (rad): 7 个关节角度，如果不提供则使用默认位置 [0,0,0,0,0,0,0]",
    )
    
    # 使用 parse_known_args() 来忽略 ROS 2 launch 系统自动添加的参数
    args, unknown = parser.parse_known_args()
    
    # 处理参数：如果指定了 --no-right，则不连接右臂
    can_left = args.can_left if args.can_left else None
    can_right = None if args.no_right else (args.can_right if args.can_right else None)
    
    if not can_left and not can_right:
        print("错误: 必须指定 --can-left 或 --can-right，或使用默认值")
        return 1
    
    # 初始化机器人（支持同时连接左臂和右臂）
    robot_left = None
    robot_right = None
    
    if can_left:
        print(f"初始化左臂机器人 (CAN: {can_left})...")
        try:
            cfg = create_agx_arm_config(robot="nero", channel=can_left)
            robot_left = AgxArmFactory.create_arm(cfg)
            robot_left.connect()
            while not robot_left.enable():
                time.sleep(0.01)
            robot_left.set_motion_mode(robot_left.MOTION_MODE.J)
            robot_left.set_speed_percent(int(args.speed))  # 转换为整数
            print("左臂机器人连接成功")
        except Exception as e:
            print(f"左臂机器人连接失败: {e}")
            robot_left = None
    
    if can_right:
        print(f"初始化右臂机器人 (CAN: {can_right})...")
        try:
            cfg = create_agx_arm_config(robot="nero", channel=can_right)
            robot_right = AgxArmFactory.create_arm(cfg)
            robot_right.connect()
            while not robot_right.enable():
                time.sleep(0.01)
            robot_right.set_motion_mode(robot_right.MOTION_MODE.J)
            robot_right.set_speed_percent(int(args.speed))  # 转换为整数
            print("右臂机器人连接成功")
        except Exception as e:
            print(f"右臂机器人连接失败: {e}")
            robot_right = None
    
    if robot_left is None and robot_right is None:
        print("错误: 无法连接任何机器人")
        return 2
    
    print(f"运动速度已设置为: {args.speed}%")
    
    try:
        # 如果只是查看状态
        if args.status:
            if robot_left:
                print_status(robot_left, "左", DEFAULT_LEFT_POSITIONS)
            if robot_right:
                print_status(robot_right, "右", DEFAULT_RIGHT_POSITIONS)
            return 0
        
        # 处理目标位置
        # 如果同时连接了左臂和右臂，支持14个关节角度（左7+右7）
        both_arms = robot_left is not None and robot_right is not None
        
        if len(args.positions) == 0:
            # 没有提供位置，使用默认位置
            if both_arms:
                targets = [[DEFAULT_LEFT_POSITIONS, DEFAULT_RIGHT_POSITIONS]]  # [左臂目标, 右臂目标]
                print(f"使用默认位置:")
                print(f"  左臂目标: {DEFAULT_LEFT_POSITIONS}")
                print(f"  右臂目标: {DEFAULT_RIGHT_POSITIONS}")
            else:
                # 单臂模式：根据连接的臂使用对应的默认位置
                if robot_left:
                    targets = [[DEFAULT_LEFT_POSITIONS]]  # 左臂默认位置
                    print(f"使用左臂默认位置: {DEFAULT_LEFT_POSITIONS}")
                else:
                    targets = [[DEFAULT_RIGHT_POSITIONS]]  # 右臂默认位置
                    print(f"使用右臂默认位置: {DEFAULT_RIGHT_POSITIONS}")
        elif len(args.positions) == 7:
            # 提供了 7 个位置（单臂）
            if both_arms:
                print("注意: 同时连接了左臂和右臂，但只提供了7个关节角度，将同时发送到两个臂")
                targets = [[args.positions, args.positions]]
            else:
                targets = [[args.positions]]
        elif len(args.positions) == 14:
            # 提供了 14 个位置（左7+右7）
            if both_arms:
                left_pos = args.positions[:7]
                right_pos = args.positions[7:14]
                targets = [[left_pos, right_pos]]
            else:
                print("错误: 提供了14个关节角度，但只连接了一个臂")
                return 1
        elif len(args.positions) % 7 == 0:
            # 提供了多个 7 的倍数
            if both_arms:
                if len(args.positions) % 14 == 0:
                    # 14的倍数，每14个为一组（左7+右7）
                    targets = []
                    for i in range(0, len(args.positions), 14):
                        left_pos = args.positions[i:i+7]
                        right_pos = args.positions[i+7:i+14]
                        targets.append([left_pos, right_pos])
                else:
                    # 7的倍数但不是14的倍数，每7个同时发送到两个臂
                    targets = []
                    for i in range(0, len(args.positions), 7):
                        pos = args.positions[i:i+7]
                        targets.append([pos, pos])
            else:
                # 单臂，每7个为一组
                targets = [[args.positions[i:i+7]] for i in range(0, len(args.positions), 7)]
        else:
            print(f"错误: 需要 7 个关节角度（单臂）或 14 个关节角度（双臂），收到 {len(args.positions)} 个")
            return 1
        
        # 发送命令
        if args.loop:
            print("开始循环发送关节目标，按 Ctrl+C 退出")
            try:
                while True:
                    for target in targets:
                        if both_arms:
                            # 同时发送到两个臂
                            left_pos, right_pos = target
                            print(f"发送左臂目标: {left_pos}")
                            print(f"发送右臂目标: {right_pos}")
                            robot_left.move_j(left_pos)
                            robot_right.move_j(right_pos)
                            if not args.no_wait:
                                # 等待两个臂都完成
                                wait_motion_done(robot_left, timeout_s=args.timeout)
                                wait_motion_done(robot_right, timeout_s=args.timeout)
                            print_status(robot_left, "左", left_pos)
                            print_status(robot_right, "右", right_pos)
                        else:
                            # 单臂发送
                            pos = target[0]
                            robot = robot_left if robot_left else robot_right
                            arm_name = "左" if robot_left else "右"
                            print(f"发送{arm_name}臂目标: {pos}")
                            robot.move_j(pos)
                            if not args.no_wait:
                                wait_motion_done(robot, timeout_s=args.timeout)
                            print_status(robot, arm_name, pos)
                        time.sleep(args.interval)
            except KeyboardInterrupt:
                print("\n已退出循环")
        else:
            # 单次发送
            for target in targets:
                if both_arms:
                    # 同时发送到两个臂
                    left_pos, right_pos = target
                    print(f"发送左臂目标: {left_pos}")
                    print(f"发送右臂目标: {right_pos}")
                    robot_left.move_j(left_pos)
                    robot_right.move_j(right_pos)
                    if not args.no_wait:
                        wait_motion_done(robot_left, timeout_s=args.timeout)
                        wait_motion_done(robot_right, timeout_s=args.timeout)
                    print_status(robot_left, "左", left_pos)
                    print_status(robot_right, "右", right_pos)
                else:
                    # 单臂发送
                    pos = target[0]
                    robot = robot_left if robot_left else robot_right
                    arm_name = "左" if robot_left else "右"
                    print(f"发送{arm_name}臂目标: {pos}")
                    robot.move_j(pos)
                    if not args.no_wait:
                        wait_motion_done(robot, timeout_s=args.timeout)
                    print_status(robot, arm_name, pos)
    
    finally:
        # 不调用 disable()，保持机械臂使能状态
        
        pass
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
