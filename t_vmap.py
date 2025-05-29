import torch
from torch import vmap
import torch.nn as nn


# ============================================================================
# 1. 基础 vmap 概念
# ============================================================================

def basic_vmap_example():
    print("=" * 50)
    print("1. 基础 vmap 示例")
    print("=" * 50)

    # 普通函数
    def square(x):
        # 在vmap内部不能使用会触发.item()的打印
        # print(f"  函数接收到的输入形状: {x.shape}")  # 这会出错
        return x ** 2

    # 单个数据
    single_data = torch.tensor(3.0)
    print(f"单个数据: {single_data}, 形状: {single_data.shape}")
    result = square(single_data)
    print(f"结果: {result}\n")

    # 批量数据 - 不使用vmap（手动循环）
    batch_data = torch.tensor([1.0, 2.0, 3.0, 4.0])
    print(f"批量数据: {batch_data}, 形状: {batch_data.shape}")
    print("手动循环处理:")
    manual_results = []
    for i, x in enumerate(batch_data):
        print(f"  处理第{i}个元素: {x}")
        result = square(x)
        manual_results.append(result)
    manual_results = torch.stack(manual_results)
    print(f"手动结果: {manual_results}\n")

    # 批量数据 - 使用vmap
    print("使用vmap处理:")
    vmap_square = vmap(square)
    vmap_results = vmap_square(batch_data)
    print(f"vmap结果: {vmap_results}")
    print(f"结果相同: {torch.allclose(manual_results, vmap_results)}\n")


# ============================================================================
# 2. 双重 vmap 示例
# ============================================================================

def nested_vmap_example():
    print("=" * 50)
    print("2. 双重 vmap 示例")
    print("=" * 50)

    def process_single_item(x):
        # print(f"    最内层函数接收: {x.shape}")
        return x * 2

    # 创建 3D 数据: [批次, 组, 元素]
    data_3d = torch.tensor([
        [[1, 2], [3, 4], [5, 6]],  # 批次0: 3组，每组2个元素
        [[7, 8], [9, 10], [11, 12]]  # 批次1: 3组，每组2个元素
    ])
    print(f"3D数据形状: {data_3d.shape} [批次, 组, 元素]")
    print(f"数据内容:\n{data_3d}\n")

    # 双重vmap: 先对组维度，再对批次维度
    double_vmap_fn = vmap(vmap(process_single_item))

    print("双重vmap处理过程:")
    result = double_vmap_fn(data_3d)
    print(f"最终结果形状: {result.shape}")
    print(f"最终结果:\n{result}\n")


# ============================================================================
# 3. 模拟无人机电机控制
# ============================================================================

class SimpleRotor(nn.Module):
    """简单的电机模型"""

    def __init__(self):
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(2.0))
        # 注意: 这里有状态会导致vmap问题！
        self.register_buffer('last_output', torch.tensor(0.0))

    def forward(self, cmd):
        # print(f"      电机接收指令形状: {cmd.shape}, 值: {cmd}")
        thrust = cmd * self.gain
        self.last_output = thrust.clone()  # 这里更新状态
        return thrust


def rotor_vmap_example():
    print("=" * 50)
    print("3. 电机 vmap 示例")
    print("=" * 50)

    rotor = SimpleRotor()

    # 单个电机指令
    single_cmd = torch.tensor(0.5)
    print(f"单个电机指令: {single_cmd}")
    single_result = rotor(single_cmd)
    print(f"单个电机输出: {single_result}\n")

    # 一个无人机的多个电机 [4个电机]
    drone_cmds = torch.tensor([0.1, 0.2, 0.3, 0.4])
    print(f"一个无人机的电机指令: {drone_cmds}, 形状: {drone_cmds.shape}")

    # 使用vmap处理一个无人机的多个电机
    vmap_rotor = vmap(rotor)
    try:
        drone_result = vmap_rotor(drone_cmds)
        print(f"一个无人机输出: {drone_result}\n")
    except Exception as e:
        print(f"❌ vmap失败: {e}")
        print("原因: 模块有状态(last_output)，vmap无法正确处理\n")


# ============================================================================
# 4. 无状态版本的电机模型
# ============================================================================

def stateless_rotor(cmd, gain=2.0):
    """无状态的电机函数"""
    # print(f"      无状态电机接收: {cmd.shape}, 值: {cmd}")
    return cmd * gain


def stateless_rotor_example():
    print("=" * 50)
    print("4. 无状态电机示例")
    print("=" * 50)

    # 一个无人机的多个电机
    drone_cmds = torch.tensor([0.1, 0.2, 0.3, 0.4])
    print(f"一个无人机指令: {drone_cmds}, 形状: {drone_cmds.shape}")

    vmap_stateless_rotor = vmap(stateless_rotor)
    drone_result = vmap_stateless_rotor(drone_cmds)
    print(f"一个无人机输出: {drone_result}\n")

    # 多个无人机的多个电机 [无人机数, 电机数]
    multi_drone_cmds = torch.tensor([
        [0.1, 0.2, 0.3, 0.4],  # 无人机0
        [0.5, 0.6, 0.7, 0.8],  # 无人机1
        [0.9, 1.0, 1.1, 1.2]  # 无人机2
    ])
    print(f"多无人机指令: 形状 {multi_drone_cmds.shape}")
    print(f"内容:\n{multi_drone_cmds}")

    # 双重vmap: 先处理电机维度，再处理无人机维度
    double_vmap_rotor = vmap(vmap(stateless_rotor))
    multi_result = double_vmap_rotor(multi_drone_cmds)
    print(f"多无人机输出: 形状 {multi_result.shape}")
    print(f"输出:\n{multi_result}\n")


# ============================================================================
# 5. 模拟你的实际场景
# ============================================================================

def simulate_your_scenario():
    print("=" * 50)
    print("5. 模拟你的实际场景")
    print("=" * 50)

    # 你的数据格式: [环境数, 无人机数, 电机数]
    rotor_cmds = torch.tensor([[
        [-0.4150, -0.4150, -0.4150, -0.4150],  # 无人机0
        [-0.4150, -0.4150, -0.4150, -0.4150],  # 无人机1
        [-0.4150, -0.4150, -0.4150, -0.4150],  # 无人机2
        [-0.4150, -0.4150, -0.4150, -0.4150]  # 无人机3
    ]])

    print(f"你的数据格式: {rotor_cmds.shape} [环境, 无人机, 电机]")
    print(f"数据内容:\n{rotor_cmds}")

    def rotor_group_forward(cmds_single_drone):
        """模拟 RotorGroup.forward，处理单个无人机"""
        print(f"    RotorGroup接收: {cmds_single_drone.shape}")
        # 模拟你的计算逻辑
        target_throttle = torch.clamp((cmds_single_drone + 1) / 2, 0, 1)
        thrusts = target_throttle * 2.0  # 简化的推力计算
        moments = thrusts * 0.1  # 简化的力矩计算
        return thrusts, moments

    # 方式1: 正确的双重vmap
    print("\n方式1: 双重vmap")
    try:
        double_vmap_fn = vmap(vmap(rotor_group_forward))
        thrusts, moments = double_vmap_fn(rotor_cmds)
        print(f"✅ 成功! 推力形状: {thrusts.shape}, 力矩形状: {moments.shape}")
        print(f"推力:\n{thrusts}")
    except Exception as e:
        print(f"❌ 失败: {e}")

    # 方式2: 手动循环验证
    print("\n方式2: 手动循环验证")
    manual_thrusts = []
    manual_moments = []

    for env_idx in range(rotor_cmds.shape[0]):  # 环境循环
        env_thrusts = []
        env_moments = []
        for drone_idx in range(rotor_cmds.shape[1]):  # 无人机循环
            drone_cmds = rotor_cmds[env_idx, drone_idx]
            print(f"  处理环境{env_idx}无人机{drone_idx}:")
            thrust, moment = rotor_group_forward(drone_cmds)
            env_thrusts.append(thrust)
            env_moments.append(moment)
        manual_thrusts.append(torch.stack(env_thrusts))
        manual_moments.append(torch.stack(env_moments))

    manual_thrusts = torch.stack(manual_thrusts)
    manual_moments = torch.stack(manual_moments)
    print(f"手动结果 - 推力形状: {manual_thrusts.shape}")


# ============================================================================
# 6. BatchedTensor 问题演示
# ============================================================================

class ProblematicRotor(nn.Module):
    """有问题的电机模型 - 会产生BatchedTensor问题"""

    def __init__(self):
        super().__init__()
        self.register_buffer('throttle', torch.zeros(4))  # 固定形状状态

    def forward(self, cmds):
        print(f"      ProblematicRotor接收: {cmds}")
        print(f"      类型: {type(cmds)}")
        if hasattr(cmds, 'shape'):
            print(f"      形状: {cmds.shape}")

        # 这里会出现维度不匹配问题
        try:
            result = cmds + self.throttle  # 维度不匹配！
            return result
        except Exception as e:
            print(f"      ❌ 计算失败: {e}")
            return cmds


def batched_tensor_problem():
    print("=" * 50)
    print("6. BatchedTensor 问题演示")
    print("=" * 50)

    problematic_rotor = ProblematicRotor()

    # 这会产生你遇到的问题
    rotor_cmds = torch.tensor([
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8]
    ])

    print(f"输入数据: {rotor_cmds.shape}")

    vmap_problematic = vmap(problematic_rotor)
    try:
        result = vmap_problematic(rotor_cmds)
        print(f"结果: {result}")
    except Exception as e:
        print(f"❌ vmap失败: {e}")


# ============================================================================
# 主函数
# ============================================================================

def main():
    print("🚁 vmap 无人机控制学习教程")
    print("本教程将帮助你理解vmap在多无人机控制中的应用\n")

    # 运行所有示例
    basic_vmap_example()
    nested_vmap_example()
    rotor_vmap_example()
    stateless_rotor_example()
    simulate_your_scenario()
    batched_tensor_problem()

    print("=" * 50)
    print("总结:")
    print("1. vmap 会逐层'剥离'张量的维度")
    print("2. 双重vmap(vmap(fn)) 会剥离前两个维度")
    print("3. 有状态的nn.Module不适合vmap，建议使用无状态函数")
    print("4. BatchedTensor是vmap的中间表示，表示还有维度未处理")
    print("5. 你的问题是RotorGroup有状态，导致vmap无法正确处理")
    print("=" * 50)


if __name__ == "__main__":
    main()
