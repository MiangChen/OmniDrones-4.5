import torch
import torch.nn as nn
from torch import vmap


# 演示你的问题
class ProblematicRotorGroup(nn.Module):
    def __init__(self, num_rotors=4):
        super().__init__()
        self.KF = nn.Parameter(torch.ones(num_rotors) * 2.0)
        # ❌ throttle 不应该是 Parameter!
        self.throttle = nn.Parameter(torch.zeros(num_rotors), requires_grad=False)

    def forward(self, cmds):
        print(f"调用前 throttle: {self.throttle.data}")

        # ❌ 在 forward 中修改 Parameter
        target = cmds * 0.1
        tau = 0.1
        self.throttle.add_(tau * (target - self.throttle))

        print(f"调用后 throttle: {self.throttle.data}")

        thrust = self.throttle * self.KF
        return thrust


def demonstrate_problems():
    print("=" * 60)
    print("演示你的代码问题")
    print("=" * 60)

    rotor = ProblematicRotorGroup()

    print("问题1: Parameter 语义错误")
    print("nn.Parameter 应该用于:")
    print("  - 需要梯度更新的模型参数")
    print("  - 通过优化器训练的权重")
    print("throttle 应该是状态，不是参数!")

    print("\n问题2: 单次调用看起来正常")
    cmd1 = torch.tensor([0.1, 0.2, 0.3, 0.4])
    result1 = rotor(cmd1)
    print(f"结果: {result1}")

    print("\n问题3: vmap 会出错")
    batch_cmds = torch.tensor([
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8]
    ])

    try:
        vmap_rotor = vmap(rotor)
        batch_results = vmap_rotor(batch_cmds)
        print(f"vmap 结果: {batch_results}")
        print(f"throttle 类型: {type(rotor.throttle)}")
    except Exception as e:
        print(f"❌ vmap 失败: {e}")


demonstrate_problems()
