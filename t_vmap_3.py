import torch
import torch.nn as nn
from torch import vmap

class FixedRotorGroup_V1(nn.Module):
    def __init__(self, rotor_config, dt: float):
        super().__init__()
        force_constants = torch.as_tensor([1.0, 1.0, 1.0, 1.0])  # 简化
        self.num_rotors = len(force_constants)
        self.dt = dt

        # ✅ 参数用 Parameter
        self.KF = nn.Parameter(force_constants * 2.0)
        self.KM = nn.Parameter(force_constants * 0.1)

        # ✅ 状态用 buffer
        self.register_buffer('throttle', torch.zeros(self.num_rotors))

        # 常量
        self.time_up = 0.15
        self.time_down = 0.15

    @property
    def tau_up(self):
        return self.dt / self.time_up

    @property
    def tau_down(self):
        return self.dt / self.time_down

    def f_inv(self, x):
        return torch.sqrt(x)

    def f(self, x):
        return x.square()

    def forward(self, cmds: torch.Tensor):
        target_throttle = self.f_inv(torch.clamp((cmds + 1) / 2, 0, 1))

        tau = torch.where(target_throttle > self.throttle, self.tau_up, self.tau_down)
        tau = torch.clamp(tau, 0, 1)

        # ✅ 修改 buffer 而不是 parameter
        self.throttle.add_(tau * (target_throttle - self.throttle))

        t = torch.clamp(self.f(self.throttle), 0., 1.)
        thrusts = t * self.KF
        moments = t * self.KM

        return thrusts, moments


def fixed_v1():
    print("\n" + "=" * 60)
    print("修正方案1: 使用 register_buffer")
    print("=" * 60)

    rotor = FixedRotorGroup_V1({'force_constants': [1, 1, 1, 1]}, dt=0.01)

    print("单次调用:")
    cmd = torch.tensor([0.1, 0.2, 0.3, 0.4])
    thrust, moment = rotor(cmd)
    print(f"✅ 推力: {thrust}")

    print("\nvmap 测试:")
    try:
        batch_cmds = torch.tensor([
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8]
        ])
        vmap_rotor = vmap(rotor)
        batch_thrust, batch_moment = vmap_rotor(batch_cmds)
        print(f"❌ 仍然有问题，throttle变成: {type(rotor.throttle)}")
    except Exception as e:
        print(f"❌ vmap 仍然失败: {e}")


fixed_v1()
