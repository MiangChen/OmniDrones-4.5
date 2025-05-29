# MIT License
#
# Copyright (c) 2023 Botian Xu, Tsinghua University
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
import torch.nn as nn


class RotorGroup(nn.Module):
    def __init__(self, rotor_config, dt: float):
        super().__init__()
        force_constants = torch.as_tensor(rotor_config["force_constants"])
        moment_constants = torch.as_tensor(rotor_config["moment_constants"])
        max_rot_vels = torch.as_tensor(rotor_config["max_rotation_velocities"]).float()
        self.num_rotors = len(force_constants)

        self.dt = dt
        self.time_up = 0.15
        self.time_down = 0.15
        self.noise_scale = 0.002

        self.KF = nn.Parameter(max_rot_vels.square() * force_constants)
        self.KM = nn.Parameter(max_rot_vels.square() * moment_constants)
        self.throttle = nn.Parameter(torch.zeros(self.num_rotors))  # 这里有问题, throttle是要能改变的量
        # self.register_buffer("throttle", torch.zeros(self.num_rotors))
        self.directions = nn.Parameter(torch.as_tensor(rotor_config["directions"]).float())

        self.tau_up = nn.Parameter(0.43 * torch.ones(self.num_rotors))
        self.tau_down = nn.Parameter(0.43 * torch.ones(self.num_rotors))

        self.f = torch.square
        self.f_inv = torch.sqrt

        self.requires_grad_(False)

    def forward(self, cmds: torch.Tensor, throttles: torch.Tensor):
        # 需要确定一下, 更改了rotor group中的throttle, 时候会影multirotor中的throttle,
        target_throttle = self.f_inv(torch.clamp((cmds + 1) / 2, 0, 1))  # 目标油门, 输入是-1~1, 归一到0~1中

        tau = torch.where(target_throttle > throttles, self.tau_up, self.tau_down)  # 响应时间,
        tau = torch.clamp(tau, 0, 1)
        throttles.add_(tau * (target_throttle - throttles))   # 根据响应时间计算油门的增加量, 一阶响应

        noise = torch.randn_like(throttles) * self.noise_scale * 0.  # 模拟噪音
        t = torch.clamp(self.f(throttles) + noise, 0., 1.)  #
        thrusts = t * self.KF  # t 是推力比例因子, 最大推力KF
        moments = (t * self.KM) * -self.directions  # 最大反作用力矩常数

        return thrusts, moments, throttles  # 返回推力\力矩\油门
