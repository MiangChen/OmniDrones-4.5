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
from typing import Sequence

class GAE(nn.Module):
    def __init__(self, gamma, lmbda):
        super().__init__()
        self.register_buffer("gamma", torch.tensor(gamma))
        self.register_buffer("lmbda", torch.tensor(lmbda))
        self.gamma: torch.Tensor
        self.lmbda: torch.Tensor

    def forward(
        self,
        reward: torch.Tensor,
        terminated: torch.Tensor,
        value: torch.Tensor,
        next_value: torch.Tensor
    ):
        num_steps = terminated.shape[1]
        # todo
        # advantages = torch.zeros_like(reward)  # torch.Size([128, 32, 1, 1])
        advantages = torch.zeros_like(value)  # torch.Size([128, 32, 1, 1])
        not_done = 1 - terminated.float()
        gae = 0
        for step in reversed(range(num_steps)): # num step = 32
            delta = (
                reward[:, step]  # torch.Size([128, 32, 1, 1]) -> torch.Size([128, 1, 1])
                + self.gamma * next_value[:, step] * not_done[:, step]   # torch.Size([128, 32, 1, 128]) -> torch.Size([128, 1, 128])
                - value[:, step]  # torch.Size([128, 32, 1, 128]) -> torch.Size([128, 1, 128])
            )  # torch.Size([128, 1, 128])

            # origin torch.Size([128, 1, 1]) = torch.Size([128, 1, 128])
            # now torch.Size([128, 1, 128]) = torch.Size([128, 1, 128])
            advantages[:, step] = gae = delta + (self.gamma * self.lmbda * not_done[:, step] * gae)
        returns = advantages + value
        return advantages, returns


def make_mlp(num_units: Sequence[int,], activation=nn.LeakyReLU):
    layers = []
    for n in num_units:
        layers.append(nn.LazyLinear(n))
        layers.append(activation())
        layers.append(nn.LayerNorm(n))
    return nn.Sequential(*layers)

