# OmniDrones

## Background Information

The original repository link is: [https://github.com/btx0424/OmniDrones](https://github.com/btx0424/OmniDrones)

The original repository supported Isaac Sim versions 2023 and 4.1.

In this repository, I have migrated it to Isaac Sim 4.5. It is recommended to use Ubuntu 22.04, as it simplifies the Isaac Sim installation process.

## Installation Method

Refer to the official `pip install` method: [https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html)

## Result
![train](docs/source/_static/train.mp4)

## Unresolved Issues

When running the following command:

```python
python examples/00_play_drones.py
```

The drones render correctly, but they are unable to fly.

![drones unable to fly](docs/source/_static/bug/bug1.jpg)

I have performed a low-level modification by relocating `self.throttle` within `omni_drones/robots/drone/multirotor.py` and `omni_drones/actuators/rotor_group.py`. 
Despite setting self.throttle to torch.tensor([[[1, 1, 1, 1]]]), the drone fails to achieve flight. I am currently uncertain of the root cause.

![source1](docs/source/_static/bug/bug2.jpg)
![source2](docs/source/_static/bug/bug3.jpg)