from setuptools import find_packages, setup

setup(
    name="omni_drones",
    version="0.1.1",
    author="btx0424@SUSTech",
    keywords=["robotics", "rl"],
    packages=find_packages("."),
    install_requires=[
        "hydra-core",
        "omegaconf",
        "wandb",
        "imageio",
        "plotly",
        "einops",
        "pandas",
        "moviepy",
        "av==13.1.0",  # av 14.0.0 doesn't work for torch 2.5.0
        # "torchrl==0.3.1", # for torch==2.2.2
        # TorchRL 0.5 requires PyTorch 2.4.0, and TorchRL 0.6 requires PyTorch 2.5.0.
        # omni drone's version requirement is 0.3.1 which is for torch 2.2.x,  and the newest version is 0.8.1
        "torchrl==0.6.0",  # for torch==2.5.0
        "tensordict==0.6.0"  # for torchrl 0.6.0

    ],
)
