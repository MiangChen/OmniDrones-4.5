# MIT License
#
# Copyright (c) 2023 Botian Xu
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

import torch
import torch.distributions as D
from torch.func import vmap
from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import UnboundedContinuousTensorSpec
from torchrl.data import Composite as CompositeSpec

# todo
import isaacsim.core.utils.prims as prim_utils
from isaacsim.util.debug_draw import _debug_draw

import omni_drones.utils.kit as kit_utils
from omni_drones.envs.isaac_env import AgentSpec, IsaacEnv
from omni_drones.robots.drone import MultirotorBase
from omni_drones.utils.torch import euler_to_quaternion, normalize, quat_rotate
from omni_drones.views import RigidPrimView

from ..utils import lemniscate, scale_time


def attach_payload(parent_path):
    import omni.physx.scripts.utils as script_utils
    from isaacsim.core.api import objects
    from pxr import UsdPhysics

    payload_prim = objects.DynamicCuboid(
        prim_path=parent_path + "/payload",
        scale=torch.tensor([0.1, 0.1, 0.15]),
        mass=0.0001,
    ).prim

    parent_prim = prim_utils.get_prim_at_path(parent_path + "/base_link")
    stage = prim_utils.get_current_stage()
    joint = script_utils.createJoint(stage, "Prismatic", payload_prim, parent_prim)
    UsdPhysics.DriveAPI.Apply(joint, "linear")
    joint.GetAttribute("physics:lowerLimit").Set(-0.15)
    joint.GetAttribute("physics:upperLimit").Set(0.15)
    joint.GetAttribute("physics:axis").Set("Z")
    joint.GetAttribute("drive:linear:physics:damping").Set(10.0)
    joint.GetAttribute("drive:linear:physics:stiffness").Set(10000.0)


class Track(IsaacEnv):
    r"""
    A basic control task. The goal for the agent is to track a reference
    lemniscate trajectory in the 3D space.

    ## Observation

    - `rpos` (3 * `future_traj_steps`): The relative position of the drone to the
      reference positions in the future `future_traj_steps` time steps.
    - `root_state` (16 + `num_rotors`): The basic information of the drone (except its position),
      containing its rotation (in quaternion), velocities (linear and angular),
      heading and up vectors, and the current throttle.
    - `time_encoding` (optional): The time encoding, which is a 4-dimensional
      vector encoding the current progress of the episode.

    ## Reward

    - `pos`: Reward for tracking the trajectory, computed from the position
      error as {math}`\exp(-a * \text{pos_error})`.
    - `up`: Reward computed from the uprightness of the drone to discourage
      large tilting.
    - `spin`: Reward computed from the spin of the drone to discourage spinning.
    - `effort`: Reward computed from the effort of the drone to optimize the
      energy consumption.

    The total reward is computed as follows:

    ```{math}
        r = r_\text{pos} + r_\text{pos} * (r_\text{up} + r_\text{heading}) + r_\text{effort}
    ```

    ## Episode End

    The episode ends when the tracking error is larger than `reset_thres`, or
    when the drone is too close to the ground, or when the episode reaches
    the maximum length.

    ## Config

    | Parameter               | Type  | Default       | Description                                                                                                                                                                                                                             |
    | ----------------------- | ----- | ------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
    | `drone_model`           | str   | "hummingbird" | Specifies the model of the drone being used in the environment.                                                                                                                                                                         |
    | `reset_thres`           | float | 0.5           | Threshold for the distance between the drone and its target, upon exceeding which the episode will be reset.                                                                                                                            |
    | `future_traj_steps`     | int   | 4             | Number of future trajectory steps the drone needs to predict.                                                                                                                                                                           |
    | `reward_distance_scale` | float | 1.2           | Scales the reward based on the distance between the drone and its target.                                                                                                                                                               |
    | `time_encoding`         | bool  | True          | Indicates whether to include time encoding in the observation space. If set to True, a 4-dimensional vector encoding the current progress of the episode is included in the observation. If set to False, this feature is not included. |
    """

    def __init__(self, cfg, headless):
        self.reset_thres = cfg.task.reset_thres
        self.reward_effort_weight = cfg.task.reward_effort_weight
        self.reward_distance_scale = cfg.task.reward_distance_scale

        self.future_traj_steps = max(int(cfg.task.future_traj_steps), 2)
        self.wind = cfg.task.wind
        self.randomization = cfg.task.get("randomization", {})
        self.has_payload = "payload" in self.randomization.keys()
        # time at which the mass of the payload changes
        self.mass_change_interval = cfg.task.get("mass_change_interval", -1)

        super().__init__(cfg, headless)

        self.drone.initialize()
        self.intrinsics = self.drone.intrinsics

        if "drone" in self.randomization:
            self.drone.setup_randomization(self.randomization["drone"])
        if "payload" in self.randomization:
            payload_cfg = self.randomization["payload"]
            self.payload_z_dist = D.Uniform(
                torch.tensor([payload_cfg["z"][0]], device=self.device),
                torch.tensor([payload_cfg["z"][1]], device=self.device),
            )
            self.payload_mass_dist = D.Uniform(
                torch.tensor([payload_cfg["mass"][0]], device=self.device),
                torch.tensor([payload_cfg["mass"][1]], device=self.device),
            )
            self.payload = RigidPrimView(
                f"/World/envs/env_*/{self.drone.name}_*/payload",
                reset_xform_properties=False,
                shape=(-1, self.drone.n),
            )
            self.payload.initialize()

        self.traj_rpy_dist = D.Uniform(
            torch.tensor([0.0, 0.0, 0.0], device=self.device) * torch.pi,
            torch.tensor([0.0, 0.0, 2.0], device=self.device) * torch.pi,
        )
        self.traj_c_dist = D.Uniform(
            torch.tensor(-0.8, device=self.device),
            torch.tensor(0.8, device=self.device),
        )
        self.traj_scale_dist = D.Uniform(
            torch.tensor([2.4, 2.4, 1.2], device=self.device),
            torch.tensor([4.0, 4.0, 1.6], device=self.device),
        )
        self.traj_w_dist = D.Uniform(
            torch.tensor(0.8, device=self.device), torch.tensor(1.6, device=self.device)
        )
        with torch.device(self.device):
            self.origin = torch.tensor([0.0, 0.0, 2.5])

            self.traj_t0 = torch.pi / 2
            self.traj_c = torch.zeros(self.num_envs)
            self.traj_scale = torch.zeros(self.num_envs, 3)
            self.traj_rot = torch.zeros(self.num_envs, 4)
            self.traj_w = torch.ones(self.num_envs)

            self.ref_pos = torch.zeros(self.num_envs, self.future_traj_steps, 3)
            self.ref_heading = torch.zeros(self.num_envs, 2)
            self.prev_action = self.action_spec.zero()

        self.alpha = 0.8

        # debug visualizations
        self.draw = _debug_draw.acquire_debug_draw_interface()

    def _design_scene(self):
        drone_model_cfg = self.cfg.task.drone_model
        self.drone, self.controller = MultirotorBase.make(
            drone_model_cfg.name, drone_model_cfg.controller
        )

        kit_utils.create_ground_plane(
            "/World/defaultGroundPlane",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        )
        drone_prim = self.drone.spawn(translations=[(0.0, 0.0, 1.5)])[0]
        if self.has_payload:
            attach_payload(drone_prim.GetPath().pathString)
        return ["/World/defaultGroundPlane"]

    def _set_specs(self):
        drone_obs_dim = (
                self.drone.state_spec.shape[-1]
                + 3 * (self.future_traj_steps - 1)
                + 2  # reference xy heading
                + self.drone.action_spec.shape[-1]  # last action
        )

        intrinsics_spec = self.drone.intrinsics_spec.to(self.device)
        intrinsics_spec["payload_mass"] = UnboundedContinuousTensorSpec(
            1, device=self.device
        )
        intrinsics_spec["payload_z"] = UnboundedContinuousTensorSpec(
            1, device=self.device
        )
        # TODO@btx0424: observe history through a Transform
        self.observation_spec = CompositeSpec(
            {
                "agents": CompositeSpec(
                    {
                        "observation": UnboundedContinuousTensorSpec(
                            (1, drone_obs_dim), device=self.device
                        ),
                        "observation_h": UnboundedContinuousTensorSpec(
                            (1, drone_obs_dim, 32), device=self.device
                        ),
                        "intrinsics": intrinsics_spec.unsqueeze(0),
                    }
                ),
            }
        ).expand(self.num_envs)
        self.action_spec = (
            CompositeSpec(
                {
                    "agents": {
                        "action": self.drone.action_spec.unsqueeze(0),
                    }
                }
            )
            .expand(self.num_envs)
            .to(self.device)
        )
        self.reward_spec = (
            CompositeSpec({"agents": {"reward": UnboundedContinuousTensorSpec((1, 1))}})
            .expand(self.num_envs)
            .to(self.device)
        )
        self.agent_spec["drone"] = AgentSpec(
            "drone",
            1,
            observation_key=("agents", "observation"),
            action_key=("agents", "action"),
            reward_key=("agents", "reward"),
            state_key=("agents", "intrinsics"),
        )

        stats_spec = CompositeSpec({
            "return": UnboundedContinuousTensorSpec(1),
            "episode_len": UnboundedContinuousTensorSpec(1),
            "tracking_error": UnboundedContinuousTensorSpec(1),
            "heading_alignment": UnboundedContinuousTensorSpec(1),
            "success": UnboundedContinuousTensorSpec(1),
        }).expand(self.num_envs).to(self.device)
        self.observation_spec["stats"] = stats_spec
        self.stats = stats_spec.zero()

        self.observation_h = self.observation_spec[("agents", "observation_h")].zero()

    def _reset_idx(self, env_ids: torch.Tensor):
        self.drone._reset_idx(env_ids)
        self.traj_c[env_ids] = self.traj_c_dist.sample(env_ids.shape)
        self.traj_rot[env_ids] = euler_to_quaternion(
            self.traj_rpy_dist.sample(env_ids.shape)
        )
        self.traj_scale[env_ids] = self.traj_scale_dist.sample(env_ids.shape)
        # randomly flip the direction
        traj_w = self.traj_w_dist.sample(env_ids.shape)
        traj_w = traj_w * torch.randn_like(traj_w).sign()
        self.traj_w[env_ids] = traj_w

        t0 = torch.full((len(env_ids),), self.traj_t0, device=self.device)
        pos_0 = lemniscate(t0, self.traj_c[env_ids]) + self.origin
        pos_1 = lemniscate(t0 + self.dt * traj_w, self.traj_c[env_ids]) + self.origin
        traj_heading = normalize(pos_1 - pos_0)
        rpy = torch.stack(
            [
                torch.zeros(len(env_ids), device=self.device),
                torch.zeros(len(env_ids), device=self.device),
                torch.arctan2(traj_heading[:, 1], traj_heading[:, 0]),
            ],
            dim=-1,
        )
        rot = euler_to_quaternion(rpy).unsqueeze(1)
        vel = torch.zeros(len(env_ids), 1, 6, device=self.device)
        self.drone.set_world_poses(pos_0 + self.envs_positions[env_ids], rot, env_ids)
        self.drone.set_velocities(vel, env_ids)

        if self.has_payload:
            # TODO@btx0424: workout a better way
            payload_z = self.payload_z_dist.sample(env_ids.shape)
            joint_indices = torch.tensor(
                [self.drone._view._dof_indices["PrismaticJoint"]], device=self.device
            )
            self.drone._view.set_joint_positions(
                payload_z, env_indices=env_ids, joint_indices=joint_indices
            )
            self.drone._view.set_joint_position_targets(
                payload_z, env_indices=env_ids, joint_indices=joint_indices
            )

            self.drone._view.set_joint_velocities(
                torch.zeros(len(env_ids), 1, device=self.device),
                env_indices=env_ids,
                joint_indices=joint_indices,
            )

            payload_mass = self.payload_mass_dist.sample(env_ids.shape)
            self.payload.set_masses(
                payload_mass * self.drone.masses[env_ids].squeeze(1),
                env_indices=env_ids,
            )

            self.intrinsics["payload_mass"][env_ids] = payload_mass.unsqueeze(1)
            self.intrinsics["payload_z"][env_ids] = payload_z.unsqueeze(1)

        self.stats[env_ids] = 0.0
        self.observation_h[env_ids] = 0.0
        self.prev_action[env_ids] = 0.0

        if self._should_render(0) and (env_ids == self.central_env_idx).any():
            # visualize the trajectory
            self.draw.clear_lines()

            traj_vis = self._compute_traj(
                self.max_episode_length, self.central_env_idx.unsqueeze(0)
            )[0]
            traj_vis = traj_vis + self.envs_positions[self.central_env_idx]
            point_list_0 = traj_vis[:-1].tolist()
            point_list_1 = traj_vis[1:].tolist()
            colors = [(1.0, 1.0, 1.0, 1.0) for _ in range(len(point_list_0))]
            sizes = [1 for _ in range(len(point_list_0))]
            self.draw.draw_lines(point_list_0, point_list_1, colors, sizes)

    def _pre_sim_step(self, tensordict: TensorDictBase):
        actions = tensordict[("agents", "action")]
        self.effort = self.drone.apply_action(actions)
        self.prev_action[:] = actions

    def _post_sim_step(self, tensordict: TensorDictBase):
        if self.has_payload and self.mass_change_interval > 0:
            change_mass = (
                (((self.progress_buf + 1) % self.mass_change_interval) == 0)
                .nonzero()
                .squeeze(-1)
            )
            payload_mass = self.payload_mass_dist.sample(change_mass.shape)
            self.intrinsics["payload_mass"][change_mass] = payload_mass.unsqueeze(1)
            self.payload.set_masses(
                payload_mass * self.drone.masses[change_mass].squeeze(1), change_mass
            )

    def _compute_state_and_obs(self):
        self.root_state = self.drone.get_state()

        self.ref_pos[:] = self._compute_traj(self.future_traj_steps, step_size=5)
        self.ref_heading[:] = normalize(self.ref_pos[:, 1, :2] - self.ref_pos[:, 0, :2])

        self.rpos = self.ref_pos - self.root_state[..., :3]
        obs = [
            self.rpos.flatten(1).unsqueeze(1),
            self.ref_heading.unsqueeze(1) - normalize(self.drone.heading[..., :2]),
            self.root_state[..., 3:],
            self.prev_action
        ]

        obs = torch.cat(obs, dim=-1)
        self.observation_h[..., -1] = obs

        return TensorDict(
            {
                "agents": {
                    "observation": obs,
                    "observation_h": self.observation_h,
                    "intrinsics": self.intrinsics,
                },
                "stats": self.stats.clone(),
            },
            self.batch_size,
        )

    def _compute_reward_and_done(self):
        pos_error = torch.norm(self.rpos[:, [0]], dim=-1)
        heading_alignment = (
                self.ref_heading.unsqueeze(1) * normalize(self.drone.heading[..., :2])
        ).sum(-1)

        reward_pose = torch.exp(-self.reward_distance_scale * pos_error)
        reward_heading = heading_alignment

        reward_effort = self.reward_effort_weight * torch.exp(-self.effort)

        reward = (
                reward_pose
                + reward_pose * (reward_heading)
                + reward_effort
        )

        misbehave = (
                (self.drone.pos[..., 2] < 0.1)
                | (pos_error > self.reset_thres)
        )
        hasnan = torch.isnan(self.drone_state).any(-1)

        terminated = misbehave | hasnan
        truncated = (self.progress_buf >= self.max_episode_length - 1).unsqueeze(-1)

        self.stats["tracking_error"].add_(pos_error)
        self.stats["heading_alignment"].add_(heading_alignment)
        self.stats["return"].add_(reward)
        self.stats["episode_len"][:] = self.progress_buf.unsqueeze(1)
        self.stats["success"][:] = truncated.float()
        self.observation_h[..., :-1] = self.observation_h[..., 1:]

        return TensorDict(
            {
                "agents": {
                    "reward": reward.unsqueeze(-1),
                },
                "done": terminated | truncated,
                "terminated": terminated,
                "truncated": truncated,
            },
            self.batch_size,
        )

    def _compute_traj(self, steps: int, env_ids=None, step_size: float = 1.0):
        if env_ids is None:
            env_ids = ...
        t = self.progress_buf[env_ids].unsqueeze(1) + step_size * torch.arange(
            steps, device=self.device
        )
        t = self.traj_t0 + scale_time(self.traj_w[env_ids].unsqueeze(1) * t * self.dt)
        traj_rot = self.traj_rot[env_ids].unsqueeze(1).expand(-1, t.shape[1], 4)

        ref_pos = vmap(lemniscate)(t, self.traj_c[env_ids])
        ref_pos = vmap(quat_rotate)(traj_rot, ref_pos) * self.traj_scale[
            env_ids
        ].unsqueeze(1)

        return self.origin + ref_pos

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            self.max_episode_length = self.cfg.env.max_episode_length
        else:
            self.max_episode_length = int(self.cfg.env.max_episode_length * 2)
        return self
