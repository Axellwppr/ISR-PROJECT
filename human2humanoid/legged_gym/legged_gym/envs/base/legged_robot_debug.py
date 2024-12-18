from legged_gym import LEGGED_GYM_ROOT_DIR, envs
import time
from warnings import WarningMessage
import numpy as np
import os

from isaacgym.torch_utils import *
from phc.utils import torch_utils
from isaacgym import gymtorch, gymapi, gymutil
import torch.nn.functional as F
import torch
from torch import Tensor
from typing import Tuple, Dict
import copy
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.math import wrap_to_pi
from legged_gym.utils.isaacgym_utils import get_euler_xyz as get_euler_xyz_in_tensor
from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils.transform import apply_rotation_to_quat_z
from .legged_robot_config import LeggedRobotCfg
from .lpf import ActionFilterButter, ActionFilterExp, ActionFilterButterTorch

from phc.utils.motion_lib_h1 import MotionLibH1
from phc.learning.network_loader import load_mcp_mlp
from smpl_sim.poselib.skeleton.skeleton3d import SkeletonTree
from termcolor import colored
from rl_games.algos_torch import torch_ext
from rsl_rl.modules import VelocityEstimator, VelocityEstimatorGRU
from easydict import EasyDict
from legged_gym.utils import task_registry
from phc.learning.network_loader import load_mlp
from typing import OrderedDict
import torch.optim as optim


class LeggedRobot(BaseTask):
    def __init__(
        self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless
    ):
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = self.cfg.viewer.debug_viz
        self.init_done = False
        self._parse_cfg(self.cfg)
        self.self_obs_size = 0
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)

        self._init_buffers()

        self.reset_idx(torch.arange(self.num_envs).to(self.device))
        self.compute_observations()  # compute initial obs vuffer.
        self.start_idx = 0

    def set_camera(self, position, lookat):
        """Set camera position and direction"""
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def step(self, actions):
        """Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        clip_actions = self.cfg.normalization.clip_actions

        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        actions = self.actions.clone()

        self.render()
        # self.actions = actions.clone()
        for _ in range(self.cfg.control.decimation):
            # random actions
            actions = torch.rand_like(actions) * 2 - 1
            # self.torques = self._compute_torques(actions).view((1, self.num_dofs))
            # breakpoint()
            # self.torques = torch.zeros_like(self.torques)
            # self.torques[:, 13] = 0
            # self.torques[:, 11] = 0
            # self.torques[:, 9] = 0
            # self.torques[:, 6] = 0
            # self.torques[:, 4] = 0
            # self.torques[:, 2] = 0

            # self.torques[:, :] = 0
            # print(self.torques)
            self.gym.set_dof_position_target_tensor(
                self.sim, gymtorch.unwrap_tensor(actions * self.cfg.control.action_scale)
            )
            # self.gym.set_dof_actuation_force_tensor(
            #     self.sim, gymtorch.unwrap_tensor(self.torques)
            # )
            self.gym.simulate(self.sim)

            if self.device == "cpu":
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)

        self.post_physics_step()

        return (
            self.obs_buf,
            self.privileged_obs_buf,
            self.rew_buf,
            self.reset_buf,
            self.extras,
        )

    def _refresh_sim_tensors(self):

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        return

    def post_physics_step(self):
        """check terminations, compute observations and rewards
        calls self._post_physics_step_callback() for common computations
        calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

    def compute_observations(self):
        self.obs_buf = torch.zeros(
            self.num_envs,
            741,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.privileged_obs_buf = torch.zeros(
            self.num_envs,
            837,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )

    def create_sim(self):
        """Creates simulation, terrain and evironments"""
        self.up_axis_idx = 2  # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(
            self.sim_device_id,
            self.graphics_device_id,
            self.physics_engine,
            self.sim_params,
        )
        mesh_type = self.cfg.terrain.mesh_type = "plane"
        if mesh_type == "plane":
            self._create_ground_plane()
        self._create_envs()

    def _process_dof_props(self, props, env_id):
        """Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id == 0:
            self.dof_pos_limits = torch.zeros(
                self.num_dof,
                2,
                dtype=torch.float,
                device=self.device,
                requires_grad=False,
            )
            self.dof_vel_limits = torch.zeros(
                self.num_dof, dtype=torch.float, device=self.device, requires_grad=False
            )
            self.torque_limits = torch.zeros(
                self.num_dof, dtype=torch.float, device=self.device, requires_grad=False
            )
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = (
                    m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                )
                self.dof_pos_limits[i, 1] = (
                    m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                )
        return props

    def _compute_torques(self, actions):
        # pd controller
        actions_scaled = self.cfg.control.action_scale * torch.rand_like(self.dof_pos)

        control_type = self.cfg.control.control_type
        if control_type == "P":
            torques = (
                100.0 * (actions_scaled + self.default_dof_pos - self.dof_pos)
                - 5.0 * self.dof_vel
            )
        print(self.torques)
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _reset_dofs(self, env_ids):
        self.dof_pos[env_ids] = 0.0
        self.dof_vel[env_ids] = 0.0

        env_ids_int32 = env_ids.to(dtype=torch.int32)

        print("before reset dof")
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )
        print("after reset dof")

    def _reset_root_states(self, env_ids):
        self.root_states[env_ids, 2] += 5.0
        env_ids_int32 = env_ids.to(dtype=torch.int32)

        # env_ids_int32 = torch.arange(self.num_envs).to(dtype=torch.int32).cuda()
        env_ids_int32 = (
            torch.arange(self.num_envs).to(dtype=torch.int32).to(self.device)
        )
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    # ----------------------------------------
    def _init_buffers(self):
        """Initialize torch tensors which will contain simulation states and processed quantities"""
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        # self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.root_states = gymtorch.wrap_tensor(actor_root_state)

        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]
        self.rpy = get_euler_xyz_in_tensor(self.base_quat)
        self.base_pos = self.root_states[: self.num_envs, 0:3]
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(
            self.num_envs, -1, 3
        )  # shape: num_envs, num_bodies, xyz axis

        # init rigid body state
        self._rigid_body_state = gymtorch.wrap_tensor(rigid_body_state)
        bodies_per_env = self._rigid_body_state.shape[0] // self.num_envs
        self._rigid_body_state_reshaped = self._rigid_body_state.view(
            self.num_envs, bodies_per_env, 13
        )
        self._rigid_body_pos = self._rigid_body_state_reshaped[
            ..., : self.num_bodies, 0:3
        ]
        self._rigid_body_rot = self._rigid_body_state_reshaped[
            ..., : self.num_bodies, 3:7
        ]
        self._rigid_body_vel = self._rigid_body_state_reshaped[
            ..., : self.num_bodies, 7:10
        ]
        self._rigid_body_ang_vel = self._rigid_body_state_reshaped[
            ..., : self.num_bodies, 10:13
        ]

        self.torques = torch.zeros(
            self.num_envs,
            self.num_actions,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.actions = torch.zeros(
            self.num_envs,
            self.num_actions,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.last_actions = torch.zeros(
            self.num_envs,
            self.num_actions,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.default_dof_pos = torch.zeros((1, 28)).unsqueeze(0)

    def _create_ground_plane(self):
        """Adds a ground plane to the simulation, sets friction and restitution based on the cfg."""
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self):
        """Creates environments:
        1. loads the robot URDF/MJCF asset,
        2. For each environment
           2.1 creates the environment,
           2.2 calls DOF and Rigid shape properties callbacks,
           2.3 create actor with these properties and add them to the env
        3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = (
            self.cfg.asset.replace_cylinder_with_capsule
        )
        asset_options.fix_base_link = False
        # asset_options.angular_damping = 0.5
        # asset_options.linear_damping = 0.1
        asset_options.disable_gravity = False
        # asset_options.override_com = True
        # asset_options.override_inertia = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS

        robot_asset = self.gym.load_asset(
            self.sim, asset_root, asset_file, asset_options
        )
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        dof_props_asset["driveMode"] = gymapi.DOF_MODE_POS

        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        # import pdb; pdb.set_trace()
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        # import ipdb; ipdb.set_trace()
        base_init_state_list = (
            self.cfg.init_state.pos
            + self.cfg.init_state.rot
            + self.cfg.init_state.lin_vel
            + self.cfg.init_state.ang_vel
        )

        self.base_init_state = to_torch(
            base_init_state_list, device=self.device, requires_grad=False
        )
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0.0, 0.0, 0.0)
        env_upper = gymapi.Vec3(0.0, 0.0, 0.0)

        self.actor_handles = []
        self.envs = []
        self.marker_handles = []

        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(
                self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs))
            )
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1.0, 1.0, (2, 1), device=self.device).squeeze(
                1
            )
            start_pose.p = gymapi.Vec3(*pos)

            self.gym.set_asset_rigid_shape_properties(
                robot_asset, rigid_shape_props_asset
            )
            actor_handle = self.gym.create_actor(
                env_handle,
                robot_asset,
                start_pose,
                self.cfg.asset.name,
                i,
                self.cfg.asset.self_collisions,
                0,
            )
            self._body_list = self.gym.get_actor_rigid_body_names(
                env_handle, actor_handle
            )
            self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props_asset)
            body_props = self.gym.get_actor_rigid_body_properties(
                env_handle, actor_handle
            )
            self.gym.set_actor_rigid_body_properties(
                env_handle, actor_handle, body_props, recomputeInertia=False
            )
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

    def _get_env_origins(self):
        """Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
        Otherwise create a grid.
        """
        self.custom_origins = False
        self.env_origins = torch.zeros(
            self.num_envs, 3, device=self.device, requires_grad=False
        )
        # create a grid of robots
        num_cols = np.floor(np.sqrt(self.num_envs))
        num_rows = np.ceil(self.num_envs / num_cols)
        xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
        spacing = self.cfg.env.env_spacing
        self.env_origins[:, 0] = spacing * xx.flatten()[: self.num_envs]
        self.env_origins[:, 1] = spacing * yy.flatten()[: self.num_envs]
        self.env_origins[:, 2] = 0.0

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales

        if isinstance(self.cfg.rewards.scales, EasyDict):
            self.reward_scales = {
                k: eval(v) if isinstance(v, str) else v
                for k, v in self.cfg.rewards.scales.items()
            }
            self.command_ranges = self.cfg.commands.ranges
        else:
            self.reward_scales = class_to_dict(self.cfg.rewards.scales)
            self.command_ranges = class_to_dict(self.cfg.commands.ranges)

        if self.cfg.terrain.mesh_type not in ["heightfield", "trimesh"]:
            self.cfg.terrain.curriculum = False

        self.max_episode_length_s = self.cfg.env.episode_length_s
        # import pdb; pdb.set_trace()
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.cfg.domain_rand.push_interval = np.ceil(
            self.cfg.domain_rand.push_interval_s / self.dt
        )
        self.cfg.domain_rand.package_loss_interval = np.ceil(
            self.cfg.domain_rand.package_loss_interval_s / self.dt
        )
        self.cfg.motion.resample_motions_for_envs_interval = int(
            np.ceil(self.cfg.motion.resample_motions_for_envs_interval_s / self.dt)
        )

    # ------------ reward functions----------------
    def render(self, sync_frame_time=False):
        super().render(sync_frame_time)
        return
