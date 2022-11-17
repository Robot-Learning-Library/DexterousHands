# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import os
import torch
import random

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *
from tasks.hand_base.base_task import BaseTask

from tasks.hand_base.vec_task import VecTask

from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import matplotlib.pyplot as plt
from PIL import Image as Im
import cv2

class AllegroHandLego(BaseTask):

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless, agent_index=[[[0, 1, 2, 3, 4, 5]], [[0, 1, 2, 3, 4, 5]]], is_multi_agent=False):

        self.cfg = cfg

        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.agent_index = agent_index

        self.is_multi_agent = is_multi_agent

        self.randomize = self.cfg["task"]["randomize"]
        self.randomization_params = self.cfg["task"]["randomization_params"]
        
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]

        self.dist_reward_scale = self.cfg["env"]["distRewardScale"]
        self.rot_reward_scale = self.cfg["env"]["rotRewardScale"]
        self.action_penalty_scale = self.cfg["env"]["actionPenaltyScale"]
        self.success_tolerance = self.cfg["env"]["successTolerance"]
        self.reach_goal_bonus = self.cfg["env"]["reachGoalBonus"]
        self.fall_dist = self.cfg["env"]["fallDistance"]
        self.fall_penalty = self.cfg["env"]["fallPenalty"]
        self.rot_eps = self.cfg["env"]["rotEps"]

        self.vel_obs_scale = 0.2  # scale factor of velocity based observations
        self.force_torque_obs_scale = 10.0  # scale factor of velocity based observations

        self.reset_position_noise = self.cfg["env"]["resetPositionNoise"]
        self.reset_rotation_noise = self.cfg["env"]["resetRotationNoise"]
        self.reset_dof_pos_noise = self.cfg["env"]["resetDofPosRandomInterval"]
        self.reset_dof_vel_noise = self.cfg["env"]["resetDofVelRandomInterval"]

        self.force_scale = self.cfg["env"].get("forceScale", 0.0)
        self.force_prob_range = self.cfg["env"].get("forceProbRange", [0.001, 0.1])
        self.force_decay = self.cfg["env"].get("forceDecay", 0.99)
        self.force_decay_interval = self.cfg["env"].get("forceDecayInterval", 0.08)
        self.rotation_axis = "y"
        if self.rotation_axis == "x":
            self.rotation_id = 0
        elif self.rotation_axis == "y":
            self.rotation_id = 1
        else:
            self.rotation_id = 2

        self.shadow_hand_dof_speed_scale = self.cfg["env"]["dofSpeedScale"]
        self.use_relative_control = self.cfg["env"]["useRelativeControl"]
        self.act_moving_average = self.cfg["env"]["actionsMovingAverage"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.reset_time = self.cfg["env"].get("resetTime", -1.0)
        self.print_success_stat = self.cfg["env"]["printNumSuccesses"]
        self.max_consecutive_successes = self.cfg["env"]["maxConsecutiveSuccesses"]
        self.av_factor = self.cfg["env"].get("averFactor", 0.1)

        self.object_type = self.cfg["env"]["objectType"]
        self.spin_coef = self.cfg["env"].get("spin_coef", 1.0)
        assert self.object_type in ["block", "egg", "pen"]

        self.ignore_z = (self.object_type == "pen")

        self.robot_asset_files_dict = {
            "normal": "urdf/franka_description/robots/franka_panda_allegro.urdf",
            "large":  "urdf/xarm6/xarm6_allegro_left_fsr_large.urdf"
        }
        self.asset_files_dict = {
            "block": "urdf/objects/cube_multicolor.urdf",
            "egg": "mjcf/box/mobility.urdf",
            "pen": "mjcf/open_ai_assets/hand/pen.xml"
        }

        # can be "full_no_vel", "full", "full_state"
        self.obs_type = self.cfg["env"]["observationType"]

        if not (self.obs_type in ["full_no_vel", "full", "full_state", "full_contact", "partial_contact"]):
            raise Exception(
                "Unknown type of observations!\nobservationType should be one of: [openai, full_no_vel, full, full_state]")

        print("Obs type:", self.obs_type)

        self.palm_name = "palm"
        self.contact_sensor_names = ["link_1.0_fsr", "link_2.0_fsr", "link_3.0_tip_fsr",
                                     "link_5.0_fsr", "link_6.0_fsr", "link_7.0_tip_fsr", "link_9.0_fsr",
                                     "link_10.0_fsr", "link_11.0_tip_fsr", "link_14.0_fsr", "link_15.0_fsr",
                                     "link_15.0_tip_fsr"]
        self.fingertip_names = ["link_3.0_tip",
                                "link_7.0_tip",
                                "link_11.0_tip",
                                "link_15.0_tip"]
        # 11, 13, 16, 20, 22, 24, 27, 29, 32, 36, 39, 40
        # self.contact_sensor_names = ["link_1.0", "link_2.0", "link_3.0_tip",
        #                              "link_5.0", "link_6.0", "link_7.0_tip", "link_9.0",
        #                              "link_10.0", "link_11.0_tip", "link_14.0", "link_15.0",
        #                              "link_15.0_tip"]
        self.stack_obs = 1
        self.num_obs_dict = {
            "full_no_vel": 50,
            "full": 72,
            "full_state": 88,
            "full_contact": 90,
            "partial_contact": 75 + 192*256*4
        }

        self.num_obs_dict = {
            "full_no_vel": 50,
            "full": 72,
            "full_state": 88,
            "full_contact": 90,
            "partial_contact": 75
        }
        self.up_axis = 'z'

        self.use_vel_obs = False
        self.fingertip_obs = True
        self.asymmetric_obs = self.cfg["env"]["asymmetric_observations"]

        num_states = 0
        if self.asymmetric_obs:
            num_states = 95

        self.one_frame_num_obs = self.num_obs_dict[self.obs_type]
        self.one_frame_num_states = num_states
        self.cfg["env"]["numObservations"] = self.num_obs_dict[self.obs_type] * self.stack_obs
        self.cfg["env"]["numStates"] = num_states * self.stack_obs
        self.cfg["env"]["numActions"] = 23

        self.cfg["device_type"] = device_type
        self.cfg["device_id"] = device_id
        self.cfg["headless"] = headless

        self.enable_camera_sensors = self.cfg["env"]["enable_camera_sensors"]

        super().__init__(cfg=self.cfg)

        self.dt = self.sim_params.dt
        control_freq_inv = self.cfg["env"].get("controlFrequencyInv", 1)
        if self.reset_time > 0.0:
            self.max_episode_length = int(round(self.reset_time/(control_freq_inv * self.dt)))
            print("Reset time: ", self.reset_time)
            print("New episode length: ", self.max_episode_length)

        if self.viewer != None:
            cam_pos = gymapi.Vec3(10.0, 5.0, 1.0)
            cam_target = gymapi.Vec3(6.0, 5.0, 0.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # get gym GPU state tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        # contact_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.jacobian_tensor = gymtorch.wrap_tensor(self.gym.acquire_jacobian_tensor(self.sim, "hand"))

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.arm_hand_default_dof_pos = torch.zeros(self.num_arm_hand_dofs, dtype=torch.float, device=self.device)
        self.arm_hand_default_dof_pos[:7] = torch.tensor([-1.2459,  0.4489, -0.1304, -2.3551,  0.2517,  2.7242, -1.6652], dtype=torch.float, device=self.device)
        self.arm_hand_default_dof_pos[7:] = to_torch([0.0, -0.174, 0.785, 0.785,
                                            0.0, -0.174, 0.785, 0.785, 0.0, -0.174, 0.785, 0.785, 0.0, -0.174, 0.785, 0.785], dtype=torch.float, device=self.device)
        self.arm_hand_prepare_dof_pos = to_torch([1.0260,  0.0671, -1.1514, -2.4576, -2.8973,  3.7172,  2.8975,
                                                0.0, -0.174, 0.785, 0.785,
                                            0.0, -0.174, 0.785, 0.785, 0.0, -0.174, 0.785, 0.785, 0.0, -0.174, 0.785, 0.785], dtype=torch.float, device=self.device)


        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.arm_hand_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_arm_hand_dofs]
        self.arm_hand_dof_pos = self.arm_hand_dof_state[..., 0]
        self.arm_hand_dof_vel = self.arm_hand_dof_state[..., 1]

        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.num_bodies = self.rigid_body_states.shape[1]

        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)

        # self.contact_tensor = gymtorch.wrap_tensor(contact_tensor).view(self.num_envs, -1)
        # print("Contact Tensor Dimension", self.contact_tensor.shape)

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        print("Num dofs: ", self.num_dofs)

        self.prev_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)

        self.global_indices = torch.arange(self.num_envs * 3, dtype=torch.int32, device=self.device).view(self.num_envs, -1)
        self.x_unit_tensor = to_torch([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = to_torch([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = to_torch([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

        self.reset_goal_buf = self.reset_buf.clone()
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.consecutive_successes = torch.zeros(1, dtype=torch.float, device=self.device)

        self.av_factor = to_torch(self.av_factor, dtype=torch.float, device=self.device)

        self.total_successes = 0
        self.total_resets = 0
        self.total_steps = 0

        # object apply random forces parameters
        self.force_decay = to_torch(self.force_decay, dtype=torch.float, device=self.device)
        self.force_prob_range = to_torch(self.force_prob_range, dtype=torch.float, device=self.device)
        self.random_force_prob = torch.exp((torch.log(self.force_prob_range[0]) - torch.log(self.force_prob_range[1]))
                                           * torch.rand(self.num_envs, device=self.device) + torch.log(self.force_prob_range[1]))

        self.rb_forces = torch.zeros((self.num_envs, self.num_bodies, 3), dtype=torch.float, device=self.device)
        self.object_pose_for_open_loop = torch.zeros_like(self.root_state_tensor[self.object_indices, 0:7])

        self.hand_base_rigid_body_index = self.gym.find_actor_rigid_body_index(self.envs[0], self.hand_indices[0], "base_link", gymapi.DOMAIN_ENV)
        print("hand_base_rigid_body_index: ", self.hand_base_rigid_body_index)

    def create_sim(self):
        self.dt = self.sim_params.dt
        self.up_axis_idx = self.set_sim_params_up_axis(self.sim_params, self.up_axis)
        self.sim_params.physx.max_gpu_contact_pairs = int(self.sim_params.physx.max_gpu_contact_pairs * 2)

        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets')

        arm_hand_asset_file = self.robot_asset_files_dict["normal"]
        # arm_hand_asset_file = "urdf/xarm6/xarm6_allegro_left.urdf"
        #"urdf/xarm6/xarm6_allegro_fsr.urdf"

        if "asset" in self.cfg["env"]:
            asset_root = self.cfg["env"]["asset"].get("assetRoot", asset_root)
            # arm_hand_asset_file = self.cfg["env"]["asset"].get("assetFileName", arm_hand_asset_file)

        object_asset_file = self.asset_files_dict[self.object_type]

        # load arm and hand.
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = False
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.angular_damping = 0.01
        # asset_options.use_mesh_materials = True
        asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
        asset_options.override_com = True
        asset_options.override_inertia = True
        asset_options.vhacd_enabled = True
        asset_options.vhacd_params = gymapi.VhacdParams()
        asset_options.vhacd_params.resolution = 2000000
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        if self.physics_engine == gymapi.SIM_PHYSX:
            asset_options.use_physx_armature = True
        arm_hand_asset = self.gym.load_asset(self.sim, asset_root, arm_hand_asset_file, asset_options)
        self.num_arm_hand_bodies = self.gym.get_asset_rigid_body_count(arm_hand_asset)
        self.num_arm_hand_shapes = self.gym.get_asset_rigid_shape_count(arm_hand_asset)
        self.num_arm_hand_dofs = self.gym.get_asset_dof_count(arm_hand_asset)
        print("Num dofs: ", self.num_arm_hand_dofs)
        print("num_arm_hand_shapes: ", self.num_arm_hand_shapes)
        print("num_arm_hand_bodies: ", self.num_arm_hand_bodies)
        self.num_arm_hand_actuators = self.num_arm_hand_dofs #self.gym.get_asset_actuator_count(shadow_hand_asset)

        # Set up each DOF.
        self.actuated_dof_indices = [i for i in range(7, self.num_arm_hand_dofs)]

        self.arm_hand_dof_lower_limits = []
        self.arm_hand_dof_upper_limits = []
        self.arm_hand_dof_default_pos = []
        self.arm_hand_dof_default_vel = []

        robot_lower_qpos = []
        robot_upper_qpos = []

        robot_dof_props = self.gym.get_asset_dof_properties(arm_hand_asset)

        for i in range(23):
            robot_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
            if i < 7:
                robot_dof_props['velocity'][i] = 3.0
            else:
                robot_dof_props['velocity'][i] = 3.0
            robot_dof_props['friction'][i] = 0.02
            robot_dof_props['stiffness'][i] = 10
            robot_dof_props['armature'][i] = 0.001

            if i < 7:
                robot_dof_props['damping'][i] = 0.1
            else:
                robot_dof_props['damping'][i] = 0.2
            robot_lower_qpos.append(robot_dof_props['lower'][i])
            robot_upper_qpos.append(robot_dof_props['upper'][i])

        self.actuated_dof_indices = to_torch(self.actuated_dof_indices, dtype=torch.long, device=self.device)
        self.arm_hand_dof_lower_limits = to_torch(robot_lower_qpos, device=self.device)
        self.arm_hand_dof_upper_limits = to_torch(robot_upper_qpos, device=self.device)
        self.arm_hand_dof_lower_qvel = to_torch(-robot_dof_props["velocity"], device=self.device)
        self.arm_hand_dof_upper_qvel = to_torch(robot_dof_props["velocity"], device=self.device)

        for i in range(self.num_arm_hand_dofs):
            self.arm_hand_dof_default_vel.append(0.0)

        self.arm_hand_dof_default_pos = to_torch(self.arm_hand_dof_default_pos, device=self.device)
        self.arm_hand_dof_default_vel = to_torch(self.arm_hand_dof_default_vel, device=self.device)

        # load manipulated object and goal assets
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.disable_gravity = True
        object_asset_options.fix_base_link = True
        # object_asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
        # object_asset_options.override_com = True
        # object_asset_options.override_inertia = True
        # object_asset_options.vhacd_enabled = True
        # object_asset_options.vhacd_params = gymapi.VhacdParams()
        # object_asset_options.vhacd_params.resolution = 100000
        # object_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        object_asset = self.gym.load_asset(self.sim, asset_root, object_asset_file, object_asset_options)

        object_asset_options.disable_gravity = True
        goal_asset = self.gym.load_asset(self.sim, asset_root, object_asset_file, object_asset_options)

        # Put objects in the scene.
        arm_hand_start_pose = gymapi.Transform()
        arm_hand_start_pose.p = gymapi.Vec3(-0.3, 0.0, 0.7)
        arm_hand_start_pose.r = gymapi.Quat().from_euler_zyx(0, 0, 0.0)

        # create table asset
        table_dims = gymapi.Vec3(1.0, 1.0, 0.6)
        table_asset_options = gymapi.AssetOptions()
        table_asset_options.fix_base_link = True
        table_asset_options.flip_visual_attachments = True
        table_asset_options.collapse_fixed_joints = True
        table_asset_options.disable_gravity = True
        table_asset_options.thickness = 0.001

        table_asset = self.gym.create_box(self.sim, table_dims.x, table_dims.y, table_dims.z, table_asset_options)
        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3(0.0, 0.0, 0.5 * table_dims.z)
        table_pose.r = gymapi.Quat().from_euler_zyx(-0., 0, 0)

        # create box asset
        box_assets = []
        box_start_poses = []

        box_thin = 0.01
        box_xyz = [0.52, 0.72, 0.2]
        box_offset = [0.1, 0, 0]

        box_asset_options = gymapi.AssetOptions()
        box_asset_options.disable_gravity = False
        box_asset_options.fix_base_link = True
        box_asset_options.flip_visual_attachments = True
        box_asset_options.collapse_fixed_joints = True
        box_asset_options.disable_gravity = True
        box_asset_options.thickness = 0.001

        box_bottom_asset = self.gym.create_box(self.sim, box_xyz[0], box_xyz[1], box_thin, table_asset_options)
        box_left_asset = self.gym.create_box(self.sim, box_xyz[0], box_thin, box_xyz[2], table_asset_options)
        box_right_asset = self.gym.create_box(self.sim, box_xyz[0], box_thin, box_xyz[2], table_asset_options)
        box_former_asset = self.gym.create_box(self.sim, box_thin, box_xyz[1], box_xyz[2], table_asset_options)
        box_after_asset = self.gym.create_box(self.sim, box_thin, box_xyz[1], box_xyz[2], table_asset_options)

        box_bottom_start_pose = gymapi.Transform()
        box_bottom_start_pose.p = gymapi.Vec3(0.0 + box_offset[0], 0.0 + box_offset[1], 0.6 + (box_thin) / 2)
        box_left_start_pose = gymapi.Transform()
        box_left_start_pose.p = gymapi.Vec3(0.0 + box_offset[0], (box_xyz[1] - box_thin) / 2 + box_offset[1], 0.6 + (box_xyz[2]) / 2)
        box_right_start_pose = gymapi.Transform()
        box_right_start_pose.p = gymapi.Vec3(0.0 + box_offset[0], -(box_xyz[1] - box_thin) / 2 + box_offset[1], 0.6 + (box_xyz[2]) / 2)
        box_former_start_pose = gymapi.Transform()
        box_former_start_pose.p = gymapi.Vec3((box_xyz[0] - box_thin) / 2 + box_offset[0], 0.0 + box_offset[1], 0.6 + (box_xyz[2]) / 2)
        box_after_start_pose = gymapi.Transform()
        box_after_start_pose.p = gymapi.Vec3(-(box_xyz[0] - box_thin) / 2 + box_offset[0], 0.0 + box_offset[1], 0.6 + (box_xyz[2]) / 2)

        box_assets.append(box_bottom_asset)
        box_assets.append(box_left_asset)
        box_assets.append(box_right_asset)
        box_assets.append(box_former_asset)
        box_assets.append(box_after_asset)
        box_start_poses.append(box_bottom_start_pose)
        box_start_poses.append(box_left_start_pose)
        box_start_poses.append(box_right_start_pose)
        box_start_poses.append(box_former_start_pose)
        box_start_poses.append(box_after_start_pose)

        object_start_pose = gymapi.Transform()
        object_start_pose.p = gymapi.Vec3(0, 0.0, -10.78)
        object_start_pose.r = gymapi.Quat().from_euler_zyx(0, 0, 1.571)

        if self.object_type == "pen":
            object_start_pose.p.z = arm_hand_start_pose.p.z + 0.02

        self.goal_displacement = gymapi.Vec3(-0.2, -0.06, -10.12)
        self.goal_displacement_tensor = to_torch(
            [self.goal_displacement.x, self.goal_displacement.y, self.goal_displacement.z], device=self.device)
        goal_start_pose = gymapi.Transform()
        goal_start_pose.p = object_start_pose.p + self.goal_displacement

        goal_start_pose.p.z -= 0.04

        lego_path = "urdf/objects/lego/urdf/"
        all_lego_files_name = os.listdir("/home/jmji/DexterousHandEnvs/assets/" + lego_path)

        lego_assets = []
        lego_start_poses = []
        self.segmentation_id = 200

        for n in range(3):
            for i, lego_file_name in enumerate(all_lego_files_name):
                lego_asset_options = gymapi.AssetOptions()
                lego_asset_options.disable_gravity = False
                # lego_asset_options.fix_base_link = True
                # lego_asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
                # lego_asset_options.override_com = True
                # lego_asset_options.override_inertia = True
                # lego_asset_options.vhacd_enabled = True
                # lego_asset_options.vhacd_params = gymapi.VhacdParams()
                # lego_asset_options.vhacd_params.resolution = 50000
                # lego_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
                lego_asset = self.gym.load_asset(self.sim, asset_root, lego_path + lego_file_name, lego_asset_options)

                lego_start_pose = gymapi.Transform()
                lego_start_pose.p = gymapi.Vec3(-0.15 + 0.1 * int(i % 4) + 0.1, -0.25 + 0.1 * int(i % 24 / 4), 0.62 + 0.15 * int(i / 24) + n * 0.2)
                lego_start_pose.r = gymapi.Quat().from_euler_zyx(0.785, 0.785, 0.0)
                # Assets visualization
                # lego_start_pose.p = gymapi.Vec3(-0.15 + 0.2 * int(i % 18) + 0.1, 0, 0.62 + 0.2 * int(i / 18) + n * 0.8 + 5.0)
                # lego_start_pose.r = gymapi.Quat().from_euler_zyx(0.0, 0, 0)

                lego_assets.append(lego_asset)
                lego_start_poses.append(lego_start_pose)

        # compute aggregate size
        max_agg_bodies = self.num_arm_hand_bodies + 2 + 1 + len(lego_assets) + 5 + 10
        max_agg_shapes = self.num_arm_hand_shapes + 2 + 1 + len(lego_assets) + 5 + 10

        self.arm_hands = []
        self.envs = []

        self.object_init_state = []
        self.lego_init_states = []
        self.hand_start_states = []

        self.hand_indices = []
        self.fingertip_indices = []
        self.object_indices = []
        self.goal_object_indices = []
        self.predict_object_indices = []
        self.table_indices = []
        self.lego_indices = []
        self.lego_segmentation_indices = []

        arm_hand_rb_count = self.gym.get_asset_rigid_body_count(arm_hand_asset)
        object_rb_count = self.gym.get_asset_rigid_body_count(object_asset)
        self.object_rb_handles = list(range(arm_hand_rb_count, arm_hand_rb_count + object_rb_count))

        self.cameras = []
        self.camera_tensors = []
        self.camera_seg_tensors = []
        self.camera_view_matrixs = []
        self.camera_proj_matrixs = []

        self.camera_props = gymapi.CameraProperties()
        self.camera_props.width = 256
        self.camera_props.height = 192
        self.camera_props.enable_tensors = True

        self.env_origin = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)
        self.pointCloudDownsampleNum = 768
        self.camera_u = torch.arange(0, self.camera_props.width, device=self.device)
        self.camera_v = torch.arange(0, self.camera_props.height, device=self.device)

        self.camera_v2, self.camera_u2 = torch.meshgrid(self.camera_v, self.camera_u, indexing='ij')

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )

            if self.aggregate_mode >= 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # add hand - collision filter = -1 to use asset collision filters set in mjcf loader
            arm_hand_actor = self.gym.create_actor(env_ptr, arm_hand_asset, arm_hand_start_pose, "hand", i, -1, 0)
            self.hand_start_states.append([arm_hand_start_pose.p.x,
                                           arm_hand_start_pose.p.y,
                                           arm_hand_start_pose.p.z,
                                           arm_hand_start_pose.r.x,
                                           arm_hand_start_pose.r.y,
                                           arm_hand_start_pose.r.z,
                                           arm_hand_start_pose.r.w,
                                           0, 0, 0, 0, 0, 0])
            self.gym.set_actor_dof_properties(env_ptr, arm_hand_actor, robot_dof_props)
            hand_idx = self.gym.get_actor_index(env_ptr, arm_hand_actor, gymapi.DOMAIN_SIM)
            self.hand_indices.append(hand_idx)

            # add object
            object_handle = self.gym.create_actor(env_ptr, object_asset, object_start_pose, "object", i, 0, 0)
            self.object_init_state.append([object_start_pose.p.x, object_start_pose.p.y, object_start_pose.p.z,
                                           object_start_pose.r.x, object_start_pose.r.y, object_start_pose.r.z, object_start_pose.r.w,
                                           0, 0, 0, 0, 0, 0])
            object_idx = self.gym.get_actor_index(env_ptr, object_handle, gymapi.DOMAIN_SIM)
            self.object_indices.append(object_idx)

            # add goal object
            goal_handle = self.gym.create_actor(env_ptr, goal_asset, goal_start_pose, "goal_object", i + self.num_envs, 0, 0)
            goal_object_idx = self.gym.get_actor_index(env_ptr, goal_handle, gymapi.DOMAIN_SIM)
            self.goal_object_indices.append(goal_object_idx)

            # add table
            table_handle = self.gym.create_actor(env_ptr, table_asset, table_pose, "table", i, -1, 0)
            # self.gym.set_rigid_body_texture(env_ptr, table_handle, 0, gymapi.MESH_VISUAL, table_texture_handle)
            table_idx = self.gym.get_actor_index(env_ptr, table_handle, gymapi.DOMAIN_SIM)
            self.gym.set_rigid_body_color(
                env_ptr, table_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(1, 0.9, 0.8)
            )
            self.table_indices.append(table_idx)
            
            table_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, table_handle)
            for object_shape_prop in table_shape_props:
                object_shape_prop.friction = 0.2
            self.gym.set_actor_rigid_shape_properties(env_ptr, table_handle, table_shape_props)

            # add box
            for box_i, box_asset in enumerate(box_assets):
                box_handle = self.gym.create_actor(env_ptr, box_asset, box_start_poses[box_i], "box_{}".format(box_i), i, 0, 0)
                # self.lego_init_state.append([lego_init_state.p.x, lego_init_state.p.y, object_start_pose.p.z,
                #                             lego_init_state.r.x, lego_init_state.r.y, object_start_pose.r.z, object_start_pose.r.w,
                #                             0, 0, 0, 0, 0, 0])
                # object_idx = self.gym.get_actor_index(env_ptr, object_handle, gymapi.DOMAIN_SIM)
                # self.object_indices.append(object_idx)
                self.gym.set_rigid_body_color(env_ptr, box_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.8, 0.4, 0))

            # add lego
            lego_idx = []
            for lego_i, lego_asset in enumerate(lego_assets):
                lego_handle = self.gym.create_actor(env_ptr, lego_asset, lego_start_poses[lego_i], "lego_{}".format(lego_i), i, 0, lego_i)
                # lego_handle = self.gym.create_actor(env_ptr, lego_asset, lego_start_poses[lego_i], "lego_{}".format(lego_i), i + self.num_envs + lego_i, -1, 0)
                self.lego_init_states.append([lego_start_poses[lego_i].p.x, lego_start_poses[lego_i].p.y, lego_start_poses[lego_i].p.z,
                                            lego_start_poses[lego_i].r.x, lego_start_poses[lego_i].r.y, lego_start_poses[lego_i].r.z, lego_start_poses[lego_i].r.w,
                                            0, 0, 0, 0, 0, 0])
                idx = self.gym.get_actor_index(env_ptr, lego_handle, gymapi.DOMAIN_SIM)
                if i == 0 and lego_i == self.segmentation_id:
                    self.lego_segmentation_index = self.gym.get_actor_index(env_ptr, lego_handle, gymapi.DOMAIN_SIM)
                lego_idx.append(idx)

                colorx = random.uniform(0, 1)
                colory = random.uniform(0, 1)
                colorz = random.uniform(0, 1)
                self.gym.set_rigid_body_color(env_ptr, lego_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(colorx, colory, colorz))
            self.lego_indices.append(lego_idx)

            if self.enable_camera_sensors:
                camera_handle = self.gym.create_camera_sensor(env_ptr, self.camera_props)
                self.gym.set_camera_location(camera_handle, env_ptr, gymapi.Vec3(0.35, 0, 1.2), gymapi.Vec3(-0.25, 0, 0))
                camera_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, camera_handle, gymapi.IMAGE_COLOR)
                torch_cam_tensor = gymtorch.wrap_tensor(camera_tensor)
                camera_seg_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, camera_handle, gymapi.IMAGE_SEGMENTATION)
                torch_cam_seg_tensor = gymtorch.wrap_tensor(camera_seg_tensor)

                cam_vinv = torch.inverse((torch.tensor(self.gym.get_camera_view_matrix(self.sim, env_ptr, camera_handle)))).to(self.device)
                cam_proj = torch.tensor(self.gym.get_camera_proj_matrix(self.sim, env_ptr, camera_handle), device=self.device)
            
            # Set up object...
            if self.object_type != "block":
                self.gym.set_rigid_body_color(
                    env_ptr, object_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.6, 0.72, 0.98)
                )
                self.gym.set_rigid_body_color(
                    env_ptr, goal_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.6, 0.72, 0.98)
                )

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            if self.enable_camera_sensors:
                origin = self.gym.get_env_origin(env_ptr)
                self.env_origin[i][0] = origin.x
                self.env_origin[i][1] = origin.y
                self.env_origin[i][2] = origin.z
                self.camera_tensors.append(torch_cam_tensor)
                self.camera_seg_tensors.append(torch_cam_seg_tensor)
                self.camera_view_matrixs.append(cam_vinv)
                self.camera_proj_matrixs.append(cam_proj)
                self.cameras.append(camera_handle)

            self.envs.append(env_ptr)
            self.arm_hands.append(arm_hand_actor)

        self.camera_rgbd_image_tensors = torch.stack(self.camera_tensors, dim=0)[:, :, :, :3].reshape(self.num_envs, -1)
        self.camera_seg_image_tensors = ((torch.stack(self.camera_seg_tensors, dim=0) == self.segmentation_id) * 255).reshape(self.num_envs, -1)
        self.emergence_reward = torch.zeros_like(self.reset_buf, device=self.device, dtype=torch.float)
        self.emergence_pixel = torch.zeros_like(self.reset_buf, device=self.device, dtype=torch.float)
        self.last_emergence_pixel = torch.zeros_like(self.reset_buf, device=self.device, dtype=torch.float)
        # Acquire specific links.
        sensor_handles = [self.gym.find_actor_rigid_body_handle(env_ptr, arm_hand_actor, sensor_name) for sensor_name in
                          self.contact_sensor_names]
        self.sensor_handle_indices = to_torch(sensor_handles, dtype=torch.int64)

        self.fingertip_handles = [self.gym.find_actor_rigid_body_handle(env_ptr, arm_hand_actor, name) for name in self.fingertip_names]
        print(self.fingertip_handles)
        object_rb_props = self.gym.get_actor_rigid_body_properties(env_ptr, object_handle)
        self.object_rb_masses = [prop.mass for prop in object_rb_props]

        self.object_init_state = to_torch(self.object_init_state, device=self.device, dtype=torch.float).view(self.num_envs, 13)
        self.goal_states = self.object_init_state.clone()
        self.goal_states[:, self.up_axis_idx] -= 0.02
        self.goal_init_state = self.goal_states.clone()
        self.hand_start_states = to_torch(self.hand_start_states, device=self.device).view(self.num_envs, 13)

        self.lego_init_states = to_torch(self.lego_init_states, device=self.device).view(self.num_envs, len(lego_assets), 13)

        self.fingertip_handles = to_torch(self.fingertip_handles, dtype=torch.long, device=self.device)
        self.object_rb_handles = to_torch(self.object_rb_handles, dtype=torch.long, device=self.device)
        self.object_rb_masses = to_torch(self.object_rb_masses, dtype=torch.float, device=self.device)

        self.hand_indices = to_torch(self.hand_indices, dtype=torch.long, device=self.device)
        self.object_indices = to_torch(self.object_indices, dtype=torch.long, device=self.device)
        self.goal_object_indices = to_torch(self.goal_object_indices, dtype=torch.long, device=self.device)
        self.lego_indices = to_torch(self.lego_indices, dtype=torch.long, device=self.device)

    def compute_reward(self, actions):
        self.rew_buf[:], self.reset_buf[:], self.reset_goal_buf[:], self.progress_buf[:], self.successes[:], self.consecutive_successes[:] = compute_hand_reward(
            torch.tensor(self.spin_coef).to(self.device), self.rew_buf, self.reset_buf, self.reset_goal_buf, self.progress_buf, self.successes, self.consecutive_successes,
            self.max_episode_length, self.object_pos, self.object_rot, self.object_angvel, self.goal_pos, self.goal_rot, self.segmentation_target_pos, self.hand_base_pos, self.emergence_reward, self.arm_hand_ff_pos, self.arm_hand_rf_pos, self.arm_hand_mf_pos, self.arm_hand_th_pos,
            self.dist_reward_scale, self.rot_reward_scale, self.rot_eps, self.actions, self.action_penalty_scale,
            self.success_tolerance, self.reach_goal_bonus, self.fall_dist, self.fall_penalty, self.rotation_id,
            self.max_consecutive_successes, self.av_factor, (self.object_type == "pen")
        )

        # self.extras['successes'] = self.successes
        # self.extras['consecutive_successes'] = self.consecutive_successes.mean()
        self.total_steps += 1
        # print("Total epoch = {}".format(int(self.total_steps/8)))

        if self.print_success_stat:
            print("Total steps = {}".format(self.total_steps))
            self.total_resets = self.total_resets + self.reset_buf.sum()
            direct_average_successes = self.total_successes + self.successes.sum()
            self.total_successes = self.total_successes + (self.successes * self.reset_buf).sum()

            # The direct average shows the overall result more quickly, but slightly undershoots long term
            # policy performance.
            print("Direct average consecutive successes = {:.1f}".format(direct_average_successes/(self.total_resets + self.num_envs)))
            if self.total_resets > 0:
                print("Post-Reset average consecutive successes = {:.1f}".format(self.total_successes/self.total_resets))

    def compute_observations(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        # self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)

        # if self.enable_camera_sensors:
        # if self.enable_camera_sensors and self.progress_buf[0] >= self.max_episode_length - 1:
        if self.enable_camera_sensors and self.progress_buf[0] % 20 == 1:
            self.gym.render_all_camera_sensors(self.sim)
            self.gym.start_access_image_tensors(self.sim)

            # camera_rgba_image = self.camera_visulization(env_id=0, is_depth_image=False)
            camera_rgba_image = self.camera_segmentation_visulization(self.camera_tensors, self.camera_seg_tensors, env_id=0, is_depth_image=False)
            self.compute_emergence_reward(self.camera_tensors, self.camera_seg_tensors, segmentation_id=self.segmentation_id)
            # self.camera_rgbd_image_tensors = torch.stack(self.camera_tensors, dim=0)[:, :, :, :3].reshape(self.num_envs, -1)
            # self.camera_seg_image_tensors = ((torch.stack(self.camera_seg_tensors, dim=0) == self.segmentation_id) * 255).reshape(self.num_envs, -1)

            cv2.imshow("Tracking", camera_rgba_image)
            cv2.waitKey(1)

            self.gym.end_access_image_tensors(self.sim)

        self.object_pose = self.root_state_tensor[self.object_indices, 0:7]
        self.object_pos = self.root_state_tensor[self.object_indices, 0:3]
        self.object_rot = self.root_state_tensor[self.object_indices, 3:7]
        self.object_linvel = self.root_state_tensor[self.object_indices, 7:10]
        self.object_angvel = self.root_state_tensor[self.object_indices, 10:13]

        self.goal_pose = self.goal_states[:, 0:7]
        self.goal_pos = self.goal_states[:, 0:3]
        self.goal_rot = self.goal_states[:, 3:7]

        self.hand_base_pose = self.rigid_body_states[:, self.hand_base_rigid_body_index, 0:7]
        self.hand_base_pos = self.rigid_body_states[:, self.hand_base_rigid_body_index, 0:3]
        self.hand_base_rot = self.rigid_body_states[:, self.hand_base_rigid_body_index, 3:7]

        self.segmentation_target_pose = self.rigid_body_states[:, self.lego_segmentation_index, 0:7]
        self.segmentation_target_pos = self.rigid_body_states[:, self.lego_segmentation_index, 0:3]
        self.segmentation_target_rot = self.rigid_body_states[:, self.lego_segmentation_index, 3:7]

        self.arm_hand_ff_pos = self.rigid_body_states[:, self.fingertip_handles[0], 0:3]
        self.arm_hand_ff_rot = self.rigid_body_states[:, self.fingertip_handles[0], 3:7]
        # self.arm_hand_ff_pos = self.arm_hand_ff_pos + quat_apply(self.arm_hand_ff_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02)
        self.arm_hand_mf_pos = self.rigid_body_states[:, self.fingertip_handles[1], 0:3]
        self.arm_hand_mf_rot = self.rigid_body_states[:, self.fingertip_handles[1], 3:7]
        # self.arm_hand_mf_pos = self.arm_hand_mf_pos + quat_apply(self.arm_hand_mf_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02)
        self.arm_hand_rf_pos = self.rigid_body_states[:, self.fingertip_handles[2], 0:3]
        self.arm_hand_rf_rot = self.rigid_body_states[:, self.fingertip_handles[2], 3:7]
        # self.arm_hand_rf_pos = self.arm_hand_rf_pos + quat_apply(self.arm_hand_rf_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02)
        # self.arm_hand_lf_pos = self.rigid_body_states[:, 20, 0:3]
        # self.arm_hand_lf_rot = self.rigid_body_states[:, 20, 3:7]
        # self.arm_hand_lf_pos = self.arm_hand_lf_pos + quat_apply(self.arm_hand_lf_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02)
        self.arm_hand_th_pos = self.rigid_body_states[:, self.fingertip_handles[3], 0:3]
        self.arm_hand_th_rot = self.rigid_body_states[:, self.fingertip_handles[3], 3:7]

        if self.obs_type == "full_no_vel":
            self.compute_full_observations(True)
        elif self.obs_type == "full":
            self.compute_full_observations()
        elif self.obs_type == "full_state":
            self.compute_full_state()
        elif self.obs_type == "full_contact":
            self.compute_contact_observations(True)
        elif self.obs_type == "partial_contact":
            self.compute_contact_observations(False)
        else:
            print("Unknown observations type!")

        if self.asymmetric_obs:
            self.compute_contact_asymmetric_observations()

    def compute_contact_asymmetric_observations(self):
            self.states_buf[:, 0:23] = unscale(self.arm_hand_dof_pos[:, 0:23],
                                                                self.arm_hand_dof_lower_limits[0:23],
                                                                self.arm_hand_dof_upper_limits[0:23])
            self.states_buf[:, 23:46] = self.vel_obs_scale * self.arm_hand_dof_vel[:, 0:23]

            self.states_buf[:, 56:79] = self.actions
            self.states_buf[:, 79:86] = self.hand_base_pose

            self.states_buf[:, 88:95] = self.segmentation_target_pose

    def compute_contact_observations(self, full_contact=True):
        self.obs_buf[:, 0:23] = unscale(self.arm_hand_dof_pos[:,0:23],
                                                            self.arm_hand_dof_lower_limits[0:23],
                                                            self.arm_hand_dof_upper_limits[0:23])
        # self.obs_buf[:, 16:23] = self.goal_pose
        self.obs_buf[:, 23:46] = self.actions

        self.obs_buf[:, 61:68] = self.hand_base_pose
        self.obs_buf[:, 68:75] = self.segmentation_target_pose
        # self.obs_buf[:, 68:75] = self.object_pose_for_open_loop

        # self.obs_buf[:, 75:192*256*3 + 75] = self.camera_rgbd_image_tensors
        # self.obs_buf[:, 192*256*3 + 75:192*256*4 + 75] = self.camera_seg_image_tensors

    def reset_target_pose(self, env_ids, apply_reset=False):
        # if int(self.total_steps/8) % 1000 == 0:
        #     torch.save(self.contact_slamer.state_dict(), "/home/jmji/isaacgym_rl/contact_slamer/ArmRotationLatentVectorStudent/model_{}.pt".format(int(self.total_steps/8)))
        rand_floats_x = torch_rand_float(-1, 1, (len(env_ids), 4), device=self.device)
        rand_floats_y = torch_rand_float(-1, 1, (len(env_ids), 4), device=self.device)

        new_rot = randomize_rotation(rand_floats_x[:, 0], rand_floats_y[:, 1],
                                     self.x_unit_tensor[env_ids],
                                     self.y_unit_tensor[env_ids])

        if apply_reset:
            self.object_pose_for_open_loop[env_ids] = self.goal_states[env_ids, 0:7]

        self.goal_states[env_ids, 0:3] = self.goal_init_state[env_ids, 0:3]
        # if not apply_reset:
        self.goal_states[env_ids, 3:7] = new_rot
        # self.goal_states[env_ids, 3:7] = self.goal_init_state[env_ids, 3:7]
        self.root_state_tensor[self.goal_object_indices[env_ids], 0:3] = self.goal_states[env_ids, 0:3] + self.goal_displacement_tensor
        self.root_state_tensor[self.goal_object_indices[env_ids], 3:7] = self.goal_states[env_ids, 3:7]
        self.root_state_tensor[self.goal_object_indices[env_ids], 7:13] = torch.zeros_like(self.root_state_tensor[self.goal_object_indices[env_ids], 7:13])

        if apply_reset:
            self.object_pose_for_open_loop[env_ids] = self.goal_states[env_ids, 0:7].clone()
            goal_object_indices = self.goal_object_indices[env_ids].to(torch.int32)
            self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                         gymtorch.unwrap_tensor(self.root_state_tensor),
                                                         gymtorch.unwrap_tensor(goal_object_indices), len(env_ids))
        self.reset_goal_buf[env_ids] = 0

    # default robot pose: [0.00, 0.782, -1.087, 3.487, 2.109, -1.415]
    def reset_idx(self, env_ids, goal_env_ids):
        # generate random values
        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), self.num_arm_hand_dofs * 2 + 5), device=self.device)

        # randomize start object poses
        self.reset_target_pose(env_ids)

        # reset rigid body forces
        self.rb_forces[env_ids, :, :] = 0.0

        # reset object
        self.root_state_tensor[self.object_indices[env_ids]] = self.object_init_state[env_ids].clone()
        self.root_state_tensor[self.object_indices[env_ids], 0:2] = self.object_init_state[env_ids, 0:2] + \
            self.reset_position_noise * rand_floats[:, 0:2]
        self.root_state_tensor[self.object_indices[env_ids], self.up_axis_idx] = self.object_init_state[env_ids, self.up_axis_idx] + \
            self.reset_position_noise * rand_floats[:, self.up_axis_idx]

        if self.obs_type == "full_contact" or "partial_contact":
            new_object_rot = randomize_rotation(torch.zeros_like(rand_floats[:, 3]),
                                                torch.zeros_like(rand_floats[:, 4]),
                                                self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids])
        else:
            new_object_rot = randomize_rotation(rand_floats[:, 3], rand_floats[:, 4], self.x_unit_tensor[env_ids],
                                                self.y_unit_tensor[env_ids])

        if self.object_type == "pen":
            rand_angle_y = torch.tensor(0.3)
            new_object_rot = randomize_rotation_pen(rand_floats[:, 3], rand_floats[:, 4], rand_angle_y,
                                                    self.x_unit_tensor[env_ids],
                                                    self.y_unit_tensor[env_ids],
                                                    self.z_unit_tensor[env_ids])

        self.root_state_tensor[self.object_indices[env_ids], 3:7] = self.object_init_state[env_ids, 3:7].clone()
        self.root_state_tensor[self.object_indices[env_ids], 7:13] = torch.zeros_like(self.root_state_tensor[self.object_indices[env_ids], 7:13])
        self.object_pose_for_open_loop[env_ids] = self.root_state_tensor[self.object_indices[env_ids], 0:7].clone()

        self.root_state_tensor[self.lego_indices[env_ids].view(-1), 0:7] = self.lego_init_states[env_ids].view(-1, 13)[:, 0:7].clone()
        self.root_state_tensor[self.lego_indices[env_ids].view(-1), 7:13] = torch.zeros_like(self.root_state_tensor[self.lego_indices[env_ids].view(-1), 7:13])

        object_indices = torch.unique(torch.cat([self.object_indices[env_ids],
                                                 self.goal_object_indices[env_ids],
                                                 self.goal_object_indices[goal_env_ids],
                                                 self.lego_indices[env_ids].view(-1)]).to(torch.int32))

        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_state_tensor),
                                                     gymtorch.unwrap_tensor(object_indices), len(object_indices))

        # reset random force probabilities
        self.random_force_prob[env_ids] = torch.exp((torch.log(self.force_prob_range[0]) - torch.log(self.force_prob_range[1]))
                                                    * torch.rand(len(env_ids), device=self.device) + torch.log(self.force_prob_range[1]))

        # reset shadow hand
        #delta_max = self.arm_hand_dof_upper_limits - self.arm_hand_dof_default_pos
        #delta_min = self.arm_hand_dof_lower_limits - self.arm_hand_dof_default_pos
        #rand_delta = delta_min + (delta_max - delta_min) * rand_floats[:, 5:5+self.num_arm_hand_dofs]

        pos = self.arm_hand_default_dof_pos #+ self.reset_dof_pos_noise * rand_delta
        self.arm_hand_dof_pos[env_ids, 0:23] = pos[0:23]
        self.arm_hand_dof_vel[env_ids, :] = self.arm_hand_dof_default_vel #+ \
        #     #self.reset_dof_vel_noise * rand_floats[:, 5+self.num_arm_hand_dofs:5+self.num_arm_hand_dofs*2]
        self.prev_targets[env_ids, :self.num_arm_hand_dofs] = pos
        self.cur_targets[env_ids, :self.num_arm_hand_dofs] = pos

        hand_indices = self.hand_indices[env_ids].to(torch.int32)
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.prev_targets),
                                                        gymtorch.unwrap_tensor(hand_indices), len(env_ids))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(hand_indices), len(env_ids))

        self.post_reset(env_ids, hand_indices)

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.successes[env_ids] = 0

    def post_reset(self, env_ids, hand_indices):
        # step physics and render each frame
        for i in range(30):
            self.render()
            self.gym.simulate(self.sim)

        if self.enable_camera_sensors:
            self.gym.render_all_camera_sensors(self.sim)
            self.gym.start_access_image_tensors(self.sim)

            # camera_rgba_image = self.camera_visulization(env_id=0, is_depth_image=False)
            camera_rgba_image = self.camera_segmentation_visulization(self.camera_tensors, self.camera_seg_tensors, env_id=0, is_depth_image=False)
            # self.compute_emergence_reward(self.camera_tensors, self.camera_seg_tensors, segmentation_id=self.segmentation_id)

            self.camera_rgbd_image_tensors = torch.stack(self.camera_tensors, dim=0)[:, :, :, :3].reshape(self.num_envs, -1)
            self.camera_seg_image_tensors = ((torch.stack(self.camera_seg_tensors, dim=0) == self.segmentation_id) * 255).reshape(self.num_envs, -1)

            for i in range(self.num_envs):
                torch_seg_tensor = self.camera_seg_tensors[i]
                self.last_emergence_pixel[i] = torch_seg_tensor[torch_seg_tensor == self.segmentation_id].shape[0]
            # cv2.imshow("Tracking", camera_rgba_image)
            # cv2.waitKey(1)

            self.gym.end_access_image_tensors(self.sim)

        self.arm_hand_dof_pos[env_ids, 0:23] = self.arm_hand_prepare_dof_pos
        self.prev_targets[env_ids, :self.num_arm_hand_dofs] = self.arm_hand_prepare_dof_pos
        self.cur_targets[env_ids, :self.num_arm_hand_dofs] = self.arm_hand_prepare_dof_pos

        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.prev_targets),
                                                        gymtorch.unwrap_tensor(hand_indices), len(env_ids))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(hand_indices), len(env_ids))
        print("post_reset finish")

    def pre_physics_step(self, actions):
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)

        # if only goals need reset, then call set API
        if len(goal_env_ids) > 0 and len(env_ids) == 0:
            self.reset_target_pose(goal_env_ids, apply_reset=True)

        # if goals need reset in addition to other envs, call set API in reset()
        elif len(goal_env_ids) > 0:
            self.reset_target_pose(goal_env_ids)

        if len(env_ids) > 0:
            self.reset_idx(env_ids, goal_env_ids)

        self.actions = actions.clone().to(self.device)

        # self.actions = torch.ones_like(actions.clone().to(self.device))
        if self.use_relative_control:
            # targets = self.prev_targets[:, self.actuated_dof_indices] + self.shadow_hand_dof_speed_scale * self.dt * self.actions
            targets = self.arm_hand_dof_pos[:, self.actuated_dof_indices] + self.shadow_hand_dof_speed_scale * self.dt * self.actions
            self.cur_targets[:, self.actuated_dof_indices] = tensor_clamp(targets,
                                                                          self.arm_hand_dof_lower_limits[self.actuated_dof_indices],
                                                                          self.arm_hand_dof_upper_limits[self.actuated_dof_indices])
        else:
            self.cur_targets[:, self.actuated_dof_indices] = scale(self.actions[:, 7:23],
                                                                   self.arm_hand_dof_lower_limits[self.actuated_dof_indices],
                                                                   self.arm_hand_dof_upper_limits[self.actuated_dof_indices])
            self.cur_targets[:, self.actuated_dof_indices] = self.act_moving_average * self.cur_targets[:,
                                                                                                        self.actuated_dof_indices] + (1.0 - self.act_moving_average) * self.prev_targets[:, self.actuated_dof_indices]
            self.cur_targets[:, self.actuated_dof_indices] = tensor_clamp(self.cur_targets[:, self.actuated_dof_indices],
                                                                          self.arm_hand_dof_lower_limits[self.actuated_dof_indices],
                                                                          self.arm_hand_dof_upper_limits[self.actuated_dof_indices])
            
            # qpos control robotic arm
            # targets = self.prev_targets[:, 6] + self.shadow_hand_dof_speed_scale * self.dt * self.actions[:, 6]
            # self.cur_targets[:, 6] = tensor_clamp(targets,
            #                                         self.arm_hand_dof_lower_limits[6],
            #                                         self.arm_hand_dof_upper_limits[6])
            # self.cur_targets[:, :7] = scale(self.actions[:, 0:7],
            #                                                        self.arm_hand_dof_lower_limits[0:7],
            #                                                        self.arm_hand_dof_upper_limits[0:7])
            # self.cur_targets[:, :7] = tensor_clamp(self.cur_targets[:, 0:7],
            #                                                               self.arm_hand_dof_lower_limits[0:7],
            #                                                               self.arm_hand_dof_upper_limits[0:7])

            # IK control robotic arm
            # target_pos = to_torch([0.05, 0.0, 0.95], device=self.device).repeat((self.num_envs, 1))
            # pos_err = target_pos - self.rigid_body_states[:, self.hand_base_rigid_body_index, 0:3].clone()
            pos_err = self.actions[:, 3:6] * 0.05
			# target_rot = actions[:, 3:7] / torch.sqrt((actions[:, 3:7]**2).sum(dim=-1)+1e-8).view(-1, 1)
            target_euler = to_torch([0.0, 1.571, 0], device=self.device).repeat((self.num_envs, 1))
            target_rot = quat_from_euler_xyz(target_euler[:, 0], target_euler[:, 1], target_euler[:, 2])
            rot_err = orientation_error(target_rot, self.rigid_body_states[:, self.hand_base_rigid_body_index, 3:7].clone())
            # rot_err = to_torch([0.0, 0.0, 0], device=self.device).repeat((self.num_envs, 1))
            # rot_err[:, 2] = self.actions[:, 6] * 0.05

            # self.add_debug_lines(self.envs[0], target_pos[0], target_rot[0])
            dpose = torch.cat([pos_err, rot_err], -1).unsqueeze(-1)
            delta = control_ik(self.jacobian_tensor[:, self.hand_base_rigid_body_index - 1, :, :-2], self.device, dpose, self.num_envs)
            targets = self.prev_targets[:, :7] + delta[:, :7]

            self.cur_targets[:, :7] = tensor_clamp(targets,
                                                    self.arm_hand_dof_lower_limits[:7],
                                                    self.arm_hand_dof_upper_limits[:7])
            # print(self.arm_hand_dof_pos[0, :7])

        self.prev_targets[:, :] = self.cur_targets[:, :]
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.cur_targets))

        if self.force_scale > 0.0:
            self.rb_forces *= torch.pow(self.force_decay, self.dt / self.force_decay_interval)

            # apply new forces
            force_indices = (torch.rand(self.num_envs, device=self.device) < self.random_force_prob).nonzero()
            self.rb_forces[force_indices, self.object_rb_handles, :] = torch.randn(
                self.rb_forces[force_indices, self.object_rb_handles, :].shape, device=self.device) * self.object_rb_masses * self.force_scale

            self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self.rb_forces), None, gymapi.LOCAL_SPACE)

    def post_physics_step(self):
        self.progress_buf += 1
        self.randomize_buf += 1

        self.compute_observations()
        self.compute_reward(self.actions)

        if self.viewer and self.debug_viz:
            # draw axes on target object
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            for i in range(self.num_envs):
                self.add_debug_lines(self.envs[i], self.segmentation_target_pos[i], self.segmentation_target_rot[i])
                self.add_debug_lines(self.envs[i], self.hand_base_pos[i], self.hand_base_rot[i])

    def add_debug_lines(self, env, pos, rot):
        posx = (pos + quat_apply(rot, to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
        posy = (pos + quat_apply(rot, to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
        posz = (pos + quat_apply(rot, to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

        p0 = pos.cpu().numpy()
        self.gym.add_lines(self.viewer, env, 1, [p0[0], p0[1], p0[2], posx[0], posx[1], posx[2]], [0.85, 0.1, 0.1])
        self.gym.add_lines(self.viewer, env, 1, [p0[0], p0[1], p0[2], posy[0], posy[1], posy[2]], [0.1, 0.85, 0.1])
        self.gym.add_lines(self.viewer, env, 1, [p0[0], p0[1], p0[2], posz[0], posz[1], posz[2]], [0.1, 0.1, 0.85])

    def camera_visulization(self, env_id=0, is_depth_image=False):
        if is_depth_image:
            camera_depth_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[env_id], self.cameras[env_id], gymapi.IMAGE_DEPTH)
            torch_depth_tensor = gymtorch.wrap_tensor(camera_depth_tensor)
            torch_depth_tensor = torch.clamp(torch_depth_tensor, -1, 1)
            torch_depth_tensor = scale(torch_depth_tensor, to_torch([0], dtype=torch.float, device=self.device),
                                                         to_torch([256], dtype=torch.float, device=self.device))
            camera_image = torch_depth_tensor.cpu().numpy()
            camera_image = Im.fromarray(camera_image)
        
        else:
            camera_rgba_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[env_id], self.cameras[env_id], gymapi.IMAGE_COLOR)
            torch_rgba_tensor = gymtorch.wrap_tensor(camera_rgba_tensor)
            camera_image = torch_rgba_tensor.cpu().numpy()
            camera_image = cv2.cvtColor(camera_image, cv2.COLOR_BGR2RGB)
        
        return camera_image

    def camera_segmentation_visulization(self, camera_tensors, camera_seg_tensors, segmentation_id=0, env_id=0, is_depth_image=False):
        torch_rgba_tensor = camera_tensors[env_id]
        torch_seg_tensor = camera_seg_tensors[env_id]
        torch_rgba_tensor[torch_seg_tensor != self.segmentation_id] = 0

        camera_image = torch_rgba_tensor.cpu().numpy()
        camera_image = cv2.cvtColor(camera_image, cv2.COLOR_BGR2RGB)

        return camera_image

    def compute_emergence_reward(self, camera_tensors, camera_seg_tensors, segmentation_id=0):
        for i in range(self.num_envs):
            torch_seg_tensor = camera_seg_tensors[i]
            self.emergence_pixel[i] = torch_seg_tensor[torch_seg_tensor == segmentation_id].shape[0]

        self.emergence_reward = (self.emergence_pixel - self.last_emergence_pixel)
        self.last_emergence_pixel = self.emergence_pixel.clone()

#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def compute_hand_reward(
    spin_coef, rew_buf, reset_buf, reset_goal_buf, progress_buf, successes, consecutive_successes,
    max_episode_length: float, object_pos, object_rot, object_angvel, target_pos, target_rot, segmentation_target_pos, hand_base_pos, emergence_reward, arm_hand_ff_pos, arm_hand_rf_pos, arm_hand_mf_pos, arm_hand_th_pos,
    dist_reward_scale: float, rot_reward_scale: float, rot_eps: float,
    actions, action_penalty_scale: float,
    success_tolerance: float, reach_goal_bonus: float, fall_dist: float,
    fall_penalty: float, rotation_id: int, max_consecutive_successes: int, av_factor: float, ignore_z_rot: bool
):
    # Distance from the hand to the object
    # goal_dist = torch.norm(hand_base_pos - segmentation_target_pos, p=2, dim=-1)
    # dist_rew = goal_dist * dist_reward_scale

    arm_hand_finger_dist = (torch.norm(segmentation_target_pos - arm_hand_ff_pos, p=2, dim=-1) + torch.norm(segmentation_target_pos - arm_hand_mf_pos, p=2, dim=-1)
                            + torch.norm(segmentation_target_pos - arm_hand_rf_pos, p=2, dim=-1) + torch.norm(segmentation_target_pos - arm_hand_th_pos, p=2, dim=-1))
    dist_rew = arm_hand_finger_dist * dist_reward_scale

    action_penalty = torch.sum(actions ** 2, dim=-1)

    # Total reward is: position distance + orientation alignment + action regularization + success bonus + fall penalty
    # emergence_reward = torch.where(arm_hand_finger_dist < 2, 
    #                     torch.where(progress_buf >= max_episode_length - 1, emergence_reward, torch.zeros_like(emergence_reward)), torch.zeros_like(emergence_reward))
    emergence_reward = torch.where(arm_hand_finger_dist < 2, emergence_reward, torch.zeros_like(emergence_reward))
        
    reward = dist_rew + emergence_reward
    print(dist_rew[0])
    print(emergence_reward[0])
    # print(rot_rew[0])
    # # print(spin_reward[0])
    # print("----finish----")

    # Fall penalty: distance to the goal is larger than a threshold
    # Check env termination conditions, including maximum success number
    resets = torch.where(arm_hand_finger_dist <= -1, torch.ones_like(reset_buf), reset_buf)
    # resets = torch.where(goal_resets == 1, torch.ones_like(resets), resets)
    # resets = torch.where(goal_resets == 1, torch.ones_like(resets), resets)

    timed_out = progress_buf >= max_episode_length - 1
    resets = torch.where(timed_out, torch.ones_like(resets), resets)

    # Apply penalty for not reaching the goal
    if max_consecutive_successes > 0:
        reward = torch.where(timed_out, reward + 0.5 * fall_penalty, reward)

    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(successes * resets.float())

    cons_successes = torch.where(num_resets > 0, av_factor*finished_cons_successes/num_resets + (1.0 - av_factor)*consecutive_successes, consecutive_successes)

    return reward, resets, reset_goal_buf, progress_buf, successes, cons_successes


@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(quat_from_angle_axis(rand0 * np.pi, x_unit_tensor),
                    quat_from_angle_axis(rand1 * np.pi, y_unit_tensor))


@torch.jit.script
def randomize_rotation_pen(rand0, rand1, max_angle, x_unit_tensor, y_unit_tensor, z_unit_tensor):
    rot = quat_mul(quat_from_angle_axis(0.5 * np.pi + rand0 * max_angle, x_unit_tensor),
                   quat_from_angle_axis(rand0 * np.pi, z_unit_tensor))
    return rot

def orientation_error(desired, current):
	cc = quat_conjugate(current)
	q_r = quat_mul(desired, cc)
	return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)

def control_ik(j_eef, device, dpose, num_envs):
	# Set controller parameters
	# IK params
    damping = 0.05
    # solve damped least squares
    j_eef_T = torch.transpose(j_eef, 1, 2)
    lmbda = torch.eye(6, device=device) * (damping ** 2)
    u = (j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda) @ dpose).view(num_envs, -1)
    return u