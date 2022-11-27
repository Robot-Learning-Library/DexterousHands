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

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *
from tasks.hand_base.base_task import BaseTask

from tasks.hand_base.vec_task import VecTask

from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

class ContactSLAMer(nn.Module):
    def __init__(self, contact_size, output_pose_size) :
        super(ContactSLAMer, self).__init__()
        self.linear1 = nn.Linear(contact_size, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, 64)
        self.output_layer = nn.Linear(64, output_pose_size)

    def forward(self, contact):
        x = F.relu(self.linear1(contact))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        output_pose = self.output_layer(x)
        return output_pose, x

class TemporaryGrad(object):
    def __enter__(self):
        self.prev = torch.is_grad_enabled()
        torch.set_grad_enabled(True)

    def __exit__(self, exc_type, exc_value, traceback):
        torch.set_grad_enabled(self.prev)

class AllegroArmOrientationLatentSpace(BaseTask):

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
            "normal": "urdf/xarm6/xarm6_allegro_left_fsr.urdf",
            "large":  "urdf/xarm6/xarm6_allegro_left_fsr_large.urdf"
        }
        self.asset_files_dict = {
            "block": "urdf/objects/cube_multicolor.urdf",
            "egg": "mjcf/open_ai_assets/hand/egg.xml",
            "pen": "mjcf/open_ai_assets/hand/pen.xml"
        }

        if "asset" in self.cfg["env"]:
            self.asset_files_dict["block"] = self.cfg["env"]["asset"].get("assetFileNameBlock", self.asset_files_dict["block"])
            self.asset_files_dict["egg"] = self.cfg["env"]["asset"].get("assetFileNameEgg", self.asset_files_dict["egg"])
            self.asset_files_dict["pen"] = self.cfg["env"]["asset"].get("assetFileNamePen", self.asset_files_dict["pen"])

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
        # 11, 13, 16, 20, 22, 24, 27, 29, 32, 36, 39, 40
        # self.contact_sensor_names = ["link_1.0", "link_2.0", "link_3.0_tip",
        #                              "link_5.0", "link_6.0", "link_7.0_tip", "link_9.0",
        #                              "link_10.0", "link_11.0_tip", "link_14.0", "link_15.0",
        #                              "link_15.0_tip"]
        self.stack_obs = 5
        self.num_obs_dict = {
            "full_no_vel": 50,
            "full": 72,
            "full_state": 88,
            "full_contact": 90,
            "partial_contact": 75 + 64
        }

        self.up_axis = 'z'

        self.use_vel_obs = False
        self.fingertip_obs = True
        self.asymmetric_obs = self.cfg["env"]["asymmetric_observations"]

        num_states = 0
        if self.asymmetric_obs:
            num_states = 95 + 64

        self.one_frame_num_obs = self.num_obs_dict[self.obs_type]
        self.one_frame_num_states = num_states
        self.cfg["env"]["numObservations"] = self.num_obs_dict[self.obs_type] * self.stack_obs
        self.cfg["env"]["numStates"] = num_states * self.stack_obs
        self.cfg["env"]["numActions"] = 16

        self.cfg["device_type"] = device_type
        self.cfg["device_id"] = device_id
        self.cfg["headless"] = headless
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

        # Zhaoheng: Add the contact!
        contact_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # Zhaoheng: Contact.
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.arm_hand_default_dof_pos = torch.zeros(self.num_arm_hand_dofs, dtype=torch.float, device=self.device)
        self.arm_hand_default_dof_pos[:6] = torch.tensor([0, -0.785, 0, 3.14, 2.4, -1.57], dtype=torch.float, device=self.device)
        self.arm_hand_default_dof_pos[6:] = to_torch([0.0, -0.174, 0.785, 0.785,
                                            0.08, -0.08, 0.872, 0.785, 0.0,  -0.174, 0.785, 0.785, 0.872,  0.436, 0.261, 0.785], dtype=torch.float, device=self.device)

        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.arm_hand_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_arm_hand_dofs]
        self.arm_hand_dof_pos = self.arm_hand_dof_state[..., 0]
        self.arm_hand_dof_vel = self.arm_hand_dof_state[..., 1]

        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.num_bodies = self.rigid_body_states.shape[1]

        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)

        # Zhaoheng: Contact.
        self.contact_tensor = gymtorch.wrap_tensor(contact_tensor).view(self.num_envs, -1)
        print("Contact Tensor Dimension", self.contact_tensor.shape)

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
        self.predict_pose = self.root_state_tensor[self.predict_object_indices, 0:7].clone()

        self.state_buf_stack_frames = []
        self.obs_buf_stack_frames = []
        self.contact_obs_buf_stack_frames = []
        for i in range(self.stack_obs):
            self.obs_buf_stack_frames.append(torch.zeros_like(self.obs_buf[:, 0:self.one_frame_num_obs]))
            self.state_buf_stack_frames.append(torch.zeros_like(self.obs_buf[:, 0:self.one_frame_num_states]))
            self.contact_obs_buf_stack_frames.append(torch.zeros_like(self.obs_buf[:, 0:self.one_frame_num_obs]))

        self.contact_slamer = ContactSLAMer(self.num_obs, 7).to(self.device)
        for param in self.contact_slamer.parameters():
            param.requires_grad_(True)
        self.contact_optimizer = optim.Adam(self.contact_slamer.parameters(), lr=0.0003)
        self.contact_obs_buf = self.obs_buf.clone()
        # self.contact_slamer.load_state_dict(torch.load("/home/jmji/isaacgym_rl/contact_slamer/ArmRotationLatentVectorStudent/model_5000.pt", map_location='cuda:0'))
        # self.contact_slamer.load_state_dict(torch.load("/home/jmji/isaacgym_rl/contact_slamer/ArmRotationLatentVectorTeacher/model_460000.pt", map_location='cuda:0'))
        # self.contact_slamer.train()

    def create_sim(self):
        self.dt = self.sim_params.dt
        self.up_axis_idx = self.set_sim_params_up_axis(self.sim_params, self.up_axis)

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
        asset_options.vhacd_params.resolution = 3000000
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
        self.actuated_dof_indices = [i for i in range(6, self.num_arm_hand_dofs)]

        self.arm_hand_dof_lower_limits = []
        self.arm_hand_dof_upper_limits = []
        self.arm_hand_dof_default_pos = []
        self.arm_hand_dof_default_vel = []

        robot_lower_qpos = []
        robot_upper_qpos = []

        robot_dof_props = self.gym.get_asset_dof_properties(arm_hand_asset)

        # Zhaoheng. This part is very important (damping)
        for i in range(22):
            robot_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
            if i < 6:
                robot_dof_props['velocity'][i] = 1.0
            else:
                robot_dof_props['velocity'][i] = 3.0
            robot_dof_props['friction'][i] = 0.02
            robot_dof_props['stiffness'][i] = 3
            robot_dof_props['armature'][i] = 0.001

            if i < 6:
                robot_dof_props['damping'][i] = 100.0
            else:
                robot_dof_props['damping'][i] = 0.2
            robot_lower_qpos.append(robot_dof_props['lower'][i])
            robot_upper_qpos.append(robot_dof_props['upper'][i])

        self.actuated_dof_indices = to_torch(self.actuated_dof_indices, dtype=torch.long, device=self.device)
        self.arm_hand_dof_lower_limits = to_torch(robot_lower_qpos, device=self.device)
        self.arm_hand_dof_upper_limits = to_torch(robot_upper_qpos, device=self.device)
        self.arm_hand_dof_lower_qvel = to_torch(-robot_dof_props["velocity"], device=self.device)
        self.arm_hand_dof_upper_qvel = to_torch(robot_dof_props["velocity"], device=self.device)


        # Set up default arm position.
        # Zhaoheng: We can set this to different positions...
        default_arm_pos = [0, -0.785, 0, 3.14, 1.57, -1.57]

        for i in range(self.num_arm_hand_dofs):
            if i < 6:
                self.arm_hand_dof_default_pos.append(default_arm_pos[i])
            else:
                self.arm_hand_dof_default_pos.append(0.0)
            self.arm_hand_dof_default_vel.append(0.0)

        self.arm_hand_dof_default_pos = to_torch(self.arm_hand_dof_default_pos, device=self.device)
        self.arm_hand_dof_default_vel = to_torch(self.arm_hand_dof_default_vel, device=self.device)

        # load manipulated object and goal assets
        object_asset_options = gymapi.AssetOptions()
        # object_asset_options.disable_gravity = True
        object_asset = self.gym.load_asset(self.sim, asset_root, object_asset_file, object_asset_options)

        object_asset_options.disable_gravity = True
        goal_asset = self.gym.load_asset(self.sim, asset_root, object_asset_file, object_asset_options)
        predict_object_asset = self.gym.load_asset(self.sim, asset_root, object_asset_file, object_asset_options)

        # Put objects in the scene.
        arm_hand_start_pose = gymapi.Transform()
        arm_hand_start_pose.p = gymapi.Vec3(0, 0.0, 0.0)
        arm_hand_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        object_start_pose = gymapi.Transform()
        object_start_pose.p = gymapi.Vec3()

        pose_dx, pose_dy, pose_dz = 0.37, 0.0, 0.49
        object_start_pose.p.x = arm_hand_start_pose.p.x + pose_dx
        object_start_pose.p.y = arm_hand_start_pose.p.y + pose_dy
        object_start_pose.p.z = arm_hand_start_pose.p.z + pose_dz

        if self.object_type == "pen":
            object_start_pose.p.z = arm_hand_start_pose.p.z + 0.02

        self.goal_displacement = gymapi.Vec3(-0.2, -0.06, 0.12)
        self.goal_displacement_tensor = to_torch(
            [self.goal_displacement.x, self.goal_displacement.y, self.goal_displacement.z], device=self.device)
        goal_start_pose = gymapi.Transform()
        goal_start_pose.p = object_start_pose.p + self.goal_displacement

        goal_start_pose.p.z -= 0.04

        self.predict_object_displacement = gymapi.Vec3(-0.1, 0.12, 0.12)
        self.predict_object_displacement_tensor = to_torch(
            [self.predict_object_displacement.x, self.predict_object_displacement.y, self.predict_object_displacement.z], device=self.device)
        predict_object_start_pose = gymapi.Transform()
        predict_object_start_pose.p = object_start_pose.p + self.predict_object_displacement
        predict_object_start_pose.r = object_start_pose.r

        predict_object_start_pose.p.z -= 0.04

        # compute aggregate size
        max_agg_bodies = self.num_arm_hand_bodies + 2 + 1
        max_agg_shapes = self.num_arm_hand_shapes + 2 + 1

        self.arm_hands = []
        self.envs = []

        self.object_init_state = []
        self.hand_start_states = []

        self.hand_indices = []
        self.fingertip_indices = []
        self.object_indices = []
        self.goal_object_indices = []
        self.predict_object_indices = []

        # self.fingertip_handles = [self.gym.find_asset_rigid_body_index(shadow_hand_asset, name) for name in self.fingertips]

        arm_hand_rb_count = self.gym.get_asset_rigid_body_count(arm_hand_asset)
        object_rb_count = self.gym.get_asset_rigid_body_count(object_asset)
        self.object_rb_handles = list(range(arm_hand_rb_count, arm_hand_rb_count + object_rb_count))

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

            # create fingertip force-torque sensors
            # if self.obs_type == "full_state" or self.asymmetric_obs:
            #     for ft_handle in self.fingertip_handles:
            #         env_sensors = []
            #         env_sensors.append(self.gym.create_force_sensor(env_ptr, ft_handle, sensor_pose))
            #         self.sensors.append(env_sensors)

            #     self.gym.enable_actor_dof_force_sensors(env_ptr, shadow_hand_actor)

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

            # add predict_object
            predict_object_handle = self.gym.create_actor(env_ptr, predict_object_asset, predict_object_start_pose, "predict_object_object", i + self.num_envs, 0, 0)
            predict_object_object_idx = self.gym.get_actor_index(env_ptr, predict_object_handle, gymapi.DOMAIN_SIM)
            self.predict_object_indices.append(predict_object_object_idx)

            arm_hand_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, arm_hand_actor)
            for i, prop in enumerate(arm_hand_shape_props):
                if i in [10, 12, 15, 19, 21, 23, 26, 28, 31, 35, 38, 39]:
                    prop.filter = 1
            self.gym.set_actor_rigid_shape_properties(env_ptr, arm_hand_actor, arm_hand_shape_props)

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

            self.envs.append(env_ptr)
            self.arm_hands.append(arm_hand_actor)

        # Acquire specific links.
        palm_handles = self.gym.find_actor_rigid_body_handle(env_ptr, arm_hand_actor, self.palm_name)
        self.palm_indices = to_torch(palm_handles, dtype=torch.int64)

        sensor_handles = [self.gym.find_actor_rigid_body_handle(env_ptr, arm_hand_actor, sensor_name) for sensor_name in
                          self.contact_sensor_names]
        self.sensor_handle_indices = to_torch(sensor_handles, dtype=torch.int64)

        print("PALM", self.palm_indices, self.sensor_handle_indices)

        object_rb_props = self.gym.get_actor_rigid_body_properties(env_ptr, object_handle)
        self.object_rb_masses = [prop.mass for prop in object_rb_props]

        self.object_init_state = to_torch(self.object_init_state, device=self.device, dtype=torch.float).view(self.num_envs, 13)
        self.goal_states = self.object_init_state.clone()
        self.goal_states[:, self.up_axis_idx] -= 0.02
        self.goal_init_state = self.goal_states.clone()
        self.hand_start_states = to_torch(self.hand_start_states, device=self.device).view(self.num_envs, 13)

        # self.fingertip_handles = to_torch(self.fingertip_handles, dtype=torch.long, device=self.device)
        self.object_rb_handles = to_torch(self.object_rb_handles, dtype=torch.long, device=self.device)
        self.object_rb_masses = to_torch(self.object_rb_masses, dtype=torch.float, device=self.device)

        self.hand_indices = to_torch(self.hand_indices, dtype=torch.long, device=self.device)
        self.object_indices = to_torch(self.object_indices, dtype=torch.long, device=self.device)
        self.goal_object_indices = to_torch(self.goal_object_indices, dtype=torch.long, device=self.device)
        self.predict_object_indices = to_torch(self.predict_object_indices, dtype=torch.long, device=self.device)

    def compute_reward(self, actions):
        self.rew_buf[:], self.reset_buf[:], self.reset_goal_buf[:], self.progress_buf[:], self.successes[:], self.consecutive_successes[:] = compute_hand_reward(
            torch.tensor(self.spin_coef).to(self.device), self.rew_buf, self.reset_buf, self.reset_goal_buf, self.progress_buf, self.successes, self.consecutive_successes,
            self.max_episode_length, self.object_pos, self.object_rot, self.object_angvel, self.goal_pos, self.goal_rot, self.thumb_pos, self.predict_pose, self.rot_loss,
            self.dist_reward_scale, self.rot_reward_scale, self.rot_eps, self.actions, self.action_penalty_scale,
            self.success_tolerance, self.reach_goal_bonus, self.fall_dist, self.fall_penalty, self.rotation_id,
            self.max_consecutive_successes, self.av_factor, (self.object_type == "pen")
        )

        self.extras['consecutive_successes'] = self.consecutive_successes.mean()
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
        # Zhaoheng.
        self.gym.refresh_net_contact_force_tensor(self.sim)
        # if self.obs_type == "full_state" or self.asymmetric_obs:
        #     self.gym.refresh_force_sensor_tensor(self.sim)
        #     self.gym.refresh_dof_force_tensor(self.sim)

        self.object_pose = self.root_state_tensor[self.object_indices, 0:7]
        self.object_pos = self.root_state_tensor[self.object_indices, 0:3]
        self.object_rot = self.root_state_tensor[self.object_indices, 3:7]
        self.object_linvel = self.root_state_tensor[self.object_indices, 7:10]
        self.object_angvel = self.root_state_tensor[self.object_indices, 10:13]

        self.goal_pose = self.goal_states[:, 0:7]
        self.goal_pos = self.goal_states[:, 0:3]
        self.goal_rot = self.goal_states[:, 3:7]

        # self.fingertip_state = self.rigid_body_states[:, self.fingertip_handles][:, :, 0:13]
        self.thumb_pos = self.rigid_body_states[:, 24, 0:3]
        self.thumb_rot = self.rigid_body_states[:, 24, 3:7]

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
            self.states_buf[:, 0:16] = unscale(self.arm_hand_dof_pos[:, 6:22],
                                                                self.arm_hand_dof_lower_limits[6:22],
                                                                self.arm_hand_dof_upper_limits[6:22])
            self.states_buf[:, 16:32] = self.vel_obs_scale * self.arm_hand_dof_vel[:, 6:22]

            self.states_buf[:, 32:35] = self.object_linvel
            self.states_buf[:, 35:38] = self.vel_obs_scale * self.object_angvel

            self.states_buf[:, 38:45] = self.goal_pose
            self.states_buf[:, 45:49] = quat_mul(self.object_rot, quat_conjugate(self.goal_rot))

            self.states_buf[:, 49:56] = self.goal_pose
            self.states_buf[:, 56:72] = self.actions
            contacts = self.contact_tensor.reshape(-1, 49, 3)  # 39+27
            contacts = contacts[:, self.sensor_handle_indices, :] # 12
            contacts = torch.norm(contacts, dim=-1)
            contacts = torch.where(contacts >= 2, 1.0, 0.0)

            self.states_buf[:, 72:84] = contacts
            self.states_buf[:, 88:95] = self.object_pose
            self.states_buf[:, 95:159] = self.pose_latent_vector.detach().clone()

            for i in range(len(self.state_buf_stack_frames) - 1):
                self.states_buf[:, (i+1) * self.one_frame_num_states:(i+2) * self.one_frame_num_states] = self.state_buf_stack_frames[i]
                self.state_buf_stack_frames[i] = self.states_buf[:, (i) * self.one_frame_num_states:(i+1) * self.one_frame_num_states].clone()

    def compute_contact_observations(self, full_contact=True):
        self.obs_buf[:, 0:16] = unscale(self.arm_hand_dof_pos[:, 6:22],
                                                            self.arm_hand_dof_lower_limits[6:22],
                                                            self.arm_hand_dof_upper_limits[6:22])
        self.obs_buf[:, 16:23] = self.goal_pose
        self.obs_buf[:, 23:39] = self.actions
        contacts = self.contact_tensor.reshape(-1, 49, 3)  # 39+27
        contacts = contacts[:, self.sensor_handle_indices, :] # 12
        contacts = torch.norm(contacts, dim=-1)
        # print(contacts[0])
        contacts = torch.where(contacts >= 2, 1.0, 0.0)
        self.obs_buf[:, 45:57] = contacts

        with TemporaryGrad():
            self.predict_pose, self.pose_latent_vector = self.predict_contact_pose(self.contact_slamer, self.contact_obs_buf)
            self.update_contact_slamer(self.predict_pose)

        # self.root_state_tensor[self.predict_object_indices, 0:3] = self.predict_pose[:, 0:3].detach() + self.predict_object_displacement_tensor
        # self.root_state_tensor[self.predict_object_indices, 3:7] = self.predict_pose[:, 3:7].detach()

        # self.obs_buf[:, 61:68] = self.object_pose
        self.obs_buf[:, 61:68] = self.predict_pose.detach().clone()
        self.obs_buf[:, 68:75] = self.object_pose_for_open_loop

        self.contact_obs_buf[:, :75] = self.obs_buf[:, :75].clone()
        self.contact_obs_buf[:, 61:68] = self.predict_pose.detach().clone()

        self.obs_buf[:, 75:139] = self.pose_latent_vector.detach().clone()
        self.contact_obs_buf[:, 75:139] = self.pose_latent_vector.detach().clone()

        for i in range(len(contacts[0])):
            if contacts[0][i] == 1.0:
                self.gym.set_rigid_body_color(
                            self.envs[0], self.hand_indices[0], self.sensor_handle_indices[i], gymapi.MESH_VISUAL, gymapi.Vec3(0.3, 1, 0.3))
            else:
                self.gym.set_rigid_body_color(
                            self.envs[0], self.hand_indices[0], self.sensor_handle_indices[i], gymapi.MESH_VISUAL, gymapi.Vec3(1, 1, 1))

        for i in range(len(self.obs_buf_stack_frames) - 1):
            self.obs_buf[:, (i+1) * self.one_frame_num_obs:(i+2) * self.one_frame_num_obs] = self.obs_buf_stack_frames[i]
            self.obs_buf_stack_frames[i] = self.obs_buf[:, (i) * self.one_frame_num_obs:(i+1) * self.one_frame_num_obs].clone()

            self.contact_obs_buf[:, (i+1) * self.one_frame_num_obs:(i+2) * self.one_frame_num_obs] = self.contact_obs_buf_stack_frames[i]
            self.contact_obs_buf_stack_frames[i] = self.contact_obs_buf[:, (i) * self.one_frame_num_obs:(i+1) * self.one_frame_num_obs].clone()

    def predict_contact_pose(self, contact_slamer, contact_buf):
        predict_pose, pose_latent_vector = contact_slamer(contact_buf)
        return predict_pose, pose_latent_vector

    def update_contact_slamer(self, predict_pose):
        self.rot_loss = torch.norm(self.object_rot - predict_pose[:, 3:7], p=2, dim=-1).mean()
        self.pos_loss = torch.norm(self.object_pos - predict_pose[:, 0:3], p=2, dim=-1).mean()
        loss = self.rot_loss + self.pos_loss
        self.contact_optimizer.zero_grad()
        loss.backward()
        self.contact_optimizer.step()
        self.extras['rot_loss'] = self.rot_loss
        self.extras['pos_loss'] = self.pos_loss

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
        self.root_state_tensor[self.predict_object_indices[env_ids], 0:3] = self.predict_pose[env_ids][:, 0:3].detach() + self.predict_object_displacement_tensor
        self.root_state_tensor[self.predict_object_indices[env_ids], 3:7] = self.predict_pose[env_ids][:, 3:7].detach()

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
        self.root_state_tensor[self.predict_object_indices[env_ids], 0:3] = self.predict_pose[env_ids][:, 0:3].detach() + self.predict_object_displacement_tensor
        self.root_state_tensor[self.predict_object_indices[env_ids], 3:7] = self.predict_pose[env_ids][:, 3:7].detach()

        object_indices = torch.unique(torch.cat([self.object_indices[env_ids],
                                                 self.goal_object_indices[env_ids],
                                                 self.goal_object_indices[goal_env_ids],
                                                 self.predict_object_indices[env_ids]]).to(torch.int32))

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
        self.arm_hand_dof_pos[env_ids, 6:22] = pos[6:22]
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

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.successes[env_ids] = 0

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
            self.cur_targets[:, self.actuated_dof_indices] = scale(self.actions,
                                                                   self.arm_hand_dof_lower_limits[self.actuated_dof_indices],
                                                                   self.arm_hand_dof_upper_limits[self.actuated_dof_indices])
            self.cur_targets[:, self.actuated_dof_indices] = self.act_moving_average * self.cur_targets[:,
                                                                                                        self.actuated_dof_indices] + (1.0 - self.act_moving_average) * self.prev_targets[:, self.actuated_dof_indices]
            self.cur_targets[:, self.actuated_dof_indices] = tensor_clamp(self.cur_targets[:, self.actuated_dof_indices],
                                                                          self.arm_hand_dof_lower_limits[self.actuated_dof_indices],
                                                                          self.arm_hand_dof_upper_limits[self.actuated_dof_indices])

        self.prev_targets[:, self.actuated_dof_indices] = self.cur_targets[:, self.actuated_dof_indices]
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.cur_targets))

        self.root_state_tensor[self.predict_object_indices, 0:3] = self.predict_pose[:, 0:3].detach() + self.predict_object_displacement_tensor
        self.root_state_tensor[self.predict_object_indices, 3:7] = self.predict_pose[:, 3:7].detach()
        object_indices = torch.unique(torch.cat([self.object_indices,
                                                 self.goal_object_indices,
                                                 self.predict_object_indices]).to(torch.int32))

        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_state_tensor),
                                                     gymtorch.unwrap_tensor(object_indices.to(torch.int32)), len(object_indices.to(torch.int32)))
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
                self.add_debug_lines(self.envs[i], self.goal_init_state[i][0:3], self.goal_init_state[i][3:7])
                self.add_debug_lines(self.envs[i], self.object_init_state[i][0:3], self.object_init_state[i][3:7])

    def add_debug_lines(self, env, pos, rot):
        posx = (pos + quat_apply(rot, to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
        posy = (pos + quat_apply(rot, to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
        posz = (pos + quat_apply(rot, to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

        p0 = pos.cpu().numpy()
        self.gym.add_lines(self.viewer, env, 1, [p0[0], p0[1], p0[2], posx[0], posx[1], posx[2]], [0.85, 0.1, 0.1])
        self.gym.add_lines(self.viewer, env, 1, [p0[0], p0[1], p0[2], posy[0], posy[1], posy[2]], [0.1, 0.85, 0.1])
        self.gym.add_lines(self.viewer, env, 1, [p0[0], p0[1], p0[2], posz[0], posz[1], posz[2]], [0.1, 0.1, 0.85])
#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_hand_reward(
    spin_coef, rew_buf, reset_buf, reset_goal_buf, progress_buf, successes, consecutive_successes,
    max_episode_length: float, object_pos, object_rot, object_angvel, target_pos, target_rot, thumb_pos, predict_pose, rot_loss,
    dist_reward_scale: float, rot_reward_scale: float, rot_eps: float,
    actions, action_penalty_scale: float,
    success_tolerance: float, reach_goal_bonus: float, fall_dist: float,
    fall_penalty: float, rotation_id: int, max_consecutive_successes: int, av_factor: float, ignore_z_rot: bool
):
    # Distance from the hand to the object
    goal_dist = torch.norm(object_pos - target_pos, p=2, dim=-1)

    if ignore_z_rot:
        success_tolerance = 2.0 * success_tolerance

    # Orientation alignment for the cube in hand and goal cube
    quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
    rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))

    dist_rew = goal_dist * dist_reward_scale
    rot_rew = 1.0/(torch.abs(rot_dist) + rot_eps) * rot_reward_scale

    predict_quat_diff = quat_mul(predict_pose[:, 3:7], quat_conjugate(target_rot))
    predict_rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(predict_quat_diff[:, 0:3], p=2, dim=-1), max=1.0))
    predict_rot_rew = 1.0/(torch.abs(predict_rot_dist) + rot_eps) * rot_reward_scale * 0.5

    action_penalty = torch.sum(actions ** 2, dim=-1)

    # spin_reward = torch.log(torch.norm(object_angvel[:, 0:3] - quat_diff[:, 0:3], p=2, dim=-1) * 0.25 + 1)
    thumb_rew = (torch.norm(object_pos - thumb_pos, p=2, dim=-1) - 0.05) * -0

    spin_reward = torch.log(torch.norm(object_angvel[:, 0:3], p=2, dim=-1) * 0.1 + 1)
    # Total reward is: position distance + orientation alignment + action regularization + success bonus + fall penalty
    reward = dist_rew + rot_rew + action_penalty * action_penalty_scale + thumb_rew
    # print(dist_rew[0])
    # print(thumb_rew[0])
    # print(rot_rew[0])
    # # print(spin_reward[0])
    # print("----finish----")

    # Find out which envs hit the goal and update successes count
    goal_resets = torch.where(torch.abs(rot_dist) <= success_tolerance, torch.ones_like(reset_goal_buf), reset_goal_buf)
    successes = successes + goal_resets

    # Success bonus: orientation is within `success_tolerance` of goal orientation
    reward = torch.where(goal_resets == 1, reward + reach_goal_bonus, reward)
    # goal_resets = torch.zeros_like(goal_resets)

    # Fall penalty: distance to the goal is larger than a threshold
    reward = torch.where(goal_dist >= fall_dist, reward + fall_penalty, reward)

    # Check env termination conditions, including maximum success number
    resets = torch.where(goal_dist >= fall_dist, torch.ones_like(reset_buf), reset_buf)
    # resets = torch.where(goal_resets == 1, torch.ones_like(resets), resets)
    # resets = torch.where(goal_resets == 1, torch.ones_like(resets), resets)

    if max_consecutive_successes > 0:
        # Reset progress buffer on goal envs if max_consecutive_successes > 0
        progress_buf = torch.where(torch.abs(rot_dist) <= success_tolerance, torch.zeros_like(progress_buf), progress_buf)
        resets = torch.where(successes >= max_consecutive_successes, torch.ones_like(resets), resets)

    timed_out = progress_buf >= max_episode_length - 1
    resets = torch.where(timed_out, torch.ones_like(resets), resets)

    # Apply penalty for not reaching the goal
    if max_consecutive_successes > 0:
        reward = torch.where(timed_out, reward + 0.5 * fall_penalty, reward)

    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(successes * resets.float())

    cons_successes = torch.where(num_resets > 0, av_factor*finished_cons_successes/num_resets + (1.0 - av_factor)*consecutive_successes, consecutive_successes)

    return reward, resets, goal_resets, progress_buf, successes, cons_successes


@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(quat_from_angle_axis(rand0 * np.pi, x_unit_tensor),
                    quat_from_angle_axis(rand1 * np.pi, y_unit_tensor))


@torch.jit.script
def randomize_rotation_pen(rand0, rand1, max_angle, x_unit_tensor, y_unit_tensor, z_unit_tensor):
    rot = quat_mul(quat_from_angle_axis(0.5 * np.pi + rand0 * max_angle, x_unit_tensor),
                   quat_from_angle_axis(rand0 * np.pi, z_unit_tensor))
    return rot
