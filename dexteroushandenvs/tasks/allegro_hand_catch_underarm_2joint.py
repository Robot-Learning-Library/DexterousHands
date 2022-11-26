# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from matplotlib.pyplot import axis
import numpy as np
import os
import random
import torch

from utils.torch_jit_utils import *
# from isaacgym.torch_utils import *

from tasks.hand_base.base_task import BaseTask
from isaacgym import gymtorch
from isaacgym import gymapi

import matplotlib.pyplot as plt
from PIL import Image as Im
import cv2

class AllegroHandCatchUnderarm2Joint(BaseTask):
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

        self.allegro_hand_dof_speed_scale = self.cfg["env"]["dofSpeedScale"]
        self.use_relative_control = self.cfg["env"]["useRelativeControl"]
        self.act_moving_average = self.cfg["env"]["actionsMovingAverage"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.reset_time = self.cfg["env"].get("resetTime", -1.0)
        self.print_success_stat = self.cfg["env"]["printNumSuccesses"]
        self.max_consecutive_successes = self.cfg["env"]["maxConsecutiveSuccesses"]
        self.av_factor = self.cfg["env"].get("averFactor", 0.01)
        print("Averaging factor: ", self.av_factor)

        control_freq_inv = self.cfg["env"].get("controlFrequencyInv", 1)
        if self.reset_time > 0.0:
            self.max_episode_length = int(round(self.reset_time/(control_freq_inv * self.sim_params.dt)))
            print("Reset time: ", self.reset_time)
            print("New episode length: ", self.max_episode_length)

        self.object_type = self.cfg["env"]["objectType"]
        assert self.object_type in ["block", "egg", "pen", "ycb/banana", "ycb/can", "ycb/mug", "ycb/brick"]

        self.ignore_z = (self.object_type == "pen")

        self.asset_files_dict = {
            "block": "urdf/objects/cube_multicolor.urdf",
            "egg": "mjcf/open_ai_assets/hand/egg.xml",
            "pen": "mjcf/open_ai_assets/hand/pen.xml",
            "ycb/banana": "urdf/ycb/011_banana/011_banana.urdf",
            "ycb/can": "urdf/ycb/010_potted_meat_can/010_potted_meat_can.urdf",
            "ycb/mug": "urdf/ycb/025_mug/025_mug.urdf",
            "ycb/brick": "urdf/ycb/061_foam_brick/061_foam_brick.urdf"
        }

        # can be "openai", "full_no_vel", "full", "full_state"
        self.obs_type = self.cfg["env"]["observationType"]

        if not (self.obs_type in ["openai", "full_no_vel", "full", "full_state"]):
            raise Exception(
                "Unknown type of observations!\nobservationType should be one of: [openai, full_no_vel, full, full_state]")

        print("Obs type:", self.obs_type)

        self.contact_sensor_names = ["link_1.0_fsr", "link_2.0_fsr", "link_3.0_tip_fsr",
                                     "link_5.0_fsr", "link_6.0_fsr", "link_7.0_tip_fsr", "link_9.0_fsr",
                                     "link_10.0_fsr", "link_11.0_tip_fsr", "link_14.0_fsr", "link_15.0_fsr"]
        # self.contact_sensor_names = ["link_1.0", "link_2.0", "link_3.0",
        #                              "link_5.0", "link_6.0", "link_7.0", "link_9.0",
        #                              "link_10.0", "link_11.0", "link_14.0", "link_15.0"]

        self.num_obs_dict = {
            "openai": 42,
            "full_no_vel": 77,
            "full": 157,
            "full_state": 111
        }
        self.num_hand_obs = 157
        self.up_axis = 'z'

        self.use_vel_obs = False
        self.fingertip_obs = True
        self.asymmetric_obs = self.cfg["env"]["asymmetric_observations"]

        num_states = 0
        if self.asymmetric_obs:
            num_states = 215

        self.cfg["env"]["numObservations"] = self.num_obs_dict[self.obs_type]
        self.cfg["env"]["numStates"] = num_states
        if self.is_multi_agent:
            self.num_agents = 2
            self.cfg["env"]["numActions"] = 18
            
        else:
            self.num_agents = 1
            self.cfg["env"]["numActions"] = 36

        self.cfg["device_type"] = device_type
        self.cfg["device_id"] = device_id
        self.cfg["headless"] = headless

        self.enable_camera_sensors = self.cfg["env"]["enableCameraSensors"]

        super().__init__(cfg=self.cfg)

        if self.viewer != None:
            cam_pos = gymapi.Vec3(10.0, 5.0, 1.0)
            cam_target = gymapi.Vec3(6.0, 5.0, 0.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # get gym GPU state tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        contact_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.allegro_hand_default_dof_pos = torch.zeros(self.num_allegro_hand_dofs, dtype=torch.float, device=self.device)
        # self.allegro_hand_default_dof_pos[:6] = torch.tensor([0, 0, -1, 3.14, 0.57, 3.14], dtype=torch.float, device=self.device)
        self.allegro_hand_default_dof_pos[:6] = torch.tensor([0, -0.785, 0, 3.14, 1.57, -1.57], dtype=torch.float, device=self.device)
        self.allegro_hand_default_dof_pos[6:] = to_torch([0.0, -0.174, 0.785, 0.785,
                                            0.0, -0.174, 0.785, 0.785, 0.0, -0.174, 0.785, 0.785, 0.0, -0.174, 0.785, 0.785], dtype=torch.float, device=self.device)

        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.allegro_hand_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_allegro_hand_dofs]
        self.allegro_hand_dof_pos = self.allegro_hand_dof_state[..., 0]
        self.allegro_hand_dof_vel = self.allegro_hand_dof_state[..., 1]

        self.allegro_hand_another_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, self.num_allegro_hand_dofs:self.num_allegro_hand_dofs*2]
        self.allegro_hand_another_dof_pos = self.allegro_hand_another_dof_state[..., 0]
        self.allegro_hand_another_dof_vel = self.allegro_hand_another_dof_state[..., 1]

        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.num_bodies = self.rigid_body_states.shape[1]

        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)
        self.hand_positions = self.root_state_tensor[:, 0:3]
        self.hand_orientations = self.root_state_tensor[:, 3:7]
        self.hand_linvels = self.root_state_tensor[:, 7:10]
        self.hand_angvels = self.root_state_tensor[:, 10:13]
        self.saved_root_tensor = self.root_state_tensor.clone() 

        self.contact_tensor = gymtorch.wrap_tensor(contact_tensor).view(self.num_envs, -1)

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        self.prev_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)

        self.x_unit_tensor = to_torch([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = to_torch([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = to_torch([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

        self.reset_goal_buf = self.reset_buf.clone()
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.consecutive_successes = torch.zeros(1, dtype=torch.float, device=self.device)

        self.av_factor = to_torch(self.av_factor, dtype=torch.float, device=self.device)
        self.object_pose_for_open_loop = torch.zeros_like(self.root_state_tensor[self.object_indices, 0:7])

        self.total_successes = 0
        self.total_resets = 0

    def create_sim(self):
        self.dt = self.sim_params.dt
        self.up_axis_idx = self.set_sim_params_up_axis(self.sim_params, self.up_axis)
        # self.sim_params.physx.max_gpu_contact_pairs = self.sim_params.physx.max_gpu_contact_pairs*10
        
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

        asset_root = "../assets"
        allegro_hand_asset_file = "urdf/xarm6/xarm6_allegro_left_fsr.urdf"
        allegro_hand_another_asset_file = "urdf/xarm6/xarm6_allegro_left_fsr.urdf"

        object_asset_file = self.asset_files_dict[self.object_type]

        # load shadow hand_ asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = False
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.angular_damping = 0.01
        asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
        asset_options.override_com = True
        asset_options.override_inertia = True
        asset_options.vhacd_enabled = True
        asset_options.vhacd_params = gymapi.VhacdParams()
        asset_options.vhacd_params.resolution = 3000000
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        if self.physics_engine == gymapi.SIM_PHYSX:
            asset_options.use_physx_armature = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        allegro_hand_asset = self.gym.load_asset(self.sim, asset_root, allegro_hand_asset_file, asset_options)
        allegro_hand_another_asset = self.gym.load_asset(self.sim, asset_root, allegro_hand_another_asset_file, asset_options)

        self.num_allegro_hand_bodies = self.gym.get_asset_rigid_body_count(allegro_hand_asset)
        self.num_allegro_hand_shapes = self.gym.get_asset_rigid_shape_count(allegro_hand_asset)
        self.num_allegro_hand_dofs = self.gym.get_asset_dof_count(allegro_hand_asset)
        self.num_allegro_hand_actuators = self.gym.get_asset_dof_count(allegro_hand_asset)
        self.num_allegro_hand_tendons = self.gym.get_asset_tendon_count(allegro_hand_asset)

        print("self.num_allegro_hand_bodies: ", self.num_allegro_hand_bodies)
        print("self.num_allegro_hand_shapes: ", self.num_allegro_hand_shapes)
        print("self.num_allegro_hand_dofs: ", self.num_allegro_hand_dofs)
        print("self.num_allegro_hand_actuators: ", self.num_allegro_hand_actuators)
        print("self.num_allegro_hand_tendons: ", self.num_allegro_hand_tendons)

        self.actuated_dof_indices = [i for i in range(16)]

        # set allegro_hand dof properties
        allegro_hand_dof_props = self.gym.get_asset_dof_properties(allegro_hand_asset)
        allegro_hand_another_dof_props = self.gym.get_asset_dof_properties(allegro_hand_another_asset)

        self.allegro_hand_dof_lower_limits = []
        self.allegro_hand_dof_upper_limits = []
        self.allegro_hand_dof_default_pos = []
        self.allegro_hand_dof_default_vel = []
        self.allegro_hand_dof_stiffness = []
        self.allegro_hand_dof_damping = []
        self.allegro_hand_dof_effort = []
        self.sensors = []
        sensor_pose = gymapi.Transform()

        for i in range(self.num_allegro_hand_dofs):
            self.allegro_hand_dof_lower_limits.append(allegro_hand_dof_props['lower'][i])
            self.allegro_hand_dof_upper_limits.append(allegro_hand_dof_props['upper'][i])
            self.allegro_hand_dof_default_pos.append(0.0)
            self.allegro_hand_dof_default_vel.append(0.0)

            allegro_hand_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
            allegro_hand_another_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
            if i < 6:
                allegro_hand_dof_props['velocity'][i] = 1.0
                allegro_hand_another_dof_props['velocity'][i] = 1.0
            else:
                allegro_hand_dof_props['velocity'][i] = 3.0
                allegro_hand_another_dof_props['velocity'][i] = 3.0

            allegro_hand_dof_props['friction'][i] = 0.02
            allegro_hand_dof_props['stiffness'][i] = 3
            allegro_hand_dof_props['armature'][i] = 0.001
            allegro_hand_another_dof_props['friction'][i] = 0.02
            allegro_hand_another_dof_props['stiffness'][i] = 3
            allegro_hand_another_dof_props['armature'][i] = 0.001
            if i < 6:
                allegro_hand_dof_props['damping'][i] = 100
                allegro_hand_another_dof_props['damping'][i] = 100
            else:
                allegro_hand_dof_props['damping'][i] = 0.2
                allegro_hand_another_dof_props['damping'][i] = 0.2

        self.actuated_dof_indices = to_torch(self.actuated_dof_indices, dtype=torch.long, device=self.device)
        self.allegro_hand_dof_lower_limits = to_torch(self.allegro_hand_dof_lower_limits, device=self.device)
        self.allegro_hand_dof_upper_limits = to_torch(self.allegro_hand_dof_upper_limits, device=self.device)
        self.allegro_hand_dof_default_pos = to_torch(self.allegro_hand_dof_default_pos, device=self.device)
        self.allegro_hand_dof_default_vel = to_torch(self.allegro_hand_dof_default_vel, device=self.device)

        # load manipulated object and goal assets
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 1000

        self.object_radius = 0.06
        object_asset = self.gym.create_sphere(self.sim, 0.06, object_asset_options)

        object_asset_options.disable_gravity = True
        goal_asset = self.gym.create_sphere(self.sim, 0.04, object_asset_options)

        allegro_hand_start_pose = gymapi.Transform()
        allegro_hand_start_pose.p = gymapi.Vec3(*get_axis_params(0, self.up_axis_idx))
        allegro_hand_start_pose.r = gymapi.Quat().from_euler_zyx(0, 0, -1.57)

        allegro_another_hand_start_pose = gymapi.Transform()
        allegro_another_hand_start_pose.p = gymapi.Vec3(0, -1.15, 0)
        allegro_another_hand_start_pose.r = gymapi.Quat().from_euler_zyx(0, 0, 1.57)

        object_start_pose = gymapi.Transform()
        object_start_pose.p = gymapi.Vec3()
        object_start_pose.p.x = allegro_hand_start_pose.p.x
        # pose_dy, pose_dz = -0.22, 0.47
        pose_dy, pose_dz = -0.3, 0.45

        object_start_pose.p.y = allegro_hand_start_pose.p.y + pose_dy
        object_start_pose.p.z = allegro_hand_start_pose.p.z + pose_dz
        object_start_pose.p = gymapi.Vec3(-0.0, -0.39, 0.39)

        if self.object_type == "pen":
            object_start_pose.p.z = allegro_hand_start_pose.p.z + 0.02

        self.goal_displacement = gymapi.Vec3(-0., 0.0, 0.)
        self.goal_displacement_tensor = to_torch(
            [self.goal_displacement.x, self.goal_displacement.y, self.goal_displacement.z], device=self.device)
        goal_start_pose = gymapi.Transform()
        goal_start_pose.p = object_start_pose.p + self.goal_displacement

        goal_start_pose.p.z -= 0.0

        # compute aggregate size
        max_agg_bodies = self.num_allegro_hand_bodies * 2 + 2
        max_agg_shapes = self.num_allegro_hand_shapes * 2 + 2

        self.allegro_hands = []
        self.envs = []

        self.object_init_state = []
        self.hand_start_states = []

        self.hand_indices = []
        self.another_hand_indices = []
        self.fingertip_indices = []
        self.object_indices = []
        self.goal_object_indices = []

        if self.enable_camera_sensors:
            self.cameras = []
            self.camera_tensors = []
            self.camera_view_matrixs = []
            self.camera_proj_matrixs = []

            self.camera_props = gymapi.CameraProperties()
            self.camera_props.width = 256
            self.camera_props.height = 256
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
            allegro_hand_actor = self.gym.create_actor(env_ptr, allegro_hand_asset, allegro_hand_start_pose, "hand", i, -1, 0)
            allegro_hand_another_actor = self.gym.create_actor(env_ptr, allegro_hand_another_asset, allegro_another_hand_start_pose, "another_hand", i, -1, 0)
            
            self.hand_start_states.append([allegro_hand_start_pose.p.x, allegro_hand_start_pose.p.y, allegro_hand_start_pose.p.z,
                                           allegro_hand_start_pose.r.x, allegro_hand_start_pose.r.y, allegro_hand_start_pose.r.z, allegro_hand_start_pose.r.w,
                                           0, 0, 0, 0, 0, 0])
            
            self.gym.set_actor_dof_properties(env_ptr, allegro_hand_actor, allegro_hand_dof_props)
            hand_idx = self.gym.get_actor_index(env_ptr, allegro_hand_actor, gymapi.DOMAIN_SIM)
            self.hand_indices.append(hand_idx)

            self.gym.set_actor_dof_properties(env_ptr, allegro_hand_another_actor, allegro_hand_another_dof_props)
            another_hand_idx = self.gym.get_actor_index(env_ptr, allegro_hand_another_actor, gymapi.DOMAIN_SIM)
            self.another_hand_indices.append(another_hand_idx)            

            # randomize colors and textures for rigid body
            num_bodies = self.gym.get_actor_rigid_body_count(env_ptr, allegro_hand_actor)
            hand_rigid_body_index = [[0,1,2,3], [4,5,6,7], [8,9,10,11], [12,13,14,15], [16,17,18,19,20], [21,22,23,24,25]]
            
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

            if self.enable_camera_sensors:
                camera_handle = self.gym.create_camera_sensor(env_ptr, self.camera_props)
                self.gym.set_camera_location(camera_handle, env_ptr, gymapi.Vec3(0.25, -0.5, 0.75), gymapi.Vec3(-0.24, -0.5, 0))
                camera_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, camera_handle, gymapi.IMAGE_DEPTH)
                torch_cam_tensor = gymtorch.wrap_tensor(camera_tensor)
                cam_vinv = torch.inverse((torch.tensor(self.gym.get_camera_view_matrix(self.sim, env_ptr, camera_handle)))).to(self.device)
                cam_proj = torch.tensor(self.gym.get_camera_proj_matrix(self.sim, env_ptr, camera_handle), device=self.device)
            
            if self.object_type != "block":
                self.gym.set_rigid_body_color(
                    env_ptr, object_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.6, 0.72, 0.98))
                self.gym.set_rigid_body_color(
                    env_ptr, goal_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.6, 0.72, 0.98))

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)
            self.allegro_hands.append(allegro_hand_actor)

            if self.enable_camera_sensors:
                origin = self.gym.get_env_origin(env_ptr)
                self.env_origin[i][0] = origin.x
                self.env_origin[i][1] = origin.y
                self.env_origin[i][2] = origin.z
                self.camera_tensors.append(torch_cam_tensor)
                self.camera_view_matrixs.append(cam_vinv)
                self.camera_proj_matrixs.append(cam_proj)
                self.cameras.append(camera_handle)

        sensor_handles = [self.gym.find_actor_rigid_body_handle(env_ptr, allegro_hand_another_actor, sensor_name) for sensor_name in
                          self.contact_sensor_names]
        
        self.sensor_handle_indices = to_torch(sensor_handles, dtype=torch.int64)

        object_rb_props = self.gym.get_actor_rigid_body_properties(env_ptr, object_handle)
        self.object_rb_masses = [prop.mass for prop in object_rb_props]

        self.object_init_state = to_torch(self.object_init_state, device=self.device, dtype=torch.float).view(self.num_envs, 13)
        self.goal_states = self.object_init_state.clone()
        self.goal_pose = self.goal_states[:, 0:7]
        self.goal_pos = self.goal_states[:, 0:3]
        self.goal_rot = self.goal_states[:, 3:7]
        # self.goal_states[:, self.up_axis_idx] -= 0.04
        self.goal_init_state = self.goal_states.clone()
        self.hand_start_states = to_torch(self.hand_start_states, device=self.device).view(self.num_envs, 13)

        self.hand_indices = to_torch(self.hand_indices, dtype=torch.long, device=self.device)
        self.another_hand_indices = to_torch(self.another_hand_indices, dtype=torch.long, device=self.device)

        self.object_indices = to_torch(self.object_indices, dtype=torch.long, device=self.device)
        self.goal_object_indices = to_torch(self.goal_object_indices, dtype=torch.long, device=self.device)

        self.init_object_tracking = True

    def compute_reward(self, actions):
        self.rew_buf[:], self.reset_buf[:], self.reset_goal_buf[:], self.progress_buf[:], self.successes[:], self.consecutive_successes[:] = compute_hand_reward(
            self.rew_buf, self.reset_buf, self.reset_goal_buf, self.progress_buf, self.successes, self.consecutive_successes,
            self.max_episode_length, self.object_pos, self.object_rot, self.goal_pos, self.goal_rot, self.allegro_left_hand_pos, self.allegro_right_hand_pos,
            self.dist_reward_scale, self.rot_reward_scale, self.rot_eps, self.actions, self.action_penalty_scale,
            self.success_tolerance, self.reach_goal_bonus, self.fall_dist, self.fall_penalty,
            self.max_consecutive_successes, self.av_factor, (self.object_type == "pen")
        )

        self.extras['successes'] = self.successes
        self.extras['consecutive_successes'] = self.consecutive_successes

        if self.print_success_stat:
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
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self.allegro_right_hand_pos = self.rigid_body_states[:, 6, 0:3]
        self.allegro_right_hand_rot = self.rigid_body_states[:, 6, 3:7]

        self.allegro_left_hand_pos = self.rigid_body_states[:, 6 + 23, 0:3]
        self.allegro_left_hand_rot = self.rigid_body_states[:, 6 + 23, 3:7]

        self.object_pose = self.root_state_tensor[self.object_indices, 0:7]
        self.object_pos = self.root_state_tensor[self.object_indices, 0:3]
        self.object_rot = self.root_state_tensor[self.object_indices, 3:7]
        self.object_linvel = self.root_state_tensor[self.object_indices, 7:10]
        self.object_angvel = self.root_state_tensor[self.object_indices, 10:13]

        self.goal_pose = self.goal_states[:, 0:7]
        self.goal_pos = self.goal_states[:, 0:3]
        self.goal_rot = self.goal_states[:, 3:7]

        if self.enable_camera_sensors:
            self.gym.render_all_camera_sensors(self.sim)
            self.gym.start_access_image_tensors(self.sim)

            self.camera_rgba_debug_fig = plt.figure("CAMERA_RGBD_DEBUG")
            # camera_rgba_image = self.camera_visulization(is_depth_image=False)
            camera_rgba_image = self.camera_tracking_visulization(is_depth_image=False)

            # plt.imshow(camera_rgba_image)
            # plt.pause(1e-9)

            self.gym.end_access_image_tensors(self.sim)

        self.compute_sim2real_observation()
        # self.compute_full_state()

        if self.asymmetric_obs:
            self.compute_sim2real_asymmetric_obs()

    def compute_sim2real_observation(self):

        self.obs_buf[:, 0:self.num_allegro_hand_dofs] = unscale(self.allegro_hand_dof_pos,
                                                            self.allegro_hand_dof_lower_limits, self.allegro_hand_dof_upper_limits)
        
        action_obs_start = 22
        self.obs_buf[:, action_obs_start:action_obs_start + 18] = self.actions[:, :18]

        # another_hand
        another_hand_start = action_obs_start + 18
        self.obs_buf[:, another_hand_start:self.num_allegro_hand_dofs + another_hand_start] = unscale(self.allegro_hand_another_dof_pos,
                                                            self.allegro_hand_dof_lower_limits, self.allegro_hand_dof_upper_limits)

        action_another_obs_start = another_hand_start + 22
        self.obs_buf[:, action_another_obs_start:action_another_obs_start + 18] = self.actions[:, 18:]

        #another hand contact
        contact_start = action_another_obs_start + 18
        contacts = self.contact_tensor.reshape(-1, 94, 3)  # 39+27
        contacts = contacts[:, self.sensor_handle_indices, :] # 12
        contacts = torch.norm(contacts, dim=-1)
        contacts = torch.where(contacts >= 1.0, 1.0, 0.0)
        self.obs_buf[:, contact_start:contact_start + 11] = contacts
        # visualize 
        # for i in range(len(contacts[0])):
        #     if contacts[0][i] == 1.0:
        #         self.gym.set_rigid_body_color(
        #                     self.envs[0], self.another_hand_indices[0], self.sensor_handle_indices[i] - 17, gymapi.MESH_VISUAL, gymapi.Vec3(1, 0.3, 0.3))
        #     else:
        #         self.gym.set_rigid_body_color(
        #                     self.envs[0], self.another_hand_indices[0], self.sensor_handle_indices[i] - 17, gymapi.MESH_VISUAL, gymapi.Vec3(0, 0.0, 0.0))

        obj_obs_start = contact_start + 11
        # self.obs_buf[:, obj_obs_start:obj_obs_start + 7] = self.object_pose
        self.obs_buf[:, obj_obs_start:obj_obs_start + 7] = self.object_pose_for_open_loop
        # self.obs_buf[:, obj_obs_start + 7:obj_obs_start + 10] = self.object_linvel
        # self.obs_buf[:, obj_obs_start + 10:obj_obs_start + 13] = self.vel_obs_scale * self.object_angvel

        goal_obs_start = obj_obs_start + 7 #
        self.obs_buf[:, goal_obs_start:goal_obs_start + 7] = self.goal_pose
        self.obs_buf[:, goal_obs_start + 7:goal_obs_start + 11] = quat_mul(self.object_rot, quat_conjugate(self.goal_rot))

    def compute_sim2real_asymmetric_obs(self):
        self.states_buf[:, 0:self.num_allegro_hand_dofs] = unscale(self.allegro_hand_dof_pos,
                                                    self.allegro_hand_dof_lower_limits, self.allegro_hand_dof_upper_limits)
        self.states_buf[:, self.num_allegro_hand_dofs:2*self.num_allegro_hand_dofs] = self.vel_obs_scale * self.allegro_hand_dof_vel

        action_obs_start = 44
        self.states_buf[:, action_obs_start:action_obs_start + 18] = self.actions[:, :18]

        # another_hand
        another_hand_start = action_obs_start + 18
        self.states_buf[:, another_hand_start:self.num_allegro_hand_dofs + another_hand_start] = unscale(self.allegro_hand_another_dof_pos,
                                                            self.allegro_hand_dof_lower_limits, self.allegro_hand_dof_upper_limits)
        self.states_buf[:, self.num_allegro_hand_dofs + another_hand_start:2*self.num_allegro_hand_dofs + another_hand_start] = self.vel_obs_scale * self.allegro_hand_another_dof_vel

        action_another_obs_start = another_hand_start + 44
        self.states_buf[:, action_another_obs_start:action_another_obs_start + 18] = self.actions[:, 18:]

        #another hand contact
        contact_start = action_another_obs_start + 18
        contacts = self.contact_tensor.reshape(-1, 94, 3)  # 39+27
        contacts = contacts[:, self.sensor_handle_indices, :] # 12
        contacts = torch.norm(contacts, dim=-1)
        contacts = torch.where(contacts >= 1.0, 1.0, 0.0)
        self.states_buf[:, contact_start:contact_start + 11] = contacts
        # visualize 
        # for i in range(len(contacts[0])):
        #     if contacts[0][i] == 1.0:
        #         self.gym.set_rigid_body_color(
        #                     self.envs[0], self.another_hand_indices[0], self.sensor_handle_indices[i] - 17, gymapi.MESH_VISUAL, gymapi.Vec3(1, 0.3, 0.3))
        #     else:
        #         self.gym.set_rigid_body_color(
        #                     self.envs[0], self.another_hand_indices[0], self.sensor_handle_indices[i] - 17, gymapi.MESH_VISUAL, gymapi.Vec3(0, 0.0, 0.0))

        obj_obs_start = contact_start + 11
        self.states_buf[:, obj_obs_start:obj_obs_start + 7] = self.object_pose
        # self.obs_buf[:, obj_obs_start:obj_obs_start + 7] = self.root_state_tensor[self.object_indices, 0:7]
        self.states_buf[:, obj_obs_start + 7:obj_obs_start + 10] = self.object_linvel
        self.states_buf[:, obj_obs_start + 10:obj_obs_start + 13] = self.vel_obs_scale * self.object_angvel

        goal_obs_start = obj_obs_start + 13 #
        self.states_buf[:, goal_obs_start:goal_obs_start + 7] = self.goal_pose
        self.states_buf[:, goal_obs_start + 7:goal_obs_start + 11] = quat_mul(self.object_rot, quat_conjugate(self.goal_rot))

    def reset_target_pose(self, env_ids, apply_reset=False):
        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), 4), device=self.device)

        new_rot = randomize_rotation(rand_floats[:, 0], rand_floats[:, 1], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids])

        self.goal_states[env_ids, 0:3] = self.goal_init_state[env_ids, 0:3]
        self.goal_states[env_ids, 1] -= 0.35
        self.goal_states[env_ids, 2] += 0.05
        self.goal_states[env_ids, 3:7] = new_rot
        self.root_state_tensor[self.goal_object_indices[env_ids], 0:3] = self.goal_states[env_ids, 0:3] + self.goal_displacement_tensor
        self.root_state_tensor[self.goal_object_indices[env_ids], 3:7] = self.goal_states[env_ids, 3:7]
        self.root_state_tensor[self.goal_object_indices[env_ids], 7:13] = torch.zeros_like(self.root_state_tensor[self.goal_object_indices[env_ids], 7:13])

        if apply_reset:
            goal_object_indices = self.goal_object_indices[env_ids].to(torch.int32)
            self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                         gymtorch.unwrap_tensor(self.root_state_tensor),
                                                         gymtorch.unwrap_tensor(goal_object_indices), len(env_ids))
        self.reset_goal_buf[env_ids] = 0

    def reset(self, env_ids, goal_env_ids):
        # randomization can happen only at reset time, since it can reset actor positions on GPU
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        # generate random values
        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), self.num_allegro_hand_dofs * 2 + 5), device=self.device)

        # randomize start object poses
        self.reset_target_pose(env_ids)

        # reset object
        self.root_state_tensor[self.object_indices[env_ids]] = self.object_init_state[env_ids].clone()
        self.root_state_tensor[self.object_indices[env_ids], 0:2] = self.object_init_state[env_ids, 0:2] + \
            self.reset_position_noise * rand_floats[:, 0:2]
        self.root_state_tensor[self.object_indices[env_ids], self.up_axis_idx] = self.object_init_state[env_ids, self.up_axis_idx] + \
            self.reset_position_noise * rand_floats[:, self.up_axis_idx]

        new_object_rot = randomize_rotation(rand_floats[:, 3], rand_floats[:, 4], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids])
        if self.object_type == "pen":
            rand_angle_y = torch.tensor(0.3)
            new_object_rot = randomize_rotation_pen(rand_floats[:, 3], rand_floats[:, 4], rand_angle_y,
                                                    self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids], self.z_unit_tensor[env_ids])

        self.root_state_tensor[self.object_indices[env_ids], 3:7] = new_object_rot
        self.root_state_tensor[self.object_indices[env_ids], 7:13] = torch.zeros_like(self.root_state_tensor[self.object_indices[env_ids], 7:13])

        self.object_pose_for_open_loop[env_ids] = self.root_state_tensor[self.object_indices[env_ids], 0:7].clone()

        object_indices = torch.unique(torch.cat([self.object_indices[env_ids],
                                                 self.goal_object_indices[env_ids],
                                                 self.goal_object_indices[goal_env_ids]]).to(torch.int32))
        # self.gym.set_actor_root_state_tensor_indexed(self.sim,
        #                                              gymtorch.unwrap_tensor(self.root_state_tensor),
        #                                              gymtorch.unwrap_tensor(object_indices), len(object_indices))

        # reset shadow hand
        pos = self.allegro_hand_default_dof_pos

        self.allegro_hand_dof_pos[env_ids, :] = pos
        self.allegro_hand_another_dof_pos[env_ids, :] = pos

        self.allegro_hand_dof_vel[env_ids, :] = self.allegro_hand_dof_default_vel + \
            self.reset_dof_vel_noise * rand_floats[:, 5+self.num_allegro_hand_dofs:5+self.num_allegro_hand_dofs*2]   

        self.allegro_hand_another_dof_vel[env_ids, :] = self.allegro_hand_dof_default_vel + \
            self.reset_dof_vel_noise * rand_floats[:, 5+self.num_allegro_hand_dofs:5+self.num_allegro_hand_dofs*2]

        self.prev_targets[env_ids, :self.num_allegro_hand_dofs] = pos
        self.cur_targets[env_ids, :self.num_allegro_hand_dofs] = pos

        self.prev_targets[env_ids, self.num_allegro_hand_dofs:self.num_allegro_hand_dofs*2] = pos
        self.cur_targets[env_ids, self.num_allegro_hand_dofs:self.num_allegro_hand_dofs*2] = pos

        hand_indices = self.hand_indices[env_ids].to(torch.int32)
        another_hand_indices = self.another_hand_indices[env_ids].to(torch.int32)
        all_hand_indices = torch.unique(torch.cat([hand_indices,
                                                 another_hand_indices]).to(torch.int32))

        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.prev_targets),
                                                        gymtorch.unwrap_tensor(all_hand_indices), len(all_hand_indices))  

        all_indices = torch.unique(torch.cat([all_hand_indices,
                                                 object_indices]).to(torch.int32))

        # self.hand_positions[all_indices.to(torch.long), :] = self.saved_root_tensor[all_indices.to(torch.long), 0:3]
        # self.hand_orientations[all_indices.to(torch.long), :] = self.saved_root_tensor[all_indices.to(torch.long), 3:7]
        # self.hand_linvels[all_indices.to(torch.long), :] = torch.zeros_like(self.saved_root_tensor[all_indices.to(torch.long), 7:10])
        # self.hand_angvels[all_indices.to(torch.long), :] = torch.zeros_like(self.saved_root_tensor[all_indices.to(torch.long), 10:13])

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(all_hand_indices), len(all_hand_indices))

        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_state_tensor),
                                                     gymtorch.unwrap_tensor(all_indices), len(all_indices))
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.successes[env_ids] = 0

        # if 1 in env_ids:
        self.init_object_tracking = True
        self.gym.clear_lines(self.viewer)

    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)

        #record trajectory
        # if self.progress_buf[0] in [0, 1]:
        #     self.save_right_allegro_hand_dof_pos = [np.array(self.allegro_hand_dof_pos[0].cpu())]
        #     self.save_left_allegro_hand_dof_pos = [np.array(self.allegro_hand_another_dof_pos[0].cpu())]
        #     self.save_right_allegro_hand_target_dof_pos = [np.array(self.actions[0, :18].cpu())]
        #     self.save_left_allegro_hand_target_dof_pos = [np.array(self.actions[0, 18:36].cpu())]
        
        #     self.save_right_allegro_hand_dof_vel = [np.array(self.allegro_hand_dof_vel[0].cpu())]

        # elif self.reset_buf[0] == 1:
        #     np.set_printoptions(suppress=True)
        #     np.set_printoptions(precision=8)

        #     np.savetxt('trajectory/left_allegro_hand_dof_pos.txt', self.save_left_allegro_hand_dof_pos, fmt='%.08f')
        #     np.savetxt('trajectory/right_allegro_hand_dof_pos.txt', self.save_right_allegro_hand_dof_pos, fmt='%.08f')
        #     np.savetxt('trajectory/left_allegro_hand_target_dof_pos.txt', self.save_left_allegro_hand_target_dof_pos, fmt='%.08f')
        #     np.savetxt('trajectory/right_allegro_hand_target_dof_pos.txt', self.save_right_allegro_hand_target_dof_pos, fmt='%.08f')
            
        #     np.savetxt('trajectory/right_allegro_hand_dof_vel.txt', self.save_right_allegro_hand_dof_vel, fmt='%.08f')

        #     print("Finish a trajectory")
        # else:
        #     self.save_right_allegro_hand_dof_pos.append(np.array(self.allegro_hand_dof_pos[0].cpu()))
        #     self.save_left_allegro_hand_dof_pos.append(np.array(self.allegro_hand_another_dof_pos[0].cpu()))
        #     self.save_right_allegro_hand_target_dof_pos.append(np.array(self.actions[0, :18].cpu()))
        #     self.save_left_allegro_hand_target_dof_pos.append(np.array(self.actions[0, 18:36].cpu()))

        #     self.save_right_allegro_hand_dof_vel.append(np.array(self.allegro_hand_dof_vel[0].cpu()))

        # traj = np.loadtxt("/home/jmji/DexterousHandEnvs/dexteroushandenvs/trajectory/35_joint_big_1_1000density/right_allegro_hand_dof_pos.txt")
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)

        # if only goals need reset, then call set API
        if len(goal_env_ids) > 0 and len(env_ids) == 0:
            self.reset_target_pose(goal_env_ids, apply_reset=True)
        # if goals need reset in addition to other envs, call set API in reset()
        elif len(goal_env_ids) > 0:
            self.reset_target_pose(goal_env_ids)

        if len(env_ids) > 0:
            self.reset(env_ids, goal_env_ids)

        if self.use_relative_control:
            # targets = self.prev_targets[:, self.actuated_dof_indices] + self.allegro_hand_dof_speed_scale * self.dt * self.actions
            targets = self.allegro_hand_dof_pos[:, self.actuated_dof_indices + 6] + self.allegro_hand_dof_speed_scale * self.dt * self.actions[:, 0:16]
            self.cur_targets[:, self.actuated_dof_indices + 6] = tensor_clamp(targets,
                                                                          self.allegro_hand_dof_lower_limits[self.actuated_dof_indices + 6], self.allegro_hand_dof_upper_limits[self.actuated_dof_indices + 6])
            targets = self.allegro_hand_another_dof_pos[:, self.actuated_dof_indices + 6] + self.allegro_hand_dof_speed_scale * self.dt * self.actions[:, 16:32]

            self.cur_targets[:, self.actuated_dof_indices + 28] = tensor_clamp(targets,
                                                                          self.allegro_hand_dof_lower_limits[self.actuated_dof_indices + 6], self.allegro_hand_dof_upper_limits[self.actuated_dof_indices + 6])
            targets = self.allegro_hand_dof_pos[:, [2, 4]] + self.allegro_hand_dof_speed_scale * self.dt * self.actions[:, 32:34]

            self.cur_targets[:, [2, 4]] = tensor_clamp(targets,
                                                                          self.allegro_hand_dof_lower_limits[[2, 4]], self.allegro_hand_dof_upper_limits[[2, 4]])
            targets = self.allegro_hand_another_dof_pos[:, [2, 4]] + self.allegro_hand_dof_speed_scale * self.dt * self.actions[:, 34:36]

            self.cur_targets[:, [24, 26]] = tensor_clamp(targets,
                                                                          self.allegro_hand_dof_lower_limits[[2, 4]], self.allegro_hand_dof_upper_limits[[2, 4]])
        
        else:
            # x-arm control
            targets = self.prev_targets[:, [2, 4]] + self.allegro_hand_dof_speed_scale * self.dt * self.actions[:, [0, 1]]
            self.cur_targets[:, [2, 4]] = tensor_clamp(targets,
                                    self.allegro_hand_dof_lower_limits[[2, 4]], self.allegro_hand_dof_upper_limits[[2, 4]])

            self.cur_targets[:, self.actuated_dof_indices + 6] = scale(self.actions[:, 2:18],
                                                                   self.allegro_hand_dof_lower_limits[self.actuated_dof_indices + 6], self.allegro_hand_dof_upper_limits[self.actuated_dof_indices + 6])
            self.cur_targets[:, self.actuated_dof_indices + 6] = tensor_clamp(self.cur_targets[:, self.actuated_dof_indices + 6],
                                                                          self.allegro_hand_dof_lower_limits[self.actuated_dof_indices + 6], self.allegro_hand_dof_upper_limits[self.actuated_dof_indices + 6])
            
            targets = self.prev_targets[:, [24, 26]] + self.allegro_hand_dof_speed_scale * self.dt * self.actions[:, [18, 19]]
            self.cur_targets[:, [24, 26]] = tensor_clamp(targets,
                                    self.allegro_hand_dof_lower_limits[[2, 4]], self.allegro_hand_dof_upper_limits[[2, 4]])

            self.cur_targets[:, self.actuated_dof_indices + 28] = scale(self.actions[:, 20:36],
                                                                   self.allegro_hand_dof_lower_limits[self.actuated_dof_indices + 6], self.allegro_hand_dof_upper_limits[self.actuated_dof_indices + 6])
            self.cur_targets[:, self.actuated_dof_indices + 28] = tensor_clamp(self.cur_targets[:, self.actuated_dof_indices + 28],
                                                                          self.allegro_hand_dof_lower_limits[self.actuated_dof_indices + 6], self.allegro_hand_dof_upper_limits[self.actuated_dof_indices + 6])

        self.prev_targets[:, :] = self.cur_targets[:, :]
        # self.cur_targets[:, 0:22] = to_torch(traj[self.progress_buf[0]], device=self.device)
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.cur_targets))

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
                self.add_debug_lines(self.envs[i], self.allegro_right_hand_pos[i], self.allegro_right_hand_rot[i])
                self.add_debug_lines(self.envs[i], self.allegro_left_hand_pos[i], self.allegro_left_hand_rot[i])

    def add_debug_lines(self, env, pos, rot, line_width=1):
        posx = (pos + quat_apply(rot, to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
        posy = (pos + quat_apply(rot, to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
        posz = (pos + quat_apply(rot, to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

        p0 = pos.cpu().numpy()
        self.gym.add_lines(self.viewer, env, line_width, [p0[0], p0[1], p0[2], posx[0], posx[1], posx[2]], [0.85, 0.1, 0.1])
        self.gym.add_lines(self.viewer, env, line_width, [p0[0], p0[1], p0[2], posy[0], posy[1], posy[2]], [0.1, 0.85, 0.1])
        self.gym.add_lines(self.viewer, env, line_width, [p0[0], p0[1], p0[2], posz[0], posz[1], posz[2]], [0.1, 0.1, 0.85])

#####################################################################
###=========================jit functions=========================###
#####################################################################
    def camera_visulization(self, is_depth_image=False):
        if is_depth_image:
            camera_depth_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[0], self.cameras[0], gymapi.IMAGE_DEPTH)
            torch_depth_tensor = gymtorch.wrap_tensor(camera_depth_tensor)
            torch_depth_tensor = torch.clamp(torch_depth_tensor, -1, 1)
            torch_depth_tensor = scale(torch_depth_tensor, to_torch([0], dtype=torch.float, device=self.device),
                                                         to_torch([256], dtype=torch.float, device=self.device))
            camera_image = torch_depth_tensor.cpu().numpy()
            camera_image = Im.fromarray(camera_image)
        
        else:
            camera_rgba_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[0], self.cameras[0], gymapi.IMAGE_COLOR)
            torch_rgba_tensor = gymtorch.wrap_tensor(camera_rgba_tensor)
            camera_image = torch_rgba_tensor.cpu().numpy()
            camera_image = Im.fromarray(camera_image)
        
        return camera_image

    def camera_tracking_visulization(self, is_depth_image=False):

        camera_rgba_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[0], self.cameras[0], gymapi.IMAGE_COLOR)
        torch_rgba_tensor = gymtorch.wrap_tensor(camera_rgba_tensor)
        camera_image = torch_rgba_tensor.cpu().numpy()
        cv2_camera_image = camera_image.copy()
        camera_image = Im.fromarray(camera_image)

        cv2_camera_image = cv2.cvtColor(cv2_camera_image, cv2.COLOR_BGR2RGB)

        if self.init_object_tracking:
            self.tracker_type = 'MIL'
            self.tracker = cv2.TrackerMIL_create()

            # 定义一个bounding box
            bbox = (143, 107, 37, 36)
            # bbox = cv2.selectROI(cv2_camera_image, False)
            # print(bbox)
            # 用第一帧初始化
            self.ok = self.tracker.init(cv2_camera_image, bbox)
            self.init_object_tracking = False

        else:
            # Start timer
            timer = cv2.getTickCount()
            # Update tracker
            ok, bbox = self.tracker.update(cv2_camera_image)

            # Solve pnp
            obj = np.array([[-0.06, -0.06, 0], [0.06, -0.06, 0], [0.06, 0.06, 0], [-0.06, 0.06, 0]], dtype=np.float32)
            pnt = np.array([[bbox[0], bbox[1]], [bbox[0] + bbox[2], bbox[1]], [bbox[0] + bbox[2], bbox[1] + bbox[3]], [bbox[0], bbox[1] + bbox[3]]], dtype=np.float32)
            mtx = np.array([[128,   0.,   128],
                [  0.,   128, 128.],
                [  0.,     0.,     1.  ]], dtype=np.float32)
            dist = np.array([[-0.,  -0., -0.,  0.,    0.]], dtype=np.float32)

            (success, rvec, tvec) = cv2.solvePnP(obj, pnt, mtx, dist)
            print(tvec)
            print(self.root_state_tensor[self.object_indices[0], 0:3])
            # tvec = self.root_state_tensor[self.object_indices[0], 0:3] - torch.tensor([0.25, -0.5, 0.75], device=self.device)
            vinv = self.camera_view_matrixs[0].double()

            # Get the camera projection matrix and get the necessary scaling
            # coefficients for deprojection
            proj = self.camera_proj_matrixs[0]
            fu = 2/proj[0, 0]
            fv = 2/proj[1, 1]

            centerU = 128
            centerV = 128
            Z = -torch.tensor(np.sqrt(tvec[0]**2+tvec[1]**2+tvec[2]**2), device=self.device, dtype=torch.double) - 0.06
            print(Z)
            bbox_tensor = torch.tensor(bbox, device=self.device, dtype=torch.double)
            X = -((bbox_tensor[0]+bbox_tensor[2]/2)-centerU)/256 * Z * fu
            Y = ((bbox_tensor[1]+bbox_tensor[3]/2)-centerV)/256 * Z * fv

            position = torch.vstack((X, Y, Z, torch.ones(1, device=self.device)))
            position = position.permute(1, 0).double()
            position = position@vinv

            points = position[:, 0:3] - self.env_origin.view(self.num_envs, 1, 3)[0]
            print(points)
            self.add_debug_lines(self.envs[0], points[0], self.allegro_right_hand_rot[0], line_width=2)

            # Cakculate FPS
            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
            # Draw bonding box
            if ok:
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(cv2_camera_image, p1, p2, (255,0,0), 2, 1)
            else:
                cv2.putText(cv2_camera_image, "Tracking failed detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            # 展示tracker类型
            cv2.putText(cv2_camera_image, self.tracker_type+"Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
            # 展示FPS
            cv2.putText(cv2_camera_image, "FPS:"+str(fps), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

        cv2.imshow("Tracking", cv2_camera_image)
        cv2.waitKey(1)

        return camera_image

    def camera_calibrate(self, cv2_camera_image, bbox):
        # obj = np.array([[-6, -6, 0], [6, -6, 0], [6, 6, 0], [-6, 6, 0]], dtype=np.float32)
        # pnt = np.array([[bbox[0], bbox[1]], [bbox[0] + bbox[2], bbox[1]], [bbox[0] + bbox[2], bbox[1] + bbox[3]], [bbox[0], bbox[1] + bbox[3]]], dtype=np.float32)
        obj = np.array([[0, 0, 0], [0, 250, 0], [0, 500, 0], [0, 750, 0], [0, 1000, 0], [250, 0, 0], [250, 250, 0], [250, 500, 0], [250, 750, 0], [250, 1000, 0]], dtype=np.float32)
        pnt = np.array([[104, 6], [95, 29], [80, 69], [44, 165], [10, 254], [154, 6], [163, 29], [177, 69], [215, 165], [246, 254]], dtype=np.float32)

        print(obj)
        print(pnt)
        size = cv2_camera_image.shape[:2]
        print(size)
        obj_points = [obj]
        img_points = [pnt]

        # 标定
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, mtx, dist)

        print("ret:", ret)
        print("mtx:\n", mtx) # 内参数矩阵--内参
        print("dist:\n", dist)  # 畸变系数--内参 
        print("rvecs:\n", rvecs)  # 旋转向量--外参
        print("tvecs:\n", tvecs ) # 平移向量--外参


@torch.jit.script
def compute_hand_reward(
    rew_buf, reset_buf, reset_goal_buf, progress_buf, successes, consecutive_successes,
    max_episode_length: float, object_pos, object_rot, target_pos, target_rot, allegro_left_hand_pos, allegro_right_hand_pos,
    dist_reward_scale: float, rot_reward_scale: float, rot_eps: float,
    actions, action_penalty_scale: float,
    success_tolerance: float, reach_goal_bonus: float, fall_dist: float,
    fall_penalty: float, max_consecutive_successes: int, av_factor: float, ignore_z_rot: bool
):
    # Distance from the hand to the object
    goal_dist = torch.norm(target_pos - object_pos, p=2, dim=-1)
    if ignore_z_rot:
        success_tolerance = 2.0 * success_tolerance

    # Orientation alignment for the cube in hand and goal cube
    quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
    rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))

    dist_rew = goal_dist
    # rot_rew = 1.0/(torch.abs(rot_dist) + rot_eps) * rot_reward_scale

    action_penalty = torch.sum(actions ** 2, dim=-1)

    # Total reward is: position distance + orientation alignment + action regularization + success bonus + fall penalty
    reward = torch.exp(-0.2*(dist_rew * dist_reward_scale + rot_dist))

    # Find out which envs hit the goal and update successes count
    goal_resets = torch.where(torch.abs(goal_dist) <= 0, torch.ones_like(reset_goal_buf), reset_goal_buf)
    successes = successes + goal_resets

    # Success bonus: orientation is within `success_tolerance` of goal orientation
    reward = torch.where(goal_resets == 1, reward + reach_goal_bonus, reward)

    # Fall penalty: distance to the goal is larger than a threashold
    reward = torch.where(object_pos[:, 2] <= 0.2, reward + fall_penalty, reward)

    # Check env termination conditions, including maximum success number
    resets = torch.where(object_pos[:, 2] <= 0.1, torch.ones_like(reset_buf), reset_buf)
    # resets = torch.where(allegro_right_hand_pos[:, 1] <= -0.8, torch.ones_like(resets), resets)

    if max_consecutive_successes > 0:
        # Reset progress buffer on goal envs if max_consecutive_successes > 0
        progress_buf = torch.where(torch.abs(rot_dist) <= success_tolerance, torch.zeros_like(progress_buf), progress_buf)
        resets = torch.where(successes >= max_consecutive_successes, torch.ones_like(resets), resets)
    resets = torch.where(progress_buf >= max_episode_length, torch.ones_like(resets), resets)

    # Apply penalty for not reaching the goal
    if max_consecutive_successes > 0:
        reward = torch.where(progress_buf >= max_episode_length, reward + 0.5 * fall_penalty, reward)

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
