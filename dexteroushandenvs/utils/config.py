# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import sys
import yaml

from isaacgym import gymapi
from isaacgym import gymutil

import numpy as np
import random
import torch


def set_np_formatting():
    np.set_printoptions(edgeitems=30, infstr='inf',
                        linewidth=4000, nanstr='nan', precision=2,
                        suppress=False, threshold=10000, formatter=None)


def warn_task_name():
    raise Exception(
        "Unrecognized task!")

def warn_algorithm_name():
    raise Exception(
                "Unrecognized algorithm!\nAlgorithm should be one of: [ppo, happo, hatrpo, mappo]")


def set_seed(seed, torch_deterministic=False):
    if seed == -1 and torch_deterministic:
        seed = 42
    elif seed == -1:
        seed = np.random.randint(0, 10000)
    print("Setting seed: {}".format(seed))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch_deterministic:
        # refer to https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.set_deterministic(True)
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    return seed


def retrieve_cfg(args, use_rlg_config=False):
    #TODO: add config files of sac, td3
    # 这里的设计有点不合理 可以修正
    if args.task == "ShadowHandOver":
        return os.path.join(args.logdir, "shadow_hand_over/{}/{}".format(args.algo, args.algo)), "cfg/{}/config.yaml".format(args.algo) , "cfg/shadow_hand_over.yaml"
    elif args.task == "ShadowHandCatchOverarm":
        return os.path.join(args.logdir, "shadow_hand_catch_overarm/{}/{}".format(args.algo, args.algo)), "cfg/{}/config.yaml".format(args.algo), "cfg/shadow_hand_catch_overarm.yaml"
    elif args.task == "ShadowHandCatchUnderarm":
        return os.path.join(args.logdir, "shadow_hand_catch_underarm/{}/{}".format(args.algo, args.algo)), "cfg/{}/config.yaml".format(args.algo), "cfg/shadow_hand_catch_underarm.yaml"
    elif args.task == "ShadowHandTwoCatchUnderarm":
        return os.path.join(args.logdir, "shadow_hand_two_catch_underarm/{}/{}".format(args.algo, args.algo)), "cfg/{}/config.yaml".format(args.algo), "cfg/shadow_hand_two_catch_underarm.yaml"
    elif args.task == "ShadowHandTwoCatchAbreast":
        return os.path.join(args.logdir, "shadow_hand_two_catch_abreast/{}/{}".format(args.algo, args.algo)), "cfg/{}/config.yaml".format(args.algo), "cfg/shadow_hand_two_catch_abreast.yaml"
    elif args.task == "ShadowHandCatchAbreast":
        return os.path.join(args.logdir, "shadow_hand_catch_abreast/{}/{}".format(args.algo, args.algo)), "cfg/{}/config.yaml".format(args.algo), "cfg/shadow_hand_catch_abreast.yaml"
    elif args.task == "ShadowHandReOrientation":
        return os.path.join(args.logdir, "shadow_hand_re_orientation/{}/{}".format(args.algo, args.algo)), "cfg/{}/re_orientation_config.yaml".format(args.algo), "cfg/shadow_hand_re_orientation.yaml"
    elif args.task == "ShadowHandOverOverarm":
        return os.path.join(args.logdir, "shadow_hand_over_overarm/{}/{}".format(args.algo, args.algo)), "cfg/{}/config.yaml".format(args.algo), "cfg/shadow_hand_over_overarm.yaml"
    # elif args.task == "ShadowHand":
    #     return os.path.join(args.logdir, "shadow_hand/{}/{}".format(args.algo, args.algo)), "cfg/{}/config.yaml".format(args.algo), "cfg/shadow_hand.yaml"
    elif args.task == "OneFrankaCabinet":
        return os.path.join(args.logdir, "franka_cabinet/{}/{}".format(args.algo, args.algo)), "cfg/{}/config.yaml".format(args.algo), "cfg/franka_cabinet.yaml"
    elif args.task == "ShadowHandLiftOverarm":
        return os.path.join(args.logdir, "shadow_hand_lift_overarm/{}/{}".format(args.algo, args.algo)), "cfg/{}/lift_config.yaml".format(args.algo), "cfg/shadow_hand_lift_overarm.yaml"
    elif args.task == "ShadowHandLiftUnderarm":
        return os.path.join(args.logdir, "shadow_hand_lift_underarm/{}/{}".format(args.algo, args.algo)), "cfg/{}/lift_config.yaml".format(args.algo), "cfg/shadow_hand_lift_underarm.yaml"
    elif args.task == "ShadowHandLift":
        return os.path.join(args.logdir, "shadow_hand_lift/{}/{}".format(args.algo, args.algo)), "cfg/{}/lift_config.yaml".format(args.algo), "cfg/shadow_hand_lift.yaml"
    elif args.task == "Humanoid":
        return os.path.join(args.logdir, "humanoid/{}/{}".format(args.algo, args.algo)), "cfg/{}/humanoid_config.yaml".format(args.algo), "cfg/humanoid.yaml"
    elif args.task == "ShadowHandThrowAbreast":
        return os.path.join(args.logdir, "shadow_hand_throw_abreast/{}/{}".format(args.algo, args.algo)), "cfg/{}/config.yaml".format(args.algo), "cfg/shadow_hand_throw_abreast.yaml"
    elif args.task == "ShadowHandCatchOver2Underarm":
        return os.path.join(args.logdir, "shadow_hand_catch_over2underarm/{}/{}".format(args.algo, args.algo)), "cfg/{}/config.yaml".format(args.algo), "cfg/shadow_hand_catch_over2underarm.yaml"
    elif args.task == "ShadowHandTest":
        return os.path.join(args.logdir, "shadow_hand_test/{}/{}".format(args.algo, args.algo)), "cfg/{}/config.yaml".format(args.algo), "cfg/shadow_hand_test.yaml"
    elif args.task == "ShadowHandLiftUnderarm2":
        return os.path.join(args.logdir, "shadow_hand_lift_underarm2/{}/{}".format(args.algo, args.algo)), "cfg/{}/config.yaml".format(args.algo), "cfg/shadow_hand_lift_underarm2.yaml"
    elif args.task == "ShadowHandBottleCap":
        return os.path.join(args.logdir, "shadow_hand_bottle_cap/{}/{}".format(args.algo, args.algo)), "cfg/{}/config.yaml".format(args.algo), "cfg/shadow_hand_bottle_cap.yaml"
    elif args.task == "ShadowHandDoorCloseInward":
        return os.path.join(args.logdir, "shadow_hand_door_close_inward/{}/{}".format(args.algo, args.algo)), "cfg/{}/config.yaml".format(args.algo), "cfg/shadow_hand_door_close_inward.yaml"
    elif args.task == "ShadowHandDoorCloseOutward":
        return os.path.join(args.logdir, "shadow_hand_door_close_outward/{}/{}".format(args.algo, args.algo)), "cfg/{}/config.yaml".format(args.algo), "cfg/shadow_hand_door_close_outward.yaml"
    elif args.task == "ShadowHandDoorOpenInward":
        return os.path.join(args.logdir, "shadow_hand_door_open_inward/{}/{}".format(args.algo, args.algo)), "cfg/{}/config.yaml".format(args.algo), "cfg/shadow_hand_door_open_inward.yaml"
    elif args.task == "ShadowHandDoorOpenOutward":
        return os.path.join(args.logdir, "shadow_hand_door_open_outward/{}/{}".format(args.algo, args.algo)), "cfg/{}/config.yaml".format(args.algo), "cfg/shadow_hand_door_open_outward.yaml"
    elif args.task == "ShadowHandKettle":
        return os.path.join(args.logdir, "shadow_hand_kettle/{}/{}".format(args.algo, args.algo)), "cfg/{}/config.yaml".format(args.algo), "cfg/shadow_hand_kettle.yaml"
    elif args.task == "ShadowHandPen":
        return os.path.join(args.logdir, "shadow_hand_pen/{}/{}".format(args.algo, args.algo)), "cfg/{}/config.yaml".format(args.algo), "cfg/shadow_hand_pen.yaml"
    elif args.task == "ShadowHandBlockStack":
        return os.path.join(args.logdir, "shadow_hand_block_stack/{}/{}".format(args.algo, args.algo)), "cfg/{}/stack_block_config.yaml".format(args.algo), "cfg/shadow_hand_block_stack.yaml"
    elif args.task == "ShadowHandSwitch":
        return os.path.join(args.logdir, "shadow_hand_switch/{}/{}".format(args.algo, args.algo)), "cfg/{}/config.yaml".format(args.algo), "cfg/shadow_hand_switch.yaml"
    elif args.task == "ShadowHandMeta":
        return os.path.join(args.logdir, "shadow_hand_meta/{}/{}".format(args.algo, args.algo)), "cfg/{}/config.yaml".format(args.algo), "cfg/meta_env_cfg/shadow_hand_meta.yaml"
    elif args.task == "ShadowHandLiftCup":
        return os.path.join(args.logdir, "shadow_hand_lift_cup/{}/{}".format(args.algo, args.algo)), "cfg/{}/config.yaml".format(args.algo), "cfg/shadow_hand_lift_cup.yaml"
    elif args.task == "ShadowHandMetaMT1":
        return os.path.join(args.logdir, "shadow_hand_meta_mt1/{}/{}".format(args.algo, args.algo)), "cfg/{}/config.yaml".format(args.algo), "cfg/meta_env_cfg/shadow_hand_meta_mt1.yaml"
    elif args.task == "ShadowHandMetaML1":
        return os.path.join(args.logdir, "shadow_hand_meta_ml1/{}/{}".format(args.algo, args.algo)), "cfg/{}/config.yaml".format(args.algo), "cfg/meta_env_cfg/shadow_hand_meta_ml1.yaml"
    elif args.task == "ShadowHandMetaMT5":
        return os.path.join(args.logdir, "shadow_hand_meta_mt5/{}/{}".format(args.algo, args.algo)), "cfg/{}/config.yaml".format(args.algo), "cfg/meta_env_cfg/shadow_hand_meta_mt5.yaml"
    elif args.task == "ShadowHandMetaMT5Door":
        return os.path.join(args.logdir, "shadow_hand_meta_mt5_door/{}/{}".format(args.algo, args.algo)), "cfg/{}/config.yaml".format(args.algo), "cfg/meta_env_cfg/shadow_hand_meta_mt5_door.yaml"
    elif args.task == "ShadowHandPushBlock":
        return os.path.join(args.logdir, "shadow_hand_push_block/{}/{}".format(args.algo, args.algo)), "cfg/{}/config.yaml".format(args.algo), "cfg/shadow_hand_push_block.yaml"
    elif args.task == "ShadowHandSwingCup":
        return os.path.join(args.logdir, "shadow_hand_swing_cup/{}/{}".format(args.algo, args.algo)), "cfg/{}/config.yaml".format(args.algo), "cfg/shadow_hand_swing_cup.yaml"
    elif args.task == "ShadowHandGraspAndPlace":
        return os.path.join(args.logdir, "shadow_hand_grasp_and_place/{}/{}".format(args.algo, args.algo)), "cfg/{}/config.yaml".format(args.algo), "cfg/shadow_hand_grasp_and_place.yaml"
    elif args.task == "ShadowHandScissors":
        return os.path.join(args.logdir, "shadow_hand_scissors/{}/{}".format(args.algo, args.algo)), "cfg/{}/config.yaml".format(args.algo), "cfg/shadow_hand_scissors.yaml"
    elif args.task == "ShadowHandMetaMT20":
        return os.path.join(args.logdir, "shadow_hand_meta_mt20/{}/{}".format(args.algo, args.algo)), "cfg/{}/config.yaml".format(args.algo), "cfg/meta_env_cfg/shadow_hand_meta_mt20.yaml"
    elif args.task == "ShadowHandCatchUnderarmWall":
        return os.path.join(args.logdir, "shadow_hand_catch_underarm_wall/{}/{}".format(args.algo, args.algo)), "cfg/{}/config.yaml".format(args.algo), "cfg/safe_env_cfg/shadow_hand_catch_underarm_wall.yaml"
    elif args.task == "AllegroHandOver":
        return os.path.join(args.logdir, "allegro_hand_over/{}/{}".format(args.algo, args.algo)), "cfg/{}/config.yaml".format(args.algo), "cfg/allegro_hand_over.yaml"
    elif args.task == "AllegroHandCatchUnderarm":
        return os.path.join(args.logdir, "allegro_hand_catch_underarm/{}/{}".format(args.algo, args.algo)), "cfg/{}/config.yaml".format(args.algo), "cfg/allegro_hand_catch_underarm.yaml"
    elif args.task == "ShadowHandPointCloud":
        return os.path.join(args.logdir, "shadow_hand_point_cloud/{}/{}".format(args.algo, args.algo)), "cfg/{}/config.yaml".format(args.algo), "cfg/shadow_hand_point_cloud.yaml"
    elif args.task == "ShadowHandDoorOpenOutwardLeft":
        return os.path.join(args.logdir, "shadow_hand_door_open_outward_left/{}/{}".format(args.algo, args.algo)), "cfg/{}/config.yaml".format(args.algo), "cfg/shadow_hand_point_cloud.yaml"
    elif args.task == "AllegroHandCatchOver2Underarm":
        return os.path.join(args.logdir, "allegro_hand_catch_over2underarm/{}/{}".format(args.algo, args.algo)), "cfg/{}/config.yaml".format(args.algo), "cfg/allegro_hand_catch_over2underarm.yaml"
    elif args.task == "AllegroHandCatchUnderarm2Joint":
        return os.path.join(args.logdir, "allegro_hand_catch_underarm_2joint/{}/{}".format(args.algo, args.algo)), "cfg/{}/config.yaml".format(args.algo), "cfg/allegro_hand_catch_underarm_2joint.yaml"
    elif args.task == "AllegroArmOrientationLatentSpace":
        return os.path.join(args.logdir, "allegro_hand_re_orientation/{}/{}".format(args.algo, args.algo)), "cfg/{}/config.yaml".format(args.algo), "cfg/allegro_hand_re_orientation.yaml"
    elif args.task == "AllegroHandLego":
        return os.path.join(args.logdir, "allegro_hand_lego/{}/{}".format(args.algo, args.algo)), "cfg/{}/config.yaml".format(args.algo), "cfg/allegro_hand_lego.yaml"
    else:
        warn_task_name()



def load_cfg(args, use_rlg_config=False):
    with open(os.path.join(os.getcwd(), args.cfg_train), 'r') as f:
        cfg_train = yaml.load(f, Loader=yaml.SafeLoader)

    with open(os.path.join(os.getcwd(), args.cfg_env), 'r') as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    # Override number of environments if passed on the command line
    if args.num_envs > 0:
        cfg["env"]["numEnvs"] = args.num_envs

    if args.episode_length > 0:
        cfg["env"]["episodeLength"] = args.episode_length

    cfg["name"] = args.task
    cfg["headless"] = args.headless

    # Set physics domain randomization
    if "task" in cfg:
        if "randomize" not in cfg["task"]:
            cfg["task"]["randomize"] = args.randomize
        else:
            cfg["task"]["randomize"] = args.randomize or cfg["task"]["randomize"]
    else:
        cfg["task"] = {"randomize": False}

    logdir = args.logdir
    if use_rlg_config:

        # Set deterministic mode
        if args.torch_deterministic:
            cfg_train["params"]["torch_deterministic"] = True

        exp_name = cfg_train["params"]["config"]['name']

        if args.experiment != 'Base':
            if args.metadata:
                exp_name = "{}_{}_{}_{}".format(args.experiment, args.task_type, args.device, str(args.physics_engine).split("_")[-1])

                if cfg["task"]["randomize"]:
                    exp_name += "_DR"
            else:
                exp_name = args.experiment

        # Override config name
        cfg_train["params"]["config"]['name'] = exp_name

        if args.resume > 0:
            cfg_train["params"]["load_checkpoint"] = True

        if args.checkpoint != "Base":
            cfg_train["params"]["load_path"] = args.checkpoint

        # Set maximum number of training iterations (epochs)
        if args.max_iterations > 0:
            cfg_train["params"]["config"]['max_epochs'] = args.max_iterations

        cfg_train["params"]["config"]["num_actors"] = cfg["env"]["numEnvs"]

        seed = cfg_train["params"].get("seed", -1)
        if args.seed is not None:
            seed = args.seed
        cfg["seed"] = seed
        cfg_train["params"]["seed"] = seed

        cfg["args"] = args
    else:

        # Set deterministic mode
        if args.torch_deterministic:
            cfg_train["torch_deterministic"] = True

        # Override seed if passed on the command line
        if args.seed is not None:
            cfg_train["seed"] = args.seed

        log_id = args.logdir
        if args.experiment != 'Base':
            if args.metadata:
                log_id = args.logdir + "_{}_{}_{}_{}".format(args.experiment, args.task_type, args.device, str(args.physics_engine).split("_")[-1])
                if cfg["task"]["randomize"]:
                    log_id += "_DR"
            else:
                log_id = args.logdir + "_{}".format(args.experiment)

        logdir = os.path.realpath(log_id)
        # os.makedirs(logdir, exist_ok=True)

    return cfg, cfg_train, logdir


def parse_sim_params(args, cfg, cfg_train):
    # initialize sim
    sim_params = gymapi.SimParams()
    sim_params.dt = 1./60.
    sim_params.num_client_threads = args.slices

    if args.physics_engine == gymapi.SIM_FLEX:
        if args.device != "cpu":
            print("WARNING: Using Flex with GPU instead of PHYSX!")
        sim_params.flex.shape_collision_margin = 0.01
        sim_params.flex.num_outer_iterations = 4
        sim_params.flex.num_inner_iterations = 10
    elif args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 0
        sim_params.physx.num_threads = 4
        sim_params.physx.use_gpu = args.use_gpu
        sim_params.physx.num_subscenes = args.subscenes
        sim_params.physx.max_gpu_contact_pairs = 8 * 1024 * 1024

    sim_params.use_gpu_pipeline = args.use_gpu_pipeline
    sim_params.physx.use_gpu = args.use_gpu

    # if sim options are provided in cfg, parse them and update/override above:
    if "sim" in cfg:
        gymutil.parse_sim_config(cfg["sim"], sim_params)

    # Override num_threads if passed on the command line
    if args.physics_engine == gymapi.SIM_PHYSX and args.num_threads > 0:
        sim_params.physx.num_threads = args.num_threads

    return sim_params


def get_args(benchmark=False, use_rlg_config=False):
    custom_parameters = [
        {"name": "--test", "action": "store_true", "default": False,
            "help": "Run trained policy, no training"},
        {"name": "--play", "action": "store_true", "default": False,
            "help": "Run trained policy, the same as test, can be used only by rl_games RL library"},
        {"name": "--resume", "type": int, "default": 0,
            "help": "Resume training or start testing from a checkpoint"},
        {"name": "--checkpoint", "type": str, "default": "Base",
            "help": "Path to the saved weights, only for rl_games RL library"},
        {"name": "--headless", "action": "store_true", "default": False,
            "help": "Force display off at all times"},
        {"name": "--horovod", "action": "store_true", "default": False,
            "help": "Use horovod for multi-gpu training, have effect only with rl_games RL library"},
        {"name": "--task", "type": str, "default": "ShadowHandOver",
            "help": "Can be BallBalance, Cartpole, CartpoleYUp, Ant, Humanoid, Anymal, FrankaCabinet, Quadcopter, ShadowHand, Ingenuity"},
        {"name": "--task_type", "type": str,
            "default": "Python", "help": "Choose Python or C++"},
        {"name": "--rl_device", "type": str, "default": "cuda:0",
            "help": "Choose CPU or GPU device for inferencing policy network"},
        {"name": "--logdir", "type": str, "default": "logs/"},
        {"name": "--experiment", "type": str, "default": "Base",
            "help": "Experiment name. If used with --metadata flag an additional information about physics engine, sim device, pipeline and domain randomization will be added to the name"},
        {"name": "--metadata", "action": "store_true", "default": False,
            "help": "Requires --experiment flag, adds physics engine, sim device, pipeline info and if domain randomization is used to the experiment name provided by user"},
        {"name": "--cfg_train", "type": str,
            "default": "Base"},
        {"name": "--cfg_env", "type": str, "default": "Base"},
        {"name": "--num_envs", "type": int, "default": 0,
            "help": "Number of environments to create - override config file"},
        {"name": "--episode_length", "type": int, "default": 0,
            "help": "Episode length, by default is read from yaml config"},
        {"name": "--seed", "type": int, "help": "Random seed"},
        {"name": "--max_iterations", "type": int, "default": 0,
            "help": "Set a maximum number of training iterations"},
        {"name": "--steps_num", "type": int, "default": -1,
            "help": "Set number of simulation steps per 1 PPO iteration. Supported only by rl_games. If not -1 overrides the config settings."},
        {"name": "--minibatch_size", "type": int, "default": -1,
            "help": "Set batch size for PPO optimization step. Supported only by rl_games. If not -1 overrides the config settings."},
        {"name": "--randomize", "action": "store_true", "default": False,
            "help": "Apply physics domain randomization"},
        {"name": "--torch_deterministic", "action": "store_true", "default": False,
            "help": "Apply additional PyTorch settings for more deterministic behaviour"},
        {"name": "--algo", "type": str, "default": "maddpg",
            "help": "Choose an algorithm"},
        {"name": "--model_dir", "type": str, "default": "",
            "help": "Choose a model dir"},
        ]

    if benchmark:
        custom_parameters += [{"name": "--num_proc", "type": int, "default": 1, "help": "Number of child processes to launch"},
                              {"name": "--random_actions", "action": "store_true",
                                  "help": "Run benchmark with random actions instead of inferencing"},
                              {"name": "--bench_len", "type": int, "default": 10,
                                  "help": "Number of timing reports"},
                              {"name": "--bench_file", "action": "store", "help": "Filename to store benchmark results"}]

    # parse arguments
    args = gymutil.parse_arguments(
        description="RL Policy",
        custom_parameters=custom_parameters)

    # allignment with examples
    args.device_id = args.compute_device_id
    args.device = args.sim_device_type if args.use_gpu_pipeline else 'cpu'

    if args.test:
        args.play = args.test
        args.train = False
    elif args.play:
        args.train = False
    else:
        args.train = True

    logdir, cfg_train, cfg_env = retrieve_cfg(args, use_rlg_config)

    if use_rlg_config == False:
        if args.horovod:
            print("Distributed multi-gpu training with Horovod is not supported by rl-pytorch. Use rl_games for distributed training.")
        if args.steps_num != -1:
            print("Setting number of simulation steps per iteration from command line is not supported by rl-pytorch.")
        if args.minibatch_size != -1:
            print("Setting minibatch size from command line is not supported by rl-pytorch.")
        if args.checkpoint != "Base":
            raise ValueError("--checkpoint is not supported by rl-pytorch. Please use --resume <iteration number>")

    # use custom parameters if provided by user
    if args.logdir == "logs/":
        args.logdir = logdir

    if args.cfg_train == "Base":
        args.cfg_train = cfg_train

    if args.cfg_env == "Base":
        args.cfg_env = cfg_env

    # if args.algo not in ["maddpg", "happo", "mappo", "hatrpo","ippo","ppo","sac","td3","ddpg","trpo"]:
    #     warn_algorithm_name()

    return args
