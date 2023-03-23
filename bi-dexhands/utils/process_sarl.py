from algorithms.rl.ppo import PPO
from algorithms.rl.dppo import DPPO
from algorithms.rl.sac import SAC
from algorithms.rl.td3 import TD3
from algorithms.rl.ddpg import DDPG
from algorithms.rl.trpo import TRPO

def process_sarl(args, env, cfg_train, logdir):
    learn_cfg = cfg_train["learn"]
    print('args: ', args)
    cfg_train['record_traj'] = args.record_traj
    if args.model_dir is not None and len(args.model_dir) > 0:
        # /logs/ShadowHand/ppo/ppo_seed3/model_20000.pt
        train_seed = args.model_dir.split("/")[-2].split("seed")[-1]
        checkpoint = args.model_dir.split("/")[-1].split(".")[0].split("_")[-1]
    else:
        train_seed = ''
        checkpoint = ''
    if args.record_traj_path is None:
        cfg_train['record_traj_path'] = f"{args.record_video_path}/{args.task}/{args.task}_{args.algo}_{train_seed}_{args.save_time_stamp}_{checkpoint}_"  # same path as video
    else:
        cfg_train['record_traj_path'] = f"{args.record_traj_path}/{args.task}/{args.task}_{args.algo}_{train_seed}_{args.save_time_stamp}_{checkpoint}_"
    is_testing = learn_cfg["test"]
    # is_testing = True
    # Override resume and testing flags if they are passed as parameters.
    if args.model_dir != "":
        # is_testing = True  # load model does not necesssarially test
        chkpt_path = args.model_dir

    if args.max_iterations != -1:
        cfg_train["learn"]["max_iterations"] = args.max_iterations
        
    logdir = logdir + "_seed{}".format(env.task.cfg["seed"])    
    
    # Only useful for dppo
    learn_cfg["learned_seed"] = args.learned_seed

    """Set up the algo system for training or inferencing."""
    model = eval(args.algo.upper())(vec_env=env,
              cfg_train = cfg_train,
              device=env.rl_device,
              sampler=learn_cfg.get("sampler", 'sequential'),
              log_dir=logdir,
              is_testing=is_testing,
              print_log=learn_cfg["print_log"],
              apply_reset=False,
              asymmetric=(env.num_states > 0)
              )

    # ppo.test("/home/hp-3070/logs/demo/scissors/ppo_seed0/model_6000.pt")
    if is_testing and args.model_dir != "":
        print("Loading model from {}".format(chkpt_path))
        model.test(chkpt_path)
    elif args.model_dir != "":
        print("Loading model from {}".format(chkpt_path))
        model.load(chkpt_path, current_learning_iteration=0)  # to still learn the desired iterations

    return model