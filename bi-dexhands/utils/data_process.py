import numpy as np

def data_process_per_sample(task_name, obs, act, obs_dim=24, act_dim=20):
    # process observation
    if task_name == 'ShadowHandOver':
        right_hand_obs_idx = np.arange(24).tolist()
        left_hand_obs_idx = np.arange(187, 211).tolist()
    elif task_name == 'ShadowHand':
        right_hand_obs_idx = np.arange(24).tolist()  # only right hand
        left_hand_obs_idx = []
    else:
        right_hand_obs_idx = np.arange(24).tolist()
        left_hand_obs_idx = np.arange(199, 223).tolist()
    obs_idx = right_hand_obs_idx + left_hand_obs_idx
    # prcess action
    if task_name == 'ShadowHandOver':
        right_hand_action_idx = np.arange(20).tolist()  # only care 20 joints on hand
        left_hand_action_idx = np.arange(20, 40).tolist()
    elif task_name == 'ShadowHand':
        right_hand_action_idx = np.arange(20).tolist()  # only care 20 joints on hand
        left_hand_action_idx = []
    else:
        right_hand_action_idx = np.arange(6, 26).tolist()  # only care 20 joints on hand
        left_hand_action_idx = np.arange(32, 52).tolist()
    action_idx = right_hand_action_idx + left_hand_action_idx
    obs = obs[:, obs_idx]  # (episode_length, dim)
    act = act[:, action_idx]
    if obs.shape[1] ==  2*obs_dim and act.shape[1] == 2*act_dim:
        obs = obs.view(-1, 2, obs_dim)
        obs = obs.view(-1, obs_dim)
        act = act.view(-1, 2, act_dim)
        act = act.view(-1, act_dim)
    # print(obs.shape, act.shape)
    return obs, act

def data_process(traj, env):
    # process observation
    if env == 'ShadowHandOver':
        right_hand_obs_idx = np.arange(24).tolist()
        left_hand_obs_idx = np.arange(187, 211).tolist()
    elif env == 'ShadowHand':
        right_hand_obs_idx = np.arange(24).tolist()  # only right hand
        left_hand_obs_idx = []
    else:
        right_hand_obs_idx = np.arange(24).tolist()
        left_hand_obs_idx = np.arange(199, 223).tolist()
    obs_idx = right_hand_obs_idx + left_hand_obs_idx
    # prcess action
    if env == 'ShadowHandOver':
        right_hand_action_idx = np.arange(20).tolist()  # only care 20 joints on hand
        left_hand_action_idx = np.arange(20, 40).tolist()
    elif env == 'ShadowHand':
        right_hand_action_idx = np.arange(20).tolist()  # only care 20 joints on hand
        left_hand_action_idx = []
    else:
        right_hand_action_idx = np.arange(6, 26).tolist()  # only care 20 joints on hand
        left_hand_action_idx = np.arange(32, 52).tolist()
    action_idx = right_hand_action_idx + left_hand_action_idx
    obs = np.array(traj['obs']).squeeze()[:, obs_idx]  # (episode_length, dim)
    act = np.array(traj['actions']).squeeze()[:, action_idx]
    # print(obs.shape, act.shape)
    return obs, act