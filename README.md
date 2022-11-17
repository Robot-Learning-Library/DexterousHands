# Bimanual Dexterous Manipulation via <br /> Multi-Agent Reinforcement Learning

<img src="assets/image_folder/cover.jpg" width="1000" border="1"/>

****
### About this repository

Dexterous manipulaiton as a common but challenging task has attracted a great deal of interest in the field of robotics. Thanks to the intersection of reinforcement learning and robotics, previous study achieves a good performance on unimanual dexterous manipulaiton. However, how to balance between hand dexterity and bimanual coordination remains an open challenge. Therefore, we provided a novel benchmark for researchers to study machine intelligence. 

Bi-DexMani is a collection of environments and algorithms for learning bimanual dexterous manipulation. 

This repository contains complex dexterous hand RL environments DexterousHandEnvs for the NVIDIA Isaac Gym high performance environments. DexterousHandEnvs is a very challenging dexterous hand manipulation environment for multi-agent reinforcement learning. We refer to some designs of existing multi-agent and dexterous hand environments, integrate their advantages, expand some new environments and unique features for multi-agent reinforcement learning. Our environments focus on the application of multi-agent algorithms to dexterous hand control, which is very challenging in traditional control algorithms. 

The difficulty of our environment is not only reflected in the challenging task content but also reflected in the ultra-high-dimensional continuous space control. The state space dimension of each environment is up to 400 dimensions in total, and the action space dimension is up to 40 dimensions. A highlight of our environment is that we use five fingers and palms of each hand as a minimum unit, you can use each finger and palm as an agent, or combine any number of them as an agent by yourself.

:star2::star2:**Click [here](#task) to check the environment introduction!**:star2::star2:  

- [Installation](#Installation)
  - [Pre-requisites](#Installation)
  - [Install from PyPI](#Install-from-PyPI)
  - [Install from source code](#Install-from-source-code)
- [Introduction to Bi-DexHands](#Introduction-to-Bi-DexHands)
  - [Demos](#Demos)
- [File Structure](#File-Structure)
- [Overview of Environments](./docs/environments.md)
- [Overview of Algorithms](./docs/algorithms.md)
- [Getting Started](#Getting-Started)
  - [Tasks](#Tasks)
  - [Training](#Training)
  - [Testing](#Testing)
  - [Plotting](#Plotting)
- [Enviroments Performance](#Enviroments-Performance)
  - [Figures](#Figures)
- [Building the Documentation](#Building-the-Documentation)
- [The Team](#The-Team)
- [License](#License)
<br></br>
****
## Installation

Details regarding installation of IsaacGym can be found [here](https://developer.nvidia.com/isaac-gym). **We currently support the `Preview Release 3` version of IsaacGym.**

### Pre-requisites

The code has been tested on Ubuntu 18.04 with Python 3.7. The minimum recommended NVIDIA driver
version for Linux is `470.74` (dictated by support of IsaacGym).

It uses [Anaconda](https://www.anaconda.com/) to create virtual environments.
To install Anaconda, follow instructions [here](https://docs.anaconda.com/anaconda/install/linux/).

Ensure that Isaac Gym works on your system by running one of the examples from the `python/examples` 
directory, like `joint_monkey.py`. Follow troubleshooting steps described in the Isaac Gym Preview 3 
install instructions if you have any trouble running the samples.

Once Isaac Gym is installed and samples work within your current python environment, install this repo:

#### Install from PyPI
Bi-DexHands is hosted on PyPI. It requires Python >= 3.6.
You can simply install Bi-DexHands from PyPI with the following command:

```bash
pip install bi-dexhands
```
#### Install from source code
You can also install this repo from source code:

```bash
pip install -e .
```

## Introduciton

This repository contains complex dexterous hand RL environments Bi-DexHands for the NVIDIA Isaac Gym high performance environments. Bi-DexHands is a  challenging dexterous hand manipulation environment for multi-agent reinforcement learning. We refer to some designs of existing multi-agent and dexterous hand environments, integrate their advantages, expand some new environments and unique features for multi-agent reinforcement learning. Our environments focus on the application of multi-agent algorithms to dexterous hand control, which is very challenging in traditional control algorithms. 

### Demos
<!-- <center class="half">
    <img src="assets/image_folder/0.gif" align="center" width="500"/><img src="assets/image_folder/3.gif" align="center" width="500"/>
</center> -->
<table><tr>
<td><img src="assets/image_folder/0.gif" border=0 width="500"></td>
<td><img src="assets/image_folder/3.gif" border=0 width="475"></td>
</tr></table>

For more demos please refer to [here](./docs/environments.md)

## Getting Started

### <span id="task">Tasks</span>

Source code for tasks can be found in `envs/tasks`. For the detailed settings of state/action/reward  please refer to [here](./docs/environments.md)

Until now we only suppose the following environments:

| Environments | Description | Demo     |
|  :----:  | :----:  | :----:  |
|ShadowHand Over| These environments involve two fixed-position hands. The hand which starts with the object must find a way to hand it over to the second hand. | <img src="assets/image_folder/0.gif" width="1000"/>    |
|ShadowHandCatch Underarm|These environments again have two hands, however now they have some additional degrees of freedom that allows them to translate/rotate their centre of masses within some constrained region. | <img src="assets/image_folder/4.gif" align="middle" width="1000"/>    |
|ShadowHandCatch Over2Underarm| This environment is is made up of half ShadowHandCatchUnderarm and half ShadowHandCatchOverarm, the object needs to be thrown from the vertical hand to the palm-up hand | <img src="assets/image_folder/2.gif" align="middle" width="1000"/>    |
|ShadowHandCatch Abreast| This environment is similar to ShadowHandCatchUnderarm, the difference is that the two hands are changed from relative to side-by-side posture. | <img src="assets/image_folder/1.gif" align="middle" width="1000"/>    |
|ShadowHandCatch TwoCatchUnderarm| These environments involve coordination between the two hands so as to throw the two objects between hands (i.e. swapping them). | <img src="assets/image_folder/two_catch.gif" align="middle" width="1000"/>    |
|ShadowHandLift Underarm | This environment requires grasping the pot handle with two hands and lifting the pot to the designated position  | <img src="assets/image_folder/3.gif" align="middle" width="1000"/>    |
|ShadowHandDoor OpenInward | This environment requires the closed door to be opened, and the door can only be pulled inwards | <img src="assets/image_folder/open_inward.gif" align="middle" width="1000"/>    |
|ShadowHandDoor OpenOutward | This environment requires a closed door to be opened and the door can only be pushed outwards  | <img src="assets/image_folder/open_outward.gif" align="middle" width="1000"/>    |
|ShadowHandDoor CloseInward | This environment requires the open door to be closed, and the door is initially open inwards | <img src="assets/image_folder/close_inward.gif" align="middle" width="1000"/>    |
|ShadowHand BottleCap | This environment involve two hands and a bottle, we need to hold the bottle with one hand and open the bottle cap with the other hand  | <img src="assets/image_folder/bottle_cap.gif" align="middle" width="1000"/>    |
<!-- |ShadowHandDoor CloseOutward | This environment requires the open door to be closed, and the door is initially open outwards  | <img src="assets/image_folder/sendpix0.gif" align="middle" width="1000"/>    | -->

### Training

#### Gym-Like API

We provide a Gym-Like API that allows us to get information from the isaac-gym environment. Our single-agent Gym-Like wrapper is the code of the Isaacgym team used, and we have developed a multi-agent Gym-Like wrapper based on it:

```python
class MultiVecTaskPython(MultiVecTask):
    # Get environment state information
    def get_state(self):
        return torch.clamp(self.task.states_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

    def step(self, actions):
        # Stack all agent actions in order and enter them into the environment
        a_hand_actions = actions[0]
        for i in range(1, len(actions)):
            a_hand_actions = torch.hstack((a_hand_actions, actions[i]))
        actions = a_hand_actions
        # Clip the actions
        actions_tensor = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        self.task.step(actions_tensor)
        # Obtain information in the environment and distinguish the observation of different agents by hand
        obs_buf = torch.clamp(self.task.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)
        hand_obs = []
        hand_obs.append(torch.cat([obs_buf[:, :self.num_hand_obs], obs_buf[:, 2*self.num_hand_obs:]], dim=1))
        hand_obs.append(torch.cat([obs_buf[:, self.num_hand_obs:2*self.num_hand_obs], obs_buf[:, 2*self.num_hand_obs:]], dim=1))
        rewards = self.task.rew_buf.unsqueeze(-1).to(self.rl_device)
        dones = self.task.reset_buf.to(self.rl_device)
        # Organize information into Multi-Agent RL format
        # Refer to https://github.com/tinyzqh/light_mappo/blob/HEAD/envs/env.py
        sub_agent_obs = []
        ...
        sub_agent_done = []
        for i in range(len(self.agent_index[0] + self.agent_index[1])):
            ...
            sub_agent_done.append(dones)
        # Transpose dim-0 and dim-1 values
        obs_all = torch.transpose(torch.stack(sub_agent_obs), 1, 0)
        ...
        done_all = torch.transpose(torch.stack(sub_agent_done), 1, 0)
        return obs_all, state_all, reward_all, done_all, info_all, None

    def reset(self):
        # Use a random action as the first action after the environment reset
        actions = 0.01 * (1 - 2 * torch.rand([self.task.num_envs, self.task.num_actions * 2], dtype=torch.float32, device=self.rl_device))
        # step the simulator
        self.task.step(actions)
        # Get the observation and state buffer in the environment, the detailed are the same as step(self, actions)
        obs_buf = torch.clamp(self.task.obs_buf, -self.clip_obs, self.clip_obs)
        ...
        obs = torch.transpose(torch.stack(sub_agent_obs), 1, 0)
        state_all = torch.transpose(torch.stack(agent_state), 1, 0)
        return obs, state_all, None
```
#### RL/Multi-Agent RL API

Similar to the Gym-Like wrapper, we also provide single-agent and multi-agent RL algorithms respectively. In order to adapt to Isaac Gym and speed up the running speed, all operations are done on the GPU using tensor, so there is no need to transfer data between the CPU and GPU, which greatly speeds up the operation.

We give an example to illustrate multi-agent RL APIs, which mainly refer to [https://github.com/cyanrain7/TRPO-in-MARL](https://github.com/cyanrain7/TRPO-in-MARL):

```python
# warmup before the main loop starts
self.warmup()
# log data
start = time.time()
episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
train_episode_rewards = torch.zeros(1, self.n_rollout_threads, device=self.device)
# main loop
for episode in range(episodes):
    if self.use_linear_lr_decay:
        self.trainer.policy.lr_decay(episode, episodes)
    done_episodes_rewards = []
    for step in range(self.episode_length):
        # Sample actions
        values, actions, action_log_probs, rnn_states, rnn_states_critic = self.collect(step)
        # Obser reward and next obs
        obs, share_obs, rewards, dones, infos, _ = self.envs.step(actions)
        dones_env = torch.all(dones, dim=1)
        reward_env = torch.mean(rewards, dim=1).flatten()
        train_episode_rewards += reward_env
        # Record reward at the end of each episode
        for t in range(self.n_rollout_threads):
            if dones_env[t]:
                done_episodes_rewards.append(train_episode_rewards[:, t].clone())
                train_episode_rewards[:, t] = 0

        data = obs, share_obs, rewards, dones, infos, \
                values, actions, action_log_probs, \
                rnn_states, rnn_states_critic
        # insert data into buffer
        self.insert(data)

    # compute return and update network
    self.compute()
    train_infos = self.train()
    # post process
    total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
    # save model
    if (episode % self.save_interval == 0 or episode == episodes - 1):
        self.save()
```

#### Training Example

For example, if you want to train your first policy with ShadowHandOver task and PPO algorithm, run this line:

```bash
python train.py --task=ShadowHandOver --algo=ppo
```

To select an algorithm, pass `--algo=ppo/mappo/happo/hatrpo/...` 
as an argument. For example, if you want to use happo algorithm, run this line:

```bash
python train.py --task=ShadowHandOver --algo=hatrpo
``` 

Supported Single-Agent RL algorithms are listed below:

Single-Agent RL: **PPO, TRPO, SAC, TD3, DDPG** 

Multi-Agent RL: **IPPO, MAPPO, MADDPG, HATRPO, HAPPO**

- [Heterogeneous-Agent Proximal Policy Optimization (HAPPO)](https://arxiv.org/pdf/2109.11251.pdf)
- [Heterogeneous-Agent Trust Region Policy Optimization (HATRPO)](https://arxiv.org/pdf/2109.11251.pdf)
- [Multi-Agent Proximal Policy Optimization (MAPPO)](https://arxiv.org/pdf/2103.01955.pdf)
- [Independent Proximal Policy Optimization (IPPO)](https://arxiv.org/pdf/2011.09533.pdf)
- [Multi-Agent Deep Deterministic Policy Gradient  (MADDPG)](https://arxiv.org/pdf/1706.02275.pdf)

For a brief introduction to these algorithms, please refer to [here](./docs/algorithms.md)

### Testing

The trained model will be saved to `logs/${Task Name}/${Algorithm Name}`folder.

To load a trained model and only perform inference (no training), pass `--test` 
as an argument, and pass `--model_dir` to specify the trained models which you want to load.
For single-agent reinforcement learning, you need to pass `--model_dir` to specify exactly what .pt model you want to load. An example is as follows:

```bash
python train.py --task=ShadowHandOver --model_dir=logs/shadow_hand_over/ppo/ppo_seed0/model_5000.pt --test
```

For multi-agent reinforcement learning, pass `--model_dir` to specify the path to the folder where all your agent model files are saved. An example is as follows:

```bash
python train.py --task=ShadowHandOver --model_dir=logs/shadow_hand_over/happo/models_seed0 --test
```

### Plotting

After training, you can convert all tfevent files into csv files and then try plotting the results.

```bash
# geenrate csv
$ python ./utils/logger/tools.py --root-dir ./logs/shadow_hand_over --refresh
# generate figures
$ python ./utils/logger/plotter.py --root-dir ./logs/shadow_hand_over --shaded-std --legend-pattern "\\w+"  --output-path=./logs/shadow_hand_over/figure.png
```

## Enviroment Performance

### Figures

We provide stable and reproducible baselins run by **PPO, HAPPO, MAPPO** algorithms. All baselines are run under the parameters of `2048 num_env` and `100M total_step`. 

<table>
    <tr>
        <th colspan="2">ShadowHandOver</th>
        <th colspan="2">ShadowHandLiftUnderarm</th>
    <tr>
    <tr>
        <td><img src="assets/image_folder/0.gif" align="middle" width="750"/></td>
        <td><img src="assets/image_folder/figures/shadow_hand_over.png" align="middle" width="750"/></td>
        <td><img src="assets/image_folder/3.gif" align="middle" width="750"/></td>
        <td><img src="assets/image_folder/figures/shadow_hand_lift_underarm.png" align="middle" width="750"/></td>
    <tr>
    <tr>
        <th colspan="2">ShadowHandCatchUnderarm</th>
        <th colspan="2">ShadowHandDoorOpenInward</th>
    <tr>
    <tr>
        <td><img src="assets/image_folder/4.gif" align="middle" width="750"/></td>
        <td><img src="assets/image_folder/figures/shadow_hand_catch_underarm.png" align="middle" width="750"/></td>
        <td><img src="assets/image_folder/open_inward.gif" align="middle" width="750"/></td>
        <td><img src="assets/image_folder/figures/shadow_hand_door_open_inward.png" align="middle" width="750"/></td>
    <tr>
    <tr>
        <th colspan="2">ShadowHandCatchOver2Underarm</th>
        <th colspan="2">ShadowHandDoorOpenOutward</th>
    <tr>
    <tr>
        <td><img src="assets/image_folder/2.gif" align="middle" width="750"/></td>
        <td><img src="assets/image_folder/figures/shadow_hand_catch_over2underarm.png" align="middle" width="750"/></td>
        <td><img src="assets/image_folder/open_outward.gif" align="middle" width="750"/></td>
        <td><img src="assets/image_folder/figures/shadow_hand_door_open_outward.png" align="middle" width="750"/></td>
    <tr>
    <tr>
        <th colspan="2">ShadowHandCatchAbreast</th>
        <th colspan="2">ShadowHandDoorCloseInward</th>
    <tr>
    <tr>
        <td><img src="assets/image_folder/1.gif" align="middle" width="750"/></td>
        <td><img src="assets/image_folder/figures/shadow_hand_catch_abreast.png" align="middle" width="750"/></td>
        <td><img src="assets/image_folder/close_inward.gif" align="middle" width="750"/></td>
        <td><img src="assets/image_folder/figures/shadow_hand_door_close_inward.png" align="middle" width="750"/></td>
    <tr>
    <tr>
        <th colspan="2">ShadowHandCatchTwoCatchUnderarm</th>
        <th colspan="2">ShadowHandDoorCloseOutward</th>
    <tr>
    <tr>
        <td><img src="assets/image_folder/two_catch.gif" align="middle" width="750"/></td>
        <td><img src="assets/image_folder/figures/shadow_hand_two_catch_underarm.png" align="middle" width="750"/></td>
        <td><img src="assets/image_folder/close_inward.gif" align="middle" width="750"/></td>
        <td><img src="assets/image_folder/figures/shadow_hand_door_close_outward.png" align="middle" width="750"/></td>
    <tr>
</table>


For more figures please refer to [here](./docs/figures.md)

## Building the Documentation

To build documentation in various formats, you will need [Sphinx](http://www.sphinx-doc.org) and the
readthedocs theme.

```bash
cd docs/
pip install -r requirements.txt
```
You can then build the documentation by running `make <format>` from the
`docs/` folder. Run `make` to get a list of all available output formats.

If you get a katex error run `npm install katex`.  If it persists, try
`npm install -g katex`

## Future Plan

- **Tasks under development**:  
  - [x] Handover, throw&catch (a 9-month-old child's behavior)
  - [ ] Pick up eyeglasses  ( an 1-year-old child's behavior )
  - [ ] Stack blocks (an 1-year-old child's behavior )
  - [ ] Put off a pen cap (a 30-month-old child's behavior)
  - [x] Open/Close a door (a 30-month-old child's behavior)
  - [ ] Unscrew a bottle top (a 30-month-old child's behavior)
  - [x] Lift a pot (a 2-year-old child's behavior)
  - [ ] Turn buttons off/on (a 5-year-old child's behavior)
  - [ ] Pour water in a teapot (an adult's behavior)

- **Meta/Multi-task algorithms**:
  - [ ] Multi-task PPO
  - [ ] Multi-task TRPO
  - [ ] Multi-task SAC
  - [ ] MAML
  - [ ] RL<sup>2 </sup>
  - [ ] PEARL




## The Team

DexterousHands is a PKU-MARL project under the leadership of Dr. [Yaodong Yang](https://www.yangyaodong.com/), it is currently maintained by [Yuanpei Chen](https://github.com/cypypccpy) and [Shengjie Wang](https://github.com/Shengjie-bob). 

It must be mentioned that in our development process, we mainly refer to the following two open source repositories: 

[https://github.com/NVIDIA-Omniverse/IsaacGymEnvs](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs) 

[https://github.com/cyanrain7/TRPO-in-MARL](https://github.com/cyanrain7/TRPO-in-MARL) 


## License

DexterousHands has a Apache license, as found in the [LICENSE](LICENSE) file.
