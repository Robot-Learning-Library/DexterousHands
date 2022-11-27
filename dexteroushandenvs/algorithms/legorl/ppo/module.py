import numpy as np
import math

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
# from algorithms.utils.mani_skill_learn.networks.backbones.pointnet import getPointNet
from algorithms.legorl.ppo.cnn import CNNLayer

class ActorCritic(nn.Module):

    def __init__(self, obs_shape, states_shape, actions_shape, initial_std, model_cfg, asymmetric=False):
        super(ActorCritic, self).__init__()

        self.asymmetric = asymmetric

        if model_cfg is None:
            actor_hidden_dim = [256, 256, 256]
            critic_hidden_dim = [256, 256, 256]
            activation = get_activation("selu")
        else:
            actor_hidden_dim = model_cfg['pi_hid_sizes']
            critic_hidden_dim = model_cfg['vf_hid_sizes']
            activation = get_activation(model_cfg['activation'])

        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(*obs_shape, actor_hidden_dim[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dim)):
            if l == len(actor_hidden_dim) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dim[l], *actions_shape))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dim[l], actor_hidden_dim[l + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        if self.asymmetric:
            critic_layers.append(nn.Linear(*states_shape, critic_hidden_dim[0]))
        else:
            critic_layers.append(nn.Linear(*obs_shape, critic_hidden_dim[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dim)):
            if l == len(critic_hidden_dim) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dim[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dim[l], critic_hidden_dim[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(self.actor)
        print(self.critic)

        # Action noise
        self.log_std = nn.Parameter(np.log(initial_std) * torch.ones(*actions_shape))

        # Initialize the weights like in stable baselines
        actor_weights = [np.sqrt(2)] * len(actor_hidden_dim)
        actor_weights.append(0.01)
        critic_weights = [np.sqrt(2)] * len(critic_hidden_dim)
        critic_weights.append(1.0)
        self.init_weights(self.actor, actor_weights)
        self.init_weights(self.critic, critic_weights)

    @staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def forward(self):
        raise NotImplementedError

    def act(self, observations, states):
        actions_mean = self.actor(observations)

        covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        distribution = MultivariateNormal(actions_mean, scale_tril=covariance)

        actions = distribution.sample()
        actions_log_prob = distribution.log_prob(actions)
        if self.asymmetric:
            value = self.critic(states)
        else:
            value = self.critic(observations)

        return actions.detach(), actions_log_prob.detach(), value.detach(), actions_mean.detach(), self.log_std.repeat(actions_mean.shape[0], 1).detach()

    def act_inference(self, observations):
        actions_mean = self.actor(observations)
        return actions_mean

    def evaluate(self, observations, states, actions):
        actions_mean = self.actor(observations)

        covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        distribution = MultivariateNormal(actions_mean, scale_tril=covariance)

        actions_log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()

        if self.asymmetric:
            value = self.critic(states)
        else:
            value = self.critic(observations)

        return actions_log_prob, entropy, value, actions_mean, self.log_std.repeat(actions_mean.shape[0], 1)

class ActorCriticCNN(nn.Module):

    def __init__(self, envs, obs_shape, states_shape, actions_shape, initial_std, model_cfg, asymmetric=False):
        super(ActorCriticCNN, self).__init__()

        self.asymmetric = asymmetric

        self.obs_dim = 62
        self.state_dim = 108
        self.feature_dim = 32

        actor_hidden_dim = model_cfg['pi_hid_sizes']
        critic_hidden_dim = model_cfg['vf_hid_sizes']
        activation = get_activation(model_cfg['activation'])

        actor_layers = []
        actor_layers.append(nn.Linear(self.feature_dim + self.obs_dim * 2, actor_hidden_dim[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dim)):
            if l == len(actor_hidden_dim) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dim[l], *actions_shape))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dim[l], actor_hidden_dim[l + 1]))
                actor_layers.append(activation)
        self.actor1 = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        if self.asymmetric:
            critic_layers.append(nn.Linear(self.feature_dim + self.state_dim * 2, critic_hidden_dim[0]))
        else:
            critic_layers.append(nn.Linear(self.feature_dim + self.state_dim * 2, critic_hidden_dim[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dim)):
            if l == len(critic_hidden_dim) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dim[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dim[l], critic_hidden_dim[l + 1]))
                critic_layers.append(activation)
        self.critic1 = nn.Sequential(*critic_layers)

        # class CNNBase(nn.Module):
        #     def __init__(self, obs_shape, feature_dim):
        #         super(CNNBase, self).__init__()

        #         self._use_orthogonal = True
        #         self._use_ReLU = True
        #         self.hidden_size = feature_dim

        #         self.cnn = CNNLayer(obs_shape, self.hidden_size, self._use_orthogonal, self._use_ReLU)

        #     def forward(self, x):
        #         x = self.cnn(x)
        #         return x

        # self.cnn = CNNBase(obs_shape=(4, 128, 128), feature_dim=self.feature_dim)
        class ShallowConv(nn.Module):
            """
            A shallow convolutional encoder from https://rll.berkeley.edu/dsae/dsae.pdf
            """
            def __init__(self, input_channel=3, output_channel=32):
                super(ShallowConv, self).__init__()
                self._input_channel = input_channel
                self._output_channel = output_channel
                self.nets = nn.Sequential(
                    torch.nn.Conv2d(input_channel, 16, kernel_size=7, stride=2, padding=3),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(16, 4, kernel_size=1, stride=1, padding=0),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(4, output_channel, kernel_size=3, stride=1, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.Flatten(),
                    torch.nn.Linear(128*128, 32),
                )

            def output_shape(self, input_shape):
                """
                Function to compute output shape from inputs to this module. 
                Args:
                    input_shape (iterable of int): shape of input. Does not include batch dimension.
                        Some modules may not need this argument, if their output does not depend 
                        on the size of the input, or if they assume fixed size input.
                Returns:
                    out_shape ([int]): list of integers corresponding to output shape
                """
                assert(len(input_shape) == 3)
                assert(input_shape[0] == self._input_channel)
                out_h = int(math.floor(input_shape[1] / 2.))
                out_w = int(math.floor(input_shape[2] / 2.))
                return [self._output_channel, out_h, out_w]

            def forward(self, inputs):
                x = self.nets(inputs)

                return x

        self.cnn = ShallowConv(input_channel=4, output_channel=4)

        class ConcatNet(nn.Module):
            def __init__(self, cnn, fc, state_dim):
                super(ConcatNet, self).__init__()
                self.cnn = cnn
                self.fc = fc
                self.split_dim = state_dim
                self.pre_fc = nn.Linear(state_dim, state_dim * 2)
            
            def forward(self, input):
                x1 = input[:, self.split_dim:].reshape(-1, 128, 128, 4)
                x1 = x1.permute(0, 3, 1, 2)
                x2 = input[:, :self.split_dim]

                y1 = self.cnn(x1)
                y2 = self.pre_fc(x2)
                inp = torch.cat((y1, y2), dim=1)
                oup = self.fc(inp)
                return oup

        self.actor = ConcatNet(self.cnn, self.actor1, self.obs_dim)
        self.critic = ConcatNet(self.cnn, self.critic1, self.state_dim)
        # self.critic = self.critic1

        print(self.actor)
        print(self.critic)

        # Action noise
        self.log_std = nn.Parameter(np.log(initial_std) * torch.ones(*actions_shape))

        # Initialize the weights like in stable baselines
    #     actor_weights = [np.sqrt(2)] * len(actor_hidden_dim)
    #     actor_weights.append(0.01)
    #     critic_weights = [np.sqrt(2)] * len(critic_hidden_dim)
    #     critic_weights.append(1.0)
    #     self.init_weights(self.actor, actor_weights)
    #     self.init_weights(self.critic, critic_weights)

    # @staticmethod
    # def init_weights(sequential, scales):
    #     [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
    #      enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def forward(self):
        raise NotImplementedError

    def act(self, observations, states):
        actions_mean = self.actor(observations)

        covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        distribution = MultivariateNormal(actions_mean, scale_tril=covariance)

        actions = distribution.sample()
        actions_log_prob = distribution.log_prob(actions)
        if self.asymmetric:
            value = self.critic(states)
        else:
            value = self.critic(observations)

        return actions.detach(), actions_log_prob.detach(), value.detach(), actions_mean.detach(), self.log_std.repeat(actions_mean.shape[0], 1).detach()

    def act_inference(self, observations):
        actions_mean = self.actor(observations)
        return actions_mean

    def evaluate(self, observations, states, actions):
        actions_mean = self.actor(observations)

        covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        distribution = MultivariateNormal(actions_mean, scale_tril=covariance)

        actions_log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()

        if self.asymmetric:
            value = self.critic(states)
        else:
            value = self.critic(observations)

        return actions_log_prob, entropy, value, actions_mean, self.log_std.repeat(actions_mean.shape[0], 1)

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
