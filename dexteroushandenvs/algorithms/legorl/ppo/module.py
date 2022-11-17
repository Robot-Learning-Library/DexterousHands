import numpy as np

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

    def __init__(self, obs_shape, states_shape, actions_shape, initial_std, model_cfg, asymmetric=False):
        super(ActorCriticCNN, self).__init__()

        self.asymmetric = asymmetric

        self.state_dim = 75
        self.feature_dim = 16

        actor_hidden_dim = model_cfg['pi_hid_sizes']
        critic_hidden_dim = model_cfg['vf_hid_sizes']
        activation = get_activation(model_cfg['activation'])

        actor_layers = []
        actor_layers.append(nn.Linear(self.feature_dim + self.state_dim, actor_hidden_dim[0]))
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
            critic_layers.append(nn.Linear(*states_shape, critic_hidden_dim[0]))
        else:
            critic_layers.append(nn.Linear(self.feature_dim + self.state_dim, critic_hidden_dim[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dim)):
            if l == len(critic_hidden_dim) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dim[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dim[l], critic_hidden_dim[l + 1]))
                critic_layers.append(activation)
        self.critic1 = nn.Sequential(*critic_layers)

        class CNNBase(nn.Module):
            def __init__(self, obs_shape, feature_dim):
                super(CNNBase, self).__init__()

                self._use_orthogonal = True
                self._use_ReLU = True
                self.hidden_size = feature_dim

                self.cnn = CNNLayer(obs_shape, self.hidden_size, self._use_orthogonal, self._use_ReLU)

            def forward(self, x):
                x = self.cnn(x)
                return x

        self.cnn = CNNBase(obs_shape=(4, 192, 256), feature_dim=self.feature_dim)

        class ConcatNet(nn.Module):
            def __init__(self, cnn, fc, state_dim):
                super(ConcatNet, self).__init__()
                self.cnn = cnn
                self.fc = fc
                self.split_dim = state_dim
                self.pre_fc = nn.Linear(state_dim, state_dim)
            
            def forward(self, input):
                x1 = input[:, self.split_dim:].reshape(-1, 192, 256, 4)
                x1 = x1.permute(0, 3, 1, 2)
                x2 = input[:, :self.split_dim]

                y1 = self.cnn(x1)
                y2 = self.pre_fc(x2)
                inp = torch.cat((y1, y2), dim=1)
                oup = self.fc(inp)
                return oup

        self.actor = ConcatNet(self.cnn, self.actor1, self.state_dim)
        # self.critic = ConcatNet(self.cnn, self.critic1, self.state_dim)
        self.critic = self.critic1

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

class ActorCriticPointCloud(nn.Module):

    def __init__(self, obs_shape, states_shape, actions_shape, initial_std, model_cfg, asymmetric=False):
        super(ActorCriticPointCloud, self).__init__()

        self.asymmetric = asymmetric

        if model_cfg is None:
            actor_hidden_dim = [256, 256, 256]
            critic_hidden_dim = [256, 256, 256]
            activation = get_activation("selu")
        else:
            actor_hidden_dim = model_cfg['pi_hid_sizes']
            critic_hidden_dim = model_cfg['vf_hid_sizes']
            activation = get_activation(model_cfg['activation'])

        self.feature_dim = 128
        self.pc_dim = 3
        self.state_dim = 398

        self.pointnet_layer = getPointNet({
            'input_feature_dim': self.pc_dim,
            'feat_dim': self.feature_dim
        })

        actor_layers = []
        actor_layers.append(nn.Linear(self.feature_dim + self.state_dim, actor_hidden_dim[0]))
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
            critic_layers.append(nn.Linear(*states_shape, critic_hidden_dim[0]))
        else:
            critic_layers.append(nn.Linear(self.feature_dim + self.state_dim, critic_hidden_dim[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dim)):
            if l == len(critic_hidden_dim) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dim[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dim[l], critic_hidden_dim[l + 1]))
                critic_layers.append(activation)
        self.critic1 = nn.Sequential(*critic_layers)

        class ConcatNet(nn.Module):
            def __init__(self, point_net, fc, state_dim):
                super(ConcatNet, self).__init__()
                self.point_net = point_net
                self.fc = fc
                self.split_dim = state_dim
            
            def forward(self, input):
                x1 = input[:, self.split_dim:].view(-1, 768, 3)
                x2 = input[:, :self.split_dim]
                y1 = self.point_net(x1)
                inp = torch.cat((x2, y1), dim=1)
                y2 = self.fc(inp)
                return y2

        self.actor = ConcatNet(self.pointnet_layer, self.actor1, self.state_dim)
        self.critic = ConcatNet(self.pointnet_layer, self.critic1, self.state_dim)

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
