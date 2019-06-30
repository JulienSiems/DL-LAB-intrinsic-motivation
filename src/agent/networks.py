import torch.nn as nn
import torch.nn.functional as F

from torchvision.models.resnet import ResNet, BasicBlock
import torch

import math


"""
CartPole network
"""


class MLP(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=400, *args, **kwargs):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x, *args, **kwargs):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# https://zablo.net/blog/post/using-resnet-for-mnist-in-pytorch-tutorial/
class ResnetVariant(ResNet):
    def __init__(self, history_length, num_actions, *args, **kwargs):
        super(ResnetVariant, self).__init__(BasicBlock, [2, 2, 2, 2], num_classes=num_actions)
        self.conv1 = nn.Conv2d(history_length, 64,
                               kernel_size=(7, 7),
                               stride=(2, 2),
                               padding=(3, 3), bias=False)

    def forward(self, x, *args, **kwargs):
        return torch.softmax(
            super(ResnetVariant, self).forward(x), dim=-1)


class LeNetVariant(nn.Module):
    """
    Adapted from LENET https://github.com/kuangliu/pytorch-cifar/blob/master/models/lenet.py
    """

    def __init__(self, history_length, num_actions, *args, **kwargs):
        super(LeNetVariant, self).__init__()
        self.conv1 = nn.Conv2d(history_length, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(7056, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_actions)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x, *args, **kwargs):
        out = F.elu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.elu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.elu(self.fc1(out))
        out = F.elu(self.fc2(out))
        out = F.softmax(self.fc3(out), dim=1)
        return out


# https://github.com/diegoalejogm/deep-q-learning/blob/master/utils/net.py
class DeepQNetwork(nn.Module):

    def __init__(self, in_dim, num_actions, history_length, duelling=False, iqn=False, activation=nn.ReLU,
                 hidden_dim=512, embedding_dim=64, *args, **kwargs):
        super(DeepQNetwork, self).__init__()
        self.duelling = duelling
        self.iqn = iqn
        self.activation = activation
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        self.conv1 = nn.Sequential(
            nn.Conv2d(history_length, 32, kernel_size=8, stride=4, padding=0),
            activation()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            activation()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            activation()
        )

        dummy_input = torch.zeros(1, in_dim[0], in_dim[1], in_dim[2])
        out_cnn = self.conv3(self.conv2(self.conv1(dummy_input)))
        self.cnn_out_dim = out_cnn.view(out_cnn.size(0), -1).shape[1]

        if iqn:
            # precompute pi * i for 0 <= i < n from eq. 4 of IQN paper
            self.n_range_pi = torch.arange(start=0, end=self.embedding_dim, dtype=torch.float32).view(1, -1) * math.pi
            self.iqn_phi = nn.Sequential(
                nn.Linear(self.embedding_dim, self.cnn_out_dim, bias=True),
                activation()
            )

        if duelling:
            self.fc1_val = nn.Sequential(
                nn.Linear(self.cnn_out_dim, self.hidden_dim, bias=True),
                activation()
            )
            self.fc2_val = nn.Sequential(
                nn.Linear(self.hidden_dim, 1, bias=True)
            )
            self.fc1_adv = nn.Sequential(
                nn.Linear(self.cnn_out_dim, self.hidden_dim, bias=True),
                activation()
            )
            self.fc2_adv = nn.Sequential(
                nn.Linear(self.hidden_dim, num_actions, bias=True)
            )
        else:
            self.fc1 = nn.Sequential(
                nn.Linear(self.cnn_out_dim, self.hidden_dim, bias=True),
                activation()
            )
            self.fc2 = nn.Sequential(
                nn.Linear(self.hidden_dim, num_actions, bias=True)
            )

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x, taus=None, *args, **kwargs):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)

        if self.iqn:
            # taus need to be of shape (batch_size, num_taus)
            batch_size, num_taus = taus.shape
            # flatten taus to shape (batch_size * num_taus, 1)
            taus = taus.view(-1, 1)
            taus_embedded = torch.cos(taus * self.n_range_pi)
            iqn_phi_out = self.iqn_phi(taus_embedded)
            x_repeated = x.repeat(1, num_taus).view(-1, self.cnn_out_dim)
            x = x_repeated * iqn_phi_out

        if self.duelling:
            x_val = self.fc1_val(x)
            x_val = self.fc2_val(x_val)
            x_adv = self.fc1_adv(x)
            x_adv = self.fc2_adv(x_adv)
            return x_val + x_adv - x_adv.mean(dim=1, keepdim=True)
        else:
            x = self.fc1(x)
            return self.fc2(x)


class Encoder(nn.Module):

    def __init__(self, history_length, filters=32):
        super(Encoder, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(history_length, filters, kernel_size=3, stride=2, padding=1),
            nn.ELU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=3, stride=2, padding=1),
            nn.ELU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=3, stride=2, padding=1),
            nn.ELU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=3, stride=2, padding=1),
            nn.ELU()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        return x


class InverseModel(nn.Module):
    """
    The inverse dynamics model (eq. 2) predicts the action taken between state s_t and s_t+1
    """

    def __init__(self, num_actions=4, input_dimension=288 * 2):
        super(InverseModel, self).__init__()
        self.hidden = nn.Sequential(
            nn.Linear(input_dimension, 256, bias=True),
            nn.ELU()
        )
        self.output = nn.Linear(256, num_actions, bias=True)

    def forward(self, phi_s_t, phi_s_tp1):
        feature_vector = torch.cat([phi_s_t, phi_s_tp1], dim=1)
        x = self.hidden(feature_vector)
        a_t_pred = self.output(x)
        return a_t_pred


class ForwardModel(nn.Module):
    """
    The forward dynamics model (eq. 4) predicts the embedding of the next state given the current state and the action taken.
    """

    def __init__(self, num_actions, dim_s=288, output_dimension=288):
        super(ForwardModel, self).__init__()
        self.hidden = nn.Sequential(
            nn.Linear(num_actions + dim_s, 256, bias=True),
            nn.ELU()
        )
        self.output = nn.Linear(256, output_dimension, bias=True)

    def forward(self, phi_s_t, a_t):
        feature_vector = torch.cat([phi_s_t, a_t], dim=1)
        x = self.hidden(feature_vector)
        phi_s_tp1_pred = self.output(x)
        return phi_s_tp1_pred
