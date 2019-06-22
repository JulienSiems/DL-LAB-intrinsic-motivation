import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
from torchvision.models.resnet import ResNet, BasicBlock, model_urls

"""
CartPole network
"""


class MLP(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=400):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_adv = nn.Linear(hidden_dim, action_dim)
        self.fc_val = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        xf = F.relu(self.fc2(x))
        # val = self.fc_val(xf)
        adv = self.fc_adv(xf)
        # x = val + adv - adv.mean()
        x = adv
        return x

    def init_zero(self):
        for param in self.parameters():
            param.data.copy_(0.0 * param.data)


class CNN(nn.Module):
    def __init__(self, history_length=1, n_classes=5):
        super(CNN, self).__init__()
        # define layers of a convolutional neural network
        self.relu = nn.ReLU(inplace=True)
        self.conv_net = nn.Sequential(
            nn.Conv2d(history_length, 16, kernel_size=8, stride=4, padding=0, bias=False),
            self.relu,
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=0, bias=False),
            self.relu,
        )
        self.fc_net = nn.Sequential(
            nn.Linear(3200, 256),
            self.relu,
        )
        self.fc_adv_net = nn.Sequential(
            nn.Linear(256, n_classes)
        )
        self.fc_val_net = nn.Sequential(
            nn.Linear(256, 1),
        )

    def forward(self, x):
        # compute forward pass
        x = self.conv_net(x)
        x = x.view(x.size(0), -1)
        xf = self.fc_net(x)
        val = self.fc_val_net(xf)
        adv = self.fc_adv_net(xf)
        x = val + adv - adv.mean()

        return x

    def init_zero(self):
        for param in self.parameters():
            param.data.copy_(0.0 * param.data)


ICM_GET_ONLY_Q_OUT = 1
ICM_GET_ONLY_FORWARD_OUT = 2
ICM_GET_ALL_OUT = 3
#
# class QModel(nn.Module):
#     def __init__(self, in_units=256, out_units=4):
#         super(QModel, self).__init__()
#         fc_out = 64
#         self.elu = nn.ELU(inplace=True)
#         self.fc_net = nn.Sequential(
#             nn.Linear(in_units, fc_out),
#             self.elu,
#         )
#         self.fc_adv_net = nn.Sequential(
#             nn.Linear(fc_out, out_units)
#         )
#         self.fc_val_net = nn.Sequential(
#             nn.Linear(fc_out, 1),
#         )
#
#     def forward(self, x):
#         xf = self.fc_net(x)
#         val = self.fc_val_net(xf)
#         adv = self.fc_adv_net(xf)
#         x = val + adv - adv.mean()
#         return x
#
#
# class ForwardModel(nn.Module):
#     def __init__(self, in_units=288+4, out_units=288):
#         super(ForwardModel, self).__init__()
#         self.elu = nn.ELU(inplace=True)
#         self.net = nn.Sequential(
#             nn.Linear(in_units, 256),
#             self.elu,
#             nn.Linear(256, out_units),
#         )
#
#     def forward(self, x):
#         x = self.net(x)
#         return x
#
#
# class InverseModel(nn.Module):
#     def __init__(self, in_units=288*2, out_units=4):
#         super(InverseModel, self).__init__()
#         self.elu = nn.ELU(inplace=True)
#         self.net = nn.Sequential(
#             nn.Linear(in_units, 256),
#             self.elu,
#             nn.Linear(256, out_units),
#         )
#
#     def forward(self, x):
#         x = self.net(x)
#         return x
#
#
# class ICMModel(nn.Module):
#     def __init__(self, in_shape=(1, 42, 42), n_classes=4):
#         super(ICMModel, self).__init__()
#         self.elu = nn.ELU(inplace=True)
#         self.cnn = nn.Sequential(
#             nn.Conv2d(in_shape[0], 32, kernel_size=3, stride=2, padding=1, bias=False),
#             self.elu,
#             nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, bias=False),
#             self.elu,
#             nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, bias=False),
#             self.elu,
#             nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, bias=False),
#             self.elu,
#         )
#
#         self.q_cnn = nn.Sequential(
#             nn.Conv2d(in_shape[0], 32, kernel_size=3, stride=2, padding=1, bias=False),
#             self.elu,
#             nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, bias=False),
#             self.elu,
#             nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, bias=False),
#             self.elu,
#             nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, bias=False),
#             self.elu,
#         )
#
#         dummy_input = torch.zeros(1, in_shape[0], in_shape[1], in_shape[2])
#         out_cnn = self.cnn(dummy_input)
#         out_cnn = out_cnn.view(out_cnn.size(0), -1)
#         cnn_out_size = out_cnn.shape[1]
#
#         self.inverse_net = InverseModel(in_units=cnn_out_size*2, out_units=n_classes)
#         self.forward_net = ForwardModel(in_units=cnn_out_size+n_classes, out_units=cnn_out_size)
#         self.q_net = QModel(in_units=cnn_out_size, out_units=n_classes)
#
#     def forward(self, state, next_state, action, mode=ICM_GET_ALL_OUT):
#         if mode == ICM_GET_ONLY_Q_OUT:
#             xt = self.q_cnn(state)
#             xt = xt.view(xt.size(0), -1)
#             q_out = self.q_net(xt)
#             inverse_out = 0
#             forward_out = 0
#         elif mode == ICM_GET_ONLY_FORWARD_OUT:
#             xt = self.cnn(state)
#             xt = xt.view(xt.size(0), -1)
#             forward_in = torch.cat([xt, action], dim=1)
#             forward_out = self.forward_net(forward_in)
#             inverse_out = 0
#             q_out = 0
#         else:
#             xt = self.cnn(state)
#             xt = xt.view(xt.size(0), -1)
#             xt1 = self.cnn(next_state)
#             xt1 = xt1.view(xt1.size(0), -1)
#             inverse_in = torch.cat([xt, xt1], dim=1)
#             forward_in = torch.cat([xt, action], dim=1)
#             inverse_out = self.inverse_net(inverse_in)
#             forward_out = self.forward_net(forward_in)
#
#             xt = self.q_cnn(state)
#             xt = xt.view(xt.size(0), -1)
#             q_out = self.q_net(xt)
#         return inverse_out, forward_out, q_out
#
#     def encode(self, x):
#         x = self.cnn(x)
#         x = x.view(x.size(0), -1)
#         return x
#
#     def init_zero(self):
#         for param in self.parameters():
#             param.data.copy_(0.0 * param.data)

class DeepQNetwork(nn.Module):

    def __init__(self, in_dim, num_actions, history_length):
        super(DeepQNetwork, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(history_length, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU()
        )

        dummy_input = torch.zeros(1, in_dim[0], in_dim[1], in_dim[2])
        x = self.conv1(dummy_input)
        x = self.conv2(x)
        x = self.conv3(x)
        out_cnn = self.conv4(x)
        out_cnn = out_cnn.view(out_cnn.size(0), -1)
        cnn_out_size = out_cnn.shape[1]

        # self.hidden = nn.Sequential(
        #     nn.Linear(cnn_out_size, 512, bias=True),
        #     nn.ELU()
        # )
        # self.out = nn.Sequential(
        #     nn.Linear(512, num_actions, bias=True)
        # )

        fc_out = 128
        self.elu = nn.ELU(inplace=True)
        self.fc_net = nn.Sequential(
            nn.Linear(cnn_out_size, fc_out),
            self.elu,
        )
        self.fc_adv_net = nn.Sequential(
            nn.Linear(fc_out, num_actions)
        )
        self.fc_val_net = nn.Sequential(
            nn.Linear(fc_out, 1),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        # x = self.hidden(x)
        # x = self.out(x)

        xf = self.fc_net(x)
        val = self.fc_val_net(xf)
        adv = self.fc_adv_net(xf)
        x = val + adv - adv.mean()

        return x


class Encoder(nn.Module):

    def __init__(self, history_length):
        super(Encoder, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(history_length, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
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

