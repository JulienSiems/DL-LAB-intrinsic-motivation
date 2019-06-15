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



