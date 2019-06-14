import torch.nn as nn
import torch.nn.functional as F

from torchvision.models.resnet import ResNet, BasicBlock
import torch

"""
CartPole network
"""


class MLP(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=400):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# https://zablo.net/blog/post/using-resnet-for-mnist-in-pytorch-tutorial/
class ResnetVariant(ResNet):
    def __init__(self, history_length, num_actions):
        super(ResnetVariant, self).__init__(BasicBlock, [2, 2, 2, 2], num_classes=num_actions)
        self.conv1 = nn.Conv2d(history_length, 64,
                               kernel_size=(7, 7),
                               stride=(2, 2),
                               padding=(3, 3), bias=False)

    def forward(self, x):
        return torch.softmax(
            super(ResnetVariant, self).forward(x), dim=-1)


class LeNetVariant(nn.Module):
    """
    Adapted from LENET https://github.com/kuangliu/pytorch-cifar/blob/master/models/lenet.py
    """

    def __init__(self, history_length, num_actions):
        super(LeNetVariant, self).__init__()
        self.conv1 = nn.Conv2d(history_length, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(7056, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_actions)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
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

    def __init__(self, num_actions, history_length):
        super(DeepQNetwork, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(history_length, 32, kernel_size=8, stride=4),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.hidden = nn.Sequential(
            nn.Linear(4096, 512, bias=True),
            nn.ReLU()
        )
        self.out = nn.Sequential(
            nn.Linear(512, num_actions, bias=True)
        )
        # Init with cuda if available
        if torch.cuda.is_available():
            self.cuda()
        self.apply(self.weights_init)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.hidden(x)
        x = self.out(x)
        return x

    @staticmethod
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # pass
            m.weight.data.normal_(0.0, 0.02)
            # nn.init.xavier_uniform(m.weight)
        if classname.find('Linear') != -1:
            pass
            # m.weight.data.normal_(0.0, 0.02)
            # m.weight.data.fill_(1)
            # nn.init.xavier_uniform(m.weight)
            # m.weight.data.normal_(0.0, 0.008)
