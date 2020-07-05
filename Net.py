from torch import nn
import torch
import torch.nn.functional as F

class DQNet(nn.Module):

    def __init__(self, input_ex, n_actions):
        super(DQNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        conv_res = self.conv3(self.conv2(self.conv1(input_ex)))
        input_shape = conv_res.view(conv_res.size(0), -1).shape[1]
        hidden_units = 200

        self.fc1 = nn.Linear(input_shape, hidden_units)
        self.fc2 = nn.Linear(hidden_units, n_actions)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.fc1(x.view(x.size(0), -1))
        return self.fc2(x)

if __name__ == '__main__':
    input = torch.ones((1, 3, 120, 80))
    n_actions = 7
    net = DQNet(input, n_actions)
    net(input)