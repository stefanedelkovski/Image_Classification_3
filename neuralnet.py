import torch
import torch.nn as nn
import torch.nn.functional as F

hp = {'batch_size': 16, 'epochs': 5, 'learning_rate': 0.0001}
classes = ['bad', 'average', 'good']


class Simple_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, (3, 3), (1, 1), padding=5 // 2)
        self.conv2 = nn.Conv2d(64, 32, (3, 3), (1, 1), padding=5 // 2)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 65 * 65, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x))
        return x

