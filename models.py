import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CNN_1d(nn.Module):
    def __init__(self, input_size, num_classes):
        super(CNN_1d, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=11, stride=1, padding='same')
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=11, stride=1, padding='same')
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=11, stride=1, padding='same')
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=11, stride=1, padding='same')
        self.conv5 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=11, stride=1, padding='same')
        self.conv6 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=11, stride=1, padding='same')
        self.conv7 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=11, stride=1, padding='same')
        self.conv8 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=11, stride=1, padding='same')
        
        self.pool1 = nn.AvgPool1d(kernel_size=2, stride=2)
        self.pool2 = nn.AvgPool1d(kernel_size=4, stride=4)
        self.fc1 = nn.Linear(8192, 2048)
        self.fc2 = nn.Linear(2048, 2048)
        self.fc3 = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = F.relu(self.conv3(x))
        x = self.pool1(x)
        x = F.relu(self.conv4(x))
        x = self.pool1(x)
        x = F.relu(self.conv5(x))
        x = self.pool1(x)
        x = F.relu(self.conv6(x))
        x = self.pool2(x)
        x = F.relu(self.conv7(x))
        x = self.pool2(x)
        x = F.relu(self.conv8(x))
        x = self.pool2(x)
        x = x.view(-1, 8192)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    