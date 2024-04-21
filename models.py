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
    
class CNN_2d(nn.Module):
    def __init__(self):
        super(CNN_2d, self).__init__()
        # input : [1, 512, 512] 
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding='same') # output: [16, 512, 512]
        self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2) # output: [16, 256, 256]
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding='same') # output: [32, 256, 256]
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2) # output: [32, 128, 128]
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding='same') # output: [64, 128, 128]
        self.avgpool3 = nn.AvgPool2d(kernel_size=2, stride=2) # output: [64, 64, 64]
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding='same') # output: [128, 64, 64]
        self.avgpool4 = nn.AvgPool2d(kernel_size=2, stride=2) # output: [128, 32, 32]
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding='same') # output: [128, 32, 32]
        self.avgpool5 = nn.AvgPool2d(kernel_size=2, stride=2) # output: [128, 16, 16]
        self.linear1 = nn.Linear(128*16*16, 2048)
        self.linear2 = nn.Linear(2048, 2048)
        self.linear3 = nn.Linear(2048, 9)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.avgpool1(x)
        x = F.relu(self.conv2(x))
        x = self.avgpool2(x)
        x = F.relu(self.conv3(x))
        x = self.avgpool3(x)
        x = F.relu(self.conv4(x))
        x = self.avgpool4(x)
        x = F.relu(self.conv5(x))
        x = self.avgpool5(x)
        x = x.view(-1, 128*16*16)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return self.linear3(x)
