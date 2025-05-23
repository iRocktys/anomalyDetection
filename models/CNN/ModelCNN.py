import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, input_channels, input_length, num_classes):
        super().__init__()
        # Bloco conv1
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=32,
                               kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm1d(32)
        # Bloco conv2
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64,
                               kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm1d(64)
        # Bloco conv3
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128,
                               kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm1d(128)
        # Bloco conv4
        self.conv4 = nn.Conv1d(in_channels=128, out_channels=256,
                               kernel_size=3, padding=1)
        self.bn4   = nn.BatchNorm1d(256)

        self.dropout = nn.Dropout(p=0.4)

        # Fully connected
        self.fc1 = nn.Linear(256 * input_length, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # x: [batch, input_channels, input_length]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))

        x = self.dropout(x)
        x = x.view(x.size(0), -1)           # flatten
        
        x = F.relu(self.fc1(x))
        return self.fc2(x)