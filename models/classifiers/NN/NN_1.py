import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepFakeNN_1(nn.Module):
    def __init__(self,input_size):
        super(DeepFakeNN_1, self).__init__()
        self.fc1 = nn.Linear(input_size[0] *input_size[1] * 3, 512)
        self.dropout1 = nn.Dropout(p=0.5)

        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(p=0.5)

        self.fc3 = nn.Linear(256, 64)
        self.dropout3 = nn.Dropout(p=0.2)

        self.fc4 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)

        x = F.relu(self.fc1(x))
        x = self.dropout1(x)

        x = F.relu(self.fc2(x))
        x = self.dropout2(x)

        x = F.relu(self.fc3(x))
        x = self.dropout3(x)

        x = self.fc4(x)
        return x