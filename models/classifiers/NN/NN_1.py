import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepFakeNN_1(nn.Module):
    def __init__(self):
        super(DeepFakeNN_1, self).__init__()
        self.fc1 = nn.Linear(224*224*3,512)
        self.fc2 = nn.Linear(512,256)
        self.fc3 = nn.Linear(128,64)
        self.fc4 = nn.Linear(64,1)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        x = torch.sigmoid(self.fc4(x))

        return x