import torch.nn as nn
from torchvision import models


class DeepFakeResNet34(nn.Module):
    def __init__(self, freeze_features=True):
        super(DeepFakeResNet34, self).__init__()

        self.resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)

        if freeze_features:
            for param in self.resnet.parameters():
                param.requires_grad = False

        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, 1)

    def forward(self, x):
        return self.resnet(x)