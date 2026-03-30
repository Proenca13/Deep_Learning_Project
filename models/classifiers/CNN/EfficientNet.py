import torch.nn as nn
from torchvision import models

class DeepFakeEfficientNet(nn.Module):
    def __init__(self, freeze_features=True):
        super(DeepFakeEfficientNet, self).__init__()

        self.base_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

        if freeze_features:
            for param in self.base_model.features.parameters():
                param.requires_grad = False

        in_features = self.base_model.classifier[1].in_features
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 1)
        )

    def forward(self, x):
        return self.base_model(x)