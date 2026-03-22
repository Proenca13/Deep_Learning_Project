import torch.nn as nn
import timm


class DeepFakeXception(nn.Module):
    def __init__(self, freeze_features=True):
        super(DeepFakeXception, self).__init__()

        self.xception = timm.create_model('legacy_xception', pretrained=True)

        if freeze_features:
            for param in self.xception.parameters():
                param.requires_grad = False

        num_ftrs = self.xception.fc.in_features
        self.xception.fc = nn.Linear(num_ftrs, 1)

    def forward(self, x):
        return self.xception(x)