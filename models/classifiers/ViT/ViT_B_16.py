import torch
import torch.nn as nn
import torchvision.models as models


class DeepFakeViT(nn.Module):
    def __init__(self, freeze_features=True):
        super(DeepFakeViT, self).__init__()

        # Load the pre-trained ViT model
        weights = models.ViT_B_16_Weights.DEFAULT
        self.vit = models.vit_b_16(weights=weights)

        # Freeze the transformer encoder layers if requested
        if freeze_features:
            for param in self.vit.parameters():
                param.requires_grad = False

        # Replace the final classification head
        num_features = self.vit.heads.head.in_features

        self.vit.heads.head = nn.Linear(num_features, 1)

    def forward(self, x):
        return self.vit(x)