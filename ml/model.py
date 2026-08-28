import torch
import torch.nn as nn
from torchvision import models


class IQAModel(nn.Module):
    """
    Image Quality Assessment model.

    Uses a pretrained ResNet-18 backbone with
    partial fine-tuning and a regression head.

    Output:
        Quality score between 0 and 1.
    """

    def __init__(self):
        super().__init__()

        # -------------------------------------------------
        # Pretrained ResNet-18
        # -------------------------------------------------

        self.backbone = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )

        # -------------------------------------------------
        # Freeze the early feature extraction layers
        # -------------------------------------------------

        for param in self.backbone.parameters():
            param.requires_grad = False

        # Fine-tune deeper layers
        for param in self.backbone.layer3.parameters():
            param.requires_grad = True

        for param in self.backbone.layer4.parameters():
            param.requires_grad = True

        # -------------------------------------------------
        # Number of ResNet features
        # -------------------------------------------------

        num_features = self.backbone.fc.in_features

        # -------------------------------------------------
        # IQA regression head
        # -------------------------------------------------

        self.backbone.fc = nn.Sequential(

            nn.Linear(
                num_features,
                256
            ),

            nn.ReLU(inplace=True),

            nn.Dropout(0.2),

            nn.Linear(
                256,
                64
            ),

            nn.ReLU(inplace=True),

            nn.Dropout(0.1),

            nn.Linear(
                64,
                1
            ),

            # Keep prediction inside [0, 1]
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.backbone(x)