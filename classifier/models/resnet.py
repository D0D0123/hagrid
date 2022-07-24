import torch

from typing import Dict
from torch import nn, Tensor
# from torchvision import models
from .custom_resnet import resnet18, resnet10, resnet20

# perceptron-posse

class ResNet(nn.Module):
    """
    Custom two headed ResNet configuration
    """

    def __init__(
            self,
            num_classes: int,
            restype: str = "ResNet18",
    ) -> None:
        """
        Custom two headed ResNet configuration

        Parameters
        ----------
        num_classes : int
            Number of classes for each task
        freezed : bool
            Freezing model parameters or not
        """

        super().__init__()

        pposse_model = None
        if restype == 'ResNet18':
            pposse_model = resnet18()
        elif restype == 'ResNet10':
            pposse_model = resnet10()
        elif restype == 'ResNet20':
            pposse_model = resnet20()

        self.backbone = None
        if pposse_model.num_layers == 4:
            self.backbone = nn.Sequential(
                pposse_model.conv1,
                pposse_model.bn1,
                pposse_model.relu,
                pposse_model.maxpool,
                pposse_model.layer1,
                pposse_model.layer2,
                pposse_model.layer3,
                pposse_model.layer4,
                pposse_model.avgpool
            )
        elif pposse_model.num_layers == 3:
            self.backbone = nn.Sequential(
                pposse_model.conv1,
                pposse_model.bn1,
                pposse_model.relu,
                pposse_model.maxpool,
                pposse_model.layer1,
                pposse_model.layer2,
                pposse_model.layer3,
                pposse_model.avgpool
            )

        num_features = pposse_model.fc.in_features

        self.classifier = nn.Sequential(
            nn.Linear(num_features, num_classes),
        )

        self.leading_hand = nn.Sequential(
            nn.Linear(num_features, 2),
        )

    def forward(self, img: Tensor) -> Dict:
        x = self.backbone(img)
        x = torch.flatten(x, 1)
        gesture = self.classifier(x)

        leading_hand = self.leading_hand(x)

        return {'gesture': gesture, 'leading_hand': leading_hand}
