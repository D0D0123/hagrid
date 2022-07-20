###################################################################################################
# perceptron-posse


from typing import Type, Any, Callable, List, Optional

import torch
import torch.nn as nn
from torch import Tensor

class CustomBlock(nn.Module):
    """
    Residual block for ResNet
    - Consists of two convolutional layers with 2-d batch normalization between followed by a dropout layer and 
    - ReLU activation function
    """
    def __init__(self, 
                in_channels, 
                out_channels, 
                stride = 1,
                identity_downsample: Optional[nn.Module] = None,
                ):
        super(CustomBlock, self).__init__()
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(0.6)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample

    def forward(self, x):
        # saving weights at beginning to reapply at end of block 
        identity = x

        # main block architecture - 2 conv layers with batch norm and relu
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        # dropout layer to help with overfitting - not necessary since our dataset is fairly small
        # x = self.dropout(x)

        # if there is a size mismatch, downsample identity before readding
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        # reapply weights from beginning of block
        x += identity
        x = self.relu(x)
        return x

#####################################

class CustomResNet(nn.Module):
    def __init__(
        self,
        block: CustomBlock,
        layers: List[int],
        num_classes: int = 1000,
    ) -> None:
        super().__init__()

        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._custom_make_layers(block, 64, layers[0], stride=1)
        self.layer2 = self._custom_make_layers(block, 128, layers[1], stride=2)
        self.layer3 = self._custom_make_layers(block, 256, layers[2], stride=2)
        self.layer4 = self._custom_make_layers(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _custom_make_layers(
                    self, 
                    block,
                    intermediate_channels,
                    num_residual_blocks,
                    stride):
        """Function to make resnet layers based on given parameters"""
        layers = []

        # used if size of identity is incompatible with output of block
        identity_downsample = nn.Sequential(
            nn.Conv2d(self.in_channels, intermediate_channels, kernel_size=1, stride=stride),
            nn.BatchNorm2d(intermediate_channels)
        )

        layers.append(
            block(
                self.in_channels, 
                intermediate_channels, 
                stride, 
                identity_downsample
            )
        )
        
        self.in_channels = intermediate_channels # 256
        for i in range(num_residual_blocks - 1):
            layers.append(
                block(
                    self.in_channels, 
                    intermediate_channels
                )
            ) # 256 -> 64, 64*4 (256) again
        
        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet(
    block: CustomBlock,
    layers: List[int],
    **kwargs: Any,
) -> CustomResNet:
    model = CustomResNet(block, layers, **kwargs)
    return model

### duplicate and customise to make custom resnet classes - perceptron-posse ###

def resnet18(progress: bool = True, **kwargs: Any) -> CustomResNet:
    r"""CustomResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(CustomBlock, [1, 1, 1, 1], **kwargs)