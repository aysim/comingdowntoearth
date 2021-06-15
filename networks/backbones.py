import torch.nn as nn
from torchvision import models

class ResNet34(nn.Module):
    def __init__(self):
        super().__init__()
        net = models.resnet34(pretrained=True)
        layers = list(net.children())[:3]
        layers_end = list(net.children())[4:-3]
        self.layers = nn.Sequential(*layers, *layers_end)

    def forward(self, x):
        return self.layers(x)

