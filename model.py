import torch.nn as nn
from torchvision import models

def get_model(num_classes=3, pretrained=True):
    model = models.mobilenet_v2(pretrained=pretrained)
    model.classifier[1] = nn.Linear(
        model.last_channel, num_classes
    )
    return model
