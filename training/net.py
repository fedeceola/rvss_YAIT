import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class Net(nn.Module):
    def __init__(self, feat_vect_dim, use_avg_pool):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4)) if use_avg_pool else nn.Identity()
        self.fc1 = nn.Linear(256 if use_avg_pool else feat_vect_dim, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.avgpool(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def load_model(model_name, feat_vect_dim=None, use_avg_pool=None):
    if model_name == "Net":
        return Net(feat_vect_dim, use_avg_pool) 

    elif model_name == "mobilenet_v2":
        model = models.mobilenet_v2(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False

        feat_vect_dim = model.classifier[-1].in_features
        model.classifier = nn.Identity()
        
        model = nn.Sequential(
            model,
            nn.AdaptiveAvgPool2d((4, 4)) if use_avg_pool else nn.Identity(),
            nn.Linear(256 if use_avg_pool else feat_vect_dim, 1)
        )

        return model

    raise NotImplementedError