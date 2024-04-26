import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models.resnet import ResNet18_Weights

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=256, atrous_rates=[6, 12, 18], multi_grid=[1, 2, 4]):
        super(ASPP, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv3x3_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=atrous_rates[0]*multi_grid[0], dilation=atrous_rates[0]*multi_grid[0])
        self.conv3x3_2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=atrous_rates[1]*multi_grid[1], dilation=atrous_rates[1]*multi_grid[1])
        self.conv3x3_3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=atrous_rates[2]*multi_grid[2], dilation=atrous_rates[2]*multi_grid[2])
        self.conv3x3_4 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, dilation=1)  # No dilation (rate=1)

        self.conv_pointwise = nn.Conv2d(out_channels * 5, out_channels, kernel_size=1)

        self.dropout = nn.Dropout2d(p=0.1)

    def forward(self, x):
        out1x1 = F.relu(self.conv1x1(x))
        out3x3_1 = F.relu(self.conv3x3_1(x))
        out3x3_2 = F.relu(self.conv3x3_2(x))
        out3x3_3 = F.relu(self.conv3x3_3(x))
        out3x3_4 = F.relu(self.conv3x3_4(x))
        
        out_concat = torch.cat((out1x1, out3x3_1, out3x3_2, out3x3_3, out3x3_4), dim=1)
        out_concat = self.dropout(out_concat)

        out = self.conv_pointwise(out_concat)
        return out

class DeepLabV3(nn.Module):
    def __init__(self, num_classes=21, in_channels=512):
        super(DeepLabV3, self).__init__()
        self.num_classes = num_classes

        self.resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.resnet.fc = nn.Identity()

        self.aspp = ASPP(in_channels=in_channels, out_channels=256)

        self.predict = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.aspp(x)
        x = F.interpolate(x, scale_factor=32, mode='bilinear', align_corners=False)

        x = self.predict(x)

        return x