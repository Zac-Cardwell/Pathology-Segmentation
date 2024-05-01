import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models.resnet import ResNet18_Weights

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=256, atrous_rates=[6, 12, 18, 24, 30]):
        super(ASPP, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv3x3_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=atrous_rates[0], dilation=atrous_rates[0])
        self.conv3x3_2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=atrous_rates[1], dilation=atrous_rates[1])
        self.conv3x3_3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=atrous_rates[2], dilation=atrous_rates[2])
        self.conv3x3_5 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=atrous_rates[3], dilation=atrous_rates[3])
        self.conv3x3_6 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=atrous_rates[4], dilation=atrous_rates[4])
        self.conv3x3_4 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, dilation=1)  # No dilation (rate=1)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.bn4 = nn.BatchNorm2d(out_channels)
        self.bn5 = nn.BatchNorm2d(out_channels)
        self.bn6 = nn.BatchNorm2d(out_channels)

        self.conv_pointwise = nn.Conv2d(out_channels * 7, out_channels, kernel_size=1)
        self.bn_pointwise = nn.BatchNorm2d(out_channels)

        self.dropout = nn.Dropout2d(p=0.1)

    def forward(self, x):
        out1x1 = F.relu(self.bn1(self.conv1x1(x)))
        out3x3_1 = F.relu(self.bn2(self.conv3x3_1(x)))
        out3x3_2 = F.relu(self.bn3(self.conv3x3_2(x)))
        out3x3_3 = F.relu(self.bn4(self.conv3x3_3(x)))
        out3x3_5 =  F.relu(self.bn5(self.conv3x3_5(x)))
        out3x3_6 =  F.relu(self.bn6(self.conv3x3_6(x)))
        out3x3_4 = F.relu(self.conv3x3_4(x))
        
        out_concat = torch.cat((out1x1, out3x3_1, out3x3_2, out3x3_3, out3x3_4, out3x3_5, out3x3_6), dim=1)
        out_concat = self.dropout(out_concat)

        out = F.relu(self.bn_pointwise(self.conv_pointwise(out_concat)))
        return out

class DeepLabV3(nn.Module):
    def __init__(self, num_classes=21, in_channels=512):
        super(DeepLabV3, self).__init__()
        self.num_classes = num_classes

        self.resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.resnet.fc = nn.Identity()
        self.backbone = nn.Sequential(*list(self.resnet.children())[:-2])  # Remove last two layers 
        

        self.aspp = ASPP(in_channels=in_channels, out_channels=256)

        self.predict = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
       
        x = self.backbone(x)
        x = self.aspp(x)
        x = F.interpolate(x, scale_factor=32, mode='bilinear', align_corners=False)

        x = self.predict(x)

        return x


'''
model = DeepLabV3( num_classes=2) 
input_tensor = torch.randn(1, 3, 256, 256)  # Example input tensor
output_tensor = model(input_tensor)
print(output_tensor.shape)'''