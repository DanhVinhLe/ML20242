import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import time
import copy
import matplotlib.pyplot as plt
import numpy as np
from typing import Type, Any, Callable, Union, List, Optional
from collections import namedtuple
import einops
from einops.layers.torch import Rearrange

class LinearSVM(nn.Module):
    def __init__(self, input_size = 28, num_classes = 10):
        super().__init__()
        self.linear = nn.Linear(input_size*input_size, num_classes)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        scores = self.linear(x) 
        return scores
#Lenet Model
class LeNet(nn.Module):
    def __init__(self, num_classes = 10, in_channels = 1):
        super(LeNet, self).__init__()
        # Các lớp tích chập
        self.in_channels = in_channels
        self.num_classes = num_classes 
        self.conv1 = nn.Conv2d(self.in_channels, 6, kernel_size=5,padding=2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5)
        
        # Các lớp fully connected
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, self.num_classes)
    def forward(self, x):
        # Các bước truyền dữ liệu qua các lớp
        x = self.conv1(x)
        x = F.relu(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
# AlexNet Model
class AlexNet(nn.Module):
    # input_size = 227
    def __init__(self, num_classes=1000, in_channels = 3):
        super(AlexNet, self).__init__()
        # Các lớp tích chập
        self.in_channels = in_channels
        self.num_classes = num_classes 
        self.features = nn.Sequential(
            # Conv1
            nn.Conv2d(self.in_channels, 96, kernel_size=11, stride=4, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Conv2
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Conv3
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            
            # Conv4
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            
            # Conv5
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        # Các lớp fully connected
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, self.num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
class VGG16(nn.Module):
    def __init__(self, num_classes=1000, in_channels = 3, dropout_rate=0.5, input_size = 227):
        super(VGG16, self).__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.layer1 = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.layer5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.input_size = input_size
        self.feature_size = self.input_size // 32
        self.avg_pool = nn.AdaptiveAvgPool2d((7, 7))
        
        self.fc1 = nn.Linear(512 * 7 * 7, 4096)
        self.dropout1 = nn.Dropout(p = dropout_rate)
        self.fc2 = nn.Linear(4096, 4096)
        self.dropout2 = nn.Dropout(p = dropout_rate)
        self.fc3 = nn.Linear(4096, self.num_classes)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

class VGG16BatchNorm(nn.Module):
    def __init__(self, num_classes=1000, in_channels = 3, dropout_rate=0.5, input_size = 227):
        super(VGG16BatchNorm, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.input_size = input_size
        self.layer1 = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.layer5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.feature_size = self.input_size // 32
        self.avg_pool = nn.AdaptiveAvgPool2d((7, 7))
        self.fc1 = nn.Linear(512 * self.feature_size * self.feature_size, 4096)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(4096, 4096)
        self.dropout2 = nn.Dropout(p=dropout_rate)
        self.fc3 = nn.Linear(4096, self.num_classes)
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

# ResNet Model
class BasicBlock(nn.Module):
    # Basic block for ResNet 18, ResNet 34
    expansion = 1
    
    def __init__(self, in_channels: int, out_channels: int, stride: int =1, 
                 downsample: Optional[nn.Module] = None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        
        self.stride = stride
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out

class BottleNeck(nn.Module):
    # Bottleneck block for ResNet 50, ResNet 101, ResNet 152
    expansion = 4
    
    def __init__(self, in_channels: int, out_channels: int, stride: int =1, 
                 downsample: Optional[nn.Module] = None):
        super(BottleNeck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        
        self.stride = stride
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out
    
class ResNet(nn.Module):
    def __init__(self, block: Type[Union[BasicBlock, BottleNeck]], layers: List[int],
                 num_classes: int = 1000, in_channels: int = 3, zero_init_residual: bool = False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        
        self.conv1 = nn.Conv2d(in_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet layers
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BottleNeck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
        
    def _make_layer(self, block: Type[Union[BasicBlock, BottleNeck]], out_channels: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        
        if stride != 1 or self.inplanes != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, out_channels * block.expansion, kernel_size= 1,
                          stride = stride, bias = False)
            )
        
        layers = []
        layers.append(block(self.inplanes, out_channels, stride, downsample))
        self.inplanes = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

def resnet18(num_classes: int = 1000, in_channels: int = 3) -> ResNet:
    """Constructs a ResNet-18 model.
    Args:
        num_classes (int): Number of classes for the output layer.
        in_channels (int): Number of input channels.
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, in_channels=in_channels)

def resnet34(num_classes: int = 1000, in_channels: int = 3) -> ResNet:
    """Constructs a ResNet-34 model.
    Args:
        num_classes (int): Number of classes for the output layer.
        in_channels (int): Number of input channels.
    """
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, in_channels=in_channels)

def resnet50(num_classes: int = 1000, in_channels: int = 3) -> ResNet:
    """Constructs a ResNet-50 model.
    Args:
        num_classes (int): Number of classes for the output layer.
        in_channels (int): Number of input channels.
    """
    return ResNet(BottleNeck, [3, 4, 6, 3], num_classes=num_classes, in_channels=in_channels)

def resnet101(num_classes: int = 1000, in_channels: int = 3) -> ResNet:
    """Constructs a ResNet-101 model.
    Args:
        num_classes (int): Number of classes for the output layer.
        in_channels (int): Number of input channels.
    """
    return ResNet(BottleNeck, [3, 4, 23, 3], num_classes=num_classes, in_channels=in_channels)

# Inception V1 model

class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out1x1, out3x3_reduce, out3x3, out5x5_reduce, out5x5, pool_proj):
        super(InceptionBlock, self).__init__()
        
        # 1x1 Convolution
        self.branch1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out1x1, kernel_size=1),
            nn.BatchNorm2d(out1x1),
            nn.ReLU(inplace=True)
        )
        
        # 1x1 Convolution followed by 3x3 Convolution
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels, out3x3_reduce, kernel_size=1),
            nn.BatchNorm2d(out3x3_reduce),
            nn.ReLU(inplace=True),
            nn.Conv2d(out3x3_reduce, out3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(out3x3),
            nn.ReLU(inplace=True)
        )
        
        # 1x1 Convolution followed by 5x5 Convolution
        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channels, out5x5_reduce, kernel_size=1),
            nn.BatchNorm2d(out5x5_reduce),
            nn.ReLU(inplace=True),
            nn.Conv2d(out5x5_reduce, out5x5, kernel_size=5, padding=2),
            nn.BatchNorm2d(out5x5),
            nn.ReLU(inplace=True)
        )
        
        # 3x3 Max Pooling followed by 1x1 Convolution
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1),
            nn.BatchNorm2d(pool_proj),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        branch1 = self.branch1x1(x)
        branch2 = self.branch3x3(x)
        branch3 = self.branch5x5(x)
        branch4 = self.branch_pool(x)
        
        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)
    
class InceptionV1(nn.Module):
    def __init__(self, num_classes = 1000, in_channels = 3):
        super(InceptionV1, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
        )
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inception3a = InceptionBlock(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionBlock(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inception4a = InceptionBlock(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = InceptionBlock(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = InceptionBlock(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = InceptionBlock(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = InceptionBlock(528, 256, 160, 320, 32, 128, 128)
        
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inception5a = InceptionBlock(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = InceptionBlock(832, 384, 192, 384, 48, 128, 128)
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.dropout = nn.Dropout(p=0.4)
        self.fc = nn.Linear(1024, self.num_classes)
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x
    

# Inception V3 model
class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels, eps = 0.001)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class InceptionA(nn.Module):
    def __init__(self, in_channels, pool_features):
        super(InceptionA, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3_1 = BasicConv2d(in_channels, 48, kernel_size=1)
        self.branch3x3_2 = BasicConv2d(48, 64, kernel_size=3, padding = 1)
        
        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding = 1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, padding = 1)
        
        self.branch_pool = BasicConv2d(in_channels, pool_features, kernel_size=1)
        self.pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        branch1 = self.branch1x1(x)
        
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)
        
        branch_pool = self.pool(x)
        branch_pool = self.branch_pool(branch_pool)
        
        outputs = [branch3x3dbl, branch3x3, branch_pool, branch1]
        return torch.cat(outputs, 1)
    
class InceptionB(nn.Module):
    # Incetion B block reduce dimensionality
    def __init__(self, in_channels):
        super(InceptionB, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 384, kernel_size=3, stride = 2)
        self.branch3x3_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3_2 = BasicConv2d(64, 96, kernel_size= 3, padding = 1)
        self.branch3x3_3 = BasicConv2d(96, 96, kernel_size=3, stride= 2)
        self.branch_pool = nn.MaxPool2d(kernel_size=3, stride=2)
        
    def forward(self, x):
        branch1 = self.branch1x1(x)
        
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch3x3 = self.branch3x3_3(branch3x3)
        
        branch_pool = self.branch_pool(x)
        
        outputs = [branch3x3, branch1, branch_pool]
        return torch.cat(outputs, 1)
    
class InceptionC(nn.Module):
    def __init__(self, in_channels, channel_7):
        super(InceptionC, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 192, kernel_size=1)
        c7 = channel_7
        
        self.branch7x7_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7_2 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7_3 = BasicConv2d(c7, 192, kernel_size=(7, 1), padding=(3, 0))
        
        self.branch7x7db_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7db_2 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7db_3 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7db_4 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7db_5 = BasicConv2d(c7, 192, kernel_size=(1, 7), padding=(0, 3))
        
        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)
        
    def forward(self, x):
        branch1 = self.branch1x1(x)
        
        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)
        
        branch7x7dbl = self.branch7x7db_1(x)
        branch7x7dbl = self.branch7x7db_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7db_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7db_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7db_5(branch7x7dbl)
        
        branch_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)(x)
        branch_pool = self.branch_pool(branch_pool)
        
        outputs = [branch7x7dbl, branch7x7, branch_pool, branch1]
        return torch.cat(outputs, 1)
    
class InceptionD(nn.Module):
    # Inception D block reduce dimensionality
    def __init__(self, in_channels):
        super(InceptionD, self).__init__()
        self.branch7x7_1 = BasicConv2d(in_channels, 192, kernel_size = 1)
        self.branch7x7_2 = BasicConv2d(192, 192, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7_3 = BasicConv2d(192, 192, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7_4 = BasicConv2d(192, 192, kernel_size=3, stride=2)
        
        self.branch3x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch3x3_2 = BasicConv2d(192, 320, kernel_size=3, stride=2)
        self.branch_pool = nn.MaxPool2d(kernel_size=3, stride=2)
        
    def forward(self, x):
        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)
        branch7x7 = self.branch7x7_4(branch7x7)
        
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        
        branch_pool = self.branch_pool(x)
        
        outputs = [branch3x3, branch7x7, branch_pool]
        return torch.cat(outputs, 1)
    
class InceptionE(nn.Module):
    def __init__(self, in_channels):
        super(InceptionE, self).__init__()
        self.branch3x3l_1 = BasicConv2d(in_channels, 448, kernel_size=1)
        self.branch3x3l_2 = BasicConv2d(448, 384, kernel_size = 3, padding =1)
        self.branch3x3l_3a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3l_3b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))
        
        self.branch3x3r_1 = BasicConv2d(in_channels, 384, kernel_size=1)
        self.branch3x3r_2a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3r_2b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))
        
        self.branch_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.branch_pool_conv = BasicConv2d(in_channels, 192, kernel_size=1)
        
        self.branch1x1 = BasicConv2d(in_channels, 320, kernel_size=1)
        
    def forward(self, x):
        branch3x3l = self.branch3x3l_1(x)
        branch3x3l = self.branch3x3l_2(branch3x3l)
        branch3x3l_3a = self.branch3x3l_3a(branch3x3l)
        branch3x3l_3b = self.branch3x3l_3b(branch3x3l)
        branch3x3l = torch.cat([branch3x3l_3a, branch3x3l_3b], 1)
        
        branch3x3r = self.branch3x3r_1(x)
        branch3x3r_2a = self.branch3x3r_2a(branch3x3r)
        branch3x3r_2b = self.branch3x3r_2b(branch3x3r)
        branch3x3r = torch.cat([branch3x3r_2a, branch3x3r_2b], 1)
        
        branch_pool = self.branch_pool(x)
        branch_pool = self.branch_pool_conv(branch_pool)
        
        branch1x1 = self.branch1x1(x)
        outputs = [branch3x3l, branch3x3r, branch_pool, branch1x1]
        return torch.cat(outputs, 1)
    
class AuxiliaryClassifier(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(AuxiliaryClassifier, self).__init__()
        self.pool1 = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv0 = BasicConv2d(in_channels, 128, kernel_size=1)
        self.conv1 = BasicConv2d(128, 768, kernel_size=5)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(768, num_classes)
        
    def forward(self, x):
        x = self.pool1(x)
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

InceptionOutput = namedtuple('InceptionOutput', ['logits', 'aux_logits'])

class InceptionV3(nn.Module):
    def __init__(self, num_classes = 1000, in_channels = 3, aux_logits = True):
        super().__init__()
        self.aux_logits = aux_logits
        self.num_classes = num_classes
        self.in_channels = in_channels
        
        self.conv1_3x3 = BasicConv2d(self.in_channels, 32, kernel_size=3, stride=2)
        self.conv2a_3x3 = BasicConv2d(32, 32, kernel_size=3)
        self.conv2b_3x3 = BasicConv2d(32, 64, kernel_size=3, padding = 1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.conv3a_3x3 = BasicConv2d(64, 80, kernel_size=1)
        self.conv3b_3x3 = BasicConv2d(80, 192, kernel_size=3)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.mixed_5a = InceptionA(in_channels=192, pool_features=32)
        self.mixed_5b = InceptionA(in_channels=256, pool_features=64)
        self.mixed_5c = InceptionA(in_channels=288, pool_features=64)
        
        self.mixed_6a = InceptionB(in_channels=288)
        self.mixed_6b = InceptionC(in_channels=768, channel_7=128)
        self.mixed_6c = InceptionC(in_channels=768, channel_7=160)
        self.mixed_6d = InceptionC(in_channels=768, channel_7=160)
        self.mixed_6e = InceptionC(in_channels=768, channel_7=192)
        
        if self.aux_logits:
            self.aux_classifier = AuxiliaryClassifier(in_channels=768, num_classes=self.num_classes)
            
        self.mixed_7a = InceptionD(in_channels=768)
        self.mixed_7b = InceptionE(in_channels=1280)
        self.mixed_7c = InceptionE(in_channels=2048)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.4)
        self.fc = nn.Linear(2048, self.num_classes)
        
    def forward(self, x):
        x = self.conv1_3x3(x)
        x = self.conv2a_3x3(x)
        x = self.conv2b_3x3(x)
        x = self.maxpool1(x)
        # print(x.shape)
        
        x = self.conv3a_3x3(x)
        x = self.conv3b_3x3(x)
        x = self.maxpool2(x)
        # print(x.shape)
        
        x = self.mixed_5a(x)
        x = self.mixed_5b(x)
        x = self.mixed_5c(x)
        # print(x.shape)
        
        x = self.mixed_6a(x)
        x = self.mixed_6b(x)
        x = self.mixed_6c(x)
        x = self.mixed_6d(x)
        x = self.mixed_6e(x)
        # print(x.shape)
        
        if self.aux_logits:
            input_aux = x
            aux = self.aux_classifier(input_aux)
        
        x = self.mixed_7a(x)
        x = self.mixed_7b(x)
        x = self.mixed_7c(x)
        # print(x.shape)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        if self.training and self.aux_logits:
            return InceptionOutput(x, aux)
        return x 
        
# Vision Transformer (ViT) model
class TransformerBlock(nn.Module):
    def __init__(self, emb_dim, num_heads, ratio_dim, dropout_rate):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=num_heads)
        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.MLP = nn.Sequential(
            nn.Linear(emb_dim, ratio_dim * emb_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(ratio_dim * emb_dim, emb_dim),
        )

    def forward(self, x):
        x_norm = self.norm1(x)
        x_norm = x_norm.permute(1, 0, 2)
        attn_out, _ = self.attention(x_norm, x_norm, x_norm)
        attn_out = attn_out.permute(1, 0, 2)
        x = x + self.dropout1(attn_out)
        norm_out = self.norm2(x)
        mlp_out = self.MLP(norm_out)
        x = x + self.dropout2(mlp_out)
        return x
    
class PatchEmbedding(nn.Module):
    def __init__(self, img_size =224, patch_size = 16, in_channels = 3, emb_dim = 768):
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.emb_dim = emb_dim
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, emb_dim, kernel_size= patch_size, stride = patch_size)
        
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        x = x.flatten(2) # (B, emb_dim, num_patches)
        x = x.transpose(1, 2) # (B, num_patches, emb_dim)
        return x
class VisionTransformer(nn.Module):
    
    def __init__(self, num_classes=1000, in_channels=3, img_size=224, patch_size=16, 
                 emb_dim=768, num_layers=12, num_heads=12, ratio_dim = 4, dropout_rate=0.1):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ratio_dim = ratio_dim
        self.emb_dim = emb_dim
        self.dropout_rate = dropout_rate
        
        # Patch Embedding
        self.patch_embed = PatchEmbedding(img_size = self.img_size, 
                                          patch_size= self.patch_size,
                                          in_channels= self.in_channels,
                                          emb_dim= self.emb_dim)
        num_patches = self.patch_embed.num_patches
        
        # Positional Encoding
        self.position_embedding = nn.Parameter(torch.randn(1, num_patches + 1, self.emb_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))
        self.dropout = nn.Dropout(dropout_rate)
        # Transformer Encoder Layers
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(emb_dim, num_heads, ratio_dim, dropout_rate) for _ in range(num_layers)
        ])
        self.layernorm = nn.LayerNorm(emb_dim)
        # Classification Head
        self.fc_out = nn.Linear(self.emb_dim, num_classes)
    
    def forward(self, x):
        # Patch Embedding
        batch = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(batch, -1, -1)
        x = torch.cat((cls_tokens, x), dim = 1)
        x += self.position_embedding
        x = self.dropout(x)
        
        for layer in self.transformer_layers:
            x = layer(x)
            
        x = self.layernorm(x)
        cls_token_final = x[:, 0]
        output = self.fc_out(cls_token_final)
        return output

# Implementation of MobileNetV3

class HardSwish(nn.Module):
    def __init__(self, inplace = True):
        super(HardSwish, self).__init__()
        self.inplace = inplace
    def forward(self, x):
        
        return x * F.relu6(x + 3, inplace=self.inplace) / 6

class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduction_ratio = 4):
        super(SqueezeExcitation, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        module_inp = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_inp* x


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio, se_ratio, activation):
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Size of the convolution kernel.
            stride (int): Stride of the convolution.
            expand_ratio (int): Expansion ratio for the depthwise separable convolution.
            se_ratio (float): Squeeze-and-excitation ratio.
            activation (str): Activation function to use ('relu' or 'hard_swish').
        """
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.use_res_connect = self.stride == 1 and in_channels == out_channels
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        else:
            self.activation = HardSwish(inplace=True)

        hidden_dim = int(round(in_channels * expand_ratio))
        layers = []
        
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                self.activation,
            ])
        
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride = stride, padding = kernel_size //2, groups = hidden_dim, bias= False),
            nn.BatchNorm2d(hidden_dim),
            self.activation,
        ])
        
        if se_ratio is not None:
            layers.append(SqueezeExcitation(hidden_dim, se_ratio))
            
        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride = 1, padding = 0, bias = False),
            nn.BatchNorm2d(out_channels)
        ])
        
        self.conv = nn.Sequential(*layers)
        
    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)
        
class MobileNetV3(nn.Module):
    def __init__(self, mode = "large", num_classes = 1000, dropout = 0.2):
        super(MobileNetV3, self).__init__()
        self.mode = mode
        if mode == "large":
            self.cfgs = [
                # in_channels, expand_ratio, out_channels, kernel_size, strice, se_ratio, activation
                [16, 16, 16, 3, 1, None, 'relu'],
                [16, 64, 24, 3, 2, None, 'relu'],
                [24, 72, 24, 3, 1, None, 'relu'],
                [24, 72, 40, 5, 2, 4, 'relu'],
                [40, 120, 40, 5, 1, 4, 'relu'],
                [40, 120, 40, 5, 1, 4, 'relu'],
                [40, 240, 80, 3, 2, None, 'hard_swish'],
                [80, 200, 80, 3, 1, None, 'hard_swish'],
                [80, 184, 80, 3, 1, None, 'hard_swish'],
                [80, 184, 80, 3, 1, None, 'hard_swish'],
                [80, 480, 112, 3, 1, 4, 'hard_swish'],
                [112, 672, 112, 3, 1, 4, 'hard_swish'],
                [112, 672, 160, 5, 2, 4, 'hard_swish'],
                [160, 960, 160, 5, 1, 4, 'hard_swish'],
                [160, 960, 160, 5, 1, 4, 'hard_swish'],
            ]
            
            in_channels = 16
            self.first_conv = nn.Sequential(
                nn.Conv2d(3, in_channels, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(in_channels),
                HardSwish(inplace=True),
            )
            
            self.last_conv = nn.Sequential(
                nn.Conv2d(160, 960, kernel_size=1, bias=False),
                nn.BatchNorm2d(960),
                HardSwish(inplace=True)
            )
            
        elif mode == 'small':
            self.cfgs = [
                # in_channels, expand_ratio, out_channels, kernel_size, strice, se_ratio, activation
                [16, 16, 16, 3, 2, 4, "relu"],       
                [16, 72, 24, 3, 2, None, "relu"],   
                [24, 88, 24, 3, 1, None, "relu"],    
                [24, 96, 40, 5, 2, 4, "hard_swish"],       
                [40, 240, 40, 5, 1, 4, "hard_swish"],      
                [40, 240, 40, 5, 1, 4, "hard_swish"],      
                [40, 120, 48, 5, 1, 4, "hard_swish"],    
                [48, 144, 48, 5, 1, 4, "hard_swish"],     
                [48, 288, 96, 5, 2, 4, "hard_swish"],      
                [96, 576, 96, 5, 1, 4, "hard_swish"],      
                [96, 576, 96, 5, 1, 4, "hard_swish"],      
            ]
            
            input_channel = 16
            self.first_conv = nn.Sequential(
                nn.Conv2d(3, input_channel, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(input_channel),
                HardSwish(inplace=True)
            )
            
            self.last_conv = nn.Sequential(
                nn.Conv2d(96, 576, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(576),
                HardSwish(inplace=True)
            )
        
        else:
            raise ValueError(f"Unsupported mode: {mode}, please use 'large' or 'small'")
        
        self.blocks = nn.ModuleList([])
        for cfg in self.cfgs:
            in_c = cfg[0]
            expand_ratio = cfg[1] / in_c
            out_c = cfg[2]
            kernel_size = cfg[3]
            stride = cfg[4]
            se_ratio = cfg[5]
            activation = cfg[6]
            
            self.blocks.append(
                InvertedResidual(in_c, out_c, kernel_size, stride, expand_ratio, se_ratio, activation)
            )
            
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        last_dim = 960 if mode == "large" else 576
        self.classifier = nn.Sequential(
            nn.Linear(last_dim, 1280), 
            HardSwish(inplace=True),
            nn.Dropout(p = dropout), 
            nn.Linear(1280, num_classes)
        )
        
    def forward(self, x):
        x = self.first_conv(x)
        for block in self.blocks:
            x = block(x)
        x = self.last_conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
            