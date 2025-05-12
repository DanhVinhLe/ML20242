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

#Lenet Model
class LeNet(nn.Module):
    def __init__(self, num_classes = 10, in_channels = 1):
        super(LeNet, self).__init__()
        # Các lớp tích chập
        self.in_channels = in_channels
        self.num_classes = num_classes 
        self.conv1 = nn.Conv2d(self.in_channels, 6, kernel_size=5)
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
        
        self.fc1 = nn.Linear(512 * self.feature_size * self.feature_size, 4096)
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
