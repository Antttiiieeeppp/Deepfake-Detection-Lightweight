## 한번에 채널 수를 늘렸을 때의 메모리 사용량 질문 해보기
# 한번에 512 채널로 conv 연산 했을 때의 메모리 사용량
# 3, 32, ..., 256, 512 채널로 conv 연산 했을 때의 메모리 사용량

import torch
import torch.nn as nn
from torchsummary import summary


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=1):
        super(SimpleCNN, self).__init__()

        # Initial convolution with aggressive downsampling
        self.initial_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # Efficient blocks with residual connections
        self.block1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Final classification layers
        self.classifier = nn.Sequential(
            nn.Linear(256, 512), nn.ReLU(), nn.Dropout(0.5), nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.initial_conv(x)

        identity = x
        x = self.block1(x)
        x = x + identity

        x = self.block2(x)
        x = self.block3(x)

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        x = self.classifier(x)
        return x