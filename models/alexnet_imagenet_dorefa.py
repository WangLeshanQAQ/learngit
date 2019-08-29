import torch
import torch.nn as nn
import torch.nn.init as init

from utils.quantizer import *


class AlexNet_Q(nn.Module):
    def __init__(self, wbit, abit, num_classes=10):
        super(AlexNet_Q, self).__init__()
        Conv2d = conv2d_Q_fn(w_bit=wbit)
        Linear = linear_Q_fn(w_bit=wbit)

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            Conv2d(64, 192, kernel_size=5, padding=2),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            activation_quantize_fn(a_bit=abit),
            nn.MaxPool2d(kernel_size=3, stride=2),

            Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            activation_quantize_fn(a_bit=abit),

            Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            activation_quantize_fn(a_bit=abit),

            Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            activation_quantize_fn(a_bit=abit),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            Linear(256 * 1 * 1, 4096),
            nn.ReLU(inplace=True),
            activation_quantize_fn(a_bit=abit),

            Linear(4096, 4096),
            nn.ReLU(inplace=True),
            activation_quantize_fn(a_bit=abit),
            nn.Linear(4096, num_classes),
        )

        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                init.xavier_normal_(module.weight.data)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
