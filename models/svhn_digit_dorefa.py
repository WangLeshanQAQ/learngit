import torch
import torch.nn as nn
import torch.nn.init as init

from utils.quantizer import *


class DigitNet_Q(nn.Module):
    def __init__(self, wbit, abit, num_classes=10):
        super(DigitNet_Q, self).__init__()
        Conv2d = conv2d_Q_fn(w_bit=wbit)
        Linear = linear_Q_fn(w_bit=wbit)

        self.features = nn.Sequential(
            # conv 0 40
            nn.Conv2d(3, 48, kernel_size=5, stride=1),  # 36
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),  # 18
            # nn.Dropout(0.2),

            # conv 1
            Conv2d(48, 64, kernel_size=3, stride=1, padding=1),  # 18
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            # nn.Dropout(0.2),
            activation_quantize_fn(a_bit=abit),

            # conv 2
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1),  # 18
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0), # 9
            # nn.Dropout(0.2),
            activation_quantize_fn(a_bit=abit),

            # conv 3
            Conv2d(64, 128, kernel_size=3, padding=0),  # 7
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            # nn.Dropout(0.2),
            activation_quantize_fn(a_bit=abit),

            # conv 4
            Conv2d(128, 128, kernel_size=3, padding=1),  # 7
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            # nn.Dropout(0.2),
            activation_quantize_fn(a_bit=abit),

            # conv 5
            Conv2d(128, 128, kernel_size=3, padding=0), # 5
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # n.MaxPool2d(kernel_size=2, stride=2, padding=1), # 5
            activation_quantize_fn(a_bit=abit),
            nn.Dropout(0.5),

            # conv 6
            Conv2d(128, 512, kernel_size=5, padding=0), # 1
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            # nn.Dropout(0.2),
            activation_quantize_fn(a_bit=abit),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1, num_classes)
        )

        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                init.xavier_normal_(module.weight.data)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
