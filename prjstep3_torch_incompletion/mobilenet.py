import os
import cv2
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, t, stride):
        super(Bottleneck, self).__init__()

        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features = nn.Sequential(OrderedDict([
                                                  ('pconv1', nn.Conv2d(in_channels, in_channels*t, kernel_size=1, stride=1, padding=0, bias=False)),
                                                  ('bn1', nn.BatchNorm2d(in_channels*t)),
                                                  ('act1', nn.ReLU6()),
                                                  ('dconv', nn.Conv2d(in_channels*t, in_channels*t, kernel_size=3, stride=stride, padding=1, bias=False)),
                                                  ('bn2', nn.BatchNorm2d(in_channels*t)),
                                                  ('act2', nn.ReLU6()),
                                                  ('pconv3', nn.Conv2d(in_channels*t, out_channels, kernel_size=1, stride=1, padding=0, bias=False)),
                                                  ('bn3', nn.BatchNorm2d(out_channels))
        ]))

    def forward(self, x):
        out = self.features(x)        
        if (self.stride==1) and (self.in_channels==self.out_channels):
            out += x
            return out


class MobileNetV2(nn.Module):
    def __init__(self, block, settings, mul, nclasses):
        super(MobileNetV2, self).__init__()

        self.num_classes = nclasses
        self.settings = settings
        self.settings['c'] = [int(elt * mul) for elt in self.settings['c']] 
        self.in_channels = int(32 * mul)
        self.out_channels = int(1280 * mul)

        self.conv0 = nn.Sequential(OrderedDict([
                                                ('conv0', nn.Conv2d(3, self.in_channels, kernel_size=1, stride=2, bias=False)),
                                                ('bn0', nn.BatchNorm2d(self.in_channels)),
                                                ('act0', nn.ReLU6())
        ]))
        self.bottleneck1 = self.build_layer(block, 
                                            self.in_channels,       # n(in_channels)
                                            self.settings['c'][0],  # n(out_channels)
                                            self.settings['t'][0],  # expansion factor
                                            self.settings['s'][0],  # stride
                                            self.settings['n'][0])  # repeat
        self.bottleneck2 = self.build_layer(block, 
                                            self.settings['c'][0],  # n(in_channels)
                                            self.settings['c'][1],  # n(out_channels)
                                            self.settings['t'][1],  # expansion factor
                                            self.settings['s'][1],  # stride
                                            self.settings['n'][1])  # repeat
        self.bottleneck3 = self.build_layer(block, 
                                            self.settings['c'][1],  # n(in_channels)
                                            self.settings['c'][2],  # n(out_channels)
                                            self.settings['t'][2],  # expansion factor
                                            self.settings['s'][2],  # stride
                                            self.settings['n'][2])  # repeat
        self.bottleneck4 = self.build_layer(block, 
                                            self.settings['c'][2],  # n(in_channels)
                                            self.settings['c'][3],  # n(out_channels)
                                            self.settings['t'][3],  # expansion factor
                                            self.settings['s'][3],  # stride
                                            self.settings['n'][3])  # repeat
        self.bottleneck5 = self.build_layer(block, 
                                            self.settings['c'][3],  # n(in_channels)
                                            self.settings['c'][4],  # n(out_channels)
                                            self.settings['t'][4],  # expansion factor
                                            self.settings['s'][4],  # stride
                                            self.settings['n'][4])  # repeat
        self.bottleneck6 = self.build_layer(block, 
                                            self.settings['c'][4],  # n(in_channels)
                                            self.settings['c'][5],  # n(out_channels)
                                            self.settings['t'][5],  # expansion factor
                                            self.settings['s'][5],  # stride
                                            self.settings['n'][5])  # repeat
        self.bottleneck7 = self.build_layer(block, 
                                            self.settings['c'][5],  # n(in_channels)
                                            self.settings['c'][6],  # n(out_channels)
                                            self.settings['t'][6],  # expansion factor
                                            self.settings['s'][6],  # stride
                                            self.settings['n'][6])  # repeat

        self.conv8 = nn.Sequential(OrderedDict([
                                                ('conv8', nn.Conv2d(self.settings['c'][6], self.out_channels, kernel_size=1, bias=False)),
                                                ('bn8', nn.BatchNorm2d(self.out_channels)),
                                                ('act8', nn.ReLU6())
        ]))

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.conv9 = nn.Conv2d(self.out_channels, self.num_classes, 1)

    def build_layer(self, block, in_channels, out_channels, t, stride, n_repeat):
        layers = []
        pre_out = in_channels
        for i in range(n_repeat):
          if i == 0:
            layers.append(block(pre_out, out_channels, t, stride))
          else:
            layers.append(block(pre_out, out_channels, t, 1))
          pre_out = out_channels

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv0(x)
        out = self.bottleneck1(out)
        out = self.bottleneck2(out)
        out = self.bottleneck3(out)
        out = self.bottleneck4(out)
        out = self.bottleneck5(out)
        out = self.bottleneck6(out)
        out = self.bottleneck7(out)
        out = self.conv8(out)
        out = self.avgpool(out)
        out = self.conv9(out)
        out = out.view(-1, self.num_classes)
        return out

